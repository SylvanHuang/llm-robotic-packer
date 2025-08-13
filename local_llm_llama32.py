# local_llm_llama32.py
# Local Meta-Llama-3.2-3B-Instruct + LoRA backend (drop-in replacement for local_llm.py)
# - Reads:  instructions/bin_state.json
# - Writes: instructions/instruction.json
# - Public API: choose_rotation_and_anchor(), call_gpt4_for_path_to_target(final_pos, feedback="")
#
# Assumptions:
#   BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
#   LoRA adapter dir at: ./models/llama32-3b   (contains adapter_model.safetensors, adapter_config.json, etc.)
#
# If your base model ID or adapter path differ, edit BASE_MODEL and LORA_DIR below.

import os, re, json
from typing import Any, Dict, List, Tuple, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# ---------- paths & setup ----------
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
BIN_STATE_PATH = os.path.join(REPO_ROOT, "instructions", "bin_state.json")
INSTR_PATH     = os.path.join(REPO_ROOT, "instructions", "instruction.json")

def _ensure_instruction_json():
    """Ensure instructions/instruction.json exists with a minimal valid structure."""
    os.makedirs(os.path.dirname(INSTR_PATH), exist_ok=True)
    default_payload = {"path": [[0.0, 0.0, 0.0]]}
    try:
        if os.path.exists(INSTR_PATH):
            with open(INSTR_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "path" in data and isinstance(data["path"], list) and data["path"]:
                return
    except Exception:
        pass
    with open(INSTR_PATH, "w") as f:
        json.dump(default_payload, f, separators=(",", ":"), ensure_ascii=False)

_ensure_instruction_json()

# Optional MPS memory hint on macOS
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ---------- model constants ----------
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # <-- change if you used a different base
LORA_DIR   = os.path.join(REPO_ROOT, "models", "llama32-3b")  # <-- your fine-tuned adapter directory
ATTN_IMPL  = "sdpa"  # Llama 3.x generally prefers SDPA; fallback to "eager" if needed

SYSTEM_MSG = (
    "You are a bin-packing assistant. "
    "Always respond in STRICT JSON only, no extra text, no code fences."
)

# ---------- module-level caches ----------
_MODEL = None
_TOK   = None
_DEVICE: torch.device | None = None
_LAST_PICK: Dict[str, Any] | None = None

# ---------- coercion helpers ----------
def _coerce_point(p) -> List[float]:
    return [float(p[0]), float(p[1]), float(p[2])]

def _coerce_path_list(path) -> List[List[float]]:
    if not isinstance(path, list):
        return []
    out = []
    for pt in path:
        if isinstance(pt, (list, tuple)) and len(pt) == 3:
            try:
                out.append(_coerce_point(pt))
            except Exception:
                continue
    return out

def _coerce_path_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {"path": []}
    p = d.get("path", [])
    return {"path": _coerce_path_list(p)}

# --------------------- utils: normalization of state ---------------------
def _coerce_triplet_from_iter(x: Iterable[Any]) -> List[float] | None:
    try:
        arr = list(x)
        if len(arr) == 3 and all(isinstance(v, (int, float)) for v in arr):
            return [float(arr[0]), float(arr[1]), float(arr[2])]
    except Exception:
        pass
    return None

def _coerce_triplet_from_dict(d: Dict[str, Any], candidates: List[Tuple[str, str, str]]) -> List[float] | None:
    for a,b,c in candidates:
        if all(k in d for k in (a,b,c)):
            try: return [float(d[a]), float(d[b]), float(d[c])]
            except Exception: continue
    return None

def _coerce_box_dims(obj: Any) -> Dict[str,float] | None:
    if isinstance(obj, (list,tuple)):
        t = _coerce_triplet_from_iter(obj)
        if t: return {"w":t[0],"h":t[1],"d":t[2]}
    if not isinstance(obj, dict): return None
    t = _coerce_triplet_from_dict(obj, [("w","h","d"),("width","height","depth"),("l","w","h")])
    if t:
        if "l" in obj and "w" in obj and "h" in obj:
            return {"w":float(obj["l"]), "h":float(obj["w"]), "d":float(obj["h"])}
        return {"w":t[0], "h":t[1], "d":t[2]}
    for key in ("dims","size","dimensions"):
        if key in obj:
            sub = obj[key]
            if isinstance(sub,(list,tuple)):
                t = _coerce_triplet_from_iter(sub)
                if t: return {"w":t[0],"h":t[1],"d":t[2]}
            if isinstance(sub,dict):
                t = _coerce_triplet_from_dict(sub,[("w","h","d"),("width","height","depth"),("l","w","h")])
                if t:
                    if "l" in sub and "w" in sub and "h" in sub:
                        return {"w":float(sub["l"]), "h":float(sub["w"]), "d":float(sub["h"])}
                    return {"w":t[0],"h":t[1],"d":t[2]}
    for v in obj.values():
        if isinstance(v,dict):
            t = _coerce_triplet_from_dict(v,[("w","h","d"),("width","height","depth"),("l","w","h")])
            if t:
                if "l" in v and "w" in v and "h" in v:
                    return {"w":float(v["l"]), "h":float(v["w"]), "d":float(v["h"])}
                return {"w":t[0],"h":t[1],"d":t[2]}
        if isinstance(v,(list,tuple)):
            t = _coerce_triplet_from_iter(v)
            if t: return {"w":t[0],"h":t[1],"d":t[2]}
    return None

def _extract_bin_dims(s: Dict[str,Any]) -> List[float]:
    if "bin_dims" in s and _coerce_triplet_from_iter(s["bin_dims"]): return _coerce_triplet_from_iter(s["bin_dims"])  # type: ignore
    if "bin_size" in s and _coerce_triplet_from_iter(s["bin_size"]): return _coerce_triplet_from_iter(s["bin_size"])  # type: ignore
    if "bin" in s:
        b = s["bin"]
        if isinstance(b,dict):
            t = _coerce_triplet_from_dict(b,[("w","h","d"),("width","height","depth")])
            if t: return t
        t = _coerce_triplet_from_iter(b)
        if t: return t
    if "env" in s and isinstance(s["env"],dict):
        return _extract_bin_dims(s["env"])
    raise KeyError("bin_dims")

def _extract_incoming_box(s: Dict[str,Any]) -> Dict[str,float]:
    for key in ("incoming_box","current_box","box","incoming"):
        if key in s:
            c = _coerce_box_dims(s[key])
            if c: return c
            raise KeyError(f"incoming_box shape under '{key}' not recognized: {str(s[key])[:200]}")
    if "env" in s and isinstance(s["env"],dict):
        return _extract_incoming_box(s["env"])
    raise KeyError("incoming_box")

def _extract_anchors_indexed(s: Dict[str,Any]) -> Any:
    if "anchors_indexed" in s: return s["anchors_indexed"]
    if "anchors" in s: return s["anchors"]
    if "env" in s and isinstance(s["env"],dict):
        e = s["env"]
        if "anchors_indexed" in e: return e["anchors_indexed"]
        if "anchors" in e: return e["anchors"]
    raise KeyError("anchors_indexed")

def _normalize_state(raw: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "bin_dims": _extract_bin_dims(raw),
        "incoming_box": _extract_incoming_box(raw),
        "anchors_indexed": _extract_anchors_indexed(raw),
    }

# --------------------- model load (cached) ---------------------
def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _dtype_for(device: torch.device):
    if device.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32

def _get_eot_id(tok) -> int | None:
    try:
        return tok.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        return None

def _load_once():
    global _MODEL, _TOK, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TOK, _DEVICE

    _DEVICE = _pick_device()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=_dtype_for(_DEVICE),
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPL,
        device_map=None,   # load on CPU then move → avoids MPS warmup issues
    )
    base.to(_DEVICE)
    base.config.use_cache = True

    if not os.path.isdir(LORA_DIR):
        raise FileNotFoundError(f"LoRA adapter not found at {LORA_DIR}")
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    _MODEL, _TOK = model, tok
    return _MODEL, _TOK, _DEVICE

# --------------------- JSON parsing/sanitizing ---------------------
def _strip_fences(s: str) -> str:
    return s.replace("```json","").replace("```","").strip()

def _balanced_json_slice(s: str) -> str | None:
    start = s.find("{")
    if start == -1: return None
    depth = 0
    for i,ch in enumerate(s[start:], start=start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def _sanitize_jsonish(s: str) -> str:
    s = _strip_fences(s)
    bal = _balanced_json_slice(s)
    if bal: s = bal
    # remove trailing commas
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # "1." -> "1.0" and ".5" -> "0.5"
    s = re.sub(r"(?<=\d)\.(?=[,\]\}\s])", ".0", s)
    s = re.sub(r"(?<=[:\s\[])\.(\d)", r"0.\1", s)
    # quote keys if missing quotes
    s = re.sub(r"([{,]\s*)([A-Za-z_][\w\-]*)(\s*:)", r'\1"\2"\3', s)
    return s

def _parse_json_or_fix(raw_text: str) -> Dict[str,Any]:
    # attempt 1: try to parse as-is (it should be only the assistant text)
    try:
        return json.loads(_strip_fences(raw_text))
    except Exception:
        pass
    # attempt 2: balanced slice or sanitize
    bal = _balanced_json_slice(raw_text)
    if bal:
        try:
            return json.loads(_strip_fences(bal))
        except Exception:
            pass
    return json.loads(_sanitize_jsonish(raw_text))

# --------------------- generation ---------------------
class StopOnEOT(StoppingCriteria):
    def __init__(self, tok):
        self.eot_id = _get_eot_id(tok)
        self.eos_id = tok.eos_token_id
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last = input_ids[0, -1].item()
        return last == self.eot_id or last == self.eos_id

def _apply_chat(tok, system: str, user: str, return_tensors="pt"):
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors=return_tensors
    )

def _generate(user_json: str, max_new_tokens: int) -> Dict[str,Any]:
    model, tok, device = _load_once()
    # Build chat prompt with the tokenizer's chat template (Llama 3.x)
    inputs = _apply_chat(tok, SYSTEM_MSG, user_json).to(device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            do_sample=False,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=[tok.eos_token_id] + ([ _get_eot_id(tok) ] if _get_eot_id(tok) is not None else []),
            stopping_criteria=StoppingCriteriaList([StopOnEOT(tok)]),
        )

    # Decode only the newly generated tokens (exclude the prompt tokens)
    gen_tokens = out[0, inputs.shape[1]:]
    text = tok.decode(gen_tokens, skip_special_tokens=True)

    # Try parse, then one self-repair retry if needed
    try:
        return _parse_json_or_fix(text)
    except Exception:
        repair_user = "Output strict JSON only. Reprint just the corrected JSON for the last instruction."
        rep_inputs = _apply_chat(tok, SYSTEM_MSG, repair_user).to(device)
        with torch.no_grad():
            out2 = model.generate(
                rep_inputs,
                do_sample=False,
                top_p=1.0,
                max_new_tokens=max_new_tokens//2,
                pad_token_id=tok.eos_token_id,
                eos_token_id=[tok.eos_token_id] + ([ _get_eot_id(tok) ] if _get_eot_id(tok) is not None else []),
                stopping_criteria=StoppingCriteriaList([StopOnEOT(tok)]),
            )
        gen2 = out2[0, rep_inputs.shape[1]:]
        text2 = tok.decode(gen2, skip_special_tokens=True)
        return _parse_json_or_fix(text2)

# --------------------- prompts ---------------------
def _build_user_pick(state: Dict[str,Any], feedback: str) -> str:
    payload = {
        "state": {
            "bin_dims": state["bin_dims"],
            "incoming_box": state["incoming_box"],
            "anchors_indexed": state["anchors_indexed"],
        },
        "instruction": (
            "Choose one rotation_index and one anchor_id from anchors_indexed. "
            "Return JSON with keys rotation_index (int) and anchor_id (string)."
        ),
    }
    if feedback: payload["feedback"] = feedback
    return json.dumps(payload, separators=(",",":"))

def _build_user_path(state: Dict[str,Any], final_pos: List[float], anchor_id: str|None, feedback: str) -> str:
    target = {"pos": final_pos}
    if anchor_id is not None: target["id"] = anchor_id
    payload = {
        "state": {
            "bin_dims": state["bin_dims"],
            "incoming_box": state["incoming_box"],
            "target_anchor": target,
        },
        "instruction": (
            "Produce a collision-free path ending exactly at target_anchor.pos. "
            "Return JSON with key path: [[x,y,z], ...]."
        ),
    }
    if feedback: payload["feedback"] = feedback
    return json.dumps(payload, separators=(",",":"))

# --------------------- public API (writes instruction.json) ---------------------
def choose_rotation_and_anchor(feedback: str = "") -> dict | None:
    global _LAST_PICK
    try:
        with open(BIN_STATE_PATH, "r") as f:
            raw = json.load(f)
        state = _normalize_state(raw)
    except Exception as e:
        print(f"[local_llm_llama32] state normalization failed: {e}")
        return None

    user_json = _build_user_pick(state, feedback)
    pick = _generate(user_json, max_new_tokens=128)

    if not isinstance(pick, dict) or "rotation_index" not in pick or "anchor_id" not in pick:
        print(f"[local_llm_llama32] invalid pick JSON: {pick}")
        return None

    # Write minimal response; simulator reads instruction.json when needed
    try:
        os.makedirs(os.path.dirname(INSTR_PATH), exist_ok=True)
        with open(INSTR_PATH, "w") as f:
            json.dump(pick, f, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:
        print(f"[local_llm_llama32] failed writing {INSTR_PATH}: {e}")

    _LAST_PICK = pick
    return pick

def call_gpt4_for_path_to_target(final_pos, feedback: str = "") -> dict | None:
    """Generate a path JSON to final_pos and write it to instructions/instruction.json."""
    # 1) Read & normalize state
    try:
        with open(BIN_STATE_PATH, "r") as f:
            raw = json.load(f)
        try:
            state = _normalize_state(raw)
        except Exception:
            state = raw
    except Exception as e:
        print(f"[local_llm_llama32] failed reading state: {e}")
        return None

    # 2) Build prompt (use last chosen anchor id if available)
    anchor_id = _LAST_PICK.get("anchor_id") if isinstance(_LAST_PICK, dict) else None
    try:
        fpos = list(map(float, final_pos))
        user_json = _build_user_path(state, fpos, anchor_id, feedback)
    except Exception as e:
        print(f"[local_llm_llama32] build_user_path failed: {e}")
        safe = {"path": [list(map(float, final_pos))]}
        os.makedirs(os.path.dirname(INSTR_PATH), exist_ok=True)
        with open(INSTR_PATH, "w") as f:
            json.dump(safe, f, separators=(",", ":"), ensure_ascii=False)
        return safe

    # 3) Generate → parse with hard fallback
    try:
        path = _generate(user_json, max_new_tokens=192)
    except Exception:
        path = {"path": [list(map(float, final_pos))]}

    # 4) Coerce to numeric path and ensure it ends at final_pos
    path = _coerce_path_dict(path)
    if not path["path"]:
        path["path"] = [fpos]
    else:
        end = path["path"][-1]
        if any(abs(a - b) > 1e-6 for a, b in zip(end, fpos)):
            path["path"].append(fpos)

    # 5) Write to instruction.json
    try:
        os.makedirs(os.path.dirname(INSTR_PATH), exist_ok=True)
        with open(INSTR_PATH, "w") as f:
            json.dump(path, f, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:
        print(f"[local_llm_llama32] failed writing {INSTR_PATH}: {e}")

    return path