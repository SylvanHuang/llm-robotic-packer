# local_llm.py
# Local Gemma‑2‑2B‑IT + LoRA backend that mirrors the old API flow:
#   - Reads:  instructions/bin_state.json
#   - Writes: instructions/instruction.json
# Robust JSON extraction + hard fallback for path so the sim never crashes.

import os, re, json
from typing import Any, Dict, List, Tuple, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# --- after these lines in local_llm.py ---
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
BIN_STATE_PATH = os.path.join(REPO_ROOT, "instructions", "bin_state.json")
INSTR_PATH     = os.path.join(REPO_ROOT, "instructions", "instruction.json")

# ADD THIS:
def _ensure_instruction_json():
    """Make sure instructions/instruction.json exists and has a path[0]."""
    os.makedirs(os.path.dirname(INSTR_PATH), exist_ok=True)
    default_payload = {"path": [[0.0, 0.0, 0.0]]}  # safe placeholder
    try:
        if os.path.exists(INSTR_PATH):
            with open(INSTR_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "path" in data and isinstance(data["path"], list) and data["path"]:
                return  # already valid
    except Exception:
        pass
    with open(INSTR_PATH, "w") as f:
        json.dump(default_payload, f, separators=(",", ":"), ensure_ascii=False)

# CALL IT immediately so env init can read it:
_ensure_instruction_json()


# Optional MPS memory hint
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

BASE_MODEL = "google/gemma-2-2b-it"
ATTN_IMPL  = "eager"   # Gemma2 recommends eager
_SYSTEM = "<|system|> You are a bin-packing assistant. Respond in STRICT JSON only.<|end|>"

# ---- cache (prevents reloading each call)
_MODEL = None
_TOK   = None
_DEVICE = None
_LAST_PICK = None


def _coerce_point(p) -> List[float]:
    # Accept [x,y,z] with numbers or numeric strings
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

# --------------------- utils: normalization ---------------------

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
            return {"w":float(obj["l"]),"h":float(obj["w"]),"d":float(obj["h"])}
        return {"w":t[0],"h":t[1],"d":t[2]}
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
                        return {"w":float(sub["l"]),"h":float(sub["w"]),"d":float(sub["h"])}
                    return {"w":t[0],"h":t[1],"d":t[2]}
    for v in obj.values():
        if isinstance(v,dict):
            t = _coerce_triplet_from_dict(v,[("w","h","d"),("width","height","depth"),("l","w","h")])
            if t:
                if "l" in v and "w" in v and "h" in v:
                    return {"w":float(v["l"]),"h":float(v["w"]),"d":float(v["h"])}
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

def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def _load_once():
    global _MODEL, _TOK, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TOK, _DEVICE

    _DEVICE = _device()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPL,
        device_map=None,      # load on CPU, then move → avoids MPS warmup issues
    )
    base.to(_DEVICE)
    base.config.use_cache = True

    # attach LoRA adapter from repo folder models/gemma-2b-it
    ckpt_dir = os.path.join(REPO_ROOT, "models", "gemma-2b-it")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"LoRA adapter not found at {ckpt_dir}")
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()

    _MODEL, _TOK = model, tok
    return _MODEL, _TOK, _DEVICE

# --------------------- JSON parsing/sanitizing ---------------------

def _assistant_segment(text: str) -> str:
    seg = text.split("<|assistant|>")[-1]
    return seg.split("<|end|>")[0]

def _strip_fences(s: str) -> str:
    return s.replace("```json","").replace("```","").strip()

def _balanced_json_slice(s: str) -> str | None:
    """Return the first balanced {...} slice if exists."""
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
    seg = _assistant_segment(raw_text)
    # attempt 1: balanced slice then parse
    bal = _balanced_json_slice(seg)
    if bal:
        try:
            return json.loads(_strip_fences(bal))
        except Exception:
            pass
    # attempt 2: sanitize then parse
    return json.loads(_sanitize_jsonish(seg))

# --------------------- generation ---------------------

class StopOnEndToken(StoppingCriteria):
    def __init__(self, tok):
        self.end_id = tok.convert_tokens_to_ids("<|end|>") if "<|end|>" in tok.get_vocab() else tok.eos_token_id
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.end_id

def _generate(prompt: str, max_new_tokens: int) -> Dict[str,Any]:
    model, tok, device = _load_once()
    inputs = tok([prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,         # deterministic
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnEndToken(tok)]),
        )
    text = tok.decode(out[0], skip_special_tokens=False)
    # try parse, then a single self‑repair retry if needed
    try:
        return _parse_json_or_fix(text)
    except Exception:
        repair = f"{_SYSTEM}\n<|user|>Output strict JSON only. Reprint just the corrected JSON.<|end|>\n<|assistant|>"
        out2 = model.generate(
            **tok([repair], return_tensors="pt").to(device),
            do_sample=False, top_p=1.0, max_new_tokens=max_new_tokens//2,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnEndToken(tok)]),
        )
        return _parse_json_or_fix(tok.decode(out2[0], skip_special_tokens=False))

# --------------------- prompts ---------------------

def _prompt(user_json: str) -> str:
    return f"{_SYSTEM}\n<|user|>{user_json}<|end|>\n<|assistant|>"

def _build_user_pick(state: Dict[str,Any], feedback: str) -> str:
    payload = {
        "state": {
            "bin_dims": state["bin_dims"],
            "incoming_box": state["incoming_box"],
            "anchors_indexed": state["anchors_indexed"],
        },
        "instruction": "Choose one rotation_index and one anchor_id from anchors_indexed. Return JSON with keys rotation_index (int) and anchor_id (string).",
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
        "instruction": "Produce a collision‑free path ending exactly at target_anchor.pos. Return JSON with key path: [[x,y,z],...].",
    }
    if feedback: payload["feedback"] = feedback
    return json.dumps(payload, separators=(",",":"))

# --------------------- public API (writes instruction.json) ---------------------

def choose_rotation_and_anchor(feedback: str = "") -> dict | None:
    global _LAST_PICK
    try:
        raw = json.load(open(BIN_STATE_PATH, "r"))
        state = _normalize_state(raw)
    except Exception as e:
        print(f"[local_llm] state normalization failed: {e}")
        return None

    pick = _generate(_prompt(_build_user_pick(state, feedback)), max_new_tokens=128)
    if not isinstance(pick, dict) or "rotation_index" not in pick or "anchor_id" not in pick:
        print(f"[local_llm] invalid pick JSON: {pick}")
        return None

    # Write exactly what the API flow expects:
    try:
        os.makedirs(os.path.dirname(INSTR_PATH), exist_ok=True)
        with open(INSTR_PATH, "w") as f:
            json.dump(pick, f, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:
        print(f"[local_llm] failed writing {INSTR_PATH}: {e}")

    _LAST_PICK = pick
    return pick

def call_gpt4_for_path_to_target(final_pos, feedback: str = "") -> dict | None:
    """Generate a path JSON to final_pos and write it to instructions/instruction.json."""
    # 1) Read & normalize state
    try:
        with open(BIN_STATE_PATH, "r") as f:
            raw = json.load(f)
        try:
            # use normalizer if defined in this file
            state = _normalize_state(raw)  # type: ignore[name-defined]
        except Exception:
            # fallback: pass raw through (only if it already matches)
            state = raw
    except Exception as e:
        print(f"[local_llm] failed reading state: {e}")
        return None

    # 2) Build prompt (derive anchor_id from last pick if available)
    anchor_id = _LAST_PICK.get("anchor_id") if isinstance(_LAST_PICK, dict) else None
    try:
        user_json = _build_user_path(state, list(map(float, final_pos)), anchor_id, feedback)
    except Exception as e:
        print(f"[local_llm] build_user_path failed: {e}")
        # minimal fallback path
        safe = {"path": [list(map(float, final_pos))]}
        os.makedirs(os.path.dirname(INSTR_PATH), exist_ok=True)
        with open(INSTR_PATH, "w") as f:
            json.dump(safe, f, separators=(",", ":"), ensure_ascii=False)
        return safe

    prompt = _prompt(user_json)

    # 3) Generate → parse with hard fallback
    try:
        path = _generate(prompt, max_new_tokens=192)
    except Exception:
        path = {"path": [list(map(float, final_pos))]}

    # ---- Coercion helpers (keep local to avoid NameError) ----
    def _coerce_point(p):
        return [float(p[0]), float(p[1]), float(p[2])]

    def _coerce_path_list(path_list):
        out = []
        if isinstance(path_list, list):
            for pt in path_list:
                if isinstance(pt, (list, tuple)) and len(pt) == 3:
                    try:
                        out.append(_coerce_point(pt))
                    except Exception:
                        continue
        return out

    def _coerce_path_dict(d):
        if not isinstance(d, dict):
            return {"path": []}
        return {"path": _coerce_path_list(d.get("path", []))}

    # 4) Coerce to numeric path and ensure it ends at final_pos
    path = _coerce_path_dict(path)
    fpos = list(map(float, final_pos))
    if not path["path"]:
        path["path"] = [fpos]
    else:
        end = path["path"][-1]
        if any(abs(a - b) > 1e-6 for a, b in zip(end, fpos)):
            path["path"].append(fpos)

    # 5) Write to instruction.json (what the simulator expects)
    try:
        os.makedirs(os.path.dirname(INSTR_PATH), exist_ok=True)
        with open(INSTR_PATH, "w") as f:
            json.dump(path, f, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:
        print(f"[local_llm] failed writing {INSTR_PATH}: {e}")

    return path
