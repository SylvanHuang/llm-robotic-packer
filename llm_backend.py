# llm_backend.py
# Router that switches between API (llm_generate.py) and local LoRA (local_llm.py),
# and ensures instructions/instruction.json exists so the env doesn't crash.

import os, json
from pathlib import Path
import config as cfg  # USE_LOCAL_LLM toggle lives here

# ---- Ensure instruction.json exists with a minimal valid structure
REPO_ROOT = Path(__file__).resolve().parent
INSTR_PATH = REPO_ROOT / "instructions" / "instruction.json"
INSTR_PATH.parent.mkdir(parents=True, exist_ok=True)

def _ensure_instruction_json():
    try:
        if INSTR_PATH.exists():
            data = json.loads(INSTR_PATH.read_text())
            if isinstance(data, dict) and isinstance(data.get("path"), list) and data["path"]:
                return
    except Exception:
        pass
    # minimal safe default so BinPacking3DEnv() can do ["path"][0]
    INSTR_PATH.write_text(json.dumps({"path": [[0.0, 0.0, 0.0]]}, separators=(",", ":")))

_ensure_instruction_json()

# llm_backend.py (only this block changes)
if cfg.USE_LOCAL_LLM:
    from local_llm_llama32 import (
        choose_rotation_and_anchor,
        call_gpt4_for_path_to_target,
    )
else:
    from llm_generate import (
        choose_rotation_and_anchor,
        call_gpt4_for_path_to_target,
    )

__all__ = ["choose_rotation_and_anchor", "call_gpt4_for_path_to_target"]
