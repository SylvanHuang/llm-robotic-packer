# config.py
# Central config for the simulator & LLM backends.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---- Backend selection ----
# True  -> use local LoRA Gemma-2B
# False -> use remote API (original llm_generate.py)
USE_LOCAL_LLM: bool = True

# ---- Local LoRA model ----
# Folder that contains adapter_model.safetensors + tokenizer files (see README)
LFD_LORA_CKPT: Path = PROJECT_ROOT / "models" / "gemma-2b-it"

# Recommended for Gemma-2: eager
ATTN_IMPL: str = "eager"   # or "sdpa"

# The simulator writes the current compact state here
BIN_STATE_PATH: Path = PROJECT_ROOT / "instructions" / "bin_state.json"

# Optional: slightly improves MPS memory reuse on macOS
PYTORCH_MPS_HIGH_WATERMARK_RATIO: str = "0.0"
