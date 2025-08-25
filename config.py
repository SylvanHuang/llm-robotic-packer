from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

USE_LOCAL_LLM: bool = True


LFD_LORA_CKPT: Path = PROJECT_ROOT / "models" / "gemma-2b-it"

# Recommended for Gemma-2: eager
ATTN_IMPL: str = "eager"   # or "sdpa"

# The simulator writes the current compact state here
BIN_STATE_PATH: Path = PROJECT_ROOT / "instructions" / "bin_state.json"

# Optional: slightly improves MPS memory reuse on macOS
PYTORCH_MPS_HIGH_WATERMARK_RATIO: str = "0.0"
