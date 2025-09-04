import os
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

API_MODEL = "gpt-4o-mini"


LOCAL_MODEL = "llama32-3b"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_DIR   = os.path.join(REPO_ROOT, "models", "llama32-3b")
ATTN_IMPL = "sdpa"  # eager or sdpa


USE_LOCAL_LLM: bool = True

if USE_LOCAL_LLM:
    from llm_local import (
        choose_rotation_and_anchor,
        generate_path,
    )
else:
    from llm_api import (
        choose_rotation_and_anchor,
        generate_path,
    )

__all__ = ["choose_rotation_and_anchor", "generate_path"]
