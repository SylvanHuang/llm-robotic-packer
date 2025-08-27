API_MODEL = "gpt-4o"
LOCAL_MODEL = "llama32-3b"


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
