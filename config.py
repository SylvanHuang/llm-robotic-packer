USE_LOCAL_LLM: bool = False

if USE_LOCAL_LLM:
    from llm_local import (
        choose_rotation_and_anchor,
        generate_path,
    )
else:
    from llm_generate import (
        choose_rotation_and_anchor,
        generate_path,
    )

__all__ = ["choose_rotation_and_anchor", "generate_path"]
