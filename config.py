import os
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


API_GPT_5o = 'openai/gpt-5-mini'
API_GPT_5_nano = 'openai/gpt-5-nano'
API_GPT_oss_120 = 'openai/gpt-oss-120b'
API_GPT_oss_20 = 'openai/gpt-oss-20b'
API_GPT_4_1 = 'openai/gpt-4.1'
API_GPT_4_1_mini = 'openai/gpt-4.1-mini'
API_GPT_4o_mini = 'openai/gpt-4o-mini'

API_CLAUDE_4 = 'anthropic/claude-sonnet-4'
API_CLAUDE_3_7 = 'anthropic/claude-3.7-sonnet'
API_CLAUDE_3_5 = 'anthropic/claude-3.5-sonnet'

API_GEMINI_FLASH_2_5 = 'google/gemini-2.5-flash'
API_GEMINI_FLASH_2_5_LITE = 'google/gemini-2.5-flash-lite'


LOCAL_MODEL = "llama32-3b"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_DIR   = os.path.join(REPO_ROOT, "models", "llama32-3b")
ATTN_IMPL = "sdpa"  # eager or sdpa


USE_LOCAL_LLM: bool = False

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
