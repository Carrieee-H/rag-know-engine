from functools import lru_cache

from know_engine.config import PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / "prompts"


@lru_cache
def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()
