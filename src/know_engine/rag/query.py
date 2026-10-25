"""Intent recognition and query rewriting (both on the light model)."""

import json
import re
from dataclasses import dataclass

import structlog

from know_engine.config import get_settings
from know_engine.prompts import load_prompt
from know_engine.rag.llm import complete, render

log = structlog.get_logger()

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
# Both calls sit in front of every answer and have a safe fallback, so they must not stall it.
CALL_TIMEOUT_SECONDS = 8


@dataclass
class Intent:
    related: bool
    intent: str
    reasoning: str = ""


def recognize_intent(question: str) -> Intent:
    """Decide whether the question needs the knowledge base. Defaults to retrieval when unsure."""
    try:
        raw = complete(get_settings().light_model, [
            {"role": "system", "content": load_prompt("intent-recognition-new-prompt.txt")},
            {"role": "user", "content": question},
        ], timeout=CALL_TIMEOUT_SECONDS)
        data = json.loads(JSON_OBJECT.search(raw).group())
        return Intent(bool(data.get("related", True)), str(data.get("intent", "")), str(data.get("reasoning", "")))
    except Exception as e:
        log.warning("intent_failed", error=str(e))
        return Intent(True, "其他", "意图识别失败，默认检索")


def format_history(history: list[tuple[str, str]]) -> str:
    """Same format as the Java QueryTransformer: ``User: ...`` / ``AI: ...`` lines."""
    return "\n".join(f"{'User' if role == 'USER' else 'AI'}: {text}" for role, text in history if text)


def rewrite_query(question: str, history: list[tuple[str, str]], prompt_name: str = "query-rewrite-prompt.txt") -> str:
    """Rewrite into a standalone search query; falls back to the original question."""
    prompt = render(load_prompt(prompt_name), chatMemory=format_history(history), query=question)
    try:
        rewritten = complete(get_settings().light_model, [{"role": "user", "content": prompt}],
                             timeout=CALL_TIMEOUT_SECONDS)
    except Exception as e:
        log.warning("rewrite_failed", error=str(e))
        return question
    # Keep only the first line, in case the model adds an explanation.
    rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
    return rewritten.removeprefix("Output：").removeprefix("Output:").strip() or question
