"""Chat-completion helpers on top of the OpenAI-compatible client."""

import re
from collections.abc import Iterable, Iterator

from know_engine.clients.llm import get_client
from know_engine.config import get_settings

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
OPEN, CLOSE = "<think>", "</think>"


def clean(text: str | None) -> str:
    """Drop reasoning blocks and the leading blank lines some models emit."""
    return THINK_BLOCK.sub("", text or "").strip()


def render(template: str, **values: str) -> str:
    """Fill ``{{name}}`` placeholders (the Java prompt files use this syntax)."""
    for name, value in values.items():
        template = template.replace("{{" + name + "}}", value)
    return template


def _options() -> dict:
    # Thinking models (qwen3.x) spend 10-50 s reasoning on a one-line rewrite; "none" turns it off.
    effort = get_settings().llm_reasoning_effort
    return {"reasoning_effort": effort} if effort else {}


def complete(model: str, messages: list[dict], temperature: float = 0.0, timeout: float | None = None) -> str:
    """``timeout`` (seconds) also disables retries: for calls on the request path that have a
    fallback, failing fast beats waiting out a rate-limit backoff."""
    client = get_client() if timeout is None else get_client().with_options(timeout=timeout, max_retries=0)
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **_options())
    return clean(response.choices[0].message.content)


def strip_leading_think(pieces: Iterable[str]) -> Iterator[str]:
    """Pass streamed text through, minus leading whitespace and leading <think> blocks."""
    state, buffer = "start", ""
    for piece in pieces:
        if state == "body":
            yield piece
            continue
        buffer += piece
        while True:
            if state == "think":
                if CLOSE not in buffer:
                    break
                buffer, state = buffer.split(CLOSE, 1)[1], "start"
            stripped = buffer.lstrip()
            if not stripped or OPEN.startswith(stripped):
                buffer = stripped  # undecided: only whitespace, or a partial "<think>"
                break
            if stripped.startswith(OPEN):
                buffer, state = stripped[len(OPEN):], "think"
                continue
            state, buffer = "body", ""
            yield stripped
            break


def stream(model: str, messages: list[dict], temperature: float = 0.3) -> Iterator[str]:
    response = get_client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=True, **_options()
    )
    pieces = (e.choices[0].delta.content or "" for e in response if e.choices)
    return strip_leading_think(p for p in pieces if p)
