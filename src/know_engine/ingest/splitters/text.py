"""LENGTH, REGEX and SEPARATOR strategies.

Sizes are in characters.
"""

import re

from know_engine.ingest.splitters.base import Chunk, merge_parts


def split_by_length(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Pack whitespace-separated words up to ``chunk_size`` characters."""
    return [Chunk(t) for t in merge_parts(re.split(r"\s+", text), chunk_size, overlap, " ")]


def split_by_regex(text: str, pattern: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Cut at matches of ``pattern``, then pack the pieces up to ``chunk_size`` characters.

    SEPARATOR uses this too: the separator is interpreted as a regex.
    """
    if not pattern:
        raise ValueError("REGEX / SEPARATOR 切分需要提供正则表达式或分隔符")
    try:
        parts = re.split(pattern, text)
    except re.error as e:
        raise ValueError(f"正则表达式无效: {e}") from e
    return [Chunk(t) for t in merge_parts(parts, chunk_size, overlap, "\n\n")]
