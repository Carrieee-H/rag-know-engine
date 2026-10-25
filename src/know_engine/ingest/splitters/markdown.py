"""Markdown header splitter with parent/child chunks (TITLE and SMART strategies).

Port of the Java MarkdownHeaderParentTextSplitter:

1. Blank lines are removed, then the text is cut at every header of level <= ``title_level``
   (headers inside fenced code blocks are ignored). Each section carries its header path as
   metadata and gets its own chunkId.
2. When ``merge_header_only`` is on (TITLE), a section that ends with a header line is merged
   with the deeper section that follows it.
3. A section longer than ``chunk_size`` is kept whole as a parent (skipEmbedding=1, not
   indexed) and also cut into character windows that point to it via parentChunkId.
"""

from know_engine.ingest.splitters.base import (
    CHUNK_ID,
    HEADER_LEVEL,
    PARENT_CHUNK_ID,
    SKIP_EMBEDDING,
    Chunk,
    new_chunk_id,
    window_split,
)

HEADER_NAMES = ["title", "subtitle", "subsubtitle", "subsubsubtitle", "subsubsubsubtitle", "subsubsubsubsubtitle"]


def _header_level(line: str, max_level: int) -> int | None:
    """Level of a Markdown ATX header (``## Title``), or None."""
    hashes = len(line) - len(line.lstrip("#"))
    if 1 <= hashes <= max_level and (len(line) == hashes or line[hashes] == " "):
        return hashes
    return None


def _sections(text: str, title_level: int) -> list[Chunk]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sections: list[Chunk] = []
    content: list[str] = []
    header_meta: dict = {}  # header path at the current position
    section_meta: dict = {}  # metadata of the section being collected
    fence = ""

    for line in lines:
        if fence:
            if line.startswith(fence):
                fence = ""
            content.append(line)
            continue
        if line.startswith(("```", "~~~")):
            fence = line[:3]
            content.append(line)
            continue

        level = _header_level(line, title_level)
        if level is None:
            content.append(line)
            continue

        if content:
            sections.append(Chunk("\n".join(content), dict(section_meta)))
            content = []
        # Pop headers at the same or a deeper level, then push this one.
        for deeper in range(level, 7):
            header_meta.pop(HEADER_NAMES[deeper - 1], None)
        header_meta[HEADER_NAMES[level - 1]] = line[level:].strip()
        header_meta[HEADER_LEVEL] = level
        header_meta[CHUNK_ID] = new_chunk_id()
        section_meta = dict(header_meta)
        content.append(line)

    if content:
        sections.append(Chunk("\n".join(content), dict(section_meta)))
    return sections


def _merge_header_only(sections: list[Chunk]) -> list[Chunk]:
    merged: list[Chunk] = []
    for section in sections:
        if merged:
            last = merged[-1]
            if last.text.split("\n")[-1].startswith("#") and len(last.metadata) < len(section.metadata):
                last.text = f"{last.text}  \n{section.text}"
                continue
        merged.append(section)
    return merged


def _split_oversized(sections: list[Chunk], chunk_size: int, overlap: int) -> list[Chunk]:
    result: list[Chunk] = []
    for section in sections:
        if len(section.text) <= chunk_size:
            result.append(section)
            continue
        parent_id = new_chunk_id()
        result.append(Chunk(section.text, {**section.metadata, CHUNK_ID: parent_id, SKIP_EMBEDDING: 1}))
        for piece in window_split(section.text, chunk_size, overlap):
            result.append(Chunk(piece, {**section.metadata, CHUNK_ID: new_chunk_id(), PARENT_CHUNK_ID: parent_id}))
    return result


def split_by_title(text: str, title_level: int, chunk_size: int, overlap: int) -> list[Chunk]:
    """TITLE: cut at headers up to ``title_level`` and merge header-only sections."""
    if not 1 <= title_level <= 6:
        raise ValueError(f"titleLevel 必须在 1 到 6 之间，当前为 {title_level}")
    sections = _merge_header_only(_sections(text, title_level))
    return _split_oversized(sections, chunk_size, overlap)


def split_smart(text: str, chunk_size: int) -> list[Chunk]:
    """SMART: every header level, no merging, overlap fixed at 10% of chunk_size."""
    return _split_oversized(_sections(text, 6), chunk_size, int(chunk_size * 0.1))
