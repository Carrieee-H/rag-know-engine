import uuid
from dataclasses import dataclass, field

# Metadata keys; camelCase because the frontend reads them.
CHUNK_ID = "chunkId"
PARENT_CHUNK_ID = "parentChunkId"
SKIP_EMBEDDING = "skipEmbedding"
HEADER_LEVEL = "headerLevel"
DOC_ID = "docId"
FILE_NAME = "fileName"
VERSION = "version"
URL = "url"


def new_chunk_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def skip_embedding(self) -> bool:
        return self.metadata.get(SKIP_EMBEDDING) == 1


def window_split(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character windows; each window starts ``overlap`` characters before the previous end."""
    if size <= 0:
        raise ValueError("chunkSize 必须大于 0")
    overlap = max(0, min(overlap, size - 1))
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return pieces


def merge_parts(parts: list[str], size: int, overlap: int, joiner: str) -> list[str]:
    """Greedily pack parts into segments of at most ``size`` characters.

    A part longer than ``size`` is cut into character windows. When a segment is closed,
    the next one starts with as many trailing parts of it as fit in ``overlap`` characters.
    """
    if size <= 0:
        raise ValueError("chunkSize 必须大于 0")
    segments: list[str] = []
    current: list[str] = []

    def length(items: list[str]) -> int:
        return sum(len(p) for p in items) + len(joiner) * max(0, len(items) - 1)

    def close() -> list[str]:
        segments.append(joiner.join(current))
        carried: list[str] = []
        for p in reversed(current):
            if length([p, *carried]) > overlap:
                break
            carried.insert(0, p)
        return carried

    for part in (p.strip() for p in parts):
        if not part:
            continue
        if len(part) > size:
            if current:
                close()
                current = []
            segments.extend(window_split(part, size, overlap))
            continue
        if current and length([*current, part]) > size:
            current = close()
            # Drop carried parts that would not leave room for the new one.
            while current and length([*current, part]) > size:
                current.pop(0)
        current.append(part)
    if current:
        segments.append(joiner.join(current))
    return segments
