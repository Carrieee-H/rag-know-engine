from dataclasses import asdict, dataclass

from know_engine.constants import SplitType
from know_engine.ingest.splitters.base import CHUNK_ID, Chunk, new_chunk_id
from know_engine.ingest.splitters.markdown import split_by_title, split_smart
from know_engine.ingest.splitters.table import split_table
from know_engine.ingest.splitters.text import split_by_length, split_by_regex

TABLE_EXTENSIONS = (".xlsx", ".xls", ".csv")


@dataclass
class SplitParams:
    split_type: SplitType
    chunk_size: int
    overlap: int = 0
    title_level: int | None = None
    regex: str | None = None
    separator: str | None = None

    def to_dict(self) -> dict:
        """Camel-case form stored in knowledge_document.extension (the frontend reads it back)."""
        d = asdict(self)
        return {
            "splitType": str(d["split_type"]), "chunkSize": d["chunk_size"], "overlap": d["overlap"],
            "titleLevel": d["title_level"], "regex": d["regex"], "separator": d["separator"],
        }


def is_table(file_name: str) -> bool:
    return file_name.lower().endswith(TABLE_EXTENSIONS)


def split_document(data: bytes, file_name: str, params: SplitParams) -> list[Chunk]:
    """Split a converted document; every returned chunk has a chunkId."""
    if params.chunk_size is None or params.chunk_size < 1:
        raise ValueError("chunkSize 必须大于 0")
    if is_table(file_name):
        # Tables are always one chunk per row, whatever strategy was requested.
        chunks = split_table(data, file_name)
    else:
        text = data.decode("utf-8", errors="replace")
        overlap = params.overlap or 0
        match params.split_type:
            case SplitType.LENGTH:
                chunks = split_by_length(text, params.chunk_size, overlap)
            case SplitType.TITLE:
                chunks = split_by_title(text, params.title_level or 1, params.chunk_size, overlap)
            case SplitType.SMART:
                chunks = split_smart(text, params.chunk_size)
            case SplitType.REGEX:
                chunks = split_by_regex(text, params.regex or "", params.chunk_size, overlap)
            case SplitType.SEPARATOR:
                chunks = split_by_regex(text, params.separator or "", params.chunk_size, overlap)
            case _:
                raise ValueError(f"不支持的切分方式: {params.split_type}")
    for chunk in chunks:
        chunk.metadata.setdefault(CHUNK_ID, new_chunk_id())
    return chunks
