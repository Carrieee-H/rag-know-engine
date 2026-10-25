"""XLSX / CSV: one chunk per data row, written as ``header：value; header：value``.

Only the first sheet is read and the first row is the header.
"""

import io
import re

import pandas as pd

from know_engine.ingest.splitters.base import Chunk

# Control characters except tab and newline.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MIDNIGHT = " 00:00:00"


def _read_rows(data: bytes, file_name: str) -> list[list[str]]:
    if file_name.lower().endswith(".csv"):
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                frame = pd.read_csv(io.BytesIO(data), header=None, dtype=str, keep_default_na=False, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("无法识别 CSV 文件编码（已尝试 UTF-8 与 GB18030）")
    else:
        frame = pd.read_excel(io.BytesIO(data), sheet_name=0, header=None, dtype=str, keep_default_na=False)
    return frame.values.tolist()


def _clean(cell) -> str:
    text = _CONTROL_CHARS.sub("", str(cell)).strip()
    # Date-only Excel cells come back as "YYYY-MM-DD 00:00:00".
    return text.removesuffix(_MIDNIGHT)


def split_table(data: bytes, file_name: str) -> list[Chunk]:
    rows = [[_clean(c) for c in row] for row in _read_rows(data, file_name)]
    if len(rows) < 2:
        return []
    headers, chunks = rows[0], []
    for row in rows[1:]:
        pairs = [f"{h}：{v}" for h, v in zip(headers, row) if h or v]
        if pairs and any(v for v in row):
            chunks.append(Chunk("; ".join(pairs)))
    return chunks
