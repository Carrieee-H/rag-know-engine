"""Data-quality gate run on the chunks of every split, before anything is stored.

Hard rules reject the split (the version stays CONVERTED, so it can be re-split with other
parameters); soft rules only produce warnings. The report is kept in the stage log and in
the document's ``extension`` (visible on the document page).
"""

from collections import Counter

from know_engine.ingest.splitters.base import Chunk

GARBLED_MAX = 0.02  # share of replacement / control / private-use characters
DUPLICATE_MAX = 0.5  # share of embeddable chunks whose text repeats another
SHORT_CHUNK_CHARS = 10
SHORT_WARN = 0.2


def _is_garbled(ch: str) -> bool:
    code = ord(ch)
    return ch == "�" or 0xE000 <= code <= 0xF8FF or (code < 32 and ch not in "\n\t\r")


def check(chunks: list[Chunk], chunk_size: int, is_table: bool) -> dict:
    embeddable = [c for c in chunks if not c.skip_embedding]
    texts = [c.text.strip() for c in embeddable]
    all_chars = "".join(texts)
    garbled = sum(1 for ch in all_chars if _is_garbled(ch)) / max(1, len(all_chars))
    counts = Counter(texts)
    duplicates = sum(n - 1 for n in counts.values() if n > 1) / max(1, len(texts))
    empty = sum(1 for t in texts if not t)
    short = sum(1 for t in texts if 0 < len(t) < SHORT_CHUNK_CHARS)
    oversized = 0 if is_table else sum(1 for t in texts if len(t) > chunk_size)
    lengths = sorted(len(t) for t in texts)

    metrics = {
        "chunks": len(chunks), "embeddable": len(embeddable), "parents": len(chunks) - len(embeddable),
        "emptyChunks": empty, "shortChunks": short, "oversizedChunks": oversized,
        "duplicateRatio": round(duplicates, 4), "garbledRatio": round(garbled, 5),
        "minLength": lengths[0] if lengths else 0, "medianLength": lengths[len(lengths) // 2] if lengths else 0,
        "maxLength": lengths[-1] if lengths else 0,
    }
    errors, warnings = [], []
    if not embeddable or empty == len(texts):
        errors.append("没有可向量化的分块")
    if garbled > GARBLED_MAX:
        errors.append(f"乱码字符占比 {garbled:.2%} 超过 {GARBLED_MAX:.0%}，请检查文档编码或解析结果")
    if duplicates > DUPLICATE_MAX:
        errors.append(f"重复分块占比 {duplicates:.0%} 超过 {DUPLICATE_MAX:.0%}，请检查切分参数")
    if empty:
        warnings.append(f"{empty} 个空分块将被跳过")
    if texts and short / len(texts) > SHORT_WARN:
        warnings.append(f"{short} 个分块少于 {SHORT_CHUNK_CHARS} 个字符，可能切得过碎")
    if oversized:
        warnings.append(f"{oversized} 个分块超过 chunkSize={chunk_size}")
    return {"passed": not errors, "errors": errors, "warnings": warnings, "metrics": metrics}
