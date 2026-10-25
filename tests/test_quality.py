from know_engine.ingest.quality import check
from know_engine.ingest.splitters.base import SKIP_EMBEDDING, Chunk


def test_clean_split_passes():
    report = check([Chunk("园区包含 A、B、C 三个生产区。"), Chunk("E-17 表示时钟偏差超过九十秒。")], 500, False)
    assert report["passed"] and not report["errors"]
    assert report["metrics"]["embeddable"] == 2


def test_garbled_text_is_rejected():
    report = check([Chunk("正常文本" * 10), Chunk("�" * 30)], 500, False)
    assert not report["passed"] and "乱码" in report["errors"][0]


def test_mostly_duplicates_is_rejected():
    report = check([Chunk("同一段")] * 5, 500, False)
    assert not report["passed"] and report["metrics"]["duplicateRatio"] == 0.8


def test_only_parents_is_rejected():
    report = check([Chunk("父分段全文", {SKIP_EMBEDDING: 1})], 500, False)
    assert not report["passed"]


def test_soft_warnings_do_not_fail():
    report = check([Chunk("短"), Chunk("这是一段足够长的正常文本内容。"), Chunk("x" * 600)], 500, False)
    assert report["passed"]
    assert any("少于" in w for w in report["warnings"]) and any("超过 chunkSize" in w for w in report["warnings"])
