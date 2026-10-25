from pathlib import Path

import pytest

from know_engine.constants import SplitType
from know_engine.ingest.splitters import SplitParams, split_document
from know_engine.ingest.splitters.base import (
    CHUNK_ID,
    PARENT_CHUNK_ID,
    SKIP_EMBEDDING,
    merge_parts,
    window_split,
)
from know_engine.ingest.splitters.markdown import split_by_title, split_smart

TEST_FILES = Path(__file__).resolve().parents[1] / "data" / "testFile"

MD = """# 园区概览
介绍文字

## 基本信息
内部代号是 SG-01

## 运行
### 值班
值班说明
```
# 代码块里的井号不是标题
```
"""


def test_window_split_overlap():
    assert window_split("abcdefghij", 4, 1) == ["abcd", "defg", "ghij"]


def test_merge_parts_respects_size_and_overlap():
    segments = merge_parts(["aa", "bb", "cc", "dd"], size=5, overlap=2, joiner=" ")
    assert segments == ["aa bb", "bb cc", "cc dd"]
    assert all(len(s) <= 5 for s in segments)


def test_merge_parts_cuts_oversized_part():
    assert merge_parts(["x" * 10], size=4, overlap=0, joiner=" ") == ["xxxx", "xxxx", "xx"]


def test_title_split_sections_and_header_path():
    chunks = split_by_title(MD, title_level=2, chunk_size=1000, overlap=0)
    texts = [c.text for c in chunks]
    assert len(chunks) == 3
    assert texts[0].startswith("# 园区概览")
    assert chunks[1].metadata["title"] == "园区概览"
    assert chunks[1].metadata["subtitle"] == "基本信息"
    # level-3 header is below title_level, so it stays inside the "运行" section
    assert "### 值班" in texts[2]
    assert len({c.metadata[CHUNK_ID] for c in chunks}) == 3


def test_header_only_section_is_merged_into_child():
    chunks = split_by_title("# A\n## B\n正文", title_level=2, chunk_size=1000, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == "# A  \n## B\n正文"


def test_code_fence_hash_is_not_a_header():
    chunks = split_smart(MD, chunk_size=1000)
    assert any("# 代码块里的井号不是标题" in c.text for c in chunks)
    assert not any(c.metadata.get("title") == "代码块里的井号不是标题" for c in chunks)


def test_oversized_section_becomes_parent_and_children():
    text = "# 长章节\n" + "字" * 250
    chunks = split_by_title(text, title_level=1, chunk_size=100, overlap=10)
    parent, children = chunks[0], chunks[1:]
    assert parent.metadata[SKIP_EMBEDDING] == 1 and parent.text == text
    assert all(c.metadata[PARENT_CHUNK_ID] == parent.metadata[CHUNK_ID] for c in children)
    assert all(len(c.text) <= 100 for c in children)
    assert len(children) == 3


def test_invalid_title_level():
    with pytest.raises(ValueError):
        split_by_title(MD, title_level=7, chunk_size=100, overlap=0)


@pytest.mark.parametrize("name", ["05_曙光智造园区资产与维护清单.xlsx", "06_曙光智造园区事件记录.csv"])
def test_table_files_one_chunk_per_row(name):
    data = (TEST_FILES / name).read_bytes()
    chunks = split_document(data, name, SplitParams(SplitType.LENGTH, chunk_size=500))
    assert chunks, "table produced no chunks"
    assert all("：" in c.text for c in chunks)
    assert all(CHUNK_ID in c.metadata for c in chunks)


def test_markdown_sample_with_title_strategy():
    name = "01_曙光智造园区概览.md"
    chunks = split_document((TEST_FILES / name).read_bytes(), name, SplitParams(SplitType.TITLE, 500, 50, 2))
    assert chunks
    assert len({c.metadata[CHUNK_ID] for c in chunks}) == len(chunks)


def test_regex_strategy_on_txt_sample():
    name = "02_曙光智造园区运行手册.txt"
    chunks = split_document((TEST_FILES / name).read_bytes(), name, SplitParams(SplitType.REGEX, 300, 0, regex=r"\n\s*\n"))
    assert chunks and all(len(c.text) <= 300 for c in chunks)
