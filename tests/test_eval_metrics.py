from know_engine.eval.metrics import NO_CONTEXT, format_contexts, parse_judge, retrieval_scores
from know_engine.rag.llm import strip_leading_think
from know_engine.rag.retrieve import Candidate, rrf_fuse


def test_retrieval_scores_chunk_level():
    s = retrieval_scores([{"a"}, {"b"}, {"c"}], ["1", "1", "2"], ["b", "x"], ["1"], top_k=3)
    assert s == {"hitAtK": True, "mrr": 0.5, "recallAtK": 0.5}


def test_retrieval_scores_counts_children_folded_into_a_parent():
    s = retrieval_scores([{"p1", "child2"}], ["1"], ["child2"], [], top_k=5)
    assert s["hitAtK"] and s["mrr"] == 1.0


def test_retrieval_scores_respects_top_k_and_doc_fallback():
    assert retrieval_scores([{"a"}, {"b"}], [], ["b"], [], top_k=1)["hitAtK"] is False
    assert retrieval_scores([{"a"}], ["9", "3"], [], ["3"], top_k=2)["mrr"] == 0.5
    assert retrieval_scores([{"a"}], ["1"], [], [], top_k=5) is None


def test_parse_judge_variants():
    assert parse_judge('{"score": 0.8, "reason": "ok"}').score == 0.8
    assert parse_judge('好的\n```json\n{"score": 1.5, "reason": "x"}\n```').score == 1.0
    assert parse_judge('结果：{"reason": "r", "score": 0}').score == 0.0


def test_format_contexts():
    assert format_contexts([]) == NO_CONTEXT
    assert format_contexts(["甲", " ", "乙"]) == "[片段1]\n甲\n\n[片段2]\n乙"


def _c(chunk_id, source):
    return Candidate(chunk_id, 1, "f", chunk_id, None, source, 1.0, sources={source})


def test_rrf_merges_by_chunk_id():
    fused = rrf_fuse([[_c("a", "VECTOR"), _c("b", "VECTOR")], [_c("b", "KEYWORD"), _c("c", "KEYWORD")]], k=60)
    assert fused[0].chunk_id == "b"
    assert fused[0].source == "BOTH"
    assert abs(fused[0].fused_score - (1 / 62 + 1 / 61)) < 1e-12


def test_strip_leading_think_across_chunks():
    pieces = ["\n\n<thi", "nk>reasoning", "...</th", "ink>\n", "答", "案"]
    assert "".join(strip_leading_think(pieces)) == "答案"
    assert "".join(strip_leading_think(["\n\n", "你好", "<think>不是开头"])) == "你好<think>不是开头"
