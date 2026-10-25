from know_engine.api.chat import sse
from know_engine.eval.runner import EvalConfig
from know_engine.ingest.dictionary import _table_terms
from know_engine.ingest.reindex import _next_name
from know_engine.rag.pipeline import RagConfig


def test_sse_framing():
    frames = list(sse(["[PROGRESS]:正在检索", "第一行\n第二行", ""]))
    assert frames[0] == "data:[PROGRESS]:正在检索\n\n"
    # A multi-line payload becomes several data: lines in one event (the page joins them with \n).
    assert frames[1] == "data:第一行\ndata:第二行\n\n"
    assert frames[2] == "data:\n\n"


def test_reindex_next_name():
    assert _next_name("know-engine-chunks", "know-engine-chunks-v1") == "know-engine-chunks-v2"
    assert _next_name("know-engine-chunks", "know-engine-chunks-v12") == "know-engine-chunks-v13"
    assert _next_name("alias", "some-index") == "alias-v2"


def test_table_terms_only_name_columns():
    row = "资产编号：AU-RB-007; 资产名称：七号装配机器人; 责任团队：机器人组; 根本原因：安全门触点松动"
    assert _table_terms(row) == {"七号装配机器人", "机器人组"}


def test_eval_config_round_trip_keeps_rag_overrides():
    config = EvalConfig(top_k=5, label="x", rag=RagConfig(vector_top_k=10, rerank_min_score=0.1))
    again = EvalConfig.from_json(config.to_json())
    assert again.rag == config.rag and again.label == "x"
    assert EvalConfig.from_json({"topK": None}).top_k == 5


def test_rag_config_ignores_unknown_fields():
    assert RagConfig.from_dict({"vector_top_k": 7, "nope": 1}).vector_top_k == 7
