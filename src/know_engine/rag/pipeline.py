"""The RAG chain as plain function calls: rewrite -> vector + keyword retrieval -> RRF ->
parent replacement -> rerank (with fallback) -> prompt -> LLM.

Each step is timed, and every knob lives in RagConfig so evaluation runs can vary one at a time.
"""

import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from know_engine.clients.llm import embed_texts
from know_engine.config import get_settings
from know_engine.prompts import load_prompt
from know_engine.rag import llm
from know_engine.rag.query import rewrite_query
from know_engine.rag.retrieve import (
    Candidate,
    keyword_search,
    replace_with_parents,
    rerank,
    rrf_fuse,
    vector_search,
)

NO_CONTEXT = "(无检索上下文)"


@dataclass
class RagConfig:
    # Defaults are the tuned values (rounds R1-R6 in docs/design.md). The initial baseline
    # (R0) was: original rewrite prompt, top_k 5/5, rerank_min_score 0.6, no fallback,
    # reranking the parent text.
    use_rewrite: bool = True
    rewrite_prompt: str = "query-rewrite-v2-prompt.txt"  # R4: keeps entities / codes / status words
    vector_top_k: int = 10  # R3
    vector_min_score: float | None = None  # ES cosine score (0..1); R1 showed it cannot separate hits
    keyword_top_k: int = 10  # R3
    rrf_k: int = 60
    rerank_min_score: float | None = 0.1  # R1: cross-encoder score after sigmoid (0..1), calibrated
    rerank_top_n: int = 5
    rerank_fallback: bool = True  # R6
    rerank_on_children: bool = True  # R2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RagConfig":
        known = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Retrieval:
    query: str
    rewritten: str
    references: list[Candidate]
    used_fallback: bool
    timings_ms: dict[str, int] = field(default_factory=dict)
    candidate_count: int = 0


class _Timer:
    def __init__(self) -> None:
        self.timings: dict[str, int] = {}

    def run(self, name: str, fn, *args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.timings[name] = round((time.perf_counter() - start) * 1000)


def retrieve(question: str, history: list[tuple[str, str]] | None = None, config: RagConfig | None = None,
             progress: Callable[[str], None] = lambda _: None) -> Retrieval:
    config = config or RagConfig()
    timer = _Timer()

    rewritten = question
    if config.use_rewrite:
        progress("正在优化您的问题...")
        rewritten = timer.run("rewrite", rewrite_query, question, history or [], config.rewrite_prompt)

    progress("正在检索知识库内容...")

    def vector_path() -> list[Candidate]:
        vector = timer.run("embed", embed_texts, [rewritten])[0]
        return timer.run("vector_search", vector_search, vector, config.vector_top_k, config.vector_min_score)

    with ThreadPoolExecutor(max_workers=2) as pool:
        vector_future = pool.submit(vector_path)
        keyword_future = pool.submit(timer.run, "keyword_search", keyword_search, rewritten, config.keyword_top_k)
        vector_hits, keyword_hits = vector_future.result(), keyword_future.result()

    progress("正在对检索结果进行排序筛选...")
    fused, kept, used_fallback = timer.run("rerank", rank_candidates, rewritten, vector_hits, keyword_hits, config)
    return Retrieval(question, rewritten, kept, used_fallback, timer.timings, len(fused))


def rank_candidates(query: str, vector_hits: list[Candidate], keyword_hits: list[Candidate],
                    config: RagConfig) -> tuple[list[Candidate], list[Candidate], bool]:
    """RRF -> rerank -> parent replacement. Returns (all candidates, kept references, used_fallback).

    With ``rerank_on_children`` the cross-encoder scores the matched child chunk and the parent's
    full text is substituted afterwards; otherwise it scores the (often long) parent text, which it
    truncates at 512 tokens.
    """
    fused = rrf_fuse([vector_hits, keyword_hits], k=config.rrf_k)
    if config.rerank_on_children:
        kept, used_fallback = rerank(query, fused, config.rerank_min_score, len(fused), config.rerank_fallback)
        return fused, replace_with_parents(kept)[:config.rerank_top_n], used_fallback
    candidates = replace_with_parents(fused)
    kept, used_fallback = rerank(query, candidates, config.rerank_min_score, config.rerank_top_n, config.rerank_fallback)
    return candidates, kept, used_fallback


def build_messages(question: str, references: list[Candidate], history: list[tuple[str, str]] | None = None) -> list[dict]:
    """System prompt + history + the question with the retrieved chunks injected."""
    contents = "\n\n".join(c.text for c in references) or NO_CONTEXT
    user = llm.render(load_prompt("rag-content-injector-prompt.txt"), userMessage=question, contents=contents)
    messages = [{"role": "system", "content": load_prompt("generic-qa-prompt.txt")}]
    for role, text in history or []:
        messages.append({"role": "user" if role == "USER" else "assistant", "content": text})
    messages.append({"role": "user", "content": user})
    return messages


def answer(question: str, references: list[Candidate], history: list[tuple[str, str]] | None = None) -> str:
    return llm.complete(get_settings().chat_model, build_messages(question, references, history), temperature=0.3)


def answer_stream(question: str, references: list[Candidate],
                  history: list[tuple[str, str]] | None = None) -> Iterator[str]:
    return llm.stream(get_settings().chat_model, build_messages(question, references, history))


def reference_dicts(retrieval: Retrieval) -> list[dict]:
    """Shape of chat_message.rag_references."""
    return [
        {
            "documentId": str(c.doc_id),
            "url": f"/api/document/{c.doc_id}/file",
            "documentTitle": c.file_name,
            "chunkId": c.chunk_id,
            "coveredChunkIds": c.covered_chunk_ids,
            "chunkContent": c.text,
            "similarityScore": c.score,
            "rerankScore": c.rerank_score,
            "retrievalSource": str(c.source),
        }
        for c in retrieval.references
    ]
