"""Two-way retrieval from Elasticsearch, RRF fusion, reranking and parent replacement."""

import threading
from dataclasses import dataclass, field, replace
from functools import lru_cache

import structlog
from sqlalchemy import select

from know_engine.clients import es
from know_engine.config import get_settings
from know_engine.constants import RetrievalSource
from know_engine.db import session_scope
from know_engine.models import KnowledgeDocument, KnowledgeSegment

log = structlog.get_logger()

SOURCE_FIELDS = ["chunk_id", "doc_id", "version_id", "parent_chunk_id", "file_name", "text"]


@dataclass
class Candidate:
    chunk_id: str
    doc_id: int
    file_name: str
    text: str
    parent_chunk_id: str | None
    source: str
    score: float  # raw score from the retriever that found it
    fused_score: float = 0.0
    rerank_score: float | None = None
    sources: set = field(default_factory=set)
    # Chunks whose text this candidate carries: itself, plus sibling children folded into the same parent.
    covered_chunk_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.covered_chunk_ids:
            self.covered_chunk_ids = [self.chunk_id]


def _active_versions() -> list[int]:
    """Current version of every document. ES should only hold these (the indexer retires old
    versions), but filtering on them keeps a half-finished retirement from leaking stale text."""
    with session_scope() as s:
        return [v for v in s.scalars(select(KnowledgeDocument.current_version_id)) if v is not None]


def _to_candidates(hits: list[dict], source: str) -> list[Candidate]:
    return [
        Candidate(
            chunk_id=h["_source"]["chunk_id"], doc_id=h["_source"]["doc_id"], file_name=h["_source"]["file_name"],
            text=h["_source"]["text"], parent_chunk_id=h["_source"].get("parent_chunk_id"),
            source=source, score=h["_score"], sources={source},
        )
        for h in hits
    ]


def vector_search(vector: list[float], top_k: int, min_score: float | None) -> list[Candidate]:
    """KNN on cosine similarity. ES reports cosine as (1 + cos) / 2, so scores are in [0, 1]."""
    knn = {
        "field": "vector", "query_vector": vector, "k": top_k, "num_candidates": max(100, top_k * 10),
        "filter": {"terms": {"version_id": _active_versions()}},
    }
    if min_score is not None:
        knn["similarity"] = 2 * min_score - 1  # ES filters on raw cosine similarity
    hits = es.get_es().search(index=es.ensure_index(), knn=knn, size=top_k, source=SOURCE_FIELDS)["hits"]["hits"]
    return _to_candidates(hits, RetrievalSource.VECTOR)


def keyword_search(query: str, top_k: int) -> list[Candidate]:
    """BM25 over the IK-analyzed text field."""
    hits = es.get_es().search(
        index=es.ensure_index(), size=top_k, source=SOURCE_FIELDS,
        query={"bool": {"must": {"match": {"text": query}}, "filter": {"terms": {"version_id": _active_versions()}}}},
    )["hits"]["hits"]
    return _to_candidates(hits, RetrievalSource.KEYWORD)


def rrf_fuse(ranked_lists: list[list[Candidate]], k: int = 60) -> list[Candidate]:
    """Reciprocal Rank Fusion keyed by chunk_id; a chunk found by both retrievers is merged into one
    candidate whose score adds up and whose sources are merged."""
    fused: dict[str, Candidate] = {}
    for ranked in ranked_lists:
        for rank, candidate in enumerate(ranked, start=1):
            entry = fused.setdefault(candidate.chunk_id, replace(candidate, sources=set(), fused_score=0.0))
            entry.fused_score += 1.0 / (k + rank)
            entry.sources |= candidate.sources
    result = sorted(fused.values(), key=lambda c: c.fused_score, reverse=True)
    for c in result:
        c.source = RetrievalSource.BOTH if len(c.sources) > 1 else next(iter(c.sources))
    return result


def replace_with_parents(candidates: list[Candidate]) -> list[Candidate]:
    """Swap a child chunk's text for its parent's full text; keep one entry per parent.

    The child's chunk_id is kept, and children folded into an already-seen parent are recorded in
    ``covered_chunk_ids``: their text is in the context, so evaluation must count them as retrieved.
    """
    parent_ids = {c.parent_chunk_id for c in candidates if c.parent_chunk_id}
    if not parent_ids:
        return candidates
    with session_scope() as s:
        parents = dict(s.execute(
            select(KnowledgeSegment.chunk_id, KnowledgeSegment.text).where(KnowledgeSegment.chunk_id.in_(parent_ids))
        ).all())
    by_parent: dict[str, Candidate] = {}
    result = []
    for c in candidates:
        if c.parent_chunk_id and c.parent_chunk_id in parents:
            if kept := by_parent.get(c.parent_chunk_id):
                kept.covered_chunk_ids.append(c.chunk_id)
                continue
            c = replace(c, text=parents[c.parent_chunk_id], covered_chunk_ids=[c.chunk_id])
            by_parent[c.parent_chunk_id] = c
        result.append(c)
    return result


_RERANK_LOCK = threading.Lock()


@lru_cache
def _reranker():
    from sentence_transformers import CrossEncoder  # heavy import, only when first needed

    return CrossEncoder(get_settings().rerank_model, local_files_only=True)


def rerank(query: str, candidates: list[Candidate], min_score: float | None, top_n: int,
           fallback: bool) -> tuple[list[Candidate], bool]:
    """Score with the BGE cross-encoder (sigmoid, 0..1), drop those below ``min_score``, keep ``top_n``.

    Returns (kept, used_fallback). With ``fallback``, an empty result falls back to the RRF order
    instead of giving the LLM no context.
    """
    if not candidates:
        return [], False
    with _RERANK_LOCK:  # one model instance shared by request / eval threads
        scores = _reranker().predict([(query, c.text) for c in candidates], batch_size=16)
    for c, s in zip(candidates, scores):
        c.rerank_score = float(s)
    ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    kept = [c for c in ranked if min_score is None or c.rerank_score >= min_score][:top_n]
    if not kept and fallback:
        return sorted(candidates, key=lambda c: c.fused_score, reverse=True)[:top_n], True
    return kept, False
