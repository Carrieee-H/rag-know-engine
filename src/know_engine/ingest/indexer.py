"""Embedding, Elasticsearch indexing and version retirement."""

import hashlib

import structlog
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from know_engine.clients import es
from know_engine.clients.llm import embed_texts
from know_engine.config import get_settings
from know_engine.constants import DocumentStatus, SegmentStatus
from know_engine.ingest.splitters.base import PARENT_CHUNK_ID
from know_engine.ingest.state import set_status
from know_engine.models import KnowledgeDocument, KnowledgeDocumentVersion, KnowledgeSegment

log = structlog.get_logger()

INDEX_BATCH_SIZE = 64


def text_hash(text: str) -> str:
    """Vectors depend on the model as well as the text, so both go into the key."""
    return hashlib.sha256(f"{get_settings().embedding_model}\n{text}".encode()).hexdigest()


def es_document(segment: KnowledgeSegment, doc: KnowledgeDocument, vector: list[float]) -> dict:
    metadata = segment.metadata_ or {}
    return {
        "chunk_id": segment.chunk_id,
        "text_hash": segment.text_hash,
        "doc_id": segment.document_id,
        "version_id": segment.document_version,
        "parent_chunk_id": metadata.get(PARENT_CHUNK_ID),
        "file_name": doc.doc_title,
        "text": segment.text,
        "vector": vector,
        "metadata": metadata,
    }


def index_segments(segments: list[KnowledgeSegment], doc: KnowledgeDocument) -> dict:
    """Embed and index segments, then mark them VECTOR_STORED. Returns reuse counts.

    A segment whose text is already indexed (typically an unchanged chunk of the previous
    version, which stays searchable until this one is done) reuses that vector instead of
    calling the embedding API.

    ES writes happen first and are idempotent (``_id`` = chunk_id), so a failure part-way
    leaves the database untouched and a retry simply overwrites what was already written.
    """
    reused = computed = 0
    for start in range(0, len(segments), INDEX_BATCH_SIZE):
        batch = segments[start:start + INDEX_BATCH_SIZE]
        for segment in batch:
            segment.text_hash = text_hash(segment.text)
        known = es.find_vectors(sorted({s.text_hash for s in batch}))
        missing = [s for s in batch if s.text_hash not in known]
        for segment, vector in zip(missing, embed_texts([s.text for s in missing]) if missing else []):
            known[segment.text_hash] = vector
        reused, computed = reused + len(batch) - len(missing), computed + len(missing)
        es.bulk_upsert([es_document(s, doc, known[s.text_hash]) for s in batch])
        for segment in batch:
            segment.status = SegmentStatus.VECTOR_STORED
            segment.embedding_id = segment.chunk_id
    return {"reused": reused, "computed": computed}


def embed_version(session: Session, version: KnowledgeDocumentVersion, doc: KnowledgeDocument) -> dict:
    """Index every pending segment of a CHUNKED version and make it the only searchable version."""
    pending = session.scalars(
        select(KnowledgeSegment)
        .where(
            KnowledgeSegment.document_version == version.version_id,
            KnowledgeSegment.status == SegmentStatus.STORED,
            KnowledgeSegment.skip_embedding.is_(False),
        )
        .order_by(KnowledgeSegment.chunk_order)
    ).all()
    counts = index_segments(list(pending), doc)
    session.flush()

    expected = session.query(KnowledgeSegment).filter(
        KnowledgeSegment.document_version == version.version_id, KnowledgeSegment.skip_embedding.is_(False)
    ).count()
    indexed = es.count_version(doc.doc_id, version.version_id)
    if indexed != expected:
        raise RuntimeError(f"ES 中的分段数 {indexed} 与应索引的分段数 {expected} 不一致")

    set_status(session, version, DocumentStatus.VECTOR_STORED)
    retire_other_versions(session, doc, keep_version_id=version.version_id)
    log.info("version_indexed", doc_id=doc.doc_id, version_id=version.version_id, **counts)
    return counts


def unindex_version(session: Session, version: KnowledgeDocumentVersion) -> None:
    """Remove a version from ES and move it back to CHUNKED (its segments are kept)."""
    es.delete_version(version.doc_id, version.version_id)
    session.execute(
        update(KnowledgeSegment)
        .where(KnowledgeSegment.document_version == version.version_id)
        .values(status=SegmentStatus.STORED, embedding_id=None)
    )
    if version.status == DocumentStatus.VECTOR_STORED:
        set_status(session, version, DocumentStatus.CHUNKED)


def retire_other_versions(session: Session, doc: KnowledgeDocument, keep_version_id: int) -> None:
    es.delete_other_versions(doc.doc_id, keep_version_id)
    others = session.scalars(
        select(KnowledgeDocumentVersion).where(
            KnowledgeDocumentVersion.doc_id == doc.doc_id,
            KnowledgeDocumentVersion.version_id != keep_version_id,
            KnowledgeDocumentVersion.status == DocumentStatus.VECTOR_STORED,
        )
    ).all()
    for other in others:
        unindex_version(session, other)
