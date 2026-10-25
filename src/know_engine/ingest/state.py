"""Status bookkeeping and per-document locking.

Every mutating operation on a document first locks its ``knowledge_document`` row:

- API requests lock with NOWAIT, so a concurrent request fails fast with a conflict
  (the Java version used Redisson locks with waitTime=0 for the same effect).
- The ingest worker locks with SKIP LOCKED and simply tries again later.

Since nobody ever waits on a lock while holding another, this cannot deadlock.
"""

from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from know_engine.ingest.errors import ConflictError, NotFoundError
from know_engine.models import KnowledgeDocument, KnowledgeDocumentVersion


def lock_document(session: Session, doc_id: int) -> KnowledgeDocument:
    stmt = select(KnowledgeDocument).where(KnowledgeDocument.doc_id == doc_id).with_for_update(nowait=True)
    try:
        doc = session.scalars(stmt).first()
    except OperationalError as e:
        if "could not obtain lock" in str(e) or "LockNotAvailable" in type(e.orig).__name__:
            raise ConflictError("文档正在处理中，请稍后再试") from e
        raise
    if doc is None:
        raise NotFoundError(f"文档不存在: {doc_id}")
    return doc


def get_version(session: Session, version_id: int, doc_id: int | None = None) -> KnowledgeDocumentVersion:
    version = session.get(KnowledgeDocumentVersion, version_id)
    if version is None:
        raise NotFoundError(f"版本不存在: {version_id}")
    if doc_id is not None and version.doc_id != doc_id:
        raise ConflictError("版本不属于该文档")
    return version


def set_status(session: Session, version: KnowledgeDocumentVersion, status: str) -> None:
    """Set a version's status; the document mirrors the status of its current version."""
    version.status = status
    doc = session.get(KnowledgeDocument, version.doc_id)
    if doc is not None and doc.current_version_id == version.version_id:
        doc.status = status
