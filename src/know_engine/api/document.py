"""/api/document and /api/segment, same routes and parameters as the Java controllers."""

import mimetypes
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Query, Response, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select

from know_engine.api import serializers
from know_engine.clients import storage
from know_engine.constants import DocumentStatus, SplitType
from know_engine.db import session_scope
from know_engine.ingest import observe, service
from know_engine.ingest.errors import ConflictError, NotFoundError
from know_engine.ingest.splitters import SplitParams
from know_engine.ingest.state import get_version
from know_engine.models import IngestStageLog, KnowledgeDocument, KnowledgeDocumentVersion, KnowledgeSegment

router = APIRouter()


def _opt_int(value: str | None, name: str) -> int | None:
    """The upload page sends empty strings for unused parameters."""
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as e:
        raise ConflictError(f"参数 {name} 必须是整数") from e


def _opt_str(value: str | None) -> str | None:
    return value if value else None


# ---------- documents ----------

@router.post("/api/document/upload")
def upload(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str | None = Form(None),
    version: str = Form("1.0.0"),
    uploadUser: str | None = Form(None),
):
    doc = service.upload(file.filename or title, file.file.read(), title, description, version or "1.0.0", uploadUser)
    return serializers.document(doc)


@router.post("/api/document/upload-version")
def upload_version(
    file: UploadFile = File(...),
    docId: int = Form(...),
    version: str = Form(...),
    changelog: str | None = Form(None),
    uploadUser: str | None = Form(None),
):
    doc = service.upload_new_version(docId, version, file.filename or "upload", file.file.read(), changelog, uploadUser)
    return serializers.document(doc)


@router.get("/api/document/versions/{doc_id}")
def list_versions(doc_id: int):
    with session_scope() as s:
        rows = s.scalars(
            select(KnowledgeDocumentVersion)
            .where(KnowledgeDocumentVersion.doc_id == doc_id)
            .order_by(KnowledgeDocumentVersion.version_id.desc())
        ).all()
        return [serializers.version(v) for v in rows]


@router.post("/api/document/switch-version")
def switch_version(docId: int, versionId: int):
    return serializers.document(service.switch_version(docId, versionId))


@router.post("/api/document/deactivate-version")
def deactivate_version(versionId: int):
    service.deactivate_version(versionId)


@router.post("/api/document/activate-version")
def activate_version(versionId: int):
    """Same as switching to that version: only one version of a document is searchable at a time."""
    with session_scope() as s:
        doc_id = get_version(s, versionId).doc_id
    service.switch_version(doc_id, versionId)


@router.post("/api/document/split/{document_id}")
def split(
    document_id: int,
    splitType: str = Query(...),
    chunkSize: str = Query(...),
    overlap: str | None = None,
    regex: str | None = None,
    titleLevel: str | None = None,
    separator: str | None = None,
) -> int:
    try:
        split_type = SplitType(splitType)
    except ValueError as e:
        raise ConflictError(f"不支持的切分方式: {splitType}") from e
    params = SplitParams(
        split_type=split_type,
        chunk_size=_opt_int(chunkSize, "chunkSize") or 0,
        overlap=_opt_int(overlap, "overlap") or 0,
        title_level=_opt_int(titleLevel, "titleLevel"),
        regex=_opt_str(regex),
        separator=_opt_str(separator),
    )
    return service.split(document_id, params)


@router.get("/api/document/page")
def page(current: int = 1, size: int = 10, docTitle: str | None = None, status: str | None = None):
    with session_scope() as s:
        stmt = select(KnowledgeDocument).where(KnowledgeDocument.deleted.is_(False))
        if docTitle:
            stmt = stmt.where(KnowledgeDocument.doc_title.ilike(f"%{docTitle}%"))
        if status:
            stmt = stmt.where(KnowledgeDocument.status == status)
        total = s.scalar(select(func.count()).select_from(stmt.subquery()))
        rows = s.scalars(
            stmt.order_by(KnowledgeDocument.created_at.desc()).offset((current - 1) * size).limit(size)
        ).all()
        return serializers.page([serializers.document(d) for d in rows], total, current, size)


@router.get("/api/document/{doc_id:int}/file")
def download_current_file(doc_id: int):
    """Original file of the current version (the chat page links references here)."""
    with session_scope() as s:
        doc = s.get(KnowledgeDocument, doc_id)
        version = s.get(KnowledgeDocumentVersion, doc.current_version_id) if doc and doc.current_version_id else None
        if version is None or not version.doc_url:
            raise NotFoundError(f"文档不存在: {doc_id}")
        uri = version.doc_url
    name = uri.rsplit("/", 1)[-1]
    media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return Response(storage.get_bytes(uri), media_type=media_type,
                    headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(name)}"})


@router.get("/api/document/{doc_id:int}")
def get_document(doc_id: int):
    with session_scope() as s:
        return serializers.document(s.get(KnowledgeDocument, doc_id))


class DocumentBody(BaseModel):
    docId: int | None = None
    docTitle: str | None = None
    description: str | None = None
    status: str | None = None
    lockVersion: int | None = None


@router.post("/api/document")
def create_document(body: DocumentBody) -> bool:
    """Create a document record without a file (kept for the page's "new" form)."""
    if not body.docTitle:
        raise ConflictError("请输入文档标题")
    with session_scope() as s:
        s.add(KnowledgeDocument(doc_title=body.docTitle, description=body.description, status=DocumentStatus.UPLOADED))
    return True


@router.put("/api/document")
def update_document(body: DocumentBody) -> bool:
    if body.docId is None:
        raise ConflictError("缺少 docId")
    return service.update_document(body.docId, body.docTitle, body.description, body.lockVersion)


@router.delete("/api/document/batch")
def delete_batch(ids: list[str] = Query(...)) -> bool:
    # Accept both ?ids=1,2 (Spring style, used by the page) and ?ids=1&ids=2.
    doc_ids = [int(x) for item in ids for x in item.split(",") if x.strip()]
    return service.delete_documents(doc_ids)


@router.delete("/api/document/{doc_id:int}")
def delete_document(doc_id: int) -> bool:
    return service.delete_documents([doc_id])


# ---------- segments ----------

def _segment_filter(stmt, documentId: int, documentVersion: int | None):
    stmt = stmt.where(KnowledgeSegment.document_id == documentId)
    if documentVersion is not None:
        stmt = stmt.where(KnowledgeSegment.document_version == documentVersion)
    return stmt


@router.get("/api/segment/page-by-document")
def segment_page(documentId: int, documentVersion: int | None = None, current: int = 1, size: int = 10):
    with session_scope() as s:
        stmt = _segment_filter(select(KnowledgeSegment), documentId, documentVersion)
        total = s.scalar(select(func.count()).select_from(stmt.subquery()))
        rows = s.scalars(stmt.order_by(KnowledgeSegment.chunk_order).offset((current - 1) * size).limit(size)).all()
        return serializers.page([serializers.segment(r) for r in rows], total, current, size)


@router.get("/api/segment/count-by-document")
def segment_count(documentId: int, documentVersion: int | None = None) -> int:
    with session_scope() as s:
        stmt = _segment_filter(select(func.count()).select_from(KnowledgeSegment), documentId, documentVersion)
        return s.scalar(stmt)


@router.get("/api/segment/{segment_id:int}")
def get_segment(segment_id: int):
    with session_scope() as s:
        return serializers.segment(s.get(KnowledgeSegment, segment_id))


class SegmentBody(BaseModel):
    id: int
    text: str
    lockVersion: int | None = None


@router.put("/api/segment")
def update_segment(body: SegmentBody) -> bool:
    return service.update_segment(body.id, body.text, body.lockVersion)


@router.delete("/api/segment/{segment_id:int}")
def delete_segment(segment_id: int) -> bool:
    return service.delete_segment(segment_id)


@router.get("/api/ingest/stats")
def ingest_stats(hours: int = 24 * 7):
    """Per-stage runs, failures, p50/p95 latency, throughput and embedding reuse."""
    return observe.stats(hours)


@router.get("/api/document/versions/{version_id:int}/lineage")
def version_lineage(version_id: int):
    """Stage history of a version: what ran, when, how long, with which parameters and results."""
    with session_scope() as s:
        rows = s.scalars(select(IngestStageLog).where(IngestStageLog.version_id == version_id)
                         .order_by(IngestStageLog.id)).all()
        return [{"stage": r.stage, "status": r.status, "durationMs": r.duration_ms, "items": r.items,
                 "detail": r.detail, "createdAt": r.created_at.isoformat()} for r in rows]
