"""Document lifecycle: upload, new version, split, version switching, deletion and segment edits.

Embedding is not done here: splitting leaves the version CHUNKED and the ingest worker
picks it up (see worker.py).
"""

import hashlib

import structlog
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from know_engine.clients import es, storage
from know_engine.constants import DocumentStatus, SegmentStatus
from know_engine.db import session_scope
from know_engine.ingest import converters, observe, quality
from know_engine.ingest.errors import ConflictError, NotFoundError
from know_engine.ingest.indexer import index_segments, unindex_version
from know_engine.ingest.reindex import require_lock_for_write
from know_engine.ingest.splitters import SplitParams, is_table, split_document
from know_engine.ingest.splitters.base import CHUNK_ID, DOC_ID, FILE_NAME, SKIP_EMBEDDING, URL, VERSION
from know_engine.ingest.state import get_version, lock_document, set_status
from know_engine.models import KnowledgeDocument, KnowledgeDocumentVersion, KnowledgeSegment

log = structlog.get_logger()

DEFAULT_USER = "admin"


def _version_key(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in version.strip().split("."))
    except ValueError as e:
        raise ConflictError(f"版本号格式不正确: {version}，应为 1.0.0 这样的数字格式") from e


def _check_not_duplicate(session: Session, content_hash: str) -> None:
    """Reject content that already exists in any document or version.

    The transaction-scoped advisory lock on the hash serializes concurrent uploads of the
    same file, so two of them cannot both pass the check.
    """
    session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:h, 0))"), {"h": content_hash})
    exists = session.scalar(
        select(func.count()).select_from(KnowledgeDocumentVersion).where(
            KnowledgeDocumentVersion.content_hash == content_hash, KnowledgeDocumentVersion.deleted.is_(False)
        )
    )
    if exists:
        raise ConflictError("文档内容已存在，请勿重复上传")


def _store_raw(session: Session, doc: KnowledgeDocument, version: KnowledgeDocumentVersion,
               file_name: str, data: bytes) -> None:
    session.flush()  # assign ids
    version.doc_url = storage.put_bytes(converters.raw_object_name(doc.doc_id, version.version_id, file_name), data)


def _convert(version_id: int, file_name: str, data: bytes, make_current: bool) -> KnowledgeDocument:
    """UPLOADED -> CONVERTING -> CONVERTED. Runs outside any lock: MinerU can take minutes."""
    with session_scope() as s:
        version = get_version(s, version_id)
        set_status(s, version, DocumentStatus.CONVERTING)
        doc_id, raw_uri = version.doc_id, version.doc_url
    try:
        with observe.stage(doc_id, version_id, "convert") as run:
            converted_uri = converters.convert(data, file_name, doc_id, version_id, raw_uri)
            run.items, run.detail = 1, {"file": file_name, "bytes": len(data)}
    except Exception as e:
        with session_scope() as s:
            version = get_version(s, version_id)
            set_status(s, version, DocumentStatus.UPLOADED)
            version.last_error = f"转换失败: {e}"[:2000]
        log.error("convert_failed", doc_id=doc_id, version_id=version_id, error=str(e))
        raise
    with session_scope() as s:
        doc = lock_document(s, doc_id)
        version = get_version(s, version_id)
        version.converted_doc_url = converted_uri
        version.last_error = None
        if make_current:
            # The old version stays searchable until this one is embedded.
            doc.current_version_id = version_id
        set_status(s, version, DocumentStatus.CONVERTED)
        return doc


def upload(file_name: str, data: bytes, title: str, description: str | None,
           version: str = "1.0.0", upload_user: str | None = None) -> KnowledgeDocument:
    converters.check_supported(file_name)
    _version_key(version)
    content_hash = hashlib.sha256(data).hexdigest()
    with session_scope() as s:
        _check_not_duplicate(s, content_hash)
        doc = KnowledgeDocument(doc_title=title, description=description, status=DocumentStatus.UPLOADED)
        s.add(doc)
        s.flush()
        record = KnowledgeDocumentVersion(
            doc_id=doc.doc_id, version=version, content_hash=content_hash,
            status=DocumentStatus.UPLOADED, upload_user=upload_user or DEFAULT_USER,
        )
        s.add(record)
        _store_raw(s, doc, record, file_name, data)
        doc.current_version_id = record.version_id
        version_id = record.version_id
    log.info("document_uploaded", doc_id=doc.doc_id, version_id=version_id, file=file_name)
    return _convert(version_id, file_name, data, make_current=False)


def upload_new_version(doc_id: int, version: str, file_name: str, data: bytes,
                       changelog: str | None = None, upload_user: str | None = None) -> KnowledgeDocument:
    converters.check_supported(file_name)
    content_hash = hashlib.sha256(data).hexdigest()
    with session_scope() as s:
        lock_document(s, doc_id)
        latest = s.scalars(
            select(KnowledgeDocumentVersion.version).where(KnowledgeDocumentVersion.doc_id == doc_id)
        ).all()
        if latest and _version_key(version) <= max(_version_key(v) for v in latest):
            newest = max(latest, key=_version_key)
            raise ConflictError(f"版本号 {version} 不大于现有最新版本号 {newest}，请使用更大的版本号")
        _check_not_duplicate(s, content_hash)
        record = KnowledgeDocumentVersion(
            doc_id=doc_id, version=version, content_hash=content_hash, status=DocumentStatus.UPLOADED,
            upload_user=upload_user or DEFAULT_USER, changelog=changelog,
        )
        s.add(record)
        _store_raw(s, s.get(KnowledgeDocument, doc_id), record, file_name, data)
        version_id = record.version_id
    log.info("version_uploaded", doc_id=doc_id, version_id=version_id, version=version)
    return _convert(version_id, file_name, data, make_current=True)


def split(doc_id: int, params: SplitParams) -> int:
    """Split the current version: CONVERTED -> CHUNKED. Returns the number of segments saved."""
    with session_scope() as s:
        doc = lock_document(s, doc_id)
        if doc.current_version_id is None:
            raise ConflictError("文档没有可切分的版本")
        version = get_version(s, doc.current_version_id)
        if version.status == DocumentStatus.CHUNKED:
            # Already split: idempotent, answer with the number of segments to embed.
            return s.scalar(select(func.count()).select_from(KnowledgeSegment).where(
                KnowledgeSegment.document_version == version.version_id, KnowledgeSegment.skip_embedding.is_(False)
            ))
        if version.status != DocumentStatus.CONVERTED or not version.converted_doc_url:
            raise ConflictError(f"文档状态为 {version.status}，只有 CONVERTED 状态的文档可以切分")

        converted_name = converters.converted_file_name(doc.doc_title, version.converted_doc_url)
        with observe.stage(doc.doc_id, version.version_id, "split") as run:
            chunks = split_document(storage.get_bytes(version.converted_doc_url), converted_name, params)
            run.items, run.detail = len(chunks), {"params": params.to_dict()}
        if not chunks:
            raise ConflictError("切分结果为空，请检查文档内容或切分参数")

        report = quality.check(chunks, params.chunk_size, is_table(converted_name))
        observe.record(doc.doc_id, version.version_id, "quality", "SUCCESS" if report["passed"] else "FAILED",
                       0, len(chunks), report)
        if not report["passed"]:
            raise ConflictError("切分结果未通过数据质量检查：" + "；".join(report["errors"]))
        chunks = [c for c in chunks if c.text.strip()]

        for order, chunk in enumerate(chunks):
            metadata = {
                **chunk.metadata,
                DOC_ID: doc.doc_id, FILE_NAME: doc.doc_title, URL: version.doc_url, VERSION: version.version_id,
            }
            s.add(KnowledgeSegment(
                text=chunk.text, chunk_id=chunk.metadata[CHUNK_ID], metadata_=metadata,
                document_id=doc.doc_id, document_version=version.version_id, chunk_order=order,
                status=SegmentStatus.STORED, skip_embedding=chunk.metadata.get(SKIP_EMBEDDING) == 1,
            ))
        doc.extension = {**(doc.extension or {}), "splitParam": params.to_dict(), "quality": report}
        set_status(s, version, DocumentStatus.CHUNKED)
        log.info("document_split", doc_id=doc_id, version_id=version.version_id, segments=len(chunks))
        return len(chunks)


def switch_version(doc_id: int, version_id: int) -> KnowledgeDocument:
    """Make a version current. The worker embeds it (if needed) and retires the old one.

    Switching to a FAILED version (including the current one) resets it for another round of retries.
    """
    with session_scope() as s:
        doc = lock_document(s, doc_id)
        version = get_version(s, version_id, doc_id)
        if version.status == DocumentStatus.FAILED:
            version.status, version.attempts, version.last_error = DocumentStatus.CHUNKED, 0, None
        elif doc.current_version_id == version_id:
            return doc
        if version.status not in (DocumentStatus.CHUNKED, DocumentStatus.VECTOR_STORED):
            raise ConflictError(f"目标版本状态为 {version.status}，需要先完成切分才能切换")
        doc.current_version_id = version_id
        doc.status = version.status
        log.info("version_switched", doc_id=doc_id, version_id=version_id)
        return doc


def deactivate_version(version_id: int) -> None:
    with session_scope() as s:
        version = get_version(s, version_id)
        doc = lock_document(s, version.doc_id)
        if doc.current_version_id == version_id:
            # The worker would immediately re-embed a current CHUNKED version.
            raise ConflictError("不能让当前版本失效，请先切换到其他版本")
        require_lock_for_write(s)
        unindex_version(s, version)


def delete_documents(doc_ids: list[int]) -> bool:
    for doc_id in doc_ids:
        with session_scope() as s:
            doc = lock_document(s, doc_id)
            require_lock_for_write(s)
            es.delete_documents([doc_id])
            s.delete(doc)  # versions and segments go with it (ON DELETE CASCADE)
        for prefix in (f"raw/{doc_id}/", f"converted/{doc_id}/"):
            try:
                storage.remove_prefix(prefix)
            except Exception as e:  # orphaned files are harmless; the rows are gone
                log.warning("minio_cleanup_failed", prefix=prefix, error=str(e))
        log.info("document_deleted", doc_id=doc_id)
    return True


def update_document(doc_id: int, title: str | None, description: str | None, lock_version: int | None) -> bool:
    with session_scope() as s:
        doc = lock_document(s, doc_id)
        if lock_version is not None and lock_version != doc.lock_version:
            raise ConflictError("文档已被修改，请刷新后重试")
        if title:
            doc.doc_title = title
        if description is not None:
            doc.description = description
    return True


def update_segment(segment_id: int, new_text: str, lock_version: int | None) -> bool:
    """Edit a segment's text; an indexed segment is re-embedded right away."""
    with session_scope() as s:
        segment = s.get(KnowledgeSegment, segment_id)
        if segment is None:
            raise NotFoundError(f"分段不存在: {segment_id}")
        doc = lock_document(s, segment.document_id)
        if segment.skip_embedding:
            raise ConflictError("父分段不支持直接修改，请修改对应的子分段")
        if lock_version is not None and lock_version != segment.lock_version:
            raise ConflictError("分段已被修改，请刷新后重试")
        if new_text == segment.text:
            return True
        segment.text = new_text
        if segment.status == SegmentStatus.VECTOR_STORED:
            require_lock_for_write(s)
            index_segments([segment], doc)
    return True


def delete_segment(segment_id: int) -> bool:
    with session_scope() as s:
        segment = s.get(KnowledgeSegment, segment_id)
        if segment is None:
            return False
        lock_document(s, segment.document_id)
        require_lock_for_write(s)
        es.get_es().options(ignore_status=404).delete(index=es.ensure_index(), id=segment.chunk_id, refresh=True)
        s.delete(segment)
    return True
