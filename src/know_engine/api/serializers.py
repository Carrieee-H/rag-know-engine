"""JSON shapes matching the Java version's Jackson output, which the static pages depend on.

Field names are camelCase; ``extension`` and segment ``metadata`` are JSON strings and
``skipEmbedding`` is 0/1, because the pages call JSON.parse / compare with 1.
"""

import json
import math
from datetime import datetime

from know_engine.models import KnowledgeDocument, KnowledgeDocumentVersion, KnowledgeSegment


def _ts(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _json_str(value) -> str | None:
    return json.dumps(value, ensure_ascii=False) if value is not None else None


def document(d: KnowledgeDocument | None) -> dict | None:
    if d is None:
        return None
    return {
        "docId": d.doc_id,
        "docTitle": d.doc_title,
        "expireDate": d.expire_date.isoformat() if d.expire_date else None,
        "status": d.status,
        "description": d.description,
        "extension": _json_str(d.extension),
        "currentVersionId": d.current_version_id,
        "createdAt": _ts(d.created_at),
        "updatedAt": _ts(d.updated_at),
        "lockVersion": d.lock_version,
        "deleted": int(d.deleted),
    }


def version(v: KnowledgeDocumentVersion) -> dict:
    return {
        "versionId": v.version_id,
        "docId": v.doc_id,
        "version": v.version,
        "docUrl": v.doc_url,
        "convertedDocUrl": v.converted_doc_url,
        "contentHash": v.content_hash,
        "status": v.status,
        "uploadUser": v.upload_user,
        "changelog": v.changelog,
        "attempts": v.attempts,
        "lastError": v.last_error,
        "createdAt": _ts(v.created_at),
        "updatedAt": _ts(v.updated_at),
        "lockVersion": v.lock_version,
    }


def segment(s: KnowledgeSegment | None) -> dict | None:
    if s is None:
        return None
    return {
        "id": s.id,
        "text": s.text,
        "chunkId": s.chunk_id,
        "metadata": _json_str(s.metadata_),
        "documentId": s.document_id,
        "documentVersion": s.document_version,
        "chunkOrder": s.chunk_order,
        "embeddingId": s.embedding_id,
        "status": s.status,
        "skipEmbedding": int(s.skip_embedding),
        "createdAt": _ts(s.created_at),
        "updatedAt": _ts(s.updated_at),
        "lockVersion": s.lock_version,
    }


def page(records: list[dict], total: int, current: int, size: int) -> dict:
    """MyBatis-Plus Page shape."""
    return {
        "records": records,
        "total": total,
        "size": size,
        "current": current,
        "pages": math.ceil(total / size) if size else 0,
    }
