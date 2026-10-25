"""Reconcile PostgreSQL (source of truth) with the Elasticsearch index.

Checks, per document: the current version's indexable chunks are all in ES, nothing else of
that document is in ES, and ES holds no chunks of documents that no longer exist. With
``repair``, a version with missing chunks is put back to CHUNKED so the worker re-indexes it
(idempotent), and orphaned ES documents are deleted.

    python -m know_engine.ingest.reconcile [--repair]
"""

import argparse

from sqlalchemy import func, select, update

from know_engine.clients import es
from know_engine.constants import DocumentStatus, SegmentStatus
from know_engine.db import session_scope
from know_engine.models import KnowledgeDocument, KnowledgeDocumentVersion, KnowledgeSegment


def _es_counts() -> dict[tuple[int, int], int]:
    """(doc_id, version_id) -> number of ES documents, via a composite aggregation."""
    counts, after = {}, None
    while True:
        composite = {"size": 1000, "sources": [{"doc": {"terms": {"field": "doc_id"}}},
                                               {"version": {"terms": {"field": "version_id"}}}]}
        if after:
            composite["after"] = after
        result = es.get_es().search(index=es.ensure_index(), size=0, aggs={"pairs": {"composite": composite}})
        agg = result["aggregations"]["pairs"]
        for bucket in agg["buckets"]:
            counts[(bucket["key"]["doc"], bucket["key"]["version"])] = bucket["doc_count"]
        after = agg.get("after_key")
        if not after or not agg["buckets"]:
            return counts


def reconcile(repair: bool = False) -> dict:
    es_counts = _es_counts()
    with session_scope() as s:
        expected = dict(s.execute(
            select(KnowledgeSegment.document_version, func.count())
            .join(KnowledgeDocument, KnowledgeDocument.current_version_id == KnowledgeSegment.document_version)
            .where(KnowledgeSegment.skip_embedding.is_(False))
            .group_by(KnowledgeSegment.document_version)
        ).all())
        versions = {v.version_id: v for v in s.scalars(select(KnowledgeDocumentVersion).where(
            KnowledgeDocumentVersion.version_id.in_(list(expected) or [-1])))}
        doc_ids = set(s.scalars(select(KnowledgeDocument.doc_id)))

        missing, stale, orphans = [], [], []
        for version_id, count in expected.items():
            v = versions[version_id]
            indexed = es_counts.get((v.doc_id, version_id), 0)
            if v.status == DocumentStatus.VECTOR_STORED and indexed != count:
                missing.append({"docId": v.doc_id, "versionId": version_id, "expected": count, "indexed": indexed})
        current = {(v.doc_id, vid) for vid, v in versions.items()}
        for (doc_id, version_id), n in es_counts.items():
            if doc_id not in doc_ids:
                orphans.append({"docId": doc_id, "versionId": version_id, "count": n})
            elif (doc_id, version_id) not in current:
                stale.append({"docId": doc_id, "versionId": version_id, "count": n})

        if repair:
            for item in missing:
                s.execute(update(KnowledgeSegment).where(KnowledgeSegment.document_version == item["versionId"])
                          .values(status=SegmentStatus.STORED))
                s.execute(update(KnowledgeDocumentVersion)
                          .where(KnowledgeDocumentVersion.version_id == item["versionId"])
                          .values(status=DocumentStatus.CHUNKED, attempts=0))
                s.execute(update(KnowledgeDocument).where(KnowledgeDocument.doc_id == item["docId"])
                          .values(status=DocumentStatus.CHUNKED))
            for item in orphans + stale:
                es.delete_version(item["docId"], item["versionId"])

    return {
        "versionsChecked": len(expected), "esDocuments": sum(es_counts.values()),
        "missing": missing, "staleVersionsInEs": stale, "orphansInEs": orphans,
        "consistent": not (missing or stale or orphans), "repaired": repair,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", action="store_true")
    print(reconcile(parser.parse_args().repair))
