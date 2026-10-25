"""Dagster orchestration for batch ingestion.

- ``source_files`` partitions: one dynamic partition per file under the watched directories.
- ``new_files_sensor``: discovers new files and requests a run for each.
- ``ingested_document`` (partitioned asset): upload + convert + split; idempotent, a file that
  is already stored is reused instead of failing.
- ``indexed_document`` (partitioned asset): waits until the ingest worker has embedded and
  indexed the document, and records chunk counts and embedding reuse as metadata.
- ``quality_gate`` asset check: the split passed the data-quality rules.
- ``reconcile_job`` (daily schedule): PostgreSQL vs Elasticsearch consistency check.

Backfills are ordinary Dagster partition backfills. Embedding itself stays with the ingest
worker, so the API, the bulk CLI and Dagster all share one state-driven path.

    dagster dev -m know_engine.orchestration.definitions
"""

import hashlib
import time
from pathlib import Path

from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    AssetKey,
    Definitions,
    DynamicPartitionsDefinition,
    MaterializeResult,
    RunRequest,
    ScheduleDefinition,
    SensorEvaluationContext,
    SensorResult,
    asset,
    asset_check,
    define_asset_job,
    job,
    op,
    sensor,
)
from sqlalchemy import func, select

from know_engine.config import PROJECT_ROOT
from know_engine.constants import DocumentStatus, SplitType
from know_engine.db import session_scope
from know_engine.ingest import reconcile, service
from know_engine.ingest.splitters import SplitParams
from know_engine.models import IngestStageLog, KnowledgeDocument, KnowledgeDocumentVersion, KnowledgeSegment

WATCHED = {"testFile": PROJECT_ROOT / "data" / "testFile", "python-zh": PROJECT_ROOT / "data" / "corpus" / "python-zh"}
PATTERNS = ("*.md", "*.txt", "*.pdf", "*.docx", "*.xlsx", "*.csv")
INDEX_TIMEOUT_SECONDS = 900

source_files = DynamicPartitionsDefinition(name="source_files")


def _path(partition_key: str) -> Path:
    collection, _, relative = partition_key.partition(":")
    return WATCHED[collection] / relative


def _split_params(path: Path) -> SplitParams:
    if path.suffix in (".md", ".pdf", ".docx"):
        return SplitParams(SplitType.TITLE, 500, 50, 2)
    return SplitParams(SplitType.LENGTH, 500, 50)


def _doc_for_hash(content_hash: str) -> KnowledgeDocument | None:
    with session_scope() as s:
        doc_id = s.scalar(select(KnowledgeDocumentVersion.doc_id).where(
            KnowledgeDocumentVersion.content_hash == content_hash))
        return s.get(KnowledgeDocument, doc_id) if doc_id else None


@asset(partitions_def=source_files, group_name="ingestion")
def ingested_document(context: AssetExecutionContext) -> MaterializeResult:
    path = _path(context.partition_key)
    data = path.read_bytes()
    doc = _doc_for_hash(hashlib.sha256(data).hexdigest())
    reused = doc is not None
    if doc is None:
        doc = service.upload(path.name, data, context.partition_key, "Dagster 批量导入")
    if doc.status == DocumentStatus.CONVERTED:
        service.split(doc.doc_id, _split_params(path))
    with session_scope() as s:
        doc = s.get(KnowledgeDocument, doc.doc_id)
        chunks = s.scalar(select(func.count()).select_from(KnowledgeSegment)
                          .where(KnowledgeSegment.document_version == doc.current_version_id))
    return MaterializeResult(metadata={"doc_id": doc.doc_id, "version_id": doc.current_version_id,
                                       "status": doc.status, "chunks": chunks, "already_stored": reused})


@asset(partitions_def=source_files, deps=[ingested_document], group_name="ingestion")
def indexed_document(context: AssetExecutionContext) -> MaterializeResult:
    doc = _doc_for_hash(hashlib.sha256(_path(context.partition_key).read_bytes()).hexdigest())
    if doc is None:
        raise RuntimeError("document not found; materialize ingested_document first")
    deadline = time.monotonic() + INDEX_TIMEOUT_SECONDS
    while True:
        with session_scope() as s:
            doc = s.get(KnowledgeDocument, doc.doc_id)
            log = s.scalars(select(IngestStageLog).where(
                IngestStageLog.version_id == doc.current_version_id, IngestStageLog.stage == "embed",
                IngestStageLog.status == "SUCCESS").order_by(IngestStageLog.id.desc())).first()
        if doc.status == DocumentStatus.VECTOR_STORED:
            detail = (log.detail or {}) if log else {}
            return MaterializeResult(metadata={"doc_id": doc.doc_id, "reused_vectors": detail.get("reused", 0),
                                               "computed_vectors": detail.get("computed", 0),
                                               "embed_ms": log.duration_ms if log else None})
        if doc.status == DocumentStatus.FAILED or time.monotonic() > deadline:
            raise RuntimeError(f"document {doc.doc_id} not indexed (status {doc.status})")
        time.sleep(2)


@asset_check(asset=ingested_document)
def quality_gate(context) -> AssetCheckResult:
    """Latest quality report of the partition's current version."""
    doc = _doc_for_hash(hashlib.sha256(_path(context.partition_key).read_bytes()).hexdigest()) \
        if getattr(context, "partition_key", None) else None
    if doc is None:
        return AssetCheckResult(passed=True, metadata={"note": "no partition context"})
    report = (doc.extension or {}).get("quality") or {}
    return AssetCheckResult(passed=bool(report.get("passed", False)),
                            metadata={"errors": str(report.get("errors")), "warnings": str(report.get("warnings"))})


ingest_job = define_asset_job("ingest_job", selection=[AssetKey("ingested_document"), AssetKey("indexed_document")],
                              partitions_def=source_files)


@sensor(job=ingest_job, minimum_interval_seconds=60)
def new_files_sensor(context: SensorEvaluationContext) -> SensorResult:
    keys = sorted(
        f"{collection}:{p.relative_to(root)}"
        for collection, root in WATCHED.items() if root.exists()
        for pattern in PATTERNS for p in root.rglob(pattern)
    )
    new = [k for k in keys if not context.instance.has_dynamic_partition("source_files", k)]
    return SensorResult(
        run_requests=[RunRequest(partition_key=k, run_key=k) for k in new],
        dynamic_partitions_requests=[source_files.build_add_request(new)] if new else [],
    )


@op
def reconcile_index(context) -> dict:
    report = reconcile.reconcile(repair=True)
    context.log.info(str(report))
    if not report["consistent"]:
        context.log.warning("index was inconsistent; affected versions queued for re-indexing")
    return report


@job
def reconcile_job():
    reconcile_index()


defs = Definitions(
    assets=[ingested_document, indexed_document],
    asset_checks=[quality_gate],
    jobs=[ingest_job, reconcile_job],
    sensors=[new_files_sensor],
    schedules=[ScheduleDefinition(job=reconcile_job, cron_schedule="0 3 * * *")],
)
