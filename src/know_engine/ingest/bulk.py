"""Bulk ingestion of a directory (backfill), with throughput reporting.

Idempotent: a file whose content is already stored is skipped, so an interrupted run can
simply be started again. Upload, conversion and splitting run on a thread pool; embedding
is done by ``--workers`` ingest workers in parallel, each claiming different versions through
``FOR UPDATE SKIP LOCKED``.

    python -m know_engine.ingest.bulk data/corpus/python-zh --pattern "*.txt" --split LENGTH --chunk-size 500
"""

import argparse
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import structlog
from sqlalchemy import func, select

from know_engine.constants import DocumentStatus, SegmentStatus, SplitType
from know_engine.db import session_scope
from know_engine.ingest import service, worker
from know_engine.ingest.errors import ConflictError
from know_engine.ingest.splitters import SplitParams
from know_engine.models import KnowledgeDocument, KnowledgeDocumentVersion, KnowledgeSegment

log = structlog.get_logger()


def _existing_hashes() -> set[str]:
    with session_scope() as s:
        return set(s.scalars(select(KnowledgeDocumentVersion.content_hash)))


def _ingest_file(path: Path, root: Path, params: SplitParams, collection: str) -> tuple[str, int | None]:
    title = f"{collection}/{path.relative_to(root)}"
    try:
        doc = service.upload(path.name, path.read_bytes(), title, f"批量导入：{collection}")
        service.split(doc.doc_id, params)
        return "ingested", doc.doc_id
    except ConflictError as e:
        return f"rejected: {e}", None


def _pending(doc_ids: list[int]) -> int:
    with session_scope() as s:
        return s.scalar(select(func.count()).select_from(KnowledgeDocument).where(
            KnowledgeDocument.doc_id.in_(doc_ids),
            KnowledgeDocument.status.notin_([DocumentStatus.VECTOR_STORED, DocumentStatus.FAILED]),
        ))


def run(root: Path, pattern: str, params: SplitParams, collection: str, threads: int, workers: int) -> dict:
    files = sorted(root.rglob(pattern))
    known = _existing_hashes()
    todo = [p for p in files if hashlib.sha256(p.read_bytes()).hexdigest() not in known]
    log.info("bulk_started", files=len(files), new=len(todo), skipped=len(files) - len(todo))

    stop = threading.Event()
    for _ in range(workers):
        threading.Thread(target=worker.run_forever, args=(stop,), daemon=True).start()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(lambda p: _ingest_file(p, root, params, collection), todo))
    split_seconds = time.perf_counter() - started
    doc_ids = [doc_id for _, doc_id in results if doc_id]
    while doc_ids and _pending(doc_ids):
        time.sleep(2)
    stop.set()
    total_seconds = time.perf_counter() - started

    with session_scope() as s:
        chunks = s.scalar(select(func.count()).select_from(KnowledgeSegment).where(
            KnowledgeSegment.document_id.in_(doc_ids), KnowledgeSegment.status == SegmentStatus.VECTOR_STORED,
        )) if doc_ids else 0
        failed = s.scalar(select(func.count()).select_from(KnowledgeDocument).where(
            KnowledgeDocument.doc_id.in_(doc_ids), KnowledgeDocument.status == DocumentStatus.FAILED,
        )) if doc_ids else 0
    rejected = [r for r, _ in results if r.startswith("rejected")]
    return {
        "files": len(files), "skippedExisting": len(files) - len(todo), "ingested": len(doc_ids),
        "rejected": len(rejected), "rejectedExamples": rejected[:3], "failedEmbedding": failed,
        "chunksIndexed": chunks, "uploadSplitSeconds": round(split_seconds, 1), "totalSeconds": round(total_seconds, 1),
        "docsPerMinute": round(len(doc_ids) / total_seconds * 60, 1) if total_seconds else None,
        "chunksPerSecond": round(chunks / total_seconds, 1) if total_seconds else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--collection", default=None, help="title prefix; defaults to the directory name")
    parser.add_argument("--split", default="LENGTH", choices=[t.value for t in SplitType])
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--title-level", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4, help="upload/convert/split threads")
    parser.add_argument("--workers", type=int, default=3, help="embedding workers")
    args = parser.parse_args()
    params = SplitParams(SplitType(args.split), args.chunk_size, args.overlap, args.title_level)
    print(run(args.root, args.pattern, params, args.collection or args.root.name, args.threads, args.workers))


if __name__ == "__main__":
    main()
