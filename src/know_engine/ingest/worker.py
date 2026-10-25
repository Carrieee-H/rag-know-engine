"""State-driven ingest worker.

Embedding is driven by database state rather than by events: the worker keeps polling for
CHUNKED versions that are their document's current version, claims one with
``FOR UPDATE SKIP LOCKED`` and embeds it. Any number of workers can run at once without
processing the same version twice, and a crash simply leaves the version CHUNKED for the next
poll, so no separate compensation job is needed.

Failures are retried with exponential backoff; after MAX_ATTEMPTS the version is marked FAILED.

Run standalone with ``python -m know_engine.ingest.worker``; the API also starts one in-process.
"""

import threading

import structlog
from sqlalchemy import func, literal_column, select

from know_engine.constants import DocumentStatus
from know_engine.db import session_scope
from know_engine.ingest import observe
from know_engine.ingest.indexer import embed_version
from know_engine.ingest.reindex import try_lock_for_write
from know_engine.ingest.state import set_status
from know_engine.models import KnowledgeDocument, KnowledgeDocumentVersion

log = structlog.get_logger()

MAX_ATTEMPTS = 5
# A version that failed n times waits 5 * (2^n - 1) seconds: 0, 5, 15, 35, 75.
BACKOFF_BASE = literal_column("interval '5 seconds'")
POLL_INTERVAL_SECONDS = 2.0


def _claim_query():
    v = KnowledgeDocumentVersion
    return (
        select(v)
        .join(KnowledgeDocument, KnowledgeDocument.current_version_id == v.version_id)
        .where(
            v.status == DocumentStatus.CHUNKED,
            v.deleted.is_(False),
            v.attempts < MAX_ATTEMPTS,
            v.updated_at <= func.now() - BACKOFF_BASE * func.power(2, v.attempts) + BACKOFF_BASE,
        )
        .order_by(v.updated_at)
        .limit(1)
        .with_for_update(of=v, skip_locked=True)
    )


def process_one() -> bool:
    """Embed one pending version. Returns False when there was nothing to claim."""
    with session_scope() as s:
        version = s.scalars(_claim_query()).first()
        if version is None:
            return False
        doc = s.scalars(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.doc_id == version.doc_id)
            .with_for_update(skip_locked=True)
        ).first()
        if doc is None:
            return False  # an API request holds the document; try again next round
        if not try_lock_for_write(s):
            return False  # the index is being rebuilt; try again next round
        version_id, doc_id = version.version_id, doc.doc_id
        try:
            with s.begin_nested(), observe.stage(doc_id, version_id, "embed") as run:
                counts = embed_version(s, version, doc)
                run.items, run.detail = counts["reused"] + counts["computed"], counts
            log.info("worker_embedded", doc_id=doc_id, version_id=version_id, **counts)
            return True
        except Exception as e:
            # The savepoint rolled back the partial work; record the failure in the same transaction.
            version.attempts += 1
            version.last_error = str(e)[:2000]
            if version.attempts >= MAX_ATTEMPTS:
                set_status(s, version, DocumentStatus.FAILED)
            log.error("worker_embed_failed", doc_id=doc_id, version_id=version_id, attempts=version.attempts, error=str(e))
            return True


def run_forever(stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    log.info("worker_started")
    while not stop.is_set():
        try:
            busy = process_one()
        except Exception as e:  # e.g. database briefly unavailable
            log.error("worker_loop_error", error=str(e))
            busy = False
        if not busy:
            stop.wait(POLL_INTERVAL_SECONDS)


def start_background() -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=run_forever, args=(stop,), name="ingest-worker", daemon=True).start()
    return stop


if __name__ == "__main__":
    try:
        run_forever()
    except KeyboardInterrupt:
        pass
