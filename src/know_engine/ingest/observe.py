"""Pipeline observability: a log row per stage run, and statistics over it."""

import time
from contextlib import contextmanager

import numpy as np
import structlog
from sqlalchemy import func, literal_column, select

from know_engine.db import session_scope
from know_engine.models import IngestStageLog

log = structlog.get_logger()


def record(doc_id: int, version_id: int, stage: str, status: str, duration_ms: int,
           items: int = 0, detail: dict | None = None) -> None:
    """Written in its own transaction, so a failed stage is logged even though its work rolled back."""
    try:
        with session_scope() as s:
            s.add(IngestStageLog(doc_id=doc_id, version_id=version_id, stage=stage, status=status,
                                 duration_ms=duration_ms, items=items, detail=detail))
    except Exception as e:  # logging must never break the pipeline
        log.warning("stage_log_failed", stage=stage, error=str(e))


class StageRun:
    def __init__(self) -> None:
        self.items = 0
        self.detail: dict = {}


@contextmanager
def stage(doc_id: int, version_id: int, name: str):
    """Time a stage and log SUCCESS or FAILED (with the error) when it ends."""
    run, start = StageRun(), time.perf_counter()
    try:
        yield run
    except Exception as e:
        record(doc_id, version_id, name, "FAILED", round((time.perf_counter() - start) * 1000), run.items,
               {**run.detail, "error": str(e)[:1000]})
        raise
    record(doc_id, version_id, name, "SUCCESS", round((time.perf_counter() - start) * 1000), run.items, run.detail)


def stats(hours: int = 24 * 7) -> dict:
    """Per-stage runs, failures, latency percentiles and throughput; embedding reuse ratio."""
    with session_scope() as s:
        rows = s.execute(
            select(IngestStageLog.stage, IngestStageLog.status, IngestStageLog.duration_ms, IngestStageLog.items,
                   IngestStageLog.detail)
            .where(IngestStageLog.created_at >= func.now() - literal_column(f"interval '{int(hours)} hours'"))
        ).all()
    result: dict = {"windowHours": hours, "stages": {}}
    for name in sorted({r.stage for r in rows}):
        runs = [r for r in rows if r.stage == name]
        ok = [r for r in runs if r.status == "SUCCESS"]
        durations = [r.duration_ms for r in ok]
        items, seconds = sum(r.items for r in ok), sum(durations) / 1000
        result["stages"][name] = {
            "runs": len(runs), "failed": len(runs) - len(ok), "items": items,
            "p50Ms": round(float(np.percentile(durations, 50))) if durations else None,
            "p95Ms": round(float(np.percentile(durations, 95))) if durations else None,
            "itemsPerSecond": round(items / seconds, 2) if seconds else None,
        }
    embeds = [r.detail or {} for r in rows if r.stage == "embed" and r.status == "SUCCESS"]
    reused, computed = sum(d.get("reused", 0) for d in embeds), sum(d.get("computed", 0) for d in embeds)
    result["embeddingReuse"] = {"reused": reused, "computed": computed,
                                "ratio": round(reused / (reused + computed), 4) if reused + computed else None}
    return result
