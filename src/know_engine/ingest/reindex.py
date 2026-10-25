"""Zero-downtime index rebuild (blue/green behind the alias).

1. Create ``<alias>-v{N+1}`` with the current mapping. Analyzers are resolved now, so a new
   IK dictionary takes effect for everything copied in.
2. Copy all documents with ES ``_reindex``. Vectors are copied too; nothing is re-embedded.
3. Verify: same document count, and sample chunks are found by both BM25 and KNN.
4. Atomically move the alias, then optionally delete the old index.

Searches keep using the old index until step 4. Index writers (the ingest worker, segment edits)
take a shared advisory lock; the rebuild takes it exclusively, so no write can land in the old
index after the copy started and then be lost in the swap.

    python -m know_engine.ingest.reindex [--delete-old]
"""

import argparse
import random
import re
import time

import structlog
from sqlalchemy import text

from know_engine.clients import es
from know_engine.config import get_settings
from know_engine.db import engine
from know_engine.ingest.errors import ConflictError

log = structlog.get_logger()

INDEX_WRITE_LOCK = 734_001  # advisory lock key shared by index writers and the rebuild
SAMPLE_SIZE = 20


def try_lock_for_write(session) -> bool:
    """Shared lock for code that writes to ES; False while a rebuild is running."""
    return bool(session.execute(text("SELECT pg_try_advisory_xact_lock_shared(:k)"), {"k": INDEX_WRITE_LOCK}).scalar())


def require_lock_for_write(session) -> None:
    """For API requests that write to ES: fail fast with a conflict while a rebuild runs."""
    if not try_lock_for_write(session):
        raise ConflictError("索引正在重建，请稍后再试")


def _physical(alias: str) -> str:
    return next(iter(es.get_es().indices.get_alias(name=alias)))


def _next_name(alias: str, current: str) -> str:
    m = re.search(r"-v(\d+)$", current)
    return f"{alias}-v{int(m.group(1)) + 1 if m else 2}"


def _verify(index: str, sample: list[dict]) -> list[str]:
    """Each sampled chunk must come back first for a KNN search with its own vector and be in the
    BM25 top 10 for its own text."""
    problems, client = [], es.get_es()
    for doc in sample:
        src = doc["_source"]
        knn = client.search(index=index, size=1, source=["chunk_id"],
                            knn={"field": "vector", "query_vector": src["vector"], "k": 1, "num_candidates": 50})
        if knn["hits"]["hits"][0]["_source"]["chunk_id"] != src["chunk_id"]:
            problems.append(f"knn miss {src['chunk_id']}")
        bm25 = client.search(index=index, size=10, source=["chunk_id"], query={"match": {"text": src["text"][:200]}})
        if src["chunk_id"] not in {h["_source"]["chunk_id"] for h in bm25["hits"]["hits"]}:
            problems.append(f"bm25 miss {src['chunk_id']}")
    return problems


def rebuild(delete_old: bool = False) -> dict:
    alias, client = es.ensure_index(), es.get_es()
    old = _physical(alias)
    new = _next_name(alias, old)
    started = time.perf_counter()
    with engine.connect() as conn:
        # Session-level exclusive lock: waits for in-flight writers, then keeps new ones out.
        conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": INDEX_WRITE_LOCK})
        try:
            client.indices.create(index=new, settings=es.INDEX_SETTINGS,
                                  mappings=es.index_mapping(get_settings().embedding_dims))
            client.reindex(source={"index": old}, dest={"index": new}, wait_for_completion=True, refresh=True)
            old_count, new_count = client.count(index=old)["count"], client.count(index=new)["count"]
            if old_count != new_count:
                raise RuntimeError(f"文档数不一致: {old}={old_count}, {new}={new_count}")
            ids = [h["_id"] for h in client.search(index=old, size=10_000, source=False)["hits"]["hits"]]
            sample_ids = random.sample(ids, min(SAMPLE_SIZE, len(ids)))
            sample = client.mget(index=new, ids=sample_ids)["docs"] if sample_ids else []
            problems = _verify(new, sample)
            if problems:
                raise RuntimeError(f"新索引校验失败: {problems[:5]}")
            client.indices.update_aliases(actions=[
                {"remove": {"index": old, "alias": alias}},
                {"add": {"index": new, "alias": alias}},
            ])
        except Exception:
            client.indices.delete(index=new, ignore_unavailable=True)
            raise
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": INDEX_WRITE_LOCK})
    if delete_old:
        client.indices.delete(index=old)
    report = {"alias": alias, "old_index": old, "new_index": new, "documents": new_count,
              "verified_samples": len(sample), "old_deleted": delete_old,
              "seconds": round(time.perf_counter() - started, 2)}
    log.info("index_rebuilt", **report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-old", action="store_true")
    print(rebuild(parser.parse_args().delete_old))
