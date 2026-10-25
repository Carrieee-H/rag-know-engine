"""Domain dictionary for the IK analyzer, built from the indexed corpus.

IK has no entry for in-house names, so 智造园区 becomes 智 / 造园 / 区 and 蓝鹭应急协调组 is
cut into single characters, which hurts BM25. This collects such terms from the data:

- values of name-like table columns (asset names, team names, ...),
- the seed list in data/ik/seed.dic (hand-reviewed words that appear in the documents).

The result is published to MinIO as ``ik/domain.dic`` (the only anonymously readable prefix),
which Elasticsearch fetches as an IK ``remote_ext_dict`` and reloads automatically (IK polls
it about once a minute). Text already indexed keeps its old tokens until it is reindexed
(see reindex.py).

    python -m know_engine.ingest.dictionary
"""

import json
import re

from sqlalchemy import select

from know_engine.clients import storage
from know_engine.config import PROJECT_ROOT, get_settings
from know_engine.db import session_scope
from know_engine.models import KnowledgeDocument, KnowledgeSegment

SEED_FILE = PROJECT_ROOT / "data" / "ik" / "seed.dic"
OBJECT_NAME = "ik/domain.dic"
NAME_COLUMNS = ("资产名称", "责任团队", "生产区")  # entity names, not free-text columns
CHINESE_TERM = re.compile(r"^[一-鿿]{2,12}$")


def _table_terms(text: str) -> set[str]:
    """Values of name-like columns in a table row chunk (``列名：值; 列名：值``)."""
    terms = set()
    for pair in text.split("; "):
        column, _, value = pair.partition("：")
        if column in NAME_COLUMNS and CHINESE_TERM.match(value.strip()):
            terms.add(value.strip())
    return terms


def collect_terms() -> list[str]:
    terms: set[str] = set()
    if SEED_FILE.exists():
        terms |= {w.strip() for w in SEED_FILE.read_text(encoding="utf-8").splitlines()
                  if w.strip() and not w.startswith("#")}
    with session_scope() as s:
        current = select(KnowledgeDocument.current_version_id)
        for text in s.scalars(select(KnowledgeSegment.text).where(KnowledgeSegment.document_version.in_(current))):
            if "：" in text and "; " in text:
                terms |= _table_terms(text)
    return sorted(terms)


def _allow_anonymous_read(prefix: str) -> None:
    bucket = get_settings().minio_bucket
    policy = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"AWS": ["*"]}, "Action": ["s3:GetObject"],
        "Resource": [f"arn:aws:s3:::{bucket}/{prefix}*"],
    }]}
    storage.get_minio().set_bucket_policy(bucket, json.dumps(policy))


def publish_dictionary() -> list[str]:
    terms = collect_terms()
    _allow_anonymous_read("ik/")
    storage.put_bytes(OBJECT_NAME, ("\n".join(terms) + "\n").encode(), "text/plain; charset=utf-8")
    return terms


if __name__ == "__main__":
    words = publish_dictionary()
    print(f"{len(words)} terms -> minio {OBJECT_NAME}")
    print(" ".join(words))
