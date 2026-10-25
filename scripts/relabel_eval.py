"""Re-map the evaluation set's ground-truth chunk IDs onto the chunks currently in the database.

The original dataset labels chunks by the Java run's chunk IDs. The Java SQL dump contains
those chunks' text, so each old chunk is matched to the current chunks of the same document
by character 3-gram overlap: among the new chunks that overlap the old one, the chosen one
maximizes (answer coverage + 0.5 * similarity to the old chunk).

Rerun this whenever documents are re-split (chunk IDs change), then rerun the evaluation.

    python scripts/relabel_eval.py --sql ../know-engine/src/main/resources/sql/know_engine.sql
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select

from know_engine.db import session_scope
from know_engine.models import KnowledgeDocument, KnowledgeSegment

DATA = Path(__file__).resolve().parents[1] / "data" / "eval"
ORIGINAL = DATA / "original"  # datasets as shipped with the Java project, never modified
INSERT = re.compile(r"INSERT INTO `(\w+)` VALUES \((.*)\);$")
SAME_CHUNK = 0.7  # 3-gram Jaccard above which two chunks are considered the same
OVERLAPS = 0.3  # share of one chunk's 3-grams found in the other


def parse_values(raw: str) -> list:
    """Split one MySQL VALUES tuple into Python values (strings, numbers, None)."""
    values, i = [], 0
    while i < len(raw):
        if raw[i] in " ,":
            i += 1
        elif raw[i] == "'":
            out, i = [], i + 1
            while raw[i] != "'":
                if raw[i] == "\\":
                    out.append({"n": "\n", "t": "\t", "r": "\r", "0": "\0"}.get(raw[i + 1], raw[i + 1]))
                    i += 2
                else:
                    out.append(raw[i])
                    i += 1
            values.append("".join(out))
            i += 1
        else:
            j = raw.find(",", i)
            j = len(raw) if j == -1 else j
            token = raw[i:j].strip()
            values.append(None if token == "NULL" else token)
            i = j
    return values


def load_java(sql_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (old chunk_id -> (doc_id, text)) and (old doc_id -> title prefix like '01')."""
    chunks, docs = {}, {}
    for line in sql_path.read_text(encoding="utf-8").splitlines():
        m = INSERT.match(line)
        if not m:
            continue
        table, values = m.group(1), parse_values(m.group(2))
        if table == "knowledge_segment":
            # id, text, chunk_id, metadata, document_id, document_version, chunk_order, ...
            chunks[values[2]] = (values[4], values[1])
        elif table == "knowledge_document":
            docs[values[0]] = values[1][:2]
    return chunks, docs


def grams(text: str) -> set[str]:
    s = re.sub(r"\s+", "", text)
    return {s[i:i + 3] for i in range(max(1, len(s) - 2))}


def containment(a: set, b: set) -> float:
    """Share of a's 3-grams that appear in b."""
    return len(a & b) / len(a) if a else 0.0


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def current_chunks() -> tuple[dict[str, list[tuple[str, str]]], dict[str, str]]:
    """Searchable chunks of each document's current version, keyed by title prefix ('01')."""
    by_prefix, doc_ids = defaultdict(list), {}
    with session_scope() as s:
        for doc in s.scalars(select(KnowledgeDocument)).all():
            prefix = doc.doc_title[:2]
            doc_ids[prefix] = str(doc.doc_id)
            rows = s.scalars(select(KnowledgeSegment).where(
                KnowledgeSegment.document_version == doc.current_version_id,
                KnowledgeSegment.skip_embedding.is_(False),
            )).all()
            by_prefix[prefix] = [(r.chunk_id, r.text) for r in rows]
    return by_prefix, doc_ids


def relabel(case: dict, java_chunks, java_docs, new_chunks, new_doc_ids, stats) -> dict:
    answer = grams(case.get("groundTruthAnswer") or "")
    question = grams(case["question"])
    new_ids, new_docs = [], []
    for old_id in case.get("groundTruthChunkIds") or []:
        old_doc, old_text = java_chunks[old_id]
        prefix = java_docs[old_doc]
        old = grams(old_text)
        scored = []
        for chunk_id, text in new_chunks[prefix]:
            new = grams(text)
            scored.append({
                "id": chunk_id,
                "same": jaccard(old, new),
                "overlap": max(containment(new, old), containment(old, new)),
                "answer": containment(answer, new),
                "question": containment(question, new),
            })
        nearby = [c for c in scored if c["overlap"] >= OVERLAPS]
        if not nearby:
            stats["unmatched"] += 1
            continue
        # Answer coverage decides; similarity to the old chunk breaks ties. This handles both
        # moved boundaries (the answer sentence ends up in the neighbouring chunk) and table rows,
        # which share their header text and differ only in values.
        best = max(nearby, key=lambda c: (c["answer"] + 0.5 * c["same"], c["question"]))
        kind, chosen = ("same" if best["same"] >= SAME_CHUNK else "rechunked"), best["id"]
        stats[kind] += 1
        if chosen not in new_ids:
            new_ids.append(chosen)
        if new_doc_ids[prefix] not in new_docs:
            new_docs.append(new_doc_ids[prefix])
    return {**case, "groundTruthChunkIds": new_ids, "groundTruthDocumentIds": new_docs}


def is_holdout(case_id: str) -> bool:
    """Deterministic ~40% hold-out split, stable across runs."""
    return int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 10 < 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql", type=Path, required=True)
    args = parser.parse_args()

    java_chunks, java_docs = load_java(args.sql)
    new_chunks, new_doc_ids = current_chunks()
    stats = defaultdict(int)

    def relabel_file(name: str) -> list[dict]:
        lines = (ORIGINAL / name).read_text(encoding="utf-8").splitlines()
        return [relabel(json.loads(line), java_chunks, java_docs, new_chunks, new_doc_ids, stats)
                for line in lines if line.strip()]

    relabeled = relabel_file("eval-test-all.jsonl")
    outputs = {
        "eval-all.jsonl": relabeled,
        "eval-tune.jsonl": [c for c in relabeled if not is_holdout(c["id"])],
        "eval-holdout.jsonl": [c for c in relabeled if is_holdout(c["id"])],
    }
    main_stats = dict(stats)  # report numbers for the main set only
    # Small legacy sets, so the report page's default path keeps working.
    outputs |= {name: relabel_file(name) for name in ("eval-test.jsonl", "eval-test1.jsonl")}
    for name, rows in outputs.items():
        (DATA / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} cases")
    print("eval-all old ground-truth chunks:", main_stats)
    print("eval-all cases without any ground-truth chunk:", sum(1 for r in relabeled if not r["groundTruthChunkIds"]))


if __name__ == "__main__":
    main()
