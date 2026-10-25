"""Evaluation datasets: JSONL files under data/eval.

Paths from the Java page (``classpath:eval/datasets/x.jsonl``) map to ``data/eval/x.jsonl``.
"""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from know_engine.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "eval"
LEGACY_PREFIX = "classpath:eval/datasets/"


@dataclass
class EvalCase:
    id: str
    question: str
    ground_truth_answer: str | None
    gt_chunk_ids: list[str]
    gt_document_ids: list[str]


def resolve(dataset_path: str) -> Path:
    name = dataset_path.removeprefix(LEGACY_PREFIX)
    path = (DATA_DIR / name).resolve()
    if DATA_DIR.resolve() not in path.parents:
        raise ValueError(f"评测集路径不合法: {dataset_path}")
    if not path.is_file():
        raise ValueError(f"评测集不存在: {dataset_path}")
    return path


def load(dataset_path: str) -> list[EvalCase]:
    cases = []
    for n, line in enumerate(resolve(dataset_path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("question"):
            raise ValueError(f"第 {n} 行缺少 question")
        cases.append(EvalCase(
            id=str(row.get("id") or f"case-{n}"),
            question=row["question"],
            ground_truth_answer=row.get("groundTruthAnswer"),
            gt_chunk_ids=[str(x) for x in row.get("groundTruthChunkIds") or []],
            gt_document_ids=[str(x) for x in row.get("groundTruthDocumentIds") or []],
        ))
    return cases


def save_upload(file_name: str, data: bytes) -> str:
    """Store an uploaded dataset and return the path to pass to /eval/run."""
    if not file_name.endswith(".jsonl"):
        raise ValueError("评测集必须是 .jsonl 文件")
    relative = f"uploads/{uuid.uuid4().hex[:8]}_{Path(file_name).name}"
    target = DATA_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    load(relative)  # validate
    return relative
