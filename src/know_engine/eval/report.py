"""Read side of evaluation runs: report, run list, comparison and CSV export.

JSON shapes are the ones eval-report.html renders.
"""

import csv
import io

from sqlalchemy import func, select

from know_engine.db import session_scope
from know_engine.ingest.errors import NotFoundError
from know_engine.models import EvalRun, EvalRunCase

METRICS = [
    ("hitAtK", "retrieval"), ("mrr", "retrieval"), ("recallAtK", "retrieval"),
    ("faithfulness", "generation"), ("relevancy", "generation"), ("answerCorrectness", "generation"),
    ("contextPrecision", "context"), ("contextRecall", "context"), ("avgLatencyMs", "retrieval"),
]
IMPROVEMENT = 0.1  # answer-correctness change that counts as improved / regressed


def _run(s, run_id: str) -> EvalRun:
    run = s.scalars(select(EvalRun).where(EvalRun.run_id == run_id)).first()
    if run is None:
        raise NotFoundError(f"评测记录不存在: {run_id}")
    return run


def _case_json(c: EvalRunCase) -> dict:
    retrieval = {k: v for k, v in (c.retrieval_scores or {}).items() if k in ("hitAtK", "mrr", "recallAtK")}
    return {
        "caseId": c.case_id, "question": c.question, "answer": c.answer, "transformContent": c.transform_content,
        "retrievedChunkIds": c.retrieved_chunk_ids, "retrievedDocumentIds": c.retrieved_document_ids,
        "ragReferences": c.rag_references, "retrievalScores": retrieval or None,
        "generationScores": c.generation_scores or None, "latencyMs": c.latency_ms,
        "timingsMs": (c.retrieval_scores or {}).get("timingsMs"), "success": c.success, "errorMessage": c.error_message,
    }


def get_report(run_id: str, include_cases: bool = True) -> dict:
    with session_scope() as s:
        run = _run(s, run_id)
        summary = run.summary_json or {}
        report = {
            "runId": run.run_id, "datasetPath": run.dataset_path, "config": run.config_json,
            "totalCases": run.total_cases, "evaluatedCases": run.evaluated_cases,
            "generationEvaluatedCases": run.generation_evaluated_cases, "status": run.status,
            "createdAt": run.created_at.isoformat(), **summary,
        }
        if include_cases:
            cases = s.scalars(select(EvalRunCase).where(EvalRunCase.run_id == run_id).order_by(EvalRunCase.id)).all()
            report["caseResults"] = [_case_json(c) for c in cases]
        return report


def list_runs(page: int, size: int) -> tuple[list[dict], int]:
    with session_scope() as s:
        total = s.scalar(select(func.count()).select_from(EvalRun))
        runs = s.scalars(select(EvalRun).order_by(EvalRun.created_at.desc()).offset((page - 1) * size).limit(size)).all()
        records = []
        for run in runs:
            summary = run.summary_json or {}
            retrieval, generation = summary.get("retrieval", {}), summary.get("generation", {})
            records.append({
                "runId": run.run_id, "createdAt": run.created_at.isoformat(), "datasetPath": run.dataset_path,
                "label": (run.config_json or {}).get("label"),
                "evaluatedCases": run.evaluated_cases, "totalCases": run.total_cases,
                "hitAtK": retrieval.get("hitAtK"), "mrr": retrieval.get("mrr"), "recallAtK": retrieval.get("recallAtK"),
                "faithfulness": generation.get("faithfulness"), "relevancy": generation.get("relevancy"),
                "answerCorrectness": generation.get("answerCorrectness"),
                "failedCount": len(summary.get("failedCaseIds", [])), "reviewCount": len(summary.get("reviewCaseIds", [])),
            })
        return records, total


def _correctness(s, run_id: str) -> dict[str, float]:
    rows = s.scalars(select(EvalRunCase).where(EvalRunCase.run_id == run_id, EvalRunCase.success.is_(True))).all()
    return {r.case_id: r.generation_scores["answerCorrectness"]
            for r in rows if (r.generation_scores or {}).get("answerCorrectness") is not None}


def _hits(s, run_id: str) -> dict[str, bool]:
    rows = s.scalars(select(EvalRunCase).where(EvalRunCase.run_id == run_id)).all()
    return {r.case_id: bool((r.retrieval_scores or {}).get("hitAtK")) for r in rows if "hitAtK" in (r.retrieval_scores or {})}


def compare(baseline_id: str, candidate_id: str) -> dict:
    with session_scope() as s:
        base, cand = _run(s, baseline_id).summary_json or {}, _run(s, candidate_id).summary_json or {}
        result: dict = {"baselineRunId": baseline_id, "candidateRunId": candidate_id}
        for metric, group in METRICS:
            b, c = base.get(group, {}).get(metric), cand.get(group, {}).get(metric)
            result[metric] = {"baseline": b, "candidate": c,
                              "delta": None if b is None or c is None else round(c - b, 4)}
        b_scores, c_scores = _correctness(s, baseline_id), _correctness(s, candidate_id)
        common = b_scores.keys() & c_scores.keys()
        result["improvedCaseIds"] = sorted(k for k in common if c_scores[k] - b_scores[k] >= IMPROVEMENT)
        result["regressedCaseIds"] = sorted(k for k in common if c_scores[k] - b_scores[k] <= -IMPROVEMENT)
        # Retrieval-level changes, useful when the judge is off.
        b_hits, c_hits = _hits(s, baseline_id), _hits(s, candidate_id)
        common = b_hits.keys() & c_hits.keys()
        result["newlyHitCaseIds"] = sorted(k for k in common if c_hits[k] and not b_hits[k])
        result["newlyMissedCaseIds"] = sorted(k for k in common if b_hits[k] and not c_hits[k])
        return result


def export_csv(run_id: str) -> str:
    report = get_report(run_id)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    columns = ["caseId", "question", "answer", "transformContent", "hitAtK", "mrr", "recallAtK", "faithfulness",
               "relevancy", "answerCorrectness", "contextPrecision", "contextRecall", "latencyMs", "success",
               "errorMessage", "retrievedChunkIds"]
    writer.writerow(columns)
    for c in report["caseResults"]:
        r, g = c["retrievalScores"] or {}, c["generationScores"] or {}
        writer.writerow([c["caseId"], c["question"], c["answer"], c["transformContent"], r.get("hitAtK"), r.get("mrr"),
                         r.get("recallAtK"), g.get("faithfulness"), g.get("relevancy"), g.get("answerCorrectness"),
                         g.get("contextPrecision"), g.get("contextRecall"), c["latencyMs"], c["success"],
                         c["errorMessage"], ";".join(c["retrievedChunkIds"] or [])])
    return "﻿" + buffer.getvalue()  # BOM so Excel opens the Chinese text correctly
