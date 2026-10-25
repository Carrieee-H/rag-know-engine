"""Batch evaluation: run every case through the real RAG chain, score it, store the results.

Inference skips intent recognition (every case is forced through retrieval) so the scores
reflect the retrieval chain, not the router.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import structlog

from know_engine.constants import EvalRunStatus
from know_engine.db import session_scope
from know_engine.eval import dataset
from know_engine.eval.dataset import EvalCase
from know_engine.eval.metrics import judge, retrieval_scores
from know_engine.models import EvalRun, EvalRunCase
from know_engine.rag import pipeline
from know_engine.rag.pipeline import RagConfig

log = structlog.get_logger()

GENERATION_METRICS = ["faithfulness", "relevancy", "answerCorrectness"]
CONTEXT_METRICS = ["contextPrecision", "contextRecall"]


@dataclass
class EvalConfig:
    top_k: int = 5
    concurrency: int = 1
    enable_llm_judge: bool = True
    review_score_threshold: float = 0.6
    review_mrr_threshold: float = 0.3
    review_recall_threshold: float = 0.3
    label: str = ""
    rag: RagConfig = field(default_factory=RagConfig)

    def to_json(self) -> dict:
        return {
            "topK": self.top_k, "concurrency": self.concurrency, "enableLlmJudge": self.enable_llm_judge,
            "reviewScoreThreshold": self.review_score_threshold, "reviewMrrThreshold": self.review_mrr_threshold,
            "reviewRecallAtKThreshold": self.review_recall_threshold, "label": self.label, "rag": asdict(self.rag),
        }

    @classmethod
    def from_json(cls, data: dict | None) -> "EvalConfig":
        data = data or {}

        def pick(key, default):
            return default if data.get(key) is None else data[key]

        return cls(
            top_k=int(pick("topK", 5)), concurrency=max(1, int(pick("concurrency", 1))),
            enable_llm_judge=bool(pick("enableLlmJudge", True)),
            review_score_threshold=float(pick("reviewScoreThreshold", 0.6)),
            review_mrr_threshold=float(pick("reviewMrrThreshold", 0.3)),
            review_recall_threshold=float(pick("reviewRecallAtKThreshold", 0.3)),
            label=str(pick("label", "")), rag=RagConfig.from_dict(data.get("rag")),
        )


def _infer(case: EvalCase, config: EvalConfig) -> dict:
    start = time.perf_counter()
    try:
        retrieval = pipeline.retrieve(case.question, config=config.rag)
        t = time.perf_counter()
        answer = pipeline.answer(case.question, retrieval.references)
        timings = {**retrieval.timings_ms, "answer": round((time.perf_counter() - t) * 1000)}
        references = pipeline.reference_dicts(retrieval)
        result = {
            "success": True, "answer": answer, "rewritten": retrieval.rewritten, "references": references,
            "chunk_ids": [r["chunkId"] for r in references], "doc_ids": [r["documentId"] for r in references],
            "timings": timings, "used_fallback": retrieval.used_fallback, "error": None,
        }
    except Exception as e:
        log.error("eval_case_failed", case=case.id, error=str(e))
        result = {"success": False, "answer": None, "rewritten": None, "references": [], "chunk_ids": [],
                  "doc_ids": [], "timings": {}, "used_fallback": False, "error": str(e)[:2000]}
    result["latency_ms"] = round((time.perf_counter() - start) * 1000)
    return result


def _evaluate(case: EvalCase, config: EvalConfig) -> dict:
    result = _infer(case, config)
    covered = [set(r.get("coveredChunkIds") or [r["chunkId"]]) for r in result["references"]]
    result["retrieval"] = retrieval_scores(covered, result["doc_ids"], case.gt_chunk_ids,
                                           case.gt_document_ids, config.top_k) if result["success"] else None
    result["generation"] = {}
    if result["success"] and config.enable_llm_judge:
        contexts = [r["chunkContent"] for r in result["references"]]
        result["generation"] = judge(case.question, result["answer"], contexts, case.ground_truth_answer)
    return result


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def _needs_review(result: dict, config: EvalConfig) -> bool:
    if not result["success"]:
        return True
    r = result["retrieval"]
    if r and (not r["hitAtK"] or r["mrr"] < config.review_mrr_threshold or r["recallAtK"] < config.review_recall_threshold):
        return True
    g = result["generation"]
    return any(g.get(m) is not None and g[m] < config.review_score_threshold
               for m in GENERATION_METRICS + CONTEXT_METRICS)


def summarize(cases: list[EvalCase], results: list[dict], config: EvalConfig) -> dict:
    ok = [r for r in results if r["success"]]
    scored = [r["retrieval"] for r in ok if r["retrieval"]]
    latencies = [r["latency_ms"] for r in ok]
    stages = sorted({k for r in ok for k in r["timings"]})
    return {
        "retrieval": {
            "hitAtK": _mean([float(s["hitAtK"]) for s in scored]),
            "mrr": _mean([s["mrr"] for s in scored]),
            "recallAtK": _mean([s["recallAtK"] for s in scored]),
            "avgLatencyMs": round(float(np.mean(latencies))) if latencies else 0,
            "p95LatencyMs": round(float(np.percentile(latencies, 95))) if latencies else 0,
        },
        "generation": {m: _mean([r["generation"][m] for r in ok if r["generation"].get(m) is not None])
                       for m in GENERATION_METRICS},
        "context": {m: _mean([r["generation"][m] for r in ok if r["generation"].get(m) is not None])
                    for m in CONTEXT_METRICS},
        "stageLatencyMs": {s: {"p50": round(float(np.percentile(v, 50))), "p95": round(float(np.percentile(v, 95)))}
                           for s in stages if (v := [r["timings"][s] for r in ok if s in r["timings"]])},
        "evaluatedCases": len(scored),
        "generationEvaluatedCases": sum(1 for r in ok if any(m in r["generation"] for m in GENERATION_METRICS)),
        "failedCaseIds": [c.id for c, r in zip(cases, results) if not r["success"]],
        "reviewCaseIds": [c.id for c, r in zip(cases, results) if _needs_review(r, config)],
        "missCaseIds": [c.id for c, r in zip(cases, results) if r["retrieval"] and not r["retrieval"]["hitAtK"]],
        "fallbackCount": sum(1 for r in ok if r["used_fallback"]),
    }


def run_batch(dataset_path: str, config: EvalConfig) -> str:
    cases = dataset.load(dataset_path)
    run_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log.info("eval_started", run_id=run_id, cases=len(cases), label=config.label)
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        results = list(pool.map(lambda c: _evaluate(c, config), cases))
    summary = summarize(cases, results, config)
    with session_scope() as s:
        s.add(EvalRun(
            run_id=run_id, dataset_path=dataset_path, config_json=config.to_json(), summary_json=summary,
            total_cases=len(cases), evaluated_cases=summary["evaluatedCases"],
            generation_evaluated_cases=summary["generationEvaluatedCases"], status=EvalRunStatus.COMPLETED,
        ))
        s.flush()
        for case, r in zip(cases, results):
            s.add(EvalRunCase(
                run_id=run_id, case_id=case.id, question=case.question, answer=r["answer"],
                transform_content=r["rewritten"], retrieved_chunk_ids=r["chunk_ids"],
                retrieved_document_ids=r["doc_ids"], rag_references=r["references"],
                retrieval_scores={**(r["retrieval"] or {}), "timingsMs": r["timings"], "usedFallback": r["used_fallback"]},
                generation_scores=r["generation"], latency_ms=r["latency_ms"], success=r["success"],
                error_message=r["error"],
            ))
    log.info("eval_finished", run_id=run_id, hit=summary["retrieval"]["hitAtK"], mrr=summary["retrieval"]["mrr"])
    return run_id
