"""Run an evaluation from the command line, with RagConfig overrides for tuning experiments.

    python -m know_engine.eval.cli eval-tune.jsonl --label baseline --no-judge
    python -m know_engine.eval.cli eval-tune.jsonl --label "rerank 0.1" --set rerank_min_score=0.1
    python -m know_engine.eval.cli --compare RUN_A RUN_B
"""

import argparse
import json

from know_engine.eval import report
from know_engine.eval.runner import EvalConfig, run_batch
from know_engine.rag.pipeline import RagConfig


def _parse_value(raw: str):
    if raw.lower() in ("none", "null"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _print_summary(run_id: str) -> None:
    r = report.get_report(run_id, include_cases=False)
    ret, gen, ctx = r["retrieval"], r["generation"], r["context"]
    print(f"run {run_id}  [{r['config'].get('label', '')}]  {r['evaluatedCases']}/{r['totalCases']} cases")
    print(f"  Hit@K={ret['hitAtK']}  MRR={ret['mrr']}  Recall@K={ret['recallAtK']}  "
          f"avg={ret['avgLatencyMs']}ms p95={ret['p95LatencyMs']}ms  fallback={r['fallbackCount']}")
    if any(v is not None for v in gen.values()):
        print(f"  faithfulness={gen['faithfulness']} relevancy={gen['relevancy']} correctness={gen['answerCorrectness']}"
              f"  ctxPrecision={ctx['contextPrecision']} ctxRecall={ctx['contextRecall']}")
    print(f"  stages: {r['stageLatencyMs']}")
    print(f"  missed: {r['missCaseIds']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--label", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)  # higher trips the model API's rate limit
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="RagConfig override")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args()

    if args.compare:
        print(json.dumps(report.compare(*args.compare), ensure_ascii=False, indent=2))
        return
    overrides = dict(item.split("=", 1) for item in args.set)
    unknown = set(overrides) - set(RagConfig.__dataclass_fields__)
    if unknown:
        parser.error(f"unknown RagConfig fields: {unknown}")
    rag = RagConfig.from_dict({k: _parse_value(v) for k, v in overrides.items()})
    config = EvalConfig(top_k=args.top_k, concurrency=args.concurrency, enable_llm_judge=not args.no_judge,
                        label=args.label, rag=rag)
    _print_summary(run_batch(args.dataset, config))


if __name__ == "__main__":
    main()
