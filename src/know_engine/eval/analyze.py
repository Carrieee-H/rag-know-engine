"""BadCase analysis: replay a case through each retrieval stage and show where the
ground-truth chunk is lost.

    python -m know_engine.eval.analyze eval-tune.jsonl aurora-md-010 aurora-txt-015 --set rerank_min_score=0.6
"""

import argparse

import numpy as np

from know_engine.clients.llm import embed_texts
from know_engine.eval import dataset
from know_engine.eval.cli import _parse_value
from know_engine.rag.pipeline import RagConfig, rank_candidates
from know_engine.rag.query import rewrite_query
from know_engine.rag.retrieve import keyword_search, replace_with_parents, rerank, rrf_fuse, vector_search


def _rank(candidates, truth: set[str]) -> str:
    ranks = [str(i) for i, c in enumerate(candidates, start=1) if truth & set(c.covered_chunk_ids)]
    return ",".join(ranks) if ranks else "-"


def diagnose(case: dataset.EvalCase, config: RagConfig) -> str:
    truth = set(case.gt_chunk_ids)
    query = rewrite_query(case.question, [], config.rewrite_prompt) if config.use_rewrite else case.question
    vector = vector_search(embed_texts([query])[0], config.vector_top_k, config.vector_min_score)
    keyword = keyword_search(query, config.keyword_top_k)
    fused, kept, fallback = rank_candidates(query, vector, keyword, config)

    if not any(truth & set(c.covered_chunk_ids) for c in fused):
        verdict = "NOT RETRIEVED: neither path returned the ground-truth chunk"
    elif not any(truth & set(c.covered_chunk_ids) for c in kept):
        gt = [c for c in fused if truth & set(c.covered_chunk_ids)]
        verdict = (f"DROPPED AT RERANK: gt rerank score {max(c.rerank_score for c in gt):.3f} "
                   f"(threshold {config.rerank_min_score}, top_n {config.rerank_top_n})")
    else:
        verdict = f"OK at rank {_rank(kept, truth)}" + (" (fallback)" if fallback else "")

    lines = [
        f"== {case.id}: {case.question}",
        f"   answer:    {case.ground_truth_answer}",
        f"   rewritten: {query}",
        (f"   gt rank -> vector {_rank(vector, truth)} | keyword {_rank(keyword, truth)} | "
         f"rrf {_rank(fused, truth)} | reranked {_rank(sorted(fused, key=lambda c: -c.rerank_score), truth)}"),
        f"   verdict:   {verdict}",
    ]
    for c in sorted(fused, key=lambda c: -c.rerank_score)[:6]:
        mark = "*" if truth & set(c.covered_chunk_ids) else " "
        lines.append(f"   {mark} rerank={c.rerank_score:.3f} {c.source:7} {c.file_name[:12]} | {c.text[:60]!r}")
    return "\n".join(lines)


def score_distribution(cases: list[dataset.EvalCase], config: RagConfig) -> str:
    """Rerank and vector scores of ground-truth vs other candidates, to pick thresholds from data."""
    rerank_gt, rerank_other, vector_gt, vector_other = [], [], [], []
    for case in cases:
        truth = set(case.gt_chunk_ids)
        query = rewrite_query(case.question, [], config.rewrite_prompt) if config.use_rewrite else case.question
        vector = vector_search(embed_texts([query])[0], config.vector_top_k, None)
        for c in vector:
            (vector_gt if truth & set(c.covered_chunk_ids) else vector_other).append(c.score)
        fused = replace_with_parents(rrf_fuse([vector, keyword_search(query, config.keyword_top_k)], k=config.rrf_k))
        rerank(query, fused, None, len(fused), False)
        for c in fused:
            (rerank_gt if truth & set(c.covered_chunk_ids) else rerank_other).append(c.rerank_score)

    def row(name: str, values: list[float]) -> str:
        if not values:
            return f"{name:22} n=0"
        p = np.percentile(values, [5, 10, 25, 50, 75, 90])
        return f"{name:22} n={len(values):3}  p5={p[0]:.3f} p10={p[1]:.3f} p25={p[2]:.3f} p50={p[3]:.3f} p75={p[4]:.3f} p90={p[5]:.3f}"

    return "\n".join([
        row("rerank ground truth", rerank_gt), row("rerank others", rerank_other),
        row("vector ground truth", vector_gt), row("vector others", vector_other),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("case_ids", nargs="*", help="default: every case")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--distribution", action="store_true", help="print score distributions instead")
    args = parser.parse_args()
    config = RagConfig.from_dict({k: _parse_value(v) for k, v in (s.split("=", 1) for s in args.set)})
    cases = [c for c in dataset.load(args.dataset) if not args.case_ids or c.id in args.case_ids]
    if args.distribution:
        print(score_distribution(cases, config))
        return
    for case in cases:
        print(diagnose(case, config))


if __name__ == "__main__":
    main()
