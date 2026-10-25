"""Retrieval metrics and LLM-as-judge scoring."""

import json
import re
from dataclasses import dataclass

from know_engine.config import get_settings
from know_engine.prompts import load_prompt
from know_engine.rag.llm import complete, render

NO_CONTEXT = "(无检索上下文)"
SCORE_JSON = re.compile(r"\{[^{}]*\"score\"[^{}]*\}", re.DOTALL)


def retrieval_scores(retrieved_chunks: list[set[str]], retrieved_docs: list[str],
                     gt_chunks: list[str], gt_docs: list[str], top_k: int) -> dict | None:
    """Hit@K, MRR and Recall@K over the top-K references, at chunk level, or document level when
    the case has no chunk ground truth.

    Each reference is the set of chunk IDs whose text it carries (a parent reference covers every
    child folded into it). Returns None when the case has no ground truth.
    """
    if gt_chunks:
        retrieved, truth = retrieved_chunks[:top_k], set(gt_chunks)
    elif gt_docs:
        retrieved, truth = [{d} for d in retrieved_docs[:top_k]], set(gt_docs)
    else:
        return None
    first_hit = next((i for i, ids in enumerate(retrieved) if ids & truth), None)
    found = set().union(*retrieved) & truth if retrieved else set()
    return {
        "hitAtK": first_hit is not None,
        "mrr": 0.0 if first_hit is None else 1.0 / (first_hit + 1),
        "recallAtK": len(found) / len(truth),
    }


@dataclass
class JudgeScore:
    score: float
    reason: str


def parse_judge(raw: str) -> JudgeScore:
    """Accept bare JSON, JSON inside other text, or a fenced block; clamp score to [0, 1]."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Judge 响应为空")
    candidates = [text]
    if m := SCORE_JSON.search(text):
        candidates.append(m.group())
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{"): text.rindex("}") + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "score" in data:
            return JudgeScore(max(0.0, min(1.0, float(data["score"]))), str(data.get("reason", "")))
    raise ValueError(f"无法解析 Judge 响应: {text[:120]}")


def format_contexts(contexts: list[str]) -> str:
    parts = [c for c in contexts if c and c.strip()]
    if not parts:
        return NO_CONTEXT
    return "\n\n".join(f"[片段{i}]\n{c}" for i, c in enumerate(parts, start=1))


def _judge(prompt_name: str, **values: str) -> JudgeScore:
    prompt = render(load_prompt(prompt_name), **values)
    return parse_judge(complete(get_settings().chat_model, [{"role": "user", "content": prompt}]))


def judge(question: str, answer: str, contexts: list[str], ground_truth: str | None) -> dict:
    """Up to five scores; each one is independent, so one failure does not lose the others."""
    scores: dict = {}
    ctx = format_contexts(contexts)
    tasks = []
    if ctx != NO_CONTEXT:
        tasks.append(("contextPrecision", "eval-context-precision-prompt.txt", {"question": question, "contexts": ctx}))
        if ground_truth:
            tasks.append(("contextRecall", "eval-context-recall-prompt.txt",
                          {"question": question, "ground_truth_answer": ground_truth, "contexts": ctx}))
    if answer and answer.strip():
        tasks.append(("faithfulness", "eval-faithfulness-prompt.txt",
                      {"question": question, "contexts": ctx, "answer": answer}))
        tasks.append(("relevancy", "eval-relevancy-prompt.txt", {"question": question, "answer": answer}))
        if ground_truth:
            tasks.append(("answerCorrectness", "eval-correctness-prompt.txt",
                          {"question": question, "ground_truth_answer": ground_truth, "answer": answer}))
    for name, prompt, values in tasks:
        try:
            result = _judge(prompt, **values)
            scores[name], scores[f"{name}Reason"] = result.score, result.reason
        except Exception as e:
            scores[f"{name}Error"] = str(e)[:300]
    return scores
