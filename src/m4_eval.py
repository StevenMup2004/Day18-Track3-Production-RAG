"""Module 4: RAGAS Evaluation - 4 metrics + failure analysis."""

import os, sys, json, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    n = min(len(questions), len(answers), len(contexts), len(ground_truths))
    if n == 0:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }

    questions = questions[:n]
    answers = answers[:n]
    contexts = contexts[:n]
    ground_truths = ground_truths[:n]

    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # type: ignore
        from datasets import Dataset  # type: ignore

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        per_question = []
        for _, row in df.iterrows():
            per_question.append(
                EvalResult(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    contexts=list(row.get("contexts", [])),
                    ground_truth=str(row.get("ground_truth", "")),
                    faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                    context_precision=float(row.get("context_precision", 0.0) or 0.0),
                    context_recall=float(row.get("context_recall", 0.0) or 0.0),
                )
            )

        def mean_metric(name: str) -> float:
            vals = [getattr(x, name) for x in per_question]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "faithfulness": mean_metric("faithfulness"),
            "answer_relevancy": mean_metric("answer_relevancy"),
            "context_precision": mean_metric("context_precision"),
            "context_recall": mean_metric("context_recall"),
            "per_question": per_question,
        }
    except Exception:
        pass

    # Fallback heuristic evaluation when ragas/datasets are unavailable.
    def toks(text: str) -> set[str]:
        return set(re.findall(r"[0-9A-Za-zÀ-ỹ]+", text.lower(), flags=re.UNICODE))

    def overlap_ratio(a: set[str], b: set[str]) -> float:
        if not a:
            return 0.0
        return len(a & b) / len(a)

    per_question: list[EvalResult] = []
    for q, a, c_list, gt in zip(questions, answers, contexts, ground_truths):
        q_tokens = toks(q)
        a_tokens = toks(a)
        gt_tokens = toks(gt)
        ctx_tokens = [toks(c) for c in c_list]
        merged_ctx = set().union(*ctx_tokens) if ctx_tokens else set()

        faith = overlap_ratio(a_tokens, merged_ctx)
        ans_rel = 0.5 * overlap_ratio(a_tokens, q_tokens) + 0.5 * overlap_ratio(a_tokens, gt_tokens)

        if ctx_tokens:
            precision_scores = []
            for c in ctx_tokens:
                if not c:
                    precision_scores.append(0.0)
                else:
                    precision_scores.append(len(c & gt_tokens) / len(c))
            ctx_prec = sum(precision_scores) / len(precision_scores)
        else:
            ctx_prec = 0.0

        ctx_rec = overlap_ratio(gt_tokens, merged_ctx) if gt_tokens else (1.0 if c_list else 0.0)

        per_question.append(
            EvalResult(
                question=q,
                answer=a,
                contexts=c_list,
                ground_truth=gt,
                faithfulness=max(0.0, min(1.0, faith)),
                answer_relevancy=max(0.0, min(1.0, ans_rel)),
                context_precision=max(0.0, min(1.0, ctx_prec)),
                context_recall=max(0.0, min(1.0, ctx_rec)),
            )
        )

    def avg(name: str) -> float:
        values = [getattr(r, name) for r in per_question]
        return sum(values) / len(values) if values else 0.0

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    normalized: list[EvalResult] = []
    for item in eval_results:
        if isinstance(item, EvalResult):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(
                EvalResult(
                    question=str(item.get("question", "")),
                    answer=str(item.get("answer", "")),
                    contexts=list(item.get("contexts", [])),
                    ground_truth=str(item.get("ground_truth", "")),
                    faithfulness=float(item.get("faithfulness", 0.0)),
                    answer_relevancy=float(item.get("answer_relevancy", 0.0)),
                    context_precision=float(item.get("context_precision", 0.0)),
                    context_recall=float(item.get("context_recall", 0.0)),
                )
            )

    scored = []
    for r in normalized:
        avg_score = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4.0
        scored.append((avg_score, r))
    scored.sort(key=lambda x: x[0])
    worst = scored[:bottom_n]

    failures = []
    for avg_score, r in worst:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]

        if worst_metric == "faithfulness" and r.faithfulness < 0.85:
            diagnosis = "LLM hallucinating"
            fix = "Tighten prompt grounding and reduce generation randomness."
        elif worst_metric == "context_recall" and r.context_recall < 0.75:
            diagnosis = "Missing relevant chunks"
            fix = "Improve chunking strategy and retrieval recall (hybrid search/query expansion)."
        elif worst_metric == "context_precision" and r.context_precision < 0.75:
            diagnosis = "Irrelevant chunks retrieved"
            fix = "Add stronger reranking and metadata filters."
        elif worst_metric == "answer_relevancy" and r.answer_relevancy < 0.80:
            diagnosis = "Answer mismatch with the question"
            fix = "Improve answer prompt template and enforce question-focused response."
        else:
            diagnosis = "Mixed quality degradation"
            fix = "Inspect retrieval and generation logs for this query and tune weakest stage."

        failures.append(
            {
                "question": r.question,
                "worst_metric": worst_metric,
                "score": float(worst_score),
                "avg_score": float(avg_score),
                "diagnosis": diagnosis,
                "suggested_fix": fix,
                "expected": r.ground_truth,
                "got": r.answer,
            }
        )
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
