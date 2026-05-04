"""Module 3: Reranking - Cross-encoder top-20 -> top-3 + latency benchmark."""

import os, sys, time, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._model_type = "heuristic"

    def _load_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker  # type: ignore

                self._model = FlagReranker(self.model_name, use_fp16=False)
                self._model_type = "flag"
            except Exception:
                try:
                    from sentence_transformers import CrossEncoder  # type: ignore

                    self._model = CrossEncoder(self.model_name)
                    self._model_type = "cross_encoder"
                except Exception:
                    self._model = None
                    self._model_type = "heuristic"
        return self._model

    @staticmethod
    def _heuristic_score(query: str, text: str, original_score: float = 0.0) -> float:
        q_tokens = set(re.findall(r"[0-9A-Za-zÀ-ỹ]+", query.lower(), flags=re.UNICODE))
        d_tokens = set(re.findall(r"[0-9A-Za-zÀ-ỹ]+", text.lower(), flags=re.UNICODE))
        if not q_tokens or not d_tokens:
            return float(original_score)
        overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        phrase_bonus = 0.25 if query.lower() in text.lower() else 0.0
        numeric_bonus = 0.1 if any(ch.isdigit() for ch in query) and any(ch.isdigit() for ch in text) else 0.0
        return 0.75 * overlap + phrase_bonus + numeric_bonus + 0.05 * float(original_score)

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 -> top-k."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]

        scores: list[float]
        if model is not None and self._model_type == "flag":
            try:
                raw_scores = model.compute_score(pairs)
                if isinstance(raw_scores, (float, int)):
                    scores = [float(raw_scores)]
                else:
                    scores = [float(s) for s in raw_scores]
            except Exception:
                scores = [
                    self._heuristic_score(query, doc.get("text", ""), float(doc.get("score", 0.0)))
                    for doc in documents
                ]
        elif model is not None and self._model_type == "cross_encoder":
            try:
                raw_scores = model.predict(pairs)
                scores = [float(s) for s in raw_scores]
            except Exception:
                scores = [
                    self._heuristic_score(query, doc.get("text", ""), float(doc.get("score", 0.0)))
                    for doc in documents
                ]
        else:
            scores = [
                self._heuristic_score(query, doc.get("text", ""), float(doc.get("score", 0.0)))
                for doc in documents
            ]

        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results: list[RerankResult] = []
        for idx, (score, doc) in enumerate(scored_docs[:top_k], start=1):
            results.append(
                RerankResult(
                    text=doc.get("text", ""),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=doc.get("metadata", {}),
                    rank=idx,
                )
            )
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest  # type: ignore

            if self._model is None:
                self._model = Ranker()
            passages = [{"text": d.get("text", "")} for d in documents]
            reranked = self._model.rerank(RerankRequest(query=query, passages=passages))[:top_k]
            output = []
            for idx, item in enumerate(reranked, start=1):
                text = item.get("text", "")
                output.append(
                    RerankResult(
                        text=text,
                        original_score=0.0,
                        rerank_score=float(item.get("score", 0.0)),
                        metadata={},
                        rank=idx,
                    )
                )
            return output
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(max(1, n_runs)):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
