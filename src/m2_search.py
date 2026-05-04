"""Module 2: Hybrid Search - BM25 (Vietnamese) + Dense + RRF (real-only)."""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_MODE,
    QDRANT_LOCAL_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    OPENAI_API_KEY,
    BM25_TOP_K,
    DENSE_TOP_K,
    HYBRID_TOP_K,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words using underthesea."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "underthesea is required for real Vietnamese tokenization. Install: pip install underthesea"
        ) from e
    return word_tokenize(text, format="text")


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks using rank_bm25."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            tokenized = segment_vietnamese(chunk.get("text", "")).split()
            self.corpus_tokens.append(tokenized)

        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception as e:
            raise RuntimeError("rank_bm25 is required for real BM25 indexing.") from e
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.documents or self.bm25 is None:
            raise RuntimeError("BM25 index has not been built. Call index() first.")

        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []

        raw_scores = self.bm25.get_scores(tokenized_query)
        scores = [float(s) for s in raw_scores]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[idx].get("text", ""),
                score=float(scores[idx]),
                metadata=self.documents[idx].get("metadata", {}),
                method="bm25",
            )
            for idx in top_indices
        ]


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except Exception as e:
            raise RuntimeError("qdrant-client is required for real dense search.") from e

        mode = (QDRANT_MODE or "local").lower().strip()
        if mode == "server":
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            try:
                self.client.get_collections()
            except Exception as e:
                raise RuntimeError(
                    f"Cannot connect to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}. "
                    "Start Qdrant service or use QDRANT_MODE=local."
                ) from e
        else:
            # Real Qdrant local engine (embedded), not mock.
            self.client = QdrantClient(path=QDRANT_LOCAL_PATH)

        self._openai_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            if not OPENAI_API_KEY:
                raise RuntimeError(
                    "OPENAI_API_KEY is missing. It is required for text-embedding-3-large."
                )
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError("openai package is required for OpenAI embeddings.") from e
            self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return self._openai_client

    @staticmethod
    def _prepare_text(text: str) -> str:
        clean = (text or "").strip()
        if not clean:
            clean = "[EMPTY]"
        # Keep within a safe size for embedding requests.
        return clean[:6000]

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        client = self._get_openai_client()
        prepared = [self._prepare_text(t) for t in texts]
        vectors: list[list[float]] = []

        batch_size = 64
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i:i + batch_size]
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            batch_vectors = [item.embedding for item in resp.data]
            vectors.extend(batch_vectors)
        return vectors

    def _encode_query(self, query: str) -> list[float]:
        vectors = self._encode_texts([query])
        return vectors[0]

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            raise ValueError("No chunks to index.")

        from qdrant_client.models import Distance, VectorParams, PointStruct  # type: ignore

        texts = [c.get("text", "") for c in chunks]
        vectors = self._encode_texts(texts)
        vec_size = len(vectors[0]) if len(vectors) > 0 else EMBEDDING_DIM
        if vec_size != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch: got {vec_size}, expected {EMBEDDING_DIM}."
            )

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
        )

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {**chunk.get("metadata", {}), "text": chunk.get("text", "")}
            vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            points.append(PointStruct(id=i, vector=vector_list, payload=payload))
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors from Qdrant."""
        query_vector = self._encode_query(query)
        query_vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        # qdrant-client >= 1.9 uses query_points(), older versions expose search().
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
            hits = list(getattr(resp, "points", []) or [])
        elif hasattr(self.client, "search"):
            try:
                hits = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
            except TypeError:
                # Older positional signature
                hits = self.client.search(collection, query_vector, limit=top_k)
        else:
            raise RuntimeError("Unsupported qdrant-client version: neither query_points() nor search() is available.")

        results: list[SearchResult] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(
                SearchResult(
                    text=text,
                    score=float(getattr(hit, "score", 0.0)),
                    metadata=payload,
                    method="dense",
                )
            )
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[tuple[str, str], dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            source = str(result.metadata.get("source", ""))
            key = (result.text, source)
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in fused
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
