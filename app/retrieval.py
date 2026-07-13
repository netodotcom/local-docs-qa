"""Retrieval strategies: dense (FAISS), sparse (BM25), hybrid fusion, re-ranking.

Pipeline:
    HybridRetriever.retrieve(question)
        dense:  embed question -> FAISS cosine top-N
        sparse: tokenize question -> BM25 top-N
        fuse:   min-max normalize each score list, weighted sum (HYBRID_ALPHA)
        -> candidate pool (RETRIEVAL_CANDIDATES chunks)

    Reranker.rerank(question, candidates, top_k)
        cross-encoder scores every (question, chunk) pair jointly
        -> final top-K, ordered by relevance, score in [0, 1] (sigmoid)

The reranker score is the relevance signal the rest of the app gates on;
the fused hybrid score is only used to pick which candidates get re-ranked.
"""

import re
from dataclasses import replace

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app import config
from app.embeddings import Embedder
from app.vector_store import SearchResult, VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    """Rescale a {chunk_id: score} map to [0, 1] within this query's results."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {chunk_id: 1.0 for chunk_id in scores}
    return {chunk_id: (score - lo) / (hi - lo) for chunk_id, score in scores.items()}


class SparseIndex:
    """BM25 keyword index over the store's chunks (built in memory at load time)."""

    def __init__(self, store: VectorStore):
        self._bm25 = BM25Okapi([_tokenize(chunk.text) for chunk in store.chunks])

    def search(self, question: str, top_k: int) -> dict[int, float]:
        scores = self._bm25.get_scores(_tokenize(question))
        top_ids = np.argsort(scores)[::-1][:top_k]
        # BM25 gives 0 to chunks sharing no terms with the query — not evidence.
        return {int(i): float(scores[i]) for i in top_ids if scores[i] > 0}


class HybridRetriever:
    """Combines dense semantic search and sparse keyword search into one pool."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store
        self.sparse = SparseIndex(store)

    def retrieve(
        self,
        question: str,
        candidates: int = config.RETRIEVAL_CANDIDATES,
        alpha: float = config.HYBRID_ALPHA,
    ) -> list[SearchResult]:
        dense_results = self.store.search(self.embedder.embed_one(question), top_k=candidates)
        dense = _min_max_normalize({r.chunk_id: r.score for r in dense_results})
        sparse = _min_max_normalize(self.sparse.search(question, top_k=candidates))

        fused = {
            chunk_id: alpha * dense.get(chunk_id, 0.0) + (1 - alpha) * sparse.get(chunk_id, 0.0)
            for chunk_id in dense.keys() | sparse.keys()
        }
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[:candidates]
        return [
            SearchResult(
                chunk_id=chunk_id,
                source=self.store.chunks[chunk_id].source,
                text=self.store.chunks[chunk_id].text,
                score=fused[chunk_id],
            )
            for chunk_id in ranked_ids
        ]


class Reranker:
    """Cross-encoder re-ranker: scores each (question, chunk) pair jointly.

    Unlike the bi-encoder used for retrieval, the cross-encoder reads question
    and chunk together, giving a much sharper relevance estimate. Scores are
    passed through a sigmoid so they land in [0, 1] regardless of model.
    """

    def __init__(self, model_name: str = config.RERANKER_MODEL):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self, question: str, results: list[SearchResult], top_k: int = config.TOP_K
    ) -> list[SearchResult]:
        if not results:
            return []
        logits = self._model.predict(
            [(question, result.text) for result in results], show_progress_bar=False
        )
        relevance = 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)))
        reranked = [
            replace(result, score=float(score))
            for result, score in zip(results, relevance)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
