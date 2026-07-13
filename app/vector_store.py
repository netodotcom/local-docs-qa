"""FAISS-backed vector index with chunk metadata, persisted to disk.

Layout inside STORAGE_DIR:
    index.faiss  — the FAISS inner-product index (vectors are unit-normalized,
                   so inner product == cosine similarity)
    chunks.json  — one entry per vector: {"source": ..., "text": ...}
    meta.json    — embedding model name and chunk count, used to detect a
                   model mismatch between ingestion and serving
"""

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app import config
from app.chunking import Chunk

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"
_META_FILE = "meta.json"


class IndexNotBuiltError(RuntimeError):
    """Raised when the index is queried before ingestion has been run."""


@dataclass
class SearchResult:
    chunk_id: int
    source: str
    text: str
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], model_name: str):
        self.index = index
        self.chunks = chunks
        self.model_name = model_name

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> "VectorStore":
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, chunks=chunks, model_name=model_name)

    @classmethod
    def load(cls, storage_dir: Path = config.STORAGE_DIR) -> "VectorStore":
        index_path = storage_dir / _INDEX_FILE
        if not index_path.exists():
            raise IndexNotBuiltError(
                f"No index found in {storage_dir}. Run `python -m app.ingest` first."
            )
        index = faiss.read_index(str(index_path))
        raw_chunks = json.loads((storage_dir / _CHUNKS_FILE).read_text(encoding="utf-8"))
        meta = json.loads((storage_dir / _META_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk(source=c["source"], text=c["text"]) for c in raw_chunks]
        return cls(index=index, chunks=chunks, model_name=meta["embedding_model"])

    def save(self, storage_dir: Path = config.STORAGE_DIR) -> None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(storage_dir / _INDEX_FILE))
        (storage_dir / _CHUNKS_FILE).write_text(
            json.dumps([{"source": c.source, "text": c.text} for c in self.chunks]),
            encoding="utf-8",
        )
        (storage_dir / _META_FILE).write_text(
            json.dumps({"embedding_model": self.model_name, "chunks": len(self.chunks)}),
            encoding="utf-8",
        )

    def search(self, query_vector: np.ndarray, top_k: int = config.TOP_K) -> list[SearchResult]:
        top_k = min(top_k, len(self.chunks))
        scores, ids = self.index.search(query_vector.reshape(1, -1), top_k)
        return [
            SearchResult(
                chunk_id=int(chunk_id),
                source=self.chunks[chunk_id].source,
                text=self.chunks[chunk_id].text,
                score=float(score),
            )
            for score, chunk_id in zip(scores[0], ids[0])
            if chunk_id != -1
        ]
