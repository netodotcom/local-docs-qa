"""Local embedding model wrapper (sentence-transformers, no API key required).

Embeddings are L2-normalized, so the inner product of two vectors is their
cosine similarity. Both the FAISS index and the grounding check rely on this.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from app import config


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dimension) float32 array of unit vectors."""
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
