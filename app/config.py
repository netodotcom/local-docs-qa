"""Central configuration, overridable via environment variables or a .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where source documents live and where the built index is stored.
DOCS_DIR = Path(os.getenv("DOCS_DIR", PROJECT_ROOT / "docs"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", PROJECT_ROOT / "storage"))

# File extensions picked up during ingestion.
DOC_EXTENSIONS = {".txt", ".md"}

# Chunking (sizes in characters).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embeddings. Runs locally via sentence-transformers; no API key needed.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Hybrid retrieval (dense FAISS + sparse BM25).
# Number of chunks finally handed to the LLM, after re-ranking.
TOP_K = int(os.getenv("TOP_K", "4"))
# Size of the fused candidate pool passed to the re-ranker.
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "12"))
# Weight of the dense (semantic) score in the fusion; 1 - alpha goes to BM25.
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

# Re-ranking. Cross-encoder run locally via sentence-transformers.
# Swap for e.g. "BAAI/bge-reranker-large" if you want a heavier model.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Minimum re-ranker relevance (sigmoid, 0..1) of the best chunk for the
# question to count as answerable from the documents at all. Cross-encoder
# scores are NOT calibrated probabilities — with the default ms-marco model,
# off-corpus questions bottom out around 2e-5 while answerable ones stay above
# 2e-3, so this gate is deliberately permissive. The LLM's own "not in context"
# check and the grounding validation catch what slips through.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.001"))

# Answer generation (Anthropic).
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-opus-4-8")
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "1024"))

# Grounding validation: the generated answer must reach this cosine similarity
# against at least one of the retrieved chunks, or the request is rejected.
ANSWER_MIN_SIMILARITY = float(os.getenv("ANSWER_MIN_SIMILARITY", "0.45"))
