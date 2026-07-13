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

# Retrieval.
TOP_K = int(os.getenv("TOP_K", "4"))
# Minimum cosine similarity between the question and the best chunk for the
# question to count as answerable from the documents at all.
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.25"))

# Answer generation (Anthropic).
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-opus-4-8")
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "1024"))

# Grounding validation: the generated answer must reach this cosine similarity
# against at least one of the retrieved chunks, or the request is rejected.
ANSWER_MIN_SIMILARITY = float(os.getenv("ANSWER_MIN_SIMILARITY", "0.45"))
