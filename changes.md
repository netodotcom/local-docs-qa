# Changes

## 2026-07-13 — Hybrid Search + Cross-Encoder Re-ranking

The query pipeline was refactored from pure dense retrieval to a two-stage
hybrid retrieval + re-ranking architecture (`app/retrieval.py`).

- **Hybrid retrieval**: every question now runs through two strategies in
  parallel — dense semantic search (FAISS over sentence-transformers
  embeddings, as before) and sparse keyword search (BM25 via `rank-bm25`,
  built in memory over the same chunks at load time). Each strategy's scores
  are min-max normalized to [0, 1] within the query, then fused as
  `HYBRID_ALPHA * dense + (1 - HYBRID_ALPHA) * sparse` (default 0.5/0.5) into
  a candidate pool of `RETRIEVAL_CANDIDATES` chunks (default 12).
- **Re-ranking**: a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`
  by default, configurable via `RERANKER_MODEL`, e.g. `BAAI/bge-reranker-large`)
  scores every (question, chunk) candidate pair jointly. The top `TOP_K`
  chunks **in re-ranked order** are what the LLM receives as context,
  maximizing context precision.
- **Relevance gate moved to the re-ranker**: the old `RETRIEVAL_MIN_SCORE`
  (raw dense cosine) was replaced by `RERANK_MIN_SCORE` (default `0.001`,
  sigmoid-mapped cross-encoder score). Measured on the sample corpus:
  off-corpus questions score ≤ 2e-5, answerable ones ≥ 2e-3.
- **API compatibility**: the `/ask` response structure is unchanged
  (`question`, `answer`, `sources[]`, `grounding_score`); `sources[].score`
  now carries the re-ranker relevance (0..1) instead of raw cosine.
- New dependency: `rank-bm25`.

## 2026-07-13 — Asynchronous, Non-Blocking Ingestion

Document ingestion was refactored so heavy work never blocks the FastAPI
event loop (`app/tasks.py`, `app/main.py`).

- **`POST /ingest` returns immediately** with HTTP **202 Accepted**, a
  `task_id`, and a `status_url`, replacing the old synchronous `/reindex`
  endpoint (breaking change: `/reindex` was removed).
- **Background worker**: the heavy lifting — reading files, chunking,
  computing embeddings, writing the FAISS index — runs via FastAPI
  `BackgroundTasks`, which executes the sync worker on a threadpool thread,
  keeping the event loop free to serve `/ask` during re-indexing.
- **Thread-safe state swap**: the new index and `RAGService` are fully
  constructed before being swapped into the shared app state as a single
  reference assignment under a `threading.Lock` — requests never observe a
  half-built index. A task registry (also lock-guarded) enforces that only
  one ingestion runs at a time; a concurrent `POST /ingest` returns
  **409 Conflict** with the active task's id.
- **`GET /ingest/status/{task_id}`** reports the task lifecycle:
  `pending` → `processing` → `completed` (with chunk count) or `failed`
  (with the error message), plus created/finished timestamps.
