"""FastAPI application exposing the question-answering endpoints.

Endpoints:
    POST /ask                      — answer a question from the indexed documents
    POST /ingest                   — start a background re-ingestion, returns 202 + task id
    GET  /ingest/status/{task_id}  — poll an ingestion task's state
    GET  /health                   — index status

Ingestion runs off the event loop: FastAPI's BackgroundTasks executes the sync
worker in a threadpool, so /ask keeps serving while documents are re-indexed.
The finished index is swapped into the app state atomically under a lock.
"""

import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from app import config, ingest
from app.embeddings import Embedder
from app.rag import (
    AnswerGenerationError,
    EmptyQuestionError,
    NoRelevantContentError,
    RAGService,
    UngroundedAnswerError,
)
from app.retrieval import HybridRetriever, Reranker
from app.tasks import IngestionInProgressError, TaskRegistry
from app.vector_store import IndexNotBuiltError, VectorStore


class AskRequest(BaseModel):
    question: str = ""


class SourceOut(BaseModel):
    source: str
    chunk_id: int
    score: float  # re-ranker relevance, 0..1
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    grounding_score: float


class IngestAccepted(BaseModel):
    task_id: str
    status: str
    status_url: str


state: dict = {"embedder": None, "reranker": None, "rag": None}
state_lock = threading.Lock()
registry = TaskRegistry()


def _build_rag(store: VectorStore) -> RAGService:
    return RAGService(
        retriever=HybridRetriever(state["embedder"], store),
        reranker=state["reranker"],
        embedder=state["embedder"],
    )


def _run_ingestion(task_id: str) -> None:
    """Background worker: rebuild the index, then swap it into the app state.

    Runs on a threadpool thread (BackgroundTasks runs sync callables via
    run_in_threadpool), so the event loop keeps serving requests. The new
    RAGService is fully constructed before the swap, and the swap itself is
    a single reference assignment under state_lock — /ask never observes a
    half-updated index.
    """
    registry.mark_processing(task_id)
    try:
        store = ingest.build_index(embedder=state["embedder"])
        rag = _build_rag(store)
    except Exception as exc:  # noqa: BLE001 — any failure must land in the task record
        registry.mark_failed(task_id, str(exc))
        return
    with state_lock:
        state["rag"] = rag
    registry.mark_completed(task_id, chunks=len(store.chunks))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are always needed; the index may not exist yet, in which case
    # /ask returns 503 until an ingestion completes.
    state["embedder"] = Embedder()
    state["reranker"] = Reranker()
    try:
        store = VectorStore.load()
        state["rag"] = _build_rag(store)
    except IndexNotBuiltError:
        state["rag"] = None
    yield


app = FastAPI(title="Local Docs Q&A", lifespan=lifespan)


@app.get("/health")
def health():
    rag: RAGService | None = state["rag"]
    return {
        "status": "ok",
        "index_ready": rag is not None,
        "chunks": len(rag.retriever.store.chunks) if rag else 0,
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker_model": config.RERANKER_MODEL,
        "answer_model": config.ANSWER_MODEL,
    }


@app.post("/ingest", response_model=IngestAccepted, status_code=202)
def start_ingestion(background_tasks: BackgroundTasks):
    """Kick off ingestion in the background and return immediately."""
    try:
        task = registry.create()
    except IngestionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    background_tasks.add_task(_run_ingestion, task.task_id)
    return IngestAccepted(
        task_id=task.task_id,
        status=task.status,
        status_url=f"/ingest/status/{task.task_id}",
    )


@app.get("/ingest/status/{task_id}")
def ingestion_status(task_id: str):
    task = registry.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown ingestion task: {task_id}")
    return task.as_dict()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    with state_lock:
        rag: RAGService | None = state["rag"]
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="Index not built yet. Add files to the docs folder and call POST /ingest.",
        )

    try:
        result = rag.ask(request.question)
    except EmptyQuestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NoRelevantContentError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except UngroundedAnswerError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except AnswerGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceOut(source=s.source, chunk_id=s.chunk_id, score=round(s.score, 4), text=s.text)
            for s in result.sources
        ],
        grounding_score=round(result.grounding_score, 4),
    )
