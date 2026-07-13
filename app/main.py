"""FastAPI application exposing the question-answering endpoint.

Endpoints:
    POST /ask      — answer a question from the indexed documents
    POST /reindex  — rebuild the index from the docs folder without restarting
    GET  /health   — index status
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
from app.vector_store import IndexNotBuiltError, VectorStore


class AskRequest(BaseModel):
    question: str = ""


class SourceOut(BaseModel):
    source: str
    chunk_id: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    grounding_score: float


state: dict = {"embedder": None, "rag": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The embedding model is always needed; the index may not exist yet, in
    # which case /ask returns 503 until /reindex (or `python -m app.ingest`).
    state["embedder"] = Embedder()
    try:
        store = VectorStore.load()
        state["rag"] = RAGService(state["embedder"], store)
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
        "chunks": len(rag.store.chunks) if rag else 0,
        "embedding_model": config.EMBEDDING_MODEL,
        "answer_model": config.ANSWER_MODEL,
    }


@app.post("/reindex")
def reindex():
    try:
        store = ingest.build_index(embedder=state["embedder"])
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    state["rag"] = RAGService(state["embedder"], store)
    return {"status": "ok", "chunks": len(store.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    rag: RAGService | None = state["rag"]
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="Index not built yet. Add files to the docs folder and call POST /reindex "
            "(or run `python -m app.ingest` and restart).",
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
