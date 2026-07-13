"""The question-answering pipeline: hybrid retrieve -> rerank -> generate -> validate.

Every failure mode maps to a typed error so the API layer can return a clear
HTTP status and message:

    EmptyQuestionError    — question is blank
    NoRelevantContentError — no chunk passes the re-ranker relevance gate, or the
                             model itself reported the context doesn't cover it
    UngroundedAnswerError — the generated answer drifted too far from the
                            retrieved chunks (grounding check failed)
    AnswerGenerationError — the model call failed or was refused
"""

from dataclasses import dataclass

import anthropic
import numpy as np

from app import config
from app.embeddings import Embedder
from app.retrieval import HybridRetriever, Reranker
from app.vector_store import SearchResult

# The model is instructed to output exactly this token when the retrieved
# context does not contain the answer.
_NOT_IN_CONTEXT = "NOT_IN_CONTEXT"

_SYSTEM_PROMPT = f"""You are a question-answering assistant restricted to a fixed set of documents.

Rules:
- Answer using ONLY the information in the provided context chunks. Do not use outside knowledge.
- Be concise and factual. Quote or closely paraphrase the source material.
- If the context does not contain the information needed to answer, reply with exactly {_NOT_IN_CONTEXT} and nothing else."""


class EmptyQuestionError(ValueError):
    pass


class NoRelevantContentError(RuntimeError):
    pass


class UngroundedAnswerError(RuntimeError):
    pass


class AnswerGenerationError(RuntimeError):
    pass


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[SearchResult]
    grounding_score: float


class RAGService:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker, embedder: Embedder):
        self.retriever = retriever
        self.reranker = reranker
        self.embedder = embedder  # used by the grounding check
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        # Lazy so the server can start (and serve /health, /ingest) without a
        # key; a missing key surfaces as a clear error on /ask instead.
        if self._client is None:
            try:
                self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
            except anthropic.AnthropicError as exc:
                raise AnswerGenerationError(
                    "Anthropic client could not be created. Set ANTHROPIC_API_KEY "
                    "in your environment or .env file."
                ) from exc
        return self._client

    def ask(self, question: str) -> Answer:
        question = (question or "").strip()
        if not question:
            raise EmptyQuestionError("Question must not be empty.")

        sources = self._retrieve(question)
        answer_text = self._generate(question, sources)
        grounding_score = self._validate(answer_text, sources)

        return Answer(
            question=question,
            answer=answer_text,
            sources=sources,
            grounding_score=grounding_score,
        )

    def _retrieve(self, question: str) -> list[SearchResult]:
        """Hybrid retrieval (dense + BM25) followed by cross-encoder re-ranking.

        The chunks handed to the LLM are exactly the re-ranker's top-K, in the
        re-ranker's order. Relevance is gated on the re-ranker score, which is
        far sharper than raw retrieval similarity.
        """
        candidates = self.retriever.retrieve(question)
        ranked = self.reranker.rerank(question, candidates)
        if not ranked or ranked[0].score < config.RERANK_MIN_SCORE:
            raise NoRelevantContentError(
                "No sufficiently relevant content was found in the indexed documents "
                f"(best re-ranker relevance {ranked[0].score:.4f} < {config.RERANK_MIN_SCORE:.4f})."
                if ranked
                else "The index contains no documents."
            )
        return ranked

    def _generate(self, question: str, sources: list[SearchResult]) -> str:
        context = "\n\n".join(
            f"[Chunk {i + 1} — {result.source}]\n{result.text}"
            for i, result in enumerate(sources)
        )
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        try:
            response = self.client.messages.create(
                model=config.ANSWER_MODEL,
                max_tokens=config.ANSWER_MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except (anthropic.AuthenticationError, TypeError) as exc:
            # The SDK raises TypeError when no credential can be resolved at all,
            # and AuthenticationError when the key is present but invalid.
            raise AnswerGenerationError(
                "Anthropic authentication failed. Set ANTHROPIC_API_KEY in your environment or .env file."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise AnswerGenerationError("Anthropic rate limit hit — retry shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise AnswerGenerationError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise AnswerGenerationError("Could not reach the Anthropic API (network error).") from exc

        if response.stop_reason == "refusal":
            raise AnswerGenerationError("The model refused to answer this question.")

        answer = "".join(block.text for block in response.content if block.type == "text").strip()

        if not answer:
            raise AnswerGenerationError("The model returned an empty answer.")
        if answer == _NOT_IN_CONTEXT:
            raise NoRelevantContentError(
                "The retrieved documents do not contain the information needed to answer this question."
            )
        return answer

    def _validate(self, answer: str, sources: list[SearchResult]) -> float:
        """Grounding check: the answer must be semantically close to a source chunk.

        Embeds the answer and compares it (cosine similarity — vectors are
        unit-normalized) against each retrieved chunk. An answer that drifted
        away from every chunk is treated as unsupported and rejected.
        """
        answer_vector = self.embedder.embed_one(answer)
        chunk_vectors = self.embedder.embed([result.text for result in sources])
        score = float(np.max(chunk_vectors @ answer_vector))

        if score < config.ANSWER_MIN_SIMILARITY:
            raise UngroundedAnswerError(
                f"Answer failed the grounding check (similarity to sources {score:.2f} "
                f"< required {config.ANSWER_MIN_SIMILARITY:.2f}). The answer was withheld "
                "because it is not close enough to the retrieved source content."
            )
        return score
