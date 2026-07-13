"""Document ingestion: read the docs folder, chunk, embed, and persist the index.

Run directly (`python -m app.ingest`) or call `build_index()` from the API's
/reindex endpoint.
"""

from pathlib import Path

from app import config
from app.chunking import Chunk, chunk_text
from app.embeddings import Embedder
from app.vector_store import VectorStore


def load_documents(docs_dir: Path = config.DOCS_DIR) -> list[tuple[str, str]]:
    """Return (filename, text) pairs for every supported file in docs_dir."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")
    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in config.DOC_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append((str(path.relative_to(docs_dir)), text))
    return documents


def build_index(embedder: Embedder | None = None) -> VectorStore:
    documents = load_documents()
    if not documents:
        raise ValueError(
            f"No documents found in {config.DOCS_DIR}. "
            f"Drop {'/'.join(sorted(config.DOC_EXTENSIONS))} files there and re-run."
        )

    chunks: list[Chunk] = []
    for filename, text in documents:
        chunks.extend(chunk_text(text, source=filename))

    embedder = embedder or Embedder()
    vectors = embedder.embed([c.text for c in chunks])

    store = VectorStore.build(chunks, vectors, model_name=embedder.model_name)
    store.save()
    print(f"Indexed {len(chunks)} chunks from {len(documents)} document(s) -> {config.STORAGE_DIR}")
    return store


if __name__ == "__main__":
    build_index()
