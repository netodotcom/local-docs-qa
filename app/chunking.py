"""Split raw document text into overlapping chunks suitable for embedding.

Strategy: split on paragraph boundaries first, then pack whole paragraphs into
chunks up to CHUNK_SIZE characters. Paragraphs longer than a chunk are hard-split
on sentence boundaries. Consecutive chunks share CHUNK_OVERLAP characters of
context so that facts sitting on a boundary are not lost.
"""

import re
from dataclasses import dataclass

from app import config


@dataclass
class Chunk:
    source: str  # filename the chunk came from
    text: str


_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    """Break a paragraph that exceeds max_size on sentence boundaries."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(paragraph):
        if current and len(current) + len(sentence) + 1 > max_size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
        # A single sentence longer than max_size gets hard-cut.
        while len(current) > max_size:
            pieces.append(current[:max_size])
            current = current[max_size:]
    if current:
        pieces.append(current)
    return pieces


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            units.extend(_split_long_paragraph(paragraph, chunk_size))
        else:
            units.append(paragraph)

    chunks: list[Chunk] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > chunk_size:
            chunks.append(Chunk(source=source, text=current))
            # Carry the tail of the previous chunk into the next one.
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{unit}".strip()
        else:
            current = f"{current}\n\n{unit}".strip()
    if current:
        chunks.append(Chunk(source=source, text=current))

    return chunks
