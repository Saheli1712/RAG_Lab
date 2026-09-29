"""Small, inspectable RAG primitives. No process-local document storage."""

import math
import re
from io import BytesIO

from pypdf import PdfReader

MAX_PDF_BYTES = 2_000_000
MAX_PAGES = 40
MAX_CHARS = 72_000
MAX_CHUNKS = 100
CHUNK_SIZE = 900
OVERLAP = 150


class DocumentError(ValueError):
    pass


def extract_pages(data: bytes) -> list[dict]:
    if not data.startswith(b"%PDF-"):
        raise DocumentError("This file is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise DocumentError("This PDF is password protected. Upload an unlocked copy.")
        if len(reader.pages) > MAX_PAGES:
            raise DocumentError(f"This demo supports at most {MAX_PAGES} pages.")
        pages = []
        for number, page in enumerate(reader.pages, 1):
            cleaned = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if cleaned:
                pages.append({"page": number, "text": cleaned})
        if not pages:
            raise DocumentError("No selectable text was found. Scanned PDFs need OCR, which this demo does not include.")
        if sum(len(p["text"]) for p in pages) > MAX_CHARS:
            raise DocumentError("This PDF contains too much text for the demo (72,000 characters maximum).")
        return pages
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("The PDF could not be read. Check that it is not damaged or encrypted.") from exc


def make_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            if end < len(text):
                boundary = text.rfind(" ", start + CHUNK_SIZE // 2, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append({"id": len(chunks) + 1, "page": page["page"], "text": piece})
            if end >= len(text):
                break
            start = max(start + 1, end - OVERLAP)
    if len(chunks) > MAX_CHUNKS:
        raise DocumentError(f"This PDF produces over {MAX_CHUNKS} chunks. Upload a shorter document.")
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Embedding dimensions do not match")
    dot = sum(x * y for x, y in zip(a, b))
    magnitude = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / magnitude if magnitude else 0.0


def top_matches(chunks: list[dict], vectors: list[list[float]], query: list[float], count: int = 4) -> list[dict]:
    if len(chunks) != len(vectors):
        raise ValueError("Chunk and vector counts do not match")
    ranked = [{**chunk, "score": round(cosine(vector, query), 4)} for chunk, vector in zip(chunks, vectors)]
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:count]
