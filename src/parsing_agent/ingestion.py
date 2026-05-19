from __future__ import annotations

import mimetypes
from pathlib import Path

from pypdf import PdfReader

from parsing_agent.models import DocumentSource

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".html", ".xml"}


def detect_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def extract_text_from_pdf(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip(), len(pages)


def _is_text_like(path: Path, media_type: str) -> bool:
    return media_type.startswith("text/") or path.suffix.lower() in _TEXT_SUFFIXES


def extract_source_text(path: Path, media_type: str) -> tuple[str | None, int | None]:
    if _is_text_like(path, media_type):
        return path.read_text(encoding="utf-8"), None
    if media_type == "application/pdf" or path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    return None, None


def build_document_source(path: Path, run_id: str) -> DocumentSource:
    resolved = path.resolve()
    media_type = detect_media_type(resolved)
    extracted_text, page_count = extract_source_text(resolved, media_type)
    return DocumentSource(
        path=resolved,
        media_type=media_type,
        size_bytes=resolved.stat().st_size,
        run_id=run_id,
        extracted_text=extracted_text,
        page_count=page_count,
    )
