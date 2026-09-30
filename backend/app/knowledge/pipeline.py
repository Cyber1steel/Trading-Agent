import json
import re
from pathlib import Path

from app.knowledge.chunker import chunk_text
from app.knowledge.cleaner import clean_text
from app.knowledge.loaders import MarkdownLoader, PDFLoader, TextLoader
from app.knowledge.loaders.base import DocumentLoadError, DocumentLoader
from app.knowledge.models import (
    LoadOptions,
    LoadedDocument,
    ProcessedDocument,
    TextSegment,
)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150

_LOADERS: dict[str, DocumentLoader] = {
    ".txt": TextLoader(),
    ".md": MarkdownLoader(),
    ".pdf": PDFLoader(),
}


class DocumentIngestionError(ValueError):
    """Raised when a source cannot be processed by the ingestion pipeline."""


def load_document(path: str | Path, options: LoadOptions | None = None) -> LoadedDocument:
    path = Path(path).expanduser()
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        supported = ", ".join(sorted(_LOADERS))
        raise DocumentIngestionError(
            f"Unsupported file extension '{path.suffix or '(none)'}'. Supported extensions: {supported}"
        )
    return loader.load(path, options)


def process_document(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    options: LoadOptions | None = None,
) -> ProcessedDocument:
    """Load, clean, and chunk one local TXT, Markdown, or PDF document."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    try:
        loaded = load_document(path, options)
    except DocumentLoadError as exc:
        raise DocumentIngestionError(str(exc)) from exc

    cleaned_parts: list[str] = []
    segments: list[TextSegment] = []
    char_offset = 0
    for page in loaded.pages:
        cleaned_page = clean_text(page.text)
        if not cleaned_page:
            continue
        if cleaned_parts:
            cleaned_parts.append("\n\n")
            char_offset += 2
        page_start = char_offset
        cleaned_parts.append(cleaned_page)
        char_offset += len(cleaned_page)
        segments.append(
            TextSegment(start=page_start, end=char_offset, page_number=page.page_number)
        )

    cleaned_document = "".join(cleaned_parts)
    if not cleaned_document:
        raise DocumentIngestionError(f"Source document contains no text: {loaded.source.file_path}")

    chunks = chunk_text(
        cleaned_document,
        loaded.source,
        chunk_size=chunk_size,
        overlap=overlap,
        segments=segments,
    )
    return ProcessedDocument(
        source=loaded.source,
        chunk_size=chunk_size,
        overlap=overlap,
        extracted_characters=len(loaded.raw_text),
        cleaned_characters=len(cleaned_document),
        chunks=chunks,
    )


def processed_output_path(document: ProcessedDocument, output_dir: str | Path) -> Path:
    """Return a predictable, collision-resistant filename for one source."""
    source_stem = Path(document.source.file_path).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source_stem).strip(".-_") or "document"
    return Path(output_dir) / f"{safe_stem}-{document.source.source_id[:10]}.json"


def write_processed_document(
    document: ProcessedDocument,
    output_dir: str | Path = "data/processed",
) -> Path:
    """Write source metadata and chunk metadata as UTF-8 JSON."""
    output_path = processed_output_path(document, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
