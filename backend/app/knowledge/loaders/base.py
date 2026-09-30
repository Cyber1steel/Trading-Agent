from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path

from app.knowledge.models import LoadOptions, LoadedDocument, PageText, SourceMetadata, SourceType


class DocumentLoadError(ValueError):
    """Raised when a local document cannot be extracted as text."""


class DocumentLoader(ABC):
    """Interface shared by local and future remote source loaders."""

    source_type: SourceType

    @abstractmethod
    def load(self, path: str | Path, options: LoadOptions | None = None) -> LoadedDocument:
        """Extract text and source metadata from a document path."""

    def source_metadata(self, path: str | Path, options: LoadOptions | None = None) -> SourceMetadata:
        options = options or LoadOptions()
        input_path = Path(path).expanduser()
        resolved_path = input_path.resolve()
        if not resolved_path.is_file():
            raise DocumentLoadError(f"Source file does not exist or is not a file: {input_path}")

        title = options.title
        if title is None or not title.strip():
            title = input_path.stem.replace("_", " ").replace("-", " ").strip()
            title = title[:1].upper() + title[1:] if title else resolved_path.name

        source_id = sha256(str(resolved_path).encode("utf-8")).hexdigest()[:24]
        return SourceMetadata(
            source_id=source_id,
            source_type=self.source_type,
            title=title,
            author=options.author,
            source_url=options.source_url,
            file_path=str(resolved_path),
            published_at=options.published_at,
        )


def read_utf8(path: Path) -> str:
    """Read UTF-8 while replacing invalid byte sequences instead of failing."""
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source_file:
            return source_file.read()
    except OSError as exc:
        raise DocumentLoadError(f"Could not read source file {path}: {exc}") from exc


def loaded_text_document(
    source: SourceMetadata,
    text: str,
) -> LoadedDocument:
    return LoadedDocument(source=source, raw_text=text, pages=[PageText(text=text)])
