from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.knowledge.loaders.base import DocumentLoadError, DocumentLoader
from app.knowledge.models import LoadOptions, LoadedDocument, PageText, SourceType


class PDFLoader(DocumentLoader):
    source_type: SourceType = "pdf"

    def load(self, path: str | Path, options: LoadOptions | None = None) -> LoadedDocument:
        path = Path(path).expanduser()
        source = self.source_metadata(path, options)
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                raise DocumentLoadError(f"Encrypted PDFs are not supported: {path}")
            pages = [
                PageText(page_number=index, text=page.extract_text() or "")
                for index, page in enumerate(reader.pages, start=1)
            ]
        except DocumentLoadError:
            raise
        except (OSError, PdfReadError) as exc:
            raise DocumentLoadError(f"Could not read PDF {path}: {exc}") from exc
        except Exception as exc:
            raise DocumentLoadError(f"Could not extract text from PDF {path}: {exc}") from exc

        combined_text = "\n\n".join(page.text for page in pages)
        if not combined_text.strip():
            raise DocumentLoadError(
                f"PDF contains no extractable text: {path}. Image-only PDFs require OCR, which is not supported."
            )
        return LoadedDocument(source=source, raw_text=combined_text, pages=pages)
