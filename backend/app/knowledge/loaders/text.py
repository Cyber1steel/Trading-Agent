from pathlib import Path

from app.knowledge.loaders.base import DocumentLoader, loaded_text_document, read_utf8
from app.knowledge.models import LoadOptions, LoadedDocument, SourceType


class TextLoader(DocumentLoader):
    source_type: SourceType = "txt"

    def load(self, path: str | Path, options: LoadOptions | None = None) -> LoadedDocument:
        path = Path(path).expanduser()
        source = self.source_metadata(path, options)
        return loaded_text_document(source, read_utf8(path))
