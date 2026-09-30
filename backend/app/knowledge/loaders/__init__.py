"""Local source document loaders."""

from app.knowledge.loaders.markdown import MarkdownLoader
from app.knowledge.loaders.pdf import PDFLoader
from app.knowledge.loaders.text import TextLoader

__all__ = ["MarkdownLoader", "PDFLoader", "TextLoader"]
