from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["pdf", "txt", "markdown", "web", "youtube", "podcast"]


class SourceMetadata(BaseModel):
    """Traceable metadata for a source document."""

    source_id: str
    source_type: SourceType
    title: str
    author: str | None = None
    source_url: str | None = None
    file_path: str
    published_at: datetime | None = None


class PageText(BaseModel):
    """Text extracted from one source page; page numbers are one-based."""

    page_number: int | None = None
    text: str


class LoadedDocument(BaseModel):
    """Text and metadata returned by a document loader."""

    source: SourceMetadata
    raw_text: str
    pages: list[PageText] = Field(default_factory=list)


class LoadOptions(BaseModel):
    """Optional metadata supplied by a caller when loading a document."""

    title: str | None = None
    author: str | None = None
    source_url: str | None = None
    published_at: datetime | None = None


class TextSegment(BaseModel):
    """A range in cleaned document text with its optional source page."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)


class DocumentChunk(BaseModel):
    """A text chunk with enough metadata to trace it back to its source."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    source_id: str
    source_type: SourceType
    title: str
    file_path: str
    source_url: str | None = None
    page_number: int | None = None
    chunk_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str = Field(min_length=1)


class ProcessedDocument(BaseModel):
    """Serializable result of cleaning and chunking a source document."""

    source: SourceMetadata
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    extracted_characters: int = Field(ge=0)
    cleaned_characters: int = Field(ge=0)
    chunks: list[DocumentChunk]
