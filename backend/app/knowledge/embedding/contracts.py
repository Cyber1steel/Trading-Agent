from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.knowledge.models import DocumentChunk, SourceMetadata, SourceType


class ChunkForEmbedding(BaseModel):
    """Embedding input that retains the Phase 2A record as its source of truth."""

    chunk: DocumentChunk
    source: SourceMetadata
    processed_file_path: Path
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector: list[float]


class EmbeddingIngestionReport(BaseModel):
    processed_file_paths: list[str]
    provider: str
    model: str
    dimension: int
    chunks_seen: int
    embeddings_created: int
    embeddings_updated: int
    embeddings_skipped: int


class RetrievalFilters(BaseModel):
    """Small, explicit metadata filter set for semantic retrieval."""

    model_config = ConfigDict(extra="forbid")

    source_id: str | None = None
    document_id: str | None = None
    source_type: SourceType | None = None
    file_path: str | None = None
    page_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def source_and_document_ids_agree(self) -> "RetrievalFilters":
        if self.source_id and self.document_id and self.source_id != self.document_id:
            raise ValueError("source_id and document_id refer to the same Phase 2A source and must agree")
        return self

    @property
    def effective_source_id(self) -> str | None:
        return self.source_id or self.document_id


class StoredEmbeddingHit(BaseModel):
    chunk_id: str
    source_id: str
    source_type: SourceType
    title: str
    author: str | None
    source_url: str | None
    source_file_path: str
    processed_file_path: str
    published_at: datetime | None
    page_number: int | None
    chunk_index: int
    char_start: int | None
    char_end: int | None
    content_sha256: str
    distance: float


class RetrievalResult(BaseModel):
    """A source-resolved chunk and its cosine distance; distance is not probability."""

    chunk_id: str
    source_id: str
    document_id: str
    source_type: SourceType
    title: str
    author: str | None
    source_url: str | None
    file_path: str
    page_number: int | None
    chunk_index: int
    char_start: int | None
    char_end: int | None
    text: str
    distance: float = Field(ge=0, le=2)
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    processed_file_path: str
    ingestion_chunk_size: int = Field(gt=0)
    ingestion_overlap: int = Field(ge=0)
