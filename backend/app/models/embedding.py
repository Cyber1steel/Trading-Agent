from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TSVECTOR

from app.core.database import Base


class EmbeddingSpace(Base):
    """One provider/model pair with a single immutable vector dimension."""

    __tablename__ = "embedding_spaces"
    __table_args__ = (
        UniqueConstraint("provider", "model", name="uq_embedding_spaces_provider_model"),
    )

    provider: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(255), primary_key=True)
    dimension: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class KnowledgeEmbedding(Base):
    """Vector and retrieval metadata, with chunk text kept in Phase 2A JSON."""

    __tablename__ = "knowledge_embeddings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["embedding_provider", "embedding_model", "embedding_dimension"],
            ["embedding_spaces.provider", "embedding_spaces.model", "embedding_spaces.dimension"],
            name="fk_knowledge_embeddings_space",
        ),
        UniqueConstraint(
            "chunk_id",
            "embedding_provider",
            "embedding_model",
            name="uq_knowledge_embedding_chunk_space",
        ),
        CheckConstraint("embedding_dimension > 0", name="ck_knowledge_embedding_dimension"),
        CheckConstraint(
            "vector_dims(embedding) = embedding_dimension",
            name="ck_knowledge_embedding_vector_dimension",
        ),
        CheckConstraint("chunk_index >= 0", name="ck_knowledge_embedding_chunk_index"),
        CheckConstraint(
            "page_number IS NULL OR page_number > 0", name="ck_knowledge_embedding_page_number"
        ),
        CheckConstraint(
            "char_start IS NULL OR char_start >= 0", name="ck_knowledge_embedding_char_start"
        ),
        CheckConstraint(
            "char_end IS NULL OR char_end >= char_start", name="ck_knowledge_embedding_char_end"
        ),
        Index(
            "ix_knowledge_embeddings_space",
            "embedding_provider",
            "embedding_model",
            "embedding_dimension",
        ),
        Index("ix_knowledge_embeddings_source", "source_id"),
        Index("ix_knowledge_embeddings_source_type", "source_type"),
        Index("ix_knowledge_embeddings_document", "source_file_path"),
        Index("ix_knowledge_embeddings_page", "page_number"),
        Index(
            "ix_knowledge_embeddings_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    chunk_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    author: Mapped[str | None] = mapped_column(String(512))
    source_url: Mapped[str | None] = mapped_column(Text)
    source_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    processed_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    page_number: Mapped[int | None] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[Any] = mapped_column(VECTOR(), nullable=False)
    # Derived lexical index only; the canonical chunk text remains in Phase 2A JSON.
    search_vector: Mapped[Any | None] = mapped_column(TSVECTOR, nullable=True)
    embedding_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
