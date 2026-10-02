"""Add pgvector embedding storage."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

revision: str = "20261002_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "embedding_spaces",
        sa.Column("provider", sa.String(length=64), primary_key=True),
        sa.Column("model", sa.String(length=255), primary_key=True),
        sa.Column("dimension", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("provider", "model", name="uq_embedding_spaces_provider_model"),
    )

    op.create_table(
        "knowledge_embeddings",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("chunk_id", sa.String(length=128), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("author", sa.String(length=512), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("source_file_path", sa.Text(), nullable=False),
        sa.Column("processed_file_path", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("embedding", VECTOR(), nullable=False),
        sa.Column("embedding_provider", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["embedding_provider", "embedding_model", "embedding_dimension"],
            ["embedding_spaces.provider", "embedding_spaces.model", "embedding_spaces.dimension"],
            name="fk_knowledge_embeddings_space",
        ),
        sa.UniqueConstraint(
            "chunk_id",
            "embedding_provider",
            "embedding_model",
            name="uq_knowledge_embedding_chunk_space",
        ),
        sa.CheckConstraint("embedding_dimension > 0", name="ck_knowledge_embedding_dimension"),
        sa.CheckConstraint(
            "vector_dims(embedding) = embedding_dimension",
            name="ck_knowledge_embedding_vector_dimension",
        ),
        sa.CheckConstraint("chunk_index >= 0", name="ck_knowledge_embedding_chunk_index"),
        sa.CheckConstraint(
            "page_number IS NULL OR page_number > 0", name="ck_knowledge_embedding_page_number"
        ),
        sa.CheckConstraint(
            "char_start IS NULL OR char_start >= 0", name="ck_knowledge_embedding_char_start"
        ),
        sa.CheckConstraint(
            "char_end IS NULL OR char_end >= char_start", name="ck_knowledge_embedding_char_end"
        ),
    )
    op.create_index(
        "ix_knowledge_embeddings_space",
        "knowledge_embeddings",
        ["embedding_provider", "embedding_model", "embedding_dimension"],
    )
    op.create_index("ix_knowledge_embeddings_source", "knowledge_embeddings", ["source_id"])
    op.create_index(
        "ix_knowledge_embeddings_source_type", "knowledge_embeddings", ["source_type"]
    )
    op.create_index(
        "ix_knowledge_embeddings_document", "knowledge_embeddings", ["source_file_path"]
    )
    op.create_index("ix_knowledge_embeddings_page", "knowledge_embeddings", ["page_number"])


def downgrade() -> None:
    op.drop_table("knowledge_embeddings")
    op.drop_table("embedding_spaces")
    # Leave the database-wide extension installed in case another schema uses it.
