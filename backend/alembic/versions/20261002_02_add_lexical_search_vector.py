"""Add PostgreSQL full-text search index for Phase 2A chunks."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261002_02"
down_revision: str | None = "20261002_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Derived index data only. Phase 2A JSON remains the canonical chunk text.
    # Existing rows remain NULL until embed_documents.py is rerun to backfill from JSON.
    op.add_column(
        "knowledge_embeddings",
        sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True),
    )
    op.create_index(
        "ix_knowledge_embeddings_search_vector",
        "knowledge_embeddings",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.execute(
        "COMMENT ON COLUMN knowledge_embeddings.search_vector IS "
        "'Derived PostgreSQL full-text index; canonical chunk text stays in Phase 2A JSON.'"
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_embeddings_search_vector", table_name="knowledge_embeddings")
    op.drop_column("knowledge_embeddings", "search_vector")
