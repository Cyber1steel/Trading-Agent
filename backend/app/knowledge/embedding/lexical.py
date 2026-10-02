"""PostgreSQL full-text retrieval for Phase 2A chunk evidence."""

from pathlib import Path
from typing import Iterable

from app.core.config import get_settings
from app.knowledge.embedding.contracts import (
    LexicalIndexingReport,
    RetrievalFilters,
    RetrievalResult,
)
from app.knowledge.embedding.errors import EmptyQueryError
from app.knowledge.embedding.ingestion import records_from_document
from app.knowledge.embedding.provenance import resolve_stored_hits
from app.knowledge.embedding.repository import EmbeddingRepository


class LexicalRetrievalService:
    def __init__(self, repository: EmbeddingRepository):
        self.repository = repository

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise EmptyQueryError("Search query must contain non-whitespace text.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        hits = self.repository.search_lexical(query.strip(), top_k, filters)
        return resolve_stored_hits(hits, retrieval_mode="lexical")


class LexicalIndexingService:
    """Backfill derived FTS vectors for existing embeddings without loading a model."""

    def __init__(self, repository: EmbeddingRepository, batch_size: int | None = None):
        self.repository = repository
        self.batch_size = batch_size if batch_size is not None else get_settings().embedding_batch_size
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def index_files(self, paths: Iterable[Path]) -> LexicalIndexingReport:
        file_paths = [Path(path) for path in paths]
        seen = updated = 0
        for path in file_paths:
            records = records_from_document(path)
            seen += len(records)
            for offset in range(0, len(records), self.batch_size):
                updated += self.repository.refresh_lexical_vectors(
                    records[offset:offset + self.batch_size]
                )
        return LexicalIndexingReport(
            processed_file_paths=[str(path.resolve()) for path in file_paths],
            chunks_seen=seen,
            database_rows_updated=updated,
        )
