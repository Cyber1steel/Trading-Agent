"""Semantic search with Phase 2A JSON files as the canonical chunk source."""

from app.embeddings.base import EmbeddingProvider
from app.embeddings.validation import validate_vector
from app.knowledge.embedding.contracts import RetrievalFilters, RetrievalResult
from app.knowledge.embedding.errors import EmptyQueryError
from app.knowledge.embedding.provenance import resolve_stored_hits
from app.knowledge.embedding.repository import EmbeddingRepository


class SemanticRetrievalService:
    def __init__(self, provider: EmbeddingProvider, repository: EmbeddingRepository):
        self.provider = provider
        self.repository = repository

    def search(self, query: str, top_k: int = 5, filters: RetrievalFilters | None = None) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise EmptyQueryError("Search query must contain non-whitespace text.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = validate_vector(self.provider.embed_text(query.strip(), input_type="query"), self.provider.dimension)
        hits = self.repository.search(vector, self.provider, top_k, filters)
        return resolve_stored_hits(hits, retrieval_mode="semantic", provider=self.provider)
