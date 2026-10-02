"""Embedding ingestion, PostgreSQL persistence, and semantic retrieval."""

from app.knowledge.embedding.contracts import RetrievalFilters, RetrievalResult
from app.knowledge.embedding.hybrid import HybridRetrievalService
from app.knowledge.embedding.ingestion import EmbeddingIngestionService
from app.knowledge.embedding.lexical import LexicalIndexingService, LexicalRetrievalService
from app.knowledge.embedding.repository import EmbeddingRepository
from app.knowledge.embedding.retrieval import SemanticRetrievalService

__all__ = [
    "EmbeddingIngestionService",
    "EmbeddingRepository",
    "HybridRetrievalService",
    "LexicalIndexingService",
    "LexicalRetrievalService",
    "RetrievalFilters",
    "RetrievalResult",
    "SemanticRetrievalService",
]
