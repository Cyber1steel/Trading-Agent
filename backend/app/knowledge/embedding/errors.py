class KnowledgeEmbeddingError(RuntimeError):
    """Base error for the embedding ingestion and retrieval layer."""


class ProcessedChunkError(KnowledgeEmbeddingError):
    """Raised when a Phase 2A JSON file or chunk record is invalid."""


class DatabaseUnavailable(KnowledgeEmbeddingError):
    """Raised when PostgreSQL cannot complete a vector-store operation."""


class VectorExtensionUnavailable(DatabaseUnavailable):
    """Raised when PostgreSQL does not have the pgvector extension installed."""


class IncompatibleEmbeddingDimension(KnowledgeEmbeddingError):
    """Raised when one provider/model identity is used with multiple dimensions."""


class EmptyQueryError(KnowledgeEmbeddingError):
    """Raised when semantic retrieval receives an empty query."""


class SourceChunkUnavailable(KnowledgeEmbeddingError):
    """Raised when a stored vector cannot resolve its original Phase 2A chunk."""


class StaleEmbeddingError(SourceChunkUnavailable):
    """Raised when the referenced Phase 2A chunk changed after embedding."""
