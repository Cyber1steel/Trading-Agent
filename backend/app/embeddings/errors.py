class EmbeddingError(RuntimeError):
    """Base error for embedding provider failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when provider configuration is inconsistent."""


class EmbeddingModelUnavailable(EmbeddingError):
    """Raised when the configured local model or runtime cannot be loaded."""


class InvalidEmbeddingError(EmbeddingError):
    """Raised when a provider returns a malformed embedding vector."""
