"""Provider-independent text embedding interfaces and implementations."""

from app.embeddings.base import EmbeddingInputType, EmbeddingProvider
from app.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingModelUnavailable,
)
from app.embeddings.fastembed_provider import FastEmbedProvider

__all__ = [
    "EmbeddingConfigurationError",
    "EmbeddingError",
    "EmbeddingInputType",
    "EmbeddingModelUnavailable",
    "EmbeddingProvider",
    "FastEmbedProvider",
]
