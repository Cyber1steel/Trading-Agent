import math
from collections.abc import Iterable

from app.embeddings.errors import InvalidEmbeddingError


def validate_vector(vector: Iterable[float], dimension: int) -> list[float]:
    """Convert a provider result to finite floats and check its configured size."""
    try:
        result = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise InvalidEmbeddingError("Embedding provider returned non-numeric vector values") from exc
    if len(result) != dimension:
        raise InvalidEmbeddingError(
            f"Embedding dimension mismatch: expected {dimension}, received {len(result)}"
        )
    if not result or not all(math.isfinite(value) for value in result):
        raise InvalidEmbeddingError("Embedding vector must contain finite numeric values")
    if not any(value != 0.0 for value in result):
        raise InvalidEmbeddingError("Embedding vector cannot be all zeroes for cosine retrieval")
    return result
