from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

EmbeddingInputType = Literal["query", "document"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Small interface required by ingestion and semantic retrieval."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_text(
        self,
        text: str,
        *,
        input_type: EmbeddingInputType = "document",
    ) -> list[float]: ...

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType = "document",
    ) -> list[list[float]]: ...
