from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.config import Settings, settings
from app.embeddings.base import EmbeddingInputType
from app.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingModelUnavailable,
    InvalidEmbeddingError,
)
from app.embeddings.validation import validate_vector


class FastEmbedProvider:
    """Local ONNX sentence embeddings with explicit offline-by-default loading."""

    provider_name = "fastembed"

    def __init__(
        self,
        config: Settings = settings,
        *,
        model_factory: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if config.embedding_provider != self.provider_name:
            raise EmbeddingConfigurationError(
                f"Configured provider is '{config.embedding_provider}', not '{self.provider_name}'"
            )
        self.model_name = config.embedding_model
        self.dimension = config.embedding_dimension
        self.batch_size = config.embedding_batch_size
        if model is not None:
            self._model = model
            return

        if model_factory is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise EmbeddingModelUnavailable(
                    "FastEmbed is not installed. Install backend/requirements.txt to enable local embeddings."
                ) from exc
            model_factory = TextEmbedding

        self._validate_supported_model(model_factory)
        cache_dir = Path(config.embedding_cache_dir).expanduser().resolve()
        try:
            self._model = model_factory(
                model_name=self.model_name,
                cache_dir=str(cache_dir),
                local_files_only=config.embedding_local_files_only,
            )
        except Exception as exc:
            if config.embedding_local_files_only:
                guidance = (
                    "Model weights are unavailable in the configured cache. Set "
                    "EMBEDDING_LOCAL_FILES_ONLY=false for an explicit first-time download, "
                    "then restore it to true for offline operation."
                )
            else:
                guidance = "Check model access, network connectivity, and EMBEDDING_CACHE_DIR."
            raise EmbeddingModelUnavailable(
                f"Could not load FastEmbed model '{self.model_name}'. {guidance}"
            ) from exc

    def _validate_supported_model(self, model_factory: Any) -> None:
        try:
            supported = model_factory.list_supported_models()
        except Exception as exc:
            raise EmbeddingModelUnavailable(
                "FastEmbed could not enumerate its supported local models. Verify the installation."
            ) from exc
        model_info = next((item for item in supported if item.get("model") == self.model_name), None)
        if model_info is None:
            raise EmbeddingConfigurationError(
                f"FastEmbed does not list model '{self.model_name}' as supported. "
                "Choose a model reported by FastEmbed and set EMBEDDING_DIMENSION explicitly."
            )
        actual_dimension = model_info.get("dim")
        if actual_dimension != self.dimension:
            raise EmbeddingConfigurationError(
                f"Configured EMBEDDING_DIMENSION={self.dimension} does not match "
                f"FastEmbed model '{self.model_name}' dimension {actual_dimension}."
            )

    @staticmethod
    def _prepare(text: str, input_type: EmbeddingInputType) -> str:
        if not text.strip():
            raise InvalidEmbeddingError("Cannot embed empty text")
        prefix = "query: " if input_type == "query" else "passage: "
        return prefix + text.strip()

    def embed_text(
        self,
        text: str,
        *,
        input_type: EmbeddingInputType = "document",
    ) -> list[float]:
        return self.embed_batch([text], input_type=input_type)[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType = "document",
    ) -> list[list[float]]:
        if not texts:
            return []
        prepared = [self._prepare(text, input_type) for text in texts]
        try:
            vectors = list(self._model.embed(prepared, batch_size=self.batch_size))
        except Exception as exc:
            raise EmbeddingModelUnavailable(
                f"FastEmbed failed while generating vectors with '{self.model_name}': {exc}"
            ) from exc
        if len(vectors) != len(prepared):
            raise InvalidEmbeddingError(
                f"FastEmbed returned {len(vectors)} vectors for {len(prepared)} input texts"
            )
        return [validate_vector(vector, self.dimension) for vector in vectors]
