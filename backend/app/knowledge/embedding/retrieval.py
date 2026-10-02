"""Semantic search with Phase 2A JSON files as the canonical chunk source."""

from pathlib import Path

from app.embeddings.base import EmbeddingProvider
from app.embeddings.validation import validate_vector
from app.knowledge.embedding.contracts import RetrievalFilters, RetrievalResult
from app.knowledge.embedding.errors import EmptyQueryError, SourceChunkUnavailable, StaleEmbeddingError
from app.knowledge.embedding.ingestion import load_processed_document
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
        cache = {}
        results = []
        for hit in hits:
            path = Path(hit.processed_file_path)
            try:
                if str(path) not in cache:
                    cache[str(path)] = load_processed_document(path)
                doc = cache[str(path)]
            except Exception as exc:
                raise SourceChunkUnavailable(f"Cannot resolve stored chunk {hit.chunk_id} from {path}.") from exc
            if doc.source.source_id != hit.source_id:
                raise StaleEmbeddingError(f"Stored provenance for {hit.chunk_id} no longer matches its Phase 2A document.")
            chunk = next((c for c in doc.chunks if c.chunk_id == hit.chunk_id), None)
            if chunk is None:
                raise SourceChunkUnavailable(f"Chunk {hit.chunk_id} is absent from {path}.")
            if chunk.source_id != hit.source_id or chunk.source_type != hit.source_type:
                raise StaleEmbeddingError(f"Chunk metadata for {hit.chunk_id} changed after embedding.")
            import hashlib
            digest = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            if digest != hit.content_sha256:
                raise StaleEmbeddingError(f"Chunk {hit.chunk_id} changed after embedding; rerun embedding ingestion.")
            results.append(RetrievalResult(
                chunk_id=hit.chunk_id, text=chunk.text, distance=hit.distance,
                source_id=hit.source_id, document_id=hit.source_id,
                source_type=hit.source_type, title=hit.title,
                author=hit.author, source_url=hit.source_url, file_path=hit.source_file_path,
                processed_file_path=hit.processed_file_path, page_number=hit.page_number,
                chunk_index=hit.chunk_index, char_start=hit.char_start, char_end=hit.char_end,
                embedding_provider=self.provider.provider_name, embedding_model=self.provider.model_name,
                embedding_dimension=self.provider.dimension,
                ingestion_chunk_size=doc.chunk_size, ingestion_overlap=doc.overlap,
            ))
        return results
