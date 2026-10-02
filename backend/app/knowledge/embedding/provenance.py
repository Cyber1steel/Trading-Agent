"""Resolve database candidates against canonical Phase 2A chunk JSON."""

import hashlib
from pathlib import Path
from typing import Literal

from app.embeddings.base import EmbeddingProvider
from app.knowledge.embedding.contracts import RetrievalResult
from app.knowledge.embedding.errors import SourceChunkUnavailable, StaleEmbeddingError
from app.knowledge.embedding.ingestion import load_processed_document


def resolve_stored_hits(
    hits: list[object],
    *,
    retrieval_mode: Literal["semantic", "lexical"],
    provider: EmbeddingProvider | None = None,
) -> list[RetrievalResult]:
    """Resolve a bounded candidate list, checking source and content identity once per file."""
    cache = {}
    results = []
    for rank, hit in enumerate(hits, start=1):
        path = Path(hit.processed_file_path)
        try:
            if str(path) not in cache:
                document = load_processed_document(path)
                cache[str(path)] = (
                    document,
                    {chunk.chunk_id: chunk for chunk in document.chunks},
                )
            document, chunks_by_id = cache[str(path)]
        except Exception as exc:
            raise SourceChunkUnavailable(
                f"Cannot resolve stored chunk {hit.chunk_id} from {path}."
            ) from exc
        if document.source.source_id != hit.source_id:
            raise StaleEmbeddingError(
                f"Stored provenance for {hit.chunk_id} no longer matches its Phase 2A document."
            )
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            raise SourceChunkUnavailable(f"Chunk {hit.chunk_id} is absent from {path}.")
        if chunk.source_id != hit.source_id or chunk.source_type != hit.source_type:
            raise StaleEmbeddingError(f"Chunk metadata for {hit.chunk_id} changed after indexing.")
        if (
            chunk.title != document.source.title
            or chunk.file_path != document.source.file_path
            or chunk.source_url != document.source.source_url
            or hit.title != document.source.title
            or hit.author != document.source.author
            or hit.source_url != document.source.source_url
            or hit.source_file_path != document.source.file_path
            or hit.page_number != chunk.page_number
            or hit.chunk_index != chunk.chunk_index
            or hit.char_start != chunk.char_start
            or hit.char_end != chunk.char_end
        ):
            raise StaleEmbeddingError(
                f"Chunk provenance for {hit.chunk_id} conflicts with its Phase 2A source record."
            )
        digest = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        if digest != hit.content_sha256:
            raise StaleEmbeddingError(
                f"Chunk {hit.chunk_id} changed after indexing; rerun knowledge ingestion."
            )

        is_semantic = retrieval_mode == "semantic"
        results.append(RetrievalResult(
            chunk_id=hit.chunk_id,
            source_id=hit.source_id,
            document_id=hit.source_id,
            source_type=hit.source_type,
            title=document.source.title,
            author=document.source.author,
            source_url=document.source.source_url,
            file_path=chunk.file_path,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            text=chunk.text,
            distance=getattr(hit, "distance", None),
            embedding_provider=provider.provider_name if provider else None,
            embedding_model=provider.model_name if provider else None,
            embedding_dimension=provider.dimension if provider else None,
            processed_file_path=hit.processed_file_path,
            ingestion_chunk_size=document.chunk_size,
            ingestion_overlap=document.overlap,
            retrieval_mode=retrieval_mode,
            lexical_score=getattr(hit, "lexical_score", None),
            semantic_rank=rank if is_semantic else None,
            lexical_rank=rank if not is_semantic else None,
        ))
    return results
