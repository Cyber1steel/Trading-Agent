"""Explicit, repeatable ingestion of Phase 2A processed documents."""

import hashlib
import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from app.core.config import get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.validation import validate_vector
from app.knowledge.embedding.contracts import ChunkForEmbedding, EmbeddingIngestionReport
from app.knowledge.embedding.errors import ProcessedChunkError
from app.knowledge.embedding.repository import EmbeddingRepository
from app.knowledge.models import ProcessedDocument


def load_processed_document(path: Path) -> ProcessedDocument:
    try:
        return ProcessedDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ProcessedChunkError(f"Cannot load Phase 2A processed document {path}: {exc}") from exc


def records_from_document(path: Path) -> list[ChunkForEmbedding]:
    doc = load_processed_document(path)
    seen: set[str] = set()
    records: list[ChunkForEmbedding] = []
    for chunk in doc.chunks:
        if chunk.chunk_id in seen:
            raise ProcessedChunkError(f"Duplicate chunk_id {chunk.chunk_id!r} in {path}.")
        seen.add(chunk.chunk_id)
        if chunk.source_id != doc.source.source_id or chunk.source_type != doc.source.source_type:
            raise ProcessedChunkError(f"Chunk {chunk.chunk_id!r} conflicts with document source metadata.")
        if chunk.char_end < chunk.char_start:
            raise ProcessedChunkError(f"Chunk {chunk.chunk_id!r} has invalid character offsets.")
        records.append(ChunkForEmbedding(
            chunk=chunk,
            source=doc.source,
            processed_file_path=path.resolve(),
            content_sha256=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            vector=[],
        ))
    if not records:
        raise ProcessedChunkError(f"Processed document {path} has no chunks.")
    return records


class EmbeddingIngestionService:
    def __init__(self, provider: EmbeddingProvider, repository: EmbeddingRepository, batch_size: int | None = None):
        self.provider = provider
        self.repository = repository
        self.batch_size = batch_size or get_settings().embedding_batch_size
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def ingest_files(self, paths: Iterable[Path]) -> EmbeddingIngestionReport:
        created = updated = skipped = documents = seen = 0
        file_paths = [Path(path) for path in paths]
        for path in file_paths:
            records = records_from_document(Path(path))
            documents += 1
            seen += len(records)
            known = self.repository.existing_content_hashes(
                [r.chunk.chunk_id for r in records], self.provider.provider_name,
                self.provider.model_name, self.provider.dimension,
            )
            changed: list[ChunkForEmbedding] = []
            unchanged: list[ChunkForEmbedding] = []
            for record in records:
                if known.get(record.chunk.chunk_id) == record.content_sha256:
                    unchanged.append(record)
                else:
                    changed.append(record)
            self.repository.refresh_provenance(unchanged, self.provider.provider_name,
                self.provider.model_name, self.provider.dimension)
            skipped += len(unchanged)
            for offset in range(0, len(changed), self.batch_size):
                batch = changed[offset:offset + self.batch_size]
                vectors = self.provider.embed_batch([r.chunk.text for r in batch], input_type="document")
                if len(vectors) != len(batch):
                    raise ProcessedChunkError("Embedding provider returned a different number of vectors than inputs.")
                embedded = [record.model_copy(update={"vector": validate_vector(v, self.provider.dimension)})
                            for record, v in zip(batch, vectors, strict=True)]
                new_count, update_count, skip_count = self.repository.upsert_batch(
                    embedded, self.provider.provider_name, self.provider.model_name, self.provider.dimension)
                created += new_count
                updated += update_count
                skipped += skip_count
        return EmbeddingIngestionReport(
            processed_file_paths=[str(path.resolve()) for path in file_paths],
            provider=self.provider.provider_name, model=self.provider.model_name,
            dimension=self.provider.dimension, chunks_seen=seen,
            embeddings_created=created, embeddings_updated=updated, embeddings_skipped=skipped,
        )
