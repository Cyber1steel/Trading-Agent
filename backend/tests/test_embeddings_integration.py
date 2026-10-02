"""Real PostgreSQL/pgvector checks; set PGVECTOR_TEST_DATABASE_URL to enable."""

import os
import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete, update
from sqlalchemy.orm import sessionmaker

from app.knowledge.embedding.contracts import ChunkForEmbedding, RetrievalFilters
from app.knowledge.embedding.errors import (
    IncompatibleEmbeddingDimension,
    SourceChunkUnavailable,
    StaleEmbeddingError,
)
from app.knowledge.embedding.lexical import LexicalRetrievalService
from app.knowledge.embedding.repository import EmbeddingRepository
from app.knowledge.embedding.retrieval import SemanticRetrievalService
from app.knowledge.models import DocumentChunk, ProcessedDocument, SourceMetadata
from app.models.embedding import EmbeddingSpace, KnowledgeEmbedding


DATABASE_URL = os.getenv("PGVECTOR_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="requires a migrated pgvector test database")


def test_pgvector_persistence_idempotency_similarity_and_filters(tmp_path: Path):
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(engine, expire_on_commit=False)
    repository = EmbeddingRepository(factory)
    model = f"integration-{uuid4()}"
    source = SourceMetadata(source_id=f"source-{uuid4()}", source_type="txt", title="Test", file_path="test.txt")
    path = tmp_path / "phase2a.json"
    records = []
    for index, (text, vector) in enumerate((("closest", [1.0, 0.0]), ("farther", [0.0, 1.0]))):
        chunk = DocumentChunk(chunk_id=f"chunk-{uuid4()}", source_id=source.source_id, source_type="txt",
            title=source.title, file_path=source.file_path, chunk_index=index, char_start=0,
            char_end=len(text), text=text)
        import hashlib
        records.append(ChunkForEmbedding(chunk=chunk, source=source, processed_file_path=path,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(), vector=vector))
    path.write_text(ProcessedDocument(
        source=source, chunk_size=100, overlap=10, extracted_characters=14,
        cleaned_characters=14, chunks=[record.chunk for record in records],
    ).model_dump_json(), encoding="utf-8")
    try:
        assert repository.upsert_batch(records, "integration", model, 2) == (2, 0, 0)
        assert repository.upsert_batch(records, "integration", model, 2) == (0, 0, 2)
        with factory() as session, session.begin():
            session.execute(
                update(KnowledgeEmbedding)
                .where(KnowledgeEmbedding.chunk_id == records[0].chunk.chunk_id)
                .values(search_vector=None)
            )
        assert repository.refresh_lexical_vectors([records[0]]) >= 1

        class QueryProvider:
            provider_name, model_name, dimension = "integration", model, 2

        hits = repository.search([1.0, 0.0], QueryProvider(), 1, RetrievalFilters(source_id=source.source_id))
        assert len(hits) == 1
        assert hits[0].chunk_id == records[0].chunk.chunk_id
        assert hits[0].distance == pytest.approx(0.0, abs=1e-6)

        lexical_hits = repository.search_lexical(
            "closest", 1, RetrievalFilters(source_id=source.source_id)
        )
        assert len(lexical_hits) == 1
        assert lexical_hits[0].chunk_id == records[0].chunk.chunk_id
        assert lexical_hits[0].lexical_score > 0
        lexical_result = LexicalRetrievalService(repository).search(
            "closest", 1, RetrievalFilters(source_id=source.source_id)
        )
        assert lexical_result[0].text == "closest"
        assert lexical_result[0].source_id == source.source_id

        class QueryProvider:
            provider_name, model_name, dimension = "integration", model, 2
            def embed_text(self, text, *, input_type="query"):
                return [1.0, 0.0]

        retrieved = SemanticRetrievalService(QueryProvider(), repository).search(
            "evidence", top_k=1, filters=RetrievalFilters(source_id=source.source_id)
        )
        assert retrieved[0].text == "closest"
        assert retrieved[0].source_id == source.source_id
        assert retrieved[0].source_url is None
        assert retrieved[0].chunk_index == 0
        assert retrieved[0].embedding_model == model

        changed_doc = ProcessedDocument(
            source=source, chunk_size=100, overlap=10, extracted_characters=13,
            cleaned_characters=13,
            chunks=[records[0].chunk.model_copy(update={"text": "content changed", "char_end": 15}), records[1].chunk],
        )
        path.write_text(changed_doc.model_dump_json(), encoding="utf-8")
        with pytest.raises(StaleEmbeddingError):
            SemanticRetrievalService(QueryProvider(), repository).search(
                "evidence", top_k=1, filters=RetrievalFilters(source_id=source.source_id)
            )
        path.write_text(ProcessedDocument(
            source=source, chunk_size=100, overlap=10, extracted_characters=7,
            cleaned_characters=7, chunks=[records[1].chunk],
        ).model_dump_json(), encoding="utf-8")
        with pytest.raises(SourceChunkUnavailable):
            SemanticRetrievalService(QueryProvider(), repository).search(
                "evidence", top_k=1, filters=RetrievalFilters(source_id=source.source_id)
            )
        changed_text = "content changed"
        changed_record = records[0].model_copy(update={
            "chunk": records[0].chunk.model_copy(update={"text": changed_text, "char_end": len(changed_text)}),
            "content_sha256": hashlib.sha256(changed_text.encode()).hexdigest(),
            "vector": [0.8, 0.2],
        })
        assert repository.upsert_batch([changed_record], "integration", model, 2) == (0, 1, 0)
        assert repository.existing_content_hashes([records[0].chunk.chunk_id], "integration", model, 2)[
            records[0].chunk.chunk_id
        ] == changed_record.content_sha256
        with pytest.raises(IncompatibleEmbeddingDimension):
            repository.existing_content_hashes([records[0].chunk.chunk_id], "integration", model, 3)
    finally:
        with factory() as session, session.begin():
            session.execute(delete(KnowledgeEmbedding).where(KnowledgeEmbedding.embedding_model == model))
            session.execute(delete(EmbeddingSpace).where(EmbeddingSpace.model == model))
        engine.dispose()
