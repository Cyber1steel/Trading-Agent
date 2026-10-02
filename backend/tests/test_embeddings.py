import hashlib
from pathlib import Path

import pytest

from app.core.config import Settings
from app.embeddings.errors import EmbeddingConfigurationError, EmbeddingModelUnavailable, InvalidEmbeddingError
from app.embeddings.fastembed_provider import FastEmbedProvider
from app.knowledge.embedding.contracts import RetrievalFilters, StoredEmbeddingHit
from app.knowledge.embedding.errors import EmptyQueryError, ProcessedChunkError
from app.knowledge.embedding.ingestion import EmbeddingIngestionService, records_from_document
from app.knowledge.embedding.retrieval import SemanticRetrievalService
from app.knowledge.models import DocumentChunk, ProcessedDocument, SourceMetadata


class FakeModel:
    def __init__(self):
        self.calls = []

    def embed(self, texts, batch_size):
        self.calls.append((list(texts), batch_size))
        return [[float(len(text)), 1.0] for text in texts]


class FakeProvider:
    provider_name = "fake"
    model_name = "test-2"
    dimension = 2

    def __init__(self): self.calls = []
    def embed_text(self, text, *, input_type="document"):
        self.calls.append((text, input_type))
        return [1.0, 1.0]
    def embed_batch(self, texts, *, input_type="document"):
        self.calls.append((list(texts), input_type))
        return [[float(len(text)), 1.0] for text in texts]


class FakeRepository:
    def __init__(self): self.rows = {}; self.search_args = None
    def existing_content_hashes(self, ids, provider, model, dimension):
        return {key: value["hash"] for key, value in self.rows.items() if key in ids}
    def refresh_provenance(self, *args): pass
    def upsert_batch(self, records, provider, model, dimension):
        created = updated = skipped = 0
        for record in records:
            prior = self.rows.get(record.chunk.chunk_id)
            if prior and prior["hash"] == record.content_sha256: skipped += 1
            else:
                created += prior is None
                updated += prior is not None
                self.rows[record.chunk.chunk_id] = {"hash": record.content_sha256, "record": record}
        return created, updated, skipped
    def search(self, vector, provider, top_k, filters):
        self.search_args = vector, top_k, filters
        return self.hits[:top_k]


def document(path: Path, text="Evidence chunk"):
    source = SourceMetadata(source_id="source-1", source_type="txt", title="A title", file_path="notes.txt")
    chunk = DocumentChunk(chunk_id="chunk-1", source_id=source.source_id, source_type="txt", title=source.title,
        file_path=source.file_path, chunk_index=0, char_start=0, char_end=len(text), text=text)
    return ProcessedDocument(source=source, chunk_size=100, overlap=10, extracted_characters=len(text),
        cleaned_characters=len(text), chunks=[chunk])


def test_fastembed_prefix_batch_dimension_and_unavailable_model(tmp_path):
    class Factory:
        @staticmethod
        def list_supported_models(): return [{"model": "tiny", "dim": 2}]
        def __new__(cls, **kwargs):
            assert kwargs["local_files_only"] is True
            return FakeModel()
    config = Settings(embedding_model="tiny", embedding_dimension=2, embedding_cache_dir=tmp_path)
    provider = FastEmbedProvider(config, model_factory=Factory)
    assert provider.embed_text("question", input_type="query") == [15.0, 1.0]
    assert provider.embed_batch(["one", "two"]) == [[12.0, 1.0], [12.0, 1.0]]
    class Missing(Factory):
        def __new__(cls, **kwargs): raise OSError("not cached")
    with pytest.raises(EmbeddingModelUnavailable, match="EMBEDDING_LOCAL_FILES_ONLY=false"):
        FastEmbedProvider(config, model_factory=Missing)
    with pytest.raises(InvalidEmbeddingError): provider.embed_batch([""])
    with pytest.raises(EmbeddingConfigurationError):
        FastEmbedProvider(Settings(embedding_model="tiny", embedding_dimension=3), model_factory=Factory)
    with pytest.raises(EmbeddingConfigurationError, match="does not list model"):
        FastEmbedProvider(Settings(embedding_model="unlisted", embedding_dimension=2), model_factory=Factory)


def test_ingestion_batches_idempotently_and_rejects_bad_json(tmp_path):
    processed = tmp_path / "source.json"
    processed.write_text(document(processed).model_dump_json(), encoding="utf-8")
    provider, repo = FakeProvider(), FakeRepository()
    service = EmbeddingIngestionService(provider, repo, batch_size=1)
    first = service.ingest_files([processed])
    second = service.ingest_files([processed])
    assert first.embeddings_created == 1 and second.embeddings_skipped == 1
    assert len(provider.calls) == 1
    broken = tmp_path / "bad.json"
    broken.write_text("{}", encoding="utf-8")
    with pytest.raises(ProcessedChunkError): records_from_document(broken)


def test_retrieval_empty_query_top_k_filters_and_provenance(tmp_path):
    processed = tmp_path / "source.json"
    doc = document(processed)
    processed.write_text(doc.model_dump_json(), encoding="utf-8")
    chunk = doc.chunks[0]
    repo = FakeRepository()
    repo.hits = [StoredEmbeddingHit(chunk_id=chunk.chunk_id, source_id=chunk.source_id,
        source_type=chunk.source_type, title=chunk.title, author=None, source_url=None,
        source_file_path=chunk.file_path, processed_file_path=str(processed), published_at=None,
        page_number=None, chunk_index=0, char_start=0, char_end=len(chunk.text),
        content_sha256=hashlib.sha256(chunk.text.encode()).hexdigest(), distance=0.12)]
    provider = FakeProvider()
    service = SemanticRetrievalService(provider, repo)
    with pytest.raises(EmptyQueryError): service.search("  ")
    filters = RetrievalFilters(document_id="source-1", source_type="txt")
    result = service.search("market setup", top_k=1, filters=filters)[0]
    assert result.text == chunk.text and result.document_id == "source-1"
    assert result.distance == 0.12 and result.page_number is None
    assert result.retrieval_mode == "semantic" and result.semantic_rank == 1
    assert repo.search_args[1:] == (1, filters)
    assert provider.calls[-1] == ("market setup", "query")


def test_stale_chunk_is_not_returned(tmp_path):
    processed = tmp_path / "source.json"
    doc = document(processed)
    processed.write_text(doc.model_dump_json(), encoding="utf-8")
    from app.knowledge.embedding.errors import StaleEmbeddingError
    repo = FakeRepository()
    chunk = doc.chunks[0]
    repo.hits = [StoredEmbeddingHit(chunk_id=chunk.chunk_id, source_id=chunk.source_id,
        source_type=chunk.source_type, title=chunk.title, author=None, source_url=None,
        source_file_path=chunk.file_path, processed_file_path=str(processed), published_at=None,
        page_number=None, chunk_index=0, char_start=0, char_end=10, content_sha256="0" * 64, distance=0.2)]
    with pytest.raises(StaleEmbeddingError): SemanticRetrievalService(FakeProvider(), repo).search("query")
