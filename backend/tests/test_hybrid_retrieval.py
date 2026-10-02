import hashlib
from pathlib import Path

import pytest
from sqlalchemy.dialects.postgresql import dialect as postgresql_dialect

from app.knowledge.embedding.contracts import RetrievalFilters, RetrievalResult, StoredLexicalHit
from app.knowledge.embedding.errors import EmptyQueryError
from app.knowledge.embedding.hybrid import HybridRetrievalService
from app.knowledge.embedding.lexical import LexicalIndexingService, LexicalRetrievalService
from app.knowledge.embedding.repository import EmbeddingRepository
from app.knowledge.models import DocumentChunk, ProcessedDocument, SourceMetadata


def write_document(path: Path, texts: list[tuple[str, str]]) -> tuple[SourceMetadata, list[DocumentChunk]]:
    source = SourceMetadata(source_id="source-eval", source_type="txt", title="Trading note", file_path="note.txt")
    chunks = [
        DocumentChunk(
            chunk_id=chunk_id,
            source_id=source.source_id,
            source_type=source.source_type,
            title=source.title,
            file_path=source.file_path,
            chunk_index=index,
            char_start=0,
            char_end=len(text),
            text=text,
        )
        for index, (chunk_id, text) in enumerate(texts)
    ]
    path.write_text(ProcessedDocument(
        source=source, chunk_size=100, overlap=10, extracted_characters=100,
        cleaned_characters=100, chunks=chunks,
    ).model_dump_json(), encoding="utf-8")
    return source, chunks


def lexical_hit(chunk, path: Path, score: float) -> StoredLexicalHit:
    return StoredLexicalHit(
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        source_type=chunk.source_type,
        title=chunk.title,
        author=None,
        source_url=None,
        source_file_path=chunk.file_path,
        processed_file_path=str(path),
        published_at=None,
        page_number=None,
        chunk_index=chunk.chunk_index,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        content_sha256=hashlib.sha256(chunk.text.encode()).hexdigest(),
        lexical_score=score,
    )


class FakeLexicalRepository:
    def __init__(self, hits):
        self.hits = hits
        self.arguments = None

    def search_lexical(self, query, top_k, filters):
        self.arguments = query, top_k, filters
        return self.hits[:top_k]


class FakeLexicalIndexRepository:
    def __init__(self):
        self.batches = []

    def refresh_lexical_vectors(self, records):
        self.batches.append(list(records))
        return len(records)


def result(chunk_id: str, distance: float | None = None, lexical_score: float | None = None,
           mode: str = "semantic") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_id="source-eval",
        document_id="source-eval",
        source_type="txt",
        title="Trading note",
        author=None,
        source_url=None,
        file_path="note.txt",
        page_number=None,
        chunk_index=0,
        char_start=0,
        char_end=10,
        text=f"text for {chunk_id}",
        distance=distance,
        embedding_provider="fake" if distance is not None else None,
        embedding_model="fake-model" if distance is not None else None,
        embedding_dimension=2 if distance is not None else None,
        processed_file_path="note.json",
        ingestion_chunk_size=100,
        ingestion_overlap=10,
        retrieval_mode=mode,
        lexical_score=lexical_score,
    )


class FakeSemantic:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k, filters):
        self.calls.append((query, top_k, filters))
        return self.results[:top_k]


class FakeLexical:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k, filters):
        self.calls.append((query, top_k, filters))
        return self.results[:top_k]


def test_lexical_retrieval_uses_structured_provenance_and_filters(tmp_path):
    path = tmp_path / "phase2a.json"
    _, chunks = write_document(path, [("s:0", "price paused after a quick move") , ("s:1", "volume became quieter")])
    filters = RetrievalFilters(source_id="source-eval", source_type="txt", page_number=1)
    repository = FakeLexicalRepository([lexical_hit(chunks[1], path, 0.73), lexical_hit(chunks[0], path, 0.2)])
    service = LexicalRetrievalService(repository)

    results = service.search("quick move", top_k=1, filters=filters)

    assert len(results) == 1 and results[0].chunk_id == "s:1"
    assert results[0].text == "volume became quieter"
    assert results[0].retrieval_mode == "lexical"
    assert results[0].lexical_score == pytest.approx(0.73)
    assert results[0].lexical_rank == 1 and results[0].distance is None
    assert results[0].source_id == "source-eval" and results[0].file_path == "note.txt"
    assert repository.arguments == ("quick move", 1, filters)


def test_lexical_empty_query_and_no_matches(tmp_path):
    repository = FakeLexicalRepository([])
    service = LexicalRetrievalService(repository)
    with pytest.raises(EmptyQueryError):
        service.search("   ")
    assert service.search("no results") == []


def test_lexical_repository_compiles_postgres_full_text_query_with_filters():
    class ResultRows:
        def all(self):
            return []

    class CompileSession:
        sql = ""
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def execute(self, statement):
            self.sql = str(statement.compile(dialect=postgresql_dialect()))
            return ResultRows()

    session = CompileSession()
    repository = EmbeddingRepository(lambda: session)
    filters = RetrievalFilters(source_id="source-eval", source_type="txt", page_number=2)

    assert repository.search_lexical('"volume quiet" OR liquidity', 7, filters) == []

    sql = session.sql.lower()
    assert "websearch_to_tsquery" in sql
    assert "ts_rank_cd" in sql
    assert " @@ " in sql
    assert "row_number() over (partition by knowledge_embeddings.chunk_id" in sql
    assert "source_id =" in sql and "source_type =" in sql and "page_number =" in sql
    assert "order by anon_1.lexical_score desc, knowledge_embeddings.chunk_id asc" in sql
    assert "limit" in sql


def test_lexical_indexing_batches_phase2a_json_without_embedding_provider(tmp_path):
    path = tmp_path / "processed.json"
    write_document(path, [("source:0", "volume became quieter"), ("source:1", "price paused")])
    repository = FakeLexicalIndexRepository()

    report = LexicalIndexingService(repository, batch_size=1).index_files([path])

    assert report.chunks_seen == 2 and report.database_rows_updated == 2
    assert len(repository.batches) == 2
    assert repository.batches[0][0].chunk.text == "volume became quieter"


def test_hybrid_rrf_dedupes_deterministically_and_preserves_scores():
    semantic = FakeSemantic([result("b", 0.1), result("a", 0.2), result("d", 0.3)])
    lexical = FakeLexical([
        result("a", lexical_score=0.8, mode="lexical"),
        result("b", lexical_score=0.7, mode="lexical"),
        result("c", lexical_score=0.6, mode="lexical"),
    ])
    service = HybridRetrievalService(semantic, lexical, semantic_candidates=4, lexical_candidates=4, rrf_k=60)
    filters = RetrievalFilters(document_id="source-eval")

    output = service.search("query", top_k=3, filters=filters)

    assert [item.chunk_id for item in output] == ["a", "b", "c"]
    assert output[0].hybrid_score == pytest.approx(1 / 62 + 1 / 61)
    assert output[0].semantic_rank == 2 and output[0].lexical_rank == 1
    assert output[0].distance == 0.2 and output[0].lexical_score == 0.8
    assert output[0].retrieval_mode == "hybrid"
    repeated = service.search("query", top_k=3, filters=filters)
    assert [item.chunk_id for item in repeated] == [item.chunk_id for item in output]
    assert semantic.calls == [("query", 4, filters)] * 2
    assert lexical.calls == [("query", 4, filters)] * 2


@pytest.mark.parametrize(
    ("semantic_results", "lexical_results", "expected_ids"),
    [
        ([result("sem")], [], ["sem"]),
        ([], [result("lex", lexical_score=0.5, mode="lexical")], ["lex"]),
        ([], [], []),
        ([result("x"), result("x")], [result("x", lexical_score=1, mode="lexical")], ["x"]),
    ],
)
def test_hybrid_one_sided_empty_and_duplicate_candidates(semantic_results, lexical_results, expected_ids):
    service = HybridRetrievalService(
        FakeSemantic(semantic_results), FakeLexical(lexical_results), rrf_k=1
    )
    assert [item.chunk_id for item in service.search("query", top_k=5)] == expected_ids


def test_hybrid_deduplicates_by_chunk_id_not_text():
    left = result("left")
    right = result("right").model_copy(update={"text": left.text})
    service = HybridRetrievalService(FakeSemantic([left, right]), FakeLexical([]), rrf_k=10)
    assert [item.chunk_id for item in service.search("query", top_k=5)] == ["left", "right"]


def test_hybrid_keeps_semantic_only_score_fields_empty_when_lexical_has_other_candidates():
    semantic = FakeSemantic([result("semantic-only")])
    lexical = FakeLexical([result("lexical-only", lexical_score=0.75, mode="lexical")])
    results = HybridRetrievalService(semantic, lexical, rrf_k=5).search("query", top_k=2)
    by_id = {item.chunk_id: item for item in results}
    assert by_id["semantic-only"].lexical_score is None
    assert by_id["lexical-only"].distance is None


def test_hybrid_rejects_invalid_input_and_candidate_counts():
    service = HybridRetrievalService(FakeSemantic([]), FakeLexical([]), rrf_k=1)
    with pytest.raises(EmptyQueryError):
        service.search(" ")
    with pytest.raises(ValueError, match="top_k"):
        service.search("query", top_k=0)
    with pytest.raises(ValueError, match="Candidate limits"):
        service.search("query", semantic_candidates=0)
