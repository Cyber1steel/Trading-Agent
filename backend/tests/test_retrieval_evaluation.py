import pytest
from pathlib import Path

from app.knowledge.embedding.contracts import RetrievalResult
from app.knowledge.embedding.evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluationDataset,
    evaluate_rankings,
    load_evaluation_dataset,
)
from app.knowledge.pipeline import process_document


def result(chunk_id: str, source_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_id=source_id,
        document_id=source_id,
        source_type="txt",
        title="Test note",
        author=None,
        source_url=None,
        file_path="note.txt",
        page_number=None,
        chunk_index=0,
        char_start=0,
        char_end=4,
        text="text",
        distance=None,
        processed_file_path="note.json",
        ingestion_chunk_size=100,
        ingestion_overlap=10,
        retrieval_mode="lexical",
        lexical_score=0.5,
    )


def dataset() -> RetrievalEvaluationDataset:
    return RetrievalEvaluationDataset(
        dataset_id="unit-test-v1",
        version=1,
        cases=[
            RetrievalEvaluationCase(
                case_id="found-at-two",
                query="meaningful query",
                expected_chunk_ids=["relevant"],
            ),
            RetrievalEvaluationCase(
                case_id="not-found",
                query="missing evidence",
                expected_source_ids=["expected-source"],
            ),
        ],
    )


def test_evaluation_metrics_and_case_failure_analysis():
    report = evaluate_rankings(
        dataset(),
        {"found-at-two": [result("noise", "other"), result("relevant", "source")]},
        retrieval_mode="lexical",
        ks=[1, 3],
    )

    assert report.retrieval_mode == "lexical" and report.ks == [1, 3]
    found = report.cases[0]
    assert found.relevant_ranks == [2]
    assert found.matched_expected_chunk_ids == ["relevant"]
    assert found.metrics_at_k["1"] == {
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "hit_rate_at_k": 0.0,
        "mrr_at_k": 0.0,
    }
    assert found.metrics_at_k["3"]["recall_at_k"] == 1.0
    assert found.metrics_at_k["3"]["precision_at_k"] == pytest.approx(1 / 3)
    assert found.metrics_at_k["3"]["hit_rate_at_k"] == 1.0
    assert found.metrics_at_k["3"]["mrr_at_k"] == 0.5

    failed = report.cases[1]
    assert failed.retrieved == []
    assert failed.relevant_ranks == []
    assert report.aggregate[1].recall_at_k == pytest.approx(0.5)
    assert report.aggregate[1].precision_at_k == pytest.approx(1 / 6)
    assert report.aggregate[1].hit_rate_at_k == 0.5
    assert report.aggregate[1].mrr_at_k == pytest.approx(0.25)


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_evaluation_reports_the_retrieval_mode(mode):
    report = evaluate_rankings(dataset(), {}, retrieval_mode=mode, ks=[2])
    assert report.retrieval_mode == mode
    assert report.aggregate[0].k == 2


def test_evaluation_rejects_empty_targets_unknown_cases_and_invalid_k():
    with pytest.raises(ValueError, match="expected chunk, source ID, or source path"):
        RetrievalEvaluationCase(case_id="empty", query="query")
    with pytest.raises(ValueError, match="positive"):
        evaluate_rankings(dataset(), {}, retrieval_mode="semantic", ks=[0])
    with pytest.raises(ValueError, match="unknown evaluation cases"):
        evaluate_rankings(dataset(), {"unknown": []}, retrieval_mode="hybrid")


def test_evaluation_dataset_resolves_phase2a_source_ids_portably(tmp_path):
    sample = tmp_path / "data" / "raw" / "notes.txt"
    sample.parent.mkdir(parents=True)
    sample.write_text("test", encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        '{"dataset_id":"portable","version":1,"cases":[{"case_id":"case",'
        '"query":"query","expected_source_paths":["data/raw/notes.txt"],'
        '"expected_chunk_indices":[0]}]}',
        encoding="utf-8",
    )

    loaded = load_evaluation_dataset(dataset_path, tmp_path)

    assert loaded.cases[0].expected_chunk_ids[0].endswith(":000000")


def test_versioned_evaluation_targets_match_the_existing_sample_chunk():
    root = Path(__file__).resolve().parents[2]
    loaded = load_evaluation_dataset(
        root / "backend" / "tests" / "data" / "retrieval_eval_v1.json", root
    )
    sample = process_document(root / "data" / "raw" / "sample_trading_notes.txt")
    assert all(case.expected_chunk_ids == [sample.chunks[0].chunk_id] for case in loaded.cases)
