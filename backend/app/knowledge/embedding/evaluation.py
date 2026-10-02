"""Small deterministic retrieval-evaluation contracts and ranking metrics."""

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.knowledge.embedding.contracts import RetrievalResult

RetrievalMode = Literal["semantic", "lexical", "hybrid"]


class RetrievalEvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    expected_source_paths: list[str] = Field(default_factory=list)
    expected_chunk_indices: list[int] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def must_have_relevant_evidence(self) -> "RetrievalEvaluationCase":
        if (
            not self.expected_chunk_ids
            and not self.expected_source_ids
            and not self.expected_source_paths
        ):
            raise ValueError("Each evaluation case needs an expected chunk, source ID, or source path")
        if self.expected_chunk_indices and not self.expected_source_paths:
            raise ValueError("Expected chunk indices require expected source paths")
        if any(index < 0 for index in self.expected_chunk_indices):
            raise ValueError("Expected chunk indices must be non-negative")
        if not self.query.strip():
            raise ValueError("Evaluation query cannot be blank")
        return self


class RetrievalEvaluationDataset(BaseModel):
    dataset_id: str
    version: int = Field(ge=1)
    cases: list[RetrievalEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self) -> "RetrievalEvaluationDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case IDs must be unique")
        return self


def load_evaluation_dataset(path: Path, repository_root: Path) -> RetrievalEvaluationDataset:
    """Load a versioned dataset and resolve relocatable source-path targets to Phase 2A IDs."""
    dataset = RetrievalEvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))
    resolved_cases = []
    root = repository_root.resolve()
    for case in dataset.cases:
        source_ids = list(case.expected_source_ids)
        chunk_ids = list(case.expected_chunk_ids)
        for relative_path in case.expected_source_paths:
            candidate = (root / relative_path).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(f"Evaluation source path escapes repository root: {relative_path}")
            if not candidate.is_file():
                raise ValueError(f"Evaluation source file does not exist: {candidate}")
            source_id = sha256(str(candidate).encode("utf-8")).hexdigest()[:24]
            if case.expected_chunk_indices:
                chunk_ids.extend(
                    f"{source_id}:{index:06d}" for index in case.expected_chunk_indices
                )
            else:
                source_ids.append(source_id)
        resolved_cases.append(case.model_copy(update={
            "expected_source_ids": sorted(set(source_ids)),
            "expected_chunk_ids": sorted(set(chunk_ids)),
        }))
    return dataset.model_copy(update={"cases": resolved_cases})


class EvaluationRankedItem(BaseModel):
    rank: int
    chunk_id: str
    source_id: str
    relevant: bool


class EvaluationCaseReport(BaseModel):
    case_id: str
    query: str
    expected_chunk_ids: list[str]
    expected_source_ids: list[str]
    retrieved: list[EvaluationRankedItem]
    matched_expected_chunk_ids: list[str]
    matched_expected_source_ids: list[str]
    relevant_ranks: list[int]
    metrics_at_k: dict[str, dict[str, float]]


class EvaluationAggregate(BaseModel):
    k: int
    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float
    mrr_at_k: float


class RetrievalEvaluationReport(BaseModel):
    dataset_id: str
    dataset_version: int
    retrieval_mode: RetrievalMode
    ks: list[int]
    cases: list[EvaluationCaseReport]
    aggregate: list[EvaluationAggregate]


def evaluate_rankings(
    dataset: RetrievalEvaluationDataset,
    rankings: Mapping[str, Sequence[RetrievalResult]],
    *,
    retrieval_mode: RetrievalMode,
    ks: Sequence[int] = (1, 3, 5),
) -> RetrievalEvaluationReport:
    """Evaluate supplied rankings; this function never invents or changes retrieval output."""
    normalized_ks = sorted(set(ks))
    if not normalized_ks or normalized_ks[0] <= 0:
        raise ValueError("Evaluation K values must be positive integers")
    known_cases = {case.case_id for case in dataset.cases}
    unexpected = set(rankings) - known_cases
    if unexpected:
        raise ValueError(f"Rankings contain unknown evaluation cases: {sorted(unexpected)}")

    reports = []
    for case in dataset.cases:
        results = list(rankings.get(case.case_id, ()))
        unique_targets = len(set(case.expected_chunk_ids)) + len(set(case.expected_source_ids))
        if unique_targets == 0:
            raise ValueError(
                f"Evaluation case {case.case_id} has unresolved source-path targets; "
                "load it with load_evaluation_dataset() first."
            )
        retrieved = [
            EvaluationRankedItem(
                rank=rank,
                chunk_id=result.chunk_id,
                source_id=result.source_id,
                relevant=(
                    result.chunk_id in case.expected_chunk_ids
                    or result.source_id in case.expected_source_ids
                ),
            )
            for rank, result in enumerate(results, start=1)
        ]
        matched_chunks = sorted(
            set(case.expected_chunk_ids) & {item.chunk_id for item in retrieved}
        )
        matched_sources = sorted(
            set(case.expected_source_ids) & {item.source_id for item in retrieved}
        )
        relevant_ranks = [item.rank for item in retrieved if item.relevant]
        metrics: dict[str, dict[str, float]] = {}
        for k in normalized_ks:
            window = retrieved[:k]
            found_targets = {
                target for target in case.expected_chunk_ids
                if any(item.chunk_id == target for item in window)
            }
            found_targets.update(
                target for target in case.expected_source_ids
                if any(item.source_id == target for item in window)
            )
            metrics[str(k)] = {
                "recall_at_k": len(found_targets) / unique_targets,
                "precision_at_k": sum(item.relevant for item in window) / k,
                "hit_rate_at_k": float(any(item.relevant for item in window)),
                "mrr_at_k": next(
                    (1.0 / item.rank for item in window if item.relevant), 0.0
                ),
            }
        reports.append(EvaluationCaseReport(
            case_id=case.case_id,
            query=case.query,
            expected_chunk_ids=case.expected_chunk_ids,
            expected_source_ids=case.expected_source_ids,
            retrieved=retrieved,
            matched_expected_chunk_ids=matched_chunks,
            matched_expected_source_ids=matched_sources,
            relevant_ranks=relevant_ranks,
            metrics_at_k=metrics,
        ))

    aggregates = []
    for k in normalized_ks:
        metric_name = str(k)
        denominator = len(reports)
        aggregates.append(EvaluationAggregate(
            k=k,
            recall_at_k=sum(row.metrics_at_k[metric_name]["recall_at_k"] for row in reports) / denominator,
            precision_at_k=sum(row.metrics_at_k[metric_name]["precision_at_k"] for row in reports) / denominator,
            hit_rate_at_k=sum(row.metrics_at_k[metric_name]["hit_rate_at_k"] for row in reports) / denominator,
            mrr_at_k=sum(row.metrics_at_k[metric_name]["mrr_at_k"] for row in reports) / denominator,
        ))
    return RetrievalEvaluationReport(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        retrieval_mode=retrieval_mode,
        ks=normalized_ks,
        cases=reports,
        aggregate=aggregates,
    )
