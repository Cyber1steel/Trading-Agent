"""Deterministic reciprocal-rank fusion of semantic and lexical candidates."""

from app.core.config import get_settings
from app.knowledge.embedding.contracts import RetrievalFilters, RetrievalResult
from app.knowledge.embedding.errors import EmptyQueryError, StaleEmbeddingError
from app.knowledge.embedding.lexical import LexicalRetrievalService
from app.knowledge.embedding.retrieval import SemanticRetrievalService


class HybridRetrievalService:
    """Combine ranked lists with RRF; raw cosine distance and FTS rank are never added."""

    def __init__(
        self,
        semantic: SemanticRetrievalService,
        lexical: LexicalRetrievalService,
        *,
        semantic_candidates: int | None = None,
        lexical_candidates: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self.semantic = semantic
        self.lexical = lexical
        self.semantic_candidates = (
            semantic_candidates if semantic_candidates is not None else settings.hybrid_semantic_candidates
        )
        self.lexical_candidates = (
            lexical_candidates if lexical_candidates is not None else settings.hybrid_lexical_candidates
        )
        self.rrf_k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k
        if min(self.semantic_candidates, self.lexical_candidates, self.rrf_k) <= 0:
            raise ValueError("Candidate limits and RRF k must be positive")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
        *,
        semantic_candidates: int | None = None,
        lexical_candidates: int | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise EmptyQueryError("Search query must contain non-whitespace text.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        sem_limit = (
            semantic_candidates if semantic_candidates is not None else self.semantic_candidates
        )
        lex_limit = lexical_candidates if lexical_candidates is not None else self.lexical_candidates
        if min(sem_limit, lex_limit) <= 0:
            raise ValueError("Candidate limits must be positive")

        # Both services apply the same filters in their database candidate query.
        semantic_results = self.semantic.search(query, sem_limit, filters)
        lexical_results = self.lexical.search(query, lex_limit, filters)
        fused: dict[str, dict[str, object]] = {}
        for mode_results, rank_field in (
            (semantic_results, "semantic_rank"),
            (lexical_results, "lexical_rank"),
        ):
            for rank, result in enumerate(mode_results, start=1):
                entry = fused.get(result.chunk_id)
                if entry is None:
                    entry = {
                        "result": result,
                        "semantic_rank": None,
                        "lexical_rank": None,
                        "lexical_score": None,
                    }
                    fused[result.chunk_id] = entry
                else:
                    existing = entry["result"]
                    if existing.source_id != result.source_id:
                        raise StaleEmbeddingError(
                            f"Chunk identity {result.chunk_id} resolved to conflicting source IDs."
                        )
                    # Prefer the semantic object because it includes model and distance fields.
                    if rank_field == "semantic_rank":
                        entry["result"] = result
                if rank_field == "lexical_rank":
                    entry["lexical_score"] = result.lexical_score
                if entry[rank_field] is None:
                    entry[rank_field] = rank

        ranked = []
        for chunk_id, entry in fused.items():
            semantic_rank = entry["semantic_rank"]
            lexical_rank = entry["lexical_rank"]
            base_result = entry["result"]
            score = sum(
                1.0 / (self.rrf_k + rank)
                for rank in (semantic_rank, lexical_rank)
                if rank is not None
            )
            lexical_score = entry["lexical_score"]
            if lexical_score is None:
                lexical_score = base_result.lexical_score
            result = base_result.model_copy(update={
                "retrieval_mode": "hybrid",
                "semantic_rank": semantic_rank,
                "lexical_rank": lexical_rank,
                "lexical_score": lexical_score,
                "hybrid_score": score,
            })
            ranked.append((score, chunk_id, result.source_id, result))

        # A future reranker can consume this fused candidate list before final top-k.
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in ranked[:top_k]]
