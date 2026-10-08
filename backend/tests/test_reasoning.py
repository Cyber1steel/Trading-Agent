"""Offline safety tests for the evidence-bound reasoning foundation."""

from datetime import timedelta
from decimal import Decimal
from hashlib import sha256
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.knowledge.embedding.contracts import RetrievalResult
from app.backtesting.engine import BacktestEngine
from app.reasoning.context import ReasoningContextBuilder
from app.reasoning.contracts import (
    CandidateStatus,
    KnowledgeEvidence,
    HistoricalValidationEvidence,
    NumericalClaim,
    ReasoningConclusion,
    ReasoningProposal,
    ReasoningRequest,
    ReasoningStatus,
    Uncertainty,
)
from app.reasoning.errors import ProviderTimeout, ProviderUnavailable, ReasoningContextError
from app.reasoning.fingerprints import context_fingerprint, digest
from app.reasoning.prompt import PromptEnvelope
from app.reasoning.provider import ProviderReply
from app.reasoning.service import ReasoningService
from app.reasoning.validation import validate_proposal
from tests.test_trade_candidate import assemble, candidate_for, make_inputs
from tests.test_backtesting import (
    FakeObservation,
    make_fake_analysis,
    make_manifest,
    make_request,
    make_strategy,
)


def _retrieved(*, chunk="chunk-1", source="source-1", document=None, text="Risk is calculated by the rules.", published=None, mode="hybrid"):
    return RetrievalResult(
        chunk_id=chunk, source_id=source, document_id=document or source, source_type="markdown",
        title="Risk notes", author=None, source_url=None, file_path="notes/risk.md",
        published_at=published or datetime(2020, 1, 1, tzinfo=timezone.utc),
        page_number=1, chunk_index=0, char_start=0, char_end=len(text), text=text,
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
        distance=0.2 if mode in ("semantic", "hybrid") else None,
        embedding_provider="fastembed" if mode in ("semantic", "hybrid") else None,
        embedding_model="model-v1" if mode in ("semantic", "hybrid") else None,
        embedding_dimension=384, processed_file_path="data/processed/risk.json",
        ingestion_chunk_size=500, ingestion_overlap=50, retrieval_mode=mode,
        lexical_score=0.5 if mode in ("lexical", "hybrid") else None,
        semantic_rank=1 if mode in ("semantic", "hybrid") else None,
        lexical_rank=2 if mode == "hybrid" else (1 if mode == "lexical" else None),
        hybrid_score=0.03 if mode == "hybrid" else None,
    )


def _context(*, candidate=None, knowledge=(), cutoff=None, analysis=None):
    if candidate is None:
        candidate, artifacts = assemble()
        analysis = analysis or artifacts[0]
    return ReasoningContextBuilder().build(
        candidate, evaluation_time=cutoff or candidate.evaluation_at,
        analysis=analysis, knowledge=tuple(knowledge),
    )


class FakeProvider:
    provider_id = "fake"
    model_id = "deterministic-test-v1"

    def __init__(self, payload=None, *, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0
        self.prompt = None

    def generate(self, prompt):
        self.calls += 1
        self.prompt = prompt
        if self.error:
            raise self.error
        value = self.payload(prompt) if callable(self.payload) else self.payload
        if isinstance(value, str):
            raw = value
        else:
            raw = json.dumps(value)
        return ProviderReply(
            provider_id=self.provider_id,
            model_id=self.model_id,
            raw_response=raw,
            request_metadata=(("temperature", 0),),
        )


def _proposal(context, **overrides):
    fields = {
        "status": "ACTIONABLE",
        "conclusion": "CANDIDATE_SUPPORTED",
        "uncertainty": "MODERATE",
        "supporting_evidence_ids": [context.deterministic_facts[0].evidence_id],
        "contradicting_evidence_ids": [],
        "missing_evidence": [],
        "assumptions": [],
        "risks": [],
        "limitations": ["This is a research explanation, not a trade instruction."],
        "knowledge_evidence_ids": [],
        "numerical_claims": [],
        "explanation": "The supplied deterministic candidate evidence supports review.",
    }
    fields.update(overrides)
    return fields


def test_context_is_immutable_and_prompt_is_deterministic():
    candidate, artifacts = assemble()
    knowledge = (_retrieved(),)
    context = _context(candidate=candidate, analysis=artifacts[0], knowledge=knowledge)
    again = _context(candidate=candidate, analysis=artifacts[0], knowledge=knowledge)
    assert context.fingerprint == again.fingerprint
    assert PromptEnvelope(ReasoningRequest(context=context, question="Explain the evidence.")).deterministic_payload() == \
        PromptEnvelope(ReasoningRequest(context=again, question="Explain the evidence.")).deterministic_payload()
    with pytest.raises((ValidationError, TypeError)):
        context.candidate.quantity = Decimal("999")
    with pytest.raises((ValidationError, TypeError)):
        context.knowledge[0].text = "mutated"
    with pytest.raises((ValidationError, TypeError)):
        context.deterministic_facts[0].value = "changed"
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        PromptEnvelope(ReasoningRequest(context=context, question="Explain.")).question = "mutated"


def test_retrieval_provenance_is_preserved_deduplicated_and_temporally_filtered():
    candidate, artifacts = assemble()
    future = _retrieved(
        chunk="future", source="future-source", text="Future market note.",
        published=candidate.evaluation_at + timedelta(days=1),
    )
    context = _context(
        candidate=candidate, analysis=artifacts[0],
        knowledge=(_retrieved(), _retrieved(), future),
    )
    assert len(context.knowledge) == 1
    evidence = context.knowledge[0]
    assert evidence.chunk_id == "chunk-1"
    assert evidence.source_id == evidence.document_id == "source-1"
    assert evidence.retrieval_mode == "hybrid"
    assert (evidence.semantic_rank, evidence.lexical_rank) == (1, 2)
    assert evidence.embedding_model == "model-v1"
    assert len(evidence.retrieved_content_sha256) == 64
    assert context.knowledge_unavailable_reason is None
    no_knowledge = _context(candidate=candidate, analysis=artifacts[0], knowledge=(future,))
    assert no_knowledge.knowledge == ()
    assert "No retrieved" in no_knowledge.knowledge_unavailable_reason
    unknown_time = _retrieved().model_copy(update={"published_at": None})
    unknown_context = _context(candidate=candidate, analysis=artifacts[0], knowledge=(unknown_time,))
    assert unknown_context.knowledge == ()
    assert no_knowledge.fingerprint == unknown_context.fingerprint


def test_retrieval_source_mismatch_and_content_digest_mismatch_are_rejected():
    candidate, artifacts = assemble()
    with pytest.raises(ValidationError):
        _context(candidate=candidate, analysis=artifacts[0], knowledge=(_retrieved(document="not-document"),))
    result = _retrieved()
    with pytest.raises(ValidationError):
        KnowledgeEvidence(
            evidence_id="knowledge:chunk-1", source_id=result.source_id,
            document_id=result.document_id, chunk_id=result.chunk_id,
            source_type=result.source_type, title=result.title, file_path=result.file_path,
            page_number=result.page_number, chunk_index=result.chunk_index,
            char_start=result.char_start, char_end=result.char_end,
            retrieved_content_sha256="0" * 64, retrieval_mode=result.retrieval_mode,
            retrieval_rank=1, text=result.text,
        )


def test_future_analysis_fingerprint_or_cutoff_mismatch_fails_closed():
    candidate, artifacts = assemble()
    future_analysis = artifacts[0].model_copy(update={"cutoff_at": candidate.evaluation_at + timedelta(minutes=1)})
    with pytest.raises(Exception, match="fingerprint|cutoff|prefix"):
        _context(candidate=candidate, analysis=future_analysis)
    with pytest.raises(Exception, match="candidate was evaluated after"):
        _context(candidate=candidate, analysis=artifacts[0], cutoff=candidate.evaluation_at - timedelta(microseconds=1))


def test_reasoning_fingerprint_changes_with_candidate_or_retrieved_source():
    candidate, artifacts = assemble()
    base = _context(candidate=candidate, analysis=artifacts[0], knowledge=(_retrieved(),))
    changed_candidate, changed_artifacts = assemble(target=Decimal("106"))
    changed = _context(candidate=changed_candidate, analysis=changed_artifacts[0], knowledge=(_retrieved(),))
    changed_source = _context(
        candidate=candidate, analysis=artifacts[0],
        knowledge=(_retrieved(chunk="other-chunk", source="other-source"),),
    )
    assert base.fingerprint != changed.fingerprint
    assert base.fingerprint != changed_source.fingerprint
    assert context_fingerprint(base) == base.fingerprint


def test_prompt_contract_version_is_part_of_reasoning_identity():
    context = _context()
    fields = context.model_dump(mode="python", exclude={"fingerprint"})
    fields["prompt_contract_version"] = "3a-prompt.2.0"
    changed = type(context).model_validate({**fields, "fingerprint": digest(fields, domain="reasoning-context")})
    assert changed.fingerprint != context.fingerprint


def test_context_builder_uses_injected_existing_retrieval_service():
    candidate, artifacts = assemble()

    class RetrievalStub:
        def __init__(self):
            self.call = None

        def search(self, query, top_k=5, filters=None):
            self.call = (query, top_k, filters)
            return [_retrieved()]

    retrieval = RetrievalStub()
    context = ReasoningContextBuilder().build_with_retrieval(
        candidate, evaluation_time=candidate.evaluation_at, question="risk assumptions",
        retrieval_service=retrieval, analysis=artifacts[0], top_k=4,
    )
    assert retrieval.call == ("risk assumptions", 4, None)
    assert [item.chunk_id for item in context.knowledge] == ["chunk-1"]


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_context_preserves_each_existing_retrieval_mode(mode):
    candidate, artifacts = assemble()
    context = _context(candidate=candidate, analysis=artifacts[0], knowledge=(_retrieved(mode=mode),))
    assert context.knowledge[0].retrieval_mode == mode
    assert context.knowledge[0].retrieval_rank == 1
    assert context.knowledge[0].semantic_rank == (1 if mode in ("semantic", "hybrid") else None)
    assert context.knowledge[0].lexical_rank == (2 if mode == "hybrid" else 1 if mode == "lexical" else None)


def test_backtest_is_explicit_historical_validation_and_requires_availability_time():
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    observations = tuple(
        FakeObservation(base + timedelta(hours=i), base + timedelta(hours=i + 1), Decimal("100") + i)
        for i in range(6)
    )
    analysis = make_fake_analysis(observations)
    strategy, setup = make_strategy()
    request = make_request(
        analysis, start=observations[0].bar_open, end=observations[4].bar_open,
        strategy=strategy, setup=setup,
    )
    backtest = BacktestEngine().run(request, make_manifest(analysis), analysis, strategy, setup)
    historical = HistoricalValidationEvidence.from_result(backtest, known_at=backtest.request.end)
    assert historical.tier.value == "HISTORICAL_VALIDATION"
    assert historical.out_of_sample_status == "NOT_ESTABLISHED"
    assert "not a forecast" in historical.limitations[0]

    candidate, _ = assemble()
    with pytest.raises(Exception, match="availability time is required"):
        ReasoningContextBuilder().build(candidate, evaluation_time=candidate.evaluation_at, backtest=backtest)
    with pytest.raises(Exception, match="not available at the reasoning cutoff"):
        ReasoningContextBuilder().build(
            candidate, evaluation_time=candidate.evaluation_at, backtest=backtest,
            backtest_available_at=candidate.evaluation_at + timedelta(seconds=1),
        )


def test_provider_valid_result_is_structured_auditable_and_repeatable():
    context = _context(knowledge=(_retrieved(),))
    provider = FakeProvider(_proposal(context, knowledge_evidence_ids=[context.knowledge[0].evidence_id]))
    request = ReasoningRequest(context=context, question="Explain the evidence.")
    result = ReasoningService(provider).reason(request)
    repeat = ReasoningService(FakeProvider(_proposal(context, knowledge_evidence_ids=[context.knowledge[0].evidence_id]))).reason(request)
    assert result.status is ReasoningStatus.ACTIONABLE
    assert result.inference_tier.value == "LLM_INFERENCE"
    assert result.candidate_id == context.candidate.candidate_id
    assert result.candidate_fingerprint == context.candidate.candidate_fingerprint
    assert result.knowledge_references[0].source_id == "source-1"
    assert result.fingerprint == repeat.fingerprint
    assert provider.prompt.contract_version
    with pytest.raises((ValidationError, TypeError)):
        result.request_metadata["new"] = "value"
    with pytest.raises((ValidationError, TypeError)):
        result.supporting_evidence[0].label = "forged"
    with pytest.raises((ValidationError, TypeError)):
        context.candidate.risk_request.provenance["forged"] = "value"


@pytest.mark.parametrize("bad", [
    {"supporting_evidence_ids": ["not-in-context"]},
    {"knowledge_evidence_ids": ["knowledge:not-retrieved"]},
    {"numerical_claims": [{"evidence_id": "not-in-context", "value": "30%"}]},
    {"explanation": "The strategy will make 30% profit."},
])
def test_unsupported_citations_claims_and_future_profitability_fail_closed(bad):
    context = _context()
    provider = FakeProvider(_proposal(context, **bad))
    result = ReasoningService(provider).reason(ReasoningRequest(context=context, question="Explain."))
    assert result.status is ReasoningStatus.INSUFFICIENT_EVIDENCE
    assert result.failure_code == "invalid_provider_response"


def test_numerical_claim_must_match_exact_deterministic_decimal():
    context = _context()
    risk = next(item for item in context.deterministic_facts if item.kind == "risk_amount")
    payload = _proposal(context, numerical_claims=[{"evidence_id": risk.evidence_id, "value": str(risk.value)}])
    # The provider contract parses numeric values strictly; the typed claim must use a Decimal instance.
    proposal = ReasoningProposal.model_validate({
        **payload,
        "numerical_claims": [{"evidence_id": risk.evidence_id, "value": risk.value}],
    })
    validate_proposal(proposal, context)
    forged = proposal.model_copy(update={
        "numerical_claims": (NumericalClaim(evidence_id=risk.evidence_id, value=Decimal("999")),)
    })
    with pytest.raises(Exception, match="conflicts"):
        validate_proposal(forged, context)


def test_unstructured_numerical_claim_in_free_text_is_rejected():
    context = _context()
    proposal = ReasoningProposal.model_validate(_proposal(context, risks=["This setup is up 30% already."]))
    with pytest.raises(Exception, match="unsupported numerical"):
        validate_proposal(proposal, context)


def test_deterministic_wait_cannot_be_upgraded_by_provider():
    artifacts = make_inputs(closes=(Decimal("9"), Decimal("9"), Decimal("9")))
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2], evaluation=artifacts[3],
        risk_request=artifacts[4], risk_result=artifacts[5], execution_assumptions=artifacts[4].execution_assumptions,
    )
    context = _context(candidate=candidate, analysis=artifacts[0])
    provider = FakeProvider(_proposal(context))
    result = ReasoningService(provider).reason(ReasoningRequest(context=context, question="Explain."))
    assert candidate.status is CandidateStatus.WAIT
    assert result.status is ReasoningStatus.WAIT
    assert provider.calls == 0


@pytest.mark.parametrize("candidate_status,closes,expected", [
    (CandidateStatus.INSUFFICIENT_EVIDENCE, (Decimal("11"),), ReasoningStatus.INSUFFICIENT_EVIDENCE),
])
def test_unavailable_or_rejected_deterministic_status_is_never_upgraded(candidate_status, closes, expected):
    artifacts = make_inputs(closes=closes)
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=artifacts[4], risk_result=artifacts[5],
        execution_assumptions=artifacts[4].execution_assumptions,
    )
    assert candidate.status is candidate_status
    context = _context(candidate=candidate, analysis=artifacts[0])
    provider = FakeProvider(_proposal(context))
    result = ReasoningService(provider).reason(ReasoningRequest(context=context, question="Explain."))
    assert result.status is expected
    assert result.deterministic_candidate_status is candidate_status
    assert provider.calls == 0


def test_rejected_candidate_stays_rejected_without_provider_call():
    artifacts = list(make_inputs())
    invalid_request = artifacts[4].model_copy(update={"stop_price": artifacts[4].entry_price})
    from app.risk.service import RiskEngineService
    invalid_result = RiskEngineService().evaluate(invalid_request)
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=invalid_request, risk_result=invalid_result,
    )
    assert candidate.status is CandidateStatus.REJECTED
    context = _context(candidate=candidate, analysis=artifacts[0])
    provider = FakeProvider(_proposal(context))
    result = ReasoningService(provider).reason(ReasoningRequest(context=context, question="Explain."))
    assert result.status is ReasoningStatus.REJECTED
    assert provider.calls == 0


@pytest.mark.parametrize("error,code", [
    (TimeoutError(), "provider_timeout"),
    (ProviderTimeout(), "ProviderTimeout"),
    (ProviderUnavailable(), "ProviderUnavailable"),
    (RuntimeError("adapter error"), "provider_failure"),
])
def test_provider_failures_return_deterministic_insufficient_result(error, code):
    context = _context()
    result = ReasoningService(FakeProvider(error=error)).reason(
        ReasoningRequest(context=context, question="Explain.")
    )
    assert result.status is ReasoningStatus.INSUFFICIENT_EVIDENCE
    assert result.failure_code == code
    assert result.supporting_evidence == ()


def test_malformed_provider_payload_is_insufficient_not_guessed():
    context = _context()
    result = ReasoningService(FakeProvider("{malformed" )).reason(
        ReasoningRequest(context=context, question="Explain.")
    )
    assert result.status is ReasoningStatus.INSUFFICIENT_EVIDENCE
    assert result.failure_code == "invalid_provider_response"


def test_context_fails_if_candidate_evidence_is_after_cutoff():
    candidate, artifacts = assemble()
    with pytest.raises(Exception):
        ReasoningContextBuilder().build(
            candidate, evaluation_time=candidate.evaluation_at - timedelta(seconds=1), analysis=artifacts[0]
        )


def test_context_and_prompt_do_not_expose_untyped_provider_dictionaries():
    context = _context()
    prompt = PromptEnvelope(ReasoningRequest(context=context, question="Compare the evidence."))
    assert "DETERMINISTIC_STRATEGY_RISK" in prompt.context_json
    assert context.market_observations[0].tier.value == "DETERMINISTIC_MARKET"
    assert context.candidate.status is CandidateStatus.ACTIONABLE
    assert "educational/reference material" in prompt.system_instructions
    assert "never an order" in prompt.system_instructions
    with pytest.raises(ValidationError):
        ReasoningRequest(context=context, question="question", arbitrary_market_data={"close": 123})


def test_unsupported_provider_returns_fail_closed_result():
    context = _context()
    result = ReasoningService(object()).reason(ReasoningRequest(context=context, question="Explain."))
    assert result.status is ReasoningStatus.INSUFFICIENT_EVIDENCE
    assert result.failure_code == "unsupported_provider"


def test_service_revalidates_context_before_any_provider_call():
    context = _context()
    corrupted = context.model_copy(update={"effective_candidate_status": CandidateStatus.WAIT})
    request = ReasoningRequest(context=context, question="Explain.").model_copy(update={"context": corrupted})
    provider = FakeProvider(_proposal(context))
    with pytest.raises(ReasoningContextError, match="integrity revalidation"):
        ReasoningService(provider).reason(request)
    assert provider.calls == 0
