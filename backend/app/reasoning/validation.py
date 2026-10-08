"""Deterministic checks over provider claims and candidate status semantics."""

import re
from decimal import Decimal, InvalidOperation

from app.reasoning.contracts import (
    CandidateStatus,
    ReasoningConclusion,
    ReasoningContext,
    ReasoningProposal,
    ReasoningStatus,
)
from app.reasoning.errors import ReasoningValidationError

_UNSAFE_FUTURE_CLAIM = re.compile(
    r"\b(will\s+(?:make|earn|return|profit)|guaranteed\s+(?:profit|return)|risk[- ]free|certain\s+profit)\b",
    re.IGNORECASE,
)
_NUMERIC_LITERAL = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)%?(?![A-Za-z0-9_])")


def validate_proposal(proposal: ReasoningProposal, context: ReasoningContext) -> None:
    facts = {item.evidence_id: item for item in context.deterministic_facts}
    knowledge = {item.evidence_id for item in context.knowledge}
    referenced = (*proposal.supporting_evidence_ids, *proposal.contradicting_evidence_ids)
    if len(referenced) != len(set(referenced)):
        raise ReasoningValidationError("duplicate evidence citation")
    unknown = set(referenced) - facts.keys()
    if unknown:
        raise ReasoningValidationError("provider cited evidence absent from the supplied context")
    if set(proposal.knowledge_evidence_ids) - knowledge:
        raise ReasoningValidationError("provider cited a knowledge source absent from retrieval results")
    if len(proposal.knowledge_evidence_ids) != len(set(proposal.knowledge_evidence_ids)):
        raise ReasoningValidationError("duplicate knowledge citation")
    for claim in proposal.numerical_claims:
        fact = facts.get(claim.evidence_id)
        if fact is None or fact.value is None:
            raise ReasoningValidationError("numerical claim does not cite a typed deterministic value")
        if type(claim.value) is not type(fact.value) or claim.value != fact.value:
            raise ReasoningValidationError("numerical claim conflicts with deterministic evidence")
    cited_values = [
        facts[evidence_id].value
        for evidence_id in (*proposal.supporting_evidence_ids, *(item.evidence_id for item in proposal.numerical_claims))
        if evidence_id in facts and facts[evidence_id].value is not None
    ]
    free_text = (
        proposal.explanation,
        *proposal.missing_evidence,
        *proposal.assumptions,
        *proposal.risks,
        *proposal.limitations,
    )
    for text in free_text:
        for token in _NUMERIC_LITERAL.findall(text):
            percentage = token.endswith("%")
            try:
                observed = Decimal(token[:-1] if percentage else token)
            except InvalidOperation as exc:
                raise ReasoningValidationError("unsupported numeric text in explanation") from exc
            if percentage:
                observed /= Decimal(100)
            if not any(
                (Decimal(value) if type(value) is int else value) == observed
                for value in cited_values
                if isinstance(value, (int, Decimal)) and type(value) is not bool
            ):
                raise ReasoningValidationError("explanation contains an unsupported numerical claim")
    if any(_UNSAFE_FUTURE_CLAIM.search(text) for text in free_text):
        raise ReasoningValidationError("unsupported future-profitability claim")

    candidate_status = context.effective_candidate_status
    if candidate_status is CandidateStatus.REJECTED:
        if proposal.status is not ReasoningStatus.REJECTED:
            raise ReasoningValidationError("reasoning cannot override a rejected deterministic candidate")
        if proposal.conclusion is not ReasoningConclusion.REJECTED:
            raise ReasoningValidationError("rejected candidate requires a rejected conclusion")
    elif candidate_status is CandidateStatus.INSUFFICIENT_EVIDENCE:
        if proposal.status is not ReasoningStatus.INSUFFICIENT_EVIDENCE:
            raise ReasoningValidationError("reasoning cannot upgrade insufficient deterministic evidence")
    elif candidate_status is CandidateStatus.WAIT:
        if proposal.status not in (ReasoningStatus.WAIT, ReasoningStatus.INSUFFICIENT_EVIDENCE):
            raise ReasoningValidationError("reasoning cannot upgrade a waiting candidate")
    elif proposal.status is ReasoningStatus.REJECTED:
        raise ReasoningValidationError("LLM inference cannot create a deterministic rejection")
    if proposal.status is ReasoningStatus.ACTIONABLE and not proposal.supporting_evidence_ids:
        raise ReasoningValidationError("an actionable explanation must cite supplied deterministic evidence")

    expected_conclusion = {
        ReasoningStatus.ACTIONABLE: (ReasoningConclusion.CANDIDATE_SUPPORTED,),
        ReasoningStatus.WAIT: (ReasoningConclusion.WAIT, ReasoningConclusion.EXPLANATION_ONLY),
        ReasoningStatus.INSUFFICIENT_EVIDENCE: (ReasoningConclusion.INSUFFICIENT_EVIDENCE, ReasoningConclusion.EXPLANATION_ONLY),
        ReasoningStatus.REJECTED: (ReasoningConclusion.REJECTED,),
    }[proposal.status]
    if proposal.conclusion not in expected_conclusion:
        raise ReasoningValidationError("reasoning conclusion conflicts with its structured status")
