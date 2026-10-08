"""Deterministic orchestrator: build prompt, call provider, validate, fingerprint."""

import json
from hashlib import sha256
from pydantic import ValidationError

from app.reasoning.contracts import (
    CandidateStatus,
    EvidenceTier,
    ReasoningConclusion,
    ReasoningProposal,
    ReasoningRequest,
    ReasoningResult,
    ReasoningStatus,
    Uncertainty,
)
from app.reasoning.errors import (
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    ReasoningContextError,
    ReasoningValidationError,
)
from app.reasoning.fingerprints import PROMPT_CONTRACT_VERSION, result_fingerprint
from app.reasoning.prompt import PromptEnvelope
from app.reasoning.validation import validate_proposal

MAX_PROVIDER_RESPONSE_CHARS = 65536


def _forced_status(context):
    return {
        CandidateStatus.REJECTED: ReasoningStatus.REJECTED,
        CandidateStatus.INSUFFICIENT_EVIDENCE: ReasoningStatus.INSUFFICIENT_EVIDENCE,
        CandidateStatus.WAIT: ReasoningStatus.WAIT,
    }.get(context.effective_candidate_status)


class ReasoningService:
    """Run a model only downstream of context validation; invalid responses fail closed."""

    def __init__(self, provider):
        self.provider = provider

    def reason(self, request: ReasoningRequest) -> ReasoningResult:
        try:
            if not isinstance(request, ReasoningRequest):
                raise TypeError("a validated ReasoningRequest is required")
            request = ReasoningRequest.model_validate(request.model_dump(mode="python"))
        except Exception as exc:
            raise ReasoningContextError("reasoning request/context failed integrity revalidation") from exc
        context = request.context
        forced = _forced_status(context)
        if forced is not None:
            conclusion = {
                ReasoningStatus.REJECTED: ReasoningConclusion.REJECTED,
                ReasoningStatus.WAIT: ReasoningConclusion.WAIT,
                ReasoningStatus.INSUFFICIENT_EVIDENCE: ReasoningConclusion.INSUFFICIENT_EVIDENCE,
            }[forced]
            return self._result(
                request, status=forced, conclusion=conclusion, uncertainty=Uncertainty.HIGH,
                explanation="The deterministic candidate status controls this result; no LLM upgrade was permitted.",
                provider_id=getattr(self.provider, "provider_id", "not_called"),
                model_id=getattr(self.provider, "model_id", "not_called"),
                failure_code=None,
            )
        if self.provider is None or not callable(getattr(self.provider, "generate", None)):
            return self._failure(
                request, "unsupported_provider", ProviderUnavailable,
                "unsupported", "unsupported",
            )
        reply = None
        try:
            reply = self.provider.generate(PromptEnvelope(request))
            if not reply.provider_id or not reply.model_id:
                raise ProviderUnavailable("provider/model identity is missing")
            if type(reply.raw_response) is not str or len(reply.raw_response) > MAX_PROVIDER_RESPONSE_CHARS:
                raise ProviderError("provider response is missing or exceeds the configured size limit")
            if type(reply.request_metadata) is not tuple or any(
                type(entry) is not tuple or len(entry) != 2 for entry in reply.request_metadata
            ):
                raise ProviderError("provider request metadata is malformed")
            metadata_names = [entry[0] for entry in reply.request_metadata]
            if any(
                type(key) is not str or (type(value) not in (str, int, bool) and value is not None)
                for key, value in reply.request_metadata
            ) or len(metadata_names) != len(set(metadata_names)):
                raise ProviderError("provider request metadata is malformed")
            if tuple(sorted(reply.request_metadata)) != reply.request_metadata:
                raise ProviderError("provider metadata must be a deterministically ordered tuple")
            if type(reply.telemetry) is not tuple or any(
                type(entry) is not tuple or len(entry) != 2 for entry in reply.telemetry
            ):
                raise ProviderError("provider telemetry is malformed")
            if any(
                type(key) is not str or (type(value) not in (str, int, float, bool) and value is not None)
                for key, value in reply.telemetry
            ) or len({entry[0] for entry in reply.telemetry}) != len(reply.telemetry):
                raise ProviderError("provider telemetry is malformed")
            metadata = dict(reply.request_metadata)
            proposal = ReasoningProposal.model_validate_json(reply.raw_response)
            validate_proposal(proposal, context)
            return self._result(
                request,
                status=proposal.status,
                conclusion=proposal.conclusion,
                uncertainty=proposal.uncertainty,
                explanation=proposal.explanation,
                supporting=tuple(item for item in context.deterministic_facts if item.evidence_id in proposal.supporting_evidence_ids),
                contradicting=tuple(item for item in context.deterministic_facts if item.evidence_id in proposal.contradicting_evidence_ids),
                missing=proposal.missing_evidence,
                assumptions=proposal.assumptions,
                risks=proposal.risks,
                limitations=proposal.limitations,
                knowledge_ids=proposal.knowledge_evidence_ids,
                provider_id=reply.provider_id,
                model_id=reply.model_id,
                metadata=metadata,
                telemetry=dict(reply.telemetry),
                provider_response_sha256=self._response_hash(reply),
                failure_code=None,
            )
        except TimeoutError:
            return self._failure(request, "provider_timeout", ProviderTimeout, "not_available", "not_available")
        except (ProviderError, ProviderUnavailable, ProviderTimeout) as exc:
            return self._failure(request, type(exc).__name__, type(exc), getattr(self.provider, "provider_id", "not_available"), getattr(self.provider, "model_id", "not_available"), reply)
        except (ValidationError, ValueError, json.JSONDecodeError, ReasoningValidationError):
            return self._failure(request, "invalid_provider_response", ReasoningValidationError, getattr(self.provider, "provider_id", "unknown"), getattr(self.provider, "model_id", "unknown"), reply)
        except Exception:
            # Adapter-specific failures never turn into a guess or a trading recommendation.
            return self._failure(request, "provider_failure", ProviderError, getattr(self.provider, "provider_id", "unknown"), getattr(self.provider, "model_id", "unknown"), reply)

    def _failure(self, request, code, _error_type, provider_id, model_id, reply=None):
        return self._result(
            request,
            status=ReasoningStatus.INSUFFICIENT_EVIDENCE,
            conclusion=ReasoningConclusion.INSUFFICIENT_EVIDENCE,
            uncertainty=Uncertainty.HIGH,
            explanation="No validated reasoning output is available. Treat the provider result as unusable.",
            provider_id=provider_id,
            model_id=model_id,
            provider_response_sha256=self._response_hash(reply),
            failure_code=code,
        )

    def _result(
        self, request, *, status, conclusion, uncertainty, explanation,
        supporting=(), contradicting=(), missing=(), assumptions=(), risks=(), limitations=(),
        knowledge_ids=(), provider_id, model_id, metadata=None, telemetry=None,
        provider_response_sha256=None, failure_code,
    ):
        context = request.context
        knowledge = {item.evidence_id: item for item in context.knowledge}
        fields = {
            "schema_version": "3a.1.0",
            "inference_tier": EvidenceTier.LLM_INFERENCE,
            "status": status,
            "deterministic_candidate_status": context.effective_candidate_status,
            "conclusion": conclusion,
            "uncertainty": uncertainty,
            "explanation": explanation,
            "supporting_evidence": tuple(supporting),
            "contradicting_evidence": tuple(contradicting),
            "missing_evidence": tuple(missing),
            "assumptions": tuple(assumptions),
            "risks": tuple(risks),
            "limitations": tuple(limitations),
            "knowledge_references": tuple(knowledge[key] for key in knowledge_ids if key in knowledge),
            "candidate_id": context.candidate.candidate_id,
            "candidate_fingerprint": context.candidate.candidate_fingerprint,
            "context_fingerprint": context.fingerprint,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "provider_id": provider_id,
            "model_id": model_id,
            "question": request.question,
            "provider_response_sha256": provider_response_sha256,
            "request_metadata": metadata or {},
            "provider_telemetry": telemetry or {},
            "failure_code": failure_code,
        }
        return ReasoningResult(**fields, fingerprint=result_fingerprint(fields))

    @staticmethod
    def _response_hash(reply):
        raw = getattr(reply, "raw_response", None)
        return sha256(raw.encode("utf-8")).hexdigest() if isinstance(raw, str) else None
