"""Offline routing, failover, continuity, and decision-integrity tests."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.reasoning.contracts import ReasoningRequest, ReasoningStatus
from app.reasoning.errors import (
    GatewayFailure,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderFailureKind,
    ProviderInvalidResponse,
    ProviderModelUnavailable,
    ProviderQuotaExhausted,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTemporaryUnavailable,
    ProviderTimeout,
)
from app.reasoning.fingerprints import PROMPT_CONTRACT_VERSION
from app.reasoning.gateway import ReasoningProviderGateway
from app.reasoning.provider import (
    ProviderAvailability,
    ProviderCapabilities,
    ProviderReply,
)
from app.reasoning.service import ReasoningService
from app.reasoning.factory import create_reasoning_gateway
from tests.test_reasoning import _context, _proposal
from tests.test_reasoning_evaluation import build_phase3b_golden_scenarios


def _caps(name, *, structured=True, availability=ProviderAvailability.AVAILABLE,
          model="model-v1", max_input_bytes=1048576, max_output_tokens=8192,
          contracts=(PROMPT_CONTRACT_VERSION,)):
    return ProviderCapabilities(
        provider_id=name,
        model_id=model,
        structured_output=structured,
        max_input_bytes=max_input_bytes,
        max_output_tokens=max_output_tokens,
        streaming=False,
        supported_contract_versions=contracts,
        availability=availability,
    )


class FakeAdapter:
    def __init__(self, name, *, response="{}", error=None, structured=True,
                 availability=ProviderAvailability.AVAILABLE, model="model-v1",
                 max_input_bytes=1048576, max_output_tokens=8192,
                 contracts=(PROMPT_CONTRACT_VERSION,)):
        self.provider_id = name
        self.model_id = model
        self.response = response
        self.error = error
        self.structured = structured
        self.availability = availability
        self.max_input_bytes = max_input_bytes
        self.max_output_tokens = max_output_tokens
        self.contracts = contracts
        self.calls = []

    @property
    def capabilities(self):
        return _caps(self.provider_id, structured=self.structured,
                     availability=self.availability, model=self.model_id,
                     max_input_bytes=self.max_input_bytes,
                     max_output_tokens=self.max_output_tokens, contracts=self.contracts)

    def generate(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return ProviderReply(self.provider_id, self.model_id, self.response)


def _gateway(first, second=None, *, required_max_output_tokens=1):
    providers = {first.provider_id: first}
    order = [first.provider_id]
    if second is not None:
        providers[second.provider_id] = second
        order.append(second.provider_id)
    return ReasoningProviderGateway(providers, order,
                                    required_max_output_tokens=required_max_output_tokens)


@pytest.mark.parametrize("error", [ProviderRateLimited("limited"), ProviderTimeout("timed out"),
                                    ProviderQuotaExhausted("quota"),
                                    ProviderTemporaryUnavailable("unavailable"),
                                    ProviderServerError("server"),
                                    ProviderModelUnavailable("model")])
def test_transient_primary_failure_fails_over_once(error):
    first = FakeAdapter("groq", error=error)
    second = FakeAdapter("openai", response='{"ok":true}')
    prompt = SimpleNamespace(contract_version=PROMPT_CONTRACT_VERSION,
                             deterministic_payload=lambda: "canonical prompt")

    reply = _gateway(first, second).generate(prompt)

    assert reply.provider_id == "openai"
    assert first.calls == [prompt]
    assert second.calls == [prompt]
    telemetry = dict(reply.telemetry)
    assert telemetry["provider_attempt_count"] == 2
    assert telemetry["failover_count"] == 1
    assert telemetry["final_provider"] == "openai"
    assert telemetry["attempt_1_failure"] == error.failure_kind.value
    assert telemetry["attempt_1_latency_ms"] >= 0
    assert telemetry["attempt_2_latency_ms"] >= 0
    assert telemetry["gateway_latency_ms"] >= 0


def test_primary_success_does_not_call_fallback_and_is_deterministic():
    first = FakeAdapter("groq", response='{"ok":true}')
    second = FakeAdapter("openai")
    prompt = SimpleNamespace(contract_version=PROMPT_CONTRACT_VERSION,
                             deterministic_payload=lambda: "canonical prompt")
    gateway = _gateway(first, second)

    replies = (gateway.generate(prompt), gateway.generate(prompt))

    assert first.calls == [prompt, prompt]
    assert second.calls == []
    assert replies[0].provider_id == replies[1].provider_id == "groq"
    assert dict(replies[0].telemetry)["failover_count"] == 0


def test_authentication_failure_is_terminal_and_does_not_call_fallback():
    first = FakeAdapter("groq", error=ProviderAuthenticationError("safe auth error"))
    second = FakeAdapter("openai")
    prompt = SimpleNamespace(contract_version=PROMPT_CONTRACT_VERSION,
                             deterministic_payload=lambda: "canonical prompt")

    with pytest.raises(GatewayFailure) as caught:
        _gateway(first, second).generate(prompt)

    assert caught.value.failure_kind.value == "AUTHENTICATION_FAILURE"
    assert first.calls and second.calls == []
    assert "safe auth error" not in str(caught.value)


def test_unsupported_structured_output_provider_is_skipped_for_fallback():
    first = FakeAdapter("groq", structured=False)
    second = FakeAdapter("openai", response='{"ok":true}')
    prompt = SimpleNamespace(contract_version=PROMPT_CONTRACT_VERSION,
                             deterministic_payload=lambda: "canonical prompt")

    reply = _gateway(first, second).generate(prompt)

    assert first.calls == []
    assert second.calls == [prompt]
    telemetry = dict(reply.telemetry)
    assert telemetry["skipped_1_reason"] == "CAPABILITY_UNSUPPORTED"


@pytest.mark.parametrize("first", [
    FakeAdapter("groq", availability=ProviderAvailability.TEMPORARILY_UNAVAILABLE),
    FakeAdapter("groq", max_input_bytes=10),
    FakeAdapter("groq", max_output_tokens=128),
    FakeAdapter("groq", contracts=("other-contract",)),
])
def test_temporary_health_and_unsatisfied_request_capabilities_are_skipped(first):
    second = FakeAdapter("openai", response='{"ok":true}')
    prompt = SimpleNamespace(contract_version=PROMPT_CONTRACT_VERSION,
                             deterministic_payload=lambda: "canonical prompt")

    reply = _gateway(first, second, required_max_output_tokens=160).generate(prompt)

    assert reply.provider_id == "openai"
    assert first.calls == [] and second.calls == [prompt]
    assert dict(reply.telemetry)["provider_attempt_count"] == 1


def test_terminal_configuration_and_unknown_failures_do_not_trigger_fallback():
    for error, expected in (
        (ProviderConfigurationError("bad config"), ProviderFailureKind.CONFIGURATION_FAILURE),
        (ProviderInvalidResponse("bad response"), ProviderFailureKind.INVALID_PROVIDER_RESPONSE),
        (ProviderError("unknown provider problem"), ProviderFailureKind.UNKNOWN_PROVIDER_FAILURE),
    ):
        first = FakeAdapter("groq", error=error)
        second = FakeAdapter("openai")
        prompt = SimpleNamespace(contract_version=PROMPT_CONTRACT_VERSION,
                                 deterministic_payload=lambda: "canonical prompt")
        with pytest.raises(GatewayFailure) as caught:
            _gateway(first, second).generate(prompt)
        assert caught.value.failure_kind is expected
        assert first.calls and second.calls == []


def test_missing_configuration_skips_provider_without_constructing_clients():
    settings = Settings(_env_file=None, groq_api_key=None, openai_api_key=None)
    from app.reasoning.factory import create_reasoning_gateway
    gateway = create_reasoning_gateway(settings)
    prompt = SimpleNamespace(contract_version=PROMPT_CONTRACT_VERSION,
                             deterministic_payload=lambda: "canonical prompt")

    with pytest.raises(GatewayFailure) as caught:
        gateway.generate(prompt)

    assert caught.value.failure_kind.value == "CONFIGURATION_FAILURE"
    assert gateway._providers["groq"]._client is None
    assert gateway._providers["openai"]._client is None


def test_trusted_provider_order_is_configurable_without_request_input():
    settings = Settings(_env_file=None, reasoning_provider_order="openai,groq",
                        groq_api_key=None, openai_api_key=None)

    gateway = create_reasoning_gateway(settings)

    assert gateway.provider_order == ("openai", "groq")


def test_no_eligible_provider_reports_request_size_and_health_classifications():
    prompt = SimpleNamespace(contract_version=PROMPT_CONTRACT_VERSION,
                             deterministic_payload=lambda: "x" * 100)
    for availability, limit, expected in (
        (ProviderAvailability.AVAILABLE, 10, ProviderFailureKind.REQUEST_TOO_LARGE),
        (ProviderAvailability.TEMPORARILY_UNAVAILABLE, 1000,
         ProviderFailureKind.TEMPORARY_UNAVAILABLE),
    ):
        adapter = FakeAdapter("groq", availability=availability, max_input_bytes=limit)
        with pytest.raises(GatewayFailure) as caught:
            _gateway(adapter).generate(prompt)
        assert caught.value.failure_kind is expected
        assert adapter.calls == []


def test_gateway_failure_returns_safe_non_actionable_result_with_attempt_telemetry():
    context = _context()
    request = ReasoningRequest(context=context, question="Explain these facts.")
    first = FakeAdapter("groq", error=ProviderTimeout("timeout"))
    second = FakeAdapter("openai", error=ProviderRateLimited("limited"))

    result = ReasoningService(_gateway(first, second)).reason(request)

    assert result.status is ReasoningStatus.INSUFFICIENT_EVIDENCE
    assert result.failure_code == "ProviderRateLimited"
    assert result.provider_id == "openai"
    assert result.provider_telemetry["provider_attempt_count"] == 2
    assert result.provider_telemetry["failover_count"] == 1
    assert result.provider_telemetry["terminal_failure_kind"] == "RATE_LIMITED"
    assert "limited" not in result.explanation


def test_gateway_passes_same_canonical_context_to_fallback_without_session_state():
    context = _context()
    request = ReasoningRequest(context=context, question="Explain the setup evidence.")
    first = FakeAdapter("groq", error=ProviderTemporaryUnavailable("offline"))
    second = FakeAdapter("openai", response=json.dumps(_proposal(context)))

    result = ReasoningService(_gateway(first, second)).reason(request)

    assert result.failure_code is None
    assert first.calls[0] is second.calls[0]
    prompt = first.calls[0]
    assert prompt is second.calls[0]
    assert prompt.deterministic_payload() == second.calls[0].deterministic_payload()
    canonical = json.loads(prompt.context_json)
    assert canonical["fingerprint"] == context.fingerprint
    assert canonical["candidate"]["candidate_id"] == str(context.candidate.candidate_id)
    assert canonical["candidate"]["strategy_fingerprint"] == context.candidate.strategy_fingerprint
    assert canonical["candidate"]["analysis_fingerprint"] == context.candidate.analysis_fingerprint
    assert canonical["deterministic_facts"] == context.model_dump(mode="json")["deterministic_facts"]
    assert prompt.question == request.question
    assert prompt.contract_version == context.prompt_contract_version


def test_malformed_successful_response_is_validated_once_without_failover():
    context = _context()
    request = ReasoningRequest(context=context, question="Explain.")
    first = FakeAdapter("groq", response='{"status":"ACTIONABLE"}')
    second = FakeAdapter("openai", response=json.dumps(_proposal(context)))

    result = ReasoningService(_gateway(first, second)).reason(request)

    assert result.status is ReasoningStatus.INSUFFICIENT_EVIDENCE
    assert result.failure_code == "invalid_provider_response"
    assert first.calls and second.calls == []


@pytest.mark.parametrize("scenario_id,expected", [
    ("golden-wait", ReasoningStatus.WAIT),
    ("golden-insufficient", ReasoningStatus.INSUFFICIENT_EVIDENCE),
    ("golden-rejected", ReasoningStatus.REJECTED),
])
def test_non_actionable_deterministic_statuses_bypass_gateway(scenario_id, expected):
    scenario = next(item for item in build_phase3b_golden_scenarios() if item.scenario_id == scenario_id)
    first = FakeAdapter("groq", response="{}")
    second = FakeAdapter("openai", response="{}")

    result = ReasoningService(_gateway(first, second)).reason(scenario.request)

    assert result.status is expected
    assert first.calls == second.calls == []


def test_gateway_enforces_two_provider_maximum_and_settings_are_trusted_only():
    first, second, third = (FakeAdapter(name) for name in ("groq", "openai", "untrusted"))
    with pytest.raises(ValueError):
        ReasoningProviderGateway({item.provider_id: item for item in (first, second, third)},
                                 ("groq", "openai", "untrusted"))
    with pytest.raises(ValidationError):
        Settings(_env_file=None, reasoning_provider_order="groq,attacker")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, groq_model="gsk_not-a-model")
