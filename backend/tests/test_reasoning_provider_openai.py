"""Offline tests for the OpenAI reasoning adapter; no requests are made."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.reasoning.errors import ProviderAuthenticationError, ProviderRateLimited, ProviderTimeout
from app.reasoning.factory import create_reasoning_provider
from app.reasoning.gateway import ReasoningProviderGateway
from app.reasoning.providers.openai import OpenAIReasoningProvider


def _settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def _prompt():
    return SimpleNamespace(
        system_instructions="instructions", context_json="{}", question="question",
        output_schema_json="{}", contract_version="3a-prompt.1.0",
        deterministic_payload=lambda: "canonical prompt",
    )


def test_provider_factory_and_import_do_not_require_credentials_or_construct_sdks():
    gateway = create_reasoning_provider(_settings(openai_api_key=None, groq_api_key=None))
    assert isinstance(gateway, ReasoningProviderGateway)
    assert isinstance(gateway._providers["openai"], OpenAIReasoningProvider)
    assert gateway._providers["openai"]._client is None
    assert gateway._providers["groq"]._client is None


def test_provider_sends_existing_contract_with_bounds_and_returns_only_safe_metadata():
    raw = '{"structured":"response"}'

    class Responses:
        kwargs = None

        def parse(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(output_text=raw, usage=SimpleNamespace(
                input_tokens=12, output_tokens=5, total_tokens=17,
            ))

    responses = Responses()
    provider = OpenAIReasoningProvider(
        _settings(openai_api_key=SecretStr("secret-test")),
        client=SimpleNamespace(responses=responses),
    )
    prompt = SimpleNamespace(
        system_instructions="instructions", context_json='{"trusted":"as data"}', question="question",
        output_schema_json="{}", deterministic_payload=lambda: "canonical prompt",
    )
    reply = provider.generate(prompt)
    assert responses.kwargs["model"] == "gpt-6.1-sol"
    assert responses.kwargs["instructions"] == "instructions"
    assert '"trusted":"as data"' in responses.kwargs["input"][0]["content"]
    assert responses.kwargs["text_format"].__name__ == "ReasoningProposal"
    assert responses.kwargs["max_output_tokens"] == 1600
    assert responses.kwargs["store"] is False
    assert reply.raw_response == raw
    telemetry = dict(reply.telemetry)
    assert telemetry["input_tokens"] == 12
    assert telemetry["output_tokens"] == 5
    assert telemetry["total_tokens"] == 17
    assert type(telemetry["latency_ms"]) is int and telemetry["latency_ms"] >= 0
    assert "secret-test" not in repr(reply)


def test_openai_sdk_retry_is_disabled_for_gateway_owned_failover(monkeypatch):
    import openai

    captured = {}
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: captured.update(kwargs) or object())
    provider = OpenAIReasoningProvider(_settings(openai_api_key=SecretStr("test-only")))

    provider._get_client()

    assert captured["max_retries"] == 0
    assert captured["timeout"] == 30


@pytest.mark.parametrize("exc,expected", [
    (TimeoutError("secret timeout"), ProviderTimeout),
])
def test_adapter_maps_request_errors_to_sanitized_categories(exc, expected):
    class Responses:
        def parse(self, **_kwargs):
            raise exc

    provider = OpenAIReasoningProvider(
        _settings(openai_api_key=SecretStr("secret-test")),
        client=SimpleNamespace(responses=Responses()),
    )
    with pytest.raises(expected) as caught:
        provider.generate(_prompt())
    assert "secret" not in str(caught.value)


def test_provider_configuration_is_bounded_and_base_url_must_be_safe():
    with pytest.raises(ValidationError):
        _settings(reasoning_timeout_seconds=0)
    with pytest.raises(ValidationError):
        _settings(reasoning_provider_order="groq,groq")
    with pytest.raises(ValidationError):
        _settings(reasoning_provider_order="groq,openai,unknown")
    with pytest.raises(ValidationError):
        _settings(reasoning_max_input_bytes=0)
    with pytest.raises(ValidationError):
        _settings(reasoning_openai_base_url="http://example.com")
    with pytest.raises(ValidationError):
        _settings(reasoning_openai_base_url="https://user:pass@example.com")
    assert _settings(reasoning_openai_base_url="http://localhost:8080/v1/").reasoning_openai_base_url == "http://localhost:8080/v1"


def test_oversized_prompt_is_rejected_before_client_initialization():
    provider = OpenAIReasoningProvider(
        _settings(reasoning_max_input_bytes=1024, openai_api_key=SecretStr("secret-test")),
    )
    from app.reasoning.errors import ProviderContextTooLarge
    with pytest.raises(ProviderContextTooLarge, match="configured input limit"):
        provider.generate(SimpleNamespace(
            system_instructions="instructions", context_json="x" * 1100, question="question",
            output_schema_json="{}", deterministic_payload=lambda: "x" * 1100,
        ))
    assert provider._client is None


def test_openai_exception_categories_are_sanitized():
    import httpx2
    from openai import APITimeoutError, AuthenticationError, RateLimitError

    cases = (
        (APITimeoutError(request=httpx2.Request("POST", "https://example.test")), ProviderTimeout),
        (AuthenticationError("secret auth body", response=httpx2.Response(401, request=httpx2.Request("POST", "https://example.test")), body=None), ProviderAuthenticationError),
        (RateLimitError("secret rate body", response=httpx2.Response(429, request=httpx2.Request("POST", "https://example.test")), body=None), ProviderRateLimited),
    )
    for exc, expected in cases:
        class Responses:
            def parse(self, **_kwargs):
                raise exc

        provider = OpenAIReasoningProvider(
            _settings(openai_api_key=SecretStr("secret-test")),
            client=SimpleNamespace(responses=Responses()),
        )
        with pytest.raises(expected) as caught:
            provider.generate(_prompt())
        assert "secret" not in str(caught.value)
