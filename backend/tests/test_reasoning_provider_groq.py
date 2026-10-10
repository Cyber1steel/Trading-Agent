"""Offline unit tests for the Groq adapter and strict schema preparation."""

from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.reasoning.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderContextTooLarge,
    ProviderInvalidResponse,
    ProviderModelUnavailable,
    ProviderQuotaExhausted,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTemporaryUnavailable,
    ProviderTimeout,
)
from app.reasoning.provider import ProviderAvailability
from app.reasoning.providers.groq import (
    STRICT_STRUCTURED_MODELS,
    GroqReasoningProvider,
    groq_strict_schema,
)


def _settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


class Prompt:
    system_instructions = "canonical safety policy"
    question = "Explain supplied evidence."
    context_json = '{"evidence":"only as data"}'
    output_schema_json = '{"type":"object"}'
    contract_version = "3a-prompt.1.0"

    def deterministic_payload(self):
        return "canonical-complete-prompt"


def _client_with_response(response):
    calls = SimpleNamespace(kwargs=None)

    def create(**kwargs):
        calls.kwargs = kwargs
        return response

    calls.client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    ))
    return calls


def test_strict_schema_is_closed_required_and_contains_only_supported_schema_keywords():
    schema = groq_strict_schema()
    allowed = {
        "$defs", "$ref", "type", "properties", "required", "additionalProperties",
        "items", "enum", "anyOf", "description",
    }

    def inspect(node):
        assert isinstance(node, dict)
        assert set(node) <= allowed
        if node.get("type") == "object" or "properties" in node:
            props = node.get("properties", {})
            assert node.get("additionalProperties") is False
            assert set(node.get("required", ())) == set(props)
        for child in node.get("properties", {}).values():
            inspect(child)
        for child in node.get("$defs", {}).values():
            inspect(child)
        if isinstance(node.get("items"), dict):
            inspect(node["items"])
        for child in node.get("anyOf", ()):
            inspect(child)

    inspect(schema)
    assert "default" not in repr(schema)
    assert schema["required"] == sorted(schema["properties"])


def test_groq_strict_model_allowlist_is_explicit_and_provider_capabilities_are_typed():
    assert "openai/gpt-oss-120b" in STRICT_STRUCTURED_MODELS
    assert "unknown-model" not in STRICT_STRUCTURED_MODELS
    adapter = GroqReasoningProvider(_settings(groq_api_key=SecretStr("test-secret")))
    assert adapter.capabilities.structured_output is True
    assert adapter.capabilities.availability is ProviderAvailability.AVAILABLE
    assert adapter.capabilities.streaming is False
    assert adapter.capabilities.supported_contract_versions == ("3a-prompt.1.0",)
    unsupported = GroqReasoningProvider(_settings(
        groq_api_key=SecretStr("test-secret"), groq_model="unknown-model",
    ))
    assert unsupported.capabilities.structured_output is False
    assert unsupported._client is None


def test_success_uses_official_sdk_strict_schema_and_returns_safe_normalized_reply():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"valid":true}', refusal=None))],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=6, total_tokens=26),
    )
    stub = _client_with_response(response)
    adapter = GroqReasoningProvider(
        _settings(groq_api_key=SecretStr("do-not-leak")), client=stub.client,
    )

    reply = adapter.generate(Prompt())

    call = stub.kwargs
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["messages"][0]["content"] == Prompt.system_instructions
    assert "only as data" in call["messages"][1]["content"]
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["response_format"]["json_schema"]["schema"] == groq_strict_schema()
    assert call["max_completion_tokens"] == 1600
    assert call["stream"] is False
    assert reply.provider_id == "groq"
    assert reply.raw_response == '{"valid":true}'
    assert dict(reply.telemetry)["total_tokens"] == 26
    assert "do-not-leak" not in repr(reply)


def test_groq_operates_without_openai_credentials_or_openai_request():
    import json

    from app.reasoning.contracts import ReasoningRequest
    from app.reasoning.factory import create_reasoning_gateway
    from app.reasoning.service import ReasoningService
    from tests.test_reasoning import _context, _proposal

    settings = _settings(groq_api_key=SecretStr("test-secret"), openai_api_key=None)
    gateway = create_reasoning_gateway(settings)
    context = _context()
    payload = json.dumps(_proposal(context))
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload, refusal=None))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )
    gateway._providers["groq"]._client = _client_with_response(response).client

    result = ReasoningService(gateway).reason(
        ReasoningRequest(context=context, question="Explain supplied evidence."),
    )

    assert result.failure_code is None
    assert result.provider_id == "groq"
    assert gateway._providers["openai"]._client is None


def test_groq_client_is_created_lazily_with_zero_sdk_retries(monkeypatch):
    import groq

    captured = {}
    monkeypatch.setattr(groq, "Groq", lambda **kwargs: captured.update(kwargs) or object())
    adapter = GroqReasoningProvider(_settings(groq_api_key=SecretStr("test-secret")))

    adapter._get_client()

    assert captured["api_key"] == "test-secret"
    assert captured["timeout"] == 30
    assert captured["max_retries"] == 0


def test_missing_groq_key_and_oversized_prompt_fail_before_sdk_call():
    missing = GroqReasoningProvider(_settings(groq_api_key=None))
    assert missing.capabilities.availability is ProviderAvailability.CONFIGURATION_ERROR
    with pytest.raises(Exception, match="not configured"):
        missing._get_client()
    oversized = GroqReasoningProvider(_settings(
        groq_api_key=SecretStr("test-secret"), reasoning_max_input_bytes=1024,
    ))
    prompt = Prompt()
    prompt.deterministic_payload = lambda: "x" * 2048
    with pytest.raises(ProviderContextTooLarge):
        oversized.generate(prompt)
    assert oversized._client is None


@pytest.mark.parametrize("error,expected", [
    (TimeoutError("secret timeout"), ProviderTimeout),
    (__import__("groq").APITimeoutError(request=httpx.Request("POST", "https://api.groq.test")),
     ProviderTimeout),
])
def test_timeout_errors_are_sanitized(error, expected):
    adapter = GroqReasoningProvider(_settings(groq_api_key=SecretStr("test-secret")),
                                    client=_client_with_error(error))
    with pytest.raises(expected) as caught:
        adapter.generate(Prompt())
    assert "secret" not in str(caught.value)


def _client_with_error(error):
    def create(**_kwargs):
        raise error
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _status_error(error_type, status, body):
    request = httpx.Request("POST", "https://api.groq.test/chat/completions")
    response = httpx.Response(status, request=request)
    return error_type("secret provider response", response=response, body=body)


@pytest.mark.parametrize("error,expected", [
    (_status_error(__import__("groq").AuthenticationError, 401,
                   {"error": {"code": "invalid_api_key"}}), ProviderAuthenticationError),
    (_status_error(__import__("groq").RateLimitError, 429,
                   {"error": {"code": "rate_limit_exceeded"}}), ProviderRateLimited),
    (_status_error(__import__("groq").RateLimitError, 429,
                   {"error": {"code": "insufficient_quota"}}), ProviderQuotaExhausted),
    (_status_error(__import__("groq").APIStatusError, 404,
                   {"error": {"code": "model_not_found"}}), ProviderModelUnavailable),
    (_status_error(__import__("groq").APIStatusError, 500,
                   {"error": {"code": "internal"}}), ProviderServerError),
    (_status_error(__import__("groq").APIStatusError, 498,
                   {"error": {"code": "capacity_exceeded"}}), ProviderTemporaryUnavailable),
    (_status_error(__import__("groq").APIStatusError, 400,
                   {"error": {"code": "invalid_request"}}), ProviderConfigurationError),
])
def test_http_provider_failures_are_normalized_without_response_leak(error, expected):
    adapter = GroqReasoningProvider(_settings(groq_api_key=SecretStr("test-secret")),
                                    client=_client_with_error(error))
    with pytest.raises(expected) as caught:
        adapter.generate(Prompt())
    assert "secret" not in str(caught.value)
    assert "provider response" not in str(caught.value)


@pytest.mark.parametrize("response", [
    SimpleNamespace(choices=[]),
    SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal=None))]),
    SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}", refusal="policy"))]),
])
def test_empty_malformed_refused_provider_responses_are_rejected(response):
    adapter = GroqReasoningProvider(_settings(groq_api_key=SecretStr("test-secret")),
                                    client=_client_with_response(response).client)
    with pytest.raises(ProviderInvalidResponse):
        adapter.generate(Prompt())


def test_oversized_provider_output_is_rejected():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="x" * 65537, refusal=None,
    ))])
    adapter = GroqReasoningProvider(_settings(groq_api_key=SecretStr("test-secret")),
                                    client=_client_with_response(response).client)
    with pytest.raises(ProviderInvalidResponse):
        adapter.generate(Prompt())
