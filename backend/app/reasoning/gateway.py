"""Deterministic, bounded routing across configured reasoning providers."""

from collections.abc import Mapping
from time import monotonic

from app.reasoning.errors import (
    GatewayFailure,
    ProviderError,
    ProviderFailureKind,
    ProviderInvalidResponse,
)
from app.reasoning.provider import (
    ProviderAvailability,
    ProviderCapabilities,
    ProviderReply,
    prompt_input_bytes,
)


class ReasoningProviderGateway:
    """Use the configured deterministic order and fail over at most once."""

    provider_id = "reasoning-gateway"
    model_id = "configured-provider-order"

    def __init__(self, providers: Mapping[str, object], provider_order, *, required_max_output_tokens=1):
        order = tuple(provider_order)
        if not order or len(order) > 2 or len(set(order)) != len(order):
            raise ValueError("gateway requires one or two unique providers in deterministic order")
        if any(name not in providers for name in order):
            raise ValueError("gateway order refers to an unregistered provider")
        self._providers = dict(providers)
        self.provider_order = order
        self.required_max_output_tokens = required_max_output_tokens

    def generate(self, prompt) -> ProviderReply:
        input_size = prompt_input_bytes(prompt)
        request_started = monotonic()
        attempted = []
        skipped = []
        eligible = []
        last_error = None
        attempt_latencies = []

        for name in self.provider_order:
            provider = self._providers[name]
            try:
                capabilities = provider.capabilities
            except Exception:
                capabilities = None
            reason = self._ineligibility(
                capabilities, prompt, input_size, self.required_max_output_tokens,
            )
            if reason is None and capabilities.provider_id != name:
                reason = ProviderFailureKind.CONFIGURATION_FAILURE
            if reason is None:
                eligible.append((name, provider, capabilities))
            else:
                skipped.append((name, reason))

        for name, provider, capabilities in eligible:
            attempt_started = monotonic()
            try:
                reply = provider.generate(prompt)
            except ProviderError as exc:
                attempted.append((name, exc.failure_kind))
                attempt_latencies.append(max(0, int((monotonic() - attempt_started) * 1000)))
                last_error = exc
                if not exc.retryable:
                    raise self._failure(
                        name, capabilities.model_id, exc, attempted, skipped, attempt_latencies,
                    ) from None
                continue
            except Exception:
                # Unknown exceptions are terminal. Never retry a request whose failure
                # category is unclear, and never surface vendor exception text.
                error = ProviderError("provider failed with an unknown error")
                attempted.append((name, error.failure_kind))
                attempt_latencies.append(max(0, int((monotonic() - attempt_started) * 1000)))
                last_error = error
                raise self._failure(name, capabilities.model_id, error, attempted, skipped,
                                    attempt_latencies) from None

            if not isinstance(reply, ProviderReply):
                error = ProviderInvalidResponse("provider returned an invalid response envelope")
                attempted.append((name, error.failure_kind))
                attempt_latencies.append(max(0, int((monotonic() - attempt_started) * 1000)))
                last_error = error
                raise self._failure(name, capabilities.model_id, error, attempted, skipped,
                                    attempt_latencies) from None
            if reply.provider_id != capabilities.provider_id or reply.model_id != capabilities.model_id:
                error = ProviderInvalidResponse("provider identity did not match its configured adapter")
                attempted.append((name, error.failure_kind))
                attempt_latencies.append(max(0, int((monotonic() - attempt_started) * 1000)))
                last_error = error
                raise self._failure(name, capabilities.model_id, error, attempted, skipped,
                                    attempt_latencies) from None
            telemetry = dict(reply.telemetry)
            attempt_latencies.append(max(0, int((monotonic() - attempt_started) * 1000)))
            telemetry.update(self._route_telemetry(
                attempted, skipped, name, include_final=True, attempt_latencies=attempt_latencies,
            ))
            telemetry["gateway_latency_ms"] = max(0, int((monotonic() - request_started) * 1000))
            return ProviderReply(
                provider_id=reply.provider_id,
                model_id=reply.model_id,
                raw_response=reply.raw_response,
                request_metadata=reply.request_metadata,
                telemetry=tuple(sorted(telemetry.items())),
            )

        if attempted:
            name, kind = attempted[-1]
            provider = self._providers[name]
            capabilities = provider.capabilities
            last_error.failure_kind = kind
            raise self._failure(name, capabilities.model_id, last_error, attempted, skipped,
                                attempt_latencies)

        reasons = {reason for _, reason in skipped}
        if ProviderFailureKind.REQUEST_TOO_LARGE in reasons:
            kind = ProviderFailureKind.REQUEST_TOO_LARGE
        elif ProviderFailureKind.CAPABILITY_UNSUPPORTED in reasons:
            kind = ProviderFailureKind.CAPABILITY_UNSUPPORTED
        elif ProviderFailureKind.TEMPORARY_UNAVAILABLE in reasons:
            kind = ProviderFailureKind.TEMPORARY_UNAVAILABLE
        else:
            kind = ProviderFailureKind.CONFIGURATION_FAILURE
        raise GatewayFailure(
            "no configured provider supports this reasoning request",
            failure_kind=kind,
            telemetry=tuple(sorted((*self._route_telemetry(attempted, skipped, "unavailable"),
                                    ("terminal_failure_kind", kind.value)))),
        )

    @staticmethod
    def _ineligibility(capabilities, prompt, input_size, required_max_output_tokens):
        if not isinstance(capabilities, ProviderCapabilities):
            return ProviderFailureKind.CONFIGURATION_FAILURE
        if capabilities.availability is ProviderAvailability.TEMPORARILY_UNAVAILABLE:
            return ProviderFailureKind.TEMPORARY_UNAVAILABLE
        if capabilities.availability is not ProviderAvailability.AVAILABLE:
            return ProviderFailureKind.CONFIGURATION_FAILURE
        if not capabilities.structured_output:
            return ProviderFailureKind.CAPABILITY_UNSUPPORTED
        if prompt.contract_version not in capabilities.supported_contract_versions:
            return ProviderFailureKind.CAPABILITY_UNSUPPORTED
        if input_size > capabilities.max_input_bytes:
            return ProviderFailureKind.REQUEST_TOO_LARGE
        if capabilities.max_output_tokens < required_max_output_tokens:
            return ProviderFailureKind.CAPABILITY_UNSUPPORTED
        return None

    @staticmethod
    def _route_telemetry(attempted, skipped, final_provider, *, include_final=False,
                         attempt_latencies=()):
        values = {
            "provider_attempt_count": len(attempted) + int(include_final),
            "failover_count": len(attempted) if include_final else max(0, len(attempted) - 1),
            "final_provider": final_provider,
        }
        for index, (name, kind) in enumerate(attempted, start=1):
            values[f"attempt_{index}_provider"] = name
            values[f"attempt_{index}_failure"] = kind.value
        for index, latency_ms in enumerate(attempt_latencies, start=1):
            values[f"attempt_{index}_latency_ms"] = latency_ms
        for index, (name, kind) in enumerate(skipped, start=1):
            values[f"skipped_{index}_provider"] = name
            values[f"skipped_{index}_reason"] = kind.value
        return tuple(sorted(values.items()))

    def _failure(self, name, model_id, error, attempted, skipped, attempt_latencies=()):
        failure = GatewayFailure(
            "configured reasoning providers could not service the request",
            failure_kind=error.failure_kind,
            telemetry=tuple(sorted((
                *self._route_telemetry(
                    attempted, skipped, name, attempt_latencies=attempt_latencies,
                ),
                ("terminal_failure_kind", error.failure_kind.value),
                ("gateway_latency_ms", sum(attempt_latencies)),
            ))),
            provider_id=name,
            model_id=model_id,
        )
        failure.failure_code = getattr(error, "failure_code", type(error).__name__)
        return failure
