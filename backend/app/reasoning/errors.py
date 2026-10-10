"""Errors raised at the reasoning boundary."""

from enum import Enum


class ProviderFailureKind(str, Enum):
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    TEMPORARY_UNAVAILABLE = "TEMPORARY_UNAVAILABLE"
    SERVER_ERROR = "SERVER_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CAPABILITY_UNSUPPORTED = "CAPABILITY_UNSUPPORTED"
    INVALID_PROVIDER_RESPONSE = "INVALID_PROVIDER_RESPONSE"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    CONFIGURATION_FAILURE = "CONFIGURATION_FAILURE"
    UNKNOWN_PROVIDER_FAILURE = "UNKNOWN_PROVIDER_FAILURE"


class ReasoningError(Exception):
    """Base class for controlled reasoning failures."""


class ReasoningContextError(ReasoningError):
    """The supplied deterministic artifacts cannot safely form a context."""


class ReasoningValidationError(ReasoningError):
    """A provider response failed the deterministic output boundary."""


class ProviderError(ReasoningError):
    """A provider could not return a usable response."""

    failure_kind = ProviderFailureKind.UNKNOWN_PROVIDER_FAILURE
    retryable = False


class ProviderTimeout(ProviderError):
    """The provider exceeded its allowed request time."""

    failure_kind = ProviderFailureKind.TIMEOUT
    retryable = True


class ProviderUnavailable(ProviderError):
    """The provider is unavailable or not configured."""

    failure_kind = ProviderFailureKind.TEMPORARY_UNAVAILABLE
    retryable = True


class ProviderContextTooLarge(ProviderError):
    """The provider rejected the prompt for exceeding context limits."""

    failure_kind = ProviderFailureKind.REQUEST_TOO_LARGE


class ProviderRateLimited(ProviderError):
    """The provider rejected the request due to rate or quota limits."""

    failure_kind = ProviderFailureKind.RATE_LIMITED
    retryable = True


class ProviderQuotaExhausted(ProviderRateLimited):
    """The provider account exhausted an applicable quota."""

    failure_kind = ProviderFailureKind.QUOTA_EXHAUSTED


class ProviderAuthenticationError(ProviderError):
    """The provider rejected configured credentials."""

    failure_kind = ProviderFailureKind.AUTHENTICATION_FAILURE


class ProviderConfigurationError(ProviderError):
    """The provider request or local provider configuration is invalid."""

    failure_kind = ProviderFailureKind.CONFIGURATION_FAILURE


class ProviderModelUnavailable(ProviderError):
    """The configured model is unavailable at the provider."""

    failure_kind = ProviderFailureKind.MODEL_UNAVAILABLE
    retryable = True


class ProviderServerError(ProviderError):
    """The provider returned a server-side failure."""

    failure_kind = ProviderFailureKind.SERVER_ERROR
    retryable = True


class ProviderTemporaryUnavailable(ProviderUnavailable):
    """The provider's service or connection is temporarily unavailable."""

    failure_kind = ProviderFailureKind.TEMPORARY_UNAVAILABLE
    retryable = True


class ProviderInvalidResponse(ProviderError):
    """The provider response is malformed or unusable."""

    failure_kind = ProviderFailureKind.INVALID_PROVIDER_RESPONSE


class GatewayFailure(ProviderError):
    """No configured, eligible provider could service this request."""

    def __init__(self, message, *, failure_kind, telemetry=(), provider_id="gateway", model_id="unavailable"):
        super().__init__(message)
        self.failure_kind = failure_kind
        self.telemetry = tuple(telemetry)
        self.provider_id = provider_id
        self.model_id = model_id
