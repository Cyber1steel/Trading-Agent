"""Errors raised at the reasoning boundary."""


class ReasoningError(Exception):
    """Base class for controlled reasoning failures."""


class ReasoningContextError(ReasoningError):
    """The supplied deterministic artifacts cannot safely form a context."""


class ReasoningValidationError(ReasoningError):
    """A provider response failed the deterministic output boundary."""


class ProviderError(ReasoningError):
    """A provider could not return a usable response."""


class ProviderTimeout(ProviderError):
    """The provider exceeded its allowed request time."""


class ProviderUnavailable(ProviderError):
    """The provider is unavailable or not configured."""


class ProviderContextTooLarge(ProviderError):
    """The provider rejected the prompt for exceeding context limits."""


class ProviderRateLimited(ProviderError):
    """The provider rejected the request due to rate or quota limits."""


class ProviderAuthenticationError(ProviderError):
    """The provider rejected configured credentials."""
