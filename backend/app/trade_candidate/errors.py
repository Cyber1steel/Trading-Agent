"""Errors raised while assembling deterministic trade candidates."""


class TradeCandidateError(ValueError):
    """Base exception for malformed candidate assembly requests."""


class CandidateInputError(TradeCandidateError):
    """Raised when an input is not one of the supported immutable contracts."""
