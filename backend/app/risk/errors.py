"""Risk-engine-specific exceptions."""


class RiskEngineError(ValueError):
    """Base exception for deterministic risk validation failures."""

    def __init__(self, *args, rule: str | None = None):
        if len(args) == 2 and rule is None:
            rule, message = args
        elif len(args) == 1:
            message = args[0]
        elif len(args) == 0:
            message = ""
        else:
            message = " ".join(str(arg) for arg in args)
        super().__init__(message)
        self.rule = rule
        self.message = message


class InsufficientEvidenceError(RiskEngineError):
    """Required inputs for risk evaluation are missing or ambiguous."""


class RiskRejectedError(RiskEngineError):
    """The candidate was evaluated fully but violates a configured rule."""
