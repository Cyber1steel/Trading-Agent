"""Errors raised while defining or evaluating deterministic setups."""


class StrategyError(ValueError):
    """Base class for invalid strategy definitions and evaluation inputs."""


class StrategyDefinitionError(StrategyError):
    """A definition is internally inconsistent or incompatible with analysis."""


class StrategyEvaluationError(StrategyError):
    """An evaluation request does not identify a valid deterministic replay."""
