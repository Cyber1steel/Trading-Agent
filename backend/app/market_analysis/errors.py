"""Hard failures raised when an analysis request is unsafe or ambiguous."""


class MarketAnalysisError(ValueError):
    """Base class for deterministic market analysis failures."""


class AnalysisDataError(MarketAnalysisError):
    """The selected stored snapshot cannot support the requested analysis."""


class AnalysisCutoffError(MarketAnalysisError):
    """Requested output or context is not knowable by the declared cutoff."""
