"""Backtesting foundation (Phase 2G).

Deterministic inter-bar backtester that replays Phase 2E AnalysisResult and
Phase 2F StrategyDefinition/Evaluation through history. Execution fills occur
at the next primary bar-open to avoid intrabar assumptions.
"""
__all__ = [
    "contracts", "engine", "execution", "ledger", "metrics", "fingerprints",
]
