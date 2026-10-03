"""Errors for deterministic market-data ingestion and persistence."""

from app.market_data.quality import DataQualityReport


class MarketDataError(RuntimeError):
    pass


class MarketDataValidationError(MarketDataError):
    def __init__(self, report: DataQualityReport):
        self.report = report
        super().__init__(f"Market data rejected with {len(report.hard_failures)} hard failure(s)")


class DatasetNotFound(MarketDataError):
    pass


class DatasetAlreadyExists(MarketDataError):
    pass


class InstrumentIdentityConflict(MarketDataError):
    pass
