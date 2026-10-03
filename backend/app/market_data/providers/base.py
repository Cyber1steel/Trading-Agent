"""Provider interface: vendor adapters return typed raw batches, never canonical rows."""

from typing import Protocol

from app.market_data.contracts import MarketDataRequest, ProviderBatch


class HistoricalMarketDataProvider(Protocol):
    provider_id: str
    provider_version: str

    def fetch_historical(self, request: MarketDataRequest) -> ProviderBatch:
        """Return data for the requested instrument, timeframe, and UTC [start, end)."""
        ...
