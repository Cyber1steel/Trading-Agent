"""In-memory deterministic fixture provider; it never makes network requests."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from uuid import UUID

from app.market_data.contracts import (
    MarketDataRequest,
    PriceBasis,
    ProviderBatch,
    ProviderCandle,
    TimestampConvention,
)
from app.market_data.errors import MarketDataError
from app.market_data.timeframes import Timeframe


class LocalFixtureProvider:
    provider_id = "local-fixture"
    provider_version = "1"

    def __init__(
        self,
        fixtures: Mapping[tuple[UUID, Timeframe], Sequence[ProviderCandle]],
        *,
        retrieved_at: datetime = datetime(2000, 1, 1, tzinfo=timezone.utc),
        price_basis: PriceBasis = PriceBasis.TRADE,
        volume_units: str = "provider-defined",
    ) -> None:
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        self.fixtures = {
            (instrument_id, Timeframe.parse(timeframe)): tuple(candles)
            for (instrument_id, timeframe), candles in fixtures.items()
        }
        self.retrieved_at = retrieved_at.astimezone(timezone.utc)
        self.price_basis = price_basis
        self.volume_units = volume_units

    def fetch_historical(self, request: MarketDataRequest) -> ProviderBatch:
        key = (request.instrument.instrument_id, request.timeframe)
        if key not in self.fixtures:
            raise MarketDataError(f"No local fixture for {request.instrument.symbol}/{request.timeframe}")
        selected = tuple(
            candle for candle in self.fixtures[key]
            if candle.timestamp.tzinfo is None
            or candle.timestamp.utcoffset() is None
            or request.start <= candle.timestamp.astimezone(timezone.utc) < request.end
        )
        return ProviderBatch(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            provider_symbol=request.instrument.symbol,
            instrument=request.instrument,
            timeframe=request.timeframe,
            requested_start=request.start,
            requested_end=request.end,
            retrieved_at=self.retrieved_at,
            timestamp_convention=TimestampConvention.BAR_OPEN,
            price_basis=self.price_basis,
            volume_units=self.volume_units,
            provider_metadata={"kind": "local_fixture"},
            candles=selected,
        )
