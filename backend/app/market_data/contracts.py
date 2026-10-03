"""Typed canonical and provider-boundary contracts for market data."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Literal
from uuid import UUID, uuid5, NAMESPACE_URL

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.market_data.quality import DataQualityReport
from app.market_data.timeframes import Timeframe


class AssetClass(str, Enum):
    EQUITY = "equity"
    FX = "fx"
    CRYPTO = "crypto"
    FUTURE = "future"
    OPTION = "option"
    INDEX = "index"
    OTHER = "other"


class TimestampConvention(str, Enum):
    BAR_OPEN = "bar_open"
    BAR_CLOSE = "bar_close"


class PriceBasis(str, Enum):
    TRADE = "trade"
    MID = "mid"
    BID = "bid"
    ASK = "ask"
    ADJUSTED = "adjusted"


def require_aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} cannot be represented in UTC") from exc


class MarketDataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def deterministic_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Instrument(MarketDataModel):
    symbol: str = Field(min_length=1, max_length=32)
    asset_class: AssetClass
    venue: str | None = Field(default=None, max_length=64)
    market: str | None = Field(default=None, max_length=64)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol or not re.fullmatch(r"[A-Z0-9][A-Z0-9._:/-]{0,31}", symbol):
            raise ValueError("Symbol must be 1-32 ASCII letters, digits, or . _ : / - characters")
        return symbol

    @field_validator("venue", "market")
    @classmethod
    def normalize_optional_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized or len(normalized) > 64:
            raise ValueError("Venue/market identity cannot be blank")
        return normalized

    @property
    def instrument_id(self) -> UUID:
        identity = "|".join((self.asset_class.value, self.venue or "", self.market or "", self.symbol))
        return uuid5(NAMESPACE_URL, f"trading-agent:instrument:{identity}")


class ProviderSymbolMapping(MarketDataModel):
    provider_id: str = Field(min_length=1, max_length=64)
    provider_symbol: str = Field(min_length=1, max_length=128)
    instrument: Instrument

    @field_validator("provider_id", "provider_symbol")
    @classmethod
    def nonblank_mapping_values(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Provider identity and symbol cannot be blank")
        return normalized


class MarketDataRequest(MarketDataModel):
    instrument: Instrument
    timeframe: Timeframe
    start: datetime
    end: datetime

    @field_validator("timeframe", mode="before")
    @classmethod
    def parse_timeframe(cls, value: str | Timeframe) -> Timeframe:
        return Timeframe.parse(value)

    @field_validator("start", "end")
    @classmethod
    def utc_bounds(cls, value: datetime, info) -> datetime:
        return require_aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def valid_range(self) -> "MarketDataRequest":
        if self.start >= self.end:
            raise ValueError("Requested range must satisfy start < end")
        return self


class ProviderCandle(MarketDataModel):
    """Typed provider row; row-level errors are reported by normalization."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    bid: Decimal | None = None
    ask: Decimal | None = None
    spread: Decimal | None = None


class ProviderBatch(MarketDataModel):
    provider_id: str = Field(min_length=1, max_length=64)
    provider_version: str = Field(min_length=1, max_length=128)
    provider_symbol: str = Field(min_length=1, max_length=128)
    instrument: Instrument
    timeframe: Timeframe
    requested_start: datetime
    requested_end: datetime
    retrieved_at: datetime
    timestamp_convention: TimestampConvention
    price_basis: PriceBasis
    volume_units: str = Field(min_length=1, max_length=64)
    provider_metadata: dict[str, str | int | bool | None] = Field(default_factory=dict)
    candles: tuple[ProviderCandle, ...]

    @field_validator("provider_id", "provider_version", "provider_symbol", "volume_units")
    @classmethod
    def nonblank_provider_metadata(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Provider metadata values cannot be blank")
        return normalized

    @field_validator("timeframe", mode="before")
    @classmethod
    def parse_timeframe(cls, value: str | Timeframe) -> Timeframe:
        return Timeframe.parse(value)

    @field_validator("requested_start", "requested_end", "retrieved_at")
    @classmethod
    def utc_metadata_timestamps(cls, value: datetime, info) -> datetime:
        return require_aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def valid_range(self) -> "ProviderBatch":
        if self.requested_start >= self.requested_end:
            raise ValueError("Provider range must satisfy start < end")
        return self


class Candle(MarketDataModel):
    """Canonical OHLCV bar; timestamp is UTC-aware and denotes bar open."""

    instrument: Instrument
    timeframe: Timeframe
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    # bid/ask are provider-supplied quote snapshots at bar open. spread is the
    # provider's spread observation for that instant (validated against both).
    bid: Decimal | None = None
    ask: Decimal | None = None
    spread: Decimal | None = None

    @field_validator("timeframe", mode="before")
    @classmethod
    def parse_timeframe(cls, value: str | Timeframe) -> Timeframe:
        return Timeframe.parse(value)

    @field_validator("timestamp")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        return require_aware_utc(value, "timestamp")

    @model_validator(mode="after")
    def valid_values(self) -> "Candle":
        prices = (self.open, self.high, self.low, self.close)
        if not all(value.is_finite() for value in prices):
            raise ValueError("OHLC values must be finite")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close) or self.high < self.low:
            raise ValueError("OHLC relationship is invalid")
        if not self.volume.is_finite() or self.volume < 0:
            raise ValueError("Volume must be finite and non-negative")
        for name, value in (("bid", self.bid), ("ask", self.ask), ("spread", self.spread)):
            if value is not None and not value.is_finite():
                raise ValueError(f"{name} must be finite")
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError("bid cannot exceed ask")
        if self.spread is not None and self.spread < 0:
            raise ValueError("spread cannot be negative")
        if self.bid is not None and self.ask is not None and self.spread is not None:
            if self.spread != self.ask - self.bid:
                raise ValueError("spread must equal ask - bid when all quote values are present")
        return self


class NormalizationResult(MarketDataModel):
    candles: tuple[Candle, ...]
    quality: DataQualityReport
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class MarketDataSlice(MarketDataModel):
    manifest: "MarketDatasetManifest"
    candles: tuple[Candle, ...]


class MarketDatasetManifest(MarketDataModel):
    dataset_id: UUID
    dataset_version: int = Field(ge=1)
    instrument: Instrument
    timeframe: Timeframe
    provider_id: str
    provider_version: str
    provider_symbol: str
    requested_start: datetime
    requested_end: datetime
    retrieved_at: datetime
    ingested_at: datetime
    timestamp_convention: TimestampConvention
    price_basis: PriceBasis
    volume_units: str
    normalization_version: str
    validation_version: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality: DataQualityReport
    provider_metadata: dict[str, str | int | bool | None] = Field(default_factory=dict)

    @field_validator(
        "provider_id", "provider_version", "provider_symbol", "volume_units",
        "normalization_version", "validation_version",
    )
    @classmethod
    def nonblank_manifest_metadata(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Manifest metadata values cannot be blank")
        return normalized

    @field_validator("requested_start", "requested_end", "retrieved_at", "ingested_at")
    @classmethod
    def utc_metadata_timestamps(cls, value: datetime, info) -> datetime:
        return require_aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def valid_range(self) -> "MarketDatasetManifest":
        if self.requested_start >= self.requested_end:
            raise ValueError("Manifest range must satisfy start < end")
        return self


def normalized_content_hash(candles: tuple[Candle, ...]) -> str:
    """Hash canonical candle content in its already-validated deterministic order."""
    serialized = json.dumps(
        [candle.model_dump(mode="json") for candle in candles],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


MarketDataSlice.model_rebuild()
