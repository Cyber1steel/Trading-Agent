"""Frozen, explicit-input contracts for deterministic market analysis."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from app.market_data.contracts import (
    Instrument, MarketDataModel, MarketDatasetManifest, PriceBasis,
    TimestampConvention, require_aware_utc,
)
from app.market_data.timeframes import Timeframe
from app.market_context.sessions import SessionClassification
from app.market_data.quality import DataQualityIssue

MAX_PERIOD = 1000
ANALYSIS_VERSION = "2e.1.0"


class DatasetRef(MarketDataModel):
    dataset_id: UUID
    dataset_version: int = Field(ge=1, strict=True)
    instrument: Instrument
    timeframe: Timeframe

    @field_validator("timeframe", mode="before")
    @classmethod
    def parse_timeframe(cls, value):
        return Timeframe.parse(value)


class AnalysisParameters(MarketDataModel):
    atr_period: int = Field(default=14, ge=1, le=MAX_PERIOD)
    swing_left: int = Field(default=2, ge=1, le=MAX_PERIOD)
    swing_right: int = Field(default=2, ge=1, le=MAX_PERIOD)


class AnalysisRequest(MarketDataModel):
    primary: DatasetRef
    higher_timeframes: tuple[DatasetRef, ...] = ()
    input_start: datetime
    analysis_start: datetime
    analysis_end: datetime
    cutoff_at: datetime
    parameters: AnalysisParameters = Field(default_factory=AnalysisParameters)

    @field_validator("input_start", "analysis_start", "analysis_end", "cutoff_at")
    @classmethod
    def utc_times(cls, value, info):
        return require_aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def valid_request(self):
        if not self.input_start <= self.analysis_start < self.analysis_end:
            raise ValueError("Require input_start <= analysis_start < analysis_end")
        refs = (self.primary, *self.higher_timeframes)
        timeframes = [ref.timeframe for ref in refs]
        if len(timeframes) != len(set(timeframes)):
            raise ValueError("Dataset timeframe references must be unique")
        for ref in refs:
            if ref.timeframe.is_calendar_anchored:
                raise ValueError("Daily and weekly calendar-anchored analysis is unsupported")
        base = self.primary.timeframe.nominal_duration
        for parent in self.higher_timeframes:
            parent_duration = parent.timeframe.nominal_duration
            if parent_duration < base or parent_duration % base:
                raise ValueError("Higher timeframe duration must be an integer multiple of primary")
        return self


class AvailabilityReason(str, Enum):
    ZERO_RANGE = "zero_range"
    NO_PREVIOUS_CLOSE = "no_previous_close"
    NON_POSITIVE_CLOSE = "non_positive_close"
    INSUFFICIENT_HISTORY = "insufficient_history"
    NO_CLOSED_PARENT_BAR = "no_closed_parent_bar"


class DecimalValue(MarketDataModel):
    value: Decimal | None
    unavailable_reason: AvailabilityReason | None = None

    @model_validator(mode="after")
    def consistent(self):
        if (self.value is None) == (self.unavailable_reason is None):
            raise ValueError("Provide exactly one of value or unavailable_reason")
        return self


class CandleFeatures(MarketDataModel):
    range: Decimal
    body: Decimal
    upper_wick: Decimal
    lower_wick: Decimal
    body_ratio: DecimalValue
    close_location: DecimalValue
    simple_return: DecimalValue


class SwingKind(str, Enum):
    HIGH = "high"
    LOW = "low"


class SwingComparison(str, Enum):
    HIGHER_HIGH = "higher_high"
    LOWER_HIGH = "lower_high"
    EQUAL_HIGH = "equal_high"
    HIGHER_LOW = "higher_low"
    LOWER_LOW = "lower_low"
    EQUAL_LOW = "equal_low"


class StructuralState(str, Enum):
    UP = "up_structure"
    DOWN = "down_structure"
    MIXED = "mixed_structure"
    INSUFFICIENT = "insufficient_structure"


class StructuralTransition(MarketDataModel):
    previous_state: StructuralState
    new_state: StructuralState
    known_at: datetime
    responsible_swing: SwingKind


class SessionTransition(MarketDataModel):
    entered: tuple[str, ...]
    exited: tuple[str, ...]


class ConfirmedSwing(MarketDataModel):
    kind: SwingKind
    candidate_at: datetime
    confirmed_at: datetime
    known_at: datetime
    price: Decimal
    comparison: SwingComparison | None = None


class DatasetProvenance(MarketDataModel):
    dataset_id: UUID
    dataset_version: int
    instrument: Instrument
    timeframe: Timeframe
    content_hash: str
    provider_id: str
    provider_version: str
    provider_symbol: str
    price_basis: PriceBasis
    timestamp_convention: TimestampConvention
    quality_warnings: tuple[DataQualityIssue, ...] = ()

    @classmethod
    def from_manifest(cls, manifest: MarketDatasetManifest):
        return cls(
            dataset_id=manifest.dataset_id, dataset_version=manifest.dataset_version,
            instrument=manifest.instrument, timeframe=manifest.timeframe,
            content_hash=manifest.content_hash, provider_id=manifest.provider_id,
            provider_version=manifest.provider_version, provider_symbol=manifest.provider_symbol,
            price_basis=manifest.price_basis,
            timestamp_convention=manifest.timestamp_convention,
            quality_warnings=manifest.quality.warnings,
        )


class HigherTimeframeContext(MarketDataModel):
    timeframe: Timeframe
    bar_open: datetime | None
    known_at: datetime
    close: Decimal | None
    structural_state: StructuralState
    confirmed_swings: tuple[ConfirmedSwing, ...] = ()
    unavailable_reason: AvailabilityReason | None = None


class MarketObservation(MarketDataModel):
    bar_open: datetime
    known_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    bid: Decimal | None = None
    ask: Decimal | None = None
    spread: Decimal | None = None
    volume_units: str
    features: CandleFeatures
    true_range: Decimal
    atr: DecimalValue
    structural_state: StructuralState
    confirmed_swings: tuple[ConfirmedSwing, ...]
    structural_transition: StructuralTransition | None
    higher_timeframes: tuple[HigherTimeframeContext, ...]
    session: SessionClassification
    session_transition: SessionTransition | None


class SelectedInput(MarketDataModel):
    provenance: DatasetProvenance
    slice_hash: str


class AnalysisResult(MarketDataModel):
    analysis_version: str
    parameters: AnalysisParameters
    input_start: datetime
    analysis_start: datetime
    analysis_end: datetime
    cutoff_at: datetime
    inputs: tuple[SelectedInput, ...]
    observations: tuple[MarketObservation, ...]
    fingerprint: str


def normalized_hash(value: object) -> str:
    """Hash stable contract JSON (Pydantic serializes Decimal and datetime explicitly)."""
    def normalize(item):
        if isinstance(item, MarketDataModel):
            return item.model_dump(mode="json")
        if isinstance(item, Decimal):
            return str(item)
        if isinstance(item, datetime):
            return require_aware_utc(item, "hash datetime").isoformat().replace("+00:00", "Z")
        if isinstance(item, UUID):
            return str(item)
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, dict):
            return {key: normalize(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(val) for val in item]
        return item
    payload = normalize(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()
