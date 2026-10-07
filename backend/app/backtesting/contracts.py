import re
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from app.market_data.contracts import MarketDatasetManifest, MarketDataModel

_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class PriceBasis(str, Enum):
    TRADE = "trade"
    BID = "bid"
    ASK = "ask"


class ExecutionAssumptions(MarketDataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    spread: Decimal = Field(...)
    slippage: Decimal = Field(...)
    pct_fee: Decimal | None = None
    fixed_fee: Decimal | None = None
    price_basis: PriceBasis = PriceBasis.TRADE
    fixed_quantity: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_decimals(self):
        if not self.spread.is_finite() or self.spread < 0:
            raise ValueError("spread must be non-negative finite Decimal")
        if not self.slippage.is_finite() or self.slippage < 0:
            raise ValueError("slippage must be non-negative finite Decimal")
        if self.pct_fee is not None and (not self.pct_fee.is_finite() or self.pct_fee < 0):
            raise ValueError("pct_fee must be non-negative Decimal")
        if self.fixed_fee is not None and (not self.fixed_fee.is_finite() or self.fixed_fee < 0):
            raise ValueError("fixed_fee must be non-negative Decimal")
        return self


class BacktestRequest(MarketDataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: UUID
    dataset_version: int = Field(..., ge=1, strict=True)
    instrument_id: UUID
    timeframe: str
    strategy_id: UUID
    strategy_version: int = Field(..., ge=1, strict=True)
    setup_id: str
    setup_version: int = Field(..., ge=1, strict=True)
    strategy_fingerprint: str = Field(...)
    setup_fingerprint: str = Field(...)
    analysis_fingerprint: str = Field(...)
    analysis_prefix_fingerprint: str = Field(...)
    start: datetime
    end: datetime
    initial_capital: Decimal
    execution: ExecutionAssumptions
    engine_version: str = Field(default="2g.1.0")

    @staticmethod
    def _require_sha256_hex(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value):
            raise ValueError(f"{field_name} must be a 64-character lowercase hex SHA-256 fingerprint")
        return value

    @field_validator("strategy_fingerprint", "setup_fingerprint", "analysis_fingerprint", "analysis_prefix_fingerprint")
    @classmethod
    def validate_fingerprints(cls, value, info):
        return cls._require_sha256_hex(value, info.field_name)

    @field_validator("start", "end")
    def require_utc(cls, value: datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("start/end must be timezone-aware UTC datetimes")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if not self.initial_capital.is_finite() or self.initial_capital <= 0:
            raise ValueError("initial_capital must be a positive Decimal")
        return self


class TradeRecord(MarketDataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    trade_id: str
    strategy_id: UUID
    setup_id: str
    entry_bar_open: datetime
    entry_price: Decimal
    entry_quantity: int
    exit_bar_open: datetime | None = None
    exit_price: Decimal | None = None
    gross_pnl: Decimal | None = None
    fees: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    net_pnl: Decimal | None = None
    return_pct: Decimal | None = None
    exit_reason: str | None = None
    request_fingerprint: str | None = None


class BacktestResult(MarketDataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request: BacktestRequest
    manifest: MarketDatasetManifest
    engine_version: str
    execution: ExecutionAssumptions
    starting_capital: Decimal
    ending_equity: Decimal
    trades: tuple[TradeRecord, ...]
    events: tuple[dict, ...]
    metrics: dict
    warnings: tuple[str, ...] = ()
    result_fingerprint: str
