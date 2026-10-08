import re
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, StrictInt, field_validator, model_validator
from app.execution.assumptions import ExecutionAssumptions, PriceBasis
from app.market_data.contracts import MarketDatasetManifest, MarketDataModel
from app.market_data.timeframes import Timeframe
from app.core.immutability import FrozenDict, deep_freeze

_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class TradeSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class BacktestRequest(MarketDataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: UUID
    dataset_version: int = Field(..., ge=1, strict=True)
    instrument_id: UUID
    side: TradeSide
    timeframe: str
    strategy_id: UUID
    strategy_version: int = Field(..., ge=1, strict=True)
    setup_id: str
    setup_version: int = Field(..., ge=1, strict=True)
    strategy_fingerprint: str = Field(...)
    setup_fingerprint: str = Field(...)
    analysis_fingerprint: str = Field(...)
    analysis_prefix_fingerprint: str = Field(...)
    pnl_to_account_rate: Decimal = Field(..., strict=True)
    start: datetime
    end: datetime
    initial_capital: Decimal = Field(..., strict=True)
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
        if not self.pnl_to_account_rate.is_finite() or self.pnl_to_account_rate <= 0:
            raise ValueError("pnl_to_account_rate must be a positive finite Decimal")
        return self

    @property
    def replay_fingerprint(self) -> str:
        """Identify the request and visible analysis prefix without future suffix data."""
        from app.backtesting.fingerprints import fingerprint
        return fingerprint(self.model_dump(mode="python", exclude={"analysis_fingerprint"}),
                          domain="backtest-request-visible")

    @field_validator("timeframe")
    @classmethod
    def normalize_timeframe(cls, value):
        return Timeframe.parse(value).value


class TradeRecord(MarketDataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    trade_id: str
    strategy_id: UUID
    setup_id: str
    side: TradeSide
    entry_bar_open: datetime
    entry_reference_price: Decimal
    entry_price: Decimal
    entry_quantity: StrictInt = Field(ge=1)
    exit_bar_open: datetime | None = None
    exit_reference_price: Decimal | None = None
    exit_price: Decimal | None = None
    gross_pnl: Decimal | None = None
    fees: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    spread_cost: Decimal = Decimal("0")
    net_pnl: Decimal | None = None
    return_pct: Decimal | None = None
    exit_reason: str | None = None
    request_fingerprint: str | None = None

    @field_validator("request_fingerprint")
    @classmethod
    def valid_request_identity(cls, value):
        if value is not None and not _FINGERPRINT_RE.fullmatch(value):
            raise ValueError("request_fingerprint must be a lowercase SHA-256 fingerprint")
        return value

    @field_validator("entry_reference_price", "entry_price", "exit_reference_price", "exit_price",
                     "gross_pnl", "fees", "slippage", "spread_cost", "net_pnl", "return_pct", mode="before")
    @classmethod
    def finite_decimal_values(cls, value):
        if value is None:
            return value
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError("trade financial values must be finite Decimals")
        return value

    @model_validator(mode="after")
    def valid_trade_state(self):
        if self.entry_reference_price <= 0 or self.entry_price <= 0:
            raise ValueError("trade entry prices must be positive")
        for name in ("fees", "slippage", "spread_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        closed_values = (self.exit_bar_open, self.exit_reference_price, self.exit_price,
                         self.gross_pnl, self.net_pnl, self.exit_reason)
        if self.exit_bar_open is None and any(value is not None for value in closed_values[1:]):
            raise ValueError("an open trade cannot contain exit or realized-result fields")
        if self.exit_bar_open is not None:
            if any(value is None for value in closed_values[1:]):
                raise ValueError("a closed trade requires exit prices, realized PnL, and an exit reason")
            if self.exit_reference_price <= 0 or self.exit_price <= 0:
                raise ValueError("trade exit prices must be positive")
            if self.exit_bar_open < self.entry_bar_open:
                raise ValueError("trade exit cannot precede entry")
            if self.net_pnl != self.gross_pnl - self.fees - self.slippage - self.spread_cost:
                raise ValueError("net PnL must equal gross PnL less fees, slippage, and spread")
        return self

    @field_validator("entry_bar_open", "exit_bar_open")
    @classmethod
    def utc_trade_timestamps(cls, value):
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trade timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)


class BacktestResult(MarketDataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request: BacktestRequest
    manifest: MarketDatasetManifest
    engine_version: str
    execution: ExecutionAssumptions
    starting_capital: Decimal
    ending_equity: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    trades: tuple[TradeRecord, ...]
    events: tuple[dict, ...]
    metrics: dict
    warnings: tuple[str, ...] = ()
    result_fingerprint: str

    @field_validator("result_fingerprint")
    @classmethod
    def valid_result_identity(cls, value):
        return BacktestRequest._require_sha256_hex(value, "result_fingerprint")

    @field_validator("starting_capital", "ending_equity", "unrealized_pnl", mode="before")
    @classmethod
    def finite_decimal_outputs(cls, value):
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError("backtest monetary results must be finite Decimals")
        return value

    @model_validator(mode="after")
    def consistent_equity(self):
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        realized = sum((trade.net_pnl for trade in self.trades if trade.net_pnl is not None), Decimal(0))
        open_count = sum(trade.net_pnl is None for trade in self.trades)
        if open_count == 0 and self.unrealized_pnl != 0:
            raise ValueError("unrealized PnL requires an open trade")
        if self.ending_equity != self.starting_capital + realized + self.unrealized_pnl:
            raise ValueError("ending equity must equal starting capital plus realized and unrealized PnL")
        return self

    @field_validator("events", "metrics", mode="after")
    @classmethod
    def immutable_output_mappings(cls, value):
        return deep_freeze(value)
