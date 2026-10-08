"""Typed, explicit execution-cost assumptions; this is not a broker interface."""

from decimal import Decimal
from enum import Enum

from pydantic import ConfigDict, Field, model_validator

from app.market_data.contracts import MarketDataModel


class PriceBasis(str, Enum):
    TRADE = "trade"
    MID = "mid"
    BID = "bid"
    ASK = "ask"


class ExecutionAssumptions(MarketDataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    spread: Decimal = Field(..., strict=True)
    slippage: Decimal = Field(..., strict=True)
    pct_fee: Decimal | None = Field(default=None, strict=True)
    fixed_fee: Decimal | None = Field(default=None, strict=True)
    price_basis: PriceBasis = PriceBasis.TRADE
    fixed_quantity: int = Field(..., ge=1, strict=True)

    @model_validator(mode="after")
    def validate_decimals(self):
        for name in ("spread", "slippage", "pct_fee", "fixed_fee"):
            value = getattr(self, name)
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError(f"{name} must be a non-negative finite Decimal")
        return self
