"""Immutable risk-engine input/output contracts."""

import re
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.execution.assumptions import ExecutionAssumptions
from app.market_data.contracts import Instrument, MarketDataModel, require_aware_utc
from app.risk.fingerprints import fingerprint
from app.core.immutability import deep_freeze

_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class RiskStatus(str, Enum):
    RISK_VALID = "RISK_VALID"
    RISK_REJECTED = "RISK_REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class RiskSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    REJECT = "reject"
    INSUFFICIENT = "insufficient"


class RiskFinding(MarketDataModel):
    rule: str
    observed_value: str | int | Decimal | None = None
    expected_constraint: str | None = None
    severity: RiskSeverity
    message: str


class RiskConfiguration(MarketDataModel):
    configuration_id: str = Field(min_length=1, max_length=128)
    configuration_version: int = Field(..., ge=1, strict=True)
    risk_per_trade: Decimal = Field(..., strict=True)
    max_position_size: int | None = Field(default=None, ge=1, strict=True)
    max_notional_exposure: Decimal | None = Field(default=None, strict=True)
    minimum_stop_distance: Decimal | None = Field(default=None, strict=True)
    maximum_stop_distance: Decimal | None = Field(default=None, strict=True)
    minimum_reward_risk: Decimal | None = Field(default=None, strict=True)
    account_currency: str = Field(min_length=3, max_length=8, pattern=r"^[A-Z]{3,8}$")
    execution_assumptions: ExecutionAssumptions | None = None
    quantity_increment: Decimal | None = Field(default=None, strict=True)

    @field_validator("risk_per_trade")
    @classmethod
    def risk_percentage(cls, value: Decimal) -> Decimal:
        if isinstance(value, bool):
            raise TypeError("risk_per_trade must be a Decimal percentage fraction")
        if not isinstance(value, Decimal) or not value.is_finite():
            raise TypeError("risk_per_trade must be a finite Decimal")
        if value <= 0 or value > Decimal("1"):
            raise ValueError("risk_per_trade must satisfy 0 < risk_per_trade <= 1")
        return value

    @field_validator("max_notional_exposure", "minimum_stop_distance", "maximum_stop_distance", "minimum_reward_risk", "quantity_increment")
    @classmethod
    def positive_decimal_or_none(cls, value):
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError("Financial values must be Decimal, not bool")
        if not isinstance(value, Decimal) or not value.is_finite():
            raise TypeError("Financial values must be finite Decimal instances")
        if value <= 0:
            raise ValueError("Financial values must be positive")
        return value

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.minimum_stop_distance is not None and self.maximum_stop_distance is not None:
            if self.minimum_stop_distance > self.maximum_stop_distance:
                raise ValueError("minimum_stop_distance must be <= maximum_stop_distance")
        return self

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="python", exclude_none=True), domain="risk-config")


class TradeRiskRequest(MarketDataModel):
    request_id: str = Field(min_length=1, max_length=128)
    risk_configuration: RiskConfiguration | None = None
    risk_configuration_fingerprint: str | None = None
    strategy_id: str | None = None
    setup_id: str | None = None
    strategy_fingerprint: str | None = None
    setup_fingerprint: str | None = None
    analysis_fingerprint: str | None = None
    evaluation_prefix_fingerprint: str | None = None
    instrument: Instrument | None = None
    side: TradeDirection | None = None
    entry_price: Decimal | None = Field(default=None, strict=True)
    stop_price: Decimal | None = Field(default=None, strict=True)
    target_price: Decimal | None = Field(default=None, strict=True)
    account_equity: Decimal | None = Field(default=None, strict=True)
    pnl_to_account_rate: Decimal | None = Field(default=None, strict=True)
    execution_assumptions: ExecutionAssumptions | None = None
    evaluation_at: datetime | None = None
    quantity_model: Literal["unit"] = "unit"
    provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator("risk_configuration_fingerprint", "strategy_fingerprint", "setup_fingerprint",
                     "analysis_fingerprint", "evaluation_prefix_fingerprint")
    @classmethod
    def fingerprint_format(cls, value):
        if value is None:
            return None
        if not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value):
            raise ValueError("Fingerprint must be a 64-character lowercase hex SHA-256 value")
        return value

    @field_validator("side")
    @classmethod
    def reject_bool_side(cls, value):
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError("side must be a TradeDirection enum value, not bool")
        return value

    @field_validator("entry_price", "stop_price", "target_price", "account_equity", "pnl_to_account_rate")
    @classmethod
    def finite_money(cls, value):
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError("Financial values must be Decimal, not bool")
        if not isinstance(value, Decimal) or not value.is_finite():
            raise TypeError("Financial values must be finite Decimal instances")
        if value <= 0:
            raise ValueError("Financial values must be positive")
        return value

    @field_validator("evaluation_at")
    @classmethod
    def require_utc(cls, value):
        if value is None:
            return None
        return require_aware_utc(value, "evaluation_at")

    @model_validator(mode="after")
    def config_fingerprint_matches(self):
        if self.risk_configuration is not None and self.risk_configuration_fingerprint is not None:
            expected = self.risk_configuration.fingerprint
            if self.risk_configuration_fingerprint != expected:
                raise ValueError("risk_configuration_fingerprint does not match the supplied configuration")
        return self

    @field_validator("provenance", mode="after")
    @classmethod
    def immutable_provenance(cls, value):
        return deep_freeze(value)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="python"), domain="risk-request")


class RiskCalculation(MarketDataModel):
    maximum_monetary_risk: Decimal | None = None
    stop_distance: Decimal | None = None
    risk_per_unit: Decimal | None = None
    position_size: Decimal | None = None
    notional_exposure: Decimal | None = None
    estimated_fees: Decimal | None = None
    estimated_spread_cost: Decimal | None = None
    estimated_slippage_impact: Decimal | None = None
    total_estimated_downside: Decimal | None = None
    reward_risk: Decimal | None = None
    reward_distance: Decimal | None = None
    execution_cost_per_unit: Decimal | None = None


class RiskResult(MarketDataModel):
    status: RiskStatus
    request_fingerprint: str
    requested_risk: Decimal | None = None
    maximum_monetary_risk: Decimal | None = None
    stop_distance: Decimal | None = None
    position_size: Decimal | None = None
    notional_exposure: Decimal | None = None
    estimated_fees: Decimal | None = None
    estimated_spread_cost: Decimal | None = None
    estimated_slippage_impact: Decimal | None = None
    total_estimated_downside: Decimal | None = None
    reward_risk: Decimal | None = None
    validation_findings: tuple[RiskFinding, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: dict[str, str] = Field(default_factory=dict)
    fingerprint: str

    @field_validator("provenance", mode="after")
    @classmethod
    def immutable_provenance(cls, value):
        return deep_freeze(value)
