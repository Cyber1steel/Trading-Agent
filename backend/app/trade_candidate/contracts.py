"""Immutable contracts for deterministic trade candidates and their evidence."""

from datetime import datetime
from decimal import Context, Decimal, localcontext
from enum import Enum
import re
from typing import Literal
from uuid import UUID

from pydantic import Field, StrictInt, field_validator, model_validator

from app.execution.assumptions import ExecutionAssumptions
from app.market_data.contracts import Instrument, MarketDataModel, require_aware_utc
from app.market_data.timeframes import Timeframe
from app.risk.contracts import RiskResult, RiskStatus, TradeDirection, TradeRiskRequest
from app.strategy.contracts import EvaluationResult, EvaluationStatus, EvidenceReference
from app.trade_candidate.fingerprints import (
    CANDIDATE_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    EXECUTION_ASSUMPTIONS_SCHEMA_VERSION,
    candidate_fingerprint,
    candidate_id_for,
)

class CandidateStatus(str, Enum):
    ACTIONABLE = "ACTIONABLE"
    WAIT = "WAIT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REJECTED = "REJECTED"


class DatasetEvidenceRef(MarketDataModel):
    dataset_id: UUID
    dataset_version: int = Field(ge=1, strict=True)
    instrument: Instrument
    timeframe: Timeframe
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_slice_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_id: str = Field(min_length=1)
    provider_version: str = Field(min_length=1)


class DefinitionEvidenceRef(MarketDataModel):
    strategy_id: UUID
    strategy_version: int = Field(ge=1, strict=True)
    strategy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    setup_id: str = Field(min_length=1)
    setup_version: int = Field(ge=1, strict=True)
    setup_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class AnalysisEvidenceRef(MarketDataModel):
    analysis_version: str = Field(min_length=1)
    analysis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_prefix_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    cutoff_at: datetime
    input_start: datetime
    analysis_start: datetime
    analysis_end: datetime

    @field_validator("cutoff_at", "input_start", "analysis_start", "analysis_end")
    @classmethod
    def utc_times(cls, value, info):
        return require_aware_utc(value, info.field_name)


class EvaluationEvidenceRef(MarketDataModel):
    status: EvaluationStatus
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_at: datetime
    source_evidence: tuple[EvidenceReference, ...]
    result: EvaluationResult

    @field_validator("evaluation_at")
    @classmethod
    def utc_evaluation_at(cls, value):
        return require_aware_utc(value, "evaluation_at")



class RiskEvidenceRef(MarketDataModel):
    configuration_id: str | None = None
    configuration_version: int | None = Field(default=None, ge=1, strict=True)
    configuration_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RiskStatus
    request: TradeRiskRequest
    result: RiskResult



class ExecutionEvidenceRef(MarketDataModel):
    schema_version: Literal[EXECUTION_ASSUMPTIONS_SCHEMA_VERSION] = EXECUTION_ASSUMPTIONS_SCHEMA_VERSION
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    assumptions: ExecutionAssumptions


class EvidencePackage(MarketDataModel):
    """Complete, immutable reference set for the facts behind one candidate."""

    schema_version: Literal[EVIDENCE_SCHEMA_VERSION] = EVIDENCE_SCHEMA_VERSION
    datasets: tuple[DatasetEvidenceRef, ...]
    analysis: AnalysisEvidenceRef
    definitions: DefinitionEvidenceRef
    evaluation: EvaluationEvidenceRef
    risk: RiskEvidenceRef
    execution: ExecutionEvidenceRef | None
    evidence: tuple[EvidenceReference, ...]

    @model_validator(mode="after")
    def ordered_unique_datasets(self):
        ordered = tuple(sorted(self.datasets, key=lambda item: (item.timeframe.value, str(item.dataset_id))))
        object.__setattr__(self, "datasets", ordered)
        return self

    @property
    def fingerprint(self) -> str:
        from app.trade_candidate.fingerprints import evidence_fingerprint
        return evidence_fingerprint(self)


class TradeCandidate(MarketDataModel):
    """Auditable research candidate; it is not an order or execution instruction."""

    schema_version: Literal[CANDIDATE_SCHEMA_VERSION] = CANDIDATE_SCHEMA_VERSION
    candidate_id: str = Field(pattern=r"^tc-[0-9a-f]{64}$")
    candidate_version: StrictInt = Field(ge=1)
    instrument: Instrument
    direction: TradeDirection | None
    entry_price: Decimal | None = Field(default=None, strict=True)
    stop_loss: Decimal | None = Field(default=None, strict=True)
    take_profits: tuple[Decimal, ...] = ()
    quantity: Decimal | None = Field(default=None, strict=True)
    account_currency: str | None = None
    risk_amount: Decimal | None = Field(default=None, strict=True)
    risk_percentage: Decimal | None = Field(default=None, strict=True)
    expected_reward_risk: Decimal | None = Field(default=None, strict=True)
    strategy_id: UUID
    strategy_version: StrictInt = Field(ge=1)
    strategy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    setup_id: str = Field(min_length=1)
    setup_version: StrictInt = Field(ge=1)
    setup_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_version: str = Field(min_length=1)
    analysis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: UUID | None
    dataset_version: StrictInt | None = Field(ge=1)
    primary_timeframe: Timeframe
    risk_configuration_id: str | None = None
    risk_configuration_version: StrictInt | None = Field(default=None, ge=1)
    risk_configuration_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    execution_assumptions_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evaluation_at: datetime
    expiry_at: datetime | None = None
    status: CandidateStatus
    findings: tuple[str, ...] = ()
    evidence: EvidencePackage
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("entry_price", "stop_loss", "quantity", "risk_amount", "risk_percentage", "expected_reward_risk", mode="before")
    @classmethod
    def finite_decimal(cls, value, info):
        if value is None:
            return None
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError(f"{info.field_name} must be a finite Decimal")
        return value

    @field_validator("take_profits", mode="before")
    @classmethod
    def finite_decimal_targets(cls, values):
        if not isinstance(values, (tuple, list)):
            raise ValueError("take_profits must be an immutable sequence of Decimal values")
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in values):
            raise ValueError("take_profits must contain finite Decimal values")
        return tuple(values)

    @field_validator("evaluation_at", "expiry_at")
    @classmethod
    def utc_candidate_times(cls, value, info):
        if value is None:
            return None
        return require_aware_utc(value, info.field_name)

    @field_validator("account_currency")
    @classmethod
    def currency_code(cls, value):
        if value is not None and not re.fullmatch(r"[A-Z]{3,8}", value):
            raise ValueError("account_currency must be an uppercase currency code")
        return value

    @model_validator(mode="after")
    def candidate_integrity(self):
        if len(self.take_profits) > 1:
            raise ValueError("The current risk contract supports at most one take-profit")
        if self.expiry_at is not None and self.expiry_at <= self.evaluation_at and self.status is CandidateStatus.ACTIONABLE:
            raise ValueError("An expired candidate cannot be ACTIONABLE")
        if self.status is CandidateStatus.ACTIONABLE:
            required = (self.entry_price, self.stop_loss, self.quantity, self.account_currency,
                        self.risk_amount, self.risk_percentage, self.evidence.execution, self.direction)
            if any(value is None for value in required):
                raise ValueError("ACTIONABLE candidates require complete price, quantity, risk, currency, and execution evidence")
            evaluation = self.evidence.evaluation.result
            risk_request = self.evidence.risk.request
            risk_result = self.evidence.risk.result
            if (evaluation.status is not EvaluationStatus.SETUP_CONFIRMED
                    or self.evidence.risk.status is not RiskStatus.RISK_VALID
                    or risk_result.request_fingerprint != risk_request.fingerprint):
                raise ValueError("ACTIONABLE candidates require confirmed evaluation and request-bound valid risk evidence")
            if (self.evidence.evaluation.status != evaluation.status
                    or self.evidence.evaluation.fingerprint != evaluation.fingerprint
                    or self.evidence.risk.request_fingerprint != risk_request.fingerprint
                    or self.evidence.risk.result_fingerprint != risk_result.fingerprint
                    or self.evidence.risk.status != risk_result.status):
                raise ValueError("ACTIONABLE evidence summaries must match their complete producer artifacts")
            if (self.strategy_id, self.strategy_version, self.strategy_fingerprint,
                self.setup_id, self.setup_version, self.setup_fingerprint,
                self.analysis_fingerprint, self.evaluation_at) != (
                evaluation.strategy_id, evaluation.strategy_version, evaluation.strategy_fingerprint,
                evaluation.setup_id, evaluation.setup_version, evaluation.setup_fingerprint,
                evaluation.analysis_fingerprint, evaluation.evaluation_at):
                raise ValueError("ACTIONABLE candidate identity must match its exact evaluation")
            if (self.evidence.definitions.strategy_id, self.evidence.definitions.strategy_version,
                self.evidence.definitions.strategy_fingerprint, self.evidence.definitions.setup_id,
                self.evidence.definitions.setup_version, self.evidence.definitions.setup_fingerprint) != (
                self.strategy_id, self.strategy_version, self.strategy_fingerprint,
                self.setup_id, self.setup_version, self.setup_fingerprint,
            ):
                raise ValueError("ACTIONABLE definition evidence must match candidate identity")
            if (self.evidence.analysis.analysis_version, self.evidence.analysis.analysis_fingerprint,
                self.evidence.analysis.evaluation_prefix_fingerprint) != (
                self.analysis_version, self.analysis_fingerprint, evaluation.evaluation_prefix_fingerprint,
            ):
                raise ValueError("ACTIONABLE analysis evidence must match candidate identity and visible prefix")
            if (risk_request.instrument != self.instrument or risk_request.side != self.direction
                    or risk_request.entry_price != self.entry_price
                    or risk_request.stop_price != self.stop_loss
                    or risk_request.target_price != (self.take_profits[0] if self.take_profits else None)
                    or risk_result.position_size != self.quantity
                    or risk_result.total_estimated_downside != self.risk_amount
                    or risk_result.reward_risk != self.expected_reward_risk):
                raise ValueError("ACTIONABLE candidate financial fields must match its exact risk artifacts")
            if (self.entry_price <= 0 or self.stop_loss <= 0 or self.quantity <= 0
                    or self.risk_amount <= 0 or self.risk_percentage <= 0):
                raise ValueError("ACTIONABLE candidate financial values must be positive")
            if ((self.direction is TradeDirection.LONG and self.stop_loss >= self.entry_price)
                    or (self.direction is TradeDirection.SHORT and self.stop_loss <= self.entry_price)):
                raise ValueError("ACTIONABLE stop must be on the loss side of entry")
            if self.take_profits:
                target = self.take_profits[0]
                if target <= 0 or ((self.direction is TradeDirection.LONG and target <= self.entry_price)
                                   or (self.direction is TradeDirection.SHORT and target >= self.entry_price)):
                    raise ValueError("ACTIONABLE target must be positive and on the reward side of entry")
            if risk_request.account_equity is None or risk_request.account_equity <= 0:
                raise ValueError("ACTIONABLE candidate requires positive account equity evidence")
            with localcontext(Context(prec=34)):
                expected_risk_percentage = self.risk_amount / risk_request.account_equity
            if self.risk_percentage != expected_risk_percentage:
                raise ValueError("ACTIONABLE risk percentage must equal risk amount divided by account equity")
            effective_assumptions = risk_request.execution_assumptions or (
                risk_request.risk_configuration.execution_assumptions
                if risk_request.risk_configuration is not None else None
            )
            execution = self.evidence.execution
            if (effective_assumptions is None or execution.assumptions != effective_assumptions
                    or execution.fingerprint != self.execution_assumptions_fingerprint):
                raise ValueError("ACTIONABLE execution assumptions must match the assumptions in its risk request")
            config = risk_request.risk_configuration
            if (self.risk_configuration_id != (config.configuration_id if config else None)
                    or self.risk_configuration_version != (config.configuration_version if config else None)
                    or self.risk_configuration_fingerprint != (config.fingerprint if config else None)):
                raise ValueError("ACTIONABLE risk configuration identity must match its risk request")
            if self.evidence.analysis.cutoff_at < self.evaluation_at:
                raise ValueError("ACTIONABLE evaluation cannot be after the analysis cutoff")
            if (self.dataset_id is None or self.dataset_version is None
                    or not any((item.dataset_id, item.dataset_version, item.instrument, item.timeframe) == (
                        self.dataset_id, self.dataset_version, self.instrument, self.primary_timeframe,
                    )
                               for item in self.evidence.datasets)):
                raise ValueError("ACTIONABLE candidate must reference a selected dataset version")
            if self.evidence.evidence != evaluation.evidence or self.evidence.evaluation.source_evidence != evaluation.evidence:
                raise ValueError("ACTIONABLE evidence package must preserve all strategy evaluation evidence")
        payload = self.model_dump(mode="python", exclude={"candidate_id", "fingerprint"})
        actual = candidate_fingerprint(payload)
        if self.fingerprint != actual or self.candidate_id != candidate_id_for(actual):
            raise ValueError("Candidate fingerprint or deterministic ID does not match its contents")
        return self
