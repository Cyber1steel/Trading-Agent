"""Immutable typed contracts for evidence-bound reasoning."""

from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from math import isfinite
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from app.backtesting.contracts import BacktestResult
from app.backtesting.fingerprints import fingerprint as backtest_fingerprint
from app.execution.assumptions import ExecutionAssumptions
from app.knowledge.models import SourceType
from app.market_analysis.contracts import AnalysisResult, MarketObservation
from app.market_data.contracts import Instrument, MarketDataModel, require_aware_utc
from app.risk.contracts import RiskResult, TradeDirection, TradeRiskRequest
from app.strategy.contracts import EvaluationResult
from app.trade_candidate.contracts import CandidateStatus, DatasetEvidenceRef, TradeCandidate
from app.core.immutability import deep_freeze
from app.reasoning.fingerprints import PROMPT_CONTRACT_VERSION, context_fingerprint


class EvidenceTier(str, Enum):
    DETERMINISTIC_MARKET = "DETERMINISTIC_MARKET"
    DETERMINISTIC_STRATEGY_RISK = "DETERMINISTIC_STRATEGY_RISK"
    HISTORICAL_VALIDATION = "HISTORICAL_VALIDATION"
    RETRIEVED_KNOWLEDGE = "RETRIEVED_KNOWLEDGE"
    LLM_INFERENCE = "LLM_INFERENCE"


class ReasoningStatus(str, Enum):
    ACTIONABLE = "ACTIONABLE"
    WAIT = "WAIT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REJECTED = "REJECTED"


class ReasoningConclusion(str, Enum):
    CANDIDATE_SUPPORTED = "CANDIDATE_SUPPORTED"
    WAIT = "WAIT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REJECTED = "REJECTED"
    EXPLANATION_ONLY = "EXPLANATION_ONLY"


class Uncertainty(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


ScalarEvidence = StrictBool | StrictInt | Decimal | StrictStr | datetime | None


class EvidenceFact(MarketDataModel):
    evidence_id: str = Field(min_length=1)
    tier: EvidenceTier
    kind: str = Field(min_length=1)
    known_at: datetime | None = None
    artifact_reference: str | None = None
    label: str = Field(min_length=1)
    value: ScalarEvidence = None

    @field_validator("known_at")
    @classmethod
    def utc_known_at(cls, value):
        return require_aware_utc(value, "known_at") if value is not None else None

    @field_validator("value")
    @classmethod
    def utc_datetime_value(cls, value):
        return require_aware_utc(value, "evidence value") if isinstance(value, datetime) else value


class KnowledgeEvidence(MarketDataModel):
    tier: Literal[EvidenceTier.RETRIEVED_KNOWLEDGE] = EvidenceTier.RETRIEVED_KNOWLEDGE
    evidence_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source_type: SourceType
    title: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    source_url: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    published_at: datetime | None = None
    retrieved_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_mode: Literal["semantic", "lexical", "hybrid"]
    retrieval_rank: int = Field(ge=1)
    semantic_rank: int | None = Field(default=None, ge=1)
    lexical_rank: int | None = Field(default=None, ge=1)
    distance: float | None = Field(default=None, ge=0, le=2)
    lexical_score: float | None = None
    hybrid_score: float | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = Field(default=None, gt=0)
    text: str = Field(min_length=1)

    @field_validator("published_at")
    @classmethod
    def utc_published_at(cls, value):
        return require_aware_utc(value, "published_at") if value is not None else None

    @model_validator(mode="after")
    def valid_span(self):
        if self.document_id != self.source_id:
            raise ValueError("Phase 2A document identity must match source identity")
        if sha256(self.text.encode("utf-8")).hexdigest() != self.retrieved_content_sha256:
            raise ValueError("retrieved text digest does not match the referenced content")
        for name in ("distance", "lexical_score", "hybrid_score"):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start")
        if self.retrieval_mode == "semantic" and self.semantic_rank is None:
            object.__setattr__(self, "semantic_rank", self.retrieval_rank)
        if self.retrieval_mode == "lexical" and self.lexical_rank is None:
            object.__setattr__(self, "lexical_rank", self.retrieval_rank)
        return self


class MarketObservationEvidence(MarketDataModel):
    tier: Literal[EvidenceTier.DETERMINISTIC_MARKET] = EvidenceTier.DETERMINISTIC_MARKET
    timeframe: str
    observation: MarketObservation

    @property
    def known_at(self):
        return self.observation.known_at

    @property
    def bar_open(self):
        return self.observation.bar_open

    @classmethod
    def from_observation(cls, observation: MarketObservation, timeframe: str):
        return cls(timeframe=timeframe, observation=observation)


class CandidateReference(MarketDataModel):
    """Cutoff-safe candidate view; excludes full-slice fields beyond the evaluation prefix."""

    candidate_id: str
    candidate_version: int = Field(ge=1, strict=True)
    candidate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: CandidateStatus
    instrument: Instrument
    direction: TradeDirection | None
    entry_price: Decimal | None
    stop_loss: Decimal | None
    take_profits: tuple[Decimal, ...]
    quantity: Decimal | None
    account_currency: str | None
    risk_amount: Decimal | None
    risk_percentage: Decimal | None
    expected_reward_risk: Decimal | None
    dataset_id: str | None
    dataset_version: int | None
    primary_timeframe: str
    analysis_version: str
    analysis_fingerprint: str
    strategy_id: str
    strategy_version: int = Field(ge=1, strict=True)
    strategy_fingerprint: str
    setup_id: str
    setup_version: int = Field(ge=1, strict=True)
    setup_fingerprint: str
    risk_request_fingerprint: str
    risk_result_fingerprint: str
    evidence_package_fingerprint: str
    datasets: tuple[DatasetEvidenceRef, ...]
    strategy_evaluation: EvaluationResult
    risk_request: TradeRiskRequest
    risk_result: RiskResult
    execution_assumptions: ExecutionAssumptions | None
    execution_assumptions_fingerprint: str | None
    evaluation_at: datetime
    expiry_at: datetime | None

    @model_validator(mode="after")
    def artifact_binding(self):
        if self.candidate_id != f"tc-{self.candidate_fingerprint}":
            raise ValueError("candidate ID does not match its fingerprint")
        if self.risk_request.fingerprint != self.risk_request_fingerprint:
            raise ValueError("risk request fingerprint does not match candidate reference")
        if self.risk_result.fingerprint != self.risk_result_fingerprint:
            raise ValueError("risk result fingerprint does not match candidate reference")
        if self.strategy_evaluation.strategy_id != UUID(self.strategy_id):
            raise ValueError("strategy evaluation identity does not match candidate reference")
        if (self.strategy_evaluation.strategy_version, self.strategy_evaluation.setup_id,
            self.strategy_evaluation.setup_version, self.strategy_evaluation.analysis_fingerprint) != (
            self.strategy_version, self.setup_id, self.setup_version, self.analysis_fingerprint
        ):
            raise ValueError("strategy evaluation artifacts do not match candidate reference")
        if self.status is CandidateStatus.ACTIONABLE:
            if self.risk_request.entry_price != self.entry_price or self.risk_request.stop_price != self.stop_loss:
                raise ValueError("risk request prices do not match candidate reference")
            if self.risk_result.position_size != self.quantity or self.risk_result.total_estimated_downside != self.risk_amount:
                raise ValueError("risk outputs do not match candidate reference")
        if self.dataset_id is not None and not any(
            str(item.dataset_id) == self.dataset_id and item.dataset_version == self.dataset_version
            and item.timeframe.value == self.primary_timeframe for item in self.datasets
        ):
            raise ValueError("primary dataset reference is missing from candidate evidence")
        return self

    @field_validator("evaluation_at", "expiry_at")
    @classmethod
    def utc_candidate_time(cls, value, info):
        return require_aware_utc(value, info.field_name) if value is not None else None

    @classmethod
    def from_candidate(cls, item: TradeCandidate):
        return cls(
            candidate_id=item.candidate_id,
            candidate_version=item.candidate_version,
            candidate_fingerprint=item.fingerprint,
            status=item.status,
            instrument=item.instrument,
            direction=item.direction,
            entry_price=item.entry_price,
            stop_loss=item.stop_loss,
            take_profits=item.take_profits,
            quantity=item.quantity,
            account_currency=item.account_currency,
            risk_amount=item.risk_amount,
            risk_percentage=item.risk_percentage,
            expected_reward_risk=item.expected_reward_risk,
            dataset_id=str(item.dataset_id) if item.dataset_id else None,
            dataset_version=item.dataset_version,
            primary_timeframe=item.primary_timeframe.value,
            analysis_version=item.analysis_version,
            analysis_fingerprint=item.analysis_fingerprint,
            strategy_id=str(item.strategy_id),
            strategy_version=item.strategy_version,
            strategy_fingerprint=item.strategy_fingerprint,
            setup_id=item.setup_id,
            setup_version=item.setup_version,
            setup_fingerprint=item.setup_fingerprint,
            risk_request_fingerprint=item.evidence.risk.request_fingerprint,
            risk_result_fingerprint=item.evidence.risk.result_fingerprint,
            evidence_package_fingerprint=item.evidence.fingerprint,
            datasets=item.evidence.datasets,
            strategy_evaluation=item.evidence.evaluation.result,
            risk_request=item.evidence.risk.request,
            risk_result=item.evidence.risk.result,
            execution_assumptions=(
                item.evidence.execution.assumptions if item.evidence.execution is not None else None
            ),
            execution_assumptions_fingerprint=item.execution_assumptions_fingerprint,
            evaluation_at=item.evaluation_at,
            expiry_at=item.expiry_at,
        )


class HistoricalValidationEvidence(MarketDataModel):
    tier: Literal[EvidenceTier.HISTORICAL_VALIDATION] = EvidenceTier.HISTORICAL_VALIDATION
    result_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str
    dataset_version: int = Field(ge=1, strict=True)
    dataset_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_id: str
    strategy_version: int = Field(ge=1, strict=True)
    setup_id: str
    setup_version: int = Field(ge=1, strict=True)
    analysis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_prefix_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine_version: str
    execution_assumptions_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_assumptions: ExecutionAssumptions
    start: datetime
    end: datetime
    known_at: datetime
    starting_capital: Decimal
    ending_equity: Decimal
    total_trades: int = Field(ge=0)
    closed_trades: int = Field(ge=0)
    net_profit: Decimal
    win_rate: Decimal | None
    maximum_drawdown: Decimal
    out_of_sample_status: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    warnings: tuple[str, ...]
    limitations: tuple[str, ...] = (
        "Historical simulation only; it is not a forecast or proof of future profitability.",
        "Walk-forward, out-of-sample, and Monte Carlo validation are not established by this repository phase.",
        "Execution realism is limited to the assumptions recorded by the backtest artifact.",
    )

    @field_validator("start", "end", "known_at")
    @classmethod
    def utc_range(cls, value, info):
        return require_aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def historical_timing(self):
        if self.end > self.known_at:
            raise ValueError("backtest evidence cannot be known before its historical range ends")
        return self

    @classmethod
    def from_result(cls, result: BacktestResult, *, known_at: datetime):
        metrics = result.metrics
        def decimal_metric(name, default=None):
            value = metrics.get(name, default)
            if value is None:
                return None
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"Backtest metric {name} is not a finite Decimal")
            return value
        def integer_metric(name):
            value = metrics.get(name, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"Backtest metric {name} is not a non-negative integer")
            return value
        return cls(
            result_fingerprint=result.result_fingerprint,
            dataset_id=str(result.request.dataset_id),
            dataset_version=result.request.dataset_version,
            dataset_content_hash=result.manifest.content_hash,
            strategy_id=str(result.request.strategy_id),
            strategy_version=result.request.strategy_version,
            setup_id=result.request.setup_id,
            setup_version=result.request.setup_version,
            analysis_fingerprint=result.request.analysis_fingerprint,
            analysis_prefix_fingerprint=result.request.analysis_prefix_fingerprint,
            request_fingerprint=result.request.replay_fingerprint,
            engine_version=result.engine_version,
            execution_assumptions_fingerprint=backtest_fingerprint(
                result.execution, domain="execution-assumptions"
            ),
            execution_assumptions=result.execution,
            start=result.request.start,
            end=result.request.end,
            known_at=known_at,
            starting_capital=result.starting_capital,
            ending_equity=result.ending_equity,
            total_trades=integer_metric("total_trades"),
            closed_trades=integer_metric("closed_trades"),
            net_profit=decimal_metric("net_profit", Decimal(0)),
            win_rate=decimal_metric("win_rate"),
            maximum_drawdown=decimal_metric("maximum_drawdown", Decimal(0)),
            warnings=result.warnings,
        )


class ReasoningContext(MarketDataModel):
    schema_version: Literal["3a.1.0"] = "3a.1.0"
    prompt_contract_version: str = PROMPT_CONTRACT_VERSION
    evaluation_time: datetime
    candidate: CandidateReference
    effective_candidate_status: CandidateStatus
    market_observations: tuple[MarketObservationEvidence, ...] = ()
    historical_validation: HistoricalValidationEvidence | None = None
    knowledge: tuple[KnowledgeEvidence, ...] = ()
    deterministic_facts: tuple[EvidenceFact, ...]
    knowledge_unavailable_reason: str | None = None
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("evaluation_time")
    @classmethod
    def utc_evaluation_time(cls, value):
        return require_aware_utc(value, "evaluation_time")

    @field_validator("market_observations", "knowledge", "deterministic_facts", mode="before")
    @classmethod
    def require_immutable_sequences(cls, value):
        if not isinstance(value, (tuple, list)):
            raise ValueError("evidence collections must be sequences")
        return tuple(value)

    @field_validator("market_observations", "knowledge", "deterministic_facts", mode="after")
    @classmethod
    def chronological_evidence(cls, value, info):
        if info.field_name == "market_observations":
            if any(value[index].known_at >= value[index + 1].known_at for index in range(len(value) - 1)):
                raise ValueError("market observations must be chronological")
        return value

    @model_validator(mode="after")
    def consistent_context(self):
        if self.prompt_contract_version != PROMPT_CONTRACT_VERSION:
            raise ValueError("reasoning context prompt contract version is unsupported")
        if self.candidate.evaluation_at > self.evaluation_time:
            raise ValueError("candidate evaluation is later than reasoning time")
        expected_status = self.candidate.status
        if (
            expected_status is CandidateStatus.ACTIONABLE
            and self.candidate.expiry_at is not None
            and self.candidate.expiry_at <= self.evaluation_time
        ):
            expected_status = CandidateStatus.WAIT
        if self.effective_candidate_status is not expected_status:
            raise ValueError("effective candidate status conflicts with deterministic status/expiry")
        if any(item.known_at > self.evaluation_time for item in self.market_observations):
            raise ValueError("future market evidence is forbidden")
        if any(item.timeframe != self.candidate.primary_timeframe for item in self.market_observations):
            raise ValueError("market observations must match the candidate primary timeframe")
        if self.historical_validation is not None and self.historical_validation.known_at > self.evaluation_time:
            raise ValueError("future backtest validation is forbidden")
        if any(item.published_at is None or item.published_at > self.evaluation_time for item in self.knowledge):
            raise ValueError("undated or future-published knowledge is forbidden")
        if any(item.known_at is not None and item.known_at > self.evaluation_time for item in self.deterministic_facts):
            raise ValueError("future deterministic evidence is forbidden")
        if any(isinstance(item.value, datetime) and item.value > self.evaluation_time
               for item in self.deterministic_facts):
            raise ValueError("future timestamp values are forbidden in deterministic facts")
        ids = [item.evidence_id for item in self.deterministic_facts]
        ids.extend(item.evidence_id for item in self.knowledge)
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique")
        expected_tiers = {
            EvidenceTier.DETERMINISTIC_MARKET,
            EvidenceTier.DETERMINISTIC_STRATEGY_RISK,
            EvidenceTier.HISTORICAL_VALIDATION,
        }
        if any(item.tier not in expected_tiers for item in self.deterministic_facts):
            raise ValueError("deterministic facts cannot be assigned knowledge or inference tiers")
        for fact in self.deterministic_facts:
            if fact.tier is EvidenceTier.DETERMINISTIC_MARKET and fact.artifact_reference != self.candidate.analysis_fingerprint:
                raise ValueError("market evidence must reference the candidate analysis artifact")
            if fact.tier is EvidenceTier.DETERMINISTIC_STRATEGY_RISK and fact.artifact_reference != self.candidate.candidate_fingerprint:
                raise ValueError("strategy/risk evidence must reference the candidate artifact")
            if fact.tier is EvidenceTier.HISTORICAL_VALIDATION and (
                self.historical_validation is None
                or fact.artifact_reference != self.historical_validation.result_fingerprint
            ):
                raise ValueError("historical facts must reference supplied backtest evidence")
        expected = context_fingerprint(self.model_copy(update={"fingerprint": "0" * 64}))
        if self.fingerprint != expected:
            raise ValueError("reasoning context fingerprint mismatch")
        return self


class ReasoningRequest(MarketDataModel):
    context: ReasoningContext
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value):
        if not value.strip():
            raise ValueError("question cannot be blank")
        return value.strip()


class NumericalClaim(MarketDataModel):
    evidence_id: str = Field(min_length=1)
    value: ScalarEvidence


class ReasoningProposal(MarketDataModel):
    status: ReasoningStatus
    conclusion: ReasoningConclusion
    uncertainty: Uncertainty
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    knowledge_evidence_ids: tuple[str, ...] = ()
    numerical_claims: tuple[NumericalClaim, ...] = ()
    explanation: str = Field(max_length=12000)


class ReasoningResult(MarketDataModel):
    schema_version: Literal["3a.1.0"] = "3a.1.0"
    inference_tier: Literal[EvidenceTier.LLM_INFERENCE] = EvidenceTier.LLM_INFERENCE
    status: ReasoningStatus
    deterministic_candidate_status: CandidateStatus
    conclusion: ReasoningConclusion
    uncertainty: Uncertainty
    explanation: str
    supporting_evidence: tuple[EvidenceFact, ...]
    contradicting_evidence: tuple[EvidenceFact, ...]
    missing_evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    risks: tuple[str, ...]
    limitations: tuple[str, ...]
    knowledge_references: tuple[KnowledgeEvidence, ...]
    candidate_id: str
    candidate_fingerprint: str
    context_fingerprint: str
    prompt_contract_version: str
    provider_id: str
    model_id: str
    question: str
    provider_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    request_metadata: dict[str, str | int | bool | None]
    provider_telemetry: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    failure_code: str | None = None
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("request_metadata", "provider_telemetry", mode="after")
    @classmethod
    def freeze_metadata(cls, value):
        for item in value.values():
            if type(item) in (int, float) and (item < 0 or (type(item) is float and not isfinite(item))):
                raise ValueError("reasoning metadata numeric values must be finite and non-negative")
        return deep_freeze(value)

    @model_validator(mode="after")
    def fingerprint_matches(self):
        from app.reasoning.fingerprints import result_fingerprint
        expected = result_fingerprint(self.model_dump(mode="python", exclude={"fingerprint"}))
        if self.fingerprint != expected:
            raise ValueError("reasoning result fingerprint mismatch")
        return self
