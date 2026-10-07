"""Immutable definitions and audit records for deterministic setup evaluation."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Literal, Union
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from app.market_context.sessions import CalendarStatus
from app.market_data.contracts import Instrument, MarketDataModel, require_aware_utc
from app.market_data.timeframes import Timeframe
from app.strategy.fingerprints import fingerprint

EvidenceValue = StrictBool | StrictInt | Decimal | StrictStr | datetime | tuple[StrictStr, ...] | None


class ParameterType(str, Enum):
    DECIMAL = "decimal"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING = "string"


class StrategyParameter(MarketDataModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    value_type: ParameterType
    value: object

    @model_validator(mode="after")
    def strict_value_type(self):
        valid = {
            ParameterType.DECIMAL: lambda v: isinstance(v, Decimal) and v.is_finite(),
            ParameterType.INTEGER: lambda v: type(v) is int,
            ParameterType.BOOLEAN: lambda v: type(v) is bool,
            ParameterType.STRING: lambda v: type(v) is str,
        }[self.value_type](self.value)
        if not valid:
            raise ValueError(f"Parameter {self.name!r} must use its declared strict type")
        return self


class FieldRef(str, Enum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    BID = "bid"
    ASK = "ask"
    SPREAD = "spread"
    CANDLE_RANGE = "features.range"
    CANDLE_BODY = "features.body"
    UPPER_WICK = "features.upper_wick"
    LOWER_WICK = "features.lower_wick"
    BODY_RATIO = "features.body_ratio"
    CLOSE_LOCATION = "features.close_location"
    SIMPLE_RETURN = "features.simple_return"
    TRUE_RANGE = "true_range"
    ATR = "atr"
    STRUCTURAL_STATE = "structural_state"
    LAST_SWING_HIGH_PRICE = "swing.last_high.price"
    LAST_SWING_HIGH_CANDIDATE_AT = "swing.last_high.candidate_at"
    LAST_SWING_HIGH_KNOWN_AT = "swing.last_high.known_at"
    LAST_SWING_LOW_PRICE = "swing.last_low.price"
    LAST_SWING_LOW_CANDIDATE_AT = "swing.last_low.candidate_at"
    LAST_SWING_LOW_KNOWN_AT = "swing.last_low.known_at"
    SESSION_WEEKEND = "session.weekend"
    SESSION_CALENDAR_STATUS = "session.calendar_status"
    SESSION_LABELS = "session.labels"
    SESSION_LABEL_COUNT = "session.label_count"
    SESSION_ENTERED_COUNT = "session_transition.entered_count"
    SESSION_EXITED_COUNT = "session_transition.exited_count"
    HIGHER_CLOSE = "higher.close"
    HIGHER_STRUCTURAL_STATE = "higher.structural_state"
    HIGHER_LAST_SWING_HIGH_PRICE = "higher.swing.last_high.price"
    HIGHER_LAST_SWING_HIGH_KNOWN_AT = "higher.swing.last_high.known_at"
    HIGHER_LAST_SWING_LOW_PRICE = "higher.swing.last_low.price"
    HIGHER_LAST_SWING_LOW_KNOWN_AT = "higher.swing.last_low.known_at"


class ComparisonOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


class ParameterRef(MarketDataModel):
    name: str = Field(min_length=1, max_length=64)


class ConditionOperand(MarketDataModel):
    value: object | None = None
    parameter: ParameterRef | None = None

    @model_validator(mode="after")
    def one_operand(self):
        if (self.parameter is None) == (self.value is None):
            raise ValueError("Provide exactly one literal value or parameter reference")
        if isinstance(self.value, float):
            raise ValueError("Float condition literals are forbidden; use Decimal")
        if isinstance(self.value, Decimal) and not self.value.is_finite():
            raise ValueError("Condition Decimal literals must be finite")
        if self.value is not None and type(self.value) not in (str, int, bool, Decimal, datetime):
            raise ValueError("Condition literals must be Decimal, integer, boolean, string, or datetime")
        if isinstance(self.value, datetime):
            object.__setattr__(self, "value", require_aware_utc(self.value, "condition datetime"))
        return self


class Compare(MarketDataModel):
    kind: Literal["compare"] = "compare"
    field: FieldRef
    operator: ComparisonOperator
    operand: ConditionOperand
    timeframe: Timeframe | None = None

    @field_validator("timeframe", mode="before")
    @classmethod
    def parse_timeframe(cls, value):
        return Timeframe.parse(value) if value is not None else None

    @model_validator(mode="after")
    def field_scope(self):
        if self.field is FieldRef.SESSION_LABELS:
            raise ValueError("Session-label membership is a required-context check, not a scalar condition field")
        higher = self.field.value.startswith("higher.")
        if higher != (self.timeframe is not None):
            raise ValueError("Higher-timeframe fields require a timeframe; primary fields forbid one")
        if self.timeframe is not None and self.timeframe.is_calendar_anchored:
            raise ValueError("Calendar-anchored higher timeframes are not supported by Phase 2E")
        return self


Condition = Annotated[Union["AllOf", "AnyOf", "Not", Compare], Field(discriminator="kind")]


class AllOf(MarketDataModel):
    kind: Literal["all"] = "all"
    conditions: tuple[Condition, ...] = Field(min_length=1)


class AnyOf(MarketDataModel):
    kind: Literal["any"] = "any"
    conditions: tuple[Condition, ...] = Field(min_length=1)


class Not(MarketDataModel):
    kind: Literal["not"] = "not"
    condition: Condition


AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()


class RequiredMarketContext(MarketDataModel):
    session_labels: tuple[str, ...] = ()
    calendar_status: CalendarStatus | None = None

    @field_validator("session_labels")
    @classmethod
    def unique_labels(cls, value):
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("Session labels must be nonblank and unique")
        return tuple(sorted(normalized))


class ExpiryRule(MarketDataModel):
    max_later_observations: int = Field(ge=1, strict=True)


class SetupDefinition(MarketDataModel):
    setup_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
    setup_version: int = Field(ge=1, strict=True)
    instrument: Instrument
    primary_timeframe: Timeframe
    required_higher_timeframes: tuple[Timeframe, ...] = ()
    required_context: RequiredMarketContext = Field(default_factory=RequiredMarketContext)
    prerequisites: Condition | None = None
    entry_conditions: Condition
    confirmation_conditions: Condition | None = None
    invalidation_conditions: Condition | None = None
    expiry: ExpiryRule

    @field_validator("primary_timeframe", mode="before")
    @classmethod
    def parse_primary_tf(cls, value):
        return Timeframe.parse(value)

    @field_validator("required_higher_timeframes", mode="before")
    @classmethod
    def parse_parent_tfs(cls, values):
        return tuple(Timeframe.parse(v) for v in values)

    @model_validator(mode="after")
    def valid_timeframes(self):
        if self.primary_timeframe.is_calendar_anchored:
            raise ValueError("Calendar-anchored primary timeframe is unsupported")
        if len(set(self.required_higher_timeframes)) != len(self.required_higher_timeframes):
            raise ValueError("Required higher timeframes must be unique")
        base = self.primary_timeframe.nominal_duration
        for timeframe in self.required_higher_timeframes:
            if timeframe == self.primary_timeframe or timeframe.is_calendar_anchored:
                raise ValueError("Required higher timeframes must be distinct fixed-duration intervals")
            if timeframe.nominal_duration < base or timeframe.nominal_duration % base:
                raise ValueError("Required higher timeframes must be integer multiples of primary")
        object.__setattr__(self, "required_higher_timeframes",
                           tuple(sorted(self.required_higher_timeframes, key=lambda item: item.value)))
        for condition in (self.prerequisites, self.entry_conditions, self.confirmation_conditions, self.invalidation_conditions):
            if condition is not None:
                for node in walk_conditions(condition):
                    if isinstance(node, Compare) and node.timeframe is not None and node.timeframe not in self.required_higher_timeframes:
                        raise ValueError(f"Condition references undeclared higher timeframe {node.timeframe}")
        return self

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="python"), domain="setup")


class StrategyDefinition(MarketDataModel):
    strategy_id: UUID
    strategy_version: int = Field(ge=1, strict=True)
    analysis_version: str = Field(min_length=1, max_length=64)
    parameters: tuple[StrategyParameter, ...] = ()
    setups: tuple[SetupDefinition, ...] = Field(min_length=1)
    fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def unique_and_consistent(self):
        names = [p.name for p in self.parameters]
        ids = [s.setup_id for s in self.setups]
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique")
        if len(ids) != len(set(ids)):
            raise ValueError("Setup IDs must be unique within a strategy")
        params = {p.name: p for p in self.parameters}
        for setup in self.setups:
            for condition in (setup.prerequisites, setup.entry_conditions, setup.confirmation_conditions, setup.invalidation_conditions):
                if condition is not None:
                    validate_parameters(condition, params)
        expected = self.calculate_fingerprint()
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("Strategy fingerprint does not match its definition")
        object.__setattr__(self, "fingerprint", expected)
        object.__setattr__(self, "setups", tuple(sorted(self.setups, key=lambda x: x.setup_id)))
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters, key=lambda x: x.name)))
        return self

    def calculate_fingerprint(self) -> str:
        payload = {
            "strategy_id": self.strategy_id, "strategy_version": self.strategy_version,
            "analysis_version": self.analysis_version,
            "parameters": tuple(sorted(self.parameters, key=lambda x: x.name)),
            "setups": tuple(sorted((s.fingerprint for s in self.setups))),
        }
        return fingerprint(payload, domain="strategy")

    def setup(self, setup_id: str, setup_version: int) -> SetupDefinition:
        for item in self.setups:
            if item.setup_id == setup_id and item.setup_version == setup_version:
                return item
        raise ValueError("Exact setup identity/version is not present in strategy definition")


class EvaluationRequest(MarketDataModel):
    strategy_id: UUID
    strategy_version: int = Field(ge=1, strict=True)
    strategy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    setup_id: str = Field(min_length=1)
    setup_version: int = Field(ge=1, strict=True)
    setup_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_prefix_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_at: datetime

    @field_validator("evaluation_at")
    @classmethod
    def utc_evaluation_time(cls, value):
        return require_aware_utc(value, "evaluation_at")

    @property
    def fingerprint(self):
        payload = self.model_dump(mode="python", exclude={"analysis_fingerprint"})
        return fingerprint(payload, domain="evaluation-request")


class TruthValue(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    INSUFFICIENT = "INSUFFICIENT"


class EvaluationStatus(str, Enum):
    SETUP_CONFIRMED = "SETUP_CONFIRMED"
    SETUP_NOT_CONFIRMED = "SETUP_NOT_CONFIRMED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class LifecycleState(str, Enum):
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class EvidenceReference(MarketDataModel):
    dataset_id: UUID
    dataset_version: StrictInt = Field(ge=1)
    instrument: Instrument
    timeframe: Timeframe
    bar_open: datetime
    known_at: datetime
    analysis_version: str
    analysis_fingerprint: str
    field: FieldRef
    value: EvidenceValue = None
    unavailable_reason: str | None = None
    condition_id: str
    context_requirement: str | None = None
    operator: ComparisonOperator | None = None
    comparison_value: EvidenceValue = None
    evaluation_prefix_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_bar_open: datetime | None = None
    supporting_candidate_at: datetime | None = None
    supporting_confirmed_at: datetime | None = None

    @field_validator("value", "comparison_value", mode="before")
    @classmethod
    def immutable_evidence_value(cls, value):
        if value is None or type(value) in (bool, int, str, Decimal, datetime):
            return value
        if isinstance(value, Enum) and isinstance(value.value, str):
            return value.value
        if isinstance(value, tuple) and all(type(item) is str for item in value):
            return value
        raise ValueError("Evidence values must be immutable scalars or tuples of strings; floats and containers are forbidden")

    @model_validator(mode="after")
    def evidence_consistency(self):
        if (self.value is None) == (self.unavailable_reason is None):
            raise ValueError("Evidence must contain a value or an unavailable reason")
        for name in ("value", "comparison_value"):
            value = getattr(self, name)
            if isinstance(value, Decimal) and not value.is_finite():
                raise ValueError("Evidence Decimal values must be finite")
            if isinstance(value, datetime):
                object.__setattr__(self, name, require_aware_utc(value, name))
        for name in ("bar_open", "known_at", "parent_bar_open", "supporting_candidate_at", "supporting_confirmed_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_aware_utc(value, name))
        return self


class ConditionTrace(MarketDataModel):
    condition_id: str
    known_at: datetime
    outcome: TruthValue
    evidence: tuple[EvidenceReference, ...]

    @field_validator("known_at")
    @classmethod
    def utc_known_at(cls, value):
        return require_aware_utc(value, "known_at")


class LifecycleTransition(MarketDataModel):
    state: LifecycleState
    known_at: datetime
    reason: str
    evidence: tuple[EvidenceReference, ...] = ()

    @field_validator("known_at")
    @classmethod
    def utc_transition_time(cls, value):
        return require_aware_utc(value, "known_at")


class EvaluationResult(MarketDataModel):
    status: EvaluationStatus
    strategy_id: UUID
    strategy_version: int = Field(ge=1, strict=True)
    strategy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    setup_id: str
    setup_version: int = Field(ge=1, strict=True)
    setup_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_prefix_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_at: datetime
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition_trace: tuple[ConditionTrace, ...]
    evidence: tuple[EvidenceReference, ...]
    lifecycle_transitions: tuple[LifecycleTransition, ...]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("evaluation_at")
    @classmethod
    def utc_result_time(cls, value):
        return require_aware_utc(value, "evaluation_at")


def walk_conditions(condition):
    if condition is None:
        return
    yield condition
    if isinstance(condition, (AllOf, AnyOf)):
        for child in condition.conditions:
            yield from walk_conditions(child)
    elif isinstance(condition, Not):
        yield from walk_conditions(condition.condition)


_FIELD_TYPES = MappingProxyType({
    FieldRef.OPEN: "decimal", FieldRef.HIGH: "decimal", FieldRef.LOW: "decimal", FieldRef.CLOSE: "decimal",
    FieldRef.VOLUME: "decimal", FieldRef.BID: "decimal", FieldRef.ASK: "decimal", FieldRef.SPREAD: "decimal",
    FieldRef.CANDLE_RANGE: "decimal", FieldRef.CANDLE_BODY: "decimal", FieldRef.UPPER_WICK: "decimal",
    FieldRef.LOWER_WICK: "decimal", FieldRef.BODY_RATIO: "decimal", FieldRef.CLOSE_LOCATION: "decimal",
    FieldRef.SIMPLE_RETURN: "decimal", FieldRef.TRUE_RANGE: "decimal", FieldRef.ATR: "decimal",
    FieldRef.STRUCTURAL_STATE: "string", FieldRef.LAST_SWING_HIGH_PRICE: "decimal",
    FieldRef.LAST_SWING_HIGH_CANDIDATE_AT: "datetime", FieldRef.LAST_SWING_HIGH_KNOWN_AT: "datetime",
    FieldRef.LAST_SWING_LOW_PRICE: "decimal", FieldRef.LAST_SWING_LOW_CANDIDATE_AT: "datetime",
    FieldRef.LAST_SWING_LOW_KNOWN_AT: "datetime", FieldRef.SESSION_WEEKEND: "boolean",
    FieldRef.SESSION_CALENDAR_STATUS: "string", FieldRef.SESSION_LABEL_COUNT: "integer",
    FieldRef.SESSION_ENTERED_COUNT: "integer", FieldRef.SESSION_EXITED_COUNT: "integer",
    FieldRef.HIGHER_CLOSE: "decimal", FieldRef.HIGHER_STRUCTURAL_STATE: "string",
    FieldRef.HIGHER_LAST_SWING_HIGH_PRICE: "decimal", FieldRef.HIGHER_LAST_SWING_HIGH_KNOWN_AT: "datetime",
    FieldRef.HIGHER_LAST_SWING_LOW_PRICE: "decimal", FieldRef.HIGHER_LAST_SWING_LOW_KNOWN_AT: "datetime",
})


def _operand_type(operand, parameters):
    if operand.parameter is not None:
        param = parameters.get(operand.parameter.name)
        if param is None:
            raise ValueError(f"Unknown strategy parameter {operand.parameter.name!r}")
        return param.value_type.value
    value = operand.value
    if isinstance(value, Decimal): return "decimal"
    if isinstance(value, datetime): return "datetime"
    if type(value) is int: return "integer"
    if type(value) is bool: return "boolean"
    if type(value) is str: return "string"
    raise ValueError("Unsupported condition literal")


def validate_parameters(condition, parameters):
    for node in walk_conditions(condition):
        if not isinstance(node, Compare):
            continue
        expected, actual = _FIELD_TYPES[node.field], _operand_type(node.operand, parameters)
        if expected != actual:
            raise ValueError(f"Condition field {node.field.value} requires {expected}, got {actual}")
        if expected in ("boolean", "string") and node.operator not in (ComparisonOperator.EQ, ComparisonOperator.NE):
            raise ValueError("Boolean and categorical fields support only EQ and NE")
