"""Stable quality issue codes and reports for market-data ingestion."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class QualitySeverity(str, Enum):
    HARD_FAILURE = "hard_failure"
    WARNING = "warning"


class QualityCode(str, Enum):
    INVALID_INSTRUMENT = "invalid_instrument"
    INVALID_TIMEFRAME = "invalid_timeframe"
    PROVIDER_METADATA_MISMATCH = "provider_metadata_mismatch"
    SYMBOL_MAPPING_MISSING = "symbol_mapping_missing"
    TIMEZONE_NAIVE = "timezone_naive_timestamp"
    IMPOSSIBLE_TIMESTAMP = "impossible_timestamp"
    OUT_OF_RANGE = "timestamp_outside_requested_range"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    NON_MONOTONIC = "non_monotonic_input"
    INVALID_OHLC = "invalid_ohlc_value"
    INVALID_OHLC_RELATIONSHIP = "invalid_ohlc_relationship"
    INVALID_VOLUME = "invalid_volume"
    INVALID_QUOTE = "contradictory_quote_data"
    GAP_UNCERTAIN = "gap_without_authoritative_calendar"
    MISSING_EXPECTED_BAR = "missing_expected_bar"
    COMPLETENESS_UNKNOWN = "completeness_unknown"
    UNUSUAL_VALUE = "unusual_but_valid_value"


class DataQualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: QualitySeverity
    code: QualityCode
    message: str
    row_index: int | None = Field(default=None, ge=0)
    timestamp: datetime | None = None

    @field_validator("timestamp")
    @classmethod
    def canonical_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Quality issue timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)


class DataQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issues: tuple[DataQualityIssue, ...] = ()
    input_record_count: int = Field(ge=0)
    accepted_record_count: int = Field(ge=0)

    @model_validator(mode="after")
    def accepted_count_within_input(self) -> "DataQualityReport":
        if self.accepted_record_count > self.input_record_count:
            raise ValueError("accepted_record_count cannot exceed input_record_count")
        return self

    @property
    def hard_failures(self) -> tuple[DataQualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == QualitySeverity.HARD_FAILURE)

    @property
    def warnings(self) -> tuple[DataQualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == QualitySeverity.WARNING)

    @property
    def can_persist(self) -> bool:
        return not self.hard_failures

    def deterministic_json(self) -> str:
        import json

        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
