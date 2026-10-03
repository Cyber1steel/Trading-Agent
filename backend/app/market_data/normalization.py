"""Strict provider-batch normalization with explicit quality findings."""

from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import Decimal

from app.market_data.contracts import (
    Candle,
    MarketDataRequest,
    NormalizationResult,
    ProviderBatch,
    ProviderSymbolMapping,
    normalized_content_hash,
)
from app.market_data.quality import (
    DataQualityIssue,
    DataQualityReport,
    QualityCode,
    QualitySeverity,
)
from app.market_data.quality_policy import CompletenessPolicy
from app.market_data.timeframes import Timeframe

NORMALIZATION_VERSION = "1"
VALIDATION_VERSION = "1"


def normalize_batch(
    batch: ProviderBatch,
    request: MarketDataRequest,
    symbol_mapping: ProviderSymbolMapping | None,
    *,
    completeness_policy: CompletenessPolicy | None = None,
) -> NormalizationResult:
    issues: list[DataQualityIssue] = []

    def failure(code: QualityCode, message: str, row: int | None = None, ts: datetime | None = None) -> None:
        issues.append(DataQualityIssue(
            severity=QualitySeverity.HARD_FAILURE, code=code, message=message,
            row_index=row, timestamp=ts if ts is not None and ts.tzinfo is not None else None,
        ))

    if batch.instrument != request.instrument:
        failure(QualityCode.INVALID_INSTRUMENT, "Provider batch instrument does not match the requested instrument.")
    if batch.timeframe != request.timeframe:
        failure(QualityCode.INVALID_TIMEFRAME, "Provider batch timeframe does not match the requested timeframe.")
    if batch.requested_start != request.start or batch.requested_end != request.end:
        failure(QualityCode.PROVIDER_METADATA_MISMATCH, "Provider batch bounds do not match the requested range.")
    if batch.timestamp_convention.value != "bar_open":
        failure(QualityCode.PROVIDER_METADATA_MISMATCH, "Only BAR-OPEN provider timestamps are supported.")
    if symbol_mapping is None:
        failure(QualityCode.SYMBOL_MAPPING_MISSING, "Provider symbol requires an explicit mapping.")
    elif (
        symbol_mapping.provider_id != batch.provider_id
        or symbol_mapping.provider_symbol != batch.provider_symbol
        or symbol_mapping.instrument != request.instrument
    ):
        failure(QualityCode.PROVIDER_METADATA_MISMATCH, "Provider symbol mapping does not match batch and request identity.")

    candidates: list[Candle] = []
    seen: set[datetime] = set()
    previous: datetime | None = None
    for row_index, raw in enumerate(batch.candles):
        timestamp = raw.timestamp
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            failure(QualityCode.TIMEZONE_NAIVE, "Provider candle timestamp is timezone-naive.", row_index)
            continue
        try:
            timestamp = timestamp.astimezone(timezone.utc)
        except (OverflowError, ValueError):
            failure(QualityCode.IMPOSSIBLE_TIMESTAMP, "Provider candle timestamp cannot be represented in UTC.", row_index)
            continue
        row_valid = True
        if not request.start <= timestamp < request.end:
            failure(QualityCode.OUT_OF_RANGE, "Candle timestamp is outside requested [start, end) bounds.", row_index, timestamp)
            row_valid = False
        if timestamp in seen:
            failure(QualityCode.DUPLICATE_TIMESTAMP, "Duplicate candle timestamp.", row_index, timestamp)
            row_valid = False
        if previous is not None and timestamp < previous:
            failure(QualityCode.NON_MONOTONIC, "Provider rows are not in strictly increasing timestamp order; input was not sorted.", row_index, timestamp)
            row_valid = False
        if previous is not None and timestamp == previous and timestamp not in seen:
            failure(QualityCode.NON_MONOTONIC, "Provider rows are not strictly increasing.", row_index, timestamp)
            row_valid = False
        seen.add(timestamp)
        previous = timestamp

        values = (raw.open, raw.high, raw.low, raw.close)
        if not all(value.is_finite() for value in values):
            failure(QualityCode.INVALID_OHLC, "OHLC values must be finite decimals.", row_index, timestamp)
            row_valid = False
        elif raw.high < max(raw.open, raw.close) or raw.low > min(raw.open, raw.close) or raw.high < raw.low:
            failure(QualityCode.INVALID_OHLC_RELATIONSHIP, "OHLC values violate high/low relationships.", row_index, timestamp)
            row_valid = False
        if not raw.volume.is_finite() or raw.volume < 0:
            failure(QualityCode.INVALID_VOLUME, "Volume must be finite and non-negative.", row_index, timestamp)
            row_valid = False
        quote_values = (raw.bid, raw.ask, raw.spread)
        if any(value is not None and not value.is_finite() for value in quote_values):
            failure(QualityCode.INVALID_QUOTE, "Bid, ask, and spread must be finite when provided.", row_index, timestamp)
            row_valid = False
        elif (
            (raw.bid is not None and raw.ask is not None and raw.bid > raw.ask)
            or (raw.spread is not None and raw.spread < 0)
            or (
                raw.bid is not None and raw.ask is not None and raw.spread is not None
                and raw.spread != raw.ask - raw.bid
            )
        ):
            failure(QualityCode.INVALID_QUOTE, "Bid/ask/spread values are contradictory.", row_index, timestamp)
            row_valid = False
        if row_valid:
            try:
                candidates.append(Candle(
                    instrument=request.instrument,
                    timeframe=request.timeframe,
                    timestamp=timestamp,
                    open=raw.open, high=raw.high, low=raw.low, close=raw.close,
                    volume=raw.volume, bid=raw.bid, ask=raw.ask, spread=raw.spread,
                ))
                if raw.volume == Decimal(0):
                    issues.append(DataQualityIssue(
                        severity=QualitySeverity.WARNING,
                        code=QualityCode.UNUSUAL_VALUE,
                        message="Zero volume is valid but is retained as a reviewable anomaly.",
                        row_index=row_index,
                        timestamp=timestamp,
                    ))
            except (ValueError, TypeError) as exc:
                failure(QualityCode.INVALID_OHLC, f"Canonical candle validation failed: {exc}", row_index, timestamp)

    ordered = tuple(candidates)
    if not issues or all(issue.severity != QualitySeverity.HARD_FAILURE for issue in issues):
        if completeness_policy is None:
            gap_found = False
            if not request.timeframe.is_calendar_anchored:
                duration = request.timeframe.nominal_duration
                gap_found = any(
                    right.timestamp - left.timestamp > duration
                    for left, right in zip(ordered, ordered[1:])
                )
            if gap_found:
                issues.append(DataQualityIssue(
                    severity=QualitySeverity.WARNING,
                    code=QualityCode.GAP_UNCERTAIN,
                    message="One or more nominal intraday intervals are missing; no authoritative market calendar was supplied.",
                ))
            issues.append(DataQualityIssue(
                severity=QualitySeverity.WARNING,
                code=QualityCode.COMPLETENESS_UNKNOWN,
                message="Dataset completeness is unknown without an authoritative market calendar.",
            ))
        else:
            expected = tuple(completeness_policy.expected_bar_opens(
                request.instrument, request.timeframe, request.start, request.end
            ))
            expected_utc_list = []
            for instant in expected:
                if instant.tzinfo is None or instant.utcoffset() is None:
                    raise ValueError("CompletenessPolicy must return timezone-aware timestamps")
                instant_utc = instant.astimezone(timezone.utc)
                if not request.start <= instant_utc < request.end:
                    raise ValueError("CompletenessPolicy returned an expected bar outside requested bounds")
                expected_utc_list.append(instant_utc)
            if expected_utc_list != sorted(set(expected_utc_list)):
                raise ValueError("CompletenessPolicy timestamps must be unique and strictly ordered")
            expected_utc = set(expected_utc_list)
            observed_utc = {candle.timestamp for candle in ordered}
            unexpected = sorted(observed_utc - expected_utc)
            for instant in unexpected:
                issues.append(DataQualityIssue(
                    severity=QualitySeverity.HARD_FAILURE,
                    code=QualityCode.INVALID_TIMEFRAME,
                    message="Candle timestamp is not expected by the authoritative completeness policy.",
                    timestamp=instant,
                ))
            missing = sorted(expected_utc - observed_utc)
            for instant in missing:
                issues.append(DataQualityIssue(
                    severity=QualitySeverity.HARD_FAILURE,
                    code=QualityCode.MISSING_EXPECTED_BAR,
                    message="An authoritative completeness policy expected a candle at this timestamp.",
                    timestamp=instant,
                ))

    hard_failures_exist = any(issue.severity == QualitySeverity.HARD_FAILURE for issue in issues)
    accepted = () if hard_failures_exist else ordered
    report = DataQualityReport(
        issues=tuple(issues),
        input_record_count=len(batch.candles),
        accepted_record_count=len(accepted),
    )
    return NormalizationResult(
        candles=accepted,
        quality=report,
        content_hash=normalized_content_hash(accepted) if not hard_failures_exist else None,
    )
