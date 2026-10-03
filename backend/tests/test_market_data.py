from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.market_context.sessions import (
    CalendarStatus,
    SessionClassifier,
    SessionDefinition,
    generic_session_definitions,
)
from app.market_data.contracts import (
    AssetClass,
    Candle,
    Instrument,
    MarketDataRequest,
    PriceBasis,
    ProviderBatch,
    ProviderCandle,
    ProviderSymbolMapping,
    TimestampConvention,
    normalized_content_hash,
)
from app.market_data.errors import MarketDataValidationError
from app.market_data.normalization import normalize_batch
from app.market_data.providers.local import LocalFixtureProvider
from app.market_data.quality import QualityCode, QualitySeverity
from app.market_data.repository import MarketDataRepository
from app.market_data.service import MarketDataService
from app.market_data.timeframes import Timeframe


UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
END = START + timedelta(hours=1)
RETRIEVED = datetime(2024, 1, 2, tzinfo=UTC)
INSTRUMENT = Instrument(symbol=" btc-usd ", asset_class=AssetClass.CRYPTO, venue="test")
REQUEST = MarketDataRequest(instrument=INSTRUMENT, timeframe=Timeframe.M1, start=START, end=END)
MAPPING = ProviderSymbolMapping(
    provider_id="local-fixture", provider_symbol="BTC-USD", instrument=INSTRUMENT
)


def provider_candle(at: datetime, *, volume: Decimal = Decimal("2"), **overrides) -> ProviderCandle:
    values = dict(
        timestamp=at,
        open=Decimal("10"),
        high=Decimal("12"),
        low=Decimal("9"),
        close=Decimal("11"),
        volume=volume,
    )
    values.update(overrides)
    return ProviderCandle(**values)


def batch(candles, **overrides) -> ProviderBatch:
    values = dict(
        provider_id="local-fixture",
        provider_version="1",
        provider_symbol="BTC-USD",
        instrument=INSTRUMENT,
        timeframe=Timeframe.M1,
        requested_start=START,
        requested_end=END,
        retrieved_at=RETRIEVED,
        timestamp_convention=TimestampConvention.BAR_OPEN,
        price_basis=PriceBasis.TRADE,
        volume_units="contracts",
        candles=tuple(candles),
    )
    values.update(overrides)
    return ProviderBatch(**values)


def test_instrument_normalization_identity_and_invalid_symbols():
    assert INSTRUMENT.symbol == "BTC-USD"
    assert INSTRUMENT.instrument_id == Instrument(
        symbol="BTC-USD", asset_class="crypto", venue="TEST"
    ).instrument_id
    with pytest.raises(ValidationError):
        Instrument(symbol="two words", asset_class="equity")


@pytest.mark.parametrize(
    ("text", "duration"),
    [
        ("1m", timedelta(minutes=1)), ("5m", timedelta(minutes=5)),
        ("15m", timedelta(minutes=15)), ("30m", timedelta(minutes=30)),
        ("1h", timedelta(hours=1)), ("4h", timedelta(hours=4)),
        ("1d", timedelta(days=1)), ("1w", timedelta(days=7)),
    ],
)
def test_timeframe_parse_and_nominal_duration(text, duration):
    timeframe = Timeframe.parse(text.upper())
    assert timeframe.value == text and timeframe.nominal_duration == duration
    assert Timeframe.parse(timeframe) is timeframe


def test_timeframe_rejects_unsupported_and_daily_weekly_are_calendar_anchored():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        Timeframe.parse("2h")
    assert Timeframe.D1.is_calendar_anchored and Timeframe.W1.is_calendar_anchored
    assert not Timeframe.H4.is_calendar_anchored


def test_candle_requires_aware_utc_timestamp_and_serializes_decimal_deterministically():
    with pytest.raises(ValidationError, match="timezone-aware"):
        Candle(instrument=INSTRUMENT, timeframe="1m", timestamp=datetime(2024, 1, 1),
               open="1", high="1", low="1", close="1", volume="0")
    candle = Candle(instrument=INSTRUMENT, timeframe="1m", timestamp=START,
                    open=Decimal("1.2300"), high=Decimal("2"), low=Decimal("1"),
                    close=Decimal("1.5"), volume=Decimal("0"))
    assert candle.timestamp.tzinfo == UTC
    assert '"open":"1.2300"' in candle.deterministic_json()
    assert candle.deterministic_json() == candle.deterministic_json()


@pytest.mark.parametrize(
    "values",
    [
        {"open": "NaN"}, {"high": "9"}, {"low": "12"}, {"close": "Infinity"},
        {"volume": "-1"}, {"volume": "NaN"},
        {"bid": "12", "ask": "11"}, {"spread": "-1"},
        {"bid": "10", "ask": "12", "spread": "1"},
    ],
)
def test_candle_rejects_invalid_ohlc_volume_and_quotes(values):
    baseline = dict(open=Decimal("10"), high=Decimal("12"), low=Decimal("9"),
                    close=Decimal("11"), volume=Decimal("1"))
    baseline.update(values)
    with pytest.raises(ValidationError):
        Candle(instrument=INSTRUMENT, timeframe="1m", timestamp=START, **baseline)


def test_request_rejects_naive_and_empty_ranges():
    with pytest.raises(ValidationError, match="timezone-aware"):
        MarketDataRequest(instrument=INSTRUMENT, timeframe="1m", start=datetime(2024, 1, 1), end=END)
    with pytest.raises(ValidationError, match="start < end"):
        MarketDataRequest(instrument=INSTRUMENT, timeframe="1m", start=START, end=START)


def test_normalization_converts_to_utc_and_hash_is_stable():
    local_offset = timezone(timedelta(hours=2))
    at = datetime(2024, 1, 1, 2, tzinfo=local_offset)
    result = normalize_batch(batch([provider_candle(at)]), REQUEST, MAPPING)
    assert result.quality.can_persist
    assert result.candles[0].timestamp == START
    assert result.content_hash == normalized_content_hash(result.candles)
    again = normalize_batch(batch([provider_candle(at)]), REQUEST, MAPPING)
    assert again.content_hash == result.content_hash
    assert result.quality.warnings
    assert result.quality.warnings[0].code == QualityCode.COMPLETENESS_UNKNOWN


def test_normalization_requires_explicit_provider_symbol_mapping():
    result = normalize_batch(batch([provider_candle(START)]), REQUEST, None)
    assert not result.quality.can_persist and not result.candles
    assert result.quality.hard_failures[0].code == QualityCode.SYMBOL_MAPPING_MISSING


def test_normalization_rejects_provider_symbol_mismatch():
    wrong = MAPPING.model_copy(update={"provider_symbol": "XBTUSD"})
    result = normalize_batch(batch([provider_candle(START)]), REQUEST, wrong)
    assert result.quality.hard_failures[0].code == QualityCode.PROVIDER_METADATA_MISMATCH


def test_normalization_rejects_timeframe_mismatch_and_non_bar_open_convention():
    wrong_timeframe = normalize_batch(
        batch([provider_candle(START)], timeframe=Timeframe.H1), REQUEST, MAPPING
    )
    assert any(issue.code == QualityCode.INVALID_TIMEFRAME for issue in wrong_timeframe.quality.hard_failures)
    wrong_convention = normalize_batch(
        batch([provider_candle(START)], timestamp_convention=TimestampConvention.BAR_CLOSE),
        REQUEST,
        MAPPING,
    )
    assert not wrong_convention.quality.can_persist


def test_zero_volume_is_valid_but_reported_as_an_anomaly():
    result = normalize_batch(
        batch([provider_candle(START, volume=Decimal("0"))]), REQUEST, MAPPING
    )
    assert result.quality.can_persist
    assert any(
        issue.severity == QualitySeverity.WARNING and issue.code == QualityCode.UNUSUAL_VALUE
        for issue in result.quality.issues
    )


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        ([provider_candle(datetime(2024, 1, 1))], QualityCode.TIMEZONE_NAIVE),
        ([provider_candle(END)], QualityCode.OUT_OF_RANGE),
        ([provider_candle(START), provider_candle(START)], QualityCode.DUPLICATE_TIMESTAMP),
        ([provider_candle(START + timedelta(minutes=1)), provider_candle(START)], QualityCode.NON_MONOTONIC),
        ([provider_candle(START, high=Decimal("8"))], QualityCode.INVALID_OHLC_RELATIONSHIP),
        ([provider_candle(START, volume=Decimal("-1"))], QualityCode.INVALID_VOLUME),
        ([provider_candle(START, bid=Decimal("4"), ask=Decimal("3"))], QualityCode.INVALID_QUOTE),
    ],
)
def test_normalization_hard_failures_never_return_partial_or_reordered_data(rows, code):
    result = normalize_batch(batch(rows), REQUEST, MAPPING)
    assert not result.quality.can_persist
    assert result.candles == () and result.quality.accepted_record_count == 0
    assert code in {issue.code for issue in result.quality.hard_failures}


def test_normalization_does_not_silently_drop_or_sort_records():
    rows = [provider_candle(START), provider_candle(START - timedelta(minutes=1))]
    result = normalize_batch(batch(rows), REQUEST, MAPPING)
    assert result.quality.input_record_count == 2
    assert result.quality.accepted_record_count == 0
    assert any(issue.code == QualityCode.NON_MONOTONIC for issue in result.quality.issues)


def test_gaps_are_warnings_without_calendar_and_hard_only_with_authoritative_policy():
    rows = [provider_candle(START), provider_candle(START + timedelta(minutes=3))]
    result = normalize_batch(batch(rows), REQUEST, MAPPING)
    assert result.quality.can_persist
    assert any(issue.code == QualityCode.GAP_UNCERTAIN for issue in result.quality.warnings)

    class EveryMinute:
        def expected_bar_opens(self, instrument, timeframe, start, end):
            return (START, START + timedelta(minutes=1), START + timedelta(minutes=2))

    strict = normalize_batch(batch(rows), REQUEST, MAPPING, completeness_policy=EveryMinute())
    assert not strict.quality.can_persist
    assert len([i for i in strict.quality.hard_failures if i.code == QualityCode.MISSING_EXPECTED_BAR]) == 2


def test_local_fixture_provider_is_deterministic_and_in_memory():
    fixture = (provider_candle(START), provider_candle(START + timedelta(minutes=1)))
    provider = LocalFixtureProvider({(INSTRUMENT.instrument_id, Timeframe.M1): fixture})
    first = provider.fetch_historical(REQUEST)
    second = provider.fetch_historical(REQUEST)
    assert first.deterministic_json() == second.deterministic_json()
    assert first.provider_id == "local-fixture"
    assert first.provider_metadata["kind"] == "local_fixture"
    assert len(first.candles) == 2


def test_market_data_service_persists_only_valid_snapshot_and_keeps_warnings():
    class FakeRepository:
        def __init__(self): self.saved = None
        def create_snapshot(self, manifest, candles): self.saved = manifest, candles

    provider = LocalFixtureProvider({(INSTRUMENT.instrument_id, Timeframe.M1): (provider_candle(START),)})
    repository = FakeRepository()
    manifest = MarketDataService(provider, repository).ingest_historical(REQUEST, symbol_mapping=MAPPING)
    assert repository.saved is not None
    assert manifest.content_hash == normalized_content_hash(repository.saved[1])
    assert manifest.quality.warnings
    assert manifest.provider_id == "local-fixture" and manifest.ingested_at.tzinfo is not None

    class InvalidProvider:
        provider_id = "local-fixture"
        provider_version = "1"
        def fetch_historical(self, request):
            return batch([provider_candle(END)])

    repository.saved = None
    with pytest.raises(MarketDataValidationError):
        MarketDataService(InvalidProvider(), repository).ingest_historical(REQUEST, symbol_mapping=MAPPING)
    assert repository.saved is None


def test_session_classification_labels_and_overlaps():
    classifier = SessionClassifier(generic_session_definitions(), calendar_timezone="UTC")
    asia = classifier.classify(datetime(2024, 1, 15, 1, tzinfo=UTC))
    overlap = classifier.classify(datetime(2024, 1, 15, 13, tzinfo=UTC))
    assert asia.labels == ("Asia",)
    assert overlap.labels == ("London", "New York")
    assert overlap.calendar_status == CalendarStatus.UNKNOWN


def test_session_timezone_dst_and_weekend_unknown_calendar():
    definition = SessionDefinition(
        name="London sample", timezone="Europe/London", local_start="01:00", local_end="03:00",
        weekdays=(6,),
    )
    classifier = SessionClassifier((definition,), calendar_timezone="Europe/London")
    before_dst = classifier.classify(datetime(2024, 3, 31, 0, 30, tzinfo=UTC))
    after_dst = classifier.classify(datetime(2024, 3, 31, 1, 30, tzinfo=UTC))
    assert before_dst.labels == ()
    assert after_dst.labels == ("London sample",)
    holiday_like = classifier.classify(datetime(2024, 12, 25, 10, tzinfo=UTC))
    assert holiday_like.calendar_status == CalendarStatus.UNKNOWN
    weekend = classifier.classify(datetime(2024, 12, 28, 12, tzinfo=UTC))
    assert weekend.weekend and weekend.calendar_status == CalendarStatus.UNKNOWN


def test_session_rejects_naive_time_and_timezone():
    classifier = SessionClassifier((), calendar_timezone="UTC")
    with pytest.raises(ValueError, match="timezone-aware"):
        classifier.classify(datetime(2024, 1, 1))
    with pytest.raises(ValidationError, match="IANA timezone"):
        SessionDefinition(name="bad", timezone="Mars/Olympus", local_start="09:00", local_end="10:00")


def test_repository_range_query_is_explicit_dataset_scoped_and_ordered():
    from types import SimpleNamespace
    from sqlalchemy.dialects.postgresql import dialect as pg_dialect
    from app.market_data.quality import DataQualityReport

    class Rows:
        def scalars(self): return self
        def all(self): return []

    class FakeSession:
        statement = None
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def get(self, model, key):
            return SimpleNamespace(
                dataset_id=key[0], dataset_version=key[1], instrument_id=INSTRUMENT.instrument_id,
                timeframe="1m", provider_id="local-fixture", provider_version="1",
                provider_symbol="BTC-USD", requested_start=START, requested_end=END,
                retrieved_at=RETRIEVED, ingested_at=RETRIEVED, timestamp_convention="bar_open",
                price_basis="trade", volume_units="contracts", normalization_version="1",
                validation_version="1", content_hash="a" * 64,
                quality_summary=DataQualityReport(
                    input_record_count=0, accepted_record_count=0
                ).model_dump(mode="json"),
                provider_metadata={},
            )
        def execute(self, statement):
            self.statement = str(statement.compile(dialect=pg_dialect()))
            return Rows()

    session = FakeSession()
    dataset_id = uuid4()
    result = MarketDataRepository(lambda: session).get_range(
        dataset_id=dataset_id, dataset_version=1, instrument=INSTRUMENT,
        timeframe=Timeframe.M1, start=START, end=END,
    )
    sql = session.statement.lower()
    assert result.manifest.dataset_id == dataset_id and result.candles == ()
    assert "market_candles.dataset_id =" in sql
    assert "market_candles.dataset_version =" in sql
    assert "market_candles.timestamp >=" in sql and "market_candles.timestamp <" in sql
    assert "order by market_candles.timestamp asc" in sql
