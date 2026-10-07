from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
from uuid import uuid4

import pytest

from app.market_analysis.contracts import (
    ANALYSIS_VERSION, AnalysisParameters, AnalysisRequest, ConfirmedSwing, DatasetRef,
    StructuralState, SwingKind,
)
from app.market_analysis.errors import AnalysisCutoffError, AnalysisDataError
from app.market_analysis.features import candle_features
from app.market_analysis.service import MarketAnalysisService
from app.market_analysis.structure import classify_swings
from app.market_analysis.swings import detect_swings
from app.market_analysis.volatility import average_true_ranges, true_ranges
from app.market_data.contracts import (
    AssetClass, Candle, Instrument, MarketDataSlice, MarketDatasetManifest,
    PriceBasis, TimestampConvention,
)
from app.market_data.quality import DataQualityReport
from app.market_data.timeframes import Timeframe

UTC = timezone.utc
BASE = datetime(2025, 1, 6, tzinfo=UTC)
INSTRUMENT = Instrument(symbol="TEST", asset_class=AssetClass.EQUITY, venue="X")


def decimal(value):
    if isinstance(value, float):
        raise TypeError("Use Decimal-native test values; floats are not accepted")
    return value if isinstance(value, Decimal) else Decimal(value)


def candle(i, high, low, close=None, timeframe=Timeframe.M1):
    high = decimal(high)
    low = decimal(low)
    close = (high + low) / Decimal(2) if close is None else decimal(close)
    return Candle(instrument=INSTRUMENT, timeframe=timeframe, timestamp=BASE + i * timeframe.nominal_duration,
                  open=close, high=high, low=low, close=close,
                  volume=Decimal("10"))


def test_features_decimal_availability_and_values():
    item = candle(0, 12, 8, 11)
    feature = candle_features(item, Decimal("10"))
    assert feature.range == Decimal("4")
    assert feature.body == 0
    assert feature.upper_wick == 1
    assert feature.lower_wick == 3
    assert feature.body_ratio.value == 0
    assert feature.close_location.value == Decimal("0.75")
    assert feature.simple_return.value == Decimal("0.1")
    assert candle_features(item).simple_return.unavailable_reason.value == "no_previous_close"


def test_zero_range_and_nonpositive_return_are_explicit():
    flat = candle(0, 5, 5, 5)
    features = candle_features(flat, Decimal("0"))
    assert features.body_ratio.value is None
    assert features.body_ratio.unavailable_reason.value == "zero_range"
    assert features.close_location.value is None
    assert features.simple_return.unavailable_reason.value == "non_positive_close"


def test_calculations_do_not_change_process_decimal_context():
    prior = getcontext().copy()
    candle_features(candle(0, 5, 1, 3), Decimal("2"))
    true_ranges((candle(0, 5, 1, 3), candle(1, 6, 2, 4)))
    current = getcontext()
    assert current.prec == prior.prec
    assert current.rounding == prior.rounding
    assert current.traps == prior.traps
    assert current.flags == prior.flags


def test_true_range_and_wilder_atr():
    values = (candle(0, 12, 10, 11), candle(1, 14, 12, 13), candle(2, 15, 13, 14))
    tr = true_ranges(values)
    assert tr == (Decimal("2"), Decimal("3"), Decimal("2"))
    atr = average_true_ranges(tr, 2)
    assert atr[0].value is None
    assert atr[1].value == Decimal("2.5")
    assert atr[2].value == Decimal("2.25")


def test_gap_true_range_uses_previous_close():
    values = (candle(0, 12, 10, 10), candle(1, 16, 15, 16))
    assert true_ranges(values)[1] == Decimal("6")


def test_swings_are_strict_and_known_only_after_right_window_closes():
    values = tuple(candle(i, h, l) for i, (h, l) in enumerate(((2, 1), (4, 2), (9, 4), (5, 3), (4, 2))))
    swings = detect_swings(values, 2, 2)
    high = next(event for event in swings if event.kind == SwingKind.HIGH)
    assert high.candidate_at == values[2].timestamp
    assert high.known_at == values[4].timestamp + Timeframe.M1.nominal_duration
    assert high.known_at > high.candidate_at
    tied = tuple(candle(i, h, l) for i, (h, l) in enumerate(((2, 1), (8, 2), (8, 3), (5, 4), (4, 2))))
    assert not any(event.kind == SwingKind.HIGH for event in detect_swings(tied, 1, 1))
    assert detect_swings(values[:4], 2, 2) == ()


def test_obvious_swing_low_and_both_boundary_windows():
    values = tuple(candle(i, h, l) for i, (h, l) in enumerate(((8, 6), (7, 4), (6, 1), (7, 5), (8, 6))))
    swings = detect_swings(values, 2, 2)
    low = next(event for event in swings if event.kind == SwingKind.LOW)
    assert low.candidate_at == values[2].timestamp
    assert not any(event.candidate_at in (values[0].timestamp, values[-1].timestamp) for event in swings)
    assert detect_swings(values[:3], 2, 2) == ()


def test_equal_lows_are_not_arbitrarily_selected():
    values = tuple(candle(i, h, l) for i, (h, l) in enumerate(((8, 6), (7, 3), (6, 3), (7, 5), (8, 6))))
    assert not any(event.kind == SwingKind.LOW for event in detect_swings(values, 1, 1))


def test_analysis_request_utc_ranges_and_explicit_timeframes():
    ref = DatasetRef(dataset_id=uuid4(), dataset_version=1, instrument=INSTRUMENT, timeframe=Timeframe.M1)
    request = AnalysisRequest(primary=ref, input_start=BASE,
        analysis_start=BASE.astimezone(timezone(timedelta(hours=2))),
        analysis_end=BASE + timedelta(minutes=5), cutoff_at=BASE + timedelta(minutes=5))
    assert request.analysis_start == BASE
    with pytest.raises(ValueError):
        AnalysisRequest(primary=ref, input_start=BASE.replace(tzinfo=None), analysis_start=BASE,
            analysis_end=BASE + timedelta(minutes=1), cutoff_at=BASE + timedelta(minutes=1))
    with pytest.raises(ValueError):
        AnalysisRequest(primary=ref, higher_timeframes=(ref,), input_start=BASE, analysis_start=BASE,
            analysis_end=BASE + timedelta(minutes=1), cutoff_at=BASE + timedelta(minutes=1))
    daily = ref.model_copy(update={"timeframe": Timeframe.D1})
    with pytest.raises(ValueError):
        AnalysisRequest(primary=daily, input_start=BASE, analysis_start=BASE,
            analysis_end=BASE + timedelta(days=1), cutoff_at=BASE + timedelta(days=1))
    with pytest.raises(ValueError):
        AnalysisRequest(primary=ref, input_start=BASE + timedelta(minutes=1), analysis_start=BASE,
            analysis_end=BASE + timedelta(minutes=1), cutoff_at=BASE + timedelta(minutes=1))
    lower_parent = ref.model_copy(update={"timeframe": Timeframe.M5})
    primary_15m = ref.model_copy(update={"timeframe": Timeframe.M15})
    with pytest.raises(ValueError):
        AnalysisRequest(primary=primary_15m, higher_timeframes=(lower_parent,), input_start=BASE,
            analysis_start=BASE, analysis_end=BASE + timedelta(minutes=15), cutoff_at=BASE + timedelta(minutes=15))


@pytest.mark.parametrize("version", [0, -1, "1", 1.5, 1.0])
def test_dataset_version_requires_a_strict_positive_integer(version):
    with pytest.raises(ValueError):
        DatasetRef(dataset_id=uuid4(), dataset_version=version,
                   instrument=INSTRUMENT, timeframe=Timeframe.M1)


def test_dataset_version_accepts_positive_integer():
    assert DatasetRef(dataset_id=uuid4(), dataset_version=1,
                      instrument=INSTRUMENT, timeframe=Timeframe.M1).dataset_version == 1


class FakeRepository:
    def __init__(self, selected):
        self.selected = selected
        self.calls = []

    def get_range(self, **kwargs):
        self.calls.append(kwargs)
        return self.selected.model_copy(update={"candles": tuple(
            candle for candle in self.selected.candles
            if kwargs["start"] <= candle.timestamp < kwargs["end"]
        )})


def dataset(candles, dataset_id, timeframe=Timeframe.M1):
    content_hash = "a" * 64
    manifest = MarketDatasetManifest(
        dataset_id=dataset_id, dataset_version=1, instrument=INSTRUMENT, timeframe=timeframe,
        provider_id="fixture", provider_version="1", provider_symbol="TEST",
        requested_start=BASE, requested_end=BASE + timedelta(hours=1), retrieved_at=BASE,
        ingested_at=BASE, timestamp_convention=TimestampConvention.BAR_OPEN,
        price_basis=PriceBasis.TRADE, volume_units="shares", normalization_version="1",
        validation_version="1", content_hash=content_hash,
        quality=DataQualityReport(input_record_count=len(candles), accepted_record_count=len(candles)),
    )
    return MarketDataSlice(manifest=manifest, candles=candles)


def test_service_is_deterministic_explicitly_scoped_and_cutoff_safe():
    bars = tuple(candle(i, 12 + i, 9 + i, 11 + i) for i in range(8))
    identity = uuid4()
    repo = FakeRepository(dataset(bars, identity))
    service = MarketAnalysisService(repo)
    request = AnalysisRequest(
        primary=DatasetRef(dataset_id=identity, dataset_version=1, instrument=INSTRUMENT, timeframe=Timeframe.M1),
        input_start=BASE, analysis_start=BASE, analysis_end=BASE + timedelta(minutes=4),
        cutoff_at=BASE + timedelta(minutes=4), parameters=AnalysisParameters(atr_period=2),
    )
    result = service.analyze(request)
    again = service.analyze(request)
    assert result.fingerprint == again.fingerprint
    assert result.analysis_version == ANALYSIS_VERSION
    assert len(result.observations) == 4
    assert result.observations[0].known_at == BASE + timedelta(minutes=1)
    assert result.observations[0].session.timestamp == result.observations[0].known_at
    assert result.observations[0].session_transition is None
    assert result.observations[1].session_transition is not None
    assert len(repo.calls) == 2
    invalid = request.model_copy(update={"cutoff_at": BASE + timedelta(minutes=2)})
    with pytest.raises(AnalysisCutoffError):
        service.analyze(invalid)


def test_future_candles_do_not_change_earlier_observations_or_events():
    primary_id, parent_id = uuid4(), uuid4()
    primary_ref = DatasetRef(dataset_id=primary_id, dataset_version=1,
        instrument=INSTRUMENT, timeframe=Timeframe.M1)
    parent_ref = DatasetRef(dataset_id=parent_id, dataset_version=1,
        instrument=INSTRUMENT, timeframe=Timeframe.M5)
    request = AnalysisRequest(primary=primary_ref, higher_timeframes=(parent_ref,),
        input_start=BASE, analysis_start=BASE, analysis_end=BASE + timedelta(minutes=20),
        cutoff_at=BASE + timedelta(minutes=20),
        parameters=AnalysisParameters(atr_period=2, swing_left=2, swing_right=2))
    baseline_primary = tuple(candle(i, h, l) for i, (h, l) in enumerate(
        ((4, 2), (5, 3), (6, 4))))
    adversarial_primary = baseline_primary + tuple(candle(i, h, l) for i, (h, l) in enumerate(
        ((7, 4), (100, 5), (8, 3), (7, 2), (6, 1)), start=3))
    baseline_parent = (candle(0, 15, 10, timeframe=Timeframe.M5),)
    adversarial_parent = baseline_parent + tuple(candle(i, h, l, timeframe=Timeframe.M5)
        for i, (h, l) in enumerate(((40, 12), (20, 11), (18, 9)), start=1))

    class SnapshotRepository:
        def __init__(self, primary_bars, parent_bars):
            self.snapshots = {
                primary_id: (primary_bars, Timeframe.M1),
                parent_id: (parent_bars, Timeframe.M5),
            }

        def get_range(self, **kwargs):
            bars, timeframe = self.snapshots[kwargs["dataset_id"]]
            selected = tuple(item for item in bars if kwargs["start"] <= item.timestamp < kwargs["end"])
            return dataset(selected, kwargs["dataset_id"], timeframe)

    baseline = MarketAnalysisService(SnapshotRepository(baseline_primary, baseline_parent)).analyze(request)
    adversarial = MarketAnalysisService(SnapshotRepository(adversarial_primary, adversarial_parent)).analyze(request)
    chosen_known_at = BASE + timedelta(minutes=3)
    before = tuple(item.model_dump(mode="json") for item in baseline.observations
                   if item.known_at <= chosen_known_at)
    after = tuple(item.model_dump(mode="json") for item in adversarial.observations
                  if item.known_at <= chosen_known_at)
    assert before == after
    assert adversarial.observations[2].higher_timeframes[0].bar_open is None
    future_candidate = BASE + timedelta(minutes=4)
    future_confirmation = BASE + timedelta(minutes=7)
    assert any(event.candidate_at == future_candidate and event.known_at == future_confirmation
               for item in adversarial.observations for event in item.confirmed_swings)


def test_parent_swing_is_visible_only_after_parent_confirmation_close():
    primary_id, parent_id = uuid4(), uuid4()
    primary_bars = tuple(candle(i, 12 + i, 8 + i) for i in range(22))
    parent_bars = tuple(candle(i, h, l, timeframe=Timeframe.M5) for i, (h, l) in enumerate(
        ((15, 10), (40, 12), (20, 11), (18, 9), (17, 8))))
    request = AnalysisRequest(
        primary=DatasetRef(dataset_id=primary_id, dataset_version=1,
            instrument=INSTRUMENT, timeframe=Timeframe.M1),
        higher_timeframes=(DatasetRef(dataset_id=parent_id, dataset_version=1,
            instrument=INSTRUMENT, timeframe=Timeframe.M5),),
        input_start=BASE, analysis_start=BASE, analysis_end=BASE + timedelta(minutes=22),
        cutoff_at=BASE + timedelta(minutes=22),
        parameters=AnalysisParameters(atr_period=1, swing_left=1, swing_right=2),
    )

    class SnapshotRepository:
        def get_range(self, **kwargs):
            bars = primary_bars if kwargs["dataset_id"] == primary_id else parent_bars
            return dataset(tuple(item for item in bars
                                 if kwargs["start"] <= item.timestamp < kwargs["end"]),
                           kwargs["dataset_id"], kwargs["timeframe"])

    result = MarketAnalysisService(SnapshotRepository()).analyze(request)
    candidate = BASE + timedelta(minutes=5)
    known = BASE + timedelta(minutes=20)
    before = next(item for item in result.observations if item.known_at == BASE + timedelta(minutes=10))
    at_confirmation = next(item for item in result.observations if item.known_at == known)
    assert not any(event.candidate_at == candidate
                   for event in before.higher_timeframes[0].confirmed_swings)
    confirmed = [event for event in at_confirmation.higher_timeframes[0].confirmed_swings
                 if event.candidate_at == candidate]
    assert len(confirmed) == 1
    assert confirmed[0].confirmed_at == known
    assert confirmed[0].known_at == known


def test_parent_close_and_cutoff_equality_vs_one_microsecond_before():
    primary_id, parent_id = uuid4(), uuid4()
    parent_close = BASE + timedelta(minutes=5)
    parent_bars = (candle(0, 20, 10, timeframe=Timeframe.M5),)
    parent_ref = DatasetRef(dataset_id=parent_id, dataset_version=1,
        instrument=INSTRUMENT, timeframe=Timeframe.M5)

    class SnapshotRepository:
        def __init__(self, primary_bar):
            self.primary_bar = primary_bar

        def get_range(self, **kwargs):
            bars = (self.primary_bar,) if kwargs["dataset_id"] == primary_id else parent_bars
            return dataset(tuple(item for item in bars
                                 if kwargs["start"] <= item.timestamp < kwargs["end"]),
                           kwargs["dataset_id"], kwargs["timeframe"])

    exact_open = parent_close - Timeframe.M1.nominal_duration
    exact_primary = candle(0, 6, 4).model_copy(update={"timestamp": exact_open})
    exact_request = AnalysisRequest(
        primary=DatasetRef(dataset_id=primary_id, dataset_version=1,
            instrument=INSTRUMENT, timeframe=Timeframe.M1),
        higher_timeframes=(parent_ref,), input_start=BASE, analysis_start=exact_open,
        analysis_end=exact_open + timedelta(microseconds=1), cutoff_at=parent_close,
        parameters=AnalysisParameters(atr_period=1),
    )
    exact = MarketAnalysisService(SnapshotRepository(exact_primary)).analyze(exact_request)
    assert exact.observations[0].known_at == parent_close
    assert exact.observations[0].higher_timeframes[0].bar_open == BASE

    before_open = exact_open - timedelta(microseconds=1)
    before_primary = candle(0, 6, 4).model_copy(update={"timestamp": before_open})
    before_close = parent_close - timedelta(microseconds=1)
    before_request = exact_request.model_copy(update={
        "analysis_start": before_open,
        "analysis_end": before_open + timedelta(microseconds=1),
        "cutoff_at": before_close,
    })
    before = MarketAnalysisService(SnapshotRepository(before_primary)).analyze(before_request)
    assert before.observations[0].known_at == before_close
    assert before.observations[0].higher_timeframes[0].bar_open is None


def test_service_rejects_wrong_manifest_identity():
    wrong_id, requested_id = uuid4(), uuid4()
    repo = FakeRepository(dataset((candle(0, 3, 1),), wrong_id))
    service = MarketAnalysisService(repo)
    request = AnalysisRequest(primary=DatasetRef(dataset_id=requested_id, dataset_version=1,
        instrument=INSTRUMENT, timeframe=Timeframe.M1), input_start=BASE, analysis_start=BASE,
        analysis_end=BASE + timedelta(minutes=1), cutoff_at=BASE + timedelta(minutes=1))
    with pytest.raises(AnalysisDataError):
        service.analyze(request)


@pytest.mark.parametrize("manifest_updates", [
    {"dataset_version": 2},
    {"instrument": Instrument(symbol="OTHER", asset_class=AssetClass.EQUITY)},
    {"timeframe": Timeframe.M5},
    {"timestamp_convention": TimestampConvention.BAR_CLOSE},
])
def test_service_rejects_version_instrument_timeframe_and_convention_mismatch(manifest_updates):
    identity = uuid4()
    selected = dataset((candle(0, 4, 2),), identity)
    selected = selected.model_copy(update={"manifest": selected.manifest.model_copy(update=manifest_updates)})
    request = AnalysisRequest(primary=DatasetRef(dataset_id=identity, dataset_version=1,
        instrument=INSTRUMENT, timeframe=Timeframe.M1), input_start=BASE, analysis_start=BASE,
        analysis_end=BASE + timedelta(minutes=1), cutoff_at=BASE + timedelta(minutes=1))
    with pytest.raises(AnalysisDataError):
        MarketAnalysisService(FakeRepository(selected)).analyze(request)


def test_parameters_and_dataset_identity_affect_fingerprint():
    bars = (candle(0, 4, 2), candle(1, 5, 3), candle(2, 6, 4))
    identity = uuid4()
    request = AnalysisRequest(primary=DatasetRef(dataset_id=identity, dataset_version=1,
        instrument=INSTRUMENT, timeframe=Timeframe.M1), input_start=BASE, analysis_start=BASE,
        analysis_end=BASE + timedelta(minutes=2), cutoff_at=BASE + timedelta(minutes=2),
        parameters=AnalysisParameters(atr_period=1))
    first = MarketAnalysisService(FakeRepository(dataset(bars, identity))).analyze(request)
    changed_parameters = request.model_copy(update={"parameters": AnalysisParameters(atr_period=2)})
    second = MarketAnalysisService(FakeRepository(dataset(bars, identity))).analyze(changed_parameters)
    changed_identity = request.model_copy(update={"primary": request.primary.model_copy(update={"dataset_id": uuid4()})})
    third = MarketAnalysisService(FakeRepository(dataset(bars, changed_identity.primary.dataset_id))).analyze(changed_identity)
    assert len({first.fingerprint, second.fingerprint, third.fingerprint}) == 3
    assert first.inputs[0].slice_hash == second.inputs[0].slice_hash


def test_selected_candle_change_changes_slice_hash_and_result_fingerprint():
    bars = (candle(0, 4, 2), candle(1, 5, 3), candle(2, 6, 4))
    identity = uuid4()
    request = AnalysisRequest(primary=DatasetRef(dataset_id=identity, dataset_version=1,
        instrument=INSTRUMENT, timeframe=Timeframe.M1), input_start=BASE, analysis_start=BASE,
        analysis_end=BASE + timedelta(minutes=3), cutoff_at=BASE + timedelta(minutes=3),
        parameters=AnalysisParameters(atr_period=1))
    changed = (bars[0].model_copy(update={"high": Decimal("7")}), *bars[1:])
    original_result = MarketAnalysisService(FakeRepository(dataset(bars, identity))).analyze(request)
    changed_result = MarketAnalysisService(FakeRepository(dataset(changed, identity))).analyze(request)
    assert original_result.inputs[0].slice_hash != changed_result.inputs[0].slice_hash
    assert original_result.fingerprint != changed_result.fingerprint


def test_mixed_structure_does_not_claim_direction():
    # The pure classifier returns an explicit insufficient state before enough highs/lows exist.
    _, transitions, state = classify_swings(())
    assert state == StructuralState.INSUFFICIENT
    assert transitions == ()


def test_structure_direction_comparisons_and_transition_are_confirmed_causally():
    events = (
        ConfirmedSwing(kind=SwingKind.HIGH, candidate_at=BASE, confirmed_at=BASE, known_at=BASE, price=Decimal("10")),
        ConfirmedSwing(kind=SwingKind.LOW, candidate_at=BASE, confirmed_at=BASE, known_at=BASE, price=Decimal("5")),
        ConfirmedSwing(kind=SwingKind.HIGH, candidate_at=BASE, confirmed_at=BASE, known_at=BASE + timedelta(minutes=1), price=Decimal("11")),
        ConfirmedSwing(kind=SwingKind.LOW, candidate_at=BASE, confirmed_at=BASE, known_at=BASE + timedelta(minutes=1), price=Decimal("6")),
        ConfirmedSwing(kind=SwingKind.HIGH, candidate_at=BASE, confirmed_at=BASE, known_at=BASE + timedelta(minutes=2), price=Decimal("9")),
        ConfirmedSwing(kind=SwingKind.LOW, candidate_at=BASE, confirmed_at=BASE, known_at=BASE + timedelta(minutes=3), price=Decimal("4")),
    )
    tagged, transitions, state = classify_swings(events)
    assert tagged[2].comparison.value == "higher_high"
    assert tagged[3].comparison.value == "higher_low"
    assert tagged[4].comparison.value == "lower_high"
    assert state == StructuralState.DOWN
    assert len(transitions) == 1
    assert transitions[0].known_at == BASE + timedelta(minutes=3)


def test_up_mixed_and_equal_structure_labels():
    def swing(kind, minute, price):
        stamp = BASE + timedelta(minutes=minute)
        return ConfirmedSwing(kind=kind, candidate_at=stamp, confirmed_at=stamp,
                              known_at=stamp, price=decimal(price))

    up = (swing(SwingKind.HIGH, 0, 10), swing(SwingKind.LOW, 1, 5),
          swing(SwingKind.HIGH, 2, 11), swing(SwingKind.LOW, 3, 6))
    assert classify_swings(up)[2] == StructuralState.UP
    mixed = up + (swing(SwingKind.HIGH, 4, 9),)
    assert classify_swings(mixed)[2] == StructuralState.MIXED
    equal = (swing(SwingKind.HIGH, 0, 10), swing(SwingKind.LOW, 1, 5),
             swing(SwingKind.HIGH, 2, 10), swing(SwingKind.LOW, 3, 5))
    tagged, _, state = classify_swings(equal)
    assert tagged[2].comparison.value == "equal_high"
    assert tagged[3].comparison.value == "equal_low"
    assert state == StructuralState.MIXED


def test_future_swing_confirmation_is_not_visible_early():
    bars = tuple(candle(i, h, l) for i, (h, l) in enumerate(((3, 2), (9, 5), (5, 3), (4, 2))))
    identity = uuid4()
    service = MarketAnalysisService(FakeRepository(dataset(bars, identity)))
    request = AnalysisRequest(primary=DatasetRef(dataset_id=identity, dataset_version=1,
        instrument=INSTRUMENT, timeframe=Timeframe.M1), input_start=BASE, analysis_start=BASE,
        analysis_end=BASE + timedelta(minutes=4), cutoff_at=BASE + timedelta(minutes=4),
        parameters=AnalysisParameters(swing_left=1, swing_right=2, atr_period=1))
    result = service.analyze(request)
    candidate_time = bars[1].timestamp
    confirmation = bars[3].timestamp + Timeframe.M1.nominal_duration
    assert not any(s.candidate_at == candidate_time for o in result.observations if o.known_at < confirmation
                   for s in o.confirmed_swings)
    assert any(s.candidate_at == candidate_time for o in result.observations if o.known_at == confirmation
               for s in o.confirmed_swings)


def test_parent_alignment_allows_exact_close_and_excludes_future_close():
    primary = tuple(candle(i, 3 + i, 1 + i) for i in range(6))
    parent = (candle(0, 20, 10, timeframe=Timeframe.M5),)
    primary_id, parent_id = uuid4(), uuid4()
    primary_ref = DatasetRef(dataset_id=primary_id, dataset_version=1, instrument=INSTRUMENT, timeframe=Timeframe.M1)
    parent_ref = DatasetRef(dataset_id=parent_id, dataset_version=1, instrument=INSTRUMENT, timeframe=Timeframe.M5)
    request = AnalysisRequest(primary=primary_ref, higher_timeframes=(parent_ref,), input_start=BASE,
        analysis_start=BASE, analysis_end=BASE + timedelta(minutes=6), cutoff_at=BASE + timedelta(minutes=6),
        parameters=AnalysisParameters(atr_period=1))

    class TwoDatasets:
        def get_range(self, **kwargs):
            chosen = primary if kwargs["timeframe"] == Timeframe.M1 else parent
            return dataset(tuple(c for c in chosen if kwargs["start"] <= c.timestamp < kwargs["end"]),
                           kwargs["dataset_id"], kwargs["timeframe"])

    result = MarketAnalysisService(TwoDatasets()).analyze(request)
    assert result.observations[3].higher_timeframes[0].bar_open is None  # base known at minute 4
    assert result.observations[4].higher_timeframes[0].bar_open == BASE  # parent closes exactly at minute 5
    assert result.observations[5].higher_timeframes[0].bar_open == BASE


@pytest.mark.parametrize("manifest_updates", [
    {"provider_id": "other-provider"},
    {"price_basis": PriceBasis.MID},
])
def test_higher_timeframe_provider_and_price_basis_must_match(manifest_updates):
    primary_id, parent_id = uuid4(), uuid4()
    primary_ref = DatasetRef(dataset_id=primary_id, dataset_version=1, instrument=INSTRUMENT, timeframe=Timeframe.M1)
    parent_ref = DatasetRef(dataset_id=parent_id, dataset_version=1, instrument=INSTRUMENT, timeframe=Timeframe.M5)
    request = AnalysisRequest(primary=primary_ref, higher_timeframes=(parent_ref,), input_start=BASE,
        analysis_start=BASE, analysis_end=BASE + timedelta(minutes=1), cutoff_at=BASE + timedelta(minutes=1))

    class InconsistentRepository:
        def get_range(self, **kwargs):
            tf = kwargs["timeframe"]
            selected = dataset((candle(0, 4, 2, timeframe=tf),), kwargs["dataset_id"], tf)
            if tf == Timeframe.M5:
                selected = selected.model_copy(update={
                    "manifest": selected.manifest.model_copy(update=manifest_updates)
                })
            return selected

    with pytest.raises(AnalysisDataError):
        MarketAnalysisService(InconsistentRepository()).analyze(request)
