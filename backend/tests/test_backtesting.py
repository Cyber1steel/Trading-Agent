from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.backtesting.contracts import BacktestRequest, ExecutionAssumptions
from app.backtesting.execution import simulate_fill
from app.backtesting.engine import BacktestEngine
from app.market_analysis.contracts import AnalysisParameters
from app.market_data.contracts import AssetClass, Instrument, MarketDatasetManifest, PriceBasis, TimestampConvention
from app.market_data.quality import DataQualityReport
from app.market_data.timeframes import Timeframe
from app.strategy.fingerprints import analysis_full_fingerprint, analysis_prefix_fingerprint


def tz_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc)


class FakeObservation:
    def __init__(self, bar_open: datetime, known_at: datetime, open_price: Decimal):
        self.bar_open = bar_open
        self.known_at = known_at
        self.open = open_price

    def model_dump(self, mode: str = "json"):
        return {
            "bar_open": self.bar_open.isoformat(),
            "known_at": self.known_at.isoformat(),
            "open": str(self.open),
        }


def make_fake_analysis(observations, *, analysis_version: str = "2e.1.0"):
    provenance = SimpleNamespace(
        dataset_id=uuid4(),
        dataset_version=1,
        instrument=Instrument(symbol="EURUSD", asset_class=AssetClass.FX),
        timeframe=Timeframe.H1,
        provider_id="fixture",
        provider_version="v1",
        provider_symbol="EURUSD",
        price_basis=PriceBasis.TRADE,
        timestamp_convention=TimestampConvention.BAR_OPEN,
        content_hash="a" * 64,
        quality_warnings=(),
    )
    analysis = SimpleNamespace(
        analysis_version=analysis_version,
        parameters=AnalysisParameters(),
        input_start=observations[0].bar_open,
        analysis_start=observations[0].bar_open,
        analysis_end=observations[-1].bar_open + timedelta(hours=1),
        cutoff_at=observations[-1].known_at,
        inputs=(SimpleNamespace(provenance=provenance),),
        observations=tuple(observations),
    )
    analysis.fingerprint = analysis_full_fingerprint(analysis)
    return analysis


def make_request(analysis, *, start, end, strategy, setup, analysis_prefix=None):
    if analysis_prefix is None:
        analysis_prefix = analysis_prefix_fingerprint(analysis, end)
    return BacktestRequest(
        dataset_id=uuid4(),
        dataset_version=1,
        instrument_id=uuid4(),
        timeframe="1h",
        strategy_id=strategy.strategy_id,
        strategy_version=strategy.strategy_version,
        setup_id=setup.setup_id,
        setup_version=setup.setup_version,
        strategy_fingerprint=strategy.fingerprint,
        setup_fingerprint=setup.fingerprint,
        analysis_fingerprint=analysis.fingerprint,
        analysis_prefix_fingerprint=analysis_prefix,
        start=start,
        end=end,
        initial_capital=Decimal("10000"),
        execution=ExecutionAssumptions(
            spread=Decimal("0.02"),
            slippage=Decimal("0.01"),
            pct_fee=Decimal("0.0"),
            fixed_fee=Decimal("0.0"),
            price_basis=PriceBasis.TRADE,
            fixed_quantity=1,
        ),
        engine_version="2g.1.0",
    )


def make_manifest():
    return MarketDatasetManifest(
        dataset_id=uuid4(),
        dataset_version=1,
        instrument=Instrument(symbol="EURUSD", asset_class=AssetClass.FX),
        timeframe=Timeframe.H1,
        provider_id="fixture",
        provider_version="v1",
        provider_symbol="EURUSD",
        requested_start=tz_utc(datetime(2020, 1, 1, 0, 0)),
        requested_end=tz_utc(datetime(2020, 1, 2, 0, 0)),
        retrieved_at=tz_utc(datetime(2020, 1, 1, 0, 0)),
        ingested_at=tz_utc(datetime(2020, 1, 1, 0, 0)),
        timestamp_convention=TimestampConvention.BAR_OPEN,
        price_basis=PriceBasis.TRADE,
        volume_units="contracts",
        normalization_version="n1",
        validation_version="v1",
        content_hash="a" * 64,
        quality=DataQualityReport(issues=(), input_record_count=1, accepted_record_count=1),
        provider_metadata={},
    )


def test_simulate_fill_applies_spread_slippage_and_fees():
    assumptions = ExecutionAssumptions(
        spread=Decimal("0.02"),
        slippage=Decimal("0.01"),
        pct_fee=Decimal("0.001"),
        fixed_fee=Decimal("0.5"),
        price_basis=PriceBasis.TRADE,
        fixed_quantity=2,
    )
    exec_res = simulate_fill(Decimal("100"), 2, assumptions)
    assert exec_res.requested_price == Decimal("100")
    assert exec_res.executed_price == Decimal("100") + Decimal("0.01") + (Decimal("0.02") / Decimal("2"))
    assert exec_res.fees > 0


def test_future_candle_invariance_and_stale_prefix_rejection(monkeypatch):
    base = tz_utc(datetime(2020, 1, 1, 0, 0))
    observations = [
        FakeObservation(base + timedelta(hours=i), base + timedelta(hours=i, minutes=60), Decimal("100") + Decimal(i))
        for i in range(6)
    ]
    analysis = make_fake_analysis(observations)
    strategy = SimpleNamespace(strategy_id=uuid4(), strategy_version=1, fingerprint="b" * 64)
    setup = SimpleNamespace(setup_id="s1", setup_version=1, fingerprint="c" * 64)

    class MockResult:
        def __init__(self, status_name, lifecycle_states=()):
            self.status = SimpleNamespace(name=status_name)
            self.lifecycle_transitions = [SimpleNamespace(state=SimpleNamespace(name=s)) for s in lifecycle_states]

    def fake_evaluate(analysis_arg, strategy_arg, setup_arg, req):
        idx = next(i for i, o in enumerate(observations) if o.known_at == req.evaluation_at)
        if idx == 0:
            return MockResult("SETUP_CONFIRMED")
        if idx == 3:
            return MockResult("SETUP_NOT_CONFIRMED", lifecycle_states=("INVALIDATED",))
        return MockResult("SETUP_NOT_CONFIRMED")

    monkeypatch.setattr("app.backtesting.engine.evaluate_setup", fake_evaluate)
    range_start = observations[0].bar_open
    range_end = observations[4].bar_open
    request = make_request(analysis, start=range_start, end=range_end, strategy=strategy, setup=setup)
    manifest = make_manifest()
    engine = BacktestEngine()
    result = engine.run(request, manifest, analysis, strategy, setup)

    future_obs = observations + [
        FakeObservation(range_end + timedelta(hours=1), range_end + timedelta(hours=2), Decimal("999")),
        FakeObservation(range_end + timedelta(hours=2), range_end + timedelta(hours=3), Decimal("998")),
    ]
    future_analysis = make_fake_analysis(future_obs)
    future_request = make_request(future_analysis, start=range_start, end=range_end, strategy=strategy, setup=setup)
    future_result = engine.run(future_request, manifest, future_analysis, strategy, setup)

    assert result.trades == future_result.trades
    assert result.events == future_result.events
    assert result.metrics == future_result.metrics
    assert result.result_fingerprint == future_result.result_fingerprint
    assert analysis_prefix_fingerprint(analysis, range_end) == analysis_prefix_fingerprint(future_analysis, range_end)

    mutated_observations = [
        FakeObservation(observations[0].bar_open, observations[0].known_at, Decimal("105"))
    ] + observations[1:]
    mutated_analysis = make_fake_analysis(mutated_observations)
    stale_prefix = analysis_prefix_fingerprint(analysis, range_end)
    assert stale_prefix != analysis_prefix_fingerprint(mutated_analysis, range_end)

    stale_request = make_request(mutated_analysis, start=range_start, end=range_end, strategy=strategy, setup=setup, analysis_prefix=stale_prefix)
    with pytest.raises(ValueError, match="analysis prefix fingerprint mismatch"):
        engine.run(stale_request, manifest, mutated_analysis, strategy, setup)

    malformed = request.model_copy(update={"analysis_prefix_fingerprint": "G" * 64})
    with pytest.raises(ValueError):
        malformed.model_validate(malformed.model_dump())


def test_rejects_invalid_chronology_and_future_higher_timeframe_data():
    base = tz_utc(datetime(2020, 1, 1, 0, 0))
    observations = [
        FakeObservation(base + timedelta(hours=i), base + timedelta(hours=i, minutes=60), Decimal("100") + Decimal(i))
        for i in range(4)
    ]
    analysis = make_fake_analysis(observations)
    strategy = SimpleNamespace(strategy_id=uuid4(), strategy_version=1, fingerprint="b" * 64)
    setup = SimpleNamespace(setup_id="s1", setup_version=1, fingerprint="c" * 64)
    request = make_request(analysis, start=observations[0].bar_open, end=observations[-1].bar_open, strategy=strategy, setup=setup)

    reordered = (observations[2], observations[0], observations[1], observations[3])
    reordered_analysis = make_fake_analysis(reordered)
    with pytest.raises(ValueError, match="chronological order"):
        BacktestEngine().run(request, make_manifest(), reordered_analysis, strategy, setup)

    future_trimmed = observations + [
        FakeObservation(observations[-1].bar_open + timedelta(hours=4), observations[-1].known_at + timedelta(hours=4), Decimal("900"))
    ]
    future_analysis = make_fake_analysis(future_trimmed)
    assert analysis_prefix_fingerprint(analysis, request.end) == analysis_prefix_fingerprint(future_analysis, request.end)


def test_invalid_fingerprint_formats_are_rejected():
    valid = "a" * 64
    with pytest.raises(ValueError):
        BacktestRequest(
            dataset_id=uuid4(),
            dataset_version=1,
            instrument_id=uuid4(),
            timeframe="1h",
            strategy_id=uuid4(),
            strategy_version=1,
            setup_id="s1",
            setup_version=1,
            strategy_fingerprint="A" * 64,
            setup_fingerprint=valid,
            analysis_fingerprint=valid,
            analysis_prefix_fingerprint=valid,
            start=tz_utc(datetime(2020, 1, 1, 0, 0)),
            end=tz_utc(datetime(2020, 1, 1, 1, 0)),
            initial_capital=Decimal("10000"),
            execution=ExecutionAssumptions(
                spread=Decimal("0.02"),
                slippage=Decimal("0.01"),
                pct_fee=Decimal("0.0"),
                fixed_fee=Decimal("0.0"),
                price_basis=PriceBasis.TRADE,
                fixed_quantity=1,
            ),
            engine_version="2g.1.0",
        )

    with pytest.raises(ValueError):
        BacktestRequest(
            dataset_id=uuid4(),
            dataset_version=1,
            instrument_id=uuid4(),
            timeframe="1h",
            strategy_id=uuid4(),
            strategy_version=1,
            setup_id="s1",
            setup_version=1,
            strategy_fingerprint=valid,
            setup_fingerprint="x" * 63,
            analysis_fingerprint=valid,
            analysis_prefix_fingerprint=valid,
            start=tz_utc(datetime(2020, 1, 1, 0, 0)),
            end=tz_utc(datetime(2020, 1, 1, 1, 0)),
            initial_capital=Decimal("10000"),
            execution=ExecutionAssumptions(
                spread=Decimal("0.02"),
                slippage=Decimal("0.01"),
                pct_fee=Decimal("0.0"),
                fixed_fee=Decimal("0.0"),
                price_basis=PriceBasis.TRADE,
                fixed_quantity=1,
            ),
            engine_version="2g.1.0",
        )
