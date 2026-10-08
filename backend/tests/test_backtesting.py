from datetime import datetime, timedelta, timezone
from decimal import Context, Decimal, localcontext
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.backtesting.contracts import BacktestRequest, ExecutionAssumptions, PriceBasis as ExecutionPriceBasis, TradeSide
from app.backtesting.execution import OrderSide, simulate_fill
from app.backtesting.engine import BacktestEngine
from app.market_analysis.contracts import (
    ANALYSIS_VERSION, AnalysisParameters, AnalysisResult, AvailabilityReason, CandleFeatures,
    DatasetProvenance, DecimalValue, MarketObservation, SelectedInput, StructuralState,
)
from app.market_context.sessions import CalendarStatus, SessionClassification
from app.market_data.contracts import AssetClass, Instrument, MarketDatasetManifest, PriceBasis, TimestampConvention
from app.market_data.quality import DataQualityReport
from app.market_data.timeframes import Timeframe
from app.strategy.fingerprints import analysis_full_fingerprint, analysis_prefix_fingerprint
from app.strategy.contracts import (
    Compare, ComparisonOperator, ConditionOperand, ExpiryRule, FieldRef,
    LifecycleState, SetupDefinition, StrategyDefinition,
)


def tz_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc)


def FakeObservation(bar_open: datetime, known_at: datetime, open: Decimal) -> MarketObservation:
    return MarketObservation(
        bar_open=bar_open, known_at=known_at, open=open, high=open + Decimal("1"),
        low=open - Decimal("1"), close=open, volume=Decimal("1"), volume_units="contracts",
        features=CandleFeatures(
            range=Decimal("2"), body=Decimal("0"), upper_wick=Decimal("1"), lower_wick=Decimal("1"),
            body_ratio=DecimalValue(value=Decimal("0")), close_location=DecimalValue(value=Decimal("0.5")),
            simple_return=DecimalValue(value=None, unavailable_reason=AvailabilityReason.NO_PREVIOUS_CLOSE),
        ), true_range=Decimal("2"), atr=DecimalValue(value=Decimal("2")),
        structural_state=StructuralState.INSUFFICIENT, confirmed_swings=(), structural_transition=None,
        higher_timeframes=(), session=SessionClassification(
            timestamp=known_at, labels=(), local_date=known_at.date(), weekend=False,
            calendar_status=CalendarStatus.UNKNOWN,
        ), session_transition=None,
    )


def make_fake_analysis(observations, *, analysis_version: str = "2e.1.0", dataset_id=None):
    instrument = Instrument(symbol="EURUSD", asset_class=AssetClass.FX)
    provenance = DatasetProvenance(
        dataset_id=dataset_id or UUID(int=100),
        dataset_version=1,
        instrument=instrument,
        timeframe=Timeframe.H1,
        provider_id="fixture",
        provider_version="v1",
        provider_symbol="EURUSD",
        price_basis=PriceBasis.TRADE,
        timestamp_convention=TimestampConvention.BAR_OPEN,
        content_hash="a" * 64,
    )
    analysis = AnalysisResult(
        analysis_version=analysis_version,
        parameters=AnalysisParameters(),
        input_start=observations[0].bar_open,
        analysis_start=observations[0].bar_open,
        analysis_end=observations[-1].bar_open + Timeframe.H1.nominal_duration,
        cutoff_at=observations[-1].known_at,
        inputs=(SelectedInput(provenance=provenance, slice_hash="b" * 64),),
        observations=tuple(observations),
        fingerprint="0" * 64,
    )
    return analysis.model_copy(update={"fingerprint": analysis_full_fingerprint(analysis)})


def make_request(analysis, *, start, end, strategy, setup, analysis_prefix=None, side=TradeSide.LONG,
                 conversion=Decimal("1")):
    if analysis_prefix is None:
        analysis_prefix = analysis_prefix_fingerprint(analysis, end)
    return BacktestRequest(
        dataset_id=analysis.inputs[0].provenance.dataset_id,
        dataset_version=1,
        instrument_id=analysis.inputs[0].provenance.instrument.instrument_id,
        side=side,
        timeframe="1h",
        strategy_id=strategy.strategy_id,
        strategy_version=strategy.strategy_version,
        setup_id=setup.setup_id,
        setup_version=setup.setup_version,
        strategy_fingerprint=strategy.fingerprint,
        setup_fingerprint=setup.fingerprint,
        analysis_fingerprint=analysis.fingerprint,
        analysis_prefix_fingerprint=analysis_prefix,
        pnl_to_account_rate=conversion,
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


def make_manifest(analysis=None):
    provenance = analysis.inputs[0].provenance if analysis is not None else None
    instrument = provenance.instrument if provenance is not None else Instrument(symbol="EURUSD", asset_class=AssetClass.FX)
    return MarketDatasetManifest(
        dataset_id=provenance.dataset_id if provenance else UUID(int=100),
        dataset_version=provenance.dataset_version if provenance else 1,
        instrument=instrument,
        timeframe=provenance.timeframe if provenance else Timeframe.H1,
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
        content_hash=provenance.content_hash if provenance else "a" * 64,
        quality=DataQualityReport(issues=(), input_record_count=1, accepted_record_count=1),
        provider_metadata={},
    )


def make_strategy():
    setup = SetupDefinition(
        setup_id="s1", setup_version=1, instrument=Instrument(symbol="EURUSD", asset_class=AssetClass.FX),
        primary_timeframe=Timeframe.H1,
        entry_conditions=Compare(field=FieldRef.CLOSE, operator=ComparisonOperator.GT,
                                 operand=ConditionOperand(value=Decimal("1"))),
        expiry=ExpiryRule(max_later_observations=3),
    )
    strategy = StrategyDefinition(strategy_id=UUID(int=200), strategy_version=1,
                                  analysis_version="2e.1.0", setups=(setup,))
    return strategy, strategy.setups[0]


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
    sell = simulate_fill(Decimal("100"), 2, assumptions, OrderSide.SELL)
    assert sell.executed_price == Decimal("100") - Decimal("0.01") - (Decimal("0.02") / Decimal("2"))
    assert exec_res.spread_cost == Decimal("0.02")
    assert exec_res.slippage_cost == Decimal("0.02")
    assert sell.spread_cost == Decimal("0.02")


def test_future_candle_invariance_and_stale_prefix_rejection(monkeypatch):
    base = tz_utc(datetime(2020, 1, 1, 0, 0))
    observations = [
        FakeObservation(base + timedelta(hours=i), base + timedelta(hours=i, minutes=60), Decimal("100") + Decimal(i))
        for i in range(6)
    ]
    analysis = make_fake_analysis(observations)
    strategy, setup = make_strategy()

    def fake_evaluate(analysis_arg, strategy_arg, setup_arg, req):
        assert req.evaluation_at == range_end
        return SimpleNamespace(lifecycle_transitions=(
            SimpleNamespace(state=LifecycleState.CONFIRMED, known_at=observations[0].known_at,
                            reason="confirmed", evidence=()),
            SimpleNamespace(state=LifecycleState.INVALIDATED, known_at=observations[3].known_at,
                            reason="invalidated", evidence=()),
        ))

    monkeypatch.setattr("app.backtesting.engine.evaluate_setup", fake_evaluate)
    range_start = observations[0].bar_open
    range_end = observations[4].bar_open
    request = make_request(analysis, start=range_start, end=range_end, strategy=strategy, setup=setup)
    manifest = make_manifest(analysis)
    engine = BacktestEngine()
    result = engine.run(request, manifest, analysis, strategy, setup)

    future_obs = observations + [
        FakeObservation(observations[-1].bar_open + timedelta(hours=1),
                       observations[-1].known_at + timedelta(hours=1), Decimal("999")),
        FakeObservation(observations[-1].bar_open + timedelta(hours=2),
                       observations[-1].known_at + timedelta(hours=2), Decimal("998")),
    ]
    future_analysis = make_fake_analysis(future_obs, dataset_id=analysis.inputs[0].provenance.dataset_id)
    future_request = make_request(future_analysis, start=range_start, end=range_end, strategy=strategy, setup=setup)
    future_result = engine.run(future_request, manifest, future_analysis, strategy, setup)

    assert result.trades == future_result.trades
    assert result.events == future_result.events
    assert result.metrics == future_result.metrics
    assert result.result_fingerprint == future_result.result_fingerprint
    assert result.unrealized_pnl == future_result.unrealized_pnl
    assert result.trades[0].request_fingerprint == request.replay_fingerprint
    assert request.replay_fingerprint == future_request.replay_fingerprint
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


def test_backtest_converts_realized_pnl_and_costs_to_account_currency(monkeypatch):
    base = tz_utc(datetime(2020, 1, 1, 0, 0))
    observations = [
        FakeObservation(base + timedelta(hours=i), base + timedelta(hours=i + 1), Decimal("100") + Decimal(i))
        for i in range(6)
    ]
    analysis = make_fake_analysis(observations)
    strategy, setup = make_strategy()

    def evaluate(_analysis, _strategy, _setup, request):
        return SimpleNamespace(lifecycle_transitions=(
            SimpleNamespace(state=LifecycleState.CONFIRMED, known_at=observations[0].known_at,
                            reason="confirmed", evidence=()),
            SimpleNamespace(state=LifecycleState.INVALIDATED, known_at=observations[3].known_at,
                            reason="invalidated", evidence=()),
        ))

    monkeypatch.setattr("app.backtesting.engine.evaluate_setup", evaluate)
    request = make_request(analysis, start=observations[0].bar_open, end=observations[5].bar_open,
                           strategy=strategy, setup=setup, conversion=Decimal("2"))
    request = request.model_copy(update={"execution": request.execution.model_copy(update={
        "pct_fee": Decimal("0.001"), "fixed_fee": Decimal("0.5"),
    })})
    result = BacktestEngine().run(request, make_manifest(analysis), analysis, strategy, setup)

    trade = result.trades[0]
    assert trade.exit_price == Decimal("103.98")
    assert trade.gross_pnl == Decimal("6")
    assert trade.fees == Decimal("2.41000")
    assert trade.net_pnl == Decimal("3.51")
    assert trade.slippage == Decimal("0.04")
    assert trade.spread_cost == Decimal("0.04")
    with localcontext(Context(prec=34)):
        assert trade.return_pct == Decimal("3.51") / (trade.entry_reference_price * Decimal(2))
    assert result.ending_equity == request.initial_capital + Decimal("3.51")
    assert result.unrealized_pnl == Decimal(0)


def test_confirmation_observation_cannot_also_exit_new_position(monkeypatch):
    base = tz_utc(datetime(2020, 1, 1, 0, 0))
    observations = [
        FakeObservation(base + timedelta(hours=i), base + timedelta(hours=i + 1), Decimal("100") + Decimal(i))
        for i in range(4)
    ]
    analysis = make_fake_analysis(observations)
    strategy, setup = make_strategy()

    def evaluate(_analysis, _strategy, _setup, request):
        return SimpleNamespace(lifecycle_transitions=(
            SimpleNamespace(state=LifecycleState.CONFIRMED, known_at=observations[0].known_at,
                            reason="confirmed", evidence=()),
            SimpleNamespace(state=LifecycleState.INVALIDATED, known_at=observations[0].known_at,
                            reason="invalidated", evidence=()),
        ))

    monkeypatch.setattr("app.backtesting.engine.evaluate_setup", evaluate)
    request = make_request(analysis, start=observations[0].bar_open, end=observations[3].bar_open,
                           strategy=strategy, setup=setup)
    result = BacktestEngine().run(request, make_manifest(analysis), analysis, strategy, setup)

    assert result.trades == ()
    assert any(event["type"] == "ENTRY_CANCELLED" for event in result.events)


def test_real_setup_evaluator_drives_chronological_backtest():
    base = tz_utc(datetime(2020, 1, 1, 0, 0))
    instrument = Instrument(symbol="EURUSD", asset_class=AssetClass.FX)
    dataset_id = UUID(int=400)
    provenance = DatasetProvenance(
        dataset_id=dataset_id, dataset_version=1, instrument=instrument, timeframe=Timeframe.H1,
        content_hash="a" * 64, provider_id="fixture", provider_version="v1", provider_symbol="EURUSD",
        price_basis=PriceBasis.TRADE, timestamp_convention=TimestampConvention.BAR_OPEN,
    )
    candles = []
    for index, close in enumerate(map(Decimal, ("9", "11", "12", "13", "14"))):
        opened = base + timedelta(hours=index)
        known = opened + timedelta(hours=1)
        candles.append(MarketObservation(
            bar_open=opened, known_at=known, open=close, high=close + Decimal("1"),
            low=close - Decimal("1"), close=close, volume=Decimal("1"), volume_units="contracts",
            features=CandleFeatures(
                range=Decimal("2"), body=Decimal("0"), upper_wick=Decimal("1"), lower_wick=Decimal("1"),
                body_ratio=DecimalValue(value=Decimal("0")), close_location=DecimalValue(value=Decimal("0.5")),
                simple_return=DecimalValue(value=None, unavailable_reason=AvailabilityReason.NO_PREVIOUS_CLOSE)
                if index == 0 else DecimalValue(value=Decimal("0.1")),
            ), true_range=Decimal("2"), atr=DecimalValue(value=Decimal("2")),
            structural_state=StructuralState.INSUFFICIENT, confirmed_swings=(),
            structural_transition=None, higher_timeframes=(),
            session=SessionClassification(timestamp=known, labels=(), local_date=known.date(), weekend=False,
                                          calendar_status=CalendarStatus.UNKNOWN),
            session_transition=None,
        ))
    end = base + timedelta(hours=5)
    analysis = AnalysisResult(
        analysis_version=ANALYSIS_VERSION, parameters=AnalysisParameters(), input_start=base,
        analysis_start=base, analysis_end=end, cutoff_at=end,
        inputs=(SelectedInput(provenance=provenance, slice_hash="b" * 64),),
        observations=tuple(candles), fingerprint="0" * 64,
    )
    analysis = analysis.model_copy(update={"fingerprint": analysis_full_fingerprint(analysis)})
    setup = SetupDefinition(
        setup_id="s1", setup_version=1, instrument=instrument, primary_timeframe=Timeframe.H1,
        entry_conditions=Compare(field=FieldRef.CLOSE, operator=ComparisonOperator.GT,
                                 operand=ConditionOperand(value=Decimal("10"))),
        invalidation_conditions=Compare(field=FieldRef.CLOSE, operator=ComparisonOperator.GT,
                                        operand=ConditionOperand(value=Decimal("12"))),
        expiry=ExpiryRule(max_later_observations=4),
    )
    strategy = StrategyDefinition(strategy_id=UUID(int=401), strategy_version=1,
                                  analysis_version=ANALYSIS_VERSION, setups=(setup,))
    request = make_request(analysis, start=base, end=end, strategy=strategy, setup=setup)

    result = BacktestEngine().run(request, make_manifest(analysis), analysis, strategy, setup)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_bar_open == candles[2].bar_open
    assert trade.exit_bar_open == candles[4].bar_open
    assert trade.entry_reference_price == Decimal("12")
    assert trade.exit_reference_price == Decimal("14")
    assert trade.gross_pnl == Decimal("2")
    assert trade.net_pnl == Decimal("1.96")
    transition_events = [event for event in result.events if event["type"] == "SETUP_STATE_TRANSITION"]
    assert [event["payload"]["state"] for event in transition_events] == ["CANDIDATE", "CONFIRMED", "INVALIDATED"]
    assert all(event["payload"]["evidence"] for event in transition_events)


def test_rejects_invalid_chronology_and_future_higher_timeframe_data():
    base = tz_utc(datetime(2020, 1, 1, 0, 0))
    observations = [
        FakeObservation(base + timedelta(hours=i), base + timedelta(hours=i, minutes=60), Decimal("100") + Decimal(i))
        for i in range(4)
    ]
    analysis = make_fake_analysis(observations)
    strategy, setup = make_strategy()
    request = make_request(analysis, start=observations[0].bar_open, end=observations[-1].bar_open, strategy=strategy, setup=setup)

    reordered = (observations[2], observations[0], observations[1], observations[3])
    reordered_analysis = make_fake_analysis(reordered, dataset_id=analysis.inputs[0].provenance.dataset_id)
    reordered_request = make_request(reordered_analysis, start=reordered[0].bar_open, end=request.end,
                                     strategy=strategy, setup=setup)
    with pytest.raises(ValueError, match="strictly chronological"):
        BacktestEngine().run(reordered_request, make_manifest(reordered_analysis), reordered_analysis, strategy, setup)

    future_trimmed = observations + [
        FakeObservation(observations[-1].bar_open + timedelta(hours=4), observations[-1].known_at + timedelta(hours=4), Decimal("900"))
    ]
    future_analysis = make_fake_analysis(future_trimmed, dataset_id=analysis.inputs[0].provenance.dataset_id)
    assert analysis_prefix_fingerprint(analysis, request.end) == analysis_prefix_fingerprint(future_analysis, request.end)


def test_backtest_rejects_inconsistent_source_provenance_and_price_basis(monkeypatch):
    base = tz_utc(datetime(2020, 1, 1, 0, 0))
    observations = [
        FakeObservation(base + timedelta(hours=i), base + timedelta(hours=i + 1), Decimal("100") + Decimal(i))
        for i in range(4)
    ]
    analysis = make_fake_analysis(observations)
    strategy, setup = make_strategy()
    request = make_request(analysis, start=observations[0].bar_open,
                           end=observations[3].bar_open, strategy=strategy, setup=setup)
    monkeypatch.setattr("app.backtesting.engine.evaluate_setup",
                        lambda *_args: SimpleNamespace(lifecycle_transitions=()))

    altered_provenance = analysis.inputs[0].provenance.model_copy(update={"provider_id": "unexpected"})
    altered_input = analysis.inputs[0].model_copy(update={"provenance": altered_provenance})
    altered_analysis = analysis.model_copy(update={"inputs": (altered_input,), "fingerprint": "0" * 64})
    altered_analysis = altered_analysis.model_copy(update={
        "fingerprint": analysis_full_fingerprint(altered_analysis),
    })
    altered_request = make_request(altered_analysis, start=request.start, end=request.end,
                                   strategy=strategy, setup=setup)
    with pytest.raises(ValueError, match="provenance does not match"):
        BacktestEngine().run(altered_request, make_manifest(analysis), altered_analysis, strategy, setup)

    wrong_execution = request.execution.model_copy(update={"price_basis": ExecutionPriceBasis.BID})
    wrong_request = request.model_copy(update={"execution": wrong_execution})
    with pytest.raises(ValueError, match="price basis"):
        BacktestEngine().run(wrong_request, make_manifest(analysis), analysis, strategy, setup)


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
