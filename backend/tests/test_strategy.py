from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.market_analysis.contracts import (
    ANALYSIS_VERSION, AnalysisParameters, AnalysisResult, AvailabilityReason,
    CandleFeatures, ConfirmedSwing, DatasetProvenance, DecimalValue, HigherTimeframeContext, MarketObservation,
    StructuralTransition,
    SelectedInput, StructuralState,
)
from app.market_context.sessions import CalendarStatus, SessionClassification
from app.market_data.contracts import AssetClass, Instrument
from app.market_data.timeframes import Timeframe
from app.strategy import (
    AllOf, AnyOf, Compare, ComparisonOperator, ConditionOperand,
    EvaluationRequest, EvaluationStatus, ExpiryRule, FieldRef, Not,
    RequiredMarketContext, SetupDefinition, StrategyDefinition, StrategyParameter, ParameterType,
    TruthValue, evaluate_setup,
)
from app.strategy.conditions import ResolvedField, evaluate_condition
from app.strategy.fingerprints import analysis_full_fingerprint, analysis_prefix_fingerprint
from app.strategy.contracts import EvidenceReference, LifecycleState
from app.strategy.errors import StrategyEvaluationError

UTC = timezone.utc
START = datetime(2025, 1, 6, tzinfo=UTC)
INSTRUMENT = Instrument(symbol="TEST", asset_class=AssetClass.EQUITY, venue="X")


def cmp(field, operator, value, timeframe=None):
    return Compare(field=field, operator=operator, operand=ConditionOperand(value=value), timeframe=timeframe)


def definition(*, entry=None, confirm=None, invalidate=None, expiry=3, parameters=(), higher=(), context=None):
    setup = SetupDefinition(
        setup_id="breakout", setup_version=1, instrument=INSTRUMENT,
        primary_timeframe=Timeframe.M1, required_higher_timeframes=higher,
        required_context=context or RequiredMarketContext(),
        entry_conditions=entry or cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal("10")),
        confirmation_conditions=confirm, invalidation_conditions=invalidate,
        expiry=ExpiryRule(max_later_observations=expiry),
    )
    strategy = StrategyDefinition(strategy_id=UUID(int=5), strategy_version=1,
                                  analysis_version=ANALYSIS_VERSION, setups=(setup,), parameters=parameters)
    return strategy, strategy.setups[0]


def analysis(closes=(Decimal("9"), Decimal("11"), Decimal("12"))):
    dataset_id = UUID(int=1)
    provenance = DatasetProvenance(
        dataset_id=dataset_id, dataset_version=1, instrument=INSTRUMENT, timeframe=Timeframe.M1,
        content_hash="a" * 64, provider_id="fixture", provider_version="1", provider_symbol="TEST",
        price_basis="trade", timestamp_convention="bar_open",
    )
    observations = []
    previous = None
    for index, close in enumerate(closes):
        close = Decimal(close)
        opened = START + index * Timeframe.M1.nominal_duration
        known = opened + Timeframe.M1.nominal_duration
        observations.append(MarketObservation(
            bar_open=opened, known_at=known, open=close, high=close + Decimal("1"),
            low=close - Decimal("1"), close=close, volume=Decimal("20"), volume_units="shares",
            features=CandleFeatures(
                range=Decimal("2"), body=Decimal("0"), upper_wick=Decimal("1"), lower_wick=Decimal("1"),
                body_ratio=DecimalValue(value=Decimal("0")), close_location=DecimalValue(value=Decimal("0.5")),
                simple_return=DecimalValue(value=Decimal("0.1") if previous is not None else None,
                    unavailable_reason=None if previous is not None else AvailabilityReason.NO_PREVIOUS_CLOSE),
            ), true_range=Decimal("2"), atr=DecimalValue(value=Decimal("2")),
            structural_state=StructuralState.INSUFFICIENT, confirmed_swings=(), structural_transition=None,
            higher_timeframes=(), session=SessionClassification(timestamp=known, labels=("London",),
                local_date=known.date(), weekend=False, calendar_status=CalendarStatus.UNKNOWN),
            session_transition=None,
        ))
        previous = close
    result = AnalysisResult(
        analysis_version=ANALYSIS_VERSION, parameters=AnalysisParameters(), input_start=START,
        analysis_start=START, analysis_end=START + len(observations) * Timeframe.M1.nominal_duration,
        cutoff_at=START + len(observations) * Timeframe.M1.nominal_duration,
        inputs=(SelectedInput(provenance=provenance, slice_hash="b" * 64),),
        observations=tuple(observations), fingerprint="0" * 64,
    )
    return result.model_copy(update={"fingerprint": analysis_full_fingerprint(result)})


def refingerprint(result):
    return result.model_copy(update={"fingerprint": analysis_full_fingerprint(result)})


def _decision_evidence(item):
    return item.model_dump(mode="json", exclude={"analysis_fingerprint"})


def request_for(result, strategy, setup, at):
    return EvaluationRequest(
        strategy_id=strategy.strategy_id, strategy_version=strategy.strategy_version,
        strategy_fingerprint=strategy.fingerprint, setup_id=setup.setup_id,
        setup_version=setup.setup_version, setup_fingerprint=setup.fingerprint,
        analysis_fingerprint=result.fingerprint,
        evaluation_prefix_fingerprint=analysis_prefix_fingerprint(result, at), evaluation_at=at,
    )


def test_frozen_contracts_and_strict_identity_versions():
    strategy, setup = definition()
    with pytest.raises(ValidationError):
        setup.setup_version = 2
    with pytest.raises(ValidationError):
        SetupDefinition(setup_id="x", setup_version=True, instrument=INSTRUMENT,
            primary_timeframe=Timeframe.M1, entry_conditions=cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal(1)),
            expiry=ExpiryRule(max_later_observations=2))
    with pytest.raises(ValidationError):
        StrategyDefinition(strategy_id=strategy.strategy_id, strategy_version=0,
            analysis_version=ANALYSIS_VERSION, setups=(setup,))
    with pytest.raises(ValidationError):
        SetupDefinition(setup_id="x", setup_version=1, instrument=INSTRUMENT,
            primary_timeframe=Timeframe.M1, entry_conditions=cmp(FieldRef.CLOSE, ComparisonOperator.GT, 1.5),
            expiry=ExpiryRule(max_later_observations=2))


def test_parameter_types_references_and_duplicate_names_are_validated():
    with pytest.raises(ValidationError):
        StrategyParameter(name="threshold", value_type=ParameterType.DECIMAL, value=1.5)
    param = StrategyParameter(name="threshold", value_type=ParameterType.DECIMAL, value=Decimal("10"))
    condition = Compare(field=FieldRef.CLOSE, operator=ComparisonOperator.GT,
                        operand=ConditionOperand(parameter={"name": "threshold"}))
    strategy, _ = definition(entry=condition, parameters=(param,))
    assert strategy.parameters[0].value == Decimal("10")
    with pytest.raises(ValidationError):
        definition(entry=condition, parameters=(param, param))
    with pytest.raises(ValidationError):
        definition(entry=Compare(field=FieldRef.SESSION_WEEKEND, operator=ComparisonOperator.EQ,
                   operand=ConditionOperand(value=Decimal("1"))))
    with pytest.raises(ValidationError):
        definition(entry=Compare(field=FieldRef.CLOSE, operator=ComparisonOperator.GT,
                   operand=ConditionOperand(parameter={"name": "missing"})))


@pytest.mark.parametrize("operator,left,right,expected", [
    (ComparisonOperator.EQ, Decimal("2"), Decimal("2"), TruthValue.TRUE),
    (ComparisonOperator.NE, Decimal("2"), Decimal("3"), TruthValue.TRUE),
    (ComparisonOperator.GT, Decimal("3"), Decimal("2"), TruthValue.TRUE),
    (ComparisonOperator.GTE, Decimal("2"), Decimal("2"), TruthValue.TRUE),
    (ComparisonOperator.LT, Decimal("1"), Decimal("2"), TruthValue.TRUE),
    (ComparisonOperator.LTE, Decimal("2"), Decimal("2"), TruthValue.TRUE),
])
def test_comparison_operators(operator, left, right, expected):
    strategy, _ = definition()
    outcome, _ = evaluate_condition(cmp(FieldRef.CLOSE, operator, right), strategy,
                                   lambda _: ResolvedField(left))
    assert outcome is expected


def test_three_valued_logic_and_nested_conditions_preserve_unavailability():
    strategy, _ = definition()
    true = cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal("1"))
    false = cmp(FieldRef.CLOSE, ComparisonOperator.LT, Decimal("1"))
    unknown = cmp(FieldRef.ATR, ComparisonOperator.GT, Decimal("1"))
    resolver = lambda node: ResolvedField(None, "ATR warmup unavailable") if node.field is FieldRef.ATR else ResolvedField(Decimal("2"))
    assert evaluate_condition(AllOf(conditions=(false, unknown)), strategy, resolver)[0] is TruthValue.FALSE
    assert evaluate_condition(AllOf(conditions=(true, unknown)), strategy, resolver)[0] is TruthValue.INSUFFICIENT
    assert evaluate_condition(AnyOf(conditions=(true, unknown)), strategy, resolver)[0] is TruthValue.TRUE
    assert evaluate_condition(AnyOf(conditions=(false, unknown)), strategy, resolver)[0] is TruthValue.INSUFFICIENT
    assert evaluate_condition(Not(condition=unknown), strategy, resolver)[0] is TruthValue.INSUFFICIENT


def test_evaluation_requires_exact_time_and_produces_provenance_evidence():
    result = analysis()
    strategy, setup = definition()
    at = result.observations[1].known_at
    evaluated = evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, at))
    assert evaluated.status is EvaluationStatus.SETUP_CONFIRMED
    assert evaluated.evidence
    evidence = evaluated.evidence[0]
    assert evidence.dataset_id == result.inputs[0].provenance.dataset_id
    assert evidence.dataset_version == 1
    assert all(item.known_at <= at for item in evaluated.evidence)
    assert evaluated.condition_trace[-1].known_at <= at
    assert evidence.analysis_fingerprint == result.fingerprint
    assert evidence.evaluation_prefix_fingerprint == analysis_prefix_fingerprint(result, evidence.known_at)
    with pytest.raises(ValueError, match="timezone-aware"):
        request_for(result, strategy, setup, at.replace(tzinfo=None))
    with pytest.raises(Exception, match="exactly identify"):
        evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, at + timedelta(microseconds=1)))


def test_required_context_is_traced_and_unknown_calendar_is_insufficient():
    result = analysis((Decimal("11"),))
    strategy, setup = definition(context=RequiredMarketContext(session_labels=("London",),
                                                               calendar_status=CalendarStatus.OPEN))
    evaluated = evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, result.observations[0].known_at))
    assert evaluated.status is EvaluationStatus.INSUFFICIENT_EVIDENCE
    context = next(trace for trace in evaluated.condition_trace if trace.evidence[0].context_requirement)
    assert context.outcome is TruthValue.INSUFFICIENT
    assert context.evidence[-1].unavailable_reason == "calendar status is unknown"


def test_left_censored_entry_is_insufficient_but_false_to_true_transition_can_start():
    strategy, setup = definition()
    one_bar = analysis((Decimal("11"),))
    result = evaluate_setup(one_bar, strategy, setup,
        request_for(one_bar, strategy, setup, one_bar.observations[0].known_at))
    assert result.status is EvaluationStatus.INSUFFICIENT_EVIDENCE
    assert result.lifecycle_transitions == ()

    bars = analysis((Decimal("9"), Decimal("11")))
    result = evaluate_setup(bars, strategy, setup,
        request_for(bars, strategy, setup, bars.observations[1].known_at))
    assert result.status is EvaluationStatus.SETUP_CONFIRMED
    assert result.lifecycle_transitions[0].state is LifecycleState.CANDIDATE
    assert result.lifecycle_transitions[0].known_at == bars.observations[1].known_at


def test_analysis_full_fingerprint_is_verified_and_prefix_tracks_visible_content():
    result = analysis((Decimal("9"), Decimal("11"), Decimal("12")))
    strategy, setup = definition()
    cutoff = result.observations[1].known_at
    request = request_for(result, strategy, setup, cutoff)

    stale = result.model_copy(update={"observations": (*result.observations[:-1],
        result.observations[-1].model_copy(update={"close": Decimal("999")}))})
    with pytest.raises(StrategyEvaluationError, match="fingerprint does not match"):
        evaluate_setup(stale, strategy, setup, request)

    changed_visible = refingerprint(result.model_copy(update={
        "observations": (result.observations[0], result.observations[1].model_copy(update={"close": Decimal("12")}),
                         result.observations[2]),
    }))
    changed_request = request_for(changed_visible, strategy, setup, cutoff)
    assert changed_request.evaluation_prefix_fingerprint != request.evaluation_prefix_fingerprint
    assert changed_request.fingerprint != request.fingerprint
    original_result = evaluate_setup(result, strategy, setup, request)
    changed_result = evaluate_setup(changed_visible, strategy, setup, changed_request)
    assert changed_result.fingerprint != original_result.fingerprint


def test_evaluation_request_rejects_wrong_prefix_identity():
    result = analysis()
    strategy, setup = definition()
    at = result.observations[1].known_at
    request = request_for(result, strategy, setup, at).model_copy(
        update={"evaluation_prefix_fingerprint": "f" * 64}
    )
    with pytest.raises(StrategyEvaluationError, match="exact supplied inputs"):
        evaluate_setup(result, strategy, setup, request)


def test_evidence_rejects_mutable_or_float_values():
    result = analysis((Decimal("9"), Decimal("11")))
    source = result.inputs[0].provenance
    common = dict(dataset_id=source.dataset_id, dataset_version=source.dataset_version,
        instrument=source.instrument, timeframe=source.timeframe, bar_open=START,
        known_at=START + Timeframe.M1.nominal_duration, analysis_version=ANALYSIS_VERSION,
        analysis_fingerprint=result.fingerprint, evaluation_prefix_fingerprint="a" * 64,
        field=FieldRef.CLOSE, condition_id="condition", value=Decimal("11"))
    for mutable in (["x"], {"x": 1}, {"x"}, 1.5):
        with pytest.raises(ValidationError):
            EvidenceReference(**common, comparison_value=mutable)
    evidence = EvidenceReference(**common, comparison_value=("London", "New York"))
    assert evidence.comparison_value == ("London", "New York")


@pytest.mark.parametrize("dataset_version", [True, False, 0, -1, "1", 1.0])
def test_evidence_reference_requires_strict_positive_dataset_version(dataset_version):
    result = analysis((Decimal("9"), Decimal("11")))
    source = result.inputs[0].provenance
    common = dict(dataset_id=source.dataset_id, instrument=source.instrument,
        timeframe=source.timeframe, bar_open=START,
        known_at=START + Timeframe.M1.nominal_duration, analysis_version=ANALYSIS_VERSION,
        analysis_fingerprint=result.fingerprint, evaluation_prefix_fingerprint="a" * 64,
        field=FieldRef.CLOSE, value=Decimal("11"), condition_id="condition")
    with pytest.raises(ValidationError):
        EvidenceReference(**common, dataset_version=dataset_version)


def test_evidence_reference_accepts_positive_integer_dataset_version():
    result = analysis((Decimal("9"), Decimal("11")))
    source = result.inputs[0].provenance
    evidence = EvidenceReference(dataset_id=source.dataset_id, dataset_version=1,
        instrument=source.instrument, timeframe=source.timeframe, bar_open=START,
        known_at=START + Timeframe.M1.nominal_duration, analysis_version=ANALYSIS_VERSION,
        analysis_fingerprint=result.fingerprint, evaluation_prefix_fingerprint="a" * 64,
        field=FieldRef.CLOSE, value=Decimal("11"), condition_id="condition")
    assert evidence.dataset_version == 1


def test_lifecycle_confirmation_invalidation_and_expiry_order_are_deterministic():
    result = analysis((Decimal("9"), Decimal("11"), Decimal("9")))
    entry = cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal("10"))
    confirm = cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal("11"))
    invalidate = cmp(FieldRef.CLOSE, ComparisonOperator.LT, Decimal("10"))
    strategy, setup = definition(entry=entry, confirm=confirm, invalidate=invalidate)
    evaluated = evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, result.observations[-1].known_at))
    assert evaluated.status is EvaluationStatus.INVALIDATED
    assert [item.state.value for item in evaluated.lifecycle_transitions] == ["CANDIDATE", "INVALIDATED"]
    assert evaluated.lifecycle_transitions[-1].known_at == result.observations[2].known_at

    expiry_data = analysis((Decimal("9"), Decimal("11"), Decimal("12")))
    expiring, expiring_setup = definition(entry=entry, confirm=cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal("11")), expiry=1)
    expired = evaluate_setup(expiry_data, expiring, expiring_setup, request_for(expiry_data, expiring, expiring_setup, expiry_data.observations[2].known_at))
    assert expired.status is EvaluationStatus.EXPIRED
    assert expired.lifecycle_transitions[-1].known_at == expiry_data.observations[2].known_at
    assert [event.state.value for event in expired.lifecycle_transitions] == ["CANDIDATE", "EXPIRED"]


def test_coincident_invalidation_expiry_and_confirmation_follow_documented_precedence():
    values = analysis((Decimal("9"), Decimal("11"), Decimal("12")))
    entry = cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal("10"))
    true_now = cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal("11"))
    # Case A: candidate reaches the expiry boundary; invalidation also wins over confirmation.
    strategy, setup = definition(entry=entry, confirm=true_now, invalidate=true_now, expiry=1)
    result = evaluate_setup(values, strategy, setup,
        request_for(values, strategy, setup, values.observations[2].known_at))
    assert result.status is EvaluationStatus.INVALIDATED
    assert [event.state.value for event in result.lifecycle_transitions] == ["CANDIDATE", "INVALIDATED"]

    # Case B: expiry precedes confirmation when both are true at the boundary.
    strategy, setup = definition(entry=entry, confirm=true_now, expiry=1)
    result = evaluate_setup(values, strategy, setup,
        request_for(values, strategy, setup, values.observations[2].known_at))
    assert result.status is EvaluationStatus.EXPIRED
    assert [event.state.value for event in result.lifecycle_transitions] == ["CANDIDATE", "EXPIRED"]


def test_same_full_inputs_repeated_evaluations_have_identical_canonical_results():
    result = analysis()
    strategy, setup = definition(confirm=cmp(FieldRef.CLOSE, ComparisonOperator.GTE, Decimal("11")))
    request = request_for(result, strategy, setup, result.observations[2].known_at)
    first = evaluate_setup(result, strategy, setup, request)
    second = evaluate_setup(result, strategy, setup, request)
    assert first.status == second.status
    assert json.dumps(first.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) == json.dumps(
        second.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert first.condition_trace == second.condition_trace
    assert first.evidence == second.evidence
    assert first.lifecycle_transitions == second.lifecycle_transitions
    assert first.fingerprint == second.fingerprint


def test_confirmation_without_confirmation_rule_occurs_at_entry_known_at():
    result = analysis((Decimal("9"), Decimal("11")))
    strategy, setup = definition()
    evaluated = evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, result.observations[1].known_at))
    assert [event.known_at for event in evaluated.lifecycle_transitions] == [result.observations[1].known_at] * 2


def test_false_confirmation_is_not_reported_as_insufficient():
    result = analysis((Decimal("9"), Decimal("11"), Decimal("12")))
    strategy, setup = definition(confirm=cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal("100")))
    evaluated = evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, result.observations[-1].known_at))
    assert evaluated.status is EvaluationStatus.SETUP_NOT_CONFIRMED


def test_unavailable_invalidation_blocks_candidate_confirmation():
    result = analysis((Decimal("9"), Decimal("11"), Decimal("12")))
    last = result.observations[-1].model_copy(update={
        "atr": DecimalValue(value=None, unavailable_reason=AvailabilityReason.INSUFFICIENT_HISTORY),
    })
    result = refingerprint(result.model_copy(update={"observations": (*result.observations[:-1], last)}))
    strategy, setup = definition(
        confirm=cmp(FieldRef.CLOSE, ComparisonOperator.GT, Decimal("10")),
        invalidate=cmp(FieldRef.ATR, ComparisonOperator.GT, Decimal("1")),
    )
    evaluated = evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, last.known_at))
    assert evaluated.status is EvaluationStatus.INSUFFICIENT_EVIDENCE
    assert [event.state.value for event in evaluated.lifecycle_transitions] == ["CANDIDATE"]


def test_future_structural_transition_is_not_exposed_as_current_state():
    result = analysis((Decimal("11"),))
    current = result.observations[0]
    future = current.known_at + timedelta(minutes=1)
    transition = StructuralTransition(previous_state=StructuralState.INSUFFICIENT,
        new_state=StructuralState.UP, known_at=future, responsible_swing="high")
    current = current.model_copy(update={"structural_state": StructuralState.UP,
        "structural_transition": transition})
    result = refingerprint(result.model_copy(update={"observations": (current,)}))
    strategy, setup = definition(entry=cmp(FieldRef.STRUCTURAL_STATE, ComparisonOperator.EQ, "up_structure"))
    evaluated = evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, current.known_at))
    assert evaluated.status is EvaluationStatus.INSUFFICIENT_EVIDENCE
    assert evaluated.evidence[0].unavailable_reason == "structural transition is not yet knowable"


def test_parent_close_equality_is_eligible_but_one_microsecond_before_is_not():
    base = analysis((Decimal("9"), Decimal("11")))
    parent_open = START
    parent_close_at = START + Timeframe.M5.nominal_duration
    parent_provenance = base.inputs[0].provenance.model_copy(update={"timeframe": Timeframe.M5})
    parent_context = HigherTimeframeContext(timeframe=Timeframe.M5, bar_open=parent_open,
        known_at=parent_close_at, close=Decimal("20"), structural_state=StructuralState.UP)
    condition = cmp(FieldRef.HIGHER_CLOSE, ComparisonOperator.GT, Decimal("10"), Timeframe.M5)
    strategy, setup = definition(entry=condition, higher=(Timeframe.M5,))

    prior_close = parent_close_at - Timeframe.M1.nominal_duration
    prior_parent = HigherTimeframeContext(timeframe=Timeframe.M5,
        bar_open=parent_close_at - Timeframe.M5.nominal_duration - Timeframe.M1.nominal_duration,
        known_at=prior_close, close=Decimal("5"), structural_state=StructuralState.INSUFFICIENT)
    prior_observation = base.observations[0].model_copy(update={
        "bar_open": prior_close - Timeframe.M1.nominal_duration,
        "known_at": prior_close, "higher_timeframes": (prior_parent,),
    })
    eligible_observation = base.observations[1].model_copy(update={
        "bar_open": parent_close_at - Timeframe.M1.nominal_duration,
        "known_at": parent_close_at, "higher_timeframes": (parent_context,),
    })
    eligible_result = refingerprint(base.model_copy(update={
        "inputs": (base.inputs[0], SelectedInput(provenance=parent_provenance, slice_hash="d" * 64)),
        "observations": (prior_observation, eligible_observation), "analysis_start": prior_observation.bar_open,
        "analysis_end": parent_close_at, "cutoff_at": parent_close_at,
    }))
    eligible_request = request_for(eligible_result, strategy, setup, parent_close_at)
    assert evaluate_setup(eligible_result, strategy, setup, eligible_request).status is EvaluationStatus.SETUP_CONFIRMED

    before_at = parent_close_at - timedelta(microseconds=1)
    before_observation = eligible_observation.model_copy(update={
        "bar_open": before_at - Timeframe.M1.nominal_duration, "known_at": before_at,
        "higher_timeframes": (parent_context.model_copy(update={"known_at": before_at}),),
    })
    before_result = refingerprint(eligible_result.model_copy(update={
        "observations": (prior_observation, before_observation), "cutoff_at": before_at,
    }))
    before_request = request_for(before_result, strategy, setup, before_at)
    assert evaluate_setup(before_result, strategy, setup, before_request).status is EvaluationStatus.INSUFFICIENT_EVIDENCE


def test_future_parent_swing_confirmation_is_not_exposed():
    base = analysis((Decimal("11"),))
    close_at = START + Timeframe.M5.nominal_duration
    future_swing = ConfirmedSwing(kind="high", candidate_at=START, confirmed_at=close_at,
        known_at=close_at + timedelta(minutes=5), price=Decimal("25"))
    parent = HigherTimeframeContext(timeframe=Timeframe.M5, bar_open=START, known_at=close_at,
        close=Decimal("20"), structural_state=StructuralState.UP, confirmed_swings=(future_swing,))
    observation = base.observations[0].model_copy(update={
        "bar_open": close_at - Timeframe.M1.nominal_duration, "known_at": close_at,
        "higher_timeframes": (parent,),
    })
    result = refingerprint(base.model_copy(update={"inputs": (base.inputs[0], SelectedInput(
        provenance=base.inputs[0].provenance.model_copy(update={"timeframe": Timeframe.M5}), slice_hash="e" * 64)),
        "observations": (observation,), "analysis_end": close_at, "cutoff_at": close_at})
    )
    condition = cmp(FieldRef.HIGHER_LAST_SWING_HIGH_PRICE, ComparisonOperator.GT, Decimal("10"), Timeframe.M5)
    setup = SetupDefinition(setup_id="swing", setup_version=1, instrument=INSTRUMENT,
        primary_timeframe=Timeframe.M1, required_higher_timeframes=(Timeframe.M5,),
        entry_conditions=condition, expiry=ExpiryRule(max_later_observations=2))
    strategy = StrategyDefinition(strategy_id=UUID(int=9), strategy_version=1,
        analysis_version=ANALYSIS_VERSION, setups=(setup,))
    setup = strategy.setups[0]
    evaluated = evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, close_at))
    assert evaluated.status is EvaluationStatus.INSUFFICIENT_EVIDENCE
    assert evaluated.evidence[0].unavailable_reason == "no confirmed parent swing"


def test_higher_structural_state_evidence_does_not_claim_swing_causality():
    base = analysis((Decimal("9"), Decimal("11")))
    close_at = START + Timeframe.M5.nominal_duration
    swing = ConfirmedSwing(kind="high", candidate_at=START, confirmed_at=close_at,
        known_at=close_at, price=Decimal("25"))
    parent = HigherTimeframeContext(timeframe=Timeframe.M5, bar_open=START,
        known_at=close_at, close=Decimal("20"), structural_state=StructuralState.UP,
        confirmed_swings=(swing,))
    prior = base.observations[0].model_copy(update={"bar_open": close_at - timedelta(minutes=2),
        "known_at": close_at - timedelta(minutes=1)})
    current = base.observations[1].model_copy(update={"bar_open": close_at - timedelta(minutes=1),
        "known_at": close_at, "higher_timeframes": (parent,)})
    result = refingerprint(base.model_copy(update={"inputs": (base.inputs[0], SelectedInput(
        provenance=base.inputs[0].provenance.model_copy(update={"timeframe": Timeframe.M5}), slice_hash="e" * 64)),
        "observations": (prior, current), "analysis_end": close_at, "cutoff_at": close_at}))
    condition = cmp(FieldRef.HIGHER_STRUCTURAL_STATE, ComparisonOperator.EQ, "up_structure", Timeframe.M5)
    strategy, setup = definition(entry=condition, higher=(Timeframe.M5,))
    result_eval = evaluate_setup(result, strategy, setup, request_for(result, strategy, setup, close_at))
    evidence = next(item for item in reversed(result_eval.evidence) if item.field is FieldRef.HIGHER_STRUCTURAL_STATE)
    assert evidence.parent_bar_open == START
    assert evidence.supporting_candidate_at is None
    assert evidence.supporting_confirmed_at is None


def test_future_primary_observations_do_not_change_earlier_evaluation():
    original = analysis((Decimal("11"), Decimal("12")))
    strategy, setup = definition(entry=cmp(FieldRef.STRUCTURAL_STATE, ComparisonOperator.EQ, "up_structure"))
    cutoff = original.observations[0].known_at
    request = request_for(original, strategy, setup, cutoff)
    baseline = evaluate_setup(original, strategy, setup, request)
    future_base = analysis((Decimal("11"), Decimal("12"), Decimal("999")))
    future_obs = future_base.observations[-1].model_copy(update={
        "structural_state": StructuralState.UP,
        "structural_transition": StructuralTransition(
            previous_state=StructuralState.INSUFFICIENT, new_state=StructuralState.UP,
            known_at=future_base.observations[-1].known_at, responsible_swing="high",
        ),
    })
    future = refingerprint(future_base.model_copy(update={
        "observations": (*future_base.observations[:-1], future_obs),
    }))
    # Full identity changes, while cutoff-visible identity and decision output do not.
    future_request = request_for(future, strategy, setup, cutoff)
    assert future_request.fingerprint == request.fingerprint
    changed = evaluate_setup(future, strategy, setup, future_request)
    assert changed.status == baseline.status
    assert changed.evaluation_prefix_fingerprint == baseline.evaluation_prefix_fingerprint
    assert changed.analysis_fingerprint != baseline.analysis_fingerprint
    assert tuple((t.condition_id, t.known_at, t.outcome) for t in changed.condition_trace) == tuple(
        (t.condition_id, t.known_at, t.outcome) for t in baseline.condition_trace)
    assert tuple(_decision_evidence(e) for e in changed.evidence) == tuple(_decision_evidence(e) for e in baseline.evidence)
    assert changed.lifecycle_transitions == baseline.lifecycle_transitions
    assert changed.fingerprint == baseline.fingerprint


def test_fingerprints_are_deterministic_and_change_with_definition_or_time():
    strategy, setup = definition()
    same_strategy, same_setup = definition(entry=setup.entry_conditions,
        parameters=strategy.parameters)
    assert strategy.fingerprint == same_strategy.fingerprint
    assert setup.fingerprint == same_setup.fingerprint
    changed_version = StrategyDefinition(strategy_id=strategy.strategy_id, strategy_version=2,
        analysis_version=ANALYSIS_VERSION, setups=(setup,))
    assert changed_version.fingerprint != strategy.fingerprint
    result = analysis()
    request1 = request_for(result, strategy, setup, result.observations[0].known_at)
    request2 = request_for(result, strategy, setup, result.observations[1].known_at)
    assert request1.fingerprint != request2.fingerprint
    assert setup.fingerprint == strategy.setup("breakout", 1).fingerprint
    assert same_setup.fingerprint == strategy.setup("breakout", 1).fingerprint
