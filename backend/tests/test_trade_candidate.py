from datetime import datetime, timedelta, timezone
from decimal import Decimal, Inexact, ROUND_UP, getcontext
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.execution.assumptions import ExecutionAssumptions
from app.market_analysis.contracts import (
    ANALYSIS_VERSION, AnalysisParameters, AnalysisResult, AvailabilityReason,
    CandleFeatures, DatasetProvenance, DecimalValue, MarketObservation,
    SelectedInput, StructuralState,
)
from app.market_context.sessions import CalendarStatus, SessionClassification
from app.market_data.contracts import AssetClass, Instrument
from app.market_data.timeframes import Timeframe
from app.risk.contracts import RiskConfiguration, RiskStatus, TradeDirection, TradeRiskRequest
from app.risk.service import RiskEngineService
from app.strategy import (
    ComparisonOperator, ConditionOperand, EvaluationRequest, EvaluationStatus,
    ExpiryRule, FieldRef, SetupDefinition, StrategyDefinition, evaluate_setup,
)
from app.strategy.fingerprints import analysis_full_fingerprint, analysis_prefix_fingerprint
from app.trade_candidate import CandidateStatus, TradeCandidate, TradeCandidateService
from app.trade_candidate.fingerprints import candidate_fingerprint, candidate_id_for
from app.trade_candidate.errors import CandidateInputError
from app.trade_candidate.service import _evaluation_fingerprint

UTC = timezone.utc
START = datetime(2026, 10, 8, 9, tzinfo=UTC)
INSTRUMENT = Instrument(symbol="EURUSD", asset_class=AssetClass.FX)


def make_analysis(closes=(Decimal("9"), Decimal("11"), Decimal("12"))):
    provenance = DatasetProvenance(
        dataset_id=UUID(int=100), dataset_version=2, instrument=INSTRUMENT,
        timeframe=Timeframe.M1, content_hash="a" * 64, provider_id="fixture",
        provider_version="1.0", provider_symbol="EURUSD", price_basis="trade",
        timestamp_convention="bar_open",
    )
    observations = []
    for index, close in enumerate(closes):
        opened = START + index * Timeframe.M1.nominal_duration
        known = opened + Timeframe.M1.nominal_duration
        prior = index > 0
        observations.append(MarketObservation(
            bar_open=opened, known_at=known, open=close, high=close + Decimal("1"),
            low=close - Decimal("1"), close=close, volume=Decimal("100"), volume_units="lots",
            features=CandleFeatures(
                range=Decimal("2"), body=Decimal("0"), upper_wick=Decimal("1"), lower_wick=Decimal("1"),
                body_ratio=DecimalValue(value=Decimal("0")), close_location=DecimalValue(value=Decimal("0.5")),
                simple_return=DecimalValue(value=Decimal("0") if prior else None,
                    unavailable_reason=None if prior else AvailabilityReason.NO_PREVIOUS_CLOSE),
            ), true_range=Decimal("2"), atr=DecimalValue(value=Decimal("2")),
            structural_state=StructuralState.INSUFFICIENT, confirmed_swings=(),
            structural_transition=None, higher_timeframes=(),
            session=SessionClassification(timestamp=known, labels=("London",), local_date=known.date(),
                weekend=False, calendar_status=CalendarStatus.UNKNOWN), session_transition=None,
        ))
    result = AnalysisResult(
        analysis_version=ANALYSIS_VERSION, parameters=AnalysisParameters(), input_start=START,
        analysis_start=START, analysis_end=START + len(observations) * Timeframe.M1.nominal_duration,
        cutoff_at=START + len(observations) * Timeframe.M1.nominal_duration,
        inputs=(SelectedInput(provenance=provenance, slice_hash="b" * 64),),
        observations=tuple(observations), fingerprint="0" * 64,
    )
    return result.model_copy(update={"fingerprint": analysis_full_fingerprint(result)})


def make_definitions(*, confirmation=None, invalidation=None, expiry=3):
    setup = SetupDefinition(
        setup_id="breakout", setup_version=2, instrument=INSTRUMENT,
        primary_timeframe=Timeframe.M1,
        entry_conditions={"kind": "compare", "field": FieldRef.CLOSE,
            "operator": ComparisonOperator.GT, "operand": ConditionOperand(value=Decimal("10"))},
        confirmation_conditions=confirmation, invalidation_conditions=invalidation,
        expiry=ExpiryRule(max_later_observations=expiry),
    )
    strategy = StrategyDefinition(strategy_id=UUID(int=22), strategy_version=3,
        analysis_version=ANALYSIS_VERSION, setups=(setup,))
    return strategy, strategy.setups[0]


def make_assumptions():
    return ExecutionAssumptions(
        spread=Decimal("0.01"), slippage=Decimal("0.01"), pct_fee=Decimal("0"),
        fixed_fee=Decimal("0"), fixed_quantity=1,
    )


def make_inputs(*, side=TradeDirection.LONG, target=Decimal("105"), closes=None, config=None,
                confirmation=None, invalidation=None, expiry=3):
    analysis = make_analysis(closes or (Decimal("9"), Decimal("11"), Decimal("12")))
    strategy, setup = make_definitions(confirmation=confirmation, invalidation=invalidation, expiry=expiry)
    at = analysis.observations[-1].known_at
    request = EvaluationRequest(
        strategy_id=strategy.strategy_id, strategy_version=strategy.strategy_version,
        strategy_fingerprint=strategy.fingerprint, setup_id=setup.setup_id,
        setup_version=setup.setup_version, setup_fingerprint=setup.fingerprint,
        analysis_fingerprint=analysis.fingerprint,
        evaluation_prefix_fingerprint=analysis_prefix_fingerprint(analysis, at), evaluation_at=at,
    )
    evaluation = evaluate_setup(analysis, strategy, setup, request)
    assumptions = make_assumptions()
    config = config or RiskConfiguration(
        configuration_id="single-trade", configuration_version=1,
        risk_per_trade=Decimal("0.01"), max_position_size=200,
        max_notional_exposure=Decimal("100000"), account_currency="USD",
        execution_assumptions=assumptions, quantity_increment=Decimal("1"),
    )
    entry = Decimal("100")
    stop = Decimal("98") if side is TradeDirection.LONG else Decimal("102")
    if side is TradeDirection.SHORT and target == Decimal("105"):
        target = Decimal("95")
    risk_request = TradeRiskRequest(
        request_id="candidate-risk-1", risk_configuration=config,
        risk_configuration_fingerprint=config.fingerprint, strategy_id=str(strategy.strategy_id),
        setup_id=setup.setup_id, strategy_fingerprint=strategy.fingerprint,
        setup_fingerprint=setup.fingerprint, analysis_fingerprint=analysis.fingerprint,
        evaluation_prefix_fingerprint=evaluation.evaluation_prefix_fingerprint,
        instrument=INSTRUMENT, side=side, entry_price=entry, stop_price=stop,
        target_price=target, account_equity=Decimal("10000"), pnl_to_account_rate=Decimal("1"),
        execution_assumptions=assumptions, evaluation_at=at,
    )
    risk_result = RiskEngineService().evaluate(risk_request)
    return analysis, strategy, setup, evaluation, risk_request, risk_result


def candidate_for(**kwargs):
    request = kwargs["risk_request"]
    config = request.risk_configuration
    kwargs.setdefault("execution_assumptions", request.execution_assumptions or (
        config.execution_assumptions if config is not None else None
    ))
    return TradeCandidateService().assemble(**kwargs)


def assemble(**kwargs):
    artifacts = make_inputs(**kwargs)
    return candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=artifacts[4], risk_result=artifacts[5],
    ), artifacts


def test_actionable_long_and_short_candidates_use_risk_outputs_without_target_recalculation():
    long, long_inputs = assemble()
    short, short_inputs = assemble(side=TradeDirection.SHORT)
    assert long.status is CandidateStatus.ACTIONABLE
    assert short.status is CandidateStatus.ACTIONABLE
    assert long.quantity == long_inputs[-1].position_size
    assert long.risk_amount == long_inputs[-1].total_estimated_downside
    assert long.risk_percentage == long.risk_amount / long_inputs[4].account_equity
    assert long.expected_reward_risk == long_inputs[-1].reward_risk == Decimal("2.5")
    assert short.direction is TradeDirection.SHORT
    assert short.expected_reward_risk == Decimal("2.5")
    assert long.fingerprint != short.fingerprint


def test_candidate_without_target_has_explicitly_unavailable_reward_risk():
    candidate, inputs = assemble(target=None)
    assert candidate.status is CandidateStatus.ACTIONABLE
    assert candidate.take_profits == ()
    assert candidate.expected_reward_risk is None
    assert inputs[-1].reward_risk is None


@pytest.mark.parametrize("side,stop,target", [
    (TradeDirection.LONG, Decimal("101"), Decimal("105")),
    (TradeDirection.SHORT, Decimal("99"), Decimal("95")),
    (TradeDirection.LONG, Decimal("98"), Decimal("99")),
    (TradeDirection.SHORT, Decimal("102"), Decimal("101")),
    (TradeDirection.LONG, Decimal("0"), Decimal("105")),
    (TradeDirection.LONG, Decimal("98"), Decimal("0")),
    (TradeDirection.LONG, Decimal("-1"), Decimal("105")),
])
def test_invalid_prices_never_produce_actionable_candidate(side, stop, target):
    artifacts = list(make_inputs(side=side))
    bad_request = artifacts[4].model_copy(update={"stop_price": stop, "target_price": target})
    artifacts[4] = bad_request
    artifacts[5] = RiskEngineService().evaluate(bad_request)
    if stop <= 0 or target <= 0:
        with pytest.raises(CandidateInputError):
            candidate_for(
                analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
                evaluation=artifacts[3], risk_request=artifacts[4], risk_result=artifacts[5],
            )
    else:
        candidate = candidate_for(
            analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
            evaluation=artifacts[3], risk_request=artifacts[4], risk_result=artifacts[5],
        )
        assert candidate.status is CandidateStatus.REJECTED


def test_incompatible_analysis_strategy_setup_and_dataset_evidence_are_rejected():
    artifacts = make_inputs()
    analysis = artifacts[0]
    changed = analysis.model_copy(update={"observations": analysis.observations[:-1]})
    candidate = candidate_for(
        analysis=changed, strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=artifacts[4], risk_result=artifacts[5],
    )
    assert candidate.status is CandidateStatus.REJECTED
    assert any("fingerprint" in item for item in candidate.findings)

    stale_evaluation = artifacts[3].model_copy(update={"strategy_version": artifacts[3].strategy_version + 1})
    candidate = candidate_for(
        analysis=analysis, strategy=artifacts[1], setup=artifacts[2],
        evaluation=stale_evaluation, risk_request=artifacts[4], risk_result=artifacts[5],
    )
    assert candidate.status is CandidateStatus.REJECTED
    assert any("evaluation identity" in item for item in candidate.findings)

    foreign_instrument = Instrument(symbol="GBPUSD", asset_class=AssetClass.FX)
    foreign_request = artifacts[4].model_copy(update={"instrument": foreign_instrument})
    foreign_result = RiskEngineService().evaluate(foreign_request)
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=foreign_request, risk_result=foreign_result,
    )
    assert candidate.status is CandidateStatus.REJECTED


def test_forged_or_stale_risk_result_and_incompatible_execution_are_rejected():
    artifacts = make_inputs()
    forged = artifacts[5].model_copy(update={"position_size": Decimal("1")})
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=artifacts[4], risk_result=forged,
    )
    assert candidate.status is CandidateStatus.REJECTED
    bad_request = artifacts[4].model_copy(update={"execution_assumptions": make_assumptions().model_copy(update={"spread": Decimal("0.50")})})
    bad_result = RiskEngineService().evaluate(bad_request)
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=bad_request, risk_result=bad_result,
        execution_assumptions=artifacts[4].execution_assumptions,
    )
    assert candidate.status is CandidateStatus.REJECTED


@pytest.mark.parametrize("field,value", [
    ("position_size", Decimal("1")),
    ("total_estimated_downside", Decimal("1")),
    ("stop_distance", Decimal("1")),
    ("reward_risk", Decimal("9")),
])
def test_modified_risk_outputs_with_stale_fingerprint_are_rejected(field, value):
    artifacts = make_inputs()
    forged = artifacts[5].model_copy(update={field: value})
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=artifacts[4], risk_result=forged,
    )
    assert candidate.status is CandidateStatus.REJECTED
    assert any("risk result fingerprint" in item for item in candidate.findings)


def test_unconfirmed_expired_invalidated_and_insufficient_evaluations_are_not_actionable():
    artifacts = list(make_inputs(closes=(Decimal("9"), Decimal("9"), Decimal("9"))))
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=artifacts[4], risk_result=artifacts[5],
    )
    assert candidate.status is CandidateStatus.WAIT

    insufficient_artifacts = make_inputs(closes=(Decimal("11"),))
    candidate = candidate_for(
        analysis=insufficient_artifacts[0], strategy=insufficient_artifacts[1], setup=insufficient_artifacts[2],
        evaluation=insufficient_artifacts[3], risk_request=insufficient_artifacts[4],
        risk_result=insufficient_artifacts[5],
    )
    assert candidate.status is CandidateStatus.INSUFFICIENT_EVIDENCE

    close_above = {"kind": "compare", "field": FieldRef.CLOSE,
        "operator": ComparisonOperator.GT, "operand": ConditionOperand(value=Decimal("11"))}
    expired_artifacts = make_inputs(closes=(Decimal("9"), Decimal("11"), Decimal("12")),
        confirmation=close_above, expiry=1)
    assert expired_artifacts[3].status is EvaluationStatus.EXPIRED
    candidate = candidate_for(
        analysis=expired_artifacts[0], strategy=expired_artifacts[1], setup=expired_artifacts[2],
        evaluation=expired_artifacts[3], risk_request=expired_artifacts[4], risk_result=expired_artifacts[5],
    )
    assert candidate.status is CandidateStatus.WAIT
    assert candidate.expiry_at == expired_artifacts[3].lifecycle_transitions[-1].known_at

    invalidated_artifacts = make_inputs(closes=(Decimal("9"), Decimal("11"), Decimal("12")),
        confirmation=close_above,
        invalidation={"kind": "compare", "field": FieldRef.CLOSE,
            "operator": ComparisonOperator.GT, "operand": ConditionOperand(value=Decimal("11"))},
        expiry=3)
    assert invalidated_artifacts[3].status is EvaluationStatus.INVALIDATED
    candidate = candidate_for(
        analysis=invalidated_artifacts[0], strategy=invalidated_artifacts[1], setup=invalidated_artifacts[2],
        evaluation=invalidated_artifacts[3], risk_request=invalidated_artifacts[4], risk_result=invalidated_artifacts[5],
    )
    assert candidate.status is CandidateStatus.WAIT


def test_risk_rejection_and_insufficient_evidence_do_not_become_actionable():
    artifacts = list(make_inputs())
    rejected = artifacts[4].model_copy(update={"stop_price": Decimal("100")})
    artifacts[5] = RiskEngineService().evaluate(rejected)
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=rejected, risk_result=artifacts[5],
    )
    assert candidate.status is CandidateStatus.REJECTED
    assert artifacts[5].status is RiskStatus.RISK_REJECTED

    no_config_request = artifacts[4].model_copy(update={
        "risk_configuration": None, "risk_configuration_fingerprint": None,
    })
    insufficient_risk = RiskEngineService().evaluate(no_config_request)
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=no_config_request, risk_result=insufficient_risk,
    )
    assert candidate.status is CandidateStatus.INSUFFICIENT_EVIDENCE

    no_direction = artifacts[4].model_copy(update={"side": None})
    no_direction_result = RiskEngineService().evaluate(no_direction)
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=no_direction, risk_result=no_direction_result,
    )
    assert candidate.status is CandidateStatus.INSUFFICIENT_EVIDENCE
    assert candidate.direction is None

    valid_artifacts = make_inputs()
    missing_assumptions = candidate_for(
        analysis=valid_artifacts[0], strategy=valid_artifacts[1], setup=valid_artifacts[2],
        evaluation=valid_artifacts[3], risk_request=valid_artifacts[4], risk_result=valid_artifacts[5],
        execution_assumptions=None,
    )
    assert missing_assumptions.status is CandidateStatus.INSUFFICIENT_EVIDENCE


def test_future_evidence_is_rejected_and_fingerprints_are_deterministic():
    artifacts = list(make_inputs())
    evidence = artifacts[3].evidence[0].model_copy(update={"known_at": artifacts[3].evaluation_at + timedelta(microseconds=1)})
    changed_eval = artifacts[3].model_copy(update={"evidence": (evidence,), "fingerprint": "0" * 64})
    changed_eval = changed_eval.model_copy(update={"fingerprint": _evaluation_fingerprint(changed_eval)})
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=changed_eval, risk_request=artifacts[4], risk_result=artifacts[5],
    )
    assert candidate.status is CandidateStatus.REJECTED
    assert any("after the candidate evaluation" in item for item in candidate.findings)
    valid, _ = assemble()
    same, _ = assemble()
    assert valid.fingerprint == same.fingerprint
    assert valid.candidate_id == same.candidate_id


def test_missing_evidence_is_insufficient_and_stale_or_incompatible_evidence_rejected():
    artifacts = list(make_inputs())
    no_evidence = artifacts[3].model_copy(update={
        "evidence": (), "condition_trace": (), "lifecycle_transitions": (), "fingerprint": "0" * 64,
    })
    no_evidence = no_evidence.model_copy(update={"fingerprint": _evaluation_fingerprint(no_evidence)})
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=no_evidence, risk_request=artifacts[4], risk_result=artifacts[5],
    )
    assert candidate.status is CandidateStatus.INSUFFICIENT_EVIDENCE

    original_ref = artifacts[3].evidence[0]
    foreign_ref = original_ref.model_copy(update={"dataset_id": UUID(int=999)})
    incompatible = artifacts[3].model_copy(update={"evidence": (foreign_ref,), "fingerprint": "0" * 64})
    incompatible = incompatible.model_copy(update={"fingerprint": _evaluation_fingerprint(incompatible)})
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=incompatible, risk_request=artifacts[4], risk_result=artifacts[5],
    )
    assert candidate.status is CandidateStatus.REJECTED
    assert any("outside the analysis inputs" in item for item in candidate.findings)

    stale_ref = original_ref.model_copy(update={"analysis_fingerprint": "f" * 64})
    stale = artifacts[3].model_copy(update={"evidence": (stale_ref,), "fingerprint": "0" * 64})
    stale = stale.model_copy(update={"fingerprint": _evaluation_fingerprint(stale)})
    candidate = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=stale, risk_request=artifacts[4], risk_result=artifacts[5],
    )
    assert candidate.status is CandidateStatus.REJECTED
    assert any("stale analysis fingerprint" in item for item in candidate.findings)


def test_material_risk_input_changes_candidate_fingerprint():
    original, artifacts = assemble()
    changed_request = artifacts[4].model_copy(update={"target_price": Decimal("106")})
    changed_result = RiskEngineService().evaluate(changed_request)
    changed = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=changed_request, risk_result=changed_result,
    )
    assert changed.status is CandidateStatus.ACTIONABLE
    assert changed.evidence.risk.request_fingerprint != original.evidence.risk.request_fingerprint
    assert changed.fingerprint != original.fingerprint

    changed_equity_request = artifacts[4].model_copy(update={"account_equity": Decimal("20000")})
    changed_equity_result = RiskEngineService().evaluate(changed_equity_request)
    changed_equity = candidate_for(
        analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
        evaluation=artifacts[3], risk_request=changed_equity_request, risk_result=changed_equity_result,
    )
    assert changed_equity.status is CandidateStatus.ACTIONABLE
    assert changed_equity.fingerprint != original.fingerprint


def test_decimal_context_is_preserved_during_candidate_assembly():
    artifacts = make_inputs()
    context = getcontext()
    original_precision, original_rounding = context.prec, context.rounding
    original_trap, original_flags = context.traps[Inexact], context.flags[Inexact]
    try:
        context.prec = 6
        context.rounding = ROUND_UP
        context.traps[Inexact] = True
        context.clear_flags()
        before = (context.prec, context.rounding, dict(context.traps), dict(context.flags))
        candidate = candidate_for(
            analysis=artifacts[0], strategy=artifacts[1], setup=artifacts[2],
            evaluation=artifacts[3], risk_request=artifacts[4], risk_result=artifacts[5],
        )
        after = (context.prec, context.rounding, dict(context.traps), dict(context.flags))
        assert candidate.status is CandidateStatus.ACTIONABLE
        assert after == before
    finally:
        context.prec = original_precision
        context.rounding = original_rounding
        context.traps[Inexact] = original_trap
        context.flags[Inexact] = original_flags


def test_candidate_and_nested_evidence_are_immutable_and_self_fingerprinting():
    candidate, _ = assemble()
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        candidate.status = CandidateStatus.REJECTED
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        candidate.evidence.evaluation.status = EvaluationStatus.EXPIRED
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        candidate.evidence.evidence += ()
    payload = candidate.model_dump(mode="python", exclude={"candidate_id", "fingerprint"})
    assert candidate.fingerprint == candidate_fingerprint(payload)
    assert candidate.candidate_id == candidate_id_for(candidate.fingerprint)
    forged_payload = candidate.model_dump(mode="python")
    forged_payload["quantity"] = Decimal("1")
    with pytest.raises(ValidationError):
        TradeCandidate(**forged_payload)


def test_evidence_package_cannot_be_mutated_or_contain_float_financial_values():
    candidate, _ = assemble()
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        candidate.evidence.datasets[0].dataset_version = 10
    data = candidate.model_dump(mode="python")
    data["entry_price"] = 100.5
    with pytest.raises(ValidationError):
        TradeCandidate(**data)
