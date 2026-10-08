from datetime import datetime, timezone
from decimal import Decimal, Inexact, ROUND_DOWN, getcontext

import pytest
from pydantic import ValidationError
from pydantic import ValidationError

from app.backtesting.contracts import ExecutionAssumptions
from app.market_data.contracts import AssetClass, Instrument, PriceBasis
from app.risk.contracts import (
    RiskConfiguration,
    RiskFinding,
    RiskSeverity,
    RiskStatus,
    TradeDirection,
    TradeRiskRequest,
)
from app.risk.service import RiskEngineService


def make_config(**overrides):
    defaults = {
        "configuration_id": "cfg-1",
        "configuration_version": 1,
        "risk_per_trade": Decimal("0.01"),
        "max_position_size": 200,
        "max_notional_exposure": Decimal("10000"),
        "minimum_stop_distance": Decimal("0.10"),
        "maximum_stop_distance": Decimal("15.00"),
        "minimum_reward_risk": Decimal("1.5"),
        "account_currency": "USD",
        "execution_assumptions": ExecutionAssumptions(
            spread=Decimal("0.00"),
            slippage=Decimal("0.00"),
            pct_fee=Decimal("0.00"),
            fixed_fee=Decimal("0.00"),
            price_basis=PriceBasis.TRADE,
            fixed_quantity=1,
        ),
        "quantity_increment": Decimal("1"),
    }
    defaults.update(overrides)
    return RiskConfiguration(**defaults)


def make_request(config=None, **overrides):
    config = config or make_config()
    defaults = {
        "request_id": "req-1",
        "risk_configuration": config,
        "risk_configuration_fingerprint": config.fingerprint,
        "instrument": Instrument(symbol="EURUSD", asset_class=AssetClass.FX),
        "side": TradeDirection.LONG,
        "entry_price": Decimal("100"),
        "stop_price": Decimal("98"),
        "target_price": Decimal("105"),
        "account_equity": Decimal("10000"),
        "pnl_to_account_rate": Decimal("1"),
        "execution_assumptions": config.execution_assumptions,
        "evaluation_at": datetime(2026, 10, 8, tzinfo=timezone.utc),
        "strategy_id": "strategy-1",
        "setup_id": "setup-1",
        "strategy_fingerprint": "a" * 64,
        "setup_fingerprint": "b" * 64,
        "analysis_fingerprint": "c" * 64,
        "evaluation_prefix_fingerprint": "d" * 64,
    }
    defaults.update(overrides)
    return TradeRiskRequest(**defaults)


def test_valid_long_risk_result():
    config = make_config()
    request = make_request(config)
    result = RiskEngineService().evaluate(request)

    assert result.status == RiskStatus.RISK_VALID
    assert result.requested_risk == Decimal("100")
    assert result.maximum_monetary_risk == Decimal("100")
    assert result.position_size == Decimal("50")
    assert result.notional_exposure == Decimal("5000")
    assert result.reward_risk == Decimal("2.5")
    assert len(result.fingerprint) == 64


def test_valid_short_risk_result():
    config = make_config()
    request = make_request(
        config,
        side=TradeDirection.SHORT,
        entry_price=Decimal("100"),
        stop_price=Decimal("105"),
        target_price=Decimal("90"),
    )

    result = RiskEngineService().evaluate(request)

    assert result.status == RiskStatus.RISK_VALID
    assert result.position_size == Decimal("20")
    assert result.reward_risk == Decimal("2")


def test_invalid_stop_relationships_are_rejected():
    config = make_config()
    long_request = make_request(config, stop_price=Decimal("100"))
    long_result = RiskEngineService().evaluate(long_request)
    assert long_result.status == RiskStatus.RISK_REJECTED

    short_request = make_request(config, side=TradeDirection.SHORT, stop_price=Decimal("99"), target_price=Decimal("90"))
    short_result = RiskEngineService().evaluate(short_request)
    assert short_result.status == RiskStatus.RISK_REJECTED


def test_missing_evidence_returns_insufficient():
    config = make_config()
    request = make_request(config, account_equity=None)
    result = RiskEngineService().evaluate(request)
    assert result.status == RiskStatus.INSUFFICIENT_EVIDENCE
    assert result.validation_findings[0].severity == RiskSeverity.INSUFFICIENT


def test_reward_risk_and_stop_distance_constraints():
    config = make_config(minimum_stop_distance=Decimal("2"), maximum_stop_distance=Decimal("4"), minimum_reward_risk=Decimal("2"))
    request = make_request(config, stop_price=Decimal("99"), target_price=Decimal("101"))
    result = RiskEngineService().evaluate(request)
    assert result.status == RiskStatus.RISK_REJECTED
    assert any(f.rule == "minimum_reward_risk" for f in result.validation_findings)

    request_two = make_request(config, stop_price=Decimal("90"))
    result_two = RiskEngineService().evaluate(request_two)
    assert result_two.status == RiskStatus.RISK_REJECTED
    assert any(f.rule == "minimum_stop_distance" for f in result_two.validation_findings)


def test_execution_costs_can_reject_over_limit():
    config = make_config(
        risk_per_trade=Decimal("0.02"),
        execution_assumptions=ExecutionAssumptions(
            spread=Decimal("0.02"),
            slippage=Decimal("0.01"),
            pct_fee=Decimal("0.25"),
            fixed_fee=Decimal("0"),
            price_basis=PriceBasis.TRADE,
            fixed_quantity=1,
        ),
        max_notional_exposure=Decimal("100000"),
    )
    request = make_request(config, account_equity=Decimal("1000"), entry_price=Decimal("100"), stop_price=Decimal("99"))
    result = RiskEngineService().evaluate(request)
    assert result.status == RiskStatus.RISK_REJECTED


def test_fixed_fees_are_converted_before_position_sizing():
    assumptions = ExecutionAssumptions(
        spread=Decimal("0"), slippage=Decimal("0"), pct_fee=Decimal("0"),
        fixed_fee=Decimal("1"), price_basis=PriceBasis.TRADE, fixed_quantity=1,
    )
    config = make_config(
        risk_per_trade=Decimal("0.01"), max_notional_exposure=Decimal("100000"),
        execution_assumptions=assumptions,
    )
    request = make_request(
        config, account_equity=Decimal("1000"), entry_price=Decimal("100"),
        stop_price=Decimal("98"), target_price=Decimal("105"),
        execution_assumptions=assumptions, pnl_to_account_rate=Decimal("2"),
    )

    result = RiskEngineService().evaluate(request)

    assert result.status == RiskStatus.RISK_VALID
    assert result.position_size == Decimal("1")
    assert result.total_estimated_downside == Decimal("8")
    assert result.total_estimated_downside <= result.maximum_monetary_risk


def test_spread_and_slippage_costs_are_reported_separately():
    assumptions = ExecutionAssumptions(
        spread=Decimal("0.2"), slippage=Decimal("0.1"), pct_fee=Decimal("0"),
        fixed_fee=Decimal("0"), price_basis=PriceBasis.TRADE, fixed_quantity=1,
    )
    config = make_config(
        execution_assumptions=assumptions,
        max_notional_exposure=Decimal("100000"),
    )
    result = RiskEngineService().evaluate(make_request(
        config, execution_assumptions=assumptions, entry_price=Decimal("100"),
        stop_price=Decimal("98"), target_price=Decimal("105"), account_equity=Decimal("1000"),
    ))

    assert result.status == RiskStatus.RISK_VALID
    assert result.position_size == Decimal("4")
    assert result.estimated_spread_cost == Decimal("0.8")
    assert result.estimated_slippage_impact == Decimal("0.8")
    assert result.total_estimated_downside == Decimal("9.6")


def test_contracts_reject_floats_bool_and_bad_fingerprints():
    with pytest.raises(ValidationError):
        RiskConfiguration(
            configuration_id="cfg-2",
            configuration_version=1,
            risk_per_trade=0.01,
            max_position_size=10,
            max_notional_exposure=Decimal("1000"),
            minimum_stop_distance=Decimal("1"),
            maximum_stop_distance=Decimal("10"),
            minimum_reward_risk=Decimal("1"),
            account_currency="USD",
            execution_assumptions=ExecutionAssumptions(
                spread=Decimal("0.02"),
                slippage=Decimal("0.01"),
                pct_fee=Decimal("0.001"),
                fixed_fee=Decimal("0"),
                price_basis=PriceBasis.TRADE,
                fixed_quantity=1,
            ),
            quantity_increment=Decimal("1"),
        )

    with pytest.raises(ValidationError):
        TradeRiskRequest(
            request_id="bad",
            risk_configuration=make_config(),
            risk_configuration_fingerprint="G" * 64,
            instrument=Instrument(symbol="EURUSD", asset_class=AssetClass.FX),
            side=TradeDirection.LONG,
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            account_equity=Decimal("1000"),
            execution_assumptions=make_config().execution_assumptions,
            evaluation_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
        )

    with pytest.raises((ValidationError, TypeError)):
        TradeRiskRequest(
            request_id="bad",
            risk_configuration=make_config(),
            risk_configuration_fingerprint=make_config().fingerprint,
            instrument=Instrument(symbol="EURUSD", asset_class=AssetClass.FX),
            side=True,
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            account_equity=Decimal("1000"),
            execution_assumptions=make_config().execution_assumptions,
            evaluation_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
        )


def test_immutable_contracts_and_config_integrity():
    config = make_config()
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        config.configuration_id = "new-id"

    request = make_request(config)
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        request.account_equity = Decimal("20000")

    tampered_payload = make_request(config).model_dump()
    tampered_payload["risk_configuration_fingerprint"] = "0" * 64
    with pytest.raises(ValidationError):
        TradeRiskRequest.model_validate(tampered_payload)


def test_quantity_rounding_and_small_prices():
    config = make_config(
        quantity_increment=Decimal("0.25"),
        risk_per_trade=Decimal("0.01"),
        max_position_size=50000,
        minimum_stop_distance=Decimal("0.00001"),
        maximum_stop_distance=Decimal("0.01"),
    )
    request = make_request(
        config,
        entry_price=Decimal("0.0005"),
        stop_price=Decimal("0.0004"),
        account_equity=Decimal("10000"),
        target_price=Decimal("0.0008"),
    )
    result = RiskEngineService().evaluate(request)

    assert result.status == RiskStatus.RISK_VALID
    assert result.position_size == Decimal("50000")
    assert result.total_estimated_downside <= result.maximum_monetary_risk
    assert result.notional_exposure == Decimal("25.0000")


def test_decimal_context_is_unchanged_by_risk_evaluation():
    context = getcontext()
    original = (context.prec, context.rounding, dict(context.traps), dict(context.flags))
    try:
        context.prec = 11
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        context.flags[Inexact] = True
        before = (context.prec, context.rounding, dict(context.traps), dict(context.flags))
        request = make_request(entry_price=Decimal("100.03"), stop_price=Decimal("98.07"))
        result = RiskEngineService().evaluate(request)
        assert result.status == RiskStatus.RISK_VALID
        after = (context.prec, context.rounding, dict(context.traps), dict(context.flags))
        assert after == before
    finally:
        context.prec, context.rounding = original[0], original[1]
        for signal, enabled in original[2].items():
            context.traps[signal] = enabled
        for signal, enabled in original[3].items():
            context.flags[signal] = enabled
