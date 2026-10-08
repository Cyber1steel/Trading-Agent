"""Deterministic mathematical risk calculations."""

from decimal import Context, Decimal, ROUND_DOWN, localcontext

from app.execution.assumptions import ExecutionAssumptions
from app.risk.contracts import RiskCalculation, RiskConfiguration, TradeDirection, TradeRiskRequest
from app.risk.errors import InsufficientEvidenceError, RiskRejectedError

_CALCULATION_CONTEXT = Context(prec=34)


def _require_decimal(value, field_name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise InsufficientEvidenceError(f"{field_name} is required")
    if not isinstance(value, Decimal) or not value.is_finite():
        raise InsufficientEvidenceError(f"{field_name} must be a finite Decimal")
    return value


def _resolve_execution_assumptions(request: TradeRiskRequest, config: RiskConfiguration) -> ExecutionAssumptions:
    assumptions = request.execution_assumptions or config.execution_assumptions
    if assumptions is None:
        raise InsufficientEvidenceError("execution assumptions are required to estimate downside")
    if not isinstance(assumptions, ExecutionAssumptions):
        raise InsufficientEvidenceError("execution assumptions must be an ExecutionAssumptions instance")
    return assumptions


def _round_down_quantity(raw_quantity: Decimal, increment: Decimal | None) -> Decimal:
    if increment is not None:
        if increment <= 0:
            raise RiskRejectedError("quantity_increment", "quantity increment must be positive")
        return (raw_quantity / increment).to_integral_value(ROUND_DOWN) * increment
    return raw_quantity.to_integral_value(ROUND_DOWN)


def calculate_trade_risk(request: TradeRiskRequest) -> RiskCalculation:
    # Isolate precision, rounding, traps, and flags from the caller's Decimal context.
    with localcontext(_CALCULATION_CONTEXT):
        return _calculate_trade_risk(request)


def _calculate_trade_risk(request: TradeRiskRequest) -> RiskCalculation:
    config = request.risk_configuration
    if config is None:
        raise InsufficientEvidenceError("risk configuration is required")
    if request.instrument is None:
        raise InsufficientEvidenceError("instrument is required")
    if request.side is None:
        raise InsufficientEvidenceError("trade direction is required")
    if request.entry_price is None:
        raise InsufficientEvidenceError("entry_price is required")
    if request.stop_price is None:
        raise InsufficientEvidenceError("stop_price is required")
    if request.account_equity is None:
        raise InsufficientEvidenceError("account_equity is required")
    if request.pnl_to_account_rate is None:
        raise InsufficientEvidenceError("PnL-to-account-currency conversion rate is required")

    entry = _require_decimal(request.entry_price, "entry_price")
    stop = _require_decimal(request.stop_price, "stop_price")
    equity = _require_decimal(request.account_equity, "account_equity")
    conversion = _require_decimal(request.pnl_to_account_rate, "pnl_to_account_rate")
    if equity <= 0:
        raise InsufficientEvidenceError("account_equity must be positive")
    if entry <= 0:
        raise InsufficientEvidenceError("entry_price must be positive")

    if request.side == TradeDirection.LONG:
        if stop >= entry:
            raise RiskRejectedError("long_stop_order", "LONG stop must be below entry")
        risk_per_unit = entry - stop
    elif request.side == TradeDirection.SHORT:
        if stop <= entry:
            raise RiskRejectedError("short_stop_order", "SHORT stop must be above entry")
        risk_per_unit = stop - entry
    else:
        raise InsufficientEvidenceError("direction must be LONG or SHORT")

    if risk_per_unit <= 0:
        raise RiskRejectedError("stop_distance", "stop distance must be positive")

    exec_assumptions = _resolve_execution_assumptions(request, config)
    requested_risk = equity * config.risk_per_trade
    maximum_monetary_risk = requested_risk
    increment = config.quantity_increment
    execution_cost_per_unit = exec_assumptions.spread + (Decimal(2) * exec_assumptions.slippage)
    percentage_fee = exec_assumptions.pct_fee or Decimal(0)
    # Reserve both entry and stop-exit costs before sizing. A position that fits
    # the price distance alone can still exceed the configured loss limit after costs.
    fee_base_per_unit = entry + stop + (Decimal(2) * execution_cost_per_unit)
    unit_loss = (risk_per_unit + execution_cost_per_unit + percentage_fee * fee_base_per_unit) * conversion
    fixed_fees = (Decimal(2) * exec_assumptions.fixed_fee
                  if exec_assumptions.fixed_fee is not None else Decimal(0))
    fixed_fees_in_account_currency = fixed_fees * conversion
    available_risk = maximum_monetary_risk - fixed_fees_in_account_currency
    if available_risk <= 0:
        raise RiskRejectedError("execution_costs", "fixed execution fees consume the entire risk budget")
    raw_position_size = available_risk / unit_loss
    position_size = _round_down_quantity(raw_position_size, increment)
    if position_size <= 0:
        raise RiskRejectedError("position size must be positive")

    if config.max_position_size is not None:
        position_size = min(position_size, Decimal(config.max_position_size))
        position_size = _round_down_quantity(position_size, increment)
        if position_size <= 0:
            raise RiskRejectedError("position_size", "configured maximum position size is below the minimum quantity increment")

    notional_exposure = position_size * entry * conversion
    if config.max_notional_exposure is not None and notional_exposure > config.max_notional_exposure:
        raise RiskRejectedError("max_notional_exposure", "notional exposure exceeds configured max_notional_exposure")

    if request.target_price is not None:
        target = _require_decimal(request.target_price, "target_price")
        if target <= 0:
            raise InsufficientEvidenceError("target_price must be positive")
        if request.side == TradeDirection.LONG:
            if target <= entry:
                raise RiskRejectedError("target_side", "LONG target_price must be above entry_price")
            reward_distance = target - entry
        else:
            if target >= entry:
                raise RiskRejectedError("target_side", "SHORT target_price must be below entry_price")
            reward_distance = entry - target
        reward_risk = reward_distance / risk_per_unit
    else:
        reward_distance = None
        reward_risk = None

    estimated_fees = (position_size * percentage_fee * fee_base_per_unit + fixed_fees) * conversion
    estimated_spread_cost = position_size * exec_assumptions.spread * conversion
    estimated_slippage_impact = position_size * Decimal(2) * exec_assumptions.slippage * conversion
    total_estimated_downside = (
        position_size * risk_per_unit * conversion
        + estimated_spread_cost + estimated_slippage_impact + estimated_fees
    )

    return RiskCalculation(
        maximum_monetary_risk=maximum_monetary_risk,
        stop_distance=risk_per_unit,
        risk_per_unit=risk_per_unit,
        position_size=position_size,
        notional_exposure=notional_exposure,
        estimated_fees=estimated_fees,
        estimated_spread_cost=estimated_spread_cost,
        estimated_slippage_impact=estimated_slippage_impact,
        total_estimated_downside=total_estimated_downside,
        reward_risk=reward_risk,
        reward_distance=reward_distance,
        execution_cost_per_unit=execution_cost_per_unit,
    )
