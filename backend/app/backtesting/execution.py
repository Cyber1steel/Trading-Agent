from decimal import Context, Decimal, localcontext
from enum import Enum
from typing import NamedTuple
from app.execution.assumptions import ExecutionAssumptions


_EXECUTION_CONTEXT = Context(prec=34)


class ExecutionResult(NamedTuple):
    requested_price: Decimal
    executed_price: Decimal
    spread: Decimal
    slippage: Decimal
    fees: Decimal
    spread_cost: Decimal
    slippage_cost: Decimal


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


def simulate_fill(requested_price: Decimal, qty: int, assumptions: ExecutionAssumptions,
                  order_side: OrderSide = OrderSide.BUY) -> ExecutionResult:
    with localcontext(_EXECUTION_CONTEXT):
        if not isinstance(requested_price, Decimal) or not requested_price.is_finite() or requested_price <= 0:
            raise ValueError("requested_price must be a positive finite Decimal")
        if type(qty) is not int or qty <= 0:
            raise ValueError("qty must be a positive integer")
        if not isinstance(order_side, OrderSide):
            raise ValueError("order_side must be BUY or SELL")
        # Trade and midpoint references apply half the quoted spread adversely
        # on each side. Bid/ask references are adjusted to the opposite quote.
        half_spread = assumptions.spread / Decimal(2)
        if assumptions.price_basis.value in ("trade", "mid"):
            spread_adjustment = half_spread if order_side is OrderSide.BUY else -half_spread
        elif assumptions.price_basis.value == "bid":
            spread_adjustment = assumptions.spread if order_side is OrderSide.BUY else Decimal(0)
        else:
            spread_adjustment = Decimal(0) if order_side is OrderSide.BUY else -assumptions.spread
        slippage_adjustment = assumptions.slippage if order_side is OrderSide.BUY else -assumptions.slippage
        executed = requested_price + spread_adjustment + slippage_adjustment
        if executed <= 0:
            raise ValueError("execution assumptions produce a non-positive fill price")
        notional = executed * Decimal(qty)
        fees = Decimal(0)
        if assumptions.pct_fee is not None:
            fees += notional * assumptions.pct_fee
        if assumptions.fixed_fee is not None:
            fees += assumptions.fixed_fee
        return ExecutionResult(
            requested_price=requested_price,
            executed_price=executed,
            spread=assumptions.spread,
            slippage=assumptions.slippage,
            fees=fees,
            spread_cost=abs(spread_adjustment) * Decimal(qty),
            slippage_cost=assumptions.slippage * Decimal(qty),
        )
