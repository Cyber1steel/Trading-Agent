"""Deterministic true range and Wilder ATR."""

from decimal import Decimal

from app.market_analysis.contracts import AvailabilityReason, DecimalValue
from app.market_analysis.features import market_context
from app.market_data.contracts import Candle


def true_ranges(candles: tuple[Candle, ...]) -> tuple[Decimal, ...]:
    values = []
    with market_context():
        for i, candle in enumerate(candles):
            span = candle.high - candle.low
            if i == 0:
                values.append(span)
            else:
                previous = candles[i - 1].close
                values.append(max(span, abs(candle.high - previous), abs(candle.low - previous)))
    return tuple(values)


def average_true_ranges(ranges: tuple[Decimal, ...], period: int) -> tuple[DecimalValue, ...]:
    result: list[DecimalValue] = []
    with market_context():
        atr = None
        for index, value in enumerate(ranges):
            if index + 1 < period:
                result.append(DecimalValue(value=None, unavailable_reason=AvailabilityReason.INSUFFICIENT_HISTORY))
            elif index + 1 == period:
                atr = sum(ranges[:period], Decimal(0)) / Decimal(period)
                result.append(DecimalValue(value=atr))
            else:
                assert atr is not None
                atr = (atr * Decimal(period - 1) + value) / Decimal(period)
                result.append(DecimalValue(value=atr))
    return tuple(result)
