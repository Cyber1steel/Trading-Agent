"""Pure candle feature calculations using Decimal only."""

from decimal import Decimal, Context, ROUND_HALF_EVEN, localcontext

from app.market_analysis.contracts import AvailabilityReason, CandleFeatures, DecimalValue
from app.market_data.contracts import Candle


def market_context():
    return localcontext(Context(prec=34, rounding=ROUND_HALF_EVEN))


def candle_features(candle: Candle, previous_close: Decimal | None = None) -> CandleFeatures:
    with market_context():
        span = candle.high - candle.low
        body = abs(candle.close - candle.open)
        upper = candle.high - max(candle.open, candle.close)
        lower = min(candle.open, candle.close) - candle.low
        if span == 0:
            ratio = DecimalValue(value=None, unavailable_reason=AvailabilityReason.ZERO_RANGE)
            location = DecimalValue(value=None, unavailable_reason=AvailabilityReason.ZERO_RANGE)
        else:
            ratio = DecimalValue(value=body / span)
            location = DecimalValue(value=(candle.close - candle.low) / span)
        if previous_close is None:
            simple_return = DecimalValue(value=None, unavailable_reason=AvailabilityReason.NO_PREVIOUS_CLOSE)
        elif previous_close <= 0 or candle.close <= 0:
            simple_return = DecimalValue(value=None, unavailable_reason=AvailabilityReason.NON_POSITIVE_CLOSE)
        else:
            simple_return = DecimalValue(value=candle.close / previous_close - Decimal(1))
        return CandleFeatures(range=span, body=body, upper_wick=upper, lower_wick=lower,
                              body_ratio=ratio, close_location=location, simple_return=simple_return)
