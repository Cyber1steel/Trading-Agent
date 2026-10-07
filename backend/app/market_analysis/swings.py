"""Strict confirmed swing detection; events are never backfilled."""

from app.market_analysis.contracts import ConfirmedSwing, SwingKind
from app.market_data.contracts import Candle


def detect_swings(candles: tuple[Candle, ...], left: int, right: int) -> tuple[ConfirmedSwing, ...]:
    events: list[ConfirmedSwing] = []
    for i in range(left, len(candles) - right):
        neighbors = (*candles[i-left:i], *candles[i+1:i+right+1])
        high = all(candles[i].high > item.high for item in neighbors)
        low = all(candles[i].low < item.low for item in neighbors)
        if high or low:
            known = candles[i + right].timestamp + candles[i].timeframe.nominal_duration
            if high:
                events.append(ConfirmedSwing(kind=SwingKind.HIGH, candidate_at=candles[i].timestamp,
                    confirmed_at=known, known_at=known, price=candles[i].high))
            if low:
                events.append(ConfirmedSwing(kind=SwingKind.LOW, candidate_at=candles[i].timestamp,
                    confirmed_at=known, known_at=known, price=candles[i].low))
    return tuple(sorted(events, key=lambda event: (event.known_at, event.candidate_at, event.kind.value)))
