"""Forward-only fixed-duration parent-bar alignment."""

from app.market_analysis.contracts import AvailabilityReason, HigherTimeframeContext, StructuralState


def align_parent_context(observation_times, parent_candles, parent_states, parent_swings, timeframe):
    """For each known-at instant select the latest parent candle already closed."""
    output = []
    pointer = 0
    swing_pointer = 0
    for known_at in observation_times:
        newly_confirmed = []
        while swing_pointer < len(parent_swings) and parent_swings[swing_pointer].known_at <= known_at:
            newly_confirmed.append(parent_swings[swing_pointer])
            swing_pointer += 1
        while pointer < len(parent_candles):
            candle = parent_candles[pointer]
            close_at = candle.timestamp + timeframe.nominal_duration
            if close_at <= known_at:
                pointer += 1
            else:
                break
        index = pointer - 1
        if index < 0:
            output.append(HigherTimeframeContext(timeframe=timeframe, bar_open=None, known_at=known_at,
                close=None, structural_state=StructuralState.INSUFFICIENT,
                unavailable_reason=AvailabilityReason.NO_CLOSED_PARENT_BAR))
        else:
            candle = parent_candles[index]
            close_at = candle.timestamp + timeframe.nominal_duration
            output.append(HigherTimeframeContext(timeframe=timeframe, bar_open=candle.timestamp,
                known_at=known_at, close=candle.close, structural_state=parent_states[index],
                confirmed_swings=tuple(s for s in newly_confirmed if s.known_at <= close_at)))
    return tuple(output)
