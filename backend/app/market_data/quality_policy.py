"""Optional authoritative completeness policy boundary."""

from collections.abc import Sequence
from typing import Protocol

from datetime import datetime

from app.market_data.contracts import Instrument
from app.market_data.timeframes import Timeframe


class CompletenessPolicy(Protocol):
    def expected_bar_opens(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> Sequence[datetime]:
        """Return authoritative UTC bar-open instants for [start, end)."""
        ...
