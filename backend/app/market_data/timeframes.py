"""Canonical market bar intervals and their nominal elapsed durations."""

from __future__ import annotations

from datetime import timedelta
from enum import Enum


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def nominal_duration(self) -> timedelta:
        """Return the nominal elapsed duration, not a market-calendar boundary."""
        return {
            Timeframe.M1: timedelta(minutes=1),
            Timeframe.M5: timedelta(minutes=5),
            Timeframe.M15: timedelta(minutes=15),
            Timeframe.M30: timedelta(minutes=30),
            Timeframe.H1: timedelta(hours=1),
            Timeframe.H4: timedelta(hours=4),
            Timeframe.D1: timedelta(days=1),
            Timeframe.W1: timedelta(days=7),
        }[self]

    @property
    def is_calendar_anchored(self) -> bool:
        """Daily/weekly bars can follow a market calendar instead of fixed UTC spans."""
        return self in {Timeframe.D1, Timeframe.W1}

    @classmethod
    def parse(cls, value: str | "Timeframe") -> "Timeframe":
        if isinstance(value, cls):
            return value
        try:
            return cls(value.strip().lower())
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"Unsupported timeframe: {value!r}") from exc

    def __str__(self) -> str:
        return self.value
