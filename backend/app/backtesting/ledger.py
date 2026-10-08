from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List
from app.core.immutability import deep_freeze


@dataclass(frozen=True)
class Event:
    seq: int
    type: str
    known_at: datetime
    reason: str
    payload: dict


class EventLedger:
    def __init__(self):
        self._events: List[Event] = []

    def append(self, type: str, known_at: datetime, reason: str, payload: dict | None = None) -> Event:
        if known_at.tzinfo is None or known_at.utcoffset() is None:
            raise ValueError("known_at must be timezone-aware")
        seq = len(self._events) + 1
        event = Event(seq=seq, type=type, known_at=known_at.astimezone(timezone.utc), reason=reason,
                      payload=deep_freeze(payload or {}))
        self._events.append(event)
        return event

    @property
    def events(self) -> List[Event]:
        return list(self._events)
