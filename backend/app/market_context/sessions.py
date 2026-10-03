"""Timezone-aware session labels; generic windows do not assert exchange hours."""

from datetime import date, datetime, time, timezone
from enum import Enum
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator

from app.market_data.contracts import MarketDataModel, require_aware_utc


class CalendarStatus(str, Enum):
    UNKNOWN = "unknown"
    OPEN = "open"
    CLOSED = "closed"


class SessionDefinition(MarketDataModel):
    name: str = Field(min_length=1, max_length=64)
    timezone: str
    local_start: time
    local_end: time
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Session name cannot be blank")
        return normalized

    @field_validator("timezone")
    @classmethod
    def valid_iana_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown IANA timezone: {value}") from exc
        return value

    @field_validator("local_start", "local_end")
    @classmethod
    def timezone_free_wall_time(cls, value: time) -> time:
        if value.tzinfo is not None:
            raise ValueError("Session window times must be local wall-clock times without tzinfo")
        return value

    @field_validator("weekdays")
    @classmethod
    def valid_weekdays(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(day < 0 or day > 6 for day in value) or len(set(value)) != len(value):
            raise ValueError("weekdays must contain unique ISO weekday indexes 0..6")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def nonempty_window(self) -> "SessionDefinition":
        if self.local_start == self.local_end:
            raise ValueError("Session start and end cannot be equal")
        return self

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def contains(self, timestamp: datetime) -> bool:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Session classification requires a timezone-aware timestamp")
        local = timestamp.astimezone(self.zone)
        current = local.timetz().replace(tzinfo=None)
        weekday = local.weekday()
        if self.local_start < self.local_end:
            return weekday in self.weekdays and self.local_start <= current < self.local_end
        # Overnight session: weekdays govern the day on which the window opens.
        if current >= self.local_start:
            return weekday in self.weekdays
        previous_weekday = (weekday - 1) % 7
        return current < self.local_end and previous_weekday in self.weekdays


class MarketCalendar(Protocol):
    def status_at(self, timestamp: datetime) -> CalendarStatus:
        """Return an authoritative market status for the supplied instant."""
        ...


class SessionClassification(MarketDataModel):
    timestamp: datetime
    labels: tuple[str, ...]
    local_date: date
    weekend: bool
    calendar_status: CalendarStatus = CalendarStatus.UNKNOWN

    @field_validator("timestamp")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        return require_aware_utc(value, "timestamp")


class SessionClassifier:
    def __init__(self, definitions: tuple[SessionDefinition, ...], *, calendar_timezone: str):
        self.definitions = definitions
        try:
            self.calendar_zone = ZoneInfo(calendar_timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown calendar IANA timezone: {calendar_timezone}") from exc
        names = [definition.name for definition in definitions]
        if len(names) != len(set(names)):
            raise ValueError("Session names must be unique")

    def classify(
        self,
        timestamp: datetime,
        *,
        market_calendar: MarketCalendar | None = None,
    ) -> SessionClassification:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Session classification requires a timezone-aware timestamp")
        utc_timestamp = timestamp.astimezone(timezone.utc)
        local_calendar_time = utc_timestamp.astimezone(self.calendar_zone)
        labels = tuple(sorted(
            definition.name for definition in self.definitions if definition.contains(utc_timestamp)
        ))
        status = market_calendar.status_at(utc_timestamp) if market_calendar else CalendarStatus.UNKNOWN
        return SessionClassification(
            timestamp=utc_timestamp,
            labels=labels,
            local_date=local_calendar_time.date(),
            weekend=local_calendar_time.weekday() >= 5,
            calendar_status=status,
        )


def generic_session_definitions() -> tuple[SessionDefinition, ...]:
    """Illustrative configurable windows, not broker or exchange trading hours."""
    weekdays = (0, 1, 2, 3, 4)
    return (
        SessionDefinition(name="Asia", timezone="Asia/Tokyo", local_start=time(9), local_end=time(18), weekdays=weekdays),
        SessionDefinition(name="London", timezone="Europe/London", local_start=time(8), local_end=time(17), weekdays=weekdays),
        SessionDefinition(name="New York", timezone="America/New_York", local_start=time(8), local_end=time(17), weekdays=weekdays),
    )
