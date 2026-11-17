from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ClockError(ValueError):
    pass


class CompetitionClock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """Injectable clock for deterministic tests and timeline replay."""

    def __init__(self, current: datetime) -> None:
        self._current = ensure_aware(current, "FrozenClock.current")

    def now(self) -> datetime:
        return self._current

    def set(self, current: datetime) -> None:
        self._current = ensure_aware(current, "FrozenClock.current")

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self._current = self._current + timedelta(seconds=seconds)


def ensure_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ClockError(f"{name} must be timezone-aware")
    return value


def ensure_utc(value: datetime, name: str) -> datetime:
    return ensure_aware(value, name).astimezone(UTC)


def zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ClockError(f"unknown timezone: {name}") from exc


__all__ = [
    "ClockError",
    "CompetitionClock",
    "FrozenClock",
    "SystemClock",
    "ensure_aware",
    "ensure_utc",
    "zone",
]
