"""Low-lag One Euro filtering for noisy webcam intent signals."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi


@dataclass
class OneEuroFilter:
    min_cutoff_hz: float = 1.5
    beta: float = 0.35
    derivative_cutoff_hz: float = 1.0
    _value: float | None = None
    _derivative: float = 0.0
    _timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        if self.min_cutoff_hz <= 0 or self.derivative_cutoff_hz <= 0 or self.beta < 0:
            raise ValueError("One Euro parameters must be positive (beta may be zero)")

    def filter(self, value: float, *, timestamp_ns: int) -> float:
        if not isfinite(value) or timestamp_ns < 0:
            raise ValueError("filter input must be finite with a non-negative timestamp")
        if self._value is None:
            self._value = value
            self._timestamp_ns = timestamp_ns
            return value
        if self._timestamp_ns is None or timestamp_ns <= self._timestamp_ns:
            raise ValueError("filter timestamps must strictly increase")
        elapsed_s = (timestamp_ns - self._timestamp_ns) / 1_000_000_000
        raw_derivative = (value - self._value) / elapsed_s
        derivative_alpha = self._alpha(self.derivative_cutoff_hz, elapsed_s)
        self._derivative += derivative_alpha * (raw_derivative - self._derivative)
        cutoff = self.min_cutoff_hz + self.beta * abs(self._derivative)
        alpha = self._alpha(cutoff, elapsed_s)
        self._value += alpha * (value - self._value)
        self._timestamp_ns = timestamp_ns
        return self._value

    @staticmethod
    def _alpha(cutoff_hz: float, elapsed_s: float) -> float:
        time_constant = 1.0 / (2.0 * pi * cutoff_hz)
        return 1.0 / (1.0 + time_constant / elapsed_s)


class IntentFilterBank:
    def __init__(self) -> None:
        self._filters: dict[str, OneEuroFilter] = {}

    def value(self, name: str, value: float, *, timestamp_ns: int) -> float:
        return self._filters.setdefault(name, OneEuroFilter()).filter(
            value, timestamp_ns=timestamp_ns
        )

    def forget(self, *names: str) -> None:
        """Discard only the named filter histories at an explicit input switch."""
        for name in names:
            self._filters.pop(name, None)
