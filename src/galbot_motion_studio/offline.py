"""Deterministic, every-frame processing for recorded-video analysis.

Live preview deliberately uses :class:`LatestWinsProcessor` to stay responsive.
That policy is wrong for a saved-video regression: dropping one decoded frame
changes the stateful calibration, filters, and safety trajectory.  This module
provides the deliberately boring alternative used by ``preview --analysis-sync``.
"""

from __future__ import annotations

from statistics import mean
from time import monotonic_ns
from typing import Callable, Generic, TypeVar

from galbot_motion_studio.realtime import DECLINED, RealtimeMetrics, TimedOutput


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class SynchronousProcessor(Generic[InputT, OutputT]):
    """Process each submitted item once, in submission order, with no queue."""

    def __init__(
        self,
        process: Callable[[InputT], OutputT],
        *,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        self._process = process
        self._clock_ns = clock_ns
        self._submitted = 0
        self._processed = 0
        self._declined = 0
        self._processing_ms: list[float] = []
        self._first_start_ns: int | None = None
        self._last_complete_ns: int | None = None

    def process(self, item: InputT) -> TimedOutput[InputT, OutputT] | None:
        """Run ``item`` immediately and return its result and host timing.

        Returns ``None`` when the callable returns ``DECLINED`` -- it produced no
        output, so there is nothing to time and nothing to hand back. Without
        this the sentinel was wrapped in a ``TimedOutput`` and counted as a
        processed frame, so the two processors disagreed about what the same
        value means and the synchronous one's metrics over-counted.
        """
        self._submitted += 1
        started = self._clock_ns()
        if self._first_start_ns is None:
            self._first_start_ns = started
        result = self._process(item)
        if result is DECLINED:
            self._declined += 1
            return None
        completed = self._clock_ns()
        processing_ms = (completed - started) / 1_000_000
        self._processed += 1
        self._processing_ms.append(processing_ms)
        self._last_complete_ns = completed
        return TimedOutput(
            item=item,
            result=result,
            processing_ms=processing_ms,
            queue_latency_ms=0.0,
            completed_host_mono_ns=completed,
        )

    @property
    def metrics(self) -> RealtimeMetrics:
        ordered = sorted(self._processing_ms)
        p95_index = max(0, int(0.95 * (len(ordered) - 1))) if ordered else 0
        duration_s = 0.0
        if self._first_start_ns is not None and self._last_complete_ns is not None:
            duration_s = (self._last_complete_ns - self._first_start_ns) / 1_000_000_000
        return RealtimeMetrics(
            submitted=self._submitted,
            processed=self._processed,
            dropped_before_processing=0,
            declined=self._declined,
            mean_processing_ms=mean(self._processing_ms) if self._processing_ms else 0.0,
            p95_processing_ms=ordered[p95_index] if ordered else 0.0,
            mean_queue_latency_ms=0.0,
            effective_fps=self._processed / duration_s if duration_s > 0 else 0.0,
        )
