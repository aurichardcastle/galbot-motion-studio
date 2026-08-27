"""Bounded latest-wins processing for responsive live preview."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from statistics import mean
from threading import Event, Thread
from time import monotonic_ns
from typing import Callable, Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class TimedOutput(Generic[InputT, OutputT]):
    item: InputT
    result: OutputT
    processing_ms: float
    queue_latency_ms: float
    completed_host_mono_ns: int


@dataclass(frozen=True)
class RealtimeMetrics:
    submitted: int
    processed: int
    dropped_before_processing: int
    mean_processing_ms: float
    p95_processing_ms: float
    mean_queue_latency_ms: float
    effective_fps: float
    #: Items the process callable declined (see DECLINED): dequeued, but no
    #: output. Counted separately so `submitted - processed - dropped - declined`
    #: still lands at zero and nothing mistakes a decline for a dropped frame.
    declined: int = 0


@dataclass(frozen=True)
class _Queued(Generic[InputT]):
    item: InputT
    submitted_ns: int


class RealtimeWorkerError(RuntimeError):
    """A background processing failure surfaced on the caller thread."""


class _Declined:
    """Singleton returned by a process callable that produced no output."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "DECLINED"


#: Return this from the process callable to say "this item yielded nothing".
#: The worker then publishes NOTHING for it -- which matters because `_results`
#: holds a single slot and overwrites: a placeholder result would silently
#: destroy the last real one before the caller ever polled for it. Nor is a
#: declined item counted as processed, because it was not.
DECLINED = _Declined()


class LatestWinsProcessor(Generic[InputT, OutputT]):
    """One worker, one pending item, one latest result.

    Submitting while the worker is busy replaces the pending item rather than
    growing a latency-producing queue. The item already executing is never
    interrupted; its result can be superseded by a newer result at the consumer.
    """

    def __init__(self, process: Callable[[InputT], OutputT], *, name: str = "motion-studio") -> None:
        self._process = process
        self._pending: Queue[_Queued[InputT] | None] = Queue(maxsize=1)
        self._results: Queue[TimedOutput[InputT, OutputT]] = Queue(maxsize=1)
        self._stop = Event()
        self._thread = Thread(target=self._run, name=name, daemon=True)
        self._started = False
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._declined = 0
        # Diagnostics use a bounded rolling window so a long camera session
        # cannot grow memory indefinitely.
        self._processing_ms: deque[float] = deque(maxlen=4096)
        self._queue_ms: deque[float] = deque(maxlen=4096)
        self._first_start_ns: int | None = None
        self._last_complete_ns: int | None = None
        self._failure: BaseException | None = None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("latest-wins processor already started")
        self._started = True
        self._thread.start()

    def submit(self, item: InputT) -> None:
        self._raise_if_failed()
        if not self._started or self._stop.is_set():
            raise RuntimeError("latest-wins processor is not running")
        queued = _Queued(item, monotonic_ns())
        self._submitted += 1
        try:
            self._pending.put_nowait(queued)
            return
        except Full:
            pass
        try:
            replaced = self._pending.get_nowait()
            if replaced is not None:
                self._dropped += 1
        except Empty:
            pass
        self._pending.put_nowait(queued)

    def poll_latest(self, *, timeout_s: float = 0.0) -> TimedOutput[InputT, OutputT] | None:
        self._raise_if_failed()
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        try:
            latest = self._results.get(timeout=timeout_s)
        except Empty:
            self._raise_if_failed()
            return None
        while True:
            try:
                latest = self._results.get_nowait()
            except Empty:
                return latest

    def close(self, *, drain: bool = True) -> None:
        if not self._started:
            return
        if drain:
            while not self._pending.empty():
                self._thread.join(timeout=0.005)
                if not self._thread.is_alive():
                    break
        self._stop.set()
        try:
            self._pending.put_nowait(None)
        except Full:
            try:
                self._pending.get_nowait()
            except Empty:
                pass
            self._pending.put_nowait(None)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("latest-wins worker did not stop")
        self._raise_if_failed()

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
            dropped_before_processing=self._dropped,
            declined=self._declined,
            mean_processing_ms=mean(self._processing_ms) if self._processing_ms else 0.0,
            p95_processing_ms=ordered[p95_index] if ordered else 0.0,
            mean_queue_latency_ms=mean(self._queue_ms) if self._queue_ms else 0.0,
            effective_fps=self._processed / duration_s if duration_s > 0 else 0.0,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            queued = self._pending.get()
            if queued is None:
                return
            started = monotonic_ns()
            if self._first_start_ns is None:
                self._first_start_ns = started
            try:
                result = self._process(queued.item)
            except BaseException as error:
                # Unhandled daemon-thread exceptions otherwise leave the UI
                # running with no new output. Retain and surface the original
                # failure at the caller's next submit/poll/close boundary.
                self._failure = error
                self._stop.set()
                return
            if result is DECLINED:
                # Nothing to publish -- see DECLINED. Counted as declined rather
                # than processed, so the three counters still add up.
                self._declined += 1
                continue
            completed = monotonic_ns()
            timed = TimedOutput(
                item=queued.item,
                result=result,
                processing_ms=(completed - started) / 1_000_000,
                queue_latency_ms=(started - queued.submitted_ns) / 1_000_000,
                completed_host_mono_ns=completed,
            )
            self._processed += 1
            self._processing_ms.append(timed.processing_ms)
            self._queue_ms.append(timed.queue_latency_ms)
            self._last_complete_ns = completed
            try:
                self._results.put_nowait(timed)
            except Full:
                try:
                    self._results.get_nowait()
                except Empty:
                    pass
                self._results.put_nowait(timed)

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RealtimeWorkerError("latest-wins worker failed") from self._failure

    def __enter__(self) -> "LatestWinsProcessor[InputT, OutputT]":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
