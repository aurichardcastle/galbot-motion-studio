"""Deterministic recorded-video source, requiring neither camera permission nor MediaPipe."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterator

from galbot_motion_studio.ports.frames import CapturedFrame


class RecordedVideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceTimelineFrame:
    """The original command-clock identity of one encoded video frame."""

    source_sequence: int
    source_mono_ns: int


class RecordedVideoSource:
    """Yield BGR frames using each frame's container presentation timestamp.

    Recorded-video timestamps are source-local presentation time, not claimed live
    wall-clock telemetry. The exporter therefore receives them as raw provenance
    and decides explicitly whether/how to resample.
    """

    def __init__(
        self,
        path: Path,
        *,
        source_clock_id: str,
        cv2_module: Any | None = None,
        source_timeline: tuple[SourceTimelineFrame, ...] | None = None,
    ) -> None:
        if not source_clock_id:
            raise ValueError("source_clock_id is required")
        self._path = path
        self._source_clock_id = source_clock_id
        self._cv2 = cv2_module
        self._source_timeline = source_timeline

    def frames(self) -> Iterator[CapturedFrame]:
        cv2 = self._cv2 or import_module("cv2")
        capture = cv2.VideoCapture(str(self._path))
        try:
            if not capture.isOpened():
                raise RecordedVideoError(f"cannot open recorded video: {self._path}")
            sequence = 0
            last_timestamp_ns = -1
            while True:
                ok, image = capture.read()
                if not ok:
                    if (
                        self._source_timeline is not None
                        and sequence != len(self._source_timeline)
                    ):
                        raise RecordedVideoError(
                            "source timeline length does not match decoded video frames"
                        )
                    return
                if self._source_timeline is None:
                    source_sequence = sequence
                    timestamp_ns = int(capture.get(cv2.CAP_PROP_POS_MSEC) * 1_000_000)
                else:
                    if sequence >= len(self._source_timeline):
                        raise RecordedVideoError(
                            "source timeline is shorter than decoded video"
                        )
                    mapped = self._source_timeline[sequence]
                    source_sequence = mapped.source_sequence
                    timestamp_ns = mapped.source_mono_ns
                if timestamp_ns <= last_timestamp_ns:
                    raise RecordedVideoError("recorded video has non-increasing frame timestamps")
                yield CapturedFrame(
                    sequence=source_sequence,
                    source_clock_id=self._source_clock_id,
                    capture_mono_ns=timestamp_ns,
                    image_bgr=image,
                    source_kind="recorded_video",
                )
                last_timestamp_ns = timestamp_ns
                sequence += 1
        finally:
            capture.release()
