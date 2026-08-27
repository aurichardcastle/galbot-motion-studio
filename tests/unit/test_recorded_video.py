from pathlib import Path

import pytest

from galbot_motion_studio.adapters.recorded_video import (
    RecordedVideoError,
    RecordedVideoSource,
    SourceTimelineFrame,
)


class FakeCapture:
    def __init__(
        self,
        frames: list[tuple[bool, object]],
        timestamps_ms: list[float],
        *,
        opened: bool = True,
    ) -> None:
        self.frames = iter(frames)
        self.timestamps_ms = iter(timestamps_ms)
        self.opened = opened
        self.current_timestamp_ms = 0.0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, object]:
        ok, image = next(self.frames, (False, None))
        if ok:
            self.current_timestamp_ms = next(self.timestamps_ms)
        return ok, image

    def get(self, _: int) -> float:
        return self.current_timestamp_ms

    def release(self) -> None:
        self.released = True


class FakeCV2:
    CAP_PROP_POS_MSEC = 0

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture

    def VideoCapture(self, _: str) -> FakeCapture:
        return self.capture


def test_recorded_video_preserves_container_timestamps() -> None:
    capture = FakeCapture([(True, "a"), (True, "b")], [40.0, 125.5])
    source = RecordedVideoSource(Path("fixture.mp4"), source_clock_id="fixture-clock", cv2_module=FakeCV2(capture))
    frames = list(source.frames())
    assert [(frame.sequence, frame.capture_mono_ns) for frame in frames] == [
        (0, 40_000_000),
        (1, 125_500_000),
    ]
    assert capture.released


def test_recorded_video_rejects_nonincreasing_timestamps() -> None:
    capture = FakeCapture([(True, "a"), (True, "b")], [40.0, 40.0])
    source = RecordedVideoSource(Path("fixture.mp4"), source_clock_id="fixture-clock", cv2_module=FakeCV2(capture))
    with pytest.raises(RecordedVideoError, match="non-increasing"):
        list(source.frames())
    assert capture.released


def test_recorded_video_uses_an_explicit_source_timeline() -> None:
    capture = FakeCapture([(True, "a"), (True, "b")], [40.0, 125.5])
    source = RecordedVideoSource(
        Path("fixture.mp4"),
        source_clock_id="fixture-clock",
        cv2_module=FakeCV2(capture),
        source_timeline=(
            SourceTimelineFrame(source_sequence=10, source_mono_ns=1_000_000_000),
            SourceTimelineFrame(source_sequence=14, source_mono_ns=1_125_000_000),
        ),
    )
    frames = list(source.frames())
    assert [(frame.sequence, frame.capture_mono_ns) for frame in frames] == [
        (10, 1_000_000_000),
        (14, 1_125_000_000),
    ]


def test_recorded_video_rejects_a_timeline_with_missing_video_frames() -> None:
    capture = FakeCapture([(True, "a"), (True, "b")], [40.0, 125.5])
    source = RecordedVideoSource(
        Path("fixture.mp4"),
        source_clock_id="fixture-clock",
        cv2_module=FakeCV2(capture),
        source_timeline=(SourceTimelineFrame(source_sequence=10, source_mono_ns=1_000_000_000),),
    )
    with pytest.raises(RecordedVideoError, match="shorter"):
        list(source.frames())


def test_recorded_video_releases_an_unopened_capture() -> None:
    capture = FakeCapture([], [], opened=False)
    source = RecordedVideoSource(Path("missing.mp4"), source_clock_id="fixture-clock", cv2_module=FakeCV2(capture))
    with pytest.raises(RecordedVideoError, match="cannot open"):
        list(source.frames())
    assert capture.released
