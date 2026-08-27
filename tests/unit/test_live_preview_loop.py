"""The live camera source, driven without a camera.

The live path is the demo path, and it is the one path a developer machine
normally cannot execute: macOS denies camera access to a non-interactive
process, so `cv2.VideoCapture(0)` fails and every live-only branch would ship
having never run.

`WebcamSource` takes its `cv2` module by injection, which is the seam that makes
the live path testable at all.  These tests pin that seam and the two properties
the rest of the live stack assumes of it -- ordered frames and strictly
increasing capture stamps.  `tests/test_live_preview_end_to_end.py` uses the
same seam to run the whole `preview` command.
"""

from __future__ import annotations

import numpy as np

from galbot_motion_studio.adapters.webcam import WebcamSource
from galbot_motion_studio.ports.frames import CapturedFrame


class FakeCapture:
    """A `cv2.VideoCapture` stand-in that serves synthetic frames indefinitely.

    Deliberately endless: a real camera does not stop, and `WebcamSource` treats
    a failed `read()` as `WebcamError` -- correctly.  The consumer bounds the run
    with `max_frames`, exactly as the live CLI does.
    """

    def __init__(self, size: tuple[int, int] = (240, 320)) -> None:
        self._index = 0
        self._size = size
        self.released = False

    def isOpened(self) -> bool:  # cv2 spelling, not ours
        return True

    def read(self):
        height, width = self._size
        image = np.full((height, width, 3), self._index % 251, dtype=np.uint8)
        self._index += 1
        return True, image

    def release(self) -> None:
        self.released = True


class FakeCv2:
    def __init__(self, capture: FakeCapture) -> None:
        self._capture = capture

    def VideoCapture(self, device):  # cv2 spelling, not ours
        return self._capture


def test_the_webcam_source_is_injectable_so_the_live_path_is_testable() -> None:
    """The seam these tests depend on, asserted rather than assumed."""
    capture = FakeCapture()
    source = WebcamSource(0, cv2_module=FakeCv2(capture))
    frames = list(source.frames(max_frames=5))
    assert len(frames) == 5
    assert all(isinstance(f, CapturedFrame) for f in frames)
    assert [f.sequence for f in frames] == sorted(f.sequence for f in frames)
    assert all(f.source_kind == "webcam:0" for f in frames)


def test_capture_timestamps_strictly_increase_across_the_live_source() -> None:
    """MediaPipe's VIDEO mode rejects a non-increasing timestamp with a bare
    ValueError, which the perception thread must not be able to trip on its own."""
    capture = FakeCapture()
    frames = list(WebcamSource(0, cv2_module=FakeCv2(capture)).frames(max_frames=12))
    stamps = [f.capture_mono_ns for f in frames]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
