import numpy as np
import pytest

from galbot_motion_studio.adapters.webcam import WebcamError, WebcamSource


class FakeCapture:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.released = False
        self.calls = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        self.calls += 1
        return True, np.zeros((4, 5, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class FakeCV2:
    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture

    def VideoCapture(self, _device: int) -> FakeCapture:
        return self.capture


def test_webcam_is_bounded_and_released() -> None:
    capture = FakeCapture()
    frames = list(WebcamSource(cv2_module=FakeCV2(capture)).frames(max_frames=2))
    assert len(frames) == 2
    assert frames[1].sequence > frames[0].sequence
    assert capture.released


def test_unopened_webcam_is_released_and_fails() -> None:
    capture = FakeCapture(opened=False)
    with pytest.raises(WebcamError):
        list(WebcamSource(cv2_module=FakeCV2(capture)).frames(max_frames=1))
    assert capture.released


def test_negative_webcam_index_is_rejected_before_native_capture() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        WebcamSource(device=-1)
