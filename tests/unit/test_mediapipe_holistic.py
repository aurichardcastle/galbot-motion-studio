import pytest
import numpy as np
from types import SimpleNamespace

from galbot_motion_studio.adapters.mediapipe_holistic import (
    HolisticDetectorError,
    MediaPipeHolisticDetector,
)
from galbot_motion_studio.contracts.human import IdentityState
from galbot_motion_studio.ports.frames import CapturedFrame, content_fingerprint_for


def test_frame_fingerprint_is_shared_by_independent_inference_consumers() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    fallback = CapturedFrame(0, "clock", 1, image, "webcam")
    asserted = CapturedFrame(0, "clock", 1, image, "webcam", "capture-digest")

    assert content_fingerprint_for(fallback) == "60daa3a5f7dbfa200f8c82840ecf5b42640b70f3b7218a4c6bbd67db542e75a4"
    assert content_fingerprint_for(asserted) == "capture-digest"


def test_holistic_detector_runs_offline_on_a_blank_fixture() -> None:
    frame = CapturedFrame(
        sequence=0,
        source_clock_id="recorded-fixture-clock",
        capture_mono_ns=1_000_000,
        image_bgr=np.zeros((48, 64, 3), dtype=np.uint8),
        source_kind="recorded_video",
    )
    with MediaPipeHolisticDetector() as detector:
        observation = detector.detect(frame, calibration_id="fixture-calibration")
    assert observation.identity is IdentityState.LOST
    assert observation.landmarks == ()
    assert observation.aggregate_confidence == 0.0


def test_holistic_detector_rejects_regressed_capture_clock_before_inference() -> None:
    first = CapturedFrame(
        sequence=0,
        source_clock_id="recorded-fixture-clock",
        capture_mono_ns=2_000_000,
        image_bgr=np.zeros((48, 64, 3), dtype=np.uint8),
        source_kind="recorded_video",
    )
    regressed = CapturedFrame(
        sequence=1,
        source_clock_id="recorded-fixture-clock",
        capture_mono_ns=1_000_000,
        image_bgr=np.zeros((48, 64, 3), dtype=np.uint8),
        source_kind="recorded_video",
    )
    with MediaPipeHolisticDetector() as detector:
        detector.detect(first, calibration_id="fixture-calibration")
        with pytest.raises(HolisticDetectorError, match="strictly monotonic"):
            detector.detect(regressed, calibration_id="fixture-calibration")


def test_optional_mediapipe_confidence_fields_do_not_crash_real_landmarks() -> None:
    landmark = SimpleNamespace(
        x=0.1,
        y=0.2,
        z=0.3,
        visibility=None,
        presence=None,
    )
    result = SimpleNamespace(
        face_landmarks=[landmark],
        pose_landmarks=[landmark],
        pose_world_landmarks=[],
        left_hand_landmarks=[landmark],
        left_hand_world_landmarks=[],
        right_hand_landmarks=[landmark],
        right_hand_world_landmarks=[],
    )

    converted = MediaPipeHolisticDetector._to_landmarks(result)

    assert len(converted) == 4
    assert all(item.visibility == 1.0 for item in converted)
    assert all(item.presence == 1.0 for item in converted)


def test_dedicated_hand_wrist_replaces_low_confidence_pose_wrist() -> None:
    pose = SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.05, presence=0.05)
    hand = SimpleNamespace(x=0.8, y=0.4, z=0.1, visibility=None, presence=None)
    result = SimpleNamespace(
        face_landmarks=[],
        pose_landmarks=[pose] * 33,
        pose_world_landmarks=[],
        left_hand_landmarks=[hand],
        left_hand_world_landmarks=[],
        right_hand_landmarks=[],
        right_hand_world_landmarks=[],
    )
    converted = {item.name: item for item in MediaPipeHolisticDetector._to_landmarks(result)}
    assert converted["pose_15"].normalized_xyz == (0.8, 0.4, 0.1)
    assert converted["pose_15"].visibility == 1.0
    assert converted["pose_15"].presence == 1.0


def test_hand_world_wrist_never_replaces_body_centred_pose_world_wrist() -> None:
    pose = SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.05, presence=0.05)
    pose_world = SimpleNamespace(x=1.0, y=2.0, z=3.0)
    hand = SimpleNamespace(x=0.8, y=0.4, z=0.1, visibility=None, presence=None)
    # MediaPipe hand-world coordinates use a wrist-centred origin.  These values
    # must stay with left_hand_0 rather than corrupting the body-centred pose arm.
    hand_world = SimpleNamespace(x=0.01, y=-0.02, z=0.03)
    result = SimpleNamespace(
        face_landmarks=[],
        pose_landmarks=[pose] * 33,
        pose_world_landmarks=[pose_world] * 33,
        left_hand_landmarks=[hand],
        left_hand_world_landmarks=[hand_world],
        right_hand_landmarks=[],
        right_hand_world_landmarks=[],
    )
    converted = {item.name: item for item in MediaPipeHolisticDetector._to_landmarks(result)}
    assert converted["pose_15"].normalized_xyz == (0.8, 0.4, 0.1)
    assert converted["pose_15"].world_xyz_m == (1.0, 2.0, 3.0)
    assert converted["left_hand_0"].world_xyz_m == (0.01, -0.02, 0.03)
