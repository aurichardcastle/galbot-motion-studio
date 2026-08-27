import pytest

from galbot_motion_studio.contracts.human import HumanObservation, IdentityState, Landmark
from galbot_motion_studio.vision.calibration import (
    CalibrationError,
    CalibrationWindowPolicy,
    create_neutral_calibration,
    create_neutral_calibration_window,
)


WINDOW_POLICY = CalibrationWindowPolicy(
    min_samples=3,
    max_window_ns=100,
    max_center_deviation_normalized=0.02,
    max_shoulder_width_deviation_normalized=0.02,
    max_eye_span_deviation_normalized=0.02,
)


def stable_observation(**changes: object) -> HumanObservation:
    values: dict[str, object] = {
        "session_id": "session",
        "sequence": 2,
        "source_clock_id": "camera-clock",
        "source_mono_ns": 2,
        "camera_id": "webcam",
        "calibration_id": "unconfigured",
        "capture_mono_ns": 2,
        "inference_complete_mono_ns": 2,
        "image_width_px": 1280,
        "image_height_px": 720,
        "identity": IdentityState.STABLE,
        "aggregate_confidence": 0.9,
        "landmarks": (
            Landmark(name="pose_11", normalized_xyz=(0.7, 0.4, 0.0), visibility=1.0, presence=1.0),
            Landmark(name="pose_12", normalized_xyz=(0.3, 0.4, 0.0), visibility=1.0, presence=1.0),
            Landmark(name="face_1", normalized_xyz=(0.5, 0.2, 0.0), visibility=1.0, presence=1.0),
            Landmark(name="face_33", normalized_xyz=(0.4, 0.2, 0.0), visibility=1.0, presence=1.0),
            Landmark(name="face_263", normalized_xyz=(0.6, 0.2, 0.0), visibility=1.0, presence=1.0),
        ),
    }
    values.update(changes)
    return HumanObservation(**values)


def test_neutral_calibration_uses_shoulder_relative_geometry() -> None:
    calibration = create_neutral_calibration(stable_observation())
    assert calibration.shoulder_center_normalized_xyz == (0.5, 0.4, 0.0)
    assert calibration.shoulder_width_normalized == pytest.approx(0.4)


def test_neutral_calibration_fails_closed_on_identity_or_visibility() -> None:
    with pytest.raises(CalibrationError, match="stable"):
        create_neutral_calibration(stable_observation(identity=IdentityState.AMBIGUOUS))
    dim_landmarks = tuple(
        landmark.model_copy(update={"visibility": 0.1}) for landmark in stable_observation().landmarks
    )
    with pytest.raises(CalibrationError, match="high-visibility"):
        create_neutral_calibration(stable_observation(landmarks=dim_landmarks))


def _window_observation(sequence: int, timestamp_ns: int, **changes: object) -> HumanObservation:
    return stable_observation(
        sequence=sequence,
        source_mono_ns=timestamp_ns,
        capture_mono_ns=timestamp_ns,
        inference_complete_mono_ns=timestamp_ns,
        **changes,
    )


def test_windowed_calibration_averages_a_stable_ordered_window() -> None:
    calibration = create_neutral_calibration_window(
        (
            _window_observation(1, 1_000),
            _window_observation(2, 1_020),
            _window_observation(3, 1_040),
        ),
        WINDOW_POLICY,
    )
    assert calibration.observation_sequence == 3
    assert calibration.shoulder_center_normalized_xyz == pytest.approx((0.5, 0.4, 0.0))
    assert calibration.shoulder_width_normalized == pytest.approx(0.4)


@pytest.mark.parametrize(
    "observations, pattern",
    [
        ((_window_observation(1, 1_000),), "too few"),
        (
            (
                _window_observation(1, 1_000),
                _window_observation(2, 1_020, camera_id="different-camera"),
                _window_observation(3, 1_040),
            ),
            "source or camera",
        ),
        (
            (
                _window_observation(1, 1_000),
                _window_observation(1, 1_020),
                _window_observation(3, 1_040),
            ),
            "strictly ordered",
        ),
        (
            (
                _window_observation(1, 1_000),
                _window_observation(2, 1_020),
                _window_observation(3, 1_200),
            ),
            "duration",
        ),
    ],
)
def test_windowed_calibration_rejects_invalid_window(
    observations: tuple[HumanObservation, ...], pattern: str
) -> None:
    with pytest.raises(CalibrationError, match=pattern):
        create_neutral_calibration_window(observations, WINDOW_POLICY)


def test_windowed_calibration_rejects_operator_motion() -> None:
    moved_landmarks = tuple(
        landmark.model_copy(update={"normalized_xyz": (0.9, 0.4, 0.0)})
        if landmark.name == "pose_11"
        else landmark
        for landmark in stable_observation().landmarks
    )
    with pytest.raises(CalibrationError, match="shoulder centre moved"):
        create_neutral_calibration_window(
            (
                _window_observation(1, 1_000),
                _window_observation(2, 1_020),
                _window_observation(3, 1_040, landmarks=moved_landmarks),
            ),
            WINDOW_POLICY,
        )


def test_window_policy_has_no_unsafe_single_frame_mode() -> None:
    with pytest.raises(ValueError, match="at least two"):
        CalibrationWindowPolicy(1, 1, 0.1, 0.1, 0.1)


def _pose_head_landmarks(nose=(0.50, 0.42, 0.0), left=(0.46, 0.40, 0.0), right=(0.54, 0.40, 0.0)):
    from galbot_motion_studio.contracts.human import Landmark

    return tuple(
        Landmark(name=name, normalized_xyz=xyz, visibility=0.95, presence=0.95)
        for name, xyz in (("pose_0", nose), ("pose_2", left), ("pose_5", right))
    )


def test_calibration_records_the_pose_head_basis_when_the_frame_carries_it() -> None:
    observation = stable_observation()
    with_pose = observation.model_copy(
        update={"landmarks": observation.landmarks + _pose_head_landmarks()}
    )
    calibration = create_neutral_calibration(with_pose)
    assert calibration.pose_nose_normalized_xyz is not None
    assert calibration.pose_eye_span_normalized is not None
    # A separate basis, not a copy of the face one: pose_2/5 are eye CENTRES and
    # face_33/263 are outer corners, so the spans must not be assumed equal.
    assert calibration.pose_eye_span_normalized != calibration.eye_span_normalized


def test_windowed_calibration_preserves_the_pose_head_fallback_basis() -> None:
    """The live CLI calls calibrate_window, so this must not drop the fallback."""
    samples = tuple(
        _window_observation(sequence, timestamp).model_copy(
            update={
                "landmarks": _window_observation(sequence, timestamp).landmarks
                + _pose_head_landmarks()
            }
        )
        for sequence, timestamp in ((1, 1_000), (2, 1_020), (3, 1_040))
    )
    calibration = create_neutral_calibration_window(samples, WINDOW_POLICY)
    assert calibration.pose_nose_normalized_xyz == pytest.approx((0.5, 0.42, 0.0))
    assert calibration.pose_eye_center_normalized_xyz == pytest.approx((0.5, 0.4, 0.0))
    assert calibration.pose_eye_span_normalized == pytest.approx(0.08)

    from galbot_motion_studio.retarget.face_pose import estimate_face_pose

    face_less = samples[-1].model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in samples[-1].landmarks
                if not landmark.name.startswith("face_")
            )
        }
    )
    intent = estimate_face_pose(face_less, calibration)
    assert intent.yaw_signal == pytest.approx(0.0)
    assert intent.pitch_signal == pytest.approx(0.0)
    assert intent.basis == "pose"


def test_a_frame_without_pose_head_landmarks_simply_has_no_fallback() -> None:
    calibration = create_neutral_calibration(stable_observation())
    assert calibration.pose_nose_normalized_xyz is None
    assert calibration.pose_eye_center_normalized_xyz is None
    assert calibration.pose_eye_span_normalized is None


def test_a_low_visibility_pose_head_basis_is_not_recorded() -> None:
    from galbot_motion_studio.contracts.human import Landmark

    observation = stable_observation()
    dim = observation.model_copy(
        update={
            "landmarks": observation.landmarks
            + tuple(
                Landmark(
                    name=name, normalized_xyz=(0.5, 0.4, 0.0), visibility=0.2, presence=0.2
                )
                for name in ("pose_0", "pose_2", "pose_5")
            )
        }
    )
    assert create_neutral_calibration(dim).pose_nose_normalized_xyz is None


def test_head_pose_uses_the_pose_basis_when_the_face_mesh_is_gone() -> None:
    """And measures it against the POSE neutral, never the face one."""
    from galbot_motion_studio.retarget.face_pose import estimate_face_pose

    observation = stable_observation()
    with_pose = observation.model_copy(
        update={"landmarks": observation.landmarks + _pose_head_landmarks()}
    )
    calibration = create_neutral_calibration(with_pose)
    # Same frame, face mesh removed, nose swung to the operator's left.
    turned = with_pose.model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in with_pose.landmarks
                if not landmark.name.startswith("face_")
            )
            + _pose_head_landmarks(nose=(0.53, 0.42, 0.0))
        }
    )
    intent = estimate_face_pose(turned, calibration)
    assert intent.yaw_signal != 0.0, "the fallback produced no yaw signal"
    # Neutral in, neutral out: the basis is self-consistent.
    unturned = with_pose.model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in with_pose.landmarks
                if not landmark.name.startswith("face_")
            )
            + _pose_head_landmarks()
        }
    )
    still = estimate_face_pose(unturned, calibration)
    assert abs(still.yaw_signal) < 1e-9 and abs(still.pitch_signal) < 1e-9


def test_head_pose_without_face_or_a_calibrated_fallback_still_fails() -> None:
    from galbot_motion_studio.retarget.face_pose import FacePoseError, estimate_face_pose

    observation = stable_observation()
    calibration = create_neutral_calibration(observation)   # no pose basis recorded
    stripped = observation.model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in observation.landmarks
                if not landmark.name.startswith("face_")
            )
            + _pose_head_landmarks()
        }
    )
    try:
        estimate_face_pose(stripped, calibration)
    except FacePoseError as error:
        assert "fallback" in str(error)
    else:
        raise AssertionError("head pose was estimated with no calibrated basis")
