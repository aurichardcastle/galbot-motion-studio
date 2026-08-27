import pytest

from galbot_motion_studio.contracts.human import IdentityState
from galbot_motion_studio.retarget.face_pose import FacePoseError, estimate_face_pose
from galbot_motion_studio.vision.calibration import create_neutral_calibration

from test_calibration import stable_observation


def test_face_pose_is_neutral_relative_and_eye_span_normalized() -> None:
    calibration = create_neutral_calibration(stable_observation())
    neutral = estimate_face_pose(stable_observation(), calibration)
    assert neutral.yaw_signal == pytest.approx(0.0)
    assert neutral.pitch_signal == pytest.approx(0.0)
    assert neutral.basis == "face_mesh"
    shifted = tuple(
        landmark.model_copy(update={"normalized_xyz": (0.55, 0.25, 0.0)})
        if landmark.name == "face_1"
        else landmark
        for landmark in stable_observation().landmarks
    )
    pose = estimate_face_pose(stable_observation(landmarks=shifted), calibration)
    # The operator view is mirrored like a selfie camera, so a positive
    # image-space nose shift must produce the opposite robot-yaw signal.
    assert pose.yaw_signal == pytest.approx(-0.25)
    assert pose.pitch_signal == pytest.approx(0.25)


def test_face_pose_fails_on_unstable_identity_or_clock_switch() -> None:
    calibration = create_neutral_calibration(stable_observation())
    with pytest.raises(FacePoseError, match="stable"):
        estimate_face_pose(stable_observation(identity=IdentityState.AMBIGUOUS), calibration)
    with pytest.raises(FacePoseError, match="clocks differ"):
        estimate_face_pose(stable_observation(source_clock_id="other-clock"), calibration)
