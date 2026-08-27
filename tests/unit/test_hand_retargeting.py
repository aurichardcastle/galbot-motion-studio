import pytest

from galbot_motion_studio.retarget.hand import (
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    estimate_hand_openness,
    openness_to_gripper_rad,
)
from galbot_motion_studio.synthetic import synthetic_observation


def test_open_and_closed_hands_map_to_parallel_gripper_endpoints() -> None:
    open_left = synthetic_observation(
        1, timestamp_ns=1_000_000_000, motion_fraction=0.0
    )
    closed_left = synthetic_observation(
        2, timestamp_ns=1_033_333_333, motion_fraction=1.0
    )
    open_score = estimate_hand_openness(open_left, "left")
    closed_score = estimate_hand_openness(closed_left, "left")
    assert open_score == pytest.approx(1.0)
    assert closed_score == pytest.approx(0.0)
    assert openness_to_gripper_rad(open_score) == GRIPPER_OPEN_RAD
    assert openness_to_gripper_rad(closed_score) == GRIPPER_CLOSED_RAD


def test_missing_hand_holds_last_gripper_instead_of_guessing() -> None:
    observation = synthetic_observation(
        1, timestamp_ns=1_000_000_000, motion_fraction=0.5
    )
    without_right = observation.model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in observation.landmarks
                if not landmark.name.startswith("right_hand_")
            )
        }
    )
    assert estimate_hand_openness(without_right, "right") is None


def test_hand_side_and_score_are_validated() -> None:
    observation = synthetic_observation(
        1, timestamp_ns=1_000_000_000, motion_fraction=0.5
    )
    with pytest.raises(ValueError, match="side"):
        estimate_hand_openness(observation, "middle")
    with pytest.raises(ValueError, match="finite"):
        openness_to_gripper_rad(float("nan"))
