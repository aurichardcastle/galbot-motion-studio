"""The swivel mapper: the operator's psi, and the live swivel-priority mode.

The properties that matter here are the ones the design rests on -- psi needs no
hips, psi survives a camera yaw, and selecting the new mode cannot disturb the
established one.
"""

from math import radians

import numpy as np
import pytest

from galbot_motion_studio.contracts.human import Landmark
from galbot_motion_studio.pipeline import mapping_config_for, mapping_hash_for
from galbot_motion_studio.retarget.left_arm import (
    LeftArmPolicy,
    _human_swivel_angle,
)

from test_calibration import stable_observation


def _landmark(name: str, world: tuple[float, float, float], confidence: float = 1.0):
    return Landmark(
        name=name,
        normalized_xyz=(0.5, 0.5, 0.0),
        world_xyz_m=world,
        visibility=confidence,
        presence=confidence,
    )


def _arm_observation(confidence: float = 1.0, *, rotation: np.ndarray | None = None):
    """A seated operator with the left elbow tucked below the shoulder.

    Deliberately carries NO hip landmarks: the whole claim is that psi does not
    need them.
    """
    points = {
        "pose_11": (0.20, 0.00, 0.00),   # left shoulder
        "pose_12": (-0.20, 0.00, 0.00),  # right shoulder
        "pose_13": (0.24, 0.28, 0.02),   # left elbow, hanging (MP world +Y is down)
        "pose_15": (0.30, 0.52, 0.10),   # left wrist, forward of the elbow
        "pose_14": (-0.24, 0.28, 0.02),
        "pose_16": (-0.30, 0.52, 0.10),
    }
    if rotation is not None:
        points = {
            name: tuple(float(value) for value in rotation @ np.asarray(world))
            for name, world in points.items()
        }
    base = stable_observation()
    keep = tuple(
        landmark
        for landmark in base.landmarks
        if not landmark.name.startswith("pose_")
    )
    return base.model_copy(
        update={
            "landmarks": keep
            + tuple(_landmark(name, world, confidence) for name, world in points.items())
        }
    )


def test_operator_swivel_is_measured_without_any_hip_landmark() -> None:
    observation = _arm_observation()
    assert not any(
        landmark.name in ("pose_23", "pose_24") for landmark in observation.landmarks
    )

    measured = _human_swivel_angle(observation, side="left", min_confidence=0.35)

    assert measured is not None
    angle_rad, confidence = measured
    assert np.isfinite(angle_rad)
    assert confidence > 0.0


def test_operator_swivel_survives_a_camera_yaw() -> None:
    """psi is a dihedral angle, so rotating the whole operator must not change it.

    This is the property that lets the shoulder line stand in for a torso basis:
    a shoulder-only DIRECTION would turn a camera yaw into a robot command, but
    an angle measured between two of the operator's own segments cannot.
    """
    yaw = radians(35.0)
    rotation = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )

    straight = _human_swivel_angle(_arm_observation(), side="left", min_confidence=0.35)
    yawed = _human_swivel_angle(
        _arm_observation(rotation=rotation), side="left", min_confidence=0.35
    )

    assert straight is not None and yawed is not None
    assert yawed[0] == pytest.approx(straight[0], abs=1e-9)


def test_operator_swivel_is_withheld_when_the_arm_is_not_confidently_seen() -> None:
    assert _human_swivel_angle(
        _arm_observation(confidence=0.10), side="left", min_confidence=0.35
    ) is None


def test_each_arm_measures_psi_from_its_own_outward_direction() -> None:
    """Sharing one reference puts the right arm on the +/-180 branch cut.

    Measured on the retained capture, a shared reference gave the right arm a
    358.1 deg range -- a full wrap mid-gesture -- against 95.1 deg per-side.
    """
    observation = _arm_observation()

    left = _human_swivel_angle(observation, side="left", min_confidence=0.35)
    right = _human_swivel_angle(observation, side="right", min_confidence=0.35)

    assert left is not None and right is not None
    # The pose is mirror-symmetric, so with each arm measuring from its OWN
    # outward direction the two psi must agree in magnitude.
    assert abs(left[0]) == pytest.approx(abs(right[0]), abs=1e-6)


def test_swivel_priority_is_a_distinct_attested_mapping() -> None:
    """A recording must never be able to claim the wrong mapper produced it."""
    hashes = {
        mapping_hash_for(mode)
        for mode in ("wrist-primary", "direction-vector", "swivel-priority")
    }

    assert len(hashes) == 3
    config = mapping_config_for("swivel-priority")
    assert config["arms"]["redundancy"] == "swivel-angle-about-shoulder-wrist-axis"
    # The wrist target is the baseline's. Only the redundancy resolution differs,
    # and the attestation has to say so.
    assert config["arms"]["wrist_target"] == "absolute-shoulder-anchored"


def test_unknown_mapping_modes_are_still_rejected() -> None:
    with pytest.raises(ValueError):
        LeftArmPolicy(mapping_mode="elbow-point")


def test_swivel_gain_must_be_finite_and_non_zero() -> None:
    """Zero would silently disable the mode while reporting success."""
    with pytest.raises(ValueError):
        LeftArmPolicy(mapping_mode="swivel-priority", swivel_gain=0.0)
    with pytest.raises(ValueError):
        LeftArmPolicy(mapping_mode="swivel-priority", swivel_gain=float("nan"))
    # Negative is legal: the two psi conventions could have run opposite ways.
    # Measured on the retained capture they do not, so the default is +1.0.
    assert LeftArmPolicy(mapping_mode="swivel-priority", swivel_gain=-1.0).swivel_gain == -1.0
