"""Head retargeting: saturate at the soft limits, reject only teleports and NaN.

The mapper used to refuse any target outside the soft band. Every refusal is a
frozen head, so the operator's own session moved 15.8 deg of yaw against ~164 deg
of available travel: turning further stopped the robot instead of driving it to
its limit. These tests pin the replacement contract -- clamp per axis, keep
producing a target -- and pin the clamp inside the hard joint limits.
"""

from math import degrees, inf, nan

import pytest

from galbot_motion_studio.model.joint_map import JointLimit, parse_revolute_joint_limits
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST
from galbot_motion_studio.retarget.head import (
    HeadMapRejection,
    HeadMappingPolicy,
    HeadPoseIntent,
    HeadRetargeter,
    HeadTarget,
)


SOFT_MARGIN_RAD = 0.09
DEADBAND_RAD = 0.02
# The canonical HOME the pipeline retargets around, so "usable travel" below is
# the travel the operator actually has rather than travel from an ideal zero.
HOME_YAW_RAD = -0.0006
HOME_PITCH_RAD = -0.0002
# Signals far past anything a face can produce, used to force saturation.
EXTREME_SIGNALS = (-50.0, -5.0, -1.0, 1.0, 5.0, 50.0)


def head_limits() -> tuple[JointLimit, JointLimit]:
    limits = parse_revolute_joint_limits(CANONICAL_MANIFEST.fixed_urdf)
    return limits["head_joint1"], limits["head_joint2"]


def soft_band(limit: JointLimit) -> tuple[float, float]:
    return limit.lower_rad + SOFT_MARGIN_RAD, limit.upper_rad - SOFT_MARGIN_RAD


def retargeter(
    *, neutral: HeadTarget = HeadTarget(yaw_rad=0.0, pitch_rad=0.0)
) -> HeadRetargeter:
    yaw_limit, pitch_limit = head_limits()
    return HeadRetargeter(
        HeadMappingPolicy(
            yaw_limit=yaw_limit,
            pitch_limit=pitch_limit,
            yaw_gain=1.0,
            pitch_up_gain=1.0,
            pitch_down_gain=0.5,
            deadband_rad=DEADBAND_RAD,
            soft_margin_rad=SOFT_MARGIN_RAD,
            max_rate_rad_s=1.0,
        ),
        neutral=neutral,
    )


def test_head_mapping_uses_asymmetric_pitch_gain() -> None:
    mapper = retargeter()
    up = mapper.map(HeadPoseIntent(yaw_signal=0.1, pitch_signal=0.1), previous=None, elapsed_s=None)
    down = mapper.map(HeadPoseIntent(yaw_signal=0.1, pitch_signal=-0.1), previous=None, elapsed_s=None)
    assert up.target == HeadTarget(yaw_rad=0.08, pitch_rad=0.08)
    assert down.target == HeadTarget(yaw_rad=0.08, pitch_rad=-0.04)
    assert (up.saturated, down.saturated) == (False, False)


def test_head_mapping_deadband_still_holds_the_neutral_pose() -> None:
    """Small signals are noise, and noise must not move the head at all."""
    mapper = retargeter()
    for yaw, pitch in ((0.01, -0.01), (0.0, 0.0), (DEADBAND_RAD, -DEADBAND_RAD)):
        still = mapper.map(HeadPoseIntent(yaw_signal=yaw, pitch_signal=pitch), previous=None, elapsed_s=None)
        assert still.target == HeadTarget(yaw_rad=0.0, pitch_rad=0.0), (yaw, pitch)
        assert still.rejection is None
        assert still.saturated is False


def test_deadband_stays_continuous_at_its_edge() -> None:
    """Just outside the deadband is a hair of motion, not a step to the raw signal."""
    mapper = retargeter()
    result = mapper.map(
        HeadPoseIntent(yaw_signal=DEADBAND_RAD + 0.001, pitch_signal=0.0),
        previous=None,
        elapsed_s=None,
    )
    assert result.target is not None
    assert result.target.yaw_rad == pytest.approx(0.001)


def test_head_mapping_saturates_pitch_at_both_ends() -> None:
    mapper = retargeter()
    _, pitch_limit = head_limits()
    lower, upper = soft_band(pitch_limit)

    top = mapper.map(HeadPoseIntent(yaw_signal=0.0, pitch_signal=5.0), previous=None, elapsed_s=None)
    bottom = mapper.map(HeadPoseIntent(yaw_signal=0.0, pitch_signal=-5.0), previous=None, elapsed_s=None)

    assert top.rejection is None and bottom.rejection is None
    assert top.target is not None and bottom.target is not None
    assert top.target.pitch_rad == pytest.approx(upper)
    assert bottom.target.pitch_rad == pytest.approx(lower)
    assert top.saturated is True and bottom.saturated is True
    # Pitch saturating must not drag yaw off its own demand.
    assert top.target.yaw_rad == pytest.approx(0.0)


def test_head_mapping_saturates_yaw_at_both_ends() -> None:
    mapper = retargeter()
    yaw_limit, _ = head_limits()
    lower, upper = soft_band(yaw_limit)

    right = mapper.map(HeadPoseIntent(yaw_signal=10.0, pitch_signal=0.0), previous=None, elapsed_s=None)
    left = mapper.map(HeadPoseIntent(yaw_signal=-10.0, pitch_signal=0.0), previous=None, elapsed_s=None)

    assert right.rejection is None and left.rejection is None
    assert right.target is not None and left.target is not None
    assert right.target.yaw_rad == pytest.approx(upper)
    assert left.target.yaw_rad == pytest.approx(lower)
    assert right.saturated is True and left.saturated is True
    assert right.target.pitch_rad == pytest.approx(0.0)


@pytest.mark.parametrize("yaw_signal", EXTREME_SIGNALS)
@pytest.mark.parametrize("pitch_signal", EXTREME_SIGNALS)
def test_targets_stay_inside_the_hard_joint_limits_however_extreme_the_demand(
    yaw_signal: float, pitch_signal: float
) -> None:
    """Clamping is only safe if the clamp lands short of the real joint stop."""
    mapper = retargeter(neutral=HeadTarget(yaw_rad=HOME_YAW_RAD, pitch_rad=HOME_PITCH_RAD))
    yaw_limit, pitch_limit = head_limits()
    result = mapper.map(
        HeadPoseIntent(yaw_signal=yaw_signal, pitch_signal=pitch_signal),
        previous=None,
        elapsed_s=None,
    )
    assert result.target is not None
    for value, limit in ((result.target.yaw_rad, yaw_limit), (result.target.pitch_rad, pitch_limit)):
        assert limit.lower_rad < value < limit.upper_rad
        # And with the whole soft margin still unspent on both sides.
        assert value >= limit.lower_rad + SOFT_MARGIN_RAD
        assert value <= limit.upper_rad - SOFT_MARGIN_RAD


@pytest.mark.parametrize("yaw_signal", EXTREME_SIGNALS)
@pytest.mark.parametrize("pitch_signal", EXTREME_SIGNALS)
def test_soft_limit_demand_is_never_a_rejection(yaw_signal: float, pitch_signal: float) -> None:
    """The regression: an out-of-range look used to freeze the head instead of pinning it."""
    result = retargeter().map(
        HeadPoseIntent(yaw_signal=yaw_signal, pitch_signal=pitch_signal),
        previous=None,
        elapsed_s=None,
    )
    assert result.rejection is None
    assert result.target is not None


def test_rate_limit_is_applied_to_the_saturated_target_not_the_raw_demand() -> None:
    """Sitting at the limit is a still head, so it must not read as a teleport."""
    mapper = retargeter()
    _, pitch_limit = head_limits()
    _, upper = soft_band(pitch_limit)
    previous = HeadTarget(yaw_rad=0.0, pitch_rad=upper - 0.001)

    result = mapper.map(
        # Raw demand is ~4.98 rad, far past any allowed step; the clamped target
        # is 0.001 rad away from ``previous``.
        HeadPoseIntent(yaw_signal=0.0, pitch_signal=5.0),
        previous=previous,
        elapsed_s=0.05,
    )

    assert result.rejection is None
    assert result.target is not None
    assert result.target.pitch_rad == pytest.approx(upper)
    assert result.saturated is True


def test_head_mapping_rejects_a_teleport_instead_of_clipping() -> None:
    mapper = retargeter()
    result = mapper.map(
        HeadPoseIntent(yaw_signal=0.2, pitch_signal=0.0),
        previous=HeadTarget(yaw_rad=0.0, pitch_rad=0.0),
        elapsed_s=0.05,
    )
    assert result.rejection is HeadMapRejection.RATE_EXCEEDED
    assert result.target is None


@pytest.mark.parametrize(
    ("yaw_signal", "pitch_signal", "elapsed_s"),
    [
        (nan, 0.0, None),
        (0.0, nan, None),
        (inf, 0.0, None),
        (0.0, -inf, None),
        (0.0, 0.0, nan),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -0.01),
    ],
)
def test_nonfinite_or_nonpositive_input_is_still_rejected(
    yaw_signal: float, pitch_signal: float, elapsed_s: float | None
) -> None:
    """Unchanged: there is no defined value to clamp a NaN toward."""
    result = retargeter().map(
        HeadPoseIntent(yaw_signal=yaw_signal, pitch_signal=pitch_signal),
        previous=HeadTarget(yaw_rad=0.0, pitch_rad=0.0),
        elapsed_s=elapsed_s,
    )
    assert result.rejection is HeadMapRejection.NONFINITE
    assert result.target is None


def test_default_gains_are_sized_to_the_measured_usable_head_range() -> None:
    """The operator fix: 0.6/0.35/0.25 tracked 15.8 deg of yaw out of ~164 available."""
    yaw_limit, pitch_limit = head_limits()
    policy = HeadMappingPolicy(yaw_limit=yaw_limit, pitch_limit=pitch_limit)
    assert (policy.yaw_gain, policy.pitch_up_gain, policy.pitch_down_gain) == (1.4, 1.2, 0.5)
    # Asymmetric because the joint is: pitch's positive side has 3x the travel.
    assert policy.pitch_up_gain > policy.pitch_down_gain


def test_measured_usable_travel_from_home_matches_the_documented_envelope() -> None:
    """The numbers the gains were sized against, pinned so a model swap surfaces here."""
    yaw_limit, pitch_limit = head_limits()
    yaw_lower, yaw_upper = soft_band(yaw_limit)
    pitch_lower, pitch_upper = soft_band(pitch_limit)
    assert round(degrees(yaw_lower - HOME_YAW_RAD), 1) == -81.9
    assert round(degrees(yaw_upper - HOME_YAW_RAD), 1) == 82.0
    assert round(degrees(pitch_lower - HOME_PITCH_RAD), 1) == -7.1
    assert round(degrees(pitch_upper - HOME_PITCH_RAD), 1) == 23.1
