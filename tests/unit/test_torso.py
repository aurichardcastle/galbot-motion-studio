from math import degrees, isfinite, nan, radians

import pytest

from galbot_motion_studio.model.joint_map import parse_revolute_joint_limits
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST
from galbot_motion_studio.retarget.torso import (
    TorsoIntent,
    TorsoMapRejection,
    TorsoMappingPolicy,
    TorsoRetargeter,
    TorsoTarget,
    shoulder_yaw_signal,
    stable_yaw_reference,
    wrapped_yaw_difference,
)

LEG_JOINT4 = parse_revolute_joint_limits(CANONICAL_MANIFEST.fixed_urdf)["leg_joint4"]
#: HOME leg_joint4 from the 2026-08-07 hardware snapshot.
NEUTRAL = TorsoTarget(yaw_rad=-0.0006)


def _retargeter(**overrides) -> TorsoRetargeter:
    policy = TorsoMappingPolicy(yaw_limit=LEG_JOINT4, **overrides)
    return TorsoRetargeter(policy, neutral=NEUTRAL)


def test_a_motionless_operator_does_not_dither_the_torso() -> None:
    # The heaviest joint on the robot must not chase resting landmark noise.
    # Measured p50 frame-to-frame yaw jitter on the 2026-08-22 trial was 1.07 deg.
    retargeter = _retargeter()
    for jitter_deg in (0.0, 0.4, 1.07, 2.8):
        result = retargeter.map(TorsoIntent(radians(jitter_deg)))
        assert result.target == NEUTRAL, f"{jitter_deg} deg moved the torso"


def test_the_deadband_edge_is_continuous() -> None:
    # A deadband that returns the raw value past its threshold steps by the
    # threshold, and the governor's acceleration limiter sees that as a jump.
    retargeter = _retargeter()
    band = retargeter.policy.deadband_rad
    just_inside = retargeter.map(TorsoIntent(band - 1e-9)).target
    just_outside = retargeter.map(TorsoIntent(band + 1e-9)).target
    assert abs(just_outside.yaw_rad - just_inside.yaw_rad) < 1e-6


def test_unity_gain_tracks_the_operator_one_for_one() -> None:
    retargeter = _retargeter()
    band = retargeter.policy.deadband_rad
    for signal in (0.2, 0.5, 1.0):
        target = retargeter.map(TorsoIntent(signal)).target
        assert abs(target.yaw_rad - (NEUTRAL.yaw_rad + signal - band)) < 1e-9


def test_the_measured_operator_range_never_saturates() -> None:
    """Ties the unity gain to the trial it was derived from.

    On 2026-08-22 the operator's torso yaw spanned -22.5..+66.6 deg at p01/p99.
    If a future gain change makes that range hit the soft limit, the torso would
    pin during ordinary use and this test is the thing that says so.
    """
    retargeter = _retargeter()
    for measured_deg in (-22.5, -11.2, 2.6, 38.0, 66.6):
        result = retargeter.map(TorsoIntent(radians(measured_deg)))
        assert not result.saturated, f"{measured_deg} deg saturated at unity gain"


def test_demand_past_the_soft_limit_saturates_instead_of_freezing() -> None:
    retargeter = _retargeter()
    margin = retargeter.policy.soft_margin_rad
    for signal in (2.0, -2.0, 10.0):
        result = retargeter.map(TorsoIntent(signal))
        assert result.target is not None, "an extreme turn froze the torso"
        assert result.saturated
        assert LEG_JOINT4.lower_rad + margin - 1e-12 <= result.target.yaw_rad
        assert result.target.yaw_rad <= LEG_JOINT4.upper_rad - margin + 1e-12
        # Never merely inside the soft band -- inside the hard limit too.
        assert LEG_JOINT4.lower_rad < result.target.yaw_rad < LEG_JOINT4.upper_rad


def test_staying_pinned_at_the_limit_is_not_a_teleport() -> None:
    # Rate control belongs to the trajectory governor, which sees the emitted
    # joint pose.  The pure mapper must therefore keep returning the same soft
    # limit instead of turning a still demand into a fake discontinuity.
    retargeter = _retargeter()
    first = retargeter.map(TorsoIntent(5.0))
    assert first.saturated
    second = retargeter.map(TorsoIntent(9.0))
    assert second.rejection is None
    assert second.target == first.target


def test_nonfinite_inputs_are_rejected() -> None:
    retargeter = _retargeter()
    assert retargeter.map(TorsoIntent(nan)).rejection is TorsoMapRejection.NONFINITE
    assert retargeter.map(TorsoIntent(float("inf"))).rejection is TorsoMapRejection.NONFINITE


def test_policy_validation() -> None:
    for bad in ({"yaw_gain": -1.0}, {"deadband_rad": nan}, {"max_rate_rad_s": -0.1}):
        try:
            TorsoMappingPolicy(yaw_limit=LEG_JOINT4, **bad)
        except ValueError as error:
            assert "finite and non-negative" in str(error)
        else:
            raise AssertionError(f"policy accepted {bad}")
    try:
        # leg_joint5's whole range is 0.329 rad; a 0.2 rad margin consumes it.
        TorsoMappingPolicy(
            yaw_limit=parse_revolute_joint_limits(CANONICAL_MANIFEST.fixed_urdf)[
                "leg_joint5"
            ],
            soft_margin_rad=0.2,
        )
    except ValueError as error:
        assert "consumes the entire joint range" in str(error)
    else:
        raise AssertionError("a margin larger than the joint range was accepted")
    with pytest.raises(ValueError, match="observation gap must be positive"):
        TorsoMappingPolicy(yaw_limit=LEG_JOINT4, max_observation_gap_ns=0)


def test_shoulder_yaw_signal_has_the_right_sign_and_magnitude() -> None:
    # Facing the camera: shoulders level in depth, zero yaw.
    assert abs(shoulder_yaw_signal((0.16, -0.42, 0.0), (-0.16, -0.42, 0.0))) < 1e-12
    # Left shoulder forward (more negative z, toward the camera) must come back
    # with the opposite sign to left shoulder back.
    left_forward = shoulder_yaw_signal((0.16, -0.42, -0.10), (-0.16, -0.42, 0.0))
    left_back = shoulder_yaw_signal((0.16, -0.42, 0.10), (-0.16, -0.42, 0.0))
    assert left_forward is not None and left_back is not None
    assert left_forward < 0 < left_back
    assert abs(left_forward + left_back) < 1e-12
    # A 45 deg turn puts the depth difference equal to the lateral extent.
    assert abs(degrees(shoulder_yaw_signal((0.16, 0.0, 0.32), (-0.16, 0.0, 0.0))) - 45.0) < 1e-9


def test_shoulder_yaw_signal_refuses_unusable_lines() -> None:
    # A collapsed shoulder line is where the 144 deg single-frame jumps came from
    # on the trial; reporting its angle would be reporting noise as a command.
    assert shoulder_yaw_signal((0.05, 0.0, 0.0), (0.0, 0.0, 0.0)) is None
    assert shoulder_yaw_signal((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) is None
    assert shoulder_yaw_signal((nan, 0.0, 0.0), (0.0, 0.0, 0.0)) is None
    assert shoulder_yaw_signal((0.16, 0.0, 0.0), (float("inf"), 0.0, 0.0)) is None
    # Exactly at the threshold is usable; the guard is on being shorter than it.
    assert shoulder_yaw_signal((0.12, 0.0, 0.0), (0.0, 0.0, 0.0)) is not None
    # A long vertical line used to pass the 3-D-width guard even though its yaw
    # denominator was nearly zero.  Its angle changes by 45 degrees for 1 mm of
    # depth noise, so the relevant horizontal projection must be rejected.
    assert shoulder_yaw_signal((0.001, 0.13, 0.0), (0.0, 0.0, 0.0)) is None
    assert shoulder_yaw_signal((0.001, 0.13, 0.001), (0.0, 0.0, 0.0)) is None


def test_calibration_reference_uses_wrapped_circular_geometry() -> None:
    reference = stable_yaw_reference(
        (radians(179.0), radians(-179.0)),
        max_absolute_yaw_rad=radians(180.0),
        max_spread_rad=radians(3.0),
    )
    assert reference is not None
    assert abs(abs(reference) - radians(180.0)) < 1e-12
    delta = wrapped_yaw_difference(radians(-179.0), radians(179.0))
    assert delta == pytest.approx(radians(2.0))


def test_calibration_reference_rejects_moving_or_off_axis_neutral() -> None:
    assert stable_yaw_reference(
        (radians(-8.0), radians(8.0)),
        max_absolute_yaw_rad=radians(30.0),
        max_spread_rad=radians(5.0),
    ) is None
    assert stable_yaw_reference(
        (radians(31.0),),
        max_absolute_yaw_rad=radians(30.0),
        max_spread_rad=radians(5.0),
    ) is None


def test_mapping_is_deterministic() -> None:
    a, b = _retargeter(), _retargeter()
    for signal in (0.0, 0.3, -0.7, 2.5):
        first = a.map(TorsoIntent(signal))
        second = b.map(TorsoIntent(signal))
        assert first == second
        assert isfinite(first.target.yaw_rad)
