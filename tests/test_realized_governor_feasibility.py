"""The governor's REALIZED step must be clearance-safe, not just the straight target.

The guarded controller validates the straight ``[current, target]`` joint-space
segment. But the trajectory governor rate-limits each joint independently, so the
pose it actually realizes lands *off* that segment and can dip below the clearance
floor even when the straight line is clear. The independent supervisor checks that
realized swept path, so it would latch a HOLD -- the "stuck at flex, can't come
down" failure seen live, and a synthetic-demo HOLD at frame 90.

``MotionStudioPipeline._feasible_governor_command`` closes that gap: it predicts
the governor's realized next pose (arm/head and the separately-governed grippers)
and eases the command back toward the current pose until that realized pose is
clear. These tests pin the fix on the REAL collision model, and guard against the
false pass where an all-ALLOW demo is achieved by simply freezing the arm.
"""

from __future__ import annotations

import pytest

from math import pi, radians, sin

from galbot_motion_studio.cli import (
    DEMO_FRAMES,
    DEMO_FRAME_NS,
    DEMO_WAVE_AMPLITUDE,
    DEMO_WAVE_CYCLES,
)
from galbot_motion_studio.pipeline import MotionStudioPipeline
from galbot_motion_studio.synthetic import synthetic_observation
from galbot_motion_studio.safety.supervisor import SafetyOutcome


def _demo_observation(sequence: int):
    timestamp = 1_000_000_000 + sequence * DEMO_FRAME_NS
    motion_fraction = 0.5 + DEMO_WAVE_AMPLITUDE * sin(
        2.0 * pi * DEMO_WAVE_CYCLES * (sequence / DEMO_FRAMES)
    )
    return synthetic_observation(
        sequence, timestamp_ns=timestamp, motion_fraction=motion_fraction
    )


def _calibrated_pipeline() -> MotionStudioPipeline:
    pipeline = MotionStudioPipeline(source_clock_id="synthetic-clock")
    pipeline.calibrate(
        synthetic_observation(0, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    )
    return pipeline


def test_synthetic_demo_is_all_allow_and_the_arm_actually_tracks() -> None:
    """Two failure modes at once, because each masks the other.

    1. The frame-90 HOLD: without realized-step validation the governor commands a
       pose whose swept path dips below the floor and the supervisor HOLDs.
    2. The FALSE PASS: an all-ALLOW demo is trivially satisfiable by freezing the
       arm (the committed straight-only controller does exactly this on the flex,
       backing off to alpha 0), which IS the stuck-at-flex bug. So passing requires
       the arm to also track a large excursion, not merely avoid HOLDs.
    """
    pipeline = _calibrated_pipeline()
    holds: list[tuple[int, tuple[str, ...]]] = []
    first_arm: dict[str, float] | None = None
    peak_displacement = 0.0

    for frame_index in range(DEMO_FRAMES):
        sequence = frame_index + 1
        observation = _demo_observation(sequence)
        result = pipeline.process_fail_closed(
            observation, now_mono_ns=observation.capture_mono_ns
        )
        if result.decision.outcome is not SafetyOutcome.ALLOW:
            holds.append((sequence, result.decision.reasons))
        arm = {
            joint.name: joint.position_rad
            for joint in result.target.joints
            if "arm_joint" in joint.name
        }
        if first_arm is None:
            first_arm = dict(arm)
        peak_displacement = max(
            peak_displacement,
            max(abs(arm[name] - first_arm[name]) for name in arm),
        )

    assert holds == [], f"supervisor HOLDs in a synthetic demo: {holds}"
    # Not frozen: the committed straight-only controller peaks near 28 deg here
    # (freezing through the flex); the realized-validated controller tracks ~128 deg.
    # 60 deg cleanly separates "tracking" from "froze at the wall".
    assert peak_displacement > radians(60), (
        f"arm barely moved ({peak_displacement:.3f} rad): an all-ALLOW demo that "
        "froze the arm is the stuck-at-flex bug, not a pass"
    )


def test_arm_warm_start_is_the_last_emitted_governor_pose() -> None:
    """IK must not lead the rate- and clearance-limited pose it is controlling."""
    pipeline = _calibrated_pipeline()
    assert pipeline._previous_left_arm == {
        name: pipeline._governed_joints[name] for name in pipeline.left_arm.joint_names
    }
    assert pipeline._previous_right_arm == {
        name: pipeline._governed_joints[name] for name in pipeline.right_arm.joint_names
    }
    observation = _demo_observation(1)
    result = pipeline.process_fail_closed(
        observation, now_mono_ns=observation.capture_mono_ns
    )
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert result.target is not None
    emitted = {
        joint.name: joint.position_rad
        for joint in result.target.joints
        if "arm_joint" in joint.name
    }
    raw_left = dict(result.target.left_ik_joints_rad or ())
    raw_right = dict(result.target.right_ik_joints_rad or ())
    assert raw_left and raw_right
    # This frame is intentionally rate-limited, so the regression would be
    # invisible if the raw solve happened to equal the emitted pose.
    assert any(
        abs(raw_left[name] - emitted[name]) > 1e-6 for name in raw_left
    ) or any(abs(raw_right[name] - emitted[name]) > 1e-6 for name in raw_right)
    assert pipeline._previous_left_arm == {
        name: emitted[name] for name in pipeline.left_arm.joint_names
    }
    assert pipeline._previous_right_arm == {
        name: emitted[name] for name in pipeline.right_arm.joint_names
    }


def _run_realized_step_scenario() -> tuple[bool, bool]:
    """Run the demo wave and report (trap seen, invariant held).

    Split out of a single test because the two answers have very different
    standing. "Did this scenario still exercise the trap?" is a question about the
    SCENARIO and is allowed to go stale. "Was every issued command's realized step
    clearance-safe?" is a safety invariant and is never allowed to go stale -- and
    it was: both assertions lived in one `xfail(strict=False)` test, the trap
    assertion came first, so the invariant line was never reached and ANY failure
    of it, including a real regression, reported as `xfailed`. Verified by forcing
    `invariant_ok = False`: the suite stayed green.
    """
    # Twice the nominal 30 Hz spacing, which is not a contrivance to make the
    # trap appear: the measured p50 source gap on the 2026-08-22 live trial was
    # 84.4 ms, so 33.3 ms is the optimistic case and 66.7 ms is nearer what the
    # pipeline actually sees. The trap needs a governor step large enough for the
    # realized pose to diverge from the straight one, and a sparse frame is
    # exactly when that happens -- which is the point of the fix.
    #
    # It stopped occurring at 33.3 ms when the swivel solver was repaired
    # (2026-08-22): a converging solver returns targets much nearer the current
    # pose, so the realized excursion shrank. The mechanism is unchanged -- the
    # realized path still runs up to 4.08 mm below the straight one on this demo
    # -- it simply no longer crosses the floor at that frame rate.
    frame_ns = DEMO_FRAME_NS * 2
    pipeline = _calibrated_pipeline()
    checker = pipeline.checker
    original = pipeline._feasible_governor_command
    saw_trap = False
    invariant_ok = True

    def spy(guarded_target, *, timestamp_ns):
        nonlocal saw_trap, invariant_ok
        current = dict(pipeline.supervisor.current_pose)
        gripper = pipeline.gripper_trajectory.preview(
            pipeline._gripper_desired, timestamp_ns=timestamp_ns
        )

        def realized_path_ok(target):
            realized = pipeline.trajectory.preview(target, timestamp_ns=timestamp_ns)
            end = {**current, **realized, **gripper}
            return bool(
                checker.check_path(current, end, steps=2, max_joint_step=0.01).ok
            )

        # The deceptive condition: the guarded target's straight path is clear (the
        # guarded controller only returns clearance-clear targets) but its realized
        # governor step is not.
        straight_clear = bool(
            checker.check_path(
                current, {**current, **guarded_target, **gripper},
                steps=2, max_joint_step=0.01,
            ).ok
        )
        if straight_clear and not realized_path_ok(guarded_target):
            saw_trap = True

        command = original(guarded_target, timestamp_ns=timestamp_ns)
        # The command the fix issues must always have a clearance-safe realized step
        # (unless earlier momentum already commits a breach, which never happens in
        # this demo -- the demo test above asserts zero HOLDs).
        if not realized_path_ok(command):
            invariant_ok = False
        return command

    pipeline._feasible_governor_command = spy
    for frame_index in range(DEMO_FRAMES):
        sequence = frame_index + 1
        motion_fraction = 0.5 + DEMO_WAVE_AMPLITUDE * sin(
            2.0 * pi * DEMO_WAVE_CYCLES * (sequence / DEMO_FRAMES)
        )
        observation = synthetic_observation(
            sequence,
            timestamp_ns=1_000_000_000 + sequence * frame_ns,
            motion_fraction=motion_fraction,
        )
        pipeline.process_fail_closed(observation, now_mono_ns=observation.capture_mono_ns)

    return saw_trap, invariant_ok


def test_every_issued_command_has_a_clearance_safe_realized_step() -> None:
    """The safety invariant. No marker, ever: this one may not go quiet."""
    _saw_trap, invariant_ok = _run_realized_step_scenario()
    assert invariant_ok, "an issued command had a clearance-unsafe realized step"


@pytest.mark.xfail(
    reason=(
        "2026-08-26: this SCENARIO no longer reproduces the realized-step trap. "
        "The look-ahead horizon was sized past the governor's braking curve "
        "(pipeline._lookahead_for), so the guarded controller now clearance-checks "
        "the path the arm can actually traverse and pre-empts the divergence this "
        "test was built to catch -- tried DEMO_FRAME_NS x2/x3/x4/x5, trap fired on "
        "no frame. NOT a licence to drop the realized-step check: that invariant is "
        "asserted by test_every_issued_command_has_a_clearance_safe_realized_step, "
        "which carries no marker. Find a scenario that exercises the trap again and "
        "remove this one."
    ),
    strict=False,
)
def test_the_realized_step_trap_still_reproduces() -> None:
    """Is the fix still demonstrably load-bearing on this scenario?"""
    saw_trap, _invariant_ok = _run_realized_step_scenario()
    assert saw_trap, (
        "no frame exercised the realized-step trap -- the test no longer proves the "
        "fix is load-bearing; find a scenario that does"
    )
