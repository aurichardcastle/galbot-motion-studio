"""GuardedPathController against the REAL model and the real 2026-08-07 incident.

The unit tests exercise the controller's logic with an injected predicate. This
module proves it works on actual collision geometry: the documented incident is a
left_arm_joint3 sweep from HOME whose endpoints are both clear but whose interior
drives left_arm_link5/6 through the leg column and then the left gripper through
the right arm. Before this controller that sweep reached the supervisor, was
rejected, and latched a frozen HOLD. Here we show the controller advances as far
along the sweep as is clear and stops there -- smooth progress, never the wall.

Uses the pipeline's own ClearanceChecker so "safe" here is identical to what the
SafetySupervisor independently enforces in production.
"""

from __future__ import annotations

import pytest

from galbot_motion_studio.pipeline import MotionStudioPipeline
from galbot_motion_studio.retarget.guarded_controller import GuardedPathController
from galbot_motion_studio.safety.clearance import HOME_QPOS


@pytest.fixture(scope="module")
def checker():
    # The real production checker (SIM profile), via the pipeline that wires it.
    return MotionStudioPipeline(source_clock_id="integration-test").checker


@pytest.fixture(scope="module")
def controller(checker):
    # Empty joint_limits isolates the collision backoff from the limit clamp
    # (the clamp is covered by the unit tests); HOME pins some joints at their
    # hard stops, so clamping here would perturb the incident geometry.
    def clearance_check(start, end):
        return checker.check_path(start, end, steps=2, max_joint_step=0.01).ok

    return GuardedPathController(
        joint_limits={},
        soft_margin_rad=0.0,
        clearance_check=clearance_check,
    )


def _clear(checker, start, end) -> bool:
    return checker.check_path(start, end, steps=2, max_joint_step=0.01).ok


def test_incident_reachable_collision_backs_off_instead_of_freezing(checker, controller):
    """j3 -0.5624 -> -0.2624: the raw target collides; the controller stops short, safe."""
    start = dict(HOME_QPOS)
    goal = dict(HOME_QPOS)
    goal["left_arm_joint3"] = -0.2624

    # Today's behaviour: driving the raw target's straight line collides, so the
    # supervisor would HOLD and the limb would freeze.
    assert not _clear(checker, start, goal)

    result = controller.solve(start, goal)

    # The controller made real progress toward the operator's pose ...
    assert result.backed_off
    assert start["left_arm_joint3"] < result.target["left_arm_joint3"] < goal["left_arm_joint3"]
    # ... and the pose it actually emits is collision-free, so the supervisor
    # ALLOWs it instead of latching a HOLD. No freeze.
    assert _clear(checker, start, result.target)
    assert 0.0 < result.alpha < 1.0


def test_incident_endpoints_clear_interior_bad_is_still_caught_and_avoided(checker, controller):
    """j3 -0.5624 -> +1.5: both endpoints clear, interior catastrophic (< -50 mm).

    The property the whole clearance module exists for. The controller must not be
    fooled by the clear endpoint: it must stop before the interior collision.
    """
    start = dict(HOME_QPOS)
    goal = dict(HOME_QPOS)
    goal["left_arm_joint3"] = 1.5

    # Both endpoints are individually clear ...
    assert checker.check_pose(start).ok
    assert checker.check_pose(goal).ok
    # ... but the swept straight line between them is not: this is the freeze case.
    assert not _clear(checker, start, goal)

    result = controller.solve(start, goal)

    assert result.backed_off
    assert result.alpha < 1.0
    # It stopped before the interior collision, short of the (clear) far endpoint.
    assert start["left_arm_joint3"] < result.target["left_arm_joint3"] < goal["left_arm_joint3"]
    # The emitted pose and the whole path to it are clear -> supervisor ALLOWs.
    assert _clear(checker, start, result.target)


def test_a_clear_target_is_passed_through_untouched(checker, controller):
    """A safe reach (j3 -0.5624 -> -0.40, known clear) must not be slowed at all."""
    start = dict(HOME_QPOS)
    goal = dict(HOME_QPOS)
    goal["left_arm_joint3"] = -0.40  # test_j3_minus_040_is_clear

    assert _clear(checker, start, goal)  # precondition: genuinely reachable

    result = controller.solve(start, goal)
    assert not result.backed_off
    assert result.alpha == 1.0
    assert result.target["left_arm_joint3"] == pytest.approx(-0.40)
