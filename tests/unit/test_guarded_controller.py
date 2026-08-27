"""Unit tests for the guard-rail-aware path controller.

No MuJoCo here: the clearance check is an injected predicate, so these exercise
the controller's decision logic (limit clamp, collision backoff, joint-set
contract) in isolation and fast.
"""

from __future__ import annotations

import pytest

from galbot_motion_studio.retarget.guarded_controller import GuardedPathController


def _always_clear(_a, _b) -> bool:
    return True


LIMITS = {"j": (-1.0, 1.0)}


def test_safe_goal_passes_through_unchanged() -> None:
    c = GuardedPathController(
        joint_limits=LIMITS, soft_margin_rad=0.09, clearance_check=_always_clear
    )
    result = c.solve({"j": 0.0}, {"j": 0.5})
    assert result.alpha == 1.0
    assert result.target == {"j": 0.5}
    assert not result.limited


def test_goal_clamped_into_soft_limit_band() -> None:
    c = GuardedPathController(
        joint_limits=LIMITS, soft_margin_rad=0.09, clearance_check=_always_clear
    )
    result = c.solve({"j": 0.0}, {"j": 5.0})  # well past the 1.0 upper stop
    assert result.target["j"] == pytest.approx(1.0 - 0.09)
    assert result.clamped_joints == ("j",)
    assert result.limited


def test_collision_backoff_moves_partway_toward_target_not_freeze() -> None:
    # A "wall": any swept step whose endpoint pushes j past 0.30 collides. The
    # proportional bisection must advance toward it and stop short, never freeze.
    def check(_a, b):
        return b["j"] <= 0.30 + 1e-9

    c = GuardedPathController(
        joint_limits={"j": (-3.14, 3.14)},
        soft_margin_rad=0.0,
        clearance_check=check,
    )
    result = c.solve({"j": 0.0}, {"j": 1.0})
    assert result.backed_off
    assert 0.0 < result.target["j"] <= 0.30  # progressed toward the wall, did not cross it
    assert 0.0 < result.alpha < 1.0
    assert check({"j": 0.0}, result.target)  # emitted pose is itself safe


def test_head_on_wall_holds_position_without_freezing() -> None:
    # Start is already at the wall; there is no safe forward motion this frame.
    def check(_a, b):
        return b["j"] <= 0.0 + 1e-9

    c = GuardedPathController(
        joint_limits={"j": (-3.14, 3.14)},
        soft_margin_rad=0.0,
        clearance_check=check,
    )
    result = c.solve({"j": 0.0}, {"j": 1.0})
    assert result.target["j"] == 0.0  # holds, is not driven into the wall
    assert result.alpha == 0.0
    assert check({"j": 0.0}, result.target)  # holding the start pose is legal/approved


def test_backoff_scales_all_joints_together_preserving_direction() -> None:
    # The wall is defined by j1 alone, but the proportional bisection must scale the
    # whole arm by one alpha so the motion stays a straight line (shape preserved)
    # when the straight step still makes progress -- the detour only kicks in when
    # the straight step is essentially cornered.
    def check(_a, b):
        return b["j1"] <= 0.5 + 1e-9

    c = GuardedPathController(
        joint_limits={"j1": (-3.14, 3.14), "j2": (-3.14, 3.14)},
        soft_margin_rad=0.0,
        clearance_check=check,
    )
    result = c.solve({"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 1.0})
    assert result.target["j1"] == pytest.approx(0.5, abs=1e-3)
    assert result.target["j2"] == pytest.approx(0.5, abs=1e-3)  # moved proportionally


def test_detour_advances_the_free_joint_when_straight_step_is_fully_blocked() -> None:
    # THE fix: the straight step is blocked from the very start (any motion of j2
    # collides), so the proportional backoff is cornered at 0. A straight-line-only
    # controller would freeze the whole arm. The per-joint detour must advance the
    # free joint (j1) toward the target while holding back the one driving the
    # collision (j2) -- so the arm keeps tracking instead of parking.
    def check(_a, b):
        return b["j2"] <= 1e-9  # any j2 motion collides; j1 is always free

    c = GuardedPathController(
        joint_limits={"j1": (-3.14, 3.14), "j2": (-3.14, 3.14)},
        soft_margin_rad=0.0,
        clearance_check=check,
        max_lookahead_rad=0.1,
    )
    result = c.solve({"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 1.0})
    assert result.backed_off
    assert result.target["j1"] == pytest.approx(0.1)  # free joint advanced to the horizon
    assert result.target["j2"] == pytest.approx(0.0)  # collision-driving joint held
    assert result.alpha > 0.0  # made real progress, did not freeze
    assert check({"j1": 0.0, "j2": 0.0}, result.target)  # emitted pose is safe


def test_detour_never_uses_the_tummy_joint_to_resolve_an_arm_blockage() -> None:
    """Whole-upper-body yaw is an operator command, not an IK escape route."""
    def check(_a, b):
        return b["left_arm_joint3"] <= 1e-9

    controller = GuardedPathController(
        joint_limits={},
        soft_margin_rad=0.0,
        clearance_check=check,
        max_lookahead_rad=0.1,
    )
    start = {"leg_joint4": 0.0, "left_arm_joint3": 0.0, "right_arm_joint3": 0.0}
    goal = {"leg_joint4": 1.0, "left_arm_joint3": 1.0, "right_arm_joint3": 1.0}
    result = controller.solve(start, goal)
    assert result.backed_off
    assert result.target["left_arm_joint3"] == pytest.approx(0.0)
    # Right arm can still make a genuine free detour; tummy stays at its
    # proportional base (zero here) rather than being used as a workaround.
    assert result.target["right_arm_joint3"] == pytest.approx(0.1)
    assert result.target["leg_joint4"] == pytest.approx(0.0)


def test_protected_tummy_turn_is_not_suppressed_by_a_cornered_arm() -> None:
    """An arm collision may hold the arm, never silently shrink tummy intent."""
    def check(_a, b):
        return b["left_arm_joint3"] <= 1e-9

    controller = GuardedPathController(
        joint_limits={},
        soft_margin_rad=0.0,
        clearance_check=check,
        max_lookahead_rad=0.1,
        protected_joints=("leg_joint4",),
    )
    start = {"leg_joint4": 0.0, "left_arm_joint3": 0.0}
    goal = {"leg_joint4": 1.0, "left_arm_joint3": 1.0}
    result = controller.solve(start, goal)
    assert result.backed_off
    assert result.target["left_arm_joint3"] == pytest.approx(0.0)
    assert result.target["leg_joint4"] == pytest.approx(0.1)


def test_tummy_collision_holds_only_tummy_and_allows_arm_detour() -> None:
    """Protection is priority, never a clearance exemption or arm freeze."""
    def check(_a, b):
        # The protected tummy's requested horizon itself is unsafe. Holding it
        # at start must still leave an independently free arm able to advance.
        return b["leg_joint4"] < 0.1

    controller = GuardedPathController(
        joint_limits={},
        soft_margin_rad=0.0,
        clearance_check=check,
        max_lookahead_rad=0.1,
        protected_joints=("leg_joint4",),
    )
    start = {"leg_joint4": 0.0, "left_arm_joint3": 0.0}
    goal = {"leg_joint4": 1.0, "left_arm_joint3": 1.0}
    result = controller.solve(start, goal)
    assert result.backed_off
    assert result.target["leg_joint4"] == pytest.approx(0.0)
    assert result.target["left_arm_joint3"] == pytest.approx(0.1)


def test_held_joint_with_zero_delta_stays_put() -> None:
    # goal == start on a joint (a held limb): it must not move regardless of alpha.
    def check(_a, _b):
        return True

    c = GuardedPathController(
        joint_limits={"j1": (-3.14, 3.14), "j2": (-3.14, 3.14)},
        soft_margin_rad=0.0,
        clearance_check=check,
    )
    result = c.solve({"j1": 0.2, "j2": 1.0}, {"j1": 0.9, "j2": 1.0})
    assert result.target["j2"] == 1.0  # unchanged


def test_context_folded_into_clearance_query() -> None:
    seen: dict[str, dict] = {}

    def check(a, b):
        seen["start"] = dict(a)
        seen["end"] = dict(b)
        return True

    c = GuardedPathController(
        joint_limits=LIMITS, soft_margin_rad=0.0, clearance_check=check
    )
    c.solve({"j": 0.0}, {"j": 0.2}, context={"left_gripper_joint": 0.1})
    assert seen["start"]["left_gripper_joint"] == 0.1
    assert seen["end"]["left_gripper_joint"] == 0.1
    assert seen["end"]["j"] == 0.2  # governed joint still present in the query


def test_context_is_not_part_of_emitted_target() -> None:
    c = GuardedPathController(
        joint_limits=LIMITS, soft_margin_rad=0.0, clearance_check=_always_clear
    )
    result = c.solve({"j": 0.0}, {"j": 0.2}, context={"left_gripper_joint": 0.1})
    assert set(result.target) == {"j"}  # grippers are governed separately


def test_mismatched_joint_sets_rejected() -> None:
    c = GuardedPathController(
        joint_limits=LIMITS, soft_margin_rad=0.0, clearance_check=_always_clear
    )
    with pytest.raises(ValueError):
        c.solve({"j": 0.0}, {"j": 0.0, "k": 0.0})


def test_non_finite_positions_rejected() -> None:
    c = GuardedPathController(
        joint_limits=LIMITS, soft_margin_rad=0.0, clearance_check=_always_clear
    )
    with pytest.raises(ValueError):
        c.solve({"j": 0.0}, {"j": float("nan")})


def test_lookahead_caps_each_joint_independently() -> None:
    # Both deltas exceed the horizon, so both cap at it. Crucially the cap is
    # per-joint, not a single global scale.
    c = GuardedPathController(
        joint_limits={},
        soft_margin_rad=0.0,
        clearance_check=_always_clear,
        max_lookahead_rad=0.2,
    )
    result = c.solve({"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 0.5})
    assert not result.backed_off
    assert result.target["j1"] == pytest.approx(0.2)
    assert result.target["j2"] == pytest.approx(0.2)


def test_lookahead_on_a_big_joint_does_not_shrink_a_small_one() -> None:
    # The per-limb independence property: a large-delta joint hitting the horizon
    # must NOT scale down an unrelated small-delta joint. A global scale would have
    # made j2 = 0.05 * (0.2/1.0) = 0.01; per-joint bounding leaves it at 0.05.
    c = GuardedPathController(
        joint_limits={},
        soft_margin_rad=0.0,
        clearance_check=_always_clear,
        max_lookahead_rad=0.2,
    )
    result = c.solve({"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 0.05})
    assert result.target["j1"] == pytest.approx(0.2)  # capped at the horizon
    assert result.target["j2"] == pytest.approx(0.05)  # untouched


def test_lookahead_does_not_shrink_a_nearby_goal() -> None:
    c = GuardedPathController(
        joint_limits={},
        soft_margin_rad=0.0,
        clearance_check=_always_clear,
        max_lookahead_rad=0.2,
    )
    result = c.solve({"j": 0.0}, {"j": 0.05})  # within the horizon
    assert result.alpha == 1.0
    assert result.target["j"] == pytest.approx(0.05)


def test_lookahead_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        GuardedPathController(
            joint_limits={},
            soft_margin_rad=0.0,
            clearance_check=_always_clear,
            max_lookahead_rad=0.0,
        )


def test_joint_without_declared_limit_passes_through() -> None:
    c = GuardedPathController(
        joint_limits={}, soft_margin_rad=0.09, clearance_check=_always_clear
    )
    result = c.solve({"free": 0.0}, {"free": 12.0})
    assert result.target["free"] == 12.0
    assert result.clamped_joints == ()


def _standoff_controller(preferred_clearance_m: float) -> GuardedPathController:
    """A controller over a 1-D 'wall': clearance falls linearly with the joint."""

    def clearance(start, end):
        # Clearance shrinks as the joint advances; blocked past 1.0 rad.
        value = 0.02 - 0.015 * max(end["j"], start["j"])
        return value if value > 0.005 else False

    return GuardedPathController(
        joint_limits={"j": (-2.0, 2.0)},
        soft_margin_rad=0.0,
        clearance_check=clearance,
        preferred_clearance_m=preferred_clearance_m,
    )


def test_without_a_standoff_the_backoff_parks_on_the_floor() -> None:
    step = _standoff_controller(0.0).solve({"j": 0.0}, {"j": 2.0})
    # 0.02 - 0.015*j > 0.005  =>  j < 1.0: the bisection converges on the wall.
    assert 0.99 < step.target["j"] <= 1.0


def test_a_standoff_keeps_margin_in_hand() -> None:
    """The property, not a particular number.

    Bisection converges on a dyadic fraction, so it lands at or inside the
    continuous answer (0.667 here) rather than on it -- conservative by
    construction, because it only ever accepts a point it has verified.
    """
    step = _standoff_controller(0.010).solve({"j": 0.0}, {"j": 2.0})
    assert step.backed_off
    achieved = 0.02 - 0.015 * step.target["j"]
    assert achieved >= 0.010, f"standoff not met: {achieved}"
    # And it is real motion, not a freeze.
    assert step.target["j"] > 0.5


def test_an_unreachable_standoff_falls_back_to_the_floor() -> None:
    """The standoff may never cost reach that was previously available."""
    unreachable = _standoff_controller(0.050).solve({"j": 0.0}, {"j": 2.0})
    baseline = _standoff_controller(0.0).solve({"j": 0.0}, {"j": 2.0})
    assert unreachable.target["j"] == baseline.target["j"]


def test_a_yes_no_predicate_still_works() -> None:
    """A caller that cannot measure distance must not be silently blocked."""
    controller = GuardedPathController(
        joint_limits={"j": (-2.0, 2.0)},
        soft_margin_rad=0.0,
        clearance_check=lambda start, end: end["j"] <= 1.0,
        preferred_clearance_m=0.010,
    )
    step = controller.solve({"j": 0.0}, {"j": 2.0})
    assert 0.99 < step.target["j"] <= 1.0


def test_a_negative_standoff_is_rejected() -> None:
    try:
        _standoff_controller(-0.001)
    except ValueError as error:
        assert "finite and non-negative" in str(error)
    else:
        raise AssertionError("a negative standoff was accepted")
