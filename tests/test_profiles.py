"""Tests for motion profiles.

There are now two load-bearing invariant tests, because a profile touches two
different objects:

``test_profile_cannot_relax_a_geometric_guarantee``
    enumerates every field of ``SupervisorPolicy`` and asserts a profile changes
    only the declared dynamics fields.

``test_profile_cannot_relax_any_other_clearance_parameter``
    enumerates every parameter of ``ClearanceChecker.__init__`` and asserts a
    profile changes only ``floor``.

Why the second one exists, and why the first one was not simply widened
----------------------------------------------------------------------
This module used to promise that a profile "may never relax a geometric or
structural guarantee". That promise is now false: ``SIM`` lowers the
self-clearance floor from 10 mm to 5 mm. The measurement behind that is in the
``profiles`` module docstring; the short version is that a 10 mm floor rejects
31.7% of clear teleop poses and 31.5% of frames during motion around a recorded
raised-hands pose, while rejecting exactly the same 100% of genuinely
interpenetrating poses that 5 mm does.

The dangerous way to land that change would have been to add ``"floor"`` to
``RELAXABLE_FIELDS`` and move on -- the enumeration test would still pass, and
the module's stated invariant would have quietly become "a profile may relax
whatever is in this set". So instead:

  * the floor lives in its own frozen set, ``RELAXABLE_CLEARANCE_PARAMS``,
    because it is a different kind of permission over a different object;
  * both sets are pinned by name below, so adding a third relaxable knob means
    editing a literal in this file with the evidence in hand;
  * the behavioural tests at the bottom assert the actual measured consequence
    on the actual model, so the number cannot drift away from its justification.

Run:
    .venv/bin/python -m pytest tests/test_profiles.py -q
"""

from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galbot_motion_studio.safety import profiles  # noqa: E402
from galbot_motion_studio.safety.clearance import (  # noqa: E402
    DEFAULT_FLOOR_M,
    ClearanceChecker,
)
from galbot_motion_studio.safety.profiles import (  # noqa: E402
    HARDWARE_CLEARANCE_FLOOR_M,
    HARDWARE_FLOOR_RATCHET_M,
    MIN_FLOOR_NOISE_RATIO,
    RELAXABLE_CLEARANCE_PARAMS,
    RELAXABLE_FIELDS,
    SIM_CLEARANCE_FLOOR_M,
    SIM_MAX_JOINT_STEP_RAD,
    STRUCTURAL_NOISE_CEILING_M,
    URDF_ARM_ACCELERATION_RAD_S2,
    URDF_JOINT_VELOCITY_RAD_S,
    MotionProfile,
    clearance_floor_for,
    clearance_kwargs_for,
    describe,
    policy_for,
)
from galbot_motion_studio.safety.supervisor import SupervisorPolicy  # noqa: E402

#: The raised-both-hands pose, solved from the operator world landmarks recorded
#: in ``tests/unit/test_left_arm_retargeting.py`` and frozen here as numbers so
#: this test does not depend on the retargeting code it was derived from.
#: Its minimum clearance is 10.119 mm, against ``right_arm_link2 |
#: torso_base_link`` -- an exact measurement, stable across every ``distmax``
#: from 12.5 mm to 120 mm.
RAISED_HANDS_QPOS = {
    "left_arm_joint1": 1.075266,
    "left_arm_joint2": -1.449637,
    "left_arm_joint3": -0.692424,
    "left_arm_joint4": -2.439735,
    "left_arm_joint5": -0.051787,
    "left_arm_joint6": -0.001403,
    "left_arm_joint7": 0.032526,
    "right_arm_joint1": -0.429205,
    "right_arm_joint2": 0.904365,
    "right_arm_joint3": 0.634309,
    "right_arm_joint4": 2.041974,
    "right_arm_joint5": 0.319922,
    "right_arm_joint6": 0.134646,
    "right_arm_joint7": 0.075907,
}

#: The 2026-08-07 incident pose: ``left_arm_link5`` driven into ``leg_link2`` at
#: -33.76 mm. Every joint is inside its URDF limit, so this is a genuine
#: self-collision and not an out-of-range pose being caught by something else.
SELF_COLLISION_QPOS = {"left_arm_joint3": -0.2624}


class TestProfiles(unittest.TestCase):
    def test_hardware_is_the_default_and_is_unchanged(self) -> None:
        """The conservative envelope must survive the existence of a faster one."""
        self.assertEqual(policy_for(MotionProfile.HARDWARE), SupervisorPolicy())

    def test_sim_matches_the_models_own_declared_limits(self) -> None:
        """SIM sets the ceiling AT the URDF limit, not above it."""
        p = policy_for(MotionProfile.SIM)
        self.assertEqual(p.max_joint_rate_rad_s, URDF_JOINT_VELOCITY_RAD_S)
        self.assertEqual(p.max_joint_acceleration_rad_s2, URDF_ARM_ACCELERATION_RAD_S2)
        self.assertEqual(p.max_joint_step_rad, SIM_MAX_JOINT_STEP_RAD)

    def test_sim_is_meaningfully_faster_than_hardware(self) -> None:
        """If the split does not actually buy expressiveness it is not worth having."""
        hw, sim = policy_for(MotionProfile.HARDWARE), policy_for(MotionProfile.SIM)
        self.assertGreater(sim.max_joint_rate_rad_s / hw.max_joint_rate_rad_s, 4.0)
        self.assertGreater(
            sim.max_joint_acceleration_rad_s2 / hw.max_joint_acceleration_rad_s2, 1.9
        )

    def test_profile_cannot_relax_a_geometric_guarantee(self) -> None:
        """THE invariant, surface 1 of 2: the supervisor policy.

        Enumerated over the real dataclass, so a newly added policy field is
        covered automatically instead of being silently exempt.
        """
        hw, sim = policy_for(MotionProfile.HARDWARE), policy_for(MotionProfile.SIM)
        for f in fields(SupervisorPolicy):
            a, b = getattr(hw, f.name), getattr(sim, f.name)
            if f.name in RELAXABLE_FIELDS:
                continue
            self.assertEqual(
                a, b,
                f"{f.name} differs between profiles; only {sorted(RELAXABLE_FIELDS)} may vary",
            )

    def test_safety_critical_fields_are_identical_by_name(self) -> None:
        """Belt and braces: name the fields explicitly as well as enumerating them."""
        hw, sim = policy_for(MotionProfile.HARDWARE), policy_for(MotionProfile.SIM)
        for name in ("soft_margin_rad", "max_ik_residual_m",
                     "clearance_match_tolerance_m", "model_hash", "tool_hash"):
            self.assertEqual(getattr(hw, name), getattr(sim, name), name)

    def test_soft_margin_still_exceeds_the_measured_drive_fault_distance(self) -> None:
        """left_arm_joint4 faulted 2.1 degrees (0.0367 rad) from its stop under load."""
        for profile in MotionProfile:
            self.assertGreater(policy_for(profile).soft_margin_rad, 0.0367)

    def test_unknown_profile_raises_rather_than_defaulting(self) -> None:
        """A typo must not silently select the permissive envelope."""
        for bad in ("SIMULATION", "sim ", "", "Sim", "hardware2", None, 3):
            with self.assertRaises((ValueError, KeyError, TypeError), msg=repr(bad)):
                policy_for(bad)  # type: ignore[arg-type]

    def test_string_values_are_accepted_for_config_files(self) -> None:
        self.assertEqual(policy_for("sim"), policy_for(MotionProfile.SIM))
        self.assertEqual(policy_for("hardware"), policy_for(MotionProfile.HARDWARE))

    def test_base_policy_is_respected_and_not_mutated(self) -> None:
        """Profiles compose with a caller-supplied base without side effects."""
        base = SupervisorPolicy(soft_margin_rad=0.12)
        out = policy_for(MotionProfile.SIM, base)
        self.assertEqual(out.soft_margin_rad, 0.12)
        self.assertEqual(out.max_joint_rate_rad_s, URDF_JOINT_VELOCITY_RAD_S)
        self.assertEqual(base.max_joint_rate_rad_s, 0.35, "base must not be mutated")

    def test_describe_states_the_envelope_plainly(self) -> None:
        self.assertIn("1.5", describe(MotionProfile.SIM))
        self.assertIn("model URDF limits", describe(MotionProfile.SIM))
        self.assertIn("0.35", describe("hardware"))
        self.assertIn("conservative", describe("hardware"))

    def test_describe_also_states_the_clearance_floor(self) -> None:
        """The floor now differs between profiles, so it belongs in clip metadata."""
        self.assertIn("clearance floor >= 5 mm", describe(MotionProfile.SIM))
        self.assertIn("clearance floor >= 10 mm", describe("hardware"))


class TestRelaxablePermissionsArePinned(unittest.TestCase):
    """The relaxable sets are the whole safety argument. Pin them by name.

    An enumeration test alone ("everything not in the set must match") is
    satisfied by adding a field to the set. These tests make that addition
    visible: it cannot happen without editing a literal here.
    """

    #: Every knob any profile is currently allowed to move, spelled out.
    EXPECTED_DYNAMICS = frozenset({
        "max_joint_rate_rad_s",
        "max_joint_acceleration_rad_s2",
        "max_joint_step_rad",
    })
    EXPECTED_CLEARANCE = frozenset({"floor"})

    def test_relaxable_clearance_params_is_exactly_the_floor(self) -> None:
        """The newly relaxable knob, named.

        ``floor`` is a geometric threshold and its presence here is an
        exception, argued from a swept measurement in the module docstring. If
        this set ever grows, the same standard applies: show that the change
        costs no detection, and that the new value stays above the model's
        structural noise.
        """
        self.assertEqual(RELAXABLE_CLEARANCE_PARAMS, self.EXPECTED_CLEARANCE)

    def test_relaxable_policy_fields_are_dynamics_only(self) -> None:
        """Subset rather than equality: only ADDING a field is dangerous."""
        extra = RELAXABLE_FIELDS - self.EXPECTED_DYNAMICS
        self.assertEqual(
            extra, frozenset(),
            f"{sorted(extra)} was added to RELAXABLE_FIELDS without updating this "
            "test; a profile may only relax the dynamics envelope",
        )

    def test_the_two_permission_sets_do_not_overlap(self) -> None:
        """They govern different objects; a shared name would be ambiguous."""
        self.assertEqual(RELAXABLE_FIELDS & RELAXABLE_CLEARANCE_PARAMS, frozenset())

    def test_profile_cannot_relax_any_other_clearance_parameter(self) -> None:
        """THE invariant, surface 2 of 2: the clearance checker.

        Enumerated over the real constructor signature, so a newly added
        checker parameter is covered automatically. ``distmax``,
        ``baseline_margin``, ``ladder_rungs``, ``tool_patterns`` and
        ``collision_groups`` decide what the checker is able to SEE; widening
        those through a profile would weaken the measurement rather than the
        threshold, and no measurement could justify that.
        """
        params = {
            name for name, p in
            inspect.signature(ClearanceChecker.__init__).parameters.items()
            if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
        }
        self.assertIn("floor", params, "ClearanceChecker lost its floor parameter")
        hw = clearance_kwargs_for(MotionProfile.HARDWARE)
        sim = clearance_kwargs_for(MotionProfile.SIM)
        for name in sorted(params - RELAXABLE_CLEARANCE_PARAMS):
            self.assertEqual(
                hw.get(name), sim.get(name),
                f"{name} differs between profiles; only "
                f"{sorted(RELAXABLE_CLEARANCE_PARAMS)} may vary",
            )

    def test_clearance_kwargs_only_names_real_checker_parameters(self) -> None:
        """A kwarg that is not a real parameter would blow up at the call site."""
        params = set(inspect.signature(ClearanceChecker.__init__).parameters)
        for profile in MotionProfile:
            unknown = set(clearance_kwargs_for(profile)) - params
            self.assertEqual(unknown, set(), f"{profile}: {sorted(unknown)}")

    def test_clearance_kwargs_leaves_every_other_parameter_at_its_default(self) -> None:
        """Absence, not restatement -- so a changed default cannot be shadowed."""
        for profile in MotionProfile:
            self.assertEqual(set(clearance_kwargs_for(profile)), {"floor"})


class TestClearanceFloorBounds(unittest.TestCase):
    def test_hardware_floor_keeps_todays_conservative_value(self) -> None:
        self.assertEqual(clearance_floor_for(MotionProfile.HARDWARE), 0.010)
        self.assertEqual(HARDWARE_CLEARANCE_FLOOR_M, 0.010)
        self.assertEqual(HARDWARE_FLOOR_RATCHET_M, 0.010)

    def test_hardware_floor_tracks_the_module_default_upward_only(self) -> None:
        """A future edit lowering clearance.DEFAULT_FLOOR_M must not loosen hardware."""
        self.assertEqual(
            HARDWARE_CLEARANCE_FLOOR_M, max(DEFAULT_FLOOR_M, HARDWARE_FLOOR_RATCHET_M)
        )
        self.assertGreaterEqual(HARDWARE_CLEARANCE_FLOOR_M, HARDWARE_FLOOR_RATCHET_M)

    def test_sim_floor_is_strictly_lower_than_hardware(self) -> None:
        """If it were not, the split would not be buying anything."""
        self.assertLess(
            clearance_floor_for(MotionProfile.SIM),
            clearance_floor_for(MotionProfile.HARDWARE),
        )

    def test_sim_floor_is_never_zero(self) -> None:
        """A zero floor is contact detection, which clearance.py exists to replace."""
        for profile in MotionProfile:
            self.assertGreater(clearance_floor_for(profile), 0.0)

    def test_sim_floor_clears_the_structural_noise_ceiling(self) -> None:
        """Measured: left_arm_link5 | left_arm_link7 sits at 1.731 mm at home.

        A floor at or below that band stops baseline-tracking the model's own
        permanently-close pairs and starts floor-comparing them, where their
        narrowphase wobble fires continuously.
        """
        self.assertEqual(STRUCTURAL_NOISE_CEILING_M, 0.001731)
        self.assertGreaterEqual(
            SIM_CLEARANCE_FLOOR_M, MIN_FLOOR_NOISE_RATIO * STRUCTURAL_NOISE_CEILING_M
        )

    def test_unknown_profile_raises_for_the_floor_too(self) -> None:
        for bad in ("SIMULATION", "sim ", "", "Sim", None, 3):
            with self.assertRaises((ValueError, KeyError, TypeError), msg=repr(bad)):
                clearance_floor_for(bad)  # type: ignore[arg-type]

    def test_import_time_validation_rejects_a_floor_inside_the_noise_band(self) -> None:
        """The guard is code, not just documentation."""
        with mock.patch.object(profiles, "SIM_CLEARANCE_FLOOR_M", 0.0015):
            with self.assertRaises(ValueError) as ctx:
                profiles._validate_floors()
        self.assertIn("structural noise ceiling", str(ctx.exception))

    def test_import_time_validation_rejects_a_zero_floor(self) -> None:
        with mock.patch.object(profiles, "SIM_CLEARANCE_FLOOR_M", 0.0):
            with self.assertRaises(ValueError) as ctx:
                profiles._validate_floors()
        self.assertIn("contact detection", str(ctx.exception))

    def test_import_time_validation_rejects_sim_stricter_than_hardware(self) -> None:
        with mock.patch.object(profiles, "SIM_CLEARANCE_FLOOR_M", 0.020):
            with self.assertRaises(ValueError) as ctx:
                profiles._validate_floors()
        self.assertIn("must not exceed", str(ctx.exception))

    def test_import_time_validation_rejects_a_loosened_hardware_floor(self) -> None:
        with mock.patch.object(profiles, "HARDWARE_CLEARANCE_FLOOR_M", 0.008):
            with self.assertRaises(ValueError) as ctx:
                profiles._validate_floors()
        self.assertIn("never loosened", str(ctx.exception))


class TestMeasuredClearanceBehaviour(unittest.TestCase):
    """The evidence, re-run against the real model.

    These are the tests that stop the 5 mm from drifting away from the argument
    that produced it. They load the shipped MJCF; each checker costs one
    baseline sweep, so they are built once for the class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = ClearanceChecker(**clearance_kwargs_for(MotionProfile.SIM))
        cls.hw = ClearanceChecker(**clearance_kwargs_for(MotionProfile.HARDWARE))

    def test_profiles_actually_reach_the_checker(self) -> None:
        self.assertEqual(self.sim.floor, 0.005)
        self.assertEqual(self.hw.floor, 0.010)

    def test_the_raised_hands_pose_is_within_a_hair_of_the_hardware_floor(self) -> None:
        """10.119 mm against 10.000 mm. The static pose passes; motion does not."""
        report = self.hw.check_pose(RAISED_HANDS_QPOS)
        self.assertAlmostEqual(report.min_distance, 0.010119, places=6)
        self.assertLess(report.min_distance - self.hw.floor, 0.0002)
        self.assertIn("right_arm_link2", str(report.min_pair.key))
        self.assertIn("torso_base_link", str(report.min_pair.key))
        self.assertTrue(report.min_pair.exact, "this must be a measurement, not a bound")
        self.assertFalse(report.min_pair.is_tool, "not a mismatched-gripper artifact")

    def test_sim_gives_the_raised_hands_pose_real_headroom(self) -> None:
        """0.119 mm becomes 5.119 mm -- the point of the whole change."""
        report = self.sim.check_pose(RAISED_HANDS_QPOS)
        self.assertGreater(report.min_distance - self.sim.floor, 0.005)

    def test_the_swept_path_to_raised_hands_passes_under_sim_and_fails_under_hardware(self) -> None:
        """The supervisor checks paths, not poses. This is the user-visible bug."""
        self.assertFalse(
            self.hw.check_path(None, RAISED_HANDS_QPOS, steps=33).ok,
            "the 10 mm floor is expected to reject motion into a visually correct pose",
        )
        self.assertTrue(
            self.sim.check_path(None, RAISED_HANDS_QPOS, steps=33).ok,
            "the 5 mm floor must accept it",
        )

    def test_both_profiles_still_reject_a_genuine_self_collision(self) -> None:
        """The 2026-08-07 incident: left_arm_link5 into leg_link2 at -33.76 mm.

        Lowering the floor must not cost any detection. This is the assertion
        that makes the change safe rather than merely convenient.
        """
        for name, checker in (("sim", self.sim), ("hardware", self.hw)):
            report = checker.check_pose(SELF_COLLISION_QPOS)
            self.assertFalse(report.ok, name)
            self.assertLess(report.min_distance, 0.0, name)
            self.assertTrue(
                any(not v.is_tool for v in report.violations),
                f"{name}: a structural violation, not just a tool-link one",
            )

    def test_sim_floor_leaves_every_structural_pair_baseline_tracked(self) -> None:
        """The lower bound, checked against the model rather than a constant.

        The four permanently-close pairs must still be recorded as baseline
        overlaps under SIM, exactly as they are under HARDWARE. If a future
        floor drops into that band this fails, and so does the demo.

        Four, not five: ``head_link2 | torso_base_link`` used to be the fifth,
        sitting at -73.671 mm because the head's collision geom is authored
        inside the torso. It is now excluded outright by
        ``clearance.EMBEDDED_BODY_PAIRS`` -- it is negative at every pose its
        joints can reach (-68.20 to -87.25 mm over 35 poses spanning the full
        head range), so it can never signal an approach-and-touch event and
        carried only noise. The four that remain are real, near-touching
        structure: ``*_arm_link5 | *_arm_link7`` at +1.171, +1.412, +1.664 and
        +1.731 mm, all of which genuinely separate and close as the wrist moves.
        """
        self.assertEqual(set(self.sim.baseline), set(self.hw.baseline))
        self.assertEqual(len(self.sim.baseline), 4)
        self.assertGreater(
            SIM_CLEARANCE_FLOOR_M, max(self.sim.baseline.values()),
            "every baseline pair must sit below the floor that recorded it",
        )

    def test_a_floor_inside_the_noise_band_fires_on_the_models_own_geometry(self) -> None:
        """Why the lower bound is real: 1.5 mm rejects a pose that is simply fine."""
        noisy = ClearanceChecker(floor=0.0015)
        self.assertLess(len(noisy.baseline), len(self.sim.baseline))
        report = noisy.check_pose(RAISED_HANDS_QPOS)
        self.assertFalse(report.ok)
        offenders = {"|".join(sorted(v.key.link_pair)) for v in report.violations}
        self.assertEqual(offenders, {"left_arm_link5|left_arm_link7"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
