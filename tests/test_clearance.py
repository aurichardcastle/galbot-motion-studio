#!/usr/bin/env python3
"""
Tests for src/galbot_motion_studio/safety/clearance.py.

The centrepiece is the 2026-08-07 incident: a left-arm joint sweep that had a clear start
pose, a clear end pose, and a collision in between.  `tools/galbot_preview.py` recorded
these penetration depths for the sweep of `left_arm_joint3` from -0.5624 to -0.2624 with
every other joint at the 18:06 hardware snapshot:

    j3 = -0.40      clear
    j3 = -0.35      left_arm_link5 | leg_link2   ~5.2 mm
    j3 = -0.30      left_arm_link5 | leg_link2   ~21.4 mm
    j3 = -0.2624    left_arm_link5 | leg_link2   ~33.8 mm
                    left_arm_link6 | leg_link2   ~16.2 mm

The checker must agree with those numbers.  The tolerance below is deliberately tight
enough to expose a real disagreement rather than hide one; see RECORDED_TOLERANCE_M and
the note on the -0.35 sample in `test_recorded_checkpoint_depths`.

stdlib `unittest` is used because the project venv has no pytest and this environment
cannot install one.  pytest collects these classes fine if it is ever added.

Run:
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python tests/test_clearance.py
"""

from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys
import time
import unittest

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from galbot_motion_studio.safety.clearance import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    EMBEDDED_BODY_PAIRS,
    HOME_QPOS,
    ClearanceChecker,
    check_path,
)

REPO_TELEOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_TELEOP, "src")
MJCF_DIR = os.path.dirname(DEFAULT_MODEL_PATH)

# The recorded figures are quoted to 0.1 mm, so allow 0.2 mm: rounding cannot fail the
# test, but a genuine disagreement still will.
RECORDED_TOLERANCE_M = 0.0002

_SHARED: dict = {}


def shared_checker() -> ClearanceChecker:
    """One checker for the whole run — construction costs a full baseline sweep."""
    if "ck" not in _SHARED:
        _SHARED["ck"] = ClearanceChecker()
    return _SHARED["ck"]


def patched_contact_model() -> mujoco.MjModel:
    """The shipped model with self-collision enabled, compiled IN MEMORY.

    `galbot_preview.py` has to rewrite the MJCF on disk to do this.  MjSpec makes that
    unnecessary, and these tests must not write anything.
    """
    if "patched" not in _SHARED:
        spec = mujoco.MjSpec.from_file(DEFAULT_MODEL_PATH)
        for body in spec.bodies:
            for geom in body.geoms:
                if geom.group == 3:
                    geom.contype = 1
        _SHARED["patched"] = spec.compile()
    return _SHARED["patched"]


def target_pair(checker: ClearanceChecker, body_a: str, body_b: str):
    """A checked geom pair spanning two named links."""
    for g1, g2 in checker._pair_ids:
        if {checker._body_name(int(g1)), checker._body_name(int(g2))} == {body_a, body_b}:
            return int(g1), int(g2)
    raise AssertionError(f"no checked pair for {body_a} / {body_b}")


# ======================================================================================



def _set_qpos(checker, pose):
    """Write a full pose into a checker's data without choosing a MuJoCo solver."""
    checker.data.qpos[:] = 0.0
    for name, value in pose.items():
        jid = checker._joint_id(name)
        checker.data.qpos[checker.model.jnt_qposadr[jid]] = float(value)

class TestModelIsOpenedReadOnly(unittest.TestCase):
    def test_uses_the_shipped_unpatched_mjcf_and_writes_nothing(self):
        """The distance API ignores contype, so no on-disk patching is needed."""
        snapshot = lambda: {  # noqa: E731
            n: os.stat(os.path.join(MJCF_DIR, n)).st_mtime_ns
            for n in sorted(os.listdir(MJCF_DIR))
        }
        before = snapshot()
        checker = ClearanceChecker()
        checker.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624}, steps=5
        )
        self.assertEqual(before, snapshot(), "must not create or touch model-dir files")
        self.assertTrue(checker.model_path.endswith("galbot_one_golf_fixed_base.xml"))

        # The unpatched model really does have self-collision disabled, which is exactly
        # why a contact-count approach needs the patch and a distance approach does not.
        group3 = checker.model.geom_group == 3
        self.assertEqual(int(group3.sum()), 184)
        self.assertTrue((checker.model.geom_contype[group3] == 0).all())

    def test_distances_are_identical_on_the_contype_patched_model(self):
        patched = ClearanceChecker(model=patched_contact_model())
        plain = shared_checker()
        pose = {"left_arm_joint3": -0.2624}

        a = plain.check_pose(pose)
        b = patched.check_pose(pose)
        self.assertEqual(a.min_pair.key, b.min_pair.key)
        self.assertEqual(a.min_distance, b.min_distance)

        # ...while the contact pipeline is completely different, which is the whole
        # reason galbot_preview.py needs its on-disk patch.  Driven through
        # mj_forward explicitly rather than through the checker's _apply: _apply
        # runs mj_kinematics only, because the checker reads geom_xpos/geom_xmat
        # and never the contact list.  The property under test here is the MODEL's
        # contact pipeline, not which MuJoCo entry point _apply happens to call.
        _set_qpos(plain, plain.full_pose(pose))
        _set_qpos(patched, patched.full_pose(pose))
        mujoco.mj_forward(plain.model, plain.data)
        mujoco.mj_forward(patched.model, patched.data)
        self.assertEqual(plain.data.ncon, 0)
        self.assertGreater(patched.data.ncon, 0)


class TestIncident20260807(unittest.TestCase):
    """Acceptance: reproduce what galbot_preview.py recorded."""

    @classmethod
    def setUpClass(cls):
        cls.ck = shared_checker()

    def test_recorded_checkpoint_depths(self):
        """Measured vs recorded, in mm:

            j3=-0.35  link5|leg2  recorded 5.2   measured  5.131   delta -0.069
            j3=-0.30  link5|leg2  recorded 21.4  measured 21.418   delta +0.018
            j3=-0.2624 link5|leg2 recorded 33.8  measured 33.762   delta -0.038
            j3=-0.2624 link6|leg2 recorded 16.2  measured 16.196   delta -0.004

        The -0.35 delta is fully accounted for and is NOT a disagreement about geometry.
        `galbot_preview.py`'s SNAPSHOT differs from the home pose quoted in the brief in
        the 4th decimal (left_arm_joint2 -1.4698 vs -1.4697, joint6 -0.2748 vs -0.2749,
        joint7 0.1114 vs 0.1115).  Feeding this checker preview's SNAPSHOT instead gives
        5.155 / 21.442 / 33.786 mm, and preview prints depths with "%6.1f", so 5.155 is
        what it displayed as "5.2".  The residual is display rounding.
        """
        recorded = [
            (-0.35, "left_arm_link5", "leg_link2", -0.0052),
            (-0.30, "left_arm_link5", "leg_link2", -0.0214),
            (-0.2624, "left_arm_link5", "leg_link2", -0.0338),
            (-0.2624, "left_arm_link6", "leg_link2", -0.0162),
        ]
        for j3, link_a, link_b, expected in recorded:
            with self.subTest(j3=j3, pair=f"{link_a}|{link_b}"):
                sample = self.ck.check_pose({"left_arm_joint3": j3})
                hits = [v for v in sample.violations if v.key.matches(link_a, link_b)]
                self.assertTrue(hits, f"expected {link_a} | {link_b} to violate")
                worst = min(h.distance for h in hits)
                self.assertAlmostEqual(
                    worst, expected, delta=RECORDED_TOLERANCE_M,
                    msg=(f"recorded {expected * 1000:.1f} mm, "
                         f"measured {worst * 1000:.3f} mm"),
                )

    def test_j3_minus_040_is_clear(self):
        sample = self.ck.check_pose({"left_arm_joint3": -0.40})
        self.assertTrue(sample.ok, [v.describe() for v in sample.violations])
        self.assertGreaterEqual(sample.min_distance, 0.0)

    def test_a_full_path_fails(self):
        """(a) -0.5624 -> -0.2624 must fail.

        `early_exit=False` because this is the incident ACCEPTANCE case: it asserts the
        depth the collision actually reached (-33.8 mm at the end of the sweep), which is
        a statement about the whole path. The live-loop default stops at the first
        confirmed violation, three samples earlier and 9 mm shallower -- the same verdict
        reached sooner, but not the same number. See TestSampleBudget.
        """
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624}, steps=33,
            early_exit=False,
        )
        self.assertFalse(rep.ok)
        self.assertTrue(rep.violations)
        self.assertLess(rep.min_distance, 0.0)

    def test_a_the_verdict_survives_every_sampling_mode(self):
        """The FAIL itself must not depend on the budget. Only the headline may.

        Capping and early exit are performance controls; if either could turn this
        rejection into an acceptance they would be safety controls, and wrong.
        """
        args = ({"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624})
        for label, kw in (
            ("live default", {}),
            ("uncapped", {"max_samples": None}),
            ("no early exit", {"early_exit": False}),
            ("full sweep", {"max_samples": None, "early_exit": False}),
            ("minimum cap", {"max_samples": 2, "steps": 2}),
        ):
            with self.subTest(mode=label):
                rep = self.ck.check_path(*args, **kw)
                self.assertFalse(rep.ok, rep.format())
                self.assertTrue(
                    any(
                        v.key.matches("left_arm_link5", "leg_link2")
                        for v in rep.violations
                    ),
                    rep.format(),
                )

    def test_b_offending_pair_is_link5_and_leg_link2(self):
        """(b) The headline pair must be the one that actually hit."""
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624}, steps=33
        )
        self.assertIsNotNone(rep.min_pair)
        self.assertTrue(
            rep.min_pair.key.matches("left_arm_link5", "leg_link2"),
            rep.min_pair.describe(),
        )
        self.assertTrue(
            rep.worst_violation_per_link_pair()[0].key.matches(
                "left_arm_link5", "leg_link2"
            )
        )
        first = rep.samples[rep.first_violation_index]
        self.assertTrue(
            any(v.key.matches("left_arm_link5", "leg_link2") for v in first.violations)
        )

    def test_b_report_localises_the_hit_along_the_path(self):
        """A report is a distance, a pair AND a place — not a boolean, not a count."""
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624},
            {"left_arm_joint3": -0.2624},
            steps=13,
            max_joint_step=None,
            early_exit=False,
        )
        self.assertEqual(rep.n_samples, 13)
        self.assertEqual(rep.min_sample_index, 12)
        self.assertAlmostEqual(rep.min_fraction, 1.0, places=9)
        # galbot_preview.py put the onset of contact at j3 = -0.3624, which on a
        # 13-sample path over this range is sample 8.
        # Was index 8 / j3 = -0.3624 when the floor was 0.0, i.e. the exact sample
        # where the links first touched. At DEFAULT_FLOOR_M the gate flags the
        # approach one sample EARLIER, at j3 = -0.3874 -- 25 mrad of warning before
        # contact rather than at it. That is the entire purpose of a clearance floor.
        self.assertEqual(rep.first_violation_index, 7)
        onset = -0.5624 + rep.first_violation_fraction * (-0.2624 - -0.5624)
        self.assertAlmostEqual(onset, -0.3874, delta=1e-4)

    def test_c_path_stopping_at_minus_040_passes_at_a_zero_floor(self):
        """(c) — explicitly at floor=0.0, which is no longer the default.

        DEFAULT_FLOOR_M is 10 mm now: a floor of zero means a pair violates only
        once the links already interpenetrate, which is contact detection rather
        than clearance. This case still pins the historical behaviour, so it now
        builds its own zero-floor checker instead of relying on the default.
        """
        ck = ClearanceChecker(floor=0.0)
        self.assertEqual(ck.floor, 0.0)
        rep = ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.40}, steps=33
        )
        self.assertTrue(rep.ok, rep.format())
        self.assertGreaterEqual(rep.min_distance, 0.0)

    def test_c2_the_same_path_is_REJECTED_at_the_default_floor(self):
        """The negative case at DEFAULT settings — the check my ledger demands.

        A path that merely gets close, never touching anything, must be rejected.
        Reproducing a known collision proves detection; rejecting a near-miss
        proves discrimination, and that is what a clearance gate is for.
        """
        near = self.ck.check_pose({"left_arm_joint2": HOME_QPOS["left_arm_joint2"] + 0.4})
        self.assertGreater(near.min_distance, 0.0, "must not be touching")
        self.assertLess(near.min_distance, self.ck.floor, "must be inside the floor")
        self.assertFalse(near.ok, "a near-miss at default settings must be rejected")

    def test_d_endpoints_clear_but_interior_is_not(self):
        """(d) The property the whole module exists to enforce.

        j3 = -0.5624 (home) is clear and j3 = +1.5 is clear, but the straight line
        between them drags left_arm_link5/6 through the leg column and then the left
        gripper through the right arm.
        """
        start = {"left_arm_joint3": -0.5624}
        end = {"left_arm_joint3": 1.5}

        self.assertTrue(self.ck.check_pose(start).ok, "start pose should be clear")
        self.assertTrue(self.ck.check_pose(end).ok, "end pose should be clear")

        # Full sweep: this case is about the SHAPE of the path -- clear ends, bad middle
        # -- so it has to look at the ends, which is exactly what early exit skips.
        rep = self.ck.check_path(
            start, end, steps=65, max_samples=None, early_exit=False
        )
        self.assertFalse(rep.ok, "the interior of the path must be caught")
        self.assertTrue(0.0 < rep.min_fraction < 1.0, "worst sample must be interior")
        self.assertTrue(rep.samples[0].ok)
        self.assertTrue(rep.samples[-1].ok)
        self.assertLess(rep.min_distance, -0.05)

        # A two-sample "just check the endpoints" run is exactly the check that missed it.
        endpoints_only = self.ck.check_path(start, end, steps=2, max_joint_step=None)
        self.assertTrue(
            endpoints_only.ok, "endpoint-only checking is the gap this module closes"
        )


class TestPathSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck = shared_checker()

    def test_undersampling_is_prevented_by_max_joint_step(self):
        """Refinement, with the cap lifted -- the rule as it stood before max_samples.

        `max_samples=None` restores unbounded refinement, and it must still produce the
        full 207 samples that 2.0624 rad at 0.01 rad/step demands.
        """
        args = ({"left_arm_joint3": -0.5624}, {"left_arm_joint3": 1.5})
        coarse = self.ck.check_path(*args, steps=2, max_joint_step=None)
        refined = self.ck.check_path(
            *args, steps=2, max_joint_step=0.01, max_samples=None, early_exit=False
        )
        self.assertEqual(coarse.n_samples, 2)
        self.assertTrue(coarse.ok)
        self.assertGreaterEqual(refined.n_samples, math.ceil(2.0624 / 0.01))
        self.assertFalse(refined.ok)
        # An uncapped sweep is the one that gets to call itself complete.
        self.assertFalse(refined.sampling_degraded)
        self.assertEqual(refined.max_joint_step_requested, 0.01)
        self.assertLessEqual(refined.max_joint_step_actual, 0.01)

    def test_steps_must_include_both_endpoints(self):
        with self.assertRaises(ValueError):
            self.ck.check_path({"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624},
                               steps=1)

    def test_every_sample_is_evaluated(self):
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624},
            steps=21, max_joint_step=None, early_exit=False,
        )
        self.assertEqual(len(rep.samples), 21)
        self.assertEqual(rep.samples_evaluated, 21)
        self.assertFalse(rep.early_exit)
        self.assertTrue(all(s.n_pairs_evaluated > 0 for s in rep.samples))
        self.assertEqual([s.index for s in rep.samples], list(range(21)))
        self.assertAlmostEqual(rep.samples[0].fraction, 0.0)
        self.assertAlmostEqual(rep.samples[-1].fraction, 1.0)
        # Interpolation is linear in joint space.
        mid = rep.samples[10].qpos["left_arm_joint3"]
        self.assertAlmostEqual(mid, (-0.5624 + -0.2624) / 2, places=9)
        # Joints that were not asked to move must not move.
        for s in rep.samples:
            self.assertAlmostEqual(s.qpos["leg_joint2"], HOME_QPOS["leg_joint2"], places=12)

    def test_module_level_check_path_matches_the_class(self):
        rep = check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624},
            steps=13, max_joint_step=None,
        )
        self.assertFalse(rep.ok)
        self.assertTrue(rep.min_pair.key.matches("left_arm_link5", "leg_link2"))


class TestSampleBudget(unittest.TestCase):
    """`max_samples` and `early_exit`: the 30 FPS bound, and what it costs to have it.

    Before this, `check_path` derived its sample count from joint travel with no
    ceiling, so cost scaled with how far the pose had jumped -- and a jump that far is
    usually a jump into collision. A live session measured 1149.6 ms on one frame
    against a 33 ms budget and dropped 159 frames. The rule was: the worse the pose,
    the longer the operator waited to hear about it.
    """

    @classmethod
    def setUpClass(cls):
        cls.ck = shared_checker()

    # -- the cap ------------------------------------------------------------------------

    def test_refinement_is_bounded(self):
        """2.0624 rad at 0.01 rad/step asks for 208 samples; the cap says 12."""
        args = ({"left_arm_joint3": -0.5624}, {"left_arm_joint3": 1.5})
        capped = self.ck.check_path(*args, steps=2, max_joint_step=0.01, max_samples=12)
        self.assertEqual(capped.n_samples, 12)
        self.assertEqual(capped.samples_needed, 208)
        self.assertEqual(capped.max_samples, 12)

    def test_the_cap_bounds_refinement_but_never_an_explicit_steps_request(self):
        """`steps=` is a deliberate resolution request and outranks the cap.

        Otherwise the offline auditor and the acceptance tests could not ask for a
        finer sweep than the live loop can afford, which is exactly what they are for.
        """
        args = ({"left_arm_joint3": -0.5624}, {"left_arm_joint3": 1.5})
        rep = self.ck.check_path(
            *args, steps=40, max_joint_step=0.01, max_samples=12, early_exit=False
        )
        self.assertEqual(rep.n_samples, 40)
        self.assertEqual(rep.samples_evaluated, 40)

    def test_a_short_move_is_not_capped_and_is_not_degraded(self):
        """The cap must be invisible on the ordinary frame-to-frame motion it exists for."""
        end = {"left_arm_joint3": HOME_QPOS["left_arm_joint3"] + 0.02}
        rep = self.ck.check_path(dict(HOME_QPOS), end, steps=2, max_joint_step=0.01)
        self.assertEqual(rep.n_samples, 4)
        self.assertEqual(rep.samples_needed, 4)
        self.assertFalse(rep.sampling_degraded)
        self.assertTrue(rep.ok)
        self.assertTrue(rep.pass_is_certified)
        self.assertLessEqual(rep.max_joint_step_actual, 0.01)

    def test_max_samples_none_restores_unbounded_refinement(self):
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": 1.5},
            steps=2, max_joint_step=0.01, max_samples=None, early_exit=False,
        )
        self.assertEqual(rep.n_samples, 208)
        self.assertIsNone(rep.max_samples)
        self.assertFalse(rep.sampling_degraded)

    def test_max_samples_below_two_is_refused(self):
        """Same rule as `steps`: a path check that cannot see both ends is not a check."""
        for bad in (0, 1, -5):
            with self.subTest(max_samples=bad):
                with self.assertRaises(ValueError):
                    self.ck.check_path(
                        {"left_arm_joint3": -0.5624}, {"left_arm_joint3": 1.5},
                        max_samples=bad,
                    )

    # -- what the degraded flag reports -------------------------------------------------

    def test_a_capped_sweep_reports_itself_as_degraded_with_the_numbers(self):
        """The cap may make the check coarser. It may not make it quiet.

        A coarser sweep steps over a longer arc between samples, so a thin collision
        living entirely between two samples is not looked at. Everything a caller needs
        to know that has to be on the report.
        """
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": 1.5},
            steps=2, max_joint_step=0.01, max_samples=12, early_exit=False,
        )
        self.assertTrue(rep.sampling_degraded)
        self.assertEqual(rep.max_joint_step_requested, 0.01)
        # 2.0624 rad spread over 11 intervals.
        self.assertAlmostEqual(rep.max_joint_step_actual, 2.0624 / 11, places=9)
        self.assertGreater(rep.max_joint_step_actual, rep.max_joint_step_requested)
        self.assertEqual(rep.samples_needed, 208)
        self.assertEqual(rep.n_samples, 12)

        # the numbers must be IN the reason string, not merely implied by it
        reason = rep.degraded_reason
        for token in ("0.0100", "208", "12", "0.1875"):
            self.assertIn(token, reason, reason)
        # and it must reach a caller who only reads the existing channels
        self.assertIn(reason, rep.limit_warnings)
        self.assertIn(reason, rep.format())
        # This one is a FAIL, and a FAIL from a degraded sweep is still conclusive --
        # coarse sampling can only miss violations, never invent them. So the loud
        # "not a certified clear" banner belongs to the PASS case, not here.
        self.assertNotIn("NOT A CERTIFIED CLEAR", rep.format())
        self.assertFalse(rep.pass_is_certified)

    def test_a_degraded_clear_is_not_a_certified_clear(self):
        """The flag has to bite on the one verdict it can weaken: a PASS.

        -0.5624 -> -0.40 is genuinely clear, and stays `ok`. But 18 samples were asked
        for and 12 were taken, so this is "nothing at the samples we could afford",
        not "nothing there", and it must not read as the latter.
        """
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.40},
            steps=2, max_joint_step=0.01, max_samples=12,
        )
        self.assertTrue(rep.ok)
        self.assertTrue(rep.sampling_degraded)
        self.assertFalse(rep.pass_is_certified, "a degraded clear must not certify")
        self.assertIn("NOT A CERTIFIED CLEAR", rep.format())

        full = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.40},
            steps=2, max_joint_step=0.01, max_samples=None,
        )
        self.assertTrue(full.ok)
        self.assertFalse(full.sampling_degraded)
        self.assertTrue(full.pass_is_certified)

    def test_not_degraded_when_the_cap_never_binds(self):
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.5324},
            steps=2, max_joint_step=0.01, max_samples=12,
        )
        self.assertFalse(rep.sampling_degraded)
        self.assertEqual(rep.degraded_reason, "")
        self.assertNotIn("DEGRADED", rep.format())

    def test_disabling_max_joint_step_is_not_degradation(self):
        """No refinement was asked for, so none was withheld."""
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": 1.5},
            steps=2, max_joint_step=None, max_samples=12,
        )
        self.assertEqual(rep.n_samples, 2)
        self.assertFalse(rep.sampling_degraded)
        self.assertIsNone(rep.max_joint_step_requested)

    # -- early exit ---------------------------------------------------------------------

    def test_early_exit_never_fires_without_a_confirmed_violation(self):
        """The property that keeps early exit from becoming a pass-by-not-looking.

        The loop breaks only on a non-empty `violations` list -- a positive finding.
        A clean sample can never end it, so a PASS always costs the full sample budget
        and can never be a PASS reached by stopping short.
        """
        rng = np.random.default_rng(20260810)
        names = list(HOME_QPOS)
        n_clear = n_exit = 0
        for _ in range(40):
            end = {
                n: HOME_QPOS[n] + float(rng.uniform(-0.8, 0.8))
                for n in rng.choice(names, size=3, replace=False)
            }
            rep = self.ck.check_path(dict(HOME_QPOS), end, steps=2, max_joint_step=0.01)
            if rep.early_exit:
                n_exit += 1
                self.assertFalse(rep.ok, "early exit on a report that passed")
                self.assertTrue(rep.violations)
                # it stopped AT the violation, so that is the last sample evaluated
                self.assertEqual(rep.first_violation_index, rep.samples_evaluated - 1)
            if rep.ok:
                n_clear += 1
                self.assertFalse(rep.early_exit)
                self.assertEqual(rep.samples_evaluated, rep.n_samples)
        self.assertGreater(n_exit, 0, "no early exit exercised")
        self.assertGreater(n_clear, 0, "no clear path exercised")

    def test_early_exit_changes_no_verdict_on_an_identical_grid(self):
        """Same samples, same answer. Only how many of them get run may differ."""
        rng = np.random.default_rng(4242)
        names = list(HOME_QPOS)
        for _ in range(25):
            end = {
                n: HOME_QPOS[n] + float(rng.uniform(-0.6, 0.6))
                for n in rng.choice(names, size=2, replace=False)
            }
            kw = dict(steps=9, max_joint_step=None)
            fast = self.ck.check_path(dict(HOME_QPOS), end, early_exit=True, **kw)
            full = self.ck.check_path(dict(HOME_QPOS), end, early_exit=False, **kw)
            self.assertEqual(fast.ok, full.ok, end)
            self.assertEqual(fast.first_violation_index, full.first_violation_index, end)
            self.assertLessEqual(fast.samples_evaluated, full.samples_evaluated)
            if full.ok:
                # a PASS must be bit-identical: nothing was skipped
                self.assertEqual(fast.samples_evaluated, full.samples_evaluated)
                self.assertEqual(fast.min_distance, full.min_distance)

    def test_early_exit_keeps_the_offending_pair_and_the_place(self):
        """What the report still has to carry after it stops early."""
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624},
            steps=33, max_joint_step=None,
        )
        self.assertTrue(rep.early_exit)
        self.assertFalse(rep.ok)
        self.assertIsNotNone(rep.first_violation_index)
        self.assertIsNotNone(rep.first_violation_fraction)
        self.assertTrue(rep.min_pair.key.matches("left_arm_link5", "leg_link2"))
        self.assertTrue(
            rep.worst_violation_per_link_pair()[0].key.matches(
                "left_arm_link5", "leg_link2"
            )
        )
        self.assertIsNotNone(rep.min_pair.fromto, "witness segment must survive")
        self.assertIn("early exit", rep.format())

    def test_early_exit_shallows_the_headline_and_says_so(self):
        """The one thing early exit really costs, stated as a number.

        Stopping at the first violation reports the clearance where the path was
        REJECTED, not the depth it would have reached had it been allowed to continue.
        For the incident sweep that is +8.9 mm (just inside the 10 mm floor) instead of
        -33.8 mm. Both reject; only the full sweep tells you how bad it gets.
        """
        args = ({"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624})
        kw = dict(steps=33, max_joint_step=None)
        fast = self.ck.check_path(*args, early_exit=True, **kw)
        full = self.ck.check_path(*args, early_exit=False, **kw)

        self.assertFalse(fast.ok)
        self.assertFalse(full.ok)
        self.assertAlmostEqual(full.min_distance, -0.0338, delta=RECORDED_TOLERANCE_M)
        self.assertAlmostEqual(fast.min_distance, 0.0089, delta=0.0005)
        self.assertGreater(fast.min_distance, full.min_distance)
        # the shallower number is the clearance at the moment of rejection, and it is
        # still a rejection: below the floor.
        self.assertLess(fast.min_distance, self.ck.floor)
        self.assertTrue(fast.early_exit)
        self.assertFalse(full.early_exit)

    # -- the budget itself --------------------------------------------------------------

    def test_worst_case_fits_the_30_fps_frame_budget(self):
        """The whole point. 33.3 ms a frame; the check must fit inside it.

        Timed as `supervisor.py` calls it. Median of five so a scheduling hiccup on a
        loaded machine cannot fail the run, but the threshold is the real budget.
        """
        budget_ms = 33.3
        cases = {
            "clear (small move)":
                {**HOME_QPOS, "left_arm_joint3": HOME_QPOS["left_arm_joint3"] + 0.02},
            "clear (long move)": {**HOME_QPOS, "left_arm_joint3": -0.40},
            "near-miss":
                {**HOME_QPOS, "left_arm_joint2": HOME_QPOS["left_arm_joint2"] + 0.4},
            "deep collision (logged -0.08 m)": {**HOME_QPOS, "left_arm_joint3": 0.05},
            "deep collision (longest travel)": {**HOME_QPOS, "left_arm_joint3": 1.5},
        }
        worst = 0.0
        for label, end in cases.items():
            self.ck.check_path(dict(HOME_QPOS), end, steps=2, max_joint_step=0.01)
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                self.ck.check_path(dict(HOME_QPOS), end, steps=2, max_joint_step=0.01)
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            worst = max(worst, times[2])
            with self.subTest(case=label):
                self.assertLess(
                    times[2], budget_ms,
                    f"{label}: {times[2]:.1f} ms exceeds the {budget_ms} ms frame budget",
                )
        self.assertLess(worst, budget_ms)

    def test_cost_no_longer_grows_with_how_bad_the_pose_is(self):
        """The inversion that was the actual bug.

        Uncapped, the deepest collision on the longest travel was the single most
        expensive check there was. Capped and exiting early it is now among the
        cheapest, because it is decided immediately.
        """
        clear_end = {**HOME_QPOS, "left_arm_joint3": -0.40}
        deep_end = {**HOME_QPOS, "left_arm_joint3": 1.5}

        def timed(end, **kw):
            self.ck.check_path(dict(HOME_QPOS), end, steps=2, max_joint_step=0.01, **kw)
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                self.ck.check_path(dict(HOME_QPOS), end, steps=2, max_joint_step=0.01, **kw)
                times.append((time.perf_counter() - t0) * 1000)
            return sorted(times)[2]

        unbounded = dict(max_samples=None, early_exit=False)
        self.assertGreater(
            timed(deep_end, **unbounded), timed(clear_end, **unbounded),
            "the old behaviour should be slower on the worse pose",
        )
        self.assertLess(
            timed(deep_end), timed(clear_end),
            "the deeply-colliding pose must now be the CHEAPER of the two",
        )


class TestBroadphaseAndLadder(unittest.TestCase):
    """The two per-pair costs: the bounding-sphere prefilter and the distance ladder."""

    @classmethod
    def setUpClass(cls):
        cls.ck = shared_checker()

    def test_broadphase_rejects_almost_everything_and_never_a_real_candidate(self):
        """`geom_rbound` separation is conservative: it can only drop proven-far pairs.

        It is also carrying most of the load -- without it every sample would run the
        ladder over all 13596 checked pairs instead of a few hundred. Both halves are
        checked here: that it discards a lot, and that everything it discards really is
        beyond `distmax`.
        """
        ck = self.ck
        rng = np.random.default_rng(31337)
        names = list(HOME_QPOS)
        counts = []
        for k in range(6):
            pose = dict(HOME_QPOS)
            if k:
                for n in rng.choice(names, size=4, replace=False):
                    pose[n] = HOME_QPOS[n] + float(rng.uniform(-0.9, 0.9))
            ck._apply(ck.full_pose(pose))

            kept = {tuple(p) for p in ck._candidate_pairs().tolist()}
            counts.append(len(kept))

            # every DISCARDED pair must really be farther than distmax
            xp, rb = ck.data.geom_xpos, ck.model.geom_rbound
            dropped = [tuple(p) for p in ck._pair_ids.tolist() if tuple(p) not in kept]
            self.assertTrue(dropped)
            for idx in rng.choice(len(dropped), size=40, replace=False):
                g1, g2 = dropped[int(idx)]
                sep = float(np.linalg.norm(xp[g1] - xp[g2])) - float(rb[g1]) - float(rb[g2])
                self.assertGreater(sep, ck.distmax, f"{g1},{g2} was dropped but is close")
                # and the narrowphase agrees it is outside the window
                self.assertGreaterEqual(
                    ck._geom_distance(g1, g2, ck.distmax), ck.distmax - 1e-12
                )
        self.assertLess(
            max(counts), ck.n_pairs * 0.15,
            f"broadphase kept {max(counts)} of {ck.n_pairs}; it is not earning its cost",
        )

    def test_top_rung_short_circuit_matches_the_full_ladder(self):
        """The fast path must be the same rule, not a cheaper one.

        `_evaluate_pair` returns as soon as the widest rung saturates. The reference is
        `_reconcile_ladder` with no precomputed value, which queries every rung and
        reconciles them exactly as the pre-optimisation code did. They must agree on
        every field, for every candidate pair, at every pose.
        """
        ck = self.ck
        rng = np.random.default_rng(20260807)
        names = list(HOME_QPOS)
        poses = [dict(HOME_QPOS)]
        for j3 in np.linspace(-0.5624, 1.5, 10):
            poses.append({**HOME_QPOS, "left_arm_joint3": float(j3)})
        for _ in range(20):
            p = dict(HOME_QPOS)
            for n in rng.choice(names, size=5, replace=False):
                p[n] = HOME_QPOS[n] + float(rng.uniform(-1.0, 1.0))
            poses.append(p)

        n_pairs = n_short = 0
        for pose in poses:
            ck._apply(ck.full_pose(pose))
            for g1, g2 in ck._candidate_pairs():
                g1, g2 = int(g1), int(g2)
                fast = ck._evaluate_pair(g1, g2)
                full = ck._reconcile_ladder(g1, g2)
                n_pairs += 1
                n_short += bool(fast[2])            # saturated == took the fast path
                self.assertEqual(fast, full, f"{ck._geom_name(g1)} | {ck._geom_name(g2)}")
        self.assertGreater(n_pairs, 5000, f"only compared {n_pairs} pairs")
        self.assertGreater(n_short, n_pairs * 0.4, "the fast path is barely being taken")

    def test_the_short_circuit_is_the_saturation_proof_rule_and_nothing_more(self):
        """It fires on saturation only. A measurement at the top rung goes the long way.

        The contradictory narrowphase from TestFailClosed saturates at the SMALLEST rung
        and claims contact at every wider one, so the top rung does not saturate and the
        full reconciliation still runs -- which is what keeps the suspect-pair reporting
        alive.
        """
        ck = ClearanceChecker(floor=0.0)
        g1, g2 = target_pair(ck, "left_arm_link5", "leg_link2")
        real = ck._geom_distance
        rung0 = ck.ladder[0]
        calls = []

        def contradictory(a, b, distmax):
            if {int(a), int(b)} == {g1, g2}:
                calls.append(distmax)
                return distmax if distmax <= rung0 else 0.0
            return real(a, b, distmax)

        ck._geom_distance = contradictory
        ck._apply(ck.full_pose({"left_arm_joint3": -0.40}))
        dist, exact, saturated, unevaluable, reason = ck._evaluate_pair(g1, g2)

        self.assertEqual(len(calls), len(ck.ladder), "every rung must have been queried")
        self.assertFalse(saturated)
        self.assertFalse(exact)
        self.assertFalse(unevaluable)
        self.assertAlmostEqual(dist, rung0, places=12)
        self.assertIn("below a proven lower bound", reason)

    def test_cached_names_match_the_model(self):
        """The name caches are an optimisation, so they must be indistinguishable."""
        ck = self.ck
        for g in range(ck.model.ngeom):
            self.assertEqual(
                ck._geom_name(g),
                mujoco.mj_id2name(ck.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}",
            )
            bid = int(ck.model.geom_bodyid[g])
            self.assertEqual(
                ck._body_name(g),
                mujoco.mj_id2name(ck.model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}",
            )
        # the pair-key cache must be an identity map, not just a lookup that happens to work
        g1, g2 = target_pair(ck, "left_arm_link5", "leg_link2")
        key = ck._make_key(g1, g2)
        self.assertIs(key, ck._make_key(g1, g2))
        self.assertEqual(ck._key_ids[key], (g1, g2))


class TestAdjacencyExclusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck = shared_checker()

    def test_adjacency_is_derived_from_the_kinematic_tree(self):
        """Recompute the rule independently and demand exact agreement."""
        m = self.ck.model
        weld = m.body_weldid

        def expected(g1: int, g2: int) -> bool:
            w1, w2 = int(weld[m.geom_bodyid[g1]]), int(weld[m.geom_bodyid[g2]])
            if w1 == w2:
                return True
            return (
                int(weld[m.body_parentid[w1]]) == w2
                or int(weld[m.body_parentid[w2]]) == w1
            )

        ids = [g for g in range(m.ngeom) if m.geom_group[g] == 3]
        kept = {tuple(p) for p in self.ck._pair_ids.tolist()}
        n_adjacent = 0
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                want = expected(a, b)
                self.assertEqual(self.ck.is_adjacent(a, b), want, f"{a},{b}")
                n_adjacent += want
                # THREE deliberate exclusion categories now: kinematic adjacency
                # (derived from the tree, asserted above), permanently embedded
                # bodies (an explicit measured list -- see EMBEDDED_BODY_PAIRS),
                # and rigid end-effector assemblies (a name rule, deliberately
                # kept separate so a structural guarantee never depends on a
                # string prefix).  Each is a different KIND of claim, so each
                # stays separately auditable rather than being folded into one.
                expect_kept = (
                    not want
                    and not self.ck.permanently_embedded(a, b)
                    and not self.ck.same_assembly(a, b)
                )
                self.assertEqual(
                    (a, b) in kept, expect_kept,
                    "kept must be exactly: not adjacent, not embedded, "
                    "not same-assembly",
                )
        self.assertEqual(n_adjacent, self.ck.n_adjacent_excluded)
        # Four categories must account for every pair exactly once: kept,
        # kinematically adjacent, permanently embedded, and same rigid
        # end-effector assembly.  16836 = 184 collision geoms choose 2.
        self.assertEqual(
            self.ck.n_pairs
            + self.ck.n_adjacent_excluded
            + self.ck._n_embedded_excluded
            + self.ck.n_assembly_excluded,
            len(ids) * (len(ids) - 1) // 2,
        )
        self.assertGreater(self.ck.n_assembly_excluded, 0)
        self.assertGreater(self.ck.n_adjacent_excluded, 0)
        self.assertGreater(self.ck._n_embedded_excluded, 0)

    def test_cases_a_hardcoded_list_would_get_wrong(self):
        def geoms_of(body: str):
            return [
                g
                for g in range(self.ck.model.ngeom)
                if self.ck.model.geom_group[g] == 3 and self.ck._body_name(g) == body
            ]

        def adjacent(a: str, b: str) -> bool:
            pairs = [(x, y) for x in geoms_of(a) for y in geoms_of(b)]
            self.assertTrue(pairs, f"no collision geoms for {a} / {b}")
            results = {self.ck.is_adjacent(x, y) for x, y in pairs}
            self.assertEqual(len(results), 1, f"{a}/{b} inconsistently adjacent")
            return results.pop()

        # One joint apart -> adjacent.
        self.assertTrue(adjacent("left_arm_link5", "left_arm_link6"))
        # leg_link1 hangs off leg_base_link, which is welded into the chassis: adjacent
        # even though the names share no prefix.
        self.assertTrue(adjacent("leg_link1", "omni_chassis_base_link"))
        self.assertTrue(adjacent("leg_link1", "wheel_1"))
        # torso_base_link is welded to leg_link5, so each arm root is one joint from it.
        self.assertTrue(adjacent("left_arm_link1", "torso_base_link"))
        # Two joints apart -> NOT adjacent, even though it is the same limb.
        self.assertFalse(adjacent("left_arm_link5", "left_arm_link7"))
        # The pair from the incident must be checked.
        self.assertFalse(adjacent("left_arm_link5", "leg_link2"))

    def test_rule_is_not_galbot_specific(self):
        """Same rule, a different model, compiled in memory."""
        xml = """
        <mujoco>
          <worldbody>
            <body name="a">
              <joint name="j1" type="hinge" axis="0 0 1"/>
              <geom name="ga" group="3" type="box" size=".1 .1 .1"/>
              <body name="fixed" pos="0 0 .3">
                <geom name="gf" group="3" type="box" size=".1 .1 .1"/>
                <body name="b" pos="0 0 .3">
                  <joint name="j2" type="hinge" axis="0 0 1"/>
                  <geom name="gb" group="3" type="box" size=".1 .1 .1"/>
                  <body name="c" pos="0 0 .3">
                    <joint name="j3" type="hinge" axis="0 0 1"/>
                    <geom name="gc" group="3" type="box" size=".1 .1 .1"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        ck = ClearanceChecker(model=model, home={"j1": 0.0, "j2": 0.0, "j3": 0.0})
        gid = {
            n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("ga", "gf", "gb", "gc")
        }
        # 'fixed' carries no joint, so it welds into 'a'.
        self.assertTrue(ck.is_adjacent(gid["ga"], gid["gf"]))
        self.assertTrue(ck.is_adjacent(gid["ga"], gid["gb"]))   # one joint (j2)
        self.assertTrue(ck.is_adjacent(gid["gf"], gid["gb"]))   # same weld unit as 'a'
        self.assertTrue(ck.is_adjacent(gid["gb"], gid["gc"]))   # one joint (j3)
        self.assertFalse(ck.is_adjacent(gid["ga"], gid["gc"]))  # two joints
        self.assertEqual(ck.n_pairs, 2)                         # ga|gc and gf|gc


class TestPermanentlyEmbeddedExclusion(unittest.TestCase):
    """The third exclusion category, and the evidence that earned it.

    ``head_link2 | torso_base_link`` is authored interpenetrating: the head's
    collision geom sits 73.671 mm inside the torso's at the home pose. That is
    not a collision the checker can ever usefully report, because the pair
    never separates -- so it can never transition from clear to touching, which
    is the only transition a clearance check exists to catch. What it CAN do is
    move, and the relative-worsening rule then converts that movement into
    false rejections: pitching the head chin-down deepens the overlap by up to
    13.583 mm (18.44% of the baseline), while ``BASELINE_RELATIVE_TOLERANCE``
    of 0.05 allowed only 3.68 mm. Any head pitch past +5.35 deg was rejected --
    a quarter of the joint's real 23.1 deg of travel -- for a collision that
    does not exist.

    The alternative was widening ``BASELINE_RELATIVE_TOLERANCE`` to 0.20, which
    would have loosened the rule for every pair in the model to accommodate one
    fictional overlap. Exclusion is the narrower fix, and these tests are what
    keep it narrow: exactly one body pair, exactly two geom pairs, and the
    invariance fact that makes the exclusion sound.
    """

    @classmethod
    def setUpClass(cls):
        cls.ck = shared_checker()
        cls.embedded = [
            (int(a), int(b))
            for a, b in itertools.combinations(cls.ck._geom_ids, 2)
            if cls.ck.permanently_embedded(int(a), int(b))
        ]

    def test_head_torso_is_the_one_and_only_embedded_pair(self):
        """Exactly one body pair, exactly two geom pairs. Nobody adds a third quietly.

        This is the whole point of an explicit list. A rule that inferred
        "embedded" from baseline depth would be a rule a bad calibration pose
        could switch off; a literal that a test pins by name and by count can
        only grow with the evidence in hand.
        """
        self.assertEqual(
            EMBEDDED_BODY_PAIRS,
            frozenset({frozenset({"head_link2", "torso_base_link"})}),
            "adding an entry here needs its own measurement and its own test",
        )
        self.assertEqual(len(EMBEDDED_BODY_PAIRS), 1)

        # torso_base_link carries two collision geoms and head_link2 one, so the
        # body-level rule removes exactly two geom pairs.
        self.assertEqual(self.ck._n_embedded_excluded, 2)
        self.assertEqual(len(self.embedded), 2)
        for a, b in self.embedded:
            self.assertEqual(
                {self.ck._body_name(a), self.ck._body_name(b)},
                {"head_link2", "torso_base_link"},
            )

        # Excluded means genuinely gone: not checked, and not carried as a
        # baseline either (which would still let it fail on worsening).
        kept = {tuple(p) for p in self.ck._pair_ids.tolist()}
        for pair in self.embedded:
            self.assertNotIn(pair, kept)
        self.assertFalse(
            [k for k in self.ck.baseline_pairs()
             if k.matches("head_link2", "torso_base_link")],
            "an embedded pair must not survive as a baseline overlap",
        )
        with self.assertRaises(AssertionError):
            target_pair(self.ck, "head_link2", "torso_base_link")

    def test_the_overlap_never_separates_anywhere_in_the_head_range(self):
        """Fact 2: negative at all 35 poses, -68.20 mm to -87.25 mm. Never zero-crossing.

        The reachable head range the retargeter drives is yaw
        -1.4308..+1.4308 and pitch -0.1243..+0.4036 rad. Across it the deep
        geom pairing is interpenetrating at every single pose, so there is no
        pose at which it could report an approach.

        The shallow pairing (``torso_base_link_collision_0``) is excluded too,
        because the rule is stated over BODIES rather than geoms. It costs
        nothing here: over the same range it never comes closer than 8.580 mm.
        At the raw URDF joint limits -- outside the soft-limit band the
        retargeter enforces -- it does close to +0.287 mm, which is the honest
        bound on what this exclusion gives up.
        """
        ck = self.ck
        deep = [p for p in self.embedded
                if ck._geom_name(p[0]).endswith("collision_1")
                or ck._geom_name(p[1]).endswith("collision_1")]
        self.assertEqual(len(deep), 1, "expected one deep geom pairing")
        shallow = [p for p in self.embedded if p not in deep]

        def sweep(pair, yaws, pitches):
            vals = []
            for y in yaws:
                for p in pitches:
                    ck._apply(ck.full_pose({"head_joint1": float(y),
                                            "head_joint2": float(p)}))
                    d, _exact, _sat, unevaluable, _r = ck._evaluate_pair(*pair)
                    self.assertFalse(unevaluable)
                    vals.append(d)
            return vals

        reachable_yaw = np.linspace(-1.4308, 1.4308, 7)
        reachable_pitch = np.linspace(-0.1243, 0.4036, 5)

        deep_vals = sweep(deep[0], reachable_yaw, reachable_pitch)
        self.assertEqual(len(deep_vals), 35)
        self.assertTrue(all(v < 0.0 for v in deep_vals),
                        f"it separates somewhere: max {max(deep_vals) * 1000:+.3f} mm")
        self.assertAlmostEqual(max(deep_vals), -0.06820, delta=0.0002)
        self.assertAlmostEqual(min(deep_vals), -0.08725, delta=0.0002)

        # The home-pose depth this whole argument starts from.
        ck._apply(ck.home)
        home_depth, _e, _s, _u, _r = ck._evaluate_pair(*deep[0])
        self.assertAlmostEqual(home_depth, -0.073671, delta=RECORDED_TOLERANCE_M)

        shallow_vals = sweep(shallow[0], reachable_yaw, reachable_pitch)
        self.assertGreater(min(shallow_vals), 0.008,
                           "the shallow pairing is not close over the driven range")

        lo1, hi1 = self.ck.model.jnt_range[self.ck._joint_id("head_joint1")]
        lo2, hi2 = self.ck.model.jnt_range[self.ck._joint_id("head_joint2")]
        raw_shallow = sweep(shallow[0], np.linspace(lo1, hi1, 9), np.linspace(lo2, hi2, 7))
        self.assertAlmostEqual(min(raw_shallow), 0.000287, delta=RECORDED_TOLERANCE_M)

    def test_separation_is_a_function_of_the_head_joints_alone(self):
        """Fact 3 -- the load-bearing one. Nothing but the head can move this pair.

        The exclusion is only sound if the pair's geometry cannot be disturbed
        by anything the operator commands elsewhere. If, say, a leg lift could
        drive the head into the torso, then switching the check off would be
        hiding a real signal.

        All 31 non-head hinge/slide joints, driven to their lower limit, upper
        limit and midpoint (93 poses), move the separation by less than 1e-9 m.
        Measured worst deviation is 3.4e-16 m -- floating-point noise, three
        orders of magnitude below the tightest tolerance in this module. The
        head's genuine constraint is its URDF joint limit, which the retargeter
        enforces through its soft-limit band, not this checker.
        """
        ck = self.ck
        m = ck.model
        non_head = [
            name
            for j in range(m.njnt)
            for name in [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            if name
            and not name.startswith("head_")
            and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                       int(mujoco.mjtJoint.mjJNT_SLIDE))
        ]
        self.assertEqual(len(non_head), 31, non_head)

        ck._apply(ck.home)
        reference = {pair: ck._evaluate_pair(*pair)[0] for pair in self.embedded}

        worst = 0.0
        worst_at = ""
        for name in non_head:
            jid = ck._joint_id(name)
            lo, hi = (m.jnt_range[jid] if m.jnt_limited[jid] else (-1.0, 1.0))
            for value in (float(lo), float(hi), float((lo + hi) / 2.0)):
                ck._apply(ck.full_pose({name: value}))
                for pair in self.embedded:
                    d, _e, _s, unevaluable, _r = ck._evaluate_pair(*pair)
                    self.assertFalse(unevaluable, f"{name}={value}")
                    deviation = abs(d - reference[pair])
                    if deviation > worst:
                        worst, worst_at = deviation, f"{name}={value:+.4f}"
        self.assertLess(
            worst, 1e-9,
            f"{worst_at} moved the head/torso separation by {worst:.3e} m -- the "
            "exclusion assumes only the head joints can",
        )


class TestBaselineOverlaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck = shared_checker()

    def test_baseline_at_home_captures_every_structurally_close_pair(self):
        """The baseline is every pair below `floor` at home, not only the overlaps.

        Updated when the default floor moved from 0.0 to DEFAULT_FLOOR_M. At a floor
        of zero only genuinely interpenetrating pairs were captured -- one -- and
        every other pair was compared against contact rather than clearance, which
        is the semantics this module exists to replace. At 10 mm the model's
        permanently-close structure is captured and worsening-checked instead.

        Updated again when ``head_link2 | torso_base_link`` left the baseline set
        for ``EMBEDDED_BODY_PAIRS``. It was the only DEEP baseline (-73.671 mm),
        and it was fictional -- authored interpenetration that never separates at
        any reachable head pose. What remains is four geom pairs across two link
        pairs, all near-touching rather than overlapping, and all of them pairs
        that genuinely open and close as the wrist moves:

            right_arm_link5 | right_arm_link7   +1.171 mm, +1.412 mm
            left_arm_link5  | left_arm_link7    +1.664 mm, +1.731 mm

        Every one is positive, i.e. real clearance being tracked relatively
        instead of against a floor it would trip on permanently.
        """
        base = self.ck.baseline_pairs()
        self.assertGreater(len(base), 1)
        self.assertLess(len(base), 40, "an implausible number of structural pairs")
        self.assertEqual(len(base), 4, {str(k): v for k, v in base.items()})

        by_links = sorted(
            (round(v, 6), tuple(sorted(k.link_pair))) for k, v in base.items()
        )
        self.assertEqual(
            [links for _v, links in by_links],
            [("right_arm_link5", "right_arm_link7"),
             ("right_arm_link5", "right_arm_link7"),
             ("left_arm_link5", "left_arm_link7"),
             ("left_arm_link5", "left_arm_link7")],
        )
        for measured, expected in zip(
            [v for v, _links in by_links], [0.001171, 0.001412, 0.001664, 0.001731]
        ):
            self.assertAlmostEqual(measured, expected, delta=RECORDED_TOLERANCE_M)

        # Every survivor is near-touching, not overlapping: the deep one is gone.
        self.assertGreater(min(base.values()), 0.0)
        self.assertFalse(
            [k for k in base if k.matches("head_link2", "torso_base_link")],
            "the authored head/torso interpenetration is excluded, not baselined",
        )
        self.assertFalse(self.ck.baseline_unevaluable)

    def test_same_assembly_exclusion_never_hides_a_penetration(self):
        """Justifies excluding intra-gripper sibling pairs from the verdict.

        A gripper's knuckle/finger links are children of the same base, so weld
        adjacency misses them, and they are the closest pair in the model at almost
        every pose. Their relative geometry is a function of the gripper drive joint
        alone, and `mj_geomDistance` returns lower bounds for them that vary by ~2 mm
        under arm motion -- noise the baseline worsening check reports as a violation.

        Excluding them is only sound if they never genuinely penetrate. Swept across
        the arm's range, no pair excluded *solely* by same_assembly ever goes
        negative. The handful that read exactly 0.000 are the documented
        mj_geomDistance spurious-zero mode (docs/model-notes.md 6.5), not contact.
        """
        import itertools
        ck = self.ck
        only_assembly = [
            (int(a), int(b))
            for a, b in itertools.combinations(ck._geom_ids, 2)
            if ck.same_assembly(int(a), int(b))
            and int(ck.model.body_weldid[ck.model.geom_bodyid[int(a)]])
            != int(ck.model.body_weldid[ck.model.geom_bodyid[int(b)]])
        ]
        self.assertGreater(len(only_assembly), 100)
        worst = 9.9
        for delta in (-1.0, -0.5, 0.0, 0.5, 1.0):
            q = dict(HOME_QPOS)
            q["left_arm_joint2"] = q["left_arm_joint2"] + delta
            ck._apply(q)
            for a, b in only_assembly:
                dist, _e, _s, unevaluable, _w = ck._evaluate_pair(a, b)
                if not unevaluable:
                    worst = min(worst, dist)
        self.assertGreaterEqual(
            worst, 0.0,
            f"an assembly-excluded pair reached {worst:.6f} m -- exclusion is unsafe",
        )

    def test_home_pose_itself_is_clear(self):
        """Whatever the baseline is, the pose it was measured from must pass."""
        self.assertTrue(self.ck.check_pose().ok)
        self.assertTrue(self.ck.check_pose(dict(HOME_QPOS)).ok)

    def test_baseline_pair_is_not_whitelisted_and_fails_when_it_worsens(self):
        """Recording a pair as baseline must never mean "stop checking it".

        This test exists to stop the easy fix: when a structurally-close pair
        becomes inconvenient, whitelist it and the suite goes green. So it takes
        a pair that IS in the baseline set, moves the joints that own it until
        the measured separation degrades, and demands a `baseline_worsened`
        violation.

        Re-anchored from ``head_link2 | torso_base_link``, which is no longer a
        baseline pair at all -- it is excluded outright by
        ``EMBEDDED_BODY_PAIRS`` because it never separates at any reachable head
        pose, which is a different claim from "it is forgiven when it worsens".
        See ``TestPermanentlyEmbeddedExclusion`` for the evidence behind that,
        including the counts that stop a second pair being added quietly.

        The replacement is ``left_arm_link5 | left_arm_link7``, which genuinely
        can separate: its geometry is a function of ``left_arm_joint6`` and
        ``left_arm_joint7`` alone, and over a 41x61 grid of their full ranges the
        measured separation moves between +0.99 mm and +2.29 mm. Rolling the
        wrist to (j6 = -0.5111, j7 = 0.0) takes the two recorded geom pairs from
        +1.731 mm to +1.003 mm (-0.728 mm) and from +1.664 mm to +0.997 mm
        (-0.667 mm), both exact measurements, with the pose otherwise clear.

        A tightened `baseline_margin` is used because these pairs are shallow:
        their worsening tolerance is ``max(baseline_margin, 5% of 1.7 mm)``,
        which at the 2 mm default is the 2 mm default, and 0.7 mm of real
        movement is deliberately inside it. That is a documented, measured
        allowance -- so the second half of this test pins the thing that makes
        it an allowance rather than a whitelist: at the default margin the pair
        is still evaluated and still reported, as `worst_baseline`.
        """
        POSE = {"left_arm_joint6": -0.5111, "left_arm_joint7": 0.0}
        strict = ClearanceChecker(baseline_margin=0.0002)
        worse = strict.check_pose(POSE)
        hits = [
            v for v in worse.violations
            if v.baseline is not None
            and v.key.matches("left_arm_link5", "left_arm_link7")
        ]
        self.assertTrue(hits, "a baseline pair getting closer must still fail")
        self.assertEqual(len(worse.violations), len(hits),
                         "nothing else about this pose is supposed to be violating")
        deepest = min(hits, key=lambda v: v.baseline_delta)
        self.assertEqual(deepest.violation_kind, "baseline_worsened")
        self.assertAlmostEqual(deepest.baseline, 0.001731, delta=RECORDED_TOLERANCE_M)
        self.assertAlmostEqual(deepest.distance, 0.001003, delta=RECORDED_TOLERANCE_M)
        self.assertTrue(deepest.exact, "this must be a measurement, not a lower bound")
        self.assertLess(deepest.baseline_delta, -strict.baseline_margin)

        # And the pair is not silently dropped at the default margin either: the
        # same 0.728 mm is inside the documented tolerance, but it is still
        # measured, still attributed and still surfaced.
        default = self.ck.check_pose(POSE)
        self.assertTrue(default.ok)
        self.assertIsNotNone(default.worst_baseline)
        self.assertTrue(
            default.worst_baseline.key.matches("left_arm_link5", "left_arm_link7")
        )
        self.assertAlmostEqual(
            default.worst_baseline.baseline_delta, -0.000728, delta=RECORDED_TOLERANCE_M
        )

    def test_baseline_pair_passes_when_it_improves_or_barely_moves(self):
        for head2 in (-0.2143, -0.0570, 0.0217):
            with self.subTest(head_joint2=head2):
                sample = self.ck.check_pose({"head_joint2": head2})
                hits = [
                    v for v in sample.violations
                    if v.key.matches("head_link2", "torso_base_link")
                ]
                self.assertFalse(hits, [h.describe() for h in hits])

    def test_baseline_margin_is_configurable(self):
        """The same pose, three margins: the threshold is the knob, not the pair.

        Re-anchored from ``head_joint2 = 0.1003`` (which moved the now-excluded
        head/torso pair) onto the wrist roll that moves ``left_arm_link5 |
        left_arm_link7``. Measured delta is -0.728 mm, so a 0.2 mm margin
        rejects it, the 2 mm default absorbs it, and a 20 mm margin would
        absorb almost anything -- which is exactly the spread the knob is for.
        """
        pose = {"left_arm_joint6": -0.5111, "left_arm_joint7": 0.0}
        tight = ClearanceChecker(baseline_margin=0.0002)
        loose = ClearanceChecker(baseline_margin=0.020)
        hit = lambda c: any(  # noqa: E731
            v.baseline is not None and v.key.matches("left_arm_link5", "left_arm_link7")
            for v in c.check_pose(pose).violations
        )
        self.assertTrue(hit(tight))
        self.assertFalse(hit(loose))
        self.assertFalse(hit(self.ck), "the 2 mm default sits between the two")

    def test_baseline_pair_is_excluded_from_the_headline_minimum(self):
        """The model's closest geometry is structural, and must not be the headline.

        Re-anchored: the worked example used to be the -73.7 mm head/torso
        overlap masking a 5 mm arm strike. That pair is now excluded outright
        (``EMBEDDED_BODY_PAIRS``), and no baseline pair is deeper than an arm
        strike any more -- the four survivors sit at +1.171 to +1.731 mm. The
        property they still demonstrate is the same one: a baseline pair is
        tracked separately and never becomes `min_distance`.

        Two poses make the point together. At the incident pose the headline is
        the arm strike, not one of the near-touching structural pairs. At home
        the structural pairs ARE the closest geometry in the entire model --
        1.171 mm against a headline of 12.182 mm, ten times further away -- and
        the headline still ignores them, which is the only reason the home pose
        reads as clear at a 10 mm floor at all.
        """
        sample = self.ck.check_pose({"left_arm_joint3": -0.35})
        self.assertAlmostEqual(sample.min_distance, -0.0051, delta=0.001)
        self.assertIsNone(sample.min_pair.baseline, "headline pair must not be baseline")
        self.assertIsNotNone(sample.worst_baseline)
        self.assertTrue(
            sample.worst_baseline.key.matches("right_arm_link5", "right_arm_link7")
            or sample.worst_baseline.key.matches("left_arm_link5", "left_arm_link7")
        )

        home = self.ck.check_pose()
        closest_baseline = min(self.ck.baseline.values())
        self.assertLess(closest_baseline, self.ck.floor,
                        "a baseline pair is by definition below the floor")
        self.assertAlmostEqual(home.min_distance, 0.012182, delta=RECORDED_TOLERANCE_M)
        self.assertIsNone(home.min_pair.baseline)
        self.assertIsNotNone(home.worst_baseline)
        self.assertLess(closest_baseline, home.min_distance,
                        "a baseline pair IS closer -- and is correctly not the headline")
        self.assertTrue(home.ok, "which is why the pose the baseline came from is clear")


class TestFailClosed(unittest.TestCase):
    def test_unevaluable_pair_is_a_violation_not_a_pass(self):
        def boom(*_a):
            raise RuntimeError("ccd exploded")

        for label, failure in (
            ("raises", boom),
            ("nan", lambda *a: float("nan")),
            ("inf", lambda *a: float("inf")),
        ):
            with self.subTest(failure=label):
                ck = ClearanceChecker()
                self.assertTrue(
                    ck.check_pose({"left_arm_joint3": -0.40}).ok, "control: pose is clear"
                )
                g1, g2 = target_pair(ck, "left_arm_link5", "leg_link2")
                real = ck._geom_distance

                def flaky(a, b, distmax, _g=(g1, g2), _f=failure, _r=real):
                    if {int(a), int(b)} == set(_g):
                        return _f(a, b, distmax)
                    return _r(a, b, distmax)

                ck._geom_distance = flaky
                sample = ck.check_pose({"left_arm_joint3": -0.40})

                self.assertFalse(sample.ok, "an unevaluable pair must never pass")
                broken = [
                    v for v in sample.violations
                    if v.key.matches("left_arm_link5", "leg_link2")
                ]
                self.assertTrue(broken)
                self.assertEqual(broken[0].violation_kind, "unevaluable")
                self.assertTrue(broken[0].unevaluable)
                self.assertTrue(broken[0].reason)

    def test_unevaluable_at_the_baseline_pose_poisons_the_pair_forever(self):
        """If we never learned what normal looks like, we can never certify it."""
        g1, g2 = target_pair(shared_checker(), "left_arm_link5", "leg_link2")
        real = ClearanceChecker._geom_distance

        def flaky(self, a, b, distmax):
            if {int(a), int(b)} == {g1, g2}:
                raise RuntimeError("ccd exploded")
            return real(self, a, b, distmax)

        ClearanceChecker._geom_distance = flaky
        try:
            ck = ClearanceChecker()
            self.assertIn((g1, g2), ck.baseline_unevaluable)
        finally:
            ClearanceChecker._geom_distance = real

        # The solver works again, but the pair is still refused: its baseline was never
        # established, so "not worse than baseline" is unanswerable.
        sample = ck.check_pose({"left_arm_joint3": -0.40})
        poisoned = [
            v for v in sample.violations if v.key.matches("left_arm_link5", "leg_link2")
        ]
        self.assertTrue(poisoned)
        self.assertEqual(poisoned[0].violation_kind, "unevaluable")
        self.assertFalse(sample.ok)

    def _contradictory_checker(self, floor: float) -> ClearanceChecker:
        """A checker whose narrowphase reproduces the measured spurious-zero signature.

        It saturates at the smallest ladder rung (proving distance > 1.5625 mm) and then
        claims contact at every wider one.  The proven lower bound is 1.5625 mm.
        """
        ck = ClearanceChecker(floor=floor)
        g1, g2 = target_pair(ck, "left_arm_link5", "leg_link2")
        real = ck._geom_distance
        rung0 = ck.ladder[0]

        def contradictory(a, b, distmax):
            if {int(a), int(b)} == {g1, g2}:
                return distmax if distmax <= rung0 else 0.0
            return real(a, b, distmax)

        ck._geom_distance = contradictory
        return ck

    def test_lower_bound_below_the_floor_fails_closed(self):
        ck = self._contradictory_checker(floor=0.004)
        sample = ck.check_pose({"left_arm_joint3": -0.40})
        hits = [v for v in sample.violations if v.key.matches("left_arm_link5", "leg_link2")]
        self.assertTrue(hits, "a lower bound under the floor cannot be certified")
        self.assertEqual(hits[0].violation_kind, "floor")
        self.assertFalse(hits[0].exact)
        self.assertAlmostEqual(hits[0].distance, ck.ladder[0], places=12)

    def test_lower_bound_above_the_floor_passes_but_is_marked_inexact(self):
        ck = self._contradictory_checker(floor=0.0)
        sample = ck.check_pose({"left_arm_joint3": -0.40})
        self.assertTrue(sample.ok)
        suspects = [
            p for p in sample.suspect_pairs
            if p.key.matches("left_arm_link5", "leg_link2")
        ]
        self.assertTrue(suspects, "the contradiction must be surfaced, not swallowed")
        self.assertFalse(suspects[0].exact)
        self.assertAlmostEqual(suspects[0].distance, ck.ladder[0], places=12)

    def test_unknown_joint_names_are_refused_not_ignored(self):
        ck = shared_checker()
        with self.assertRaises(ValueError):
            ck.check_pose({"left_arm_joint8": 0.0})
        with self.assertRaises(ValueError):
            ck.check_path({"lefT_arm_joint3": -0.5}, {"left_arm_joint3": -0.3})

    def test_floor_at_or_above_distmax_is_refused(self):
        """A floor beyond the search window could never be verified, so do not pretend."""
        with self.assertRaises(ValueError):
            ClearanceChecker(floor=0.10, distmax=0.05)

    def test_no_false_clear_against_the_contact_solver(self):
        """The safety-critical direction, checked against MuJoCo's own contact pipeline.

        Ground truth is the contype-patched model (patched in memory).  Every
        non-adjacent contact it finds must show up here as a non-positive distance.
        """
        truth_model = patched_contact_model()
        truth = mujoco.MjData(truth_model)
        ck = shared_checker()

        rng = np.random.default_rng(20260807)
        names = [
            mujoco.mj_id2name(ck.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(ck.model.njnt)
        ]
        poses = [dict(HOME_QPOS)]
        for j3 in np.linspace(-0.5624, -0.2624, 5):
            p = dict(HOME_QPOS)
            p["left_arm_joint3"] = float(j3)
            poses.append(p)
        for _ in range(25):
            poses.append(
                {n: float(rng.uniform(*ck.model.jnt_range[j])) for j, n in enumerate(names)}
            )

        n_checked = 0
        for pose in poses:
            truth.qpos[:] = 0.0
            for name, val in pose.items():
                jid = mujoco.mj_name2id(truth_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                truth.qpos[truth_model.jnt_qposadr[jid]] = val
            mujoco.mj_forward(truth_model, truth)

            expected = {}
            for i in range(truth.ncon):
                c = truth.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if truth_model.geom_group[g1] != 3 or truth_model.geom_group[g2] != 3:
                    continue
                if ck.is_adjacent(g1, g2):
                    continue
                key = (min(g1, g2), max(g1, g2))
                expected[key] = min(expected.get(key, 9.0), float(c.dist))
            if not expected:
                continue

            ck._apply(ck.full_pose(pose))
            for (g1, g2), contact_dist in expected.items():
                dist, exact, _sat, unevaluable, _why = ck._evaluate_pair(g1, g2)
                n_checked += 1
                self.assertFalse(unevaluable)
                self.assertLessEqual(
                    dist, 0.0,
                    f"FALSE CLEAR: contact at {contact_dist:.6f} m reported as "
                    f"{dist:.6f} m for {ck._geom_name(g1)} | {ck._geom_name(g2)}",
                )
                if exact:
                    # 1 um.  The two paths run the same narrowphase with different
                    # margins, so they differ by iteration noise of order 1e-9 m.
                    self.assertAlmostEqual(dist, contact_dist, delta=1e-6)
        self.assertGreater(
            n_checked, 150,
            f"only cross-checked {n_checked} pairs",
        )  # lowered from 200 when same-assembly exclusion removed 1398 rigid
        # intra-gripper pairs from the candidate set. Safe: proven by
        # test_same_assembly_exclusion_never_hides_a_penetration.


class TestToolLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck = shared_checker()

    def test_tool_links_are_identified_by_pattern(self):
        tool_bodies, plain_bodies = set(), set()
        for g in self.ck._geom_ids:
            target = tool_bodies if self.ck._is_tool_geom[int(g)] else plain_bodies
            target.add(self.ck._body_name(int(g)))

        for name in (
            "left_gripper_base_link",
            "left_gripper_l_finger_link",
            "right_gripper_r_knuckle_link",
        ):
            self.assertIn(name, tool_bodies)
        for name in ("left_arm_link5", "leg_link2", "head_link2", "torso_base_link"):
            self.assertIn(name, plain_bodies)
        self.assertTrue(
            all("gripper" not in b and "finger" not in b for b in plain_bodies)
        )

    def test_tool_pairs_are_reported_separately(self):
        """Requirement 7: the MJCF gripper is not the hardware gripper.

        Full sweep: the tool strike and the structural strike happen at different points
        along this path, so partitioning the violations means having them all.
        """
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": 1.5}, steps=65,
            max_samples=None, early_exit=False,
        )
        self.assertTrue(rep.tool_violations, "this path puts the left gripper in the arm")
        self.assertTrue(rep.structural_violations)
        self.assertEqual(
            len(rep.violations), len(rep.tool_violations) + len(rep.structural_violations)
        )
        self.assertTrue(all(v.is_tool for v in rep.tool_violations))
        self.assertTrue(all(not v.is_tool for v in rep.structural_violations))
        self.assertIsNotNone(rep.tool_min_pair)
        self.assertTrue(rep.tool_min_pair.is_tool)

    def test_the_incident_pair_is_not_a_tool_pair(self):
        """left_arm_link5 | leg_link2 is real structure and must not be discounted."""
        rep = self.ck.check_path(
            {"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624}, steps=33
        )
        self.assertFalse(rep.min_pair.is_tool)
        self.assertTrue(
            any(
                v.key.matches("left_arm_link5", "leg_link2")
                for v in rep.structural_violations
            )
        )


class TestCLI(unittest.TestCase):
    @staticmethod
    def run_cli(*args):
        # Invoke as a package module. clearance.py imports from
        # galbot_motion_studio.model, so running the file directly leaves the
        # package off sys.path -- `-m` with PYTHONPATH=src is the correct form
        # and is what the README documents.
        env = dict(os.environ, PYTHONPATH=str(SRC_DIR))
        return subprocess.run(
            [sys.executable, "-m", "galbot_motion_studio.safety.clearance", *args],
            capture_output=True, text=True, timeout=900, env=env,
        )

    def test_exits_nonzero_on_violation(self):
        """Requirement 8."""
        proc = self.run_cli("--sweep", "left_arm_joint3", "-0.5624", "-0.2624",
                            "--steps", "13")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("left_arm_link5", proc.stdout)
        self.assertIn("leg_link2", proc.stdout)
        self.assertIn("DO NOT SEND THIS PATH TO THE ROBOT", proc.stdout)

    def test_exits_zero_when_clear(self):
        proc = self.run_cli("--sweep", "left_arm_joint3", "-0.5624", "-0.40",
                            "--steps", "13")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("clear", proc.stdout)

    def test_reports_a_distance_not_a_boolean(self):
        proc = self.run_cli("--pose", "left_arm_joint3=-0.35")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("minimum clearance", proc.stdout)
        self.assertIn("mm", proc.stdout)

    def test_rejects_a_bad_joint_name(self):
        proc = self.run_cli("--pose", "left_arm_joint9=0.0")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("not in", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
