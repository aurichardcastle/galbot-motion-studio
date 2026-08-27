#!/usr/bin/env python3
"""
clearance.py — minimum-CLEARANCE checking for the Galbot G1 MuJoCo model.

Why this exists
---------------
`tools/galbot_preview.py` answers "is this pose in contact?".  That is a boolean.
A pose that misses the leg column by 2 mm and a pose that misses it by half a metre
both report "0 collisions", and both look equally safe to the operator.

On 2026-08-07 that gap let a left-arm sweep put a gripper 5.2 cm from the robot's own
head, and drove `left_arm_link5` into `leg_link2`.  The start pose was clear.  The end
pose was clear.  The middle was not, and nothing was checking the middle.

This module reports DISTANCE, not contact, and it checks the whole path:

    from galbot_motion_studio.safety.clearance import check_path, HOME_QPOS
    rep = check_path({"left_arm_joint3": -0.5624}, {"left_arm_joint3": -0.2624}, steps=33)
    rep.ok              # False
    rep.min_distance    # -0.0338  (metres; negative == penetration)
    rep.min_pair        # left_arm_link5 | leg_link2
    rep.min_fraction    # where along the path the worst sample was

Unspecified joints default to the 2026-08-07 18:06 hardware snapshot (`HOME_QPOS`).


What was learned about this model (measured, not assumed)
---------------------------------------------------------
1. `mj_geomDistance` does NOT need the contype patch.
   `galbot_preview.py` has to rewrite the MJCF on disk (contype 0 -> 1) because the
   contact pipeline prunes pairs in a compile-time broadphase.  `mj_geomDistance` calls
   the narrowphase directly and ignores contype/conaffinity entirely.  Verified: the
   shipped, unpatched `galbot_one_golf_fixed_base.xml` gives byte-identical distances to
   the patched copy (e.g. left_arm_link5 | leg_link2 = -33.76 mm at j3 = -0.2624 in both).
   So this module loads the shipped XML read-only and writes nothing to disk.

   (Aside for whoever maintains galbot_preview.py: on mujoco 3.11 the on-disk patch is no
   longer necessary even for the CONTACT approach.  `mujoco.MjSpec.from_file(xml)`, set
   `geom.contype = 1` on the group-3 geoms, then `spec.compile()` reproduces the patched
   model in memory — same 12 contacts at j3 = -0.2624.  Not changing that tool here.)

2. `mj_geomDistance` has a spurious-zero failure mode on mesh/mesh pairs.
   When `distmax` is large relative to the geometry, the convex narrowphase sometimes
   collapses and returns exactly 0.0 for geoms that are demonstrably far apart.  Measured
   example at the home pose: `leg_link1_collision_2` vs `left_gripper_r_knuckle_link_collision_2`,
   whose bounding spheres are 263 mm apart, returns

       distmax=0.10 -> 0.10   (correct: "farther than 0.10")
       distmax=0.50 -> 0.00   (wrong, plus a nonsense 0.4 m witness segment)

   The threshold is pair-dependent (some pairs break at distmax=0.01).  The error is
   always in the same direction — a spurious 0.0, i.e. a false alarm, never a false clear
   (checked against the contact solver over 3535 real contact pairs across 134 poses:
   zero false clears).  `_evaluate_pair` defends against it with a ladder of increasing
   `distmax` queries and the API's own contract, `f(distmax) == min(true, distmax)`:
   a rung that saturates is a proof that the true distance exceeds that rung, so a later
   rung that returns less than a proven lower bound is rejected and the lower bound is
   kept instead.  Such pairs are reported with `exact=False` and listed in
   `report.suspect_pairs`.

3. The model overlaps itself at every pose.
   With structurally adjacent pairs excluded, exactly one overlap survives at the home
   pose: `head_link2 | torso_base_link` at -73.7 mm.  It is recorded as a baseline, but
   it is NOT whitelisted — it still fails if it gets deeper than baseline by more than
   `baseline_margin`.

Cost, and why it is bounded
---------------------------
`check_path` runs inside a 30 FPS teleop loop, where the whole frame budget is 33 ms.
Until 2026-08-10 its cost was unbounded and, worse, *proportional to how bad the pose
was*: `max_joint_step` refinement derived the sample count from the joint travel with no
ceiling, so a large joint delta produced hundreds of samples and every one of them swept
the geom pairs through a 7-rung distance ladder.  A live session measured 1149.6 ms on a
single deeply-colliding frame (clearance -81.6 mm) and dropped 159 frames.  Cost scaling
with badness is exactly backwards: the worse the pose, the longer the operator waited to
be told about it.

Three bounds now apply, in the order they bite:

1. `max_samples` (default `DEFAULT_MAX_SAMPLES`) caps how far `max_joint_step` may
   inflate the sample count.  It does NOT override an explicit `steps=` request — that is
   the caller deliberately asking for a resolution.  When the cap forces a step coarser
   than `max_joint_step` asked for, `report.sampling_degraded` is True, the shortfall is
   spelled out in `report.degraded_reason` and pushed into `report.limit_warnings`, and
   `report.pass_is_certified` is False.  A coarser sweep can step over a thin collision,
   so a PASS from a degraded sweep is "nothing at the samples we could afford", not
   "nothing there".  It is never silently treated as equivalent.
2. `early_exit` (default True) stops sampling at the first sample with a CONFIRMED
   violation.  The verdict is already decided at that point and no later sample can undo
   it, so the remaining samples are pure cost.  This only ever shortens a FAIL; a PASS
   still evaluates every planned sample, so it can never become a pass-by-not-looking.
   The price is that `min_distance`/`min_pair` then describe the path only up to the
   first violation — flagged as `report.early_exit`, with `report.samples_evaluated`.
3. Per pair, the distance ladder is now entered at its widest rung.  Saturation there
   ("nothing inside `distmax`") is the module's existing proof rule, so it settles the
   pair in one narrowphase call instead of seven.  Measured on this model, 60.1% of
   candidate pairs saturate at the top rung (7008 of 11656 evaluations over 60 poses),
   and the short circuit has never disagreed with the full ladder (see
   `test_top_rung_short_circuit_matches_the_full_ladder`).

Measured end to end, as `supervisor.py` calls it (`steps=2, max_joint_step=0.01`):

    case                                     before            after
    clear, small move (0.02 rad)          10.2 ms    4 samples     5.1 ms   4 samples
    clear, long move (0.16 rad)           51.1 ms   18 samples    16.3 ms  12 samples
    near-miss, 6.25 mm                   106.1 ms   41 samples    13.8 ms  10 of 12
    deep collision, the logged -0.087 m  242.3 ms   63 samples     5.8 ms   4 of 12
    deep collision, longest travel      1200.9 ms  208 samples     2.8 ms   2 of 12

The verdict is identical in every one of those rows.  The worst case fell from 1200.9 ms
to 16.3 ms, and — the actual point — it no longer lives on the worst pose.  It now lives
on a CLEAR path, because a clear path is the only thing that spends the whole budget.

The bounding-sphere broadphase in `_candidate_pairs` predates this work and is doing most
of the heavy lifting already: it rejects ~99% of the 13596 checked pairs before any
narrowphase call (130 survive at the home pose, 1022 at the worst contorted pose found in
a 441-pose sweep).  It was kept and made cheaper by hoisting the static `geom_rbound`
sums out of the per-sample path; there is nothing further to win there.

Scope
-----
Self-clearance only: geoms in the MJCF's `class="collision"` (group 3).  The world floor
plane is group 0 and is NOT checked.  This module does not know about the environment.

Tool links
----------
The shipped model has a GENERIC parallel-jaw gripper on both arms.  The real robot has a
HUILING gripper on the left and a suction cup on the right, so every distance that
involves a `*gripper*` / `*suction*` / `*finger*` / `*tcp*` link is geometrically wrong
by an unknown amount.  Those pairs are evaluated but flagged separately
(`report.tool_violations`, `PairResult.is_tool`) so callers can treat them differently.

CLI
---
    python3 clearance.py --sweep left_arm_joint3 -0.5624 -0.2624
    python3 clearance.py --start left_arm_joint3=-0.5624 --end left_arm_joint3=-0.2624
    python3 clearance.py --pose left_arm_joint3=-0.30
Exit code is 1 on any violation, 0 when clear.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import mujoco
import numpy as np

from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST

__all__ = [
    "HOME_QPOS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_MAX_SAMPLES",
    "TOOL_LINK_PATTERNS",
    "PairKey",
    "PairResult",
    "SampleReport",
    "ClearanceReport",
    "ClearanceChecker",
    "check_path",
    "check_pose",
    "default_checker",
]


DEFAULT_MODEL_PATH = os.environ.get(
    "GALBOT_G1_MJCF",
    str(CANONICAL_MANIFEST.fixed_mjcf),
)

# Measured on the real robot at 18:06 on 2026-08-07, before anything moved.
# Every check is expressed as a deviation from this pose, and the baseline overlap set
# is computed here.
HOME_QPOS: Dict[str, float] = {
    "left_arm_joint1": 1.8199,
    "left_arm_joint2": -1.4697,
    "left_arm_joint3": -0.5624,
    "left_arm_joint4": -1.7742,
    "left_arm_joint5": -0.0077,
    "left_arm_joint6": -0.2749,
    "left_arm_joint7": 0.1115,
    "right_arm_joint1": -1.7820,
    "right_arm_joint2": 1.4846,
    "right_arm_joint3": 0.5983,
    "right_arm_joint4": 1.8141,
    "right_arm_joint5": -0.0069,
    "right_arm_joint6": 0.6207,
    "right_arm_joint7": -0.0406,
    "leg_joint1": 0.4980,
    "leg_joint2": 1.4961,
    "leg_joint3": 0.9994,
    "leg_joint4": -0.0006,
    "leg_joint5": -0.0003,
    "head_joint1": -0.0006,
    "head_joint2": -0.0002,
}

# Links whose geometry in this MJCF does not match the hardware (see module docstring).
#: Default minimum-clearance floor, in metres.
#:
#: **This must never be 0.0.** A floor of zero means a pair violates only at
#: negative distance -- i.e. once the two links already interpenetrate -- which is
#: contact detection, the exact semantics this module exists to replace. The
#: incident this project was founded on was a gripper reaching 5.2 cm from the
#: robot's own head while a contact check reported "0 collisions".
#:
#: 10 mm is chosen from the model, not invented. At the home pose it captures every
#: structurally-close pair as baseline -- the intra-gripper four-bar linkages
#: (sibling links ~0.18 mm apart, which weld-tree adjacency does not exclude because
#: they share a parent rather than being parent/child) and pairs like
#: ``*_arm_link5 | *_arm_link7`` at 1.17 mm. Those then receive the baseline
#: worsening check instead of a floor comparison, so real clearance is no longer
#: masked by permanent structure. Raising this value is safe; lowering it toward
#: zero re-introduces contact semantics.
#: Link-name prefixes identifying a rigid end-effector assembly.
#:
#: Links sharing one of these prefixes belong to the same gripper, whose internal
#: geometry is a mimic-driven four-bar linkage: their relative pose is a function
#: of the gripper drive joint alone and is invariant under every arm motion.
#: Comparing them to each other is therefore meaningless, and worse than
#: meaningless in practice -- ``mj_geomDistance`` returns lower bounds that vary by
#: ~2 mm across arm poses for these rigid pairs, which the baseline worsening check
#: then reports as a violation. Measured: rotating ``left_arm_joint1`` by 0.2 rad
#: "moved" ``left_gripper_l_knuckle_link | left_gripper_l_finger_link`` from
#: 8.551 mm to 6.250 mm. It did not move.
#:
#: These pairs are excluded from the verdict and reported separately. They are also
#: tool links, whose geometry does not match the real robot at all (the model ships
#: a generic gripper on both arms; the hardware carries a HUILING gripper and a
#: suction cup), so a number derived from them was never trustworthy.
ASSEMBLY_PREFIXES: Tuple[str, ...] = ("left_gripper_", "right_gripper_")

#: Body pairs whose collision primitives are modelled permanently interpenetrating at
#: every pose their joints can reach. Such a pair can never transition from clear to
#: touching, so it carries no collision signal -- only noise, which the relative
#: worsening rule then converts into false rejections.
#:
#: ``head_link2 | torso_base_link``: the head's collision geom is authored ~74 mm
#: inside the torso. Measured over 35 poses spanning the full reachable head range
#: (yaw -1.4308..+1.4308, pitch -0.1243..+0.4036) the separation is negative at every
#: one, -68.20 mm to -87.25 mm. It never separates. Pitching the head drives it
#: monotonically deeper -- worst case 13.58 mm at full chin-down, 18.4% of the
#: baseline -- so ``_worsening_tolerance`` rejected any pitch past +5.35 deg, capping
#: operator head-nod at a quarter of the joint's real 23.1 deg of travel.
#:
#: Excluding it beats widening BASELINE_RELATIVE_TOLERANCE, which would have to reach
#: 0.20 to cover the same range and would loosen the rule for every pair in the model
#: to accommodate one fictional overlap.
#:
#: Safe because the pair's relative geometry is a function of the head joints ALONE:
#: all 31 non-head hinge/slide joints, set to their lower limit, upper limit and
#: midpoint, changed the separation by less than 1e-9. The head's genuine constraint
#: is its URDF joint limit, which the retargeter enforces via its soft-limit band.
#:
#: Deliberately an explicit, measured list rather than a rule inferred from baseline
#: depth. A gate that silently stops watching a pair because it happens to look
#: embedded at HOME is a gate a bad calibration pose can switch off.
EMBEDDED_BODY_PAIRS: frozenset[frozenset[str]] = frozenset(
    {frozenset({"head_link2", "torso_base_link"})}
)


#: Fraction of an existing overlap depth that a baseline pair may degrade by.
#: See ``ClearanceChecker._worsening_tolerance``.
BASELINE_RELATIVE_TOLERANCE = 0.05

DEFAULT_FLOOR_M = 0.010

TOOL_LINK_PATTERNS: Tuple[str, ...] = ("*gripper*", "*suction*", "*finger*", "*tcp*")

#: Ceiling on the sample count that `max_joint_step` refinement may ask for.
#:
#: Chosen from measurement, against the 33.3 ms budget of a 30 FPS loop.  Per-sample cost
#: is dominated by how many geom pairs survive the bounding-sphere broadphase, which
#: varies with the pose.  Measured on this model (`check_pose`, one sample):
#:
#:     home pose, clear           130 candidate pairs   1.35 ms
#:     j3 = +0.6, colliding       435 candidate pairs   4.04 ms
#:     worst of a 441-pose sweep 1022 candidate pairs   7.47 ms
#:
#: A pose dense enough to cost 4-7 ms is dense *because* links are close, so it violates
#: and trips `early_exit` in the first sample or two.  The case that actually spends the
#: whole sample budget is therefore a CLEAR path, where the candidate count stays in the
#: 130-200 band and a sample costs ~1.4 ms.  Measured worst case of a full live-loop
#: `check_path` (`steps=2, max_joint_step=0.01`) against the cap:
#:
#:     max_samples   6    8.5 ms      max_samples  16   23.7 ms
#:     max_samples   8   11.5 ms      max_samples  20   26.3 ms
#:     max_samples  12   17.1 ms      max_samples  24   37.0 ms  <- over budget
#:
#: 12 keeps the worst case at roughly half the frame, leaving the other half for the rest
#: of the pipeline and about 2x headroom on a slower machine.  The worst case is a clear
#: path at every cap, which is the point: cost no longer grows with how bad the pose is.
#:
#: Raising this buys finer sampling at linear cost; lowering it makes degraded sweeps
#: more common.  It caps refinement only — an explicit `steps=` is always honoured.
DEFAULT_MAX_SAMPLES = 12

# Geom groups treated as collision geometry.  The MJCF puts every `class="collision"`
# geom in group 3; visuals are group 2 and the world floor plane is group 0.
COLLISION_GEOM_GROUPS: Tuple[int, ...] = (3,)

# Saturation test tolerance.  mj_geomDistance returns `distmax` bit-exactly when it finds
# nothing inside the margin, so this only guards against representation noise.
_SAT_TOL = 1e-12


# --------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class PairKey:
    """A canonical, order-independent identifier for a geom pair."""

    geom1: str
    geom2: str
    body1: str
    body2: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.body1} | {self.body2}"

    @property
    def link_pair(self) -> Tuple[str, str]:
        """The (body, body) pair, which is what an operator actually cares about.

        Order follows geom id, which is stable but arbitrary; use `matches` to compare.
        """
        return (self.body1, self.body2)

    @property
    def geom_pair(self) -> Tuple[str, str]:
        return (self.geom1, self.geom2)

    def matches(self, link_a: str, link_b: str) -> bool:
        """Order-independent link-pair comparison."""
        return sorted(self.link_pair) == sorted((link_a, link_b))


@dataclass
class PairResult:
    """One geom pair evaluated at one pose."""

    key: PairKey
    distance: float
    """Signed distance in metres.  Negative == penetration depth.

    ALWAYS a lower bound on the true distance: when `exact` is False the true distance is
    at least this value, never less.  Every pass/fail decision uses it in that sense, so
    an uncertain pair can only ever fail, never falsely pass.
    """
    exact: bool = True
    """False when the value is a proven lower bound rather than a measured distance
    (saturated at `distmax`, or the narrowphase contradicted itself — see module docs)."""
    saturated: bool = False
    """True when nothing was found inside `distmax`; `distance` == `distmax`."""
    unevaluable: bool = False
    """True when the pair could not be evaluated at all.  Always a violation."""
    reason: str = ""
    is_tool: bool = False
    """True when either link matches TOOL_LINK_PATTERNS."""
    baseline: Optional[float] = None
    """Home-pose distance, when this pair is baseline-constrained (see ClearanceChecker)."""
    violated: bool = False
    violation_kind: str = ""
    """'' | 'floor' | 'baseline_worsened' | 'unevaluable'"""
    fromto: Optional[Tuple[float, ...]] = None
    """Witness segment (x1,y1,z1,x2,y2,z2) in world coordinates, when available."""
    sample_index: int = 0
    """Which sample along the path this came from."""
    fraction: float = 0.0
    """Where along the path (0 == start, 1 == end)."""

    @property
    def baseline_delta(self) -> Optional[float]:
        """How much better (+) or worse (-) than the home-pose baseline."""
        if self.baseline is None:
            return None
        return self.distance - self.baseline

    def describe(self) -> str:
        mm = self.distance * 1000.0
        if self.unevaluable:
            body = f"UNEVALUABLE ({self.reason})"
        elif self.saturated:
            body = f">= {mm:.1f} mm (nothing inside the {mm:.0f} mm window)"
        elif self.exact:
            body = f"{mm:+8.2f} mm" + ("  [PENETRATING]" if self.distance < 0 else "")
        else:
            body = f">= {mm:+.2f} mm (lower bound; narrowphase inconsistent)"
        tags = []
        if self.is_tool:
            tags.append("TOOL-LINK: geometry does not match hardware")
        if self.baseline is not None:
            tags.append(
                f"baseline {self.baseline * 1000:+.2f} mm, "
                f"delta {self.baseline_delta * 1000:+.2f} mm"
            )
        if self.violated:
            tags.append(f"VIOLATION[{self.violation_kind}]")
        suffix = ("   " + "; ".join(tags)) if tags else ""
        return (
            f"{self.key.body1} | {self.key.body2}   {body}{suffix}\n"
            f"        sample {self.sample_index} (t={self.fraction:.3f})   "
            f"geoms: {self.key.geom1} | {self.key.geom2}"
        )


def _severity(r: PairResult) -> Tuple[int, float]:
    """Sort key: unevaluable first, then the most negative margin.

    For a baseline-constrained pair the meaningful margin is how far it has degraded from
    its baseline; for every other pair it is the distance itself.
    """
    if r.unevaluable:
        return (0, 0.0)
    if r.baseline is not None:
        return (1, float(r.baseline_delta))
    return (1, float(r.distance))


@dataclass
class SampleReport:
    """One interpolated pose along a path."""

    index: int
    fraction: float
    qpos: Dict[str, float]
    min_distance: float
    min_pair: Optional[PairResult]
    violations: List[PairResult] = field(default_factory=list)
    suspect_pairs: List[PairResult] = field(default_factory=list)
    tool_min_distance: Optional[float] = None
    tool_min_pair: Optional[PairResult] = None
    worst_baseline: Optional[PairResult] = None
    n_pairs_evaluated: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations


@dataclass
class ClearanceReport:
    """The result of checking a whole path."""

    ok: bool
    min_distance: float
    """Smallest clearance seen at any sample, over all non-baseline pairs.

    Baseline-overlapping pairs are excluded from this number because they are structurally
    overlapped at every pose and would otherwise permanently dominate it (head_link2 |
    torso_base_link sits at -73.7 mm no matter what the arms do).  They are still checked,
    via `worst_baseline`.
    """
    min_pair: Optional[PairResult]
    min_sample_index: int
    min_fraction: float
    n_samples: int
    """How many samples the path was PLANNED with, after `max_samples` was applied.

    `samples_evaluated` is how many were actually run; the two differ only when
    `early_exit` fired.
    """
    steps_requested: int
    samples: List[SampleReport] = field(default_factory=list)
    violations: List[PairResult] = field(default_factory=list)
    tool_violations: List[PairResult] = field(default_factory=list)
    structural_violations: List[PairResult] = field(default_factory=list)
    suspect_pairs: List[PairResult] = field(default_factory=list)
    tool_min_distance: Optional[float] = None
    tool_min_pair: Optional[PairResult] = None
    worst_baseline: Optional[PairResult] = None
    limit_warnings: List[str] = field(default_factory=list)
    config: Dict[str, float] = field(default_factory=dict)

    # -- sampling accounting ------------------------------------------------------------
    samples_evaluated: int = 0
    """How many samples were actually evaluated.  Less than `n_samples` iff `early_exit`."""
    samples_needed: int = 0
    """How many samples `max_joint_step` asked for, before `max_samples` capped it."""
    max_samples: Optional[int] = None
    """The refinement cap in force; None means the count was not capped."""
    max_joint_step_requested: Optional[float] = None
    """The per-step joint motion the caller asked not to exceed (rad).  None == disabled."""
    max_joint_step_actual: float = 0.0
    """The largest per-step joint motion actually checked (rad)."""
    sampling_degraded: bool = False
    """True when the sweep is COARSER than `max_joint_step` asked for.

    This is the one flag that can weaken a PASS.  A coarser sweep steps over a longer
    arc of joint space between samples, so a thin collision that lives entirely between
    two samples is not seen.  `ok` still means exactly what it always meant — "no
    violation at any sample evaluated" — but on a degraded report that is a statement
    about the samples, not about the path.  Use `pass_is_certified` to tell them apart.
    """
    degraded_reason: str = ""
    """Human-readable statement of the shortfall, with numbers.  Empty when not degraded.

    Also appended to `limit_warnings`, so any caller that already surfaces those (the CLI
    prints them with a leading `!`) shows the degradation without being changed.
    """
    early_exit: bool = False
    """True when sampling stopped at a confirmed violation with samples left unevaluated.

    Only ever set on a FAIL: the loop stops after `SampleReport.violations` is non-empty,
    which is a positive finding, never an absence of one.  The verdict is unaffected — no
    later sample could have cleared an already-confirmed violation.  What IS affected is
    the headline: `min_distance`, `min_pair`, `tool_min_pair` and `worst_baseline`
    describe the path only up to `samples_evaluated`, so the reported depth is the worst
    seen BEFORE the decision, not the worst on the path.  Re-run with `early_exit=False`
    for the full picture (the CLI does this by default).
    """

    @property
    def pass_is_certified(self) -> bool:
        """True when a PASS from this report means what a PASS normally means.

        False on a degraded sweep, where `ok` only says "nothing at the samples we could
        afford to take".  A caller gating motion on `ok` alone should check this too.
        """
        return self.ok and not self.sampling_degraded

    @property
    def first_violation_index(self) -> Optional[int]:
        for s in self.samples:
            if s.violations:
                return s.index
        return None

    @property
    def first_violation_fraction(self) -> Optional[float]:
        i = self.first_violation_index
        return None if i is None else self.samples[i].fraction

    @property
    def violating_link_pairs(self) -> List[Tuple[str, str]]:
        """Distinct link pairs that violated, worst first."""
        return [r.key.link_pair for r in self.worst_violation_per_link_pair()]

    def worst_violation_per_link_pair(self) -> List[PairResult]:
        """One `PairResult` per offending link pair — the worst one seen, worst first.

        A single link pair usually violates at many samples and across several of its
        geoms; the operator wants "which two links, and how bad did it get".
        """
        worst: Dict[Tuple[str, ...], PairResult] = {}
        for v in self.violations:
            k = tuple(sorted(v.key.link_pair))
            cur = worst.get(k)
            if cur is None or _severity(v) < _severity(cur):
                worst[k] = v
        return sorted(worst.values(), key=_severity)

    def distinct_suspect_pairs(self) -> List[PairResult]:
        """One `PairResult` per link pair whose narrowphase result was contradictory."""
        seen: Dict[Tuple[str, ...], PairResult] = {}
        for p in self.suspect_pairs:
            seen.setdefault(tuple(sorted(p.key.link_pair)), p)
        return list(seen.values())

    def format(self) -> str:
        out: List[str] = []
        cfg = self.config
        out.append(
            "clearance: floor {:.1f} mm | distmax {:.0f} mm | baseline margin {:.1f} mm | "
            "{} samples (requested {}){}".format(
                cfg.get("floor", 0.0) * 1000,
                cfg.get("distmax", 0.0) * 1000,
                cfg.get("baseline_margin", 0.0) * 1000,
                self.n_samples,
                self.steps_requested,
                ""
                if self.samples_evaluated in (0, self.n_samples)
                else f" | {self.samples_evaluated} evaluated",
            )
        )
        for w in self.limit_warnings:
            out.append(f"  ! {w}")
        if self.early_exit:
            out.append(
                "  early exit: stopped at sample {} of {} on a confirmed violation; "
                "{} sample(s) not evaluated.".format(
                    self.samples_evaluated - 1,
                    self.n_samples - 1,
                    self.n_samples - self.samples_evaluated,
                )
            )
            out.append(
                "    the verdict is final, but the minimum below is the worst seen "
                "BEFORE the decision, not the worst on the path "
                "(re-run with early_exit=False for that)."
            )
        if self.min_pair is None:
            out.append(
                "  minimum clearance: nothing within the search window at any sample"
            )
        else:
            out.append(
                "  minimum clearance: {:+.2f} mm at sample {}/{} (fraction {:.3f})".format(
                    self.min_distance * 1000,
                    self.min_sample_index,
                    self.n_samples - 1,
                    self.min_fraction,
                )
            )
            out.append("      " + self.min_pair.describe())
        if self.tool_min_pair is not None:
            out.append("  closest TOOL-link pair (geometry unreliable, see module docs):")
            out.append("      " + self.tool_min_pair.describe())
        if self.worst_baseline is not None:
            out.append("  worst baseline-overlap pair:")
            out.append("      " + self.worst_baseline.describe())
        distinct_suspects = self.distinct_suspect_pairs()
        if distinct_suspects:
            out.append(
                f"  {len(distinct_suspects)} link pair(s) where the narrowphase "
                "contradicted itself; proven lower bounds used instead:"
            )
            for p in distinct_suspects[:5]:
                out.append("      " + p.describe())
            if len(distinct_suspects) > 5:
                out.append(f"      ... and {len(distinct_suspects) - 5} more")
        if self.violations:
            worst = self.worst_violation_per_link_pair()
            out.append(
                f"  *** {len(worst)} OFFENDING LINK PAIR(S), "
                f"{len(self.violations)} violating sample-pairs ***"
            )
            out.append(
                "  first violating sample: {} (fraction {:.3f})".format(
                    self.first_violation_index, self.first_violation_fraction
                )
            )
            for v in worst:
                out.append("      " + v.describe())
            if self.tool_violations:
                out.append(
                    f"  ({len(self.tool_violations)} violating sample-pairs involve a tool "
                    "link and are geometrically unreliable)"
                )
            out.append("  DO NOT SEND THIS PATH TO THE ROBOT")
        elif self.sampling_degraded:
            out.append("  *** SAMPLING DEGRADED — THIS IS NOT A CERTIFIED CLEAR ***")
            out.append(f"      {self.degraded_reason}")
            out.append(
                "      no violation was found, but a collision thinner than the step "
                "above would not have been looked at."
            )
        else:
            out.append("  clear")
        return "\n".join(out)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return self.format()


# --------------------------------------------------------------------------------------
# Checker
# --------------------------------------------------------------------------------------


class ClearanceChecker:
    """Minimum-distance self-clearance checker for a MuJoCo model.

    Pass/fail rule, per geom pair, per sample:

    * structurally adjacent pairs (same rigid weld body, or weld-parent/child, i.e.
      separated by exactly one joint) are not checked at all;
    * a pair that could not be evaluated is a violation ("fail closed");
    * a baseline-constrained pair (one whose home-pose distance was already below
      `floor`) violates when it gets worse than its baseline by more than
      `baseline_margin`;
    * every other pair violates when its distance drops below `floor`.

    Every decision uses a value that is a lower bound on the true distance, so
    uncertainty can only produce a false alarm, never a false pass.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        home: Optional[Mapping[str, float]] = None,
        floor: float = DEFAULT_FLOOR_M,
        distmax: float = 0.05,
        baseline_margin: float = 0.002,
        ladder_rungs: int = 6,
        tool_patterns: Sequence[str] = TOOL_LINK_PATTERNS,
        collision_groups: Sequence[int] = COLLISION_GEOM_GROUPS,
        model: Optional["mujoco.MjModel"] = None,
    ) -> None:
        if distmax <= 0.0:
            raise ValueError("distmax must be > 0")
        if floor >= distmax:
            raise ValueError(
                f"floor ({floor}) must be < distmax ({distmax}); nothing beyond distmax "
                "is measured, so a floor at or above it could never be verified"
            )
        if baseline_margin < 0.0:
            raise ValueError("baseline_margin must be >= 0")
        if ladder_rungs < 1:
            raise ValueError("ladder_rungs must be >= 1")

        # `model` lets a caller (or a test) supply an already-compiled MjModel — e.g. one
        # built with MjSpec — instead of loading from disk.  Nothing is ever written.
        self.model_path = model_path if model is None else "<in-memory MjModel>"
        self.model = mujoco.MjModel.from_xml_path(model_path) if model is None else model
        self.data = mujoco.MjData(self.model)
        self.floor = float(floor)
        self.distmax = float(distmax)
        self.baseline_margin = float(baseline_margin)
        self.tool_patterns = tuple(tool_patterns)
        self.collision_groups = tuple(collision_groups)

        # Name lookups are static for the life of the model, and `mj_id2name` costs
        # ~0.73 us a call.  `_make_key` needs four of them per geom pair, which at a few
        # hundred candidate pairs a sample was 0.45 ms/sample of pure string plumbing --
        # a fifth of the home-pose budget spent re-deriving constants.  Resolve them once.
        self._geom_names: Tuple[str, ...] = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            for g in range(self.model.ngeom)
        )
        self._body_names_by_geom: Tuple[str, ...] = tuple(
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])
            )
            or f"body{int(self.model.geom_bodyid[g])}"
            for g in range(self.model.ngeom)
        )
        self._key_cache: Dict[Tuple[int, int], PairKey] = {}
        self._key_ids: Dict[PairKey, Tuple[int, int]] = {}
        self._joint_ids: Dict[str, int] = {}

        # Increasing ladder of distmax values, e.g. distmax/32 ... distmax.
        #
        # `floor` is inserted as a rung. Without it the decision threshold can fall
        # in a gap between geometric rungs: with distmax=0.05 and 6 rungs the ladder
        # is 1.5625, 3.125, 6.25, 12.5, 25, 50 mm, and a 10 mm floor sits between two
        # of them. A pair could then be proven ">= 6.25 mm" and still be unresolvable
        # against 10 mm, so the conservative branch rejected it -- blocking motion
        # that was in fact clear. Making the threshold directly queryable turns that
        # into a decidable question: saturation AT the floor proves `true > floor`.
        rungs = {self.distmax / (2 ** k) for k in range(ladder_rungs - 1, -1, -1)}
        if 0.0 < floor < self.distmax:
            rungs.add(float(floor))
        self.ladder: Tuple[float, ...] = tuple(sorted(rungs))

        self.home: Dict[str, float] = dict(HOME_QPOS if home is None else home)
        self._validate_joint_names(self.home)

        self._geom_ids = self._collision_geom_ids()
        self._pair_ids, self._n_adjacent_excluded = self._build_pairs()
        self._is_tool_geom = {
            g: self._matches_tool(self._geom_name(g)) or self._matches_tool(self._body_name(g))
            for g in self._geom_ids
        }

        # Broadphase constants.  `geom_rbound` is a compile-time property of the model, so
        # the per-pair radius sum never changes; only `geom_xpos` does.  Hoisting it out
        # leaves the per-sample broadphase as one gather plus one norm.
        self._pair_a = np.ascontiguousarray(self._pair_ids[:, 0])
        self._pair_b = np.ascontiguousarray(self._pair_ids[:, 1])
        self._pair_rsum = (
            self.model.geom_rbound[self._pair_a] + self.model.geom_rbound[self._pair_b]
        )
        # Broadphase scratch, built lazily on first use (see _candidate_pairs).
        self._bp_flat: np.ndarray | None = None
        self._bp_gids: np.ndarray | None = None
        self._bp_threshold_sq: np.ndarray | None = None
        #: Per-frame pose cache.  None disables it; the pipeline opens one per frame.
        self._frame_memo: Dict[Tuple, SampleReport] | None = None

        # Baseline: home-pose distance of every pair that is already below the floor
        # at home.  Keyed by (geom1_id, geom2_id).  NOT a whitelist — see _classify.
        self.baseline: Dict[Tuple[int, int], float] = {}
        self.baseline_unevaluable: Dict[Tuple[int, int], str] = {}
        self._build_baseline()

    # -- model introspection ------------------------------------------------------------

    def _geom_name(self, g: int) -> str:
        return self._geom_names[int(g)]

    def _body_name(self, g: int) -> str:
        return self._body_names_by_geom[int(g)]

    def _joint_id(self, name: str) -> int:
        """`mj_name2id` for a joint, memoised.  -1 when the joint is not in the model."""
        jid = self._joint_ids.get(name)
        if jid is None:
            jid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            self._joint_ids[name] = jid
        return jid

    def _matches_tool(self, name: str) -> bool:
        low = name.lower()
        return any(fnmatch.fnmatch(low, pat) for pat in self.tool_patterns)

    def _collision_geom_ids(self) -> np.ndarray:
        ids = [
            g
            for g in range(self.model.ngeom)
            if int(self.model.geom_group[g]) in self.collision_groups
        ]
        if not ids:
            raise ValueError(
                f"no geoms in groups {self.collision_groups} in {self.model_path}"
            )
        return np.asarray(ids, dtype=np.int32)

    def _weld_parent(self, weld: int) -> int:
        """The weld body one joint up the tree from `weld`."""
        return int(self.model.body_weldid[self.model.body_parentid[int(weld)]])

    def is_adjacent(self, g1: int, g2: int) -> bool:
        """True when two geoms are structurally adjacent and must not be checked.

        Derived from the model's kinematic tree, never hardcoded.  `body_weldid` collapses
        every chain of fixed (jointless) bodies into a single rigid unit, so:

        * same weld body  -> the geoms are bolted to the same rigid part;
        * weld-parent/child -> the parts are separated by exactly one joint, and are
          near each other by construction at every pose.

        This is the same rule MuJoCo's own broadphase applies before generating contacts,
        which is why `galbot_preview.py` never sees these pairs either.
        """
        w1 = int(self.model.body_weldid[self.model.geom_bodyid[int(g1)]])
        w2 = int(self.model.body_weldid[self.model.geom_bodyid[int(g2)]])
        if w1 == w2:
            return True
        return self._weld_parent(w1) == w2 or self._weld_parent(w2) == w1

    def same_assembly(self, g1: int, g2: int) -> bool:
        """True when both geoms belong to the same rigid end-effector assembly.

        The weld rule above catches parent/child and same-weld pairs, but a gripper's
        four-bar linkage is a set of *siblings* -- ``*_knuckle_link``,
        ``*_inner_knuckle_link`` and ``*_finger_link`` are all children of
        ``*_gripper_base_link``, separated from each other by two joints rather than
        one. Weld adjacency therefore misses them, and they are the closest pair in
        the whole model at almost every pose (~0.18 mm), so they dominate
        ``min_distance`` and mask real structural clearance.

        Their relative geometry is a function of the gripper drive joint alone, so
        comparing them under arm motion measures nothing. See ``ASSEMBLY_PREFIXES``.
        """
        n1 = self._body_name(int(g1))
        n2 = self._body_name(int(g2))
        return any(
            n1.startswith(prefix) and n2.startswith(prefix)
            for prefix in ASSEMBLY_PREFIXES
        )

    def permanently_embedded(self, g1: int, g2: int) -> bool:
        """True for pairs modelled interpenetrating at every reachable pose.

        See ``EMBEDDED_BODY_PAIRS``. Kept separate from both ``is_adjacent`` (purely
        kinematic-tree derived) and ``same_assembly`` (rigid end-effector siblings):
        this is a third, distinct claim -- that a pair's separation never changes sign
        anywhere in its joints' range -- and each rule should stay auditable alone.
        """
        pair = frozenset({self._body_name(int(g1)), self._body_name(int(g2))})
        return pair in EMBEDDED_BODY_PAIRS

    def _build_pairs(self) -> Tuple[np.ndarray, int]:
        ids = self._geom_ids
        pairs = []
        excluded = 0
        assembly = 0
        embedded = 0
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if self.is_adjacent(int(a), int(b)):
                    excluded += 1
                elif self.permanently_embedded(int(a), int(b)):
                    embedded += 1
                elif self.same_assembly(int(a), int(b)):
                    # Deliberately a SEPARATE category from adjacency. `is_adjacent`
                    # is derived purely from the kinematic tree and must stay that
                    # way; this is a name-based rule about rigid end-effector
                    # assemblies, and conflating the two would make a structural
                    # guarantee depend on a string prefix.
                    assembly += 1
                else:
                    pairs.append((int(a), int(b)))
        self._n_assembly_excluded = assembly
        self._n_embedded_excluded = embedded
        if not pairs:
            raise ValueError("every collision pair was excluded as adjacent")
        return np.asarray(pairs, dtype=np.int32), excluded

    @property
    def n_pairs(self) -> int:
        return int(self._pair_ids.shape[0])

    @property
    def n_adjacent_excluded(self) -> int:
        return self._n_adjacent_excluded

    @property
    def n_assembly_excluded(self) -> int:
        """Pairs skipped because both links belong to one rigid end-effector assembly."""
        return self._n_assembly_excluded

    def joint_names(self) -> List[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.njnt)
        ]

    def _validate_joint_names(self, pose: Mapping[str, float]) -> None:
        unknown = [n for n in pose if self._joint_id(n) < 0]
        if unknown:
            raise ValueError(
                f"joints not in {os.path.basename(self.model_path)}: {sorted(unknown)}. "
                "Refusing to silently ignore them — a typo'd joint name would mean the "
                "pose you checked is not the pose you send."
            )

    # -- pose handling ------------------------------------------------------------------

    def full_pose(self, overrides: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
        """`home` with `overrides` applied.  Raises on unknown joint names."""
        pose = dict(self.home)
        if overrides:
            self._validate_joint_names(overrides)
            pose.update({k: float(v) for k, v in overrides.items()})
        return pose

    def tcp_positions(
        self, overrides: Optional[Mapping[str, float]] = None
    ) -> Dict[str, Tuple[float, float, float]]:
        """World-frame TCP position of each arm at ``home`` + ``overrides``.

        Exists so the safety supervisor can recompute a declared IK residual for
        itself rather than trusting the retargeter that produced it. It shares
        this class's pose handling deliberately: the same `home` fill and the same
        `mj_forward`, so the geometry it reports is the geometry the clearance
        check is reasoning about, not a second opinion assembled differently.
        """
        pose = self.full_pose(overrides)
        self._apply(pose)
        out: Dict[str, Tuple[float, float, float]] = {}
        for side in ("left", "right"):
            body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_gripper_tcp_link"
            )
            if body < 0:
                continue
            position = self.data.xpos[body]
            out[side] = (float(position[0]), float(position[1]), float(position[2]))
        return out

    def limit_warnings(self, pose: Mapping[str, float]) -> List[str]:
        out = []
        for name, val in pose.items():
            jid = self._joint_id(name)
            if jid < 0 or not self.model.jnt_limited[jid]:
                continue
            lo, hi = self.model.jnt_range[jid]
            if val < lo - 1e-9 or val > hi + 1e-9:
                out.append(
                    f"{name}={val:+.4f} is outside its limit [{lo:+.4f}, {hi:+.4f}]"
                )
        return out

    def _apply(self, pose: Mapping[str, float]) -> None:
        self.data.qpos[:] = 0.0
        for name, val in pose.items():
            jid = self._joint_id(name)
            if jid < 0:
                raise ValueError(f"joint not in model: {name}")
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(val)
        # Kinematics only.  mj_geomDistance reads geom_xpos and geom_xmat plus
        # static model mesh data, and mj_kinematics computes exactly those; the
        # extra comPos/tendon/transmission/CRB/factorM/collision/constraint work
        # in mj_forward is never read here.  Verified bit-identical over 60 random
        # poses (max |geom_xpos diff| and |geom_xmat diff| both exactly 0.0).  The
        # preview SINK still needs mj_forward -- this is the checker only.
        mujoco.mj_kinematics(self.model, self.data)

    # -- distance evaluation ------------------------------------------------------------

    def _geom_distance(self, g1: int, g2: int, distmax: float) -> float:
        """Raw `mj_geomDistance` call.  Overridable so tests can force a failure."""
        return float(
            mujoco.mj_geomDistance(
                self.model, self.data, int(g1), int(g2), float(distmax), None
            )
        )

    def _witness(self, g1: int, g2: int, distmax: float) -> Optional[Tuple[float, ...]]:
        try:
            ft = np.zeros(6, dtype=np.float64)
            self._geom_distance_with_fromto(int(g1), int(g2), float(distmax), ft)
            if not np.all(np.isfinite(ft)) or not np.any(ft):
                return None
            return tuple(float(x) for x in ft)
        except Exception:
            return None

    def _geom_distance_with_fromto(
        self, g1: int, g2: int, distmax: float, fromto: np.ndarray
    ) -> float:
        return float(
            mujoco.mj_geomDistance(
                self.model, self.data, int(g1), int(g2), float(distmax), fromto
            )
        )

    def _evaluate_pair(self, g1: int, g2: int) -> Tuple[float, bool, bool, bool, str]:
        """Return (distance, exact, saturated, unevaluable, reason) for one geom pair.

        `distance` is always a LOWER BOUND on the true separation.

        Enters the ladder at its WIDEST rung, which is `distmax`.  Saturation there means
        the narrowphase found nothing at all inside the search window, which under this
        module's own proof rule ("a rung that saturates is a proof that the true distance
        exceeds that rung") settles the pair outright: `true >= distmax`, no narrower rung
        can say anything stronger, and the answer is the same one the full ladder reaches
        via its `not measurements` branch.  That is one narrowphase call instead of seven,
        and on this model it covers 59% of candidate pairs -- the single largest saving in
        the whole check.

        This is a fast path, not a different rule.  If the top rung does NOT saturate the
        remaining rungs are queried and `_reconcile_ladder` decides, exactly as before;
        the equivalence is pinned by `test_top_rung_short_circuit_matches_the_full_ladder`,
        which compares the two implementations pair-for-pair over a pose sweep.

        Anything that raises, returns a non-finite value, or produces no usable
        information at all is unevaluable — which the caller turns into a violation.
        """
        top = self.ladder[-1]
        try:
            v_top = self._geom_distance(g1, g2, top)
        except Exception as exc:  # fail closed on anything at all
            return (float("-inf"), False, False, True, f"{type(exc).__name__}: {exc}")
        if not np.isfinite(v_top):
            return (float("-inf"), False, False, True, f"non-finite result {v_top!r}")
        if v_top >= top - _SAT_TOL:
            # Nothing found inside the widest window: true distance >= distmax.
            return (self.distmax, False, True, False, "")
        return self._reconcile_ladder(g1, g2, v_top)

    def _reconcile_ladder(
        self, g1: int, g2: int, v_top: Optional[float] = None
    ) -> Tuple[float, bool, bool, bool, str]:
        """The full ladder: query every rung and cross-check them against each other.

        The MuJoCo contract is `f(distmax) == min(true_distance, distmax)`.  Query a
        ladder of increasing distmax values and use that contract as a cross-check:

        * a rung where `f(dm) == dm` saturated: proof that `true > dm`;
        * a rung where `f(dm) <  dm` claims a measurement of `true`;
        * a claimed measurement below a proven lower bound is impossible, so it is the
          measurement that is wrong (this is the spurious-zero failure mode documented at
          the top of the file).  Keep the lower bound and mark the pair inexact.

        `v_top` is the widest rung's value when the caller already has it; the reference
        behaviour (every rung queried here) is what the equivalence test drives.
        """
        values: List[float] = [0.0] * len(self.ladder)
        rungs = self.ladder if v_top is None else self.ladder[:-1]
        if v_top is not None:
            values[-1] = float(v_top)
        for i, dm in enumerate(rungs):
            try:
                v = self._geom_distance(g1, g2, dm)
            except Exception as exc:  # fail closed on anything at all
                return (float("-inf"), False, False, True, f"{type(exc).__name__}: {exc}")
            if not np.isfinite(v):
                return (float("-inf"), False, False, True, f"non-finite result {v!r}")
            values[i] = v

        # One pass instead of two list comprehensions plus max/min over them. Same
        # rungs, same values, same comparisons -- this loop runs ~580k times a
        # frame, so the allocations were the cost, not the arithmetic.
        lower_bound = float("-inf")
        best = float("inf")
        has_measurement = False
        has_saturated = False
        for dm, v in zip(self.ladder, values):
            if v >= dm - _SAT_TOL:
                has_saturated = True
                lower_bound = max(lower_bound, dm)
            else:
                has_measurement = True
                best = min(best, v)

        if not has_measurement:
            # Nothing found inside the widest window: true distance >= distmax.
            return (self.distmax, False, True, False, "")

        if has_saturated and best < lower_bound - _SAT_TOL:
            return (
                lower_bound,
                False,
                False,
                False,
                f"narrowphase returned {best:.6f} m below a proven lower bound of "
                f"{lower_bound:.6f} m",
            )
        return (best, True, False, False, "")

    def _candidate_pairs(self) -> np.ndarray:
        """Bounding-sphere prefilter — the cheap broadphase, run before any narrowphase.

        `geom_rbound` is the radius of the bounding sphere centred on `geom_xpos`, so
        `|x1-x2| - r1 - r2 > distmax` proves the true distance exceeds distmax.  This is
        the same conservative test MuJoCo's own broadphase uses; it never discards a pair
        that could be inside the window (verified against unfiltered evaluation).

        It is worth its cost by a wide margin and then some: of the 13596 checked pairs it
        passes 130 at the home pose and 1022 at the worst pose found in a 441-pose sweep,
        i.e. it removes 92-99% of the work before the seven-rung ladder is entered.  The
        arithmetic itself is ~0.31 ms a sample, against ~10 us a pair for the ladder — so
        skipping the filter entirely and evaluating all 13596 pairs would cost about 140
        ms a sample instead of 1-6 ms.

        `r1 + r2` is a compile-time constant of the model, so it is precomputed in
        `__init__`; only the gather and the norm are per-sample.  The comparison is
        arithmetically identical to the earlier inline form.
        """
        if self._bp_flat is None:
            geoms = np.unique(np.concatenate([self._pair_a, self._pair_b]))
            index = {int(g): i for i, g in enumerate(geoms)}
            rows = np.array([index[int(g)] for g in self._pair_a], dtype=np.int64)
            cols = np.array([index[int(g)] for g in self._pair_b], dtype=np.int64)
            self._bp_gids = geoms.astype(np.int64)
            self._bp_flat = (rows * len(geoms) + cols).astype(np.int64)
            threshold = self._pair_rsum + self.distmax
            # Squared form of |d| <= rsum + distmax (both sides non-negative), with
            # the threshold deliberately inflated so the test can only ever ADMIT a
            # pair the exact form keeps -- provably a superset, never a dropped pair.
            self._bp_threshold_sq = threshold * threshold * (1.0 + 1e-9) + 1e-15
        points = self.data.geom_xpos[self._bp_gids]
        squares = np.einsum("ij,ij->i", points, points)
        gram = squares[:, None] + squares[None, :] - 2.0 * (points @ points.T)
        return self._pair_ids[gram.ravel()[self._bp_flat] <= self._bp_threshold_sq]

    def _worsening_tolerance(self, baseline: float) -> float:
        """How much a baseline-constrained pair may degrade before it is a violation.

        A flat absolute margin is the wrong test for a pair that is already deeply
        interpenetrated. ``head_link2 | torso_base_link`` sits **73.7 mm** inside
        itself at the home pose -- the head geometry is simply embedded in the torso
        geometry in this model -- and rotating the head sweeps that number by about
        2 mm. With a flat 2 mm margin, merely *looking around* registered as a
        developing collision and blocked the demo at frame 30.

        A pair 73 mm inside another cannot meaningfully collide harder; what matters
        is a proportionally significant change. So the tolerance is the larger of the
        absolute margin and a fraction of the existing overlap depth. For the
        head/torso pair that yields ~3.7 mm, which absorbs the rotation sweep while
        still catching a real 10 mm deterioration. For a pair whose baseline is
        shallow -- the ordinary case -- ``baseline_margin`` still governs, unchanged.
        """
        return max(self.baseline_margin, BASELINE_RELATIVE_TOLERANCE * abs(baseline))

    def _make_key(self, g1: int, g2: int) -> PairKey:
        """The (frozen, hashable) identity of a geom pair.

        Memoised: a pair's names never change, and the same few hundred pairs are
        rebuilt at every sample of every path.  `_key_ids` is the inverse, so a caller
        holding a `PairResult` can get back to the geom ids without a name lookup.
        """
        ids = (int(g1), int(g2))
        key = self._key_cache.get(ids)
        if key is None:
            key = PairKey(
                geom1=self._geom_names[ids[0]],
                geom2=self._geom_names[ids[1]],
                body1=self._body_names_by_geom[ids[0]],
                body2=self._body_names_by_geom[ids[1]],
            )
            self._key_cache[ids] = key
            self._key_ids[key] = ids
        return key

    # -- baseline -----------------------------------------------------------------------

    def _build_baseline(self) -> None:
        """Record, at the home pose, every pair already below `floor`.

        These are the model's permanent structural overlaps (head/torso, and — before the
        adjacency rule removes them — leg/chassis and leg/wheels).  They are recorded, not
        forgiven: `_classify` still fails them if they get worse.
        """
        self._apply(self.home)
        for g1, g2 in self._candidate_pairs():
            dist, exact, saturated, unevaluable, reason = self._evaluate_pair(int(g1), int(g2))
            key = (int(g1), int(g2))
            if unevaluable:
                self.baseline_unevaluable[key] = reason
            elif dist < self.floor:
                self.baseline[key] = dist

    def baseline_pairs(self) -> Dict[PairKey, float]:
        """Human-readable view of the baseline overlap set."""
        return {self._make_key(g1, g2): v for (g1, g2), v in self.baseline.items()}

    # -- classification -----------------------------------------------------------------

    def _classify(self, g1: int, g2: int) -> PairResult:
        key_ids = (int(g1), int(g2))
        dist, exact, saturated, unevaluable, reason = self._evaluate_pair(int(g1), int(g2))
        baseline = self.baseline.get(key_ids)
        res = PairResult(
            key=self._make_key(g1, g2),
            distance=dist,
            exact=exact,
            saturated=saturated,
            unevaluable=unevaluable,
            reason=reason,
            is_tool=bool(self._is_tool_geom[int(g1)] or self._is_tool_geom[int(g2)]),
            baseline=baseline,
        )

        if key_ids in self.baseline_unevaluable:
            # We never established what "normal" looks like for this pair, so we can never
            # say it is fine.
            res.unevaluable = True
            res.violated = True
            res.violation_kind = "unevaluable"
            res.reason = res.reason or (
                "unevaluable at the home pose: " + self.baseline_unevaluable[key_ids]
            )
            return res

        if unevaluable:
            res.violated = True
            res.violation_kind = "unevaluable"
            return res

        if baseline is not None:
            if dist < baseline - self._worsening_tolerance(baseline):
                res.violated = True
                res.violation_kind = "baseline_worsened"
        elif dist < self.floor:
            res.violated = True
            res.violation_kind = "floor"
        return res

    # -- public API ---------------------------------------------------------------------

    def check_pose(
        self,
        qpos_dict: Optional[Mapping[str, float]] = None,
        index: int = 0,
        fraction: float = 0.0,
    ) -> SampleReport:
        """Evaluate one pose.  Unspecified joints fall back to `home`."""
        pose = self.full_pose(qpos_dict)
        # Within a single frame, pose -> SampleReport is a pure function: _apply is
        # deterministic (zero qpos, write the pose, run kinematics) and the baseline
        # is immutable.  Census over 120 recorded frames: 2,384 of 11,718 samples
        # (20.3%) are exact re-evaluations of a pose already decided this frame --
        # sample 0 of every check_path in a bisection is the same start pose.  This
        # computes each distinct pose once instead.  Nothing is skipped, sampled or
        # coarsened: every distinct pose still runs the full ladder, and the key is
        # the exact float pose so a different pose can never hit a cached entry.
        memo = self._frame_memo
        key = None
        if memo is not None:
            key = (
                tuple(sorted((name, float(v)) for name, v in pose.items())),
                int(index),
                float(fraction),
            )
            cached = memo.get(key)
            if cached is not None:
                return cached
        self._apply(pose)

        min_res: Optional[PairResult] = None
        tool_min: Optional[PairResult] = None
        worst_baseline: Optional[PairResult] = None
        violations: List[PairResult] = []
        suspects: List[PairResult] = []
        candidates = self._candidate_pairs()

        for g1, g2 in candidates:
            res = self._classify(int(g1), int(g2))
            res.sample_index = index
            res.fraction = fraction
            if res.violated:
                violations.append(res)
            if not res.exact and not res.saturated and not res.unevaluable:
                suspects.append(res)
            if res.unevaluable:
                continue
            if res.baseline is not None:
                if worst_baseline is None or (
                    res.baseline_delta < worst_baseline.baseline_delta
                ):
                    worst_baseline = res
                continue  # keep permanent overlaps out of the headline minimum
            if min_res is None or res.distance < min_res.distance:
                min_res = res
            if res.is_tool and (tool_min is None or res.distance < tool_min.distance):
                tool_min = res

        for res in [min_res, tool_min, worst_baseline] + violations:
            if res is not None and res.fromto is None and not res.unevaluable:
                g1, g2 = self._key_ids[res.key]
                res.fromto = self._witness(g1, g2, self.distmax)

        report = SampleReport(
            index=index,
            fraction=fraction,
            qpos=pose,
            min_distance=self.distmax if min_res is None else min_res.distance,
            min_pair=min_res,
            violations=violations,
            suspect_pairs=suspects,
            tool_min_distance=None if tool_min is None else tool_min.distance,
            tool_min_pair=tool_min,
            worst_baseline=worst_baseline,
            n_pairs_evaluated=int(candidates.shape[0]),
        )
        if key is not None:
            memo[key] = report
        return report

    def check_path(
        self,
        start_qpos_dict: Optional[Mapping[str, float]] = None,
        end_qpos_dict: Optional[Mapping[str, float]] = None,
        steps: int = 33,
        max_joint_step: Optional[float] = 0.01,
        max_samples: Optional[int] = DEFAULT_MAX_SAMPLES,
        early_exit: bool = True,
    ) -> ClearanceReport:
        """Linearly interpolate between two poses and evaluate the samples.

        This is the whole point of the module.  The 2026-08-07 collision had a clear start
        pose and a clear end pose; only the interior was in collision.

        `steps` is the number of samples, endpoints included (minimum 2).  When
        `max_joint_step` is set (default 0.01 rad) the sample count is increased if
        needed so that no joint moves further than that between consecutive samples —
        silent under-sampling is exactly the failure this module exists to prevent.

        `max_samples` bounds that increase.  Without it the count is a function of joint
        travel alone, so a big commanded jump costs unbounded time inside a 33 ms frame
        (measured: 208 samples, 1070 ms, on one live frame).  It caps ONLY the automatic
        refinement — an explicit `steps=` is a deliberate resolution request and is always
        honoured — and `None` restores the old unbounded behaviour.  When the cap bites,
        the sweep is coarser than `max_joint_step` asked for and could step over a thin
        collision, so the report says so: `sampling_degraded`, `degraded_reason`, an entry
        in `limit_warnings`, a banner in `format()`, and `pass_is_certified` False.  The
        degradation is never treated as equivalent to a full sweep.

        `early_exit` stops the sweep at the first sample that CONFIRMS a violation.  That
        verdict cannot be reversed by a later sample, so the rest is pure cost — and it is
        the deeply-colliding poses that were costing the most, which is backwards.  It can
        only ever shorten a FAIL: a clean sample never terminates the loop, so a PASS
        still evaluates every planned sample and can never become a pass-by-not-looking.
        What it does cost is the headline — see `ClearanceReport.early_exit`.

        `n_samples` is the planned count; `samples_evaluated` is how many ran.
        """
        if steps < 2:
            raise ValueError("steps must be >= 2 (both endpoints must be evaluated)")
        if max_samples is not None and int(max_samples) < 2:
            raise ValueError(
                "max_samples must be >= 2 (both endpoints must be evaluated) or None "
                "for no cap"
            )

        start = self.full_pose(start_qpos_dict)
        end = self.full_pose(end_qpos_dict)
        names = sorted(set(start) | set(end))
        a = np.array([start[n] for n in names], dtype=np.float64)
        b = np.array([end[n] for n in names], dtype=np.float64)
        travel = float(np.max(np.abs(b - a))) if names else 0.0

        requested_step = (
            float(max_joint_step) if max_joint_step and max_joint_step > 0 else None
        )
        n_samples = int(steps)
        samples_needed = n_samples
        if requested_step is not None:
            samples_needed = int(np.ceil(travel / requested_step)) + 1
            allowed = (
                samples_needed
                if max_samples is None
                else min(samples_needed, int(max_samples))
            )
            n_samples = max(n_samples, allowed)
        actual_step = travel / (n_samples - 1) if n_samples > 1 else 0.0
        degraded = requested_step is not None and n_samples < samples_needed
        degraded_reason = ""
        if degraded:
            degraded_reason = (
                "sampling degraded: max_joint_step={:.4f} rad needed {} samples, "
                "capped to {} (max_samples={}, steps={}); the largest joint step actually "
                "checked is {:.4f} rad, {:.1f}x coarser than requested. A collision "
                "thinner than that step could lie between two samples and was not looked "
                "at, so a clear verdict here is NOT equivalent to a full sweep."
            ).format(
                requested_step,
                samples_needed,
                n_samples,
                max_samples,
                int(steps),
                actual_step,
                (actual_step / requested_step) if requested_step else float("inf"),
            )

        warnings = []
        for pose in (start, end):
            for w in self.limit_warnings(pose):
                if w not in warnings:
                    warnings.append(w)
        if degraded_reason:
            warnings.append(degraded_reason)

        samples: List[SampleReport] = []
        exited_early = False
        for i, t in enumerate(np.linspace(0.0, 1.0, n_samples)):
            q = a + (b - a) * t
            sample = self.check_pose(
                dict(zip(names, (float(x) for x in q))), index=i, fraction=float(t)
            )
            samples.append(sample)
            if early_exit and sample.violations:
                # A CONFIRMED violation, never a suspicion and never an absence: this
                # branch is reachable only with a non-empty `violations` list, so the
                # report is already `ok=False` and no later sample could change it.
                exited_early = i < n_samples - 1
                break

        violations = [v for s in samples for v in s.violations]
        suspects = [p for s in samples for p in s.suspect_pairs]

        best_idx, best = 0, None
        for s in samples:
            if s.min_pair is not None and (best is None or s.min_pair.distance < best.distance):
                best, best_idx = s.min_pair, s.index
        tool_best = None
        for s in samples:
            if s.tool_min_pair is not None and (
                tool_best is None or s.tool_min_pair.distance < tool_best.distance
            ):
                tool_best = s.tool_min_pair
        base_worst = None
        for s in samples:
            if s.worst_baseline is not None and (
                base_worst is None
                or s.worst_baseline.baseline_delta < base_worst.baseline_delta
            ):
                base_worst = s.worst_baseline

        return ClearanceReport(
            ok=not violations,
            min_distance=self.distmax if best is None else best.distance,
            min_pair=best,
            min_sample_index=best_idx,
            min_fraction=samples[best_idx].fraction,
            n_samples=n_samples,
            steps_requested=int(steps),
            samples=samples,
            violations=violations,
            tool_violations=[v for v in violations if v.is_tool],
            structural_violations=[v for v in violations if not v.is_tool],
            suspect_pairs=suspects,
            tool_min_distance=None if tool_best is None else tool_best.distance,
            tool_min_pair=tool_best,
            worst_baseline=base_worst,
            limit_warnings=warnings,
            config={
                "floor": self.floor,
                "distmax": self.distmax,
                "baseline_margin": self.baseline_margin,
            },
            samples_evaluated=len(samples),
            samples_needed=samples_needed,
            max_samples=None if max_samples is None else int(max_samples),
            max_joint_step_requested=requested_step,
            max_joint_step_actual=actual_step,
            sampling_degraded=degraded,
            degraded_reason=degraded_reason,
            early_exit=exited_early,
        )


# --------------------------------------------------------------------------------------
# Module-level convenience
# --------------------------------------------------------------------------------------

_DEFAULT: Dict[Tuple, ClearanceChecker] = {}


def default_checker(**kwargs) -> ClearanceChecker:
    """A cached `ClearanceChecker`.  Building one costs a baseline sweep, so reuse it."""
    key = tuple(sorted(kwargs.items()))
    if key not in _DEFAULT:
        _DEFAULT[key] = ClearanceChecker(**kwargs)
    return _DEFAULT[key]


def check_path(
    start_qpos_dict: Optional[Mapping[str, float]] = None,
    end_qpos_dict: Optional[Mapping[str, float]] = None,
    steps: int = 33,
    max_joint_step: Optional[float] = 0.01,
    max_samples: Optional[int] = DEFAULT_MAX_SAMPLES,
    early_exit: bool = True,
    **checker_kwargs,
) -> ClearanceReport:
    """Check a linear joint-space path.  See `ClearanceChecker.check_path`."""
    return default_checker(**checker_kwargs).check_path(
        start_qpos_dict,
        end_qpos_dict,
        steps=steps,
        max_joint_step=max_joint_step,
        max_samples=max_samples,
        early_exit=early_exit,
    )


def check_pose(
    qpos_dict: Optional[Mapping[str, float]] = None, **checker_kwargs
) -> SampleReport:
    """Check a single pose.  See `ClearanceChecker.check_pose`."""
    return default_checker(**checker_kwargs).check_pose(qpos_dict)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_assignments(items: Optional[Sequence[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected joint=value, got {item!r}")
        name, _, val = item.partition("=")
        try:
            out[name.strip()] = float(val)
        except ValueError:
            raise SystemExit(f"not a number: {item!r}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Minimum-clearance checker for the Galbot G1 MuJoCo model.",
        epilog="Unspecified joints default to the 2026-08-07 18:06 hardware snapshot.",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--start", nargs="+", metavar="JOINT=VAL",
                    help="path start, as deviations from the home pose")
    ap.add_argument("--end", nargs="+", metavar="JOINT=VAL", help="path end")
    ap.add_argument("--pose", nargs="+", metavar="JOINT=VAL",
                    help="check a single pose instead of a path")
    ap.add_argument("--sweep", nargs=3, metavar=("JOINT", "FROM", "TO"),
                    help="shorthand for --start JOINT=FROM --end JOINT=TO")
    ap.add_argument("--steps", type=int, default=33)
    ap.add_argument("--max-joint-step", type=float, default=0.01,
                    help="refine sampling so no joint moves more than this per step "
                         "(rad; 0 disables)")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap on how far --max-joint-step may inflate the sample count "
                         f"(0 = uncapped, the CLI default; the library default is "
                         f"{DEFAULT_MAX_SAMPLES}, chosen for a 30 FPS loop). This tool is "
                         "an offline auditor, so it sweeps the whole path by default and "
                         "never reports a degraded clear unless you ask it to.")
    ap.add_argument("--early-exit", action="store_true",
                    help="stop at the first confirmed violation instead of sweeping the "
                         "whole path. Faster, but the reported minimum is then the worst "
                         "seen BEFORE the decision, not the worst on the path. Off here, "
                         "on by default in the library.")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR_M,
                    help="minimum acceptable clearance in metres (default 0)")
    ap.add_argument("--distmax", type=float, default=0.05,
                    help="distance search window in metres (default 0.05)")
    ap.add_argument("--baseline-margin", type=float, default=0.002,
                    help="how much deeper a baseline overlap may get before it fails (m)")
    ap.add_argument("--verbose", action="store_true", help="print every sample")
    args = ap.parse_args(argv)

    try:
        ck = ClearanceChecker(
            model_path=args.model,
            floor=args.floor,
            distmax=args.distmax,
            baseline_margin=args.baseline_margin,
        )
    except Exception as exc:
        print(f"could not build checker: {exc}", file=sys.stderr)
        return 2

    print(f"model: {ck.model_path}")
    print(
        f"collision geoms: {len(ck._geom_ids)}   pairs checked: {ck.n_pairs}   "
        f"structurally adjacent pairs excluded: {ck.n_adjacent_excluded}"
    )
    base = ck.baseline_pairs()
    print(f"baseline overlaps at the home pose: {len(base)} (tracked, not whitelisted)")
    for key, val in sorted(base.items(), key=lambda kv: kv[1]):
        print(f"    {key.body1} | {key.body2}   {val * 1000:+.2f} mm")
    if ck.baseline_unevaluable:
        print(f"    ! {len(ck.baseline_unevaluable)} pair(s) unevaluable at home "
              "— they will always be reported as violations")
    print()

    try:
        if args.sweep:
            joint, lo, hi = args.sweep[0], float(args.sweep[1]), float(args.sweep[2])
            start, end = {joint: lo}, {joint: hi}
        elif args.pose:
            start = end = _parse_assignments(args.pose)
        else:
            start = _parse_assignments(args.start)
            end = _parse_assignments(args.end) if args.end else dict(start)
        rep = ck.check_path(
            start, end, steps=args.steps,
            max_joint_step=(args.max_joint_step or None),
            max_samples=(args.max_samples or None),
            early_exit=bool(args.early_exit),
        )
    except Exception as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 2

    print(rep.format())
    if args.verbose:
        print("\nper-sample minimum clearance:")
        for s in rep.samples:
            tag = "VIOLATION" if s.violations else "ok"
            pair = f"{s.min_pair.key.body1} | {s.min_pair.key.body2}" if s.min_pair else "-"
            print(f"  [{s.index:3d}] t={s.fraction:.3f}  {s.min_distance * 1000:+9.2f} mm  "
                  f"{pair:45s} {tag}")
    return 1 if not rep.ok else 0


if __name__ == "__main__":
    sys.exit(main())
