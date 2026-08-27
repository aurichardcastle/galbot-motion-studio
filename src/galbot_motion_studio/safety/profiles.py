"""Motion profiles: one gate, two declared envelopes.

The supervisor's default rate and acceleration caps were chosen for a physical
robot. Applied to a simulation they are the reason the demo looks like a statue:

    policy default   max_joint_rate = 0.35 rad/s    max_accel = 1.0 rad/s^2
    the model itself  velocity      = 1.5  rad/s    accel     = 2.0 rad/s^2

That is **4.3x tighter than the robot's own declared velocity limit**, and
``cli.py`` was slowed to 5 Hz to stay underneath it. Natural hand-swaying at
roughly 1 Hz needs on the order of 1-2 rad/s at the shoulder, so the cap -- not
the retargeting, not the solver -- is what bounds expressiveness.

The fix is not to loosen safety. It is that **one policy cannot serve both a
simulation and a robot**, and pretending otherwise forces a choice between a
demo nobody can see and an envelope nobody should ship to hardware.

The invariant that makes this safe
----------------------------------
**A profile may relax the dynamics envelope, and exactly one geometric
threshold: the self-clearance floor. Nothing else, ever.**

The second clause is new, and it is a deliberate weakening of what this module
used to promise ("it may never relax a geometric or structural guarantee").
That promise was too strong to survive contact with a measurement -- see *Why
SIM needs its own clearance floor* below -- but weakening it silently would be
the exact failure this file exists to prevent. So the floor is named, bounded,
validated at import, and enumerated in its own frozen set.

Relaxable dynamics (how fast the target may move) -- ``SupervisorPolicy`` fields,
enumerated in ``RELAXABLE_FIELDS``:
    ``max_joint_rate_rad_s``          -- rate limiter
    ``max_joint_acceleration_rad_s2`` -- acceleration limiter
    ``max_joint_step_rad``            -- per-command discontinuity limiter

Relaxable geometry (how close two links may come) -- ``ClearanceChecker``
constructor parameters, enumerated in ``RELAXABLE_CLEARANCE_PARAMS``:
    ``floor``                         -- minimum-clearance threshold

Never relaxable, identical in every profile:
    ``soft_margin_rad``       -- joint-limit margin. 0.09 rad, because
                                 ``left_arm_joint4`` faulted into ``DriverError``
                                 2.1 degrees from its mechanical stop under
                                 gravity load on 2026-08-07. The margin must
                                 exceed where the drive gives up.
    ``max_ik_residual_m``     -- a converged-but-poor solution is a silently
                                 wrong pose regardless of how fast it is reached.
    ``clearance_match_tolerance_m`` -- geometry binding.
    ``distmax``, ``baseline_margin``, ``ladder_rungs``, ``tool_patterns``,
    ``collision_groups`` -- these decide what the checker is *able to see*.
                                 Widening them through a profile would weaken the
                                 measurement itself rather than the threshold it
                                 is compared against, and no measurement can
                                 justify that.
    the clearance gate itself, the joint hard limits, and fail-closed behaviour.

``test_profiles.py`` asserts this invariant over both surfaces -- the
``SupervisorPolicy`` dataclass and the ``ClearanceChecker`` signature -- and
additionally pins the contents of both relaxable sets by name. Adding a third
relaxable knob therefore means editing a literal set in a test, with the
evidence in hand, rather than discovering later that a profile quietly moved a
safety threshold.

Numbers, and where they come from
---------------------------------
``SIM`` dynamics are read from the shipped URDF, not invented. All nine
controlled joints (``head_joint1``, ``head_joint2``, ``left_arm_joint1..7``)
declare ``velocity="1.5"``; the seven arm joints declare ``acceleration="2"`` and
``jerk="12"``. ``SIM`` therefore sets the ceiling *at* the model's own limit and
no higher -- the simulation is allowed to move the robot exactly as fast as the
robot's description says the robot can move, and not one rad/s faster.

``HARDWARE`` keeps the existing conservative values unchanged and remains the
default. Nothing about real dynamics has been measured: the sustained command
rate of ``set_joint_positions`` is unknown, ``t_motion`` is unbounded (there is
no known cancel API), and the only hardware excursion ever executed was 22.8
degrees. Opening the hardware envelope would be guessing with a robot.

Why SIM needs its own clearance floor
-------------------------------------
``clearance.DEFAULT_FLOOR_M`` is 10 mm, chosen from the home pose alone. Swept
across a range of real motion it turns out to be too large a number for this
robot's shoulder geometry, and the symptom is a demo that intermittently stops
following the operator.

**What the floor is competing against.** ``min_distance`` is not free to grow.
``leg_link3_collision_1 | torso_base_link_collision_0`` sits at a pose-invariant
**12.407 mm** (measured; spread 0.0000 mm across 40 arm poses, because the leg is
never commanded in teleop) and it is an ordinary checked pair, so it caps the
headline clearance number at every pose. The entire observable range is
``(0, 12.407] mm``, and a 10 mm floor consumes **80.6%** of it. It is not "10 mm
of margin"; it is 2.4 mm of usable signal.

**The structural noise band -- the hard lower bound.** Several pairs are
permanently close and carry no information about self-collision. Measured at the
home pose:

    intra-gripper four-bar linkages          0.180 mm  (372 pairs, already
                                                        excluded by
                                                        ``ASSEMBLY_PREFIXES``)
    ``right_arm_link5 | right_arm_link7``    1.171 mm, 1.412 mm
    ``left_arm_link5  | left_arm_link7``     1.664 mm, 1.731 mm
    ``torso_base_link | head_link2``       -73.665 mm  (the head geometry is
                                                        embedded in the torso)
    ``torso_base_link | *_arm_link1``        never evaluated -- every one of its
                                                        2x4 geom pairs is
                                                        weld-adjacent, so the
                                                        kinematic-tree rule
                                                        already removes it.

A pair below the floor at the home pose is recorded as *baseline* and gets the
relative-worsening check instead of a floor comparison. The floor must therefore
sit **above 1.731 mm**, the deepest of those pairs, or they leave the baseline
set, become floor-compared, and their ~2 mm narrowphase wobble fires forever.
Measured directly: at a 1.50 mm floor, **120 of 120** jittered raised-hands poses
are rejected, all on ``left_arm_link5 | left_arm_link7``. At 1.80 mm all five
pairs are baseline-tracked again; at 3.00 mm, 0 of 120 are rejected.

**What 10 mm actually rejects.** A raised-both-hands pose derived from recorded
operator landmarks clears by **10.119 mm** -- 0.119 mm of headroom. That number
is a genuine measurement, not a numerics artifact: ``mj_geomDistance`` on
``torso_base_link_collision_0 | right_arm_link2_collision_2`` returns 10.119 mm
consistently for every ``distmax`` from 12.5 mm through 120 mm. The right upper
arm really does pass about 10 mm from the torso shell when a person raises both
hands, and that pose is *visually perfect*. Consequences, measured:

    jitter of sigma = 0.02 rad (1.1 deg) about that pose   63/200 rejected (31.5%)
    the swept path home -> raised hands                    FAILS
    every one of those 63 rejections came from one pair,
    ``right_arm_link2 | torso_base_link``

That is the "the robot randomly stops following me" report, exactly. Note the
static pose itself passes, at 10.119 mm against 10.000 mm -- it is only *motion*
that fails, which is why a single-pose check never caught this.

**The floor sweep.** 612 clear and 283 penetrating poses, labelled by an
independent criterion (does any checked pair actually interpenetrate), drawn
from the retargeter's own output, interpolations between its poses, per-frame
jitter, and single-joint sweeps of +/-60 deg about home:

    floor    clear poses rejected      penetrating poses rejected
     2 mm       5/612   ( 0.8%)              283/283  (100%)
     3 mm       5/612   ( 0.8%)              283/283  (100%)
     5 mm       5/612   ( 0.8%)              283/283  (100%)
     6 mm       8/612   ( 1.3%)              283/283  (100%)
     8 mm      17/612   ( 2.8%)              283/283  (100%)
    10 mm     194/612   (31.7%)              283/283  (100%)
    12 mm     339/612   (55.4%)              283/283  (100%)

**Every floor from 2 mm to 12 mm rejects 100% of genuine self-collisions.** The
floor's discriminating power does not depend on its value anywhere in that
range; only its false-rejection rate does, and that rate has a cliff between
8 mm and 10 mm where it jumps 11.4x. The two classes are cleanly separated -- the
worst clear pose measures +0.843 mm, the best penetrating pose -0.376 mm -- which
is why moving the threshold inside that gap costs nothing. Lowering the floor
buys acceptance of natural motion and gives up no detection.

**Why 5 mm.** It is 2.9x the 1.731 mm structural noise ceiling; it sits below the
1st percentile of the clear-pose distribution (5.648 mm), so ordinary teleop
never approaches it; it gives the raised-hands pose 5.119 mm of headroom instead
of 0.119 mm; it takes jitter rejection from 63/200 to 0/200 and the swept path
from FAIL to PASS; and it still rejects all 283 penetrating poses -- 5 mm and
2 mm reject exactly the same 5 clear poses, so nothing is bought by going lower
either. It is exactly half the hardware floor, which is the point: it should read
as a simulation number, not as a slightly nicer safety number.

**Why only in SIM.** The two floors protect against different things. In
simulation nothing can be damaged, so the floor exists to stop *visually wrong*
output -- an arm through the torso -- and the sweep above shows 5 mm does that as
completely as 10 mm. On hardware the floor stands between a mispredicted pose and
a bent link; the tool geometry does not match this model at all (generic grippers
here, a HUILING gripper and a suction cup on the robot); and the honest position
is that 10 mm was never validated as *sufficient* for hardware either, only as
arbitrary. So hardware keeps it, and ``HARDWARE_FLOOR_RATCHET_M`` makes it
structurally impossible for the hardware floor to fall below today's value even
if ``clearance.DEFAULT_FLOOR_M`` is lowered later.

What this does not change: the gate still runs on every sample of every swept
path, still fails closed, still rejects unevaluable pairs, and still tracks the
baseline overlaps. One threshold moves, and only in simulation.
"""

from __future__ import annotations

from dataclasses import replace
from enum import StrEnum

from galbot_motion_studio.safety.clearance import DEFAULT_FLOOR_M
from galbot_motion_studio.safety.supervisor import SupervisorPolicy

# --- from third_party/galbot_one_golf_description/urdf/galbot_one_golf_fixed_base.urdf
# head_joint1/2:        velocity="1.5"   (no acceleration or jerk declared)
# left_arm_joint1..7:   velocity="1.5"   acceleration="2"   jerk="12"
URDF_JOINT_VELOCITY_RAD_S = 1.5
URDF_ARM_ACCELERATION_RAD_S2 = 2.0

#: ``SupervisorPolicy`` fields a profile is permitted to change. Anything outside
#: this set is a safety invariant and must be byte-identical across every profile.
RELAXABLE_FIELDS = frozenset(
    {"max_joint_rate_rad_s", "max_joint_acceleration_rad_s2", "max_joint_step_rad"}
)
SIM_MAX_JOINT_STEP_RAD = 0.10

#: ``ClearanceChecker`` constructor parameters a profile is permitted to change.
#:
#: Separate from ``RELAXABLE_FIELDS`` because it is a different kind of
#: permission over a different object, and collapsing the two would let a
#: geometric threshold ride in on a dynamics argument. ``floor`` is the only
#: member; see the module docstring for the measurement that earned it.
RELAXABLE_CLEARANCE_PARAMS = frozenset({"floor"})

#: Deepest permanently-close pair in the model, measured at the home pose:
#: ``left_arm_link5 | left_arm_link7`` at 1.731 mm. Pairs at or below the floor at
#: home are baseline-tracked rather than floor-compared, so a floor must exceed
#: this or the model's own fixed structure is reported as a developing collision.
#: Measured consequence of getting it wrong: a 1.50 mm floor rejects 120 of 120
#: jittered raised-hands poses.
STRUCTURAL_NOISE_CEILING_M = 0.001731

#: Minimum ratio between a profile's floor and the structural noise ceiling.
#: ``mj_geomDistance`` lower bounds on the rigid ``*_arm_link5 | *_arm_link7``
#: pairs wobble by roughly 2 mm across arm poses (see ``ASSEMBLY_PREFIXES`` in
#: ``clearance.py``), so a floor merely *above* 1.731 mm is not enough; it has to
#: clear the wobble as well.
MIN_FLOOR_NOISE_RATIO = 2.5

#: The hardware floor may be tightened but never loosened, whatever
#: ``clearance.DEFAULT_FLOOR_M`` becomes. This is today's value, pinned here so
#: that lowering the shared default cannot silently widen the envelope a physical
#: robot runs under.
HARDWARE_FLOOR_RATCHET_M = 0.010

#: Hardware self-clearance floor. Follows ``clearance.DEFAULT_FLOOR_M`` upward,
#: never downward.
HARDWARE_CLEARANCE_FLOOR_M = max(DEFAULT_FLOOR_M, HARDWARE_FLOOR_RATCHET_M)

#: Simulation self-clearance floor, in metres. See the module docstring.
SIM_CLEARANCE_FLOOR_M = 0.005


class MotionProfile(StrEnum):
    """Declared motion envelope. Recorded in every clip so a dataset can never
    be mistaken for one captured under hardware limits."""

    HARDWARE = "hardware"
    SIM = "sim"


def _validate_floors() -> None:
    """Fail at import if the floor constants stop making sense.

    A clearance floor is only meaningful between two bounds and both are
    load-bearing: below the structural noise it fires on the model's own fixed
    geometry, and above the hardware floor a simulation profile would be
    claiming to be safer than the robot. Checked at import rather than only in
    tests, so that an edit to these constants cannot ship behind a skipped run.
    """
    # Hardware first: it is the more fundamental invariant, and if it is ever
    # violated that is what the reader needs to be told about, not a knock-on
    # complaint that the simulation floor now compares badly against it.
    if HARDWARE_CLEARANCE_FLOOR_M < HARDWARE_FLOOR_RATCHET_M:
        raise ValueError(
            "the hardware clearance floor may be tightened but never loosened"
        )
    if not SIM_CLEARANCE_FLOOR_M > 0.0:
        raise ValueError(
            "SIM_CLEARANCE_FLOOR_M must be > 0; a floor of zero is contact "
            "detection, which is the semantics clearance.py exists to replace"
        )
    if SIM_CLEARANCE_FLOOR_M < MIN_FLOOR_NOISE_RATIO * STRUCTURAL_NOISE_CEILING_M:
        raise ValueError(
            f"SIM_CLEARANCE_FLOOR_M ({SIM_CLEARANCE_FLOOR_M}) is within "
            f"{MIN_FLOOR_NOISE_RATIO}x of the structural noise ceiling "
            f"({STRUCTURAL_NOISE_CEILING_M}); the permanently-close "
            "*_arm_link5 | *_arm_link7 pairs would leave the baseline set and "
            "the gate would reject continuously"
        )
    if SIM_CLEARANCE_FLOOR_M > HARDWARE_CLEARANCE_FLOOR_M:
        raise ValueError(
            "SIM_CLEARANCE_FLOOR_M must not exceed HARDWARE_CLEARANCE_FLOOR_M; a "
            "split that made simulation stricter than the robot would mean "
            "hardware runs on the less-tested threshold"
        )


_validate_floors()


def policy_for(profile: MotionProfile | str, base: SupervisorPolicy | None = None) -> SupervisorPolicy:
    """Return the supervisor policy for ``profile``.

    ``HARDWARE`` returns ``base`` unchanged -- today's conservative envelope,
    and the default everywhere. ``SIM`` widens only declared dynamics fields.

    The clearance floor is deliberately not here: ``SupervisorPolicy`` does not
    carry one, the ``ClearanceChecker`` does. See :func:`clearance_kwargs_for`.

    Raises ``ValueError`` on an unknown profile rather than falling through to a
    default, because a typo silently selecting the permissive envelope is
    exactly the failure this module exists to prevent.
    """
    base = base or SupervisorPolicy()
    profile = MotionProfile(profile)  # raises ValueError on anything unrecognised

    if profile is MotionProfile.HARDWARE:
        return base
    return replace(
        base,
        max_joint_rate_rad_s=URDF_JOINT_VELOCITY_RAD_S,
        max_joint_acceleration_rad_s2=URDF_ARM_ACCELERATION_RAD_S2,
        # One latest-wins drop at 30 Hz spans two 0.05 rad periods while still
        # satisfying the model rate and acceleration ceilings. Hardware retains
        # the conservative 0.05 rad discontinuity cap.
        max_joint_step_rad=SIM_MAX_JOINT_STEP_RAD,
    )


def clearance_floor_for(profile: MotionProfile | str) -> float:
    """Return the minimum self-clearance floor, in metres, for ``profile``.

    ``HARDWARE`` is 10 mm -- ``clearance.DEFAULT_FLOOR_M``, ratcheted so it can
    only ever rise. ``SIM`` is 5 mm; the module docstring carries the swept
    measurement that justifies it.

    Raises ``ValueError`` on an unknown profile, for the same reason
    :func:`policy_for` does.
    """
    profile = MotionProfile(profile)
    if profile is MotionProfile.HARDWARE:
        return HARDWARE_CLEARANCE_FLOOR_M
    return SIM_CLEARANCE_FLOOR_M


def clearance_kwargs_for(profile: MotionProfile | str) -> dict[str, float]:
    """Keyword arguments for this profile's ``ClearanceChecker``.

        checker = ClearanceChecker(model=model, home=HOME_QPOS,
                                   **clearance_kwargs_for(profile))

    Only the floor is profile-dependent. Every other constructor argument -- the
    search window, the baseline margin, the ladder, the tool patterns, the
    collision groups -- is deliberately absent so it keeps its default and stays
    identical in every profile. Returning a dict rather than a built checker
    keeps this module free of a MuJoCo model load.
    """
    return {"floor": clearance_floor_for(profile)}


def describe(profile: MotionProfile | str) -> str:
    """One-line human summary, for the UI and for clip metadata."""
    profile = MotionProfile(profile)
    p = policy_for(profile)
    floor_mm = clearance_floor_for(profile) * 1000.0
    suffix = " (model URDF limits)" if profile is MotionProfile.SIM else " (conservative)"
    return (
        f"{profile.value}: rate <= {p.max_joint_rate_rad_s} rad/s, "
        f"accel <= {p.max_joint_acceleration_rad_s2} rad/s^2{suffix}, "
        f"clearance floor >= {floor_mm:g} mm"
    )
