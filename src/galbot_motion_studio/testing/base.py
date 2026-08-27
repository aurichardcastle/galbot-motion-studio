"""Robot adapter protocol for the Galbot G1, modelled on the real SDK surface.

This module defines the *shape* of the hardware boundary. It contains no fault
behaviour -- see ``fault_injection.py`` for the test double that misbehaves the
way the real robot was measured to misbehave on 2026-08-07.

Why the protocol exists
-----------------------
The teleop stack runs offline against a simulator that always completes and
always returns SUCCESS. Real hardware does neither. Everything downstream of
this boundary must be written against the protocol, never against the concrete
simulator, so that the safety layer can be driven by a fake that fails.

Provenance of every value in this file
--------------------------------------
MEASURED   -- retained hardware observations, not included in this repository.
DOCUMENTED -- public model and interface documentation.
LOCAL      -- a choice made by this module. Not measured, not documented.
              Every LOCAL choice is flagged inline so nobody mistakes it for evidence.
"""

from __future__ import annotations

import enum
from typing import Mapping, Protocol, Sequence, runtime_checkable

__all__ = [
    "ControlStatus",
    "MOTION_FAILURE_STATUSES",
    "G1_JOINT_GROUPS",
    "DEFAULT_CONTROL_GROUPS",
    "GROUP_JOINTS",
    "JOINT_LIMITS",
    "MEASURED_POSE_2026_08_07_1806",
    "JointLimit",
    "RobotAdapter",
    "AdapterError",
    "UninitializedGetterError",
    "MissingTeardownError",
    "ToleranceUnknownError",
    "limits_for_group",
    "clamp_to_limits",
]


# --------------------------------------------------------------------------
# Status enum
# --------------------------------------------------------------------------
class ControlStatus(enum.Enum):
    """Execution status of a robot control command.

    DOCUMENTED: the full member list is transcribed from the SDK 1.9.0
    ``ControlStatus`` enum reference. The task brief requires SUCCESS, TIMEOUT,
    FAULT and INVALID_INPUT at minimum; the remaining members are included so
    that code written against this protocol does not assume the enum is closed.
    A safety layer that ``match``es on four names and falls through on a fifth
    is a bug waiting for hardware to find it.

    Only three of these were ever observed from ``set_joint_positions`` on real
    hardware -- SUCCESS, TIMEOUT and FAULT (VISIT2_FIELD_LOG.md, 19:10 and
    19:42). INVALID_INPUT was observed from ``forward_kinematics`` with a bad
    frame name (transcript lines 2538-2542). The rest are DOCUMENTED only:
    this project has never seen them.
    """

    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    FAULT = "FAULT"
    INVALID_INPUT = "INVALID_INPUT"
    # Documented but never observed by this project:
    COMM_DISCONNECTED = "COMM_DISCONNECTED"
    DATA_FETCH_FAILED = "DATA_FETCH_FAILED"
    INIT_FAILED = "INIT_FAILED"
    IN_PROGRESS = "IN_PROGRESS"
    PUBLISH_FAIL = "PUBLISH_FAIL"
    STOPPED_UNREACHED = "STOPPED_UNREACHED"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ControlStatus.{self.name}"


#: Every status that is not SUCCESS. Provided so callers can write
#: ``status in MOTION_FAILURE_STATUSES`` instead of ``status != SUCCESS``,
#: which reads the same but survives someone adding a second success-ish member.
MOTION_FAILURE_STATUSES = frozenset(s for s in ControlStatus if s is not ControlStatus.SUCCESS)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class AdapterError(RuntimeError):
    """Base class for every error this adapter layer raises."""


class UninitializedGetterError(AdapterError):
    """A getter was called before ``init()`` or after ``init()`` returned False.

    MEASURED (VISIT2_FIELD_LOG.md, "What survives as genuinely reportable", item 1):
    on real hardware this path does not raise. It dereferences null and kills
    the interpreter with SIGSEGV and no Python traceback. The SDK guards some
    entry points against uninitialised use but not the getters.

    A test double must be *observable* rather than fatal, so this exception
    stands in for the segfault. Failure-matrix row R2.
    """


class MissingTeardownError(AdapterError):
    """The documented shutdown sequence was not followed.

    DOCUMENTED (SDK 1.9.0, ``destroy``): "This is the last step of the shutdown
    sequence: request_shutdown() -> wait_for_shutdown() -> destroy()."

    MEASURED (VISIT2_FIELD_LOG.md, Retractions, "The 17:35 segfault was mine"):
    omitting the sequence produced SIGSEGV on exit; re-running the identical
    read *with* the sequence exited 0. A/B settled in one minute.

    Again, a test double must be observable, so the omission raises here rather
    than crashing the interpreter.
    """


class ToleranceUnknownError(AdapterError):
    """A judgement fell inside the *unmeasured* acceptance-tolerance band.

    The band is the open interval (0.0062, 0.0359) rad -- see
    ``fault_injection.ACCEPTANCE_TOLERANCE_*``. Two data points bound it; no
    data point sits inside it. Rather than invent a threshold, the fake refuses
    to answer by default and raises this. See
    ``fault_injection.UnknownBandPolicy`` for the alternatives.
    """


# --------------------------------------------------------------------------
# Joint model
# --------------------------------------------------------------------------
class JointLimit:
    """Lower/upper position limit for one joint, in radians.

    MEASURED/DOCUMENTED: extracted from ``urdf/galbot_one_golf.urdf`` in
    the vendored `galbot_one_golf_description` model package.

    Note the SDK has no joint-limit getter at all (field log finding 1d):
    ``KinematicsBoundary`` exists as a type but nothing on ``GalbotRobot``
    returns one. The URDF is the only published source for these numbers, and
    that is itself the reason this class lives here rather than being read from
    the robot.
    """

    __slots__ = ("name", "lower", "upper")

    def __init__(self, name: str, lower: float, upper: float) -> None:
        if lower > upper:
            raise ValueError(f"{name}: lower {lower} > upper {upper}")
        self.name = name
        self.lower = lower
        self.upper = upper

    @property
    def span(self) -> float:
        return self.upper - self.lower

    def clamp(self, value: float) -> float:
        return min(self.upper, max(self.lower, value))

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"JointLimit({self.name!r}, {self.lower!r}, {self.upper!r})"


#: DOCUMENTED: SDK 1.9.0 ``G1JointGroup`` enum. ``chassis`` is passive in
#: joint-position control.
G1_JOINT_GROUPS: tuple[str, ...] = (
    "chassis",
    "head",
    "left_arm",
    "left_dexhand",
    "left_gripper",
    "left_suction_cup",
    "leg",
    "right_arm",
    "right_dexhand",
    "right_gripper",
    "right_suction_cup",
)

#: DOCUMENTED: SDK example comment -- "Set head joint group; if empty, defaults
#: to whole body joints ["leg", "head", "left_arm", "right_arm"]".
DEFAULT_CONTROL_GROUPS: tuple[str, ...] = ("leg", "head", "left_arm", "right_arm")


def _arm_joints(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_arm_joint{i}" for i in range(1, 8))


#: Joint membership per group. MEASURED counts (field log 17:50): leg 5,
#: head 2, each arm 7, chassis 4 (chassis only with ``only_active_joint=False``).
#: The groups this fake models are the four that were exercised on hardware
#: plus chassis; the dexhand/gripper/suction groups are declared in
#: ``G1_JOINT_GROUPS`` but deliberately not modelled -- nothing about them was
#: measured, and a plausible-looking fake of an unmeasured group is worse than
#: no fake at all.
GROUP_JOINTS: Mapping[str, tuple[str, ...]] = {
    "leg": tuple(f"leg_joint{i}" for i in range(1, 6)),
    "head": ("head_joint1", "head_joint2"),
    "left_arm": _arm_joints("left"),
    "right_arm": _arm_joints("right"),
    "chassis": tuple(f"chassis_joint{i}" for i in range(1, 5)),
}

#: MEASURED: URDF <limit> tags in the vendored model.
#: The arms are mirrored, not identical -- joint4 and joint6 have their bounds
#: swapped between left and right. Copying a left-arm pose to the right by
#: negation is correct for those joints and would be wrong if limits were
#: symmetric.
JOINT_LIMITS: Mapping[str, JointLimit] = {
    limit.name: limit
    for limit in (
        JointLimit("leg_joint1", 0.0, 0.9374),
        JointLimit("leg_joint2", 0.0, 2.5847),
        JointLimit("leg_joint3", 0.0, 2.3262),
        JointLimit("leg_joint4", -1.5906, 1.5906),
        JointLimit("leg_joint5", -0.1645, 0.1645),
        JointLimit("head_joint1", -1.5208, 1.5208),
        # The joint that produced finding 1f. Strongly asymmetric.
        JointLimit("head_joint2", -0.2143461, 0.4935988),
        JointLimit("left_arm_joint1", -3.00432619, 3.00432619),
        JointLimit("left_arm_joint2", -1.608062789, 1.608062789),
        JointLimit("left_arm_joint3", -2.916972222, 2.916972222),
        JointLimit("left_arm_joint4", -2.5679938779914946, 1.869862177),
        JointLimit("left_arm_joint5", -2.916972222, 2.916972222),
        JointLimit("left_arm_joint6", -0.8226646259971648, 0.7353981633974483),
        JointLimit("left_arm_joint7", -1.538202778, 1.538202778),
        JointLimit("right_arm_joint1", -3.00432619, 3.00432619),
        JointLimit("right_arm_joint2", -1.608062789, 1.608062789),
        JointLimit("right_arm_joint3", -2.916972222, 2.916972222),
        JointLimit("right_arm_joint4", -1.869862177, 2.5679938779914946),
        JointLimit("right_arm_joint5", -2.916972222, 2.916972222),
        JointLimit("right_arm_joint6", -0.7353981633974483, 0.8226646259971648),
        JointLimit("right_arm_joint7", -1.538202778, 1.538202778),
    )
}

#: MEASURED: the 18:06 full-state snapshot on 2026-08-07 (transcript lines
#: 2031-2087, tabulated in the vendored model). Used as the fake's
#: start pose so that tests begin from a pose the robot actually held.
#: chassis is LOCAL: zeros, never measured, and passive anyway.
MEASURED_POSE_2026_08_07_1806: Mapping[str, float] = {
    "leg_joint1": 0.4980,
    "leg_joint2": 1.4961,
    "leg_joint3": 0.9994,
    "leg_joint4": -0.0006,
    "leg_joint5": -0.0003,
    "head_joint1": -0.0006,
    "head_joint2": -0.0002,
    "left_arm_joint1": 1.8199,
    "left_arm_joint2": -1.4698,
    "left_arm_joint3": -0.5624,
    "left_arm_joint4": -1.7742,
    "left_arm_joint5": -0.0077,
    "left_arm_joint6": -0.2748,
    "left_arm_joint7": 0.1114,
    "right_arm_joint1": -1.7820,
    "right_arm_joint2": 1.4846,
    "right_arm_joint3": 0.5983,
    "right_arm_joint4": 1.8141,
    "right_arm_joint5": -0.0069,
    "right_arm_joint6": 0.6207,
    "right_arm_joint7": -0.0406,
    "chassis_joint1": 0.0,
    "chassis_joint2": 0.0,
    "chassis_joint3": 0.0,
    "chassis_joint4": 0.0,
}


def limits_for_group(group: str) -> tuple[JointLimit, ...]:
    """Limits for every joint in ``group``, in the group's canonical order.

    Returns an empty tuple for an unknown group. That is deliberate and
    MEASURED: ``["legs"]`` (the documented-but-wrong plural, finding 1b)
    returned 0 joints on hardware with no exception raised.
    """
    return tuple(JOINT_LIMITS[name] for name in GROUP_JOINTS.get(group, ()) if name in JOINT_LIMITS)


def clamp_to_limits(group: str, positions: Sequence[float]) -> list[float]:
    """Clamp ``positions`` to the URDF limits of ``group``'s joints.

    This is the *mechanism* behind finding 1f, exposed as a plain function so
    a caller can pre-clamp and compare. Joints with no limit entry (chassis)
    pass through unchanged.
    """
    names = GROUP_JOINTS.get(group, ())
    out: list[float] = []
    for index, value in enumerate(positions):
        if index < len(names) and names[index] in JOINT_LIMITS:
            out.append(JOINT_LIMITS[names[index]].clamp(value))
        else:
            out.append(value)
    return out


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------
@runtime_checkable
class RobotAdapter(Protocol):
    """The hardware boundary. Everything above this is testable on a laptop.

    Divergences from the real SDK signature, all deliberate, all listed here so
    nobody has to diff the docs to find them:

    * ``get_joint_positions`` returns ``dict[str, list[float]]`` keyed by group.
      The SDK returns a flat ``list[float]`` whose meaning depends entirely on
      the caller having reconstructed the group expansion order correctly. The
      flat list is a foot-gun -- the SDK docs themselves warn that
      "joint_groups are expanded to concrete joint_names in group order" -- so
      this protocol keys the result and an implementation wrapping the real SDK
      is responsible for the expansion.
    * ``set_joint_positions`` takes ``joint_names`` as a keyword-only argument
      at the end rather than as the SDK's third positional parameter, so that
      the positional order matches the order used throughout this project
      (positions, groups, is_blocking, speed, timeout). SDK precedence is
      preserved: when ``joint_names`` is non-empty it overrides ``groups``.
    * ``timeout`` defaults to 20.0 s. NOTE: the SDK 1.9.0 documented default is
      ``timeout_s = 15.0``. The 20.0 here matches the value the 7 Aug scripts
      actually passed, which is the number the measured behaviour was produced
      under -- six printed timeouts at 20 s is the 120 s floor in the field
      log's abort analysis. Flagged because the two numbers disagree and the
      docs are not wrong; the scripts simply overrode them.
    """

    def init(self, sensor_set: Sequence[object] | None = None) -> bool:
        """Initialise the robot. Returns success as a bare bool.

        MEASURED (field log, reportable item 2): on failure this returns False
        with no exception and nothing on stdout or stderr. The reason is written
        to ``~/galbot_sdk_log/g1/`` and nothing tells the caller to look there.
        The natural next action -- calling a getter -- then crashes the process.

        MEASURED (field log, finding 1g / R6): a True return does NOT mean the
        robot will move. A latched controller fault leaves ``init()`` returning
        True and every getter returning correct live values.
        Never treat True as a health check.
        """
        ...

    def get_joint_positions(self, groups: Sequence[str] = ()) -> dict[str, list[float]]:
        """Current joint positions, keyed by group. Empty ``groups`` means
        ``DEFAULT_CONTROL_GROUPS``. Unknown group -> empty list, no exception."""
        ...

    def get_joint_group_names(self) -> list[str]:
        """Available joint group names."""
        ...

    def set_joint_positions(
        self,
        positions: Sequence[float],
        groups: Sequence[str] = (),
        is_blocking: bool = True,
        speed: float = 0.2,
        timeout: float = 20.0,
        *,
        joint_names: Sequence[str] = (),
    ) -> ControlStatus:
        """Command joint positions. Blocking by default.

        MEASURED: the status ladder is not what the names suggest.
        SUCCESS may not have reached the target (finding 1f);
        TIMEOUT means *partial* motion -- six of seven joints still reached
        home in the observed case; FAULT means zero motion, rejected outright.
        Read the joints back. Every time.
        """
        ...

    def request_shutdown(self) -> None:
        """Step 1 of 3. Sends the shutdown signal."""
        ...

    def wait_for_shutdown(self) -> None:
        """Step 2 of 3. Blocks until modules have shut down."""
        ...

    def destroy(self) -> None:
        """Step 3 of 3. MUST be last. Omitting the sequence SIGSEGVs on hardware."""
        ...
