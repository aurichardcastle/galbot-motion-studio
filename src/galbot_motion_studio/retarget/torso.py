"""Torso yaw mapping: turn the robot's upper body as the operator turns theirs.

The G1 has no waist joint.  Its entire upper body -- both arm mounts and the
head -- hangs off ``torso_base_link``, which is carried by a five-DOF ``leg_*``
column.  Measured on the canonical fixed-base model, the Jacobian of
``torso_base_link`` with respect to that column separates cleanly::

    leg_joint1..3   rotate about world Y and translate in X and Z  (lift + pitch)
    leg_joint4      pure rotation about world Z, zero translation  (YAW)
    leg_joint5      pure rotation about world X, zero translation  (ROLL)

Only ``leg_joint4`` is mapped here, and the reason is measurement, not caution
for its own sake.  On the 2026-08-22 trial (1192 frames with both shoulders
visible) the operator's own torso spanned 49.2 deg between p05 and p95 and
+-66.6 deg at p01/p99.  ``leg_joint4`` allows +-91.2 deg, so yaw fits with
headroom at unity gain.  Roll spanned 29.7 deg p05..p95 against ``leg_joint5``'s
+-9.4 deg of travel -- the actuator is smaller than the operator's resting
variation, so a roll mapping would spend most of its time clipped, which reads
as an end-stop rattle rather than as tracking.  The lift chain changes both the
arm workspace and the self-collision geometry and is a separate problem.

Yaw is a *rigid rotation of the whole upper body*, which is what makes it safe
to add to an arm pipeline that was built without it: the arms' shoulder-relative
targets are unchanged in the torso frame, so the caller must rotate its arm
offsets by the commanded yaw and solve arm IK against a base pose that carries
it.  Doing one without the other silently aims the arms at where the shoulder
used to be.  See ``TorsoTarget.yaw_rad``.

Semantics deliberately distinguish the robot's reachable range from the
camera's observable range.  A demand inside the calibrated, front-visible
camera span can saturate into the robot soft band.  A back-facing or edge-on
operator pose is *not* saturated: MediaPipe can relabel the shoulders there,
so it must be held by the caller rather than turned into a confident command in
the opposite direction.  The live trajectory governor, not this pure mapper,
applies the torso slew limit to the pose that is actually emitted.

Nothing here authorizes motion.  The independent safety supervisor still checks
every emitted pose, and the swept clearance check must be given the commanded
torso angle -- evaluating it at the home torso pose would be blind to precisely
the motion this module creates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import atan2, cos, isfinite, radians, sin, sqrt

from galbot_motion_studio.model.joint_map import JointLimit

#: The one joint that yaws the G1's upper body. Named here rather than in the
#: arm retargeter because the arms only need to know *which* joint carries
#: them; this module owns what that joint means.
TORSO_YAW_JOINT = "leg_joint4"

#: The body every arm mount and the head hang off. Its orientation IS the
#: torso frame, so callers read the frame from forward kinematics here rather
#: than reconstructing it from the joint angle.
TORSO_BODY = "torso_base_link"


class TorsoMapRejection(StrEnum):
    NONFINITE = "NONFINITE"


@dataclass(frozen=True)
class TorsoIntent:
    """Neutral-relative torso yaw signal in radians, not a calibrated angle.

    Positive is the operator's left shoulder moving toward the camera, i.e. the
    direction ``shoulder_yaw_signal`` produces for a leftward turn.
    """

    yaw_signal: float


def shoulder_yaw_signal(
    left_world_xyz_m: tuple[float, float, float],
    right_world_xyz_m: tuple[float, float, float],
    *,
    min_shoulder_width_m: float = 0.12,
) -> float | None:
    """Torso yaw from the shoulder line, in radians, or ``None`` if unusable.

    MediaPipe's ``world_xyz_m`` pose landmarks are metric and hip-centred, so the
    shoulder line's rotation about the vertical is directly the operator's torso
    yaw -- no calibration constant, and no dependence on how far away they stand.
    That last property is why this uses the world landmarks rather than the
    normalized ones: a normalized shoulder line shortens both when the operator
    turns and when they lean back, and those are not the same command.

    ``min_shoulder_width_m`` rejects a degenerate line rather than reporting the
    wildly wrong angle a near-zero denominator produces.  On the 2026-08-22 trial
    the shoulder width was 0.316 m at p50 but 0.134 m at p01, so short lines do
    occur and they are the frames that carry the 144 deg single-frame jumps.
    """
    values = (*left_world_xyz_m, *right_world_xyz_m)
    if not all(isfinite(value) for value in values):
        return None
    dx = left_world_xyz_m[0] - right_world_xyz_m[0]
    dz = left_world_xyz_m[2] - right_world_xyz_m[2]
    # ``atan2`` below uses the horizontal x/z projection.  Guard that exact
    # quantity, not the full 3-D line: a nearly vertical 13 cm shoulder line
    # otherwise passes the old guard, and a millimetre of z noise can swing its
    # reported yaw by 45 degrees.
    if sqrt(dx * dx + dz * dz) < min_shoulder_width_m:
        return None
    # atan2 against the horizontal extent: the shoulder line's depth difference
    # over its lateral extent IS the yaw.
    return atan2(dz, dx)


def wrapped_yaw_difference(observed_rad: float, reference_rad: float) -> float | None:
    """Return the principal observed-minus-reference yaw in ``[-pi, pi]``.

    A neutral shoulder line can legitimately lie either side of the atan2 wrap.
    Subtracting raw angles would turn a two-degree movement around +/-180
    degrees into a 358-degree robot command, so every calibration-relative
    comparison comes through this function.
    """
    if not (isfinite(observed_rad) and isfinite(reference_rad)):
        return None
    difference = observed_rad - reference_rad
    return atan2(sin(difference), cos(difference))


def stable_yaw_reference(
    samples_rad: tuple[float, ...],
    *,
    max_absolute_yaw_rad: float,
    max_spread_rad: float,
) -> float | None:
    """Circular mean for a square, still torso calibration window.

    This intentionally returns ``None`` rather than guessing when the window
    lacks an observable shoulder yaw, is pointed too far away from the camera,
    or rotates while it is meant to be neutral.  The pipeline then holds only
    the torso; head and arms still retain their independently valid calibration.
    """
    if not samples_rad or not all(isfinite(value) for value in samples_rad):
        return None
    if not (isfinite(max_absolute_yaw_rad) and max_absolute_yaw_rad > 0):
        raise ValueError("max_absolute_yaw_rad must be finite and positive")
    if not (isfinite(max_spread_rad) and max_spread_rad >= 0):
        raise ValueError("max_spread_rad must be finite and non-negative")
    mean_sin = sum(sin(value) for value in samples_rad)
    mean_cos = sum(cos(value) for value in samples_rad)
    # Opposite samples leave a tiny cosine residual on most platforms.  A
    # near-zero resultant has no defined circular direction, so fail closed
    # before it can masquerade as a neutral reference.
    if sqrt(mean_sin * mean_sin + mean_cos * mean_cos) <= 1e-12 * len(samples_rad):
        return None
    reference = atan2(mean_sin, mean_cos)
    if abs(reference) > max_absolute_yaw_rad:
        return None
    if any(
        (difference := wrapped_yaw_difference(value, reference)) is None
        or abs(difference) > max_spread_rad
        for value in samples_rad
    ):
        return None
    return reference


@dataclass(frozen=True)
class TorsoMappingPolicy:
    yaw_limit: JointLimit
    #: Unity. The operator's measured p01..p99 torso yaw is +-66.6 deg and the
    #: soft band on ``leg_joint4`` is +-86.0 deg, so 1.0 tracks one-for-one and
    #: still never saturates inside the observed range. A gain above 1.0 would
    #: buy expressiveness only by spending that entire margin.
    yaw_gain: float = 1.0
    #: 0.05 rad = 2.9 deg. Measured frame-to-frame yaw jitter on the same trial
    #: was 1.07 deg at p50 and 9.21 deg at p95, so this sits above the resting
    #: noise while costing under 6% of the 49.2 deg p05..p95 working span. It
    #: does not replace the caller's filter; it stops a motionless operator from
    #: dithering the heaviest joint on the robot.
    deadband_rad: float = 0.05
    soft_margin_rad: float = 0.09
    #: A direct shoulder-line yaw is only unambiguous while the person remains
    #: substantially front-facing.  Beyond this the detector can relabel left
    #: and right shoulders, which is indistinguishable from an opposite turn
    #: from the line alone.  The pipeline holds the torso outside this range.
    max_camera_yaw_rad: float = radians(80.0)
    #: Operator-relative turn span that the robot is permitted to map.  Keeping
    #: this five degrees inside the raw camera span leaves the entire signed
    #: working range available after a valid (+/-5 degree) square calibration.
    max_relative_yaw_rad: float = radians(75.0)
    #: Calibration must be much squarer than the working span.  It prevents a
    #: side-on neutral from spending the whole observable range on one turn.
    max_neutral_camera_yaw_rad: float = radians(5.0)
    #: Calibration is a still window, not a trajectory.  A wider angular spread
    #: means there is no defensible neutral torso reference.
    max_neutral_spread_rad: float = 0.12
    #: 750 ms, derived from the MEASURED live cadence, not the retained 30 fps
    #: source cadence the old 150 ms came from. Over the 285 intervals of the
    #: 2026-08-26 camera trial (artifacts/test-run-2/live.json) the interval
    #: between processed observations was p50 98 ms, p95 364 ms, p99 569 ms,
    #: max 616 ms: 150 ms fired on 99 of 285 and was the top hold reason, and
    #: even 600 ms still clips the tail of NORMAL cadence (1 interval over).
    #: The horizon must sit above the whole normal-cadence distribution or it
    #: reports missing data that is not missing, so it is taken as the observed
    #: max plus ~20% for an unsampled tail: 616 ms -> 750 ms (0 of 285 exceed).
    #: That is still far below the multi-second dropout it exists to catch, and
    #: identity across the gap is guarded by the face-front requirement and the
    #: 20 deg semantic step, not by this number.
    max_observation_gap_ns: int = 750_000_000
    #: A source yaw cannot jump twenty degrees between accepted observations
    #: without either a tracker identity/shoulder-label failure or an unobserved
    #: interval. Hold and require a continuous return instead of treating the
    #: new value as a physically possible torso command. This is a perception
    #: continuity guard, not the robot's trajectory rate (the governor owns
    #: that separately).
    max_observation_step_rad: float = radians(20.0)
    #: PROVISIONAL, not measured. No operator-comfort trial has been run for
    #: torso slew, so this is deliberately NOT chosen for feel: it is the
    #: tightest rate any active authority already declares for this joint. The
    #: URDF gives ``leg_joint4`` 1.0 rad/s and the hardware supervisor caps every
    #: joint at 0.35 rad/s, so 0.35 is the binding one and is what is used. It
    #: must be re-derived from an operator trial before any release claim; see
    #: the open "measured operator tolerance" item.
    max_rate_rad_s: float = 0.35

    def __post_init__(self) -> None:
        numeric = (
            self.yaw_gain,
            self.deadband_rad,
            self.soft_margin_rad,
            self.max_camera_yaw_rad,
            self.max_relative_yaw_rad,
            self.max_neutral_camera_yaw_rad,
            self.max_neutral_spread_rad,
            self.max_observation_gap_ns,
            self.max_observation_step_rad,
            self.max_rate_rad_s,
        )
        if not all(isfinite(value) and value >= 0 for value in numeric):
            raise ValueError("torso policy values must be finite and non-negative")
        if self.max_observation_gap_ns <= 0:
            raise ValueError("torso observation gap must be positive")
        if self.yaw_limit.upper_rad - self.yaw_limit.lower_rad <= 2 * self.soft_margin_rad:
            raise ValueError("soft margin consumes the entire joint range")
        if self.max_neutral_camera_yaw_rad > self.max_camera_yaw_rad:
            raise ValueError("neutral camera yaw limit cannot exceed working camera yaw")
        if self.max_relative_yaw_rad > self.max_camera_yaw_rad:
            raise ValueError("relative yaw limit cannot exceed camera yaw limit")


@dataclass(frozen=True)
class TorsoTarget:
    """Commanded ``leg_joint4`` angle.

    Callers that also drive the arms MUST rotate their shoulder-relative arm
    offsets by this angle and solve arm IK against a base pose carrying it.
    """

    yaw_rad: float


@dataclass(frozen=True)
class TorsoMappingResult:
    target: TorsoTarget | None
    rejection: TorsoMapRejection | None = None
    #: Demand ran past the soft limit and was clamped, so the torso is pinned and
    #: no longer following.
    saturated: bool = False


class TorsoRetargeter:
    def __init__(self, policy: TorsoMappingPolicy, *, neutral: TorsoTarget) -> None:
        self._policy = policy
        self._neutral = neutral

    @property
    def policy(self) -> TorsoMappingPolicy:
        return self._policy

    @property
    def neutral(self) -> TorsoTarget:
        return self._neutral

    def map(
        self,
        intent: TorsoIntent,
    ) -> TorsoMappingResult:
        if not isfinite(intent.yaw_signal):
            return TorsoMappingResult(None, TorsoMapRejection.NONFINITE)

        demanded = TorsoTarget(
            yaw_rad=self._neutral.yaw_rad
            + self._deadband(intent.yaw_signal) * self._policy.yaw_gain
        )
        target, saturated = self._saturate(demanded)
        return TorsoMappingResult(target, saturated=saturated)

    def _deadband(self, value: float) -> float:
        magnitude = abs(value)
        if magnitude <= self._policy.deadband_rad:
            return 0.0
        # Subtract the threshold so the output is continuous at the deadband edge;
        # returning ``value`` directly steps and trips the acceleration limit.
        return (magnitude - self._policy.deadband_rad) * (1.0 if value > 0 else -1.0)

    def _saturate(self, target: TorsoTarget) -> tuple[TorsoTarget, bool]:
        lower, upper = self._soft_band(self._policy.yaw_limit)
        yaw = min(max(target.yaw_rad, lower), upper)
        if yaw == target.yaw_rad:
            return target, False
        return TorsoTarget(yaw_rad=yaw), True

    def _soft_band(self, limit: JointLimit) -> tuple[float, float]:
        """Inclusive soft band, strictly inside the hard limit by the margin."""
        return (
            limit.lower_rad + self._policy.soft_margin_rad,
            limit.upper_rad - self._policy.soft_margin_rad,
        )
