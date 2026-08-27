"""Head yaw/pitch mapping that saturates at the soft limits instead of rejecting.

Out-of-range demand is clamped into the soft-limit band, per axis, rather than
refused. Clamping is safe by construction here: the soft band is strictly inside
the URDF hard limits, so a saturated target is a legal joint position with the
full soft margin still in hand. Rejecting it instead was worse than unsafe, it
was useless -- a rejected frame freezes the robot head, so looking far to one
side stopped tracking exactly when the operator was asking for the most motion.

Only two things are still rejected outright, because clipping cannot make either
of them safe: non-finite input (no defined value to clamp toward) and a demanded
step that exceeds the rate limit (a teleport, checked against the clamped
target). Everything else produces a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from galbot_motion_studio.model.joint_map import JointLimit


class HeadMapRejection(StrEnum):
    NONFINITE = "NONFINITE"
    # Retained, never produced. The mapper saturates at the soft limits now, but
    # recordings made before that change carry "head map: OUTSIDE_SOFT_LIMIT" as
    # a HOLD reason (see artifacts/live-webcam-uat/motion_clip.json), and this
    # member is what those strings decode to. A frozen head is no longer a
    # rejection; if this value ever appears in a fresh recording, something is
    # constructing the result by hand.
    OUTSIDE_SOFT_LIMIT = "OUTSIDE_SOFT_LIMIT"
    RATE_EXCEEDED = "RATE_EXCEEDED"


@dataclass(frozen=True)
class HeadPoseIntent:
    """Neutral-relative face-orientation signals, not calibrated physical angles."""

    yaw_signal: float
    pitch_signal: float
    #: The landmark basis that produced the signals. Face mesh and pose eyes
    #: have independently calibrated neutral spans, so a source transition is a
    #: perception event rather than an interchangeable sample.
    basis: str = "unknown"


@dataclass(frozen=True)
class HeadMappingPolicy:
    yaw_limit: JointLimit
    pitch_limit: JointLimit
    # Sized to the measured usable travel from HOME on the canonical fixed-base
    # model, not chosen for feel. head_joint1 soft band is -1.4308..+1.4308 rad
    # about HOME -0.0006, i.e. -81.9..+82.0 deg; head_joint2 soft band is
    # -0.1243..+0.4036 rad about HOME -0.0002, i.e. -7.1..+23.1 deg.
    # Yaw therefore has ~82 deg either way and 1.4 stays expressive without
    # sitting on the limit. Pitch is strongly asymmetric: the positive-delta side
    # driven by pitch_up_gain has 23.1 deg of travel, the negative side driven by
    # pitch_down_gain only 7.1 deg, so one shared pitch gain would either waste
    # the long side or pin the short one. Saturation makes an over-large gain
    # cost expressiveness rather than tracking, but these are sized to stay off
    # the limits during normal head motion.
    yaw_gain: float = 1.4
    pitch_up_gain: float = 1.2
    pitch_down_gain: float = 0.5
    deadband_rad: float = 0.025
    soft_margin_rad: float = 0.09
    max_rate_rad_s: float = 0.25

    def __post_init__(self) -> None:
        numeric = (
            self.yaw_gain,
            self.pitch_up_gain,
            self.pitch_down_gain,
            self.deadband_rad,
            self.soft_margin_rad,
            self.max_rate_rad_s,
        )
        if not all(isfinite(value) and value >= 0 for value in numeric):
            raise ValueError("head policy values must be finite and non-negative")
        for limit in (self.yaw_limit, self.pitch_limit):
            if limit.upper_rad - limit.lower_rad <= 2 * self.soft_margin_rad:
                raise ValueError("soft margin consumes the entire joint range")


@dataclass(frozen=True)
class HeadTarget:
    yaw_rad: float
    pitch_rad: float


@dataclass(frozen=True)
class HeadMappingResult:
    target: HeadTarget | None
    rejection: HeadMapRejection | None = None
    # True when demand ran past a soft limit on either axis and was clamped, so
    # the head is pinned and no longer following. Trailing and defaulted: every
    # existing caller reads ``target``/``rejection`` positionally or by name.
    saturated: bool = False


class HeadRetargeter:
    def __init__(self, policy: HeadMappingPolicy, *, neutral: HeadTarget) -> None:
        self._policy = policy
        self._neutral = neutral

    @property
    def policy(self) -> HeadMappingPolicy:
        """The live policy, exposed for independent pipeline-envelope tests."""
        return self._policy

    def map(
        self,
        intent: HeadPoseIntent,
        *,
        previous: HeadTarget | None,
        elapsed_s: float | None,
    ) -> HeadMappingResult:
        values = (intent.yaw_signal, intent.pitch_signal, elapsed_s)
        if any(value is not None and not isfinite(value) for value in values):
            return HeadMappingResult(None, HeadMapRejection.NONFINITE)
        if elapsed_s is not None and elapsed_s <= 0:
            return HeadMappingResult(None, HeadMapRejection.NONFINITE)

        yaw_delta = self._deadband(intent.yaw_signal) * self._policy.yaw_gain
        pitch_gain = self._policy.pitch_up_gain if intent.pitch_signal >= 0 else self._policy.pitch_down_gain
        pitch_delta = self._deadband(intent.pitch_signal) * pitch_gain
        demanded = HeadTarget(
            yaw_rad=self._neutral.yaw_rad + yaw_delta,
            pitch_rad=self._neutral.pitch_rad + pitch_delta,
        )
        # Saturate before the rate check, not after: the clamped target is what
        # would be commanded, so it is what the rate limit has to be true of.
        # Checking the raw demand instead would turn "already at the limit and
        # staying there" into a teleport rejection, which is the freeze again.
        target, saturated = self._saturate(demanded)
        if previous is not None and elapsed_s is not None:
            allowed_step = self._policy.max_rate_rad_s * elapsed_s
            if max(abs(target.yaw_rad - previous.yaw_rad), abs(target.pitch_rad - previous.pitch_rad)) > allowed_step:
                return HeadMappingResult(None, HeadMapRejection.RATE_EXCEEDED)
        return HeadMappingResult(target, saturated=saturated)

    def _deadband(self, value: float) -> float:
        magnitude = abs(value)
        if magnitude <= self._policy.deadband_rad:
            return 0.0
        # Subtract the threshold so the output is continuous at the deadband edge.
        # Returning ``value`` directly creates a step that trips acceleration limits.
        return (magnitude - self._policy.deadband_rad) * (1.0 if value > 0 else -1.0)

    def _saturate(self, target: HeadTarget) -> tuple[HeadTarget, bool]:
        """Clamp each axis into its own soft band, independently.

        Per axis on purpose: yaw has ~82 deg of headroom and pitch has 7, so a
        joint clamp on the short axis must not cost travel on the long one.
        """
        yaw = _clamp(target.yaw_rad, *self._soft_band(self._policy.yaw_limit))
        pitch = _clamp(target.pitch_rad, *self._soft_band(self._policy.pitch_limit))
        if yaw == target.yaw_rad and pitch == target.pitch_rad:
            return target, False
        return HeadTarget(yaw_rad=yaw, pitch_rad=pitch), True

    def _soft_band(self, limit: JointLimit) -> tuple[float, float]:
        """Inclusive soft band, strictly inside the hard limit by the margin.

        ``HeadMappingPolicy.__post_init__`` rejects a margin that consumes the
        range, so this band is always non-empty and lower < upper holds.
        """
        return (
            limit.lower_rad + self._policy.soft_margin_rad,
            limit.upper_rad - self._policy.soft_margin_rad,
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
