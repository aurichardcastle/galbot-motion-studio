"""Core contracts. Named joints prevent accidental ordering-dependent commands."""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from json import dumps
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    """Strict immutable boundary object used by the simulator-only core."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class SafetyOutcome(StrEnum):
    ALLOW = "ALLOW"
    HOLD = "HOLD"
    FAULT = "FAULT"


class Envelope(Contract):
    schema_version: str = "1.0"
    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    source_clock_id: str = Field(min_length=1)
    source_mono_ns: int = Field(ge=0)


class JointTarget(Contract):
    name: str = Field(min_length=1)
    position_rad: float
    max_velocity_rad_s: float | None = Field(default=None, gt=0)
    max_acceleration_rad_s2: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def positions_are_finite(self) -> "JointTarget":
        values = (self.position_rad, self.max_velocity_rad_s, self.max_acceleration_rad_s2)
        if any(value is not None and not isfinite(value) for value in values):
            raise ValueError("joint target values must be finite")
        return self


class RobotTarget(Envelope):
    joints: tuple[JointTarget, ...] = Field(min_length=1)
    arm_generation: str = Field(min_length=1)
    model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_hash: str = Field(min_length=1)
    mapping_hash: str = Field(min_length=1)
    ik_residual_m: float = Field(ge=0)
    predicted_clearance_m: float
    #: World-frame wrist targets the declared ``ik_residual_m`` was measured
    #: against, per arm. Present so a second authority can RECOMPUTE the residual
    #: from the commanded joints instead of taking the retargeter's word for it --
    #: the same discipline `predicted_clearance_m` is already held to. ``None``
    #: for a limb that was held this frame and therefore has no fresh target.
    left_wrist_target_world_m: tuple[float, float, float] | None = None
    right_wrist_target_world_m: tuple[float, float, float] | None = None
    #: The arm joint angles the residual was measured on -- the IK SOLUTION, not
    #: the joints in ``joints``. Those two are not the same pose: the guarded
    #: controller and the trajectory governor both sit between the solve and the
    #: command, and the commanded TCP legitimately trails the solved one by
    #: 15.6 mm at p50 and up to 99 mm on a fast frame (measured, 2026-08-22).
    #: Verifying the declared residual against the commanded joints therefore
    #: measures tracking lag and not IK quality, which is why the solution is
    #: carried explicitly instead of being inferred.
    left_ik_joints_rad: tuple[tuple[str, float], ...] | None = None
    right_ik_joints_rad: tuple[tuple[str, float], ...] | None = None

    @model_validator(mode="after")
    def target_is_well_formed(self) -> "RobotTarget":
        if not isfinite(self.ik_residual_m) or not isfinite(self.predicted_clearance_m):
            raise ValueError("target metrics must be finite")
        for wrist in (self.left_wrist_target_world_m, self.right_wrist_target_world_m):
            if wrist is not None and not all(isfinite(value) for value in wrist):
                raise ValueError("wrist target coordinates must be finite")
        names = tuple(joint.name for joint in self.joints)
        if len(names) != len(set(names)):
            raise ValueError("joint target names must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        """Stable content binding used by a separately-clocked safety decision."""
        canonical = dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()


class SafetyDecision(Envelope):
    outcome: SafetyOutcome
    arm_generation: str = Field(min_length=1)
    target_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def failures_have_reasons(self) -> "SafetyDecision":
        if self.outcome is not SafetyOutcome.ALLOW and not self.reasons:
            raise ValueError("HOLD and FAULT decisions require at least one reason")
        if self.outcome is SafetyOutcome.ALLOW and self.target_fingerprint is None:
            raise ValueError("ALLOW decisions require the exact target fingerprint")
        return self


class ApprovedCommand(Contract):
    """Capability object created only after a matching ALLOW decision."""

    target: RobotTarget
    decision: SafetyDecision

    @model_validator(mode="after")
    def decision_matches_target(self) -> "ApprovedCommand":
        if self.decision.outcome is not SafetyOutcome.ALLOW:
            raise ValueError("only ALLOW decisions may produce an ApprovedCommand")
        if self.decision.session_id != self.target.session_id:
            raise ValueError("decision and target session IDs differ")
        if self.decision.sequence != self.target.sequence:
            raise ValueError("decision and target sequences differ")
        if self.decision.arm_generation != self.target.arm_generation:
            raise ValueError("decision and target arm generations differ")
        if self.decision.target_fingerprint != self.target.fingerprint:
            raise ValueError("decision is not bound to this exact target")
        return self
