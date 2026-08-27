"""Fail-closed supervisor for the offline MuJoCo preview path.

This module deliberately exposes no transition to physical ``LIVE`` control.  It
creates capability-bound :class:`ApprovedCommand` objects for the simulator only,
after independently checking provenance, timing, joint limits, dynamics, and the
complete interpolated clearance path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite, sqrt
from secrets import token_hex
from typing import Mapping

from galbot_motion_studio.contracts.core import (
    ApprovedCommand,
    RobotTarget,
    SafetyDecision,
    SafetyOutcome,
)
from galbot_motion_studio.model.joint_map import JointLimit, parse_revolute_joint_limits
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST
from galbot_motion_studio.safety.clearance import ClearanceChecker, ClearanceReport


class SupervisorState(StrEnum):
    QUERY_ONLY = "QUERY_ONLY"
    PREVIEW = "PREVIEW"
    HOLD = "HOLD"
    FAULT = "FAULT"


@dataclass(frozen=True)
class SupervisorPolicy:
    model_hash: str = CANONICAL_MANIFEST.fixed_mjcf_sha256
    tool_hash: str = "canonical-generic-grippers-sim-only"
    soft_margin_rad: float = 0.09
    max_ik_residual_m: float = 0.03
    max_joint_rate_rad_s: float = 0.35
    max_joint_acceleration_rad_s2: float = 1.0
    max_joint_step_rad: float = 0.05
    clearance_match_tolerance_m: float = 1e-6
    #: How far the recomputed wrist residual may exceed the declared one before
    #: the declaration counts as understating it. Larger than the clearance
    #: tolerance on purpose: the supervisor rebuilds the pose from the target's
    #: joint angles, so its forward kinematics differ from the retargeter's by
    #: float rounding through a seven-joint chain rather than by an exact replay.
    ik_residual_match_tolerance_m: float = 1e-4

    def __post_init__(self) -> None:
        values = (
            self.soft_margin_rad,
            self.max_ik_residual_m,
            self.max_joint_rate_rad_s,
            self.max_joint_acceleration_rad_s2,
            self.max_joint_step_rad,
            self.clearance_match_tolerance_m,
        )
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError("supervisor policy values must be finite and non-negative")


@dataclass(frozen=True)
class SupervisionResult:
    decision: SafetyDecision
    approved: ApprovedCommand | None
    clearance: ClearanceReport | None
    state_before: SupervisorState
    state_after: SupervisorState


class SafetySupervisor:
    """Stateful, simulator-only command authority.

    ``HOLD`` and ``FAULT`` are latched.  Continuing after either requires an
    explicit call to :meth:`resume_preview` or :meth:`reset_fault`, which also
    rotates the arm generation so queued targets cannot be replayed.
    """

    def __init__(
        self,
        checker: ClearanceChecker,
        *,
        initial_pose: Mapping[str, float],
        source_clock_id: str,
        policy: SupervisorPolicy | None = None,
    ) -> None:
        if not source_clock_id:
            raise ValueError("source_clock_id is required")
        self.checker = checker
        self.policy = policy or SupervisorPolicy()
        self.source_clock_id = source_clock_id
        self.state = SupervisorState.QUERY_ONLY
        self.arm_generation = "query-only"
        self.current_pose = checker.full_pose(initial_pose)
        self._last_sequence: int | None = None
        self._last_source_mono_ns: int | None = None
        self._last_velocity: dict[str, float] = {}
        self._pending_clearance: tuple[tuple[tuple[str, float], ...], ClearanceReport] | None = None
        self._limits: dict[str, JointLimit] = parse_revolute_joint_limits(
            CANONICAL_MANIFEST.fixed_urdf
        )

    def start_preview(self, *, source_mono_ns: int | None = None) -> str:
        if self.state is SupervisorState.FAULT:
            raise RuntimeError("fault must be explicitly reset before preview")
        self._rotate_generation(source_mono_ns=source_mono_ns)
        self.state = SupervisorState.PREVIEW
        return self.arm_generation

    def resume_preview(self, *, source_mono_ns: int | None = None) -> str:
        if self.state is not SupervisorState.HOLD:
            raise RuntimeError("resume_preview is only valid from HOLD")
        self._rotate_generation(source_mono_ns=source_mono_ns)
        self.state = SupervisorState.PREVIEW
        return self.arm_generation

    def reset_fault(self) -> None:
        if self.state is not SupervisorState.FAULT:
            raise RuntimeError("reset_fault is only valid from FAULT")
        self.state = SupervisorState.QUERY_ONLY
        self.arm_generation = "query-only"
        self._reset_ordering()

    def predict_clearance(self, joints: Mapping[str, float]) -> ClearanceReport:
        """Compute the same swept check that :meth:`evaluate` will repeat."""
        end = dict(self.current_pose)
        end.update(joints)
        report = self.checker.check_path(
            self.current_pose,
            end,
            steps=2,
            max_joint_step=0.01,
        )
        self._pending_clearance = (self._pose_key(end), report)
        return report

    def hold(
        self,
        *,
        session_id: str,
        sequence: int,
        source_mono_ns: int,
        now_mono_ns: int,
        reason: str,
    ) -> SafetyDecision:
        """Latch a perception/retargeting HOLD before a RobotTarget exists."""
        if not reason:
            raise ValueError("a HOLD reason is required")
        before = self.state
        # `source_mono_ns` is the SOURCE frame's capture time and both paths below
        # must use it, not `now_mono_ns`.
        #
        # This parameter was accepted and then silently ignored: every HOLD was
        # stamped with the PROCESSING clock, which always leads the frame being
        # processed. A HOLD therefore landed ahead of every later frame's capture
        # time, and `MotionClip`'s strict-increase validator rejected the clip at
        # save time. Measured on a live 1196-frame session with holds in it: the
        # whole recording was lost after the operator had already quit cleanly.
        # An ALLOW has always used the capture time, so HOLD was also the odd one
        # out. Analysis-sync never tripped it, which is why the suite stayed green.
        if before is SupervisorState.FAULT:
            return SafetyDecision(
                session_id=session_id,
                sequence=sequence,
                source_clock_id=self.source_clock_id,
                source_mono_ns=source_mono_ns,
                arm_generation=self.arm_generation,
                outcome=SafetyOutcome.FAULT,
                reasons=(f"FAULT remains latched; additional rejection: {reason}",),
            )
        self.state = SupervisorState.HOLD
        return SafetyDecision(
            session_id=session_id,
            sequence=sequence,
            source_clock_id=self.source_clock_id,
            source_mono_ns=source_mono_ns,
            arm_generation=self.arm_generation,
            outcome=SafetyOutcome.HOLD,
            reasons=(f"{before}: {reason}",),
        )

    def fault(
        self,
        *,
        session_id: str,
        sequence: int,
        now_mono_ns: int,
        reason: str,
        source_mono_ns: int | None = None,
    ) -> SafetyDecision:
        """Latch an unexpected internal failure as FAULT; never infer permission.

        ``source_mono_ns`` is the SOURCE frame's capture time. It defaults to
        ``now_mono_ns`` only for callers that have no observation to hand.

        Defaulting it was a real defect: ``now_mono_ns`` is the processing clock
        and always runs slightly ahead of the frame being processed, so a FAULT
        landed in the future and the next correctly stamped frame was behind it.
        On a live 1196-frame session that made `MotionClip`'s strict-increase
        validator reject the whole recording at save time, losing all of it.
        """
        if not reason:
            raise ValueError("a FAULT reason is required")
        self.state = SupervisorState.FAULT
        return SafetyDecision(
            session_id=session_id,
            sequence=sequence,
            source_clock_id=self.source_clock_id,
            source_mono_ns=now_mono_ns if source_mono_ns is None else source_mono_ns,
            arm_generation=self.arm_generation,
            outcome=SafetyOutcome.FAULT,
            reasons=(reason,),
        )

    def evaluate(self, target: RobotTarget, *, now_mono_ns: int) -> SupervisionResult:
        before = self.state
        try:
            return self._evaluate_checked(target, now_mono_ns=now_mono_ns)
        except BaseException as error:
            self._pending_clearance = None
            self.state = SupervisorState.FAULT
            decision = self._decision(
                target,
                now_mono_ns,
                SafetyOutcome.FAULT,
                (f"unexpected supervisor failure: {type(error).__name__}: {error}",),
            )
            return SupervisionResult(decision, None, None, before, self.state)

    def _evaluate_checked(
        self, target: RobotTarget, *, now_mono_ns: int
    ) -> SupervisionResult:
        before = self.state
        reasons: list[str] = []
        fault = False
        clearance: ClearanceReport | None = None

        if self.state is not SupervisorState.PREVIEW:
            reasons.append(f"supervisor is {self.state}")
            fault = self.state is SupervisorState.FAULT
        if target.source_clock_id != self.source_clock_id:
            reasons.append("source clock mismatch")
            fault = True
        if target.arm_generation != self.arm_generation:
            reasons.append("arm generation mismatch")
            fault = True
        if target.model_hash != self.policy.model_hash:
            reasons.append("model hash mismatch")
            fault = True
        if target.tool_hash != self.policy.tool_hash:
            reasons.append("tool hash mismatch")
            fault = True
        if self._last_sequence is not None and target.sequence <= self._last_sequence:
            reasons.append("reordered or duplicate target")
            fault = True
        if self._last_source_mono_ns is not None and target.source_mono_ns <= self._last_source_mono_ns:
            reasons.append("non-monotonic source timestamp")
            fault = True
        if target.source_mono_ns > now_mono_ns:
            reasons.append("target timestamp is in the future")
            fault = True
        if target.ik_residual_m > self.policy.max_ik_residual_m:
            reasons.append("IK residual exceeds policy")
        reasons.extend(self._ik_residual_reasons(target))

        names = {joint.name for joint in target.joints}
        unknown = sorted(names - self._limits.keys() | names - self.current_pose.keys())
        if unknown:
            reasons.append(f"unknown or non-revolute joints: {unknown}")
            fault = True

        if not unknown:
            reasons.extend(self._joint_limit_reasons(target))
            reasons.extend(self._dynamic_reasons(target))

        if reasons:
            self._pending_clearance = None
            self.state = SupervisorState.FAULT if fault else SupervisorState.HOLD
            outcome = SafetyOutcome.FAULT if fault else SafetyOutcome.HOLD
            decision = self._decision(target, now_mono_ns, outcome, tuple(reasons))
            return SupervisionResult(decision, None, clearance, before, self.state)

        end = dict(self.current_pose)
        end.update({joint.name: joint.position_rad for joint in target.joints})
        try:
            key = self._pose_key(end)
            if self._pending_clearance is not None and self._pending_clearance[0] == key:
                clearance = self._pending_clearance[1]
            else:
                clearance = self.checker.check_path(
                    self.current_pose,
                    end,
                    steps=2,
                    max_joint_step=0.01,
                )
            if not isfinite(clearance.min_distance):
                raise ValueError("clearance report contains a non-finite distance")
            if not clearance.ok:
                reasons.append("interpolated clearance path rejected")
            if (
                abs(clearance.min_distance - target.predicted_clearance_m)
                > self.policy.clearance_match_tolerance_m
            ):
                reasons.append("declared clearance does not match independent check")
        except BaseException as error:
            reasons.append(f"clearance unevaluable: {type(error).__name__}: {error}")
            fault = True

        if reasons:
            self._pending_clearance = None
            self.state = SupervisorState.FAULT if fault else SupervisorState.HOLD
            outcome = SafetyOutcome.FAULT if fault else SafetyOutcome.HOLD
            decision = self._decision(target, now_mono_ns, outcome, tuple(reasons))
            return SupervisionResult(decision, None, clearance, before, self.state)

        decision = self._decision(target, now_mono_ns, SafetyOutcome.ALLOW, ())
        approved = ApprovedCommand(target=target, decision=decision)
        elapsed_s = self._elapsed_s(target)
        if elapsed_s is not None and elapsed_s > 0:
            self._last_velocity = {
                joint.name: (joint.position_rad - self.current_pose[joint.name]) / elapsed_s
                for joint in target.joints
            }
        else:
            self._last_velocity = {joint.name: 0.0 for joint in target.joints}
        self.current_pose.update({joint.name: joint.position_rad for joint in target.joints})
        self._pending_clearance = None
        self._last_sequence = target.sequence
        self._last_source_mono_ns = target.source_mono_ns
        return SupervisionResult(decision, approved, clearance, before, self.state)

    def _ik_residual_reasons(self, target: RobotTarget) -> list[str]:
        """Recompute the wrist residual instead of believing the declared one.

        Until this existed the residual was the one safety-relevant number in the
        target that NOTHING checked: clearance was independently recomputed and
        cross-checked against the declaration, joint limits and dynamics were
        recomputed, and the residual alone was taken on trust from the same
        retargeter that produced the pose. A solver that converged badly and then
        reported a small residual would have been approved -- one authority, not
        two, on the number that says whether the robot is where the operator put
        it.

        Only an UNDERSTATED residual is a reason. A target that declares MORE
        error than it achieved is conservative, is already checked against
        ``max_ik_residual_m`` above, and is exactly what the frame-to-frame
        carry-over of a held limb's last residual produces.
        """
        claims = (
            ("left", target.left_wrist_target_world_m, target.left_ik_joints_rad),
            ("right", target.right_wrist_target_world_m, target.right_ik_joints_rad),
        )
        reasons: list[str] = []
        for side, wrist, solution in claims:
            if wrist is None and solution is None:
                continue
            if wrist is None or solution is None:
                reasons.append(f"{side} IK residual claim is incomplete")
                continue
            pose = dict(self.current_pose)
            pose.update({name: float(value) for name, value in solution})
            try:
                achieved = self.checker.tcp_positions(pose)
            except Exception as error:  # a broken FK must fault closed, not pass
                reasons.append(
                    f"could not verify {side} IK residual: {type(error).__name__}: {error}"
                )
                continue
            tcp = achieved.get(side)
            if tcp is None or not all(isfinite(value) for value in tcp):
                reasons.append(f"{side} TCP could not be verified")
                continue
            residual = sqrt(sum((a - b) ** 2 for a, b in zip(tcp, wrist)))
            if not isfinite(residual):
                reasons.append(f"{side} IK residual is not finite")
            elif residual > self.policy.max_ik_residual_m:
                reasons.append(f"{side} recomputed IK residual exceeds policy")
            elif residual > target.ik_residual_m + self.policy.ik_residual_match_tolerance_m:
                reasons.append(f"{side} declared IK residual understates the achieved one")
        return reasons

    def _joint_limit_reasons(self, target: RobotTarget) -> list[str]:
        reasons = []
        margin = self.policy.soft_margin_rad
        for joint in target.joints:
            limit = self._limits[joint.name]
            if not (
                limit.lower_rad + margin
                <= joint.position_rad
                <= limit.upper_rad - margin
            ):
                reasons.append(f"{joint.name} breaches its soft joint limit")
        return reasons

    def _dynamic_reasons(self, target: RobotTarget) -> list[str]:
        elapsed_s = self._elapsed_s(target)
        reasons = []
        for joint in target.joints:
            step = abs(joint.position_rad - self.current_pose[joint.name])
            if step > self.policy.max_joint_step_rad + 1e-12:
                reasons.append(f"{joint.name} step exceeds policy")
        if elapsed_s is None:
            return reasons
        if elapsed_s <= 0:
            return reasons + ["non-positive target time step"]
        for joint in target.joints:
            velocity = (joint.position_rad - self.current_pose[joint.name]) / elapsed_s
            rate_limit = min(
                self.policy.max_joint_rate_rad_s,
                joint.max_velocity_rad_s or self.policy.max_joint_rate_rad_s,
            )
            if abs(velocity) > rate_limit + 1e-12:
                reasons.append(f"{joint.name} rate exceeds policy")
            if joint.name in self._last_velocity:
                acceleration = abs(velocity - self._last_velocity[joint.name]) / elapsed_s
                acceleration_limit = min(
                    self.policy.max_joint_acceleration_rad_s2,
                    joint.max_acceleration_rad_s2
                    or self.policy.max_joint_acceleration_rad_s2,
                )
                if acceleration > acceleration_limit + 1e-12:
                    reasons.append(f"{joint.name} acceleration exceeds policy")
        return reasons

    def _elapsed_s(self, target: RobotTarget) -> float | None:
        if self._last_source_mono_ns is None:
            return None
        return (target.source_mono_ns - self._last_source_mono_ns) / 1_000_000_000

    def _decision(
        self,
        target: RobotTarget,
        now_mono_ns: int,
        outcome: SafetyOutcome,
        reasons: tuple[str, ...],
    ) -> SafetyDecision:
        return SafetyDecision(
            session_id=target.session_id,
            sequence=target.sequence,
            source_clock_id=self.source_clock_id,
            source_mono_ns=now_mono_ns,
            arm_generation=target.arm_generation,
            outcome=outcome,
            target_fingerprint=target.fingerprint if outcome is SafetyOutcome.ALLOW else None,
            reasons=reasons,
        )

    def _rotate_generation(self, *, source_mono_ns: int | None = None) -> None:
        self.arm_generation = token_hex(16)
        self._reset_ordering(source_mono_ns=source_mono_ns)

    def _reset_ordering(self, *, source_mono_ns: int | None = None) -> None:
        self._last_sequence = None
        self._last_source_mono_ns = source_mono_ns
        self._last_velocity = (
            {name: 0.0 for name in self.current_pose}
            if source_mono_ns is not None
            else {}
        )
        self._pending_clearance = None

    @staticmethod
    def _pose_key(pose: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        return tuple(sorted((name, float(value)) for name, value in pose.items()))
