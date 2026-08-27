"""Deterministic MotionClip replay through fresh safety and clearance checks."""

from __future__ import annotations

from dataclasses import dataclass

from galbot_motion_studio.contracts.core import SafetyDecision, SafetyOutcome
from galbot_motion_studio.ports.command import CommandReceipt, CommandSink
from galbot_motion_studio.recording import MotionClip
from galbot_motion_studio.safety.profiles import clearance_floor_for, policy_for
from galbot_motion_studio.safety.supervisor import SafetySupervisor, SupervisorState


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayResult:
    receipts: tuple[CommandReceipt, ...]
    held_frames: int
    joint_targets: tuple[tuple[tuple[str, float], ...], ...]
    #: Preserved separately when the final source event regressed in time and
    #: therefore cannot join the monotonic command stream.
    terminal_fault: SafetyDecision | None = None


def replay_clip(
    clip: MotionClip,
    *,
    supervisor: SafetySupervisor,
    sink: CommandSink,
) -> ReplayResult:
    if clip.model_hash != supervisor.policy.model_hash:
        raise ReplayError("clip model hash does not match the active model")
    if clip.tool_hash != supervisor.policy.tool_hash:
        raise ReplayError("clip tool hash does not match the active simulated tool")
    expected_policy = policy_for(clip.motion_profile)
    if (
        supervisor.policy.max_joint_rate_rad_s
        != expected_policy.max_joint_rate_rad_s
        or supervisor.policy.max_joint_acceleration_rad_s2
        != expected_policy.max_joint_acceleration_rad_s2
        or supervisor.policy.max_joint_step_rad != expected_policy.max_joint_step_rad
    ):
        raise ReplayError("clip motion profile does not match the replay supervisor")
    # The floor lives on the checker, not the policy, so it needs its own check.
    # A clip records its profile and nothing else about geometry; the floor is
    # re-derived from that profile exactly as the dynamics above are. Without
    # this, a supervisor built from a default-floor checker would replay a SIM
    # clip under the 10 mm hardware floor and silently hold frames the recording
    # approved -- which is not a replay of that clip.
    expected_floor = clearance_floor_for(clip.motion_profile)
    if supervisor.checker.floor != expected_floor:
        raise ReplayError(
            f"clip motion profile '{clip.motion_profile}' requires a "
            f"{expected_floor * 1000:g} mm self-clearance floor; the replay "
            f"supervisor's checker uses {supervisor.checker.floor * 1000:g} mm"
        )
    approved_frames = [
        frame
        for frame in clip.frames
        if frame.target is not None and frame.decision.outcome is SafetyOutcome.ALLOW
    ]
    if not approved_frames:
        _replay_terminal_fault(clip, supervisor)
        return ReplayResult((), len(clip.frames), (), clip.terminal_fault)
    first_source_ns = approved_frames[0].target.source_mono_ns
    initial_source_ns = (
        clip.initial_source_mono_ns
        if clip.initial_source_mono_ns is not None
        else max(0, first_source_ns - clip.control_period_ns)
    )
    generation = supervisor.start_preview(
        source_mono_ns=initial_source_ns
    )
    receipts: list[CommandReceipt] = []
    joint_targets: list[tuple[tuple[str, float], ...]] = []
    held = 0
    for frame in clip.frames:
        if frame.target is None or frame.decision.outcome is not SafetyOutcome.ALLOW:
            held += 1
            # Live HOLD rotates the generation and zeroes reconstructed velocity
            # before the next valid target. Replaying by merely skipping held
            # rows incorrectly bridges dynamics across the dropout and can reject
            # the exact acceleration-governed target that live execution allowed.
            if supervisor.state is SupervisorState.PREVIEW:
                supervisor.hold(
                    session_id=frame.decision.session_id,
                    sequence=frame.source_sequence,
                    source_mono_ns=frame.source_mono_ns,
                    now_mono_ns=frame.source_mono_ns,
                    reason="recorded non-command frame",
                )
            continue
        if supervisor.state is SupervisorState.HOLD:
            generation = supervisor.resume_preview(
                source_mono_ns=max(
                    0, frame.target.source_mono_ns - clip.control_period_ns
                )
            )
        joints = {joint.name: joint.position_rad for joint in frame.target.joints}
        clearance = supervisor.predict_clearance(joints)
        target = frame.target.model_copy(
            update={
                "arm_generation": generation,
                "predicted_clearance_m": clearance.min_distance,
            }
        )
        result = supervisor.evaluate(target, now_mono_ns=target.source_mono_ns)
        if result.approved is None:
            raise ReplayError(
                f"frame {frame.source_sequence} failed fresh safety checks: "
                f"{result.decision.reasons}"
            )
        receipts.append(sink.submit(result.approved))
        joint_targets.append(
            tuple((joint.name, joint.position_rad) for joint in result.approved.target.joints)
        )
    _replay_terminal_fault(clip, supervisor)
    return ReplayResult(tuple(receipts), held, tuple(joint_targets), clip.terminal_fault)


def _replay_terminal_fault(clip: MotionClip, supervisor: SafetySupervisor) -> None:
    """Reconstruct a terminal ingress FAULT without forging its capture time."""
    fault = clip.terminal_fault
    if fault is None:
        return
    # MotionClip validates the outcome, but keep replay fail-closed if an older
    # or manually constructed object reaches this boundary without that check.
    if fault.outcome is not SafetyOutcome.FAULT:
        raise ReplayError("terminal fault does not have FAULT outcome")
    supervisor.fault(
        session_id=fault.session_id,
        sequence=fault.sequence,
        source_mono_ns=fault.source_mono_ns,
        now_mono_ns=fault.source_mono_ns,
        reason=fault.reasons[0],
    )
