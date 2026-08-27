import pytest

from galbot_motion_studio.contracts.core import (
    ApprovedCommand,
    JointTarget,
    RobotTarget,
    SafetyDecision,
    SafetyOutcome,
)
from galbot_motion_studio.ports.command import NullCommandSink


def test_null_sink_records_without_hardware_side_effect() -> None:
    target = RobotTarget(
        session_id="test-session",
        sequence=7,
        source_clock_id="test-clock",
        source_mono_ns=1,
        arm_generation="arm-1",
        joints=(JointTarget(name="head_joint1", position_rad=0.1),),
        model_hash="a" * 64,
        tool_hash="simulated-tool",
        mapping_hash="mapping-v1",
        ik_residual_m=0.0,
        predicted_clearance_m=1.0,
    )
    decision = SafetyDecision(
        session_id="test-session",
        sequence=7,
        source_clock_id="test-clock",
        source_mono_ns=2,
        arm_generation="arm-1",
        outcome=SafetyOutcome.ALLOW,
        target_fingerprint=target.fingerprint,
    )
    sink = NullCommandSink()
    receipt = sink.submit(ApprovedCommand(target=target, decision=decision))
    assert receipt.accepted and receipt.sink == "null"
    assert sink.submissions == [ApprovedCommand(target=target, decision=decision)]


def test_approval_cannot_be_reused_for_a_changed_target() -> None:
    target = RobotTarget(
        session_id="test-session",
        sequence=7,
        source_clock_id="mac-clock",
        source_mono_ns=1,
        arm_generation="arm-1",
        joints=(JointTarget(name="head_joint1", position_rad=0.1),),
        model_hash="a" * 64,
        tool_hash="simulated-tool",
        mapping_hash="mapping-v1",
        ik_residual_m=0.0,
        predicted_clearance_m=1.0,
    )
    changed_target = target.model_copy(update={"joints": (JointTarget(name="head_joint1", position_rad=0.2),)})
    decision = SafetyDecision(
        session_id="test-session",
        sequence=7,
        source_clock_id="hpu-clock",
        source_mono_ns=2,
        arm_generation="arm-1",
        outcome=SafetyOutcome.ALLOW,
        target_fingerprint=target.fingerprint,
    )
    with pytest.raises(ValueError, match="exact target"):
        ApprovedCommand(target=changed_target, decision=decision)


def test_authority_ids_and_model_hash_are_strict() -> None:
    with pytest.raises(ValueError, match="session_id"):
        RobotTarget(
            session_id="",
            sequence=0,
            source_clock_id="clock",
            source_mono_ns=0,
            arm_generation="arm",
            joints=(JointTarget(name="head_joint1", position_rad=0.0),),
            model_hash="z" * 64,
            tool_hash="tool",
            mapping_hash="map",
            ik_residual_m=0.0,
            predicted_clearance_m=1.0,
        )
    with pytest.raises(ValueError, match="source_clock_id"):
        RobotTarget(
            session_id="session",
            sequence=0,
            source_clock_id="",
            source_mono_ns=0,
            arm_generation="arm",
            joints=(JointTarget(name="head_joint1", position_rad=0.0),),
            model_hash="a" * 64,
            tool_hash="tool",
            mapping_hash="map",
            ik_residual_m=0.0,
            predicted_clearance_m=1.0,
        )
    with pytest.raises(ValueError, match="model_hash"):
        RobotTarget(
            session_id="session",
            sequence=0,
            source_clock_id="clock",
            source_mono_ns=0,
            arm_generation="arm",
            joints=(JointTarget(name="head_joint1", position_rad=0.0),),
            model_hash="z" * 64,
            tool_hash="tool",
            mapping_hash="map",
            ik_residual_m=0.0,
            predicted_clearance_m=1.0,
        )
