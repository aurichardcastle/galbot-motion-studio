import mujoco
import pytest

from galbot_motion_studio.adapters.mujoco_preview import MujocoPreviewSink
from galbot_motion_studio.contracts.core import ApprovedCommand, JointTarget, RobotTarget, SafetyDecision, SafetyOutcome


def test_preview_applies_named_approved_target_only_to_digital_twin() -> None:
    target = RobotTarget(
        session_id="preview", sequence=1, source_clock_id="clock", source_mono_ns=1, arm_generation="preview-1",
        joints=(JointTarget(name="head_joint1", position_rad=0.2),), model_hash="a" * 64,
        tool_hash="sim", mapping_hash="map", ik_residual_m=0.0, predicted_clearance_m=1.0,
    )
    decision = SafetyDecision(
        session_id="preview", sequence=1, source_clock_id="clock", source_mono_ns=2, arm_generation="preview-1",
        outcome=SafetyOutcome.ALLOW, target_fingerprint=target.fingerprint,
    )
    sink = MujocoPreviewSink()
    receipt = sink.submit(ApprovedCommand(target=target, decision=decision))
    joint_id = mujoco.mj_name2id(sink.model, mujoco.mjtObj.mjOBJ_JOINT, "head_joint1")
    joint = sink.model.jnt_qposadr[joint_id]
    assert receipt.sink == "mujoco-preview"
    assert sink.data.qpos[joint] == 0.2


def test_preview_rejects_any_object_without_allow_capability() -> None:
    sink = MujocoPreviewSink()
    before = sink.data.qpos.copy()
    with pytest.raises(TypeError, match="ApprovedCommand"):
        sink.submit(object())
    assert (sink.data.qpos == before).all()
