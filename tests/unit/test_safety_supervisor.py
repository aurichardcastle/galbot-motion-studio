from dataclasses import replace

from galbot_motion_studio.contracts.core import JointTarget, RobotTarget, SafetyOutcome
from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST
from galbot_motion_studio.safety.clearance import ClearanceChecker, HOME_QPOS
from galbot_motion_studio.safety.supervisor import (
    SafetySupervisor,
    SupervisorPolicy,
    SupervisorState,
)


def make_supervisor() -> SafetySupervisor:
    model = load_verified_fixed_base_model()
    checker = ClearanceChecker(model=model, home=HOME_QPOS)
    return SafetySupervisor(
        checker,
        initial_pose=HOME_QPOS,
        source_clock_id="camera-clock",
    )


def make_target(
    supervisor: SafetySupervisor,
    *,
    sequence: int = 1,
    timestamp: int = 1_000_000_000,
    joint_name: str = "head_joint1",
    position: float = 0.02,
    model_hash: str = CANONICAL_MANIFEST.fixed_mjcf_sha256,
    arm_generation: str | None = None,
) -> RobotTarget:
    report = supervisor.predict_clearance({joint_name: position})
    return RobotTarget(
        session_id="preview",
        sequence=sequence,
        source_clock_id="camera-clock",
        source_mono_ns=timestamp,
        arm_generation=arm_generation or supervisor.arm_generation,
        joints=(JointTarget(name=joint_name, position_rad=position),),
        model_hash=model_hash,
        tool_hash=SupervisorPolicy().tool_hash,
        mapping_hash="mapping-v1",
        ik_residual_m=0.0,
        predicted_clearance_m=report.min_distance,
    )


def test_supervisor_approves_only_after_independent_swept_check() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    result = supervisor.evaluate(make_target(supervisor), now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert result.approved is not None
    assert result.clearance is not None and result.clearance.ok
    assert supervisor.current_pose["head_joint1"] == 0.02


def test_hash_mismatch_faults_and_rejects_old_generation_after_reset() -> None:
    supervisor = make_supervisor()
    old_generation = supervisor.start_preview()
    bad = make_target(supervisor, model_hash="0" * 64)
    result = supervisor.evaluate(bad, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert result.approved is None
    assert supervisor.state is SupervisorState.FAULT

    supervisor.reset_fault()
    supervisor.start_preview()
    replay = make_target(supervisor, arm_generation=old_generation)
    result = supervisor.evaluate(replay, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.FAULT


def test_limit_and_rate_rejections_latch_hold_before_command_creation() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    too_close_to_limit = make_target(supervisor, position=1.50)
    result = supervisor.evaluate(too_close_to_limit, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert result.approved is None
    assert supervisor.state is SupervisorState.HOLD

    supervisor.resume_preview()
    first = make_target(supervisor, position=0.0)
    assert supervisor.evaluate(first, now_mono_ns=1_000_000_001).approved is not None
    fast = make_target(
        supervisor,
        sequence=2,
        timestamp=1_010_000_000,
        position=0.2,
    )
    result = supervisor.evaluate(fast, now_mono_ns=1_010_000_001)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert "rate exceeds" in " ".join(result.decision.reasons)


def test_declared_clearance_cannot_be_substituted() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    target = make_target(supervisor).model_copy(update={"predicted_clearance_m": 1.0})
    result = supervisor.evaluate(target, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert result.approved is None
    assert "does not match" in " ".join(result.decision.reasons)


def test_clearance_exception_faults_without_an_approved_command() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    target = make_target(supervisor)
    supervisor._pending_clearance = None

    def raising_clearance(*_args, **_kwargs):
        raise RuntimeError("injected geometry failure")

    supervisor.checker.check_path = raising_clearance
    result = supervisor.evaluate(target, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert result.approved is None
    assert "clearance unevaluable" in " ".join(result.decision.reasons)


def test_nonfinite_clearance_report_faults_closed() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    target = make_target(supervisor)
    report = supervisor.predict_clearance({"head_joint1": 0.02})
    supervisor._pending_clearance = None
    supervisor.checker.check_path = lambda *_args, **_kwargs: replace(
        report, min_distance=float("nan")
    )
    result = supervisor.evaluate(target, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert result.approved is None
    assert "non-finite" in " ".join(result.decision.reasons)


def test_source_timestamp_regression_latches_fault() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    first = make_target(supervisor, timestamp=1_000_000_000, position=0.0)
    assert supervisor.evaluate(first, now_mono_ns=1_000_000_001).approved is not None
    regressed = make_target(
        supervisor,
        sequence=2,
        timestamp=900_000_000,
        position=0.0,
    )
    result = supervisor.evaluate(regressed, now_mono_ns=1_100_000_000)
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert result.approved is None
    assert "non-monotonic" in " ".join(result.decision.reasons)


def test_first_target_and_long_gap_cannot_bypass_per_step_limit() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    first_jump = make_target(supervisor, position=0.2)
    result = supervisor.evaluate(first_jump, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert "step exceeds" in " ".join(result.decision.reasons)

    supervisor = make_supervisor()
    supervisor.start_preview()
    first = make_target(supervisor, position=0.0)
    assert supervisor.evaluate(first, now_mono_ns=1_000_000_001).approved is not None
    delayed_jump = make_target(
        supervisor,
        sequence=2,
        timestamp=21_000_000_000,
        position=0.2,
    )
    result = supervisor.evaluate(delayed_jump, now_mono_ns=21_000_000_001)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert "step exceeds" in " ".join(result.decision.reasons)


def test_rejected_dynamics_skip_expensive_clearance() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    target = make_target(supervisor, position=0.2)
    supervisor._pending_clearance = None

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("clearance ran after a conclusive dynamics rejection")

    supervisor.checker.check_path = must_not_run
    result = supervisor.evaluate(target, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert "step exceeds" in " ".join(result.decision.reasons)
    assert "AssertionError" not in " ".join(result.decision.reasons)


def test_duplicate_timestamp_and_uncontrolled_joint_fault_without_raising() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    first = make_target(supervisor, position=0.0)
    assert supervisor.evaluate(first, now_mono_ns=1_000_000_001).approved is not None
    duplicate_time = make_target(
        supervisor,
        sequence=2,
        timestamp=1_000_000_000,
        position=0.0,
    )
    result = supervisor.evaluate(duplicate_time, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert result.approved is None

    supervisor = make_supervisor()
    supervisor.start_preview()
    safe = make_target(supervisor, position=0.0)
    uncontrolled = safe.model_copy(
        update={
                "joints": (
                    JointTarget(
                        name="left_gripper_r_finger_joint", position_rad=0.0
                    ),
                ),
        }
    )
    result = supervisor.evaluate(uncontrolled, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert "unknown or non-revolute" in " ".join(result.decision.reasons)


def test_keyboard_interrupt_from_clearance_becomes_fault_decision() -> None:
    supervisor = make_supervisor()
    supervisor.start_preview()
    target = make_target(supervisor)
    supervisor._pending_clearance = None

    def interrupting_clearance(*_args, **_kwargs):
        raise KeyboardInterrupt("injected interrupt")

    supervisor.checker.check_path = interrupting_clearance
    result = supervisor.evaluate(target, now_mono_ns=1_000_000_001)
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert result.approved is None
    assert supervisor.state is SupervisorState.FAULT
    assert "KeyboardInterrupt" in " ".join(result.decision.reasons)


def preview_supervisor() -> SafetySupervisor:
    supervisor = make_supervisor()
    supervisor.start_preview()
    return supervisor


def _left_arm_solution(supervisor: SafetySupervisor, *, reach: float = 0.0):
    """A left-arm IK 'solution' plus the TCP it actually reaches."""
    joints = tuple(
        (f"left_arm_joint{index}", reach if index == 2 else 0.0)
        for index in range(1, 8)
    )
    pose = dict(supervisor.current_pose)
    pose.update(dict(joints))
    return joints, supervisor.checker.tcp_positions(pose)["left"]


def _residual_target(supervisor, *, declared, wrist, joints, sequence=1):
    commanded = supervisor.predict_clearance({"head_joint1": 0.02})
    return RobotTarget(
        session_id="preview",
        sequence=sequence,
        source_clock_id="camera-clock",
        source_mono_ns=1_000_000_000,
        arm_generation=supervisor.arm_generation,
        joints=(JointTarget(name="head_joint1", position_rad=0.02),),
        model_hash=CANONICAL_MANIFEST.fixed_mjcf_sha256,
        tool_hash=SupervisorPolicy().tool_hash,
        mapping_hash="mapping-v1",
        ik_residual_m=declared,
        predicted_clearance_m=commanded.min_distance,
        left_wrist_target_world_m=wrist,
        left_ik_joints_rad=joints,
    )


def test_an_honest_ik_residual_claim_is_approved() -> None:
    supervisor = preview_supervisor()
    joints, tcp = _left_arm_solution(supervisor)
    # The solution lands exactly on the declared wrist target: residual zero.
    target = _residual_target(supervisor, declared=0.0, wrist=tcp, joints=joints)
    assert supervisor.evaluate(target, now_mono_ns=1_000_000_001).decision.outcome is (
        SafetyOutcome.ALLOW
    )


def test_a_fabricated_ik_residual_is_caught() -> None:
    """The point of the check: a small declared residual on a pose that misses.

    Before the supervisor recomputed this, ``ik_residual_m`` was the one
    safety-relevant number in the target that nothing verified -- a solver could
    converge badly, report 0.0, and be approved on its own say-so.
    """
    supervisor = preview_supervisor()
    joints, tcp = _left_arm_solution(supervisor)
    far = (tcp[0] + 0.25, tcp[1], tcp[2])
    target = _residual_target(supervisor, declared=0.0, wrist=far, joints=joints)
    decision = supervisor.evaluate(target, now_mono_ns=1_000_000_001).decision
    assert decision.outcome is SafetyOutcome.HOLD
    assert any("recomputed IK residual exceeds policy" in r for r in decision.reasons)


def test_an_overstated_ik_residual_is_not_a_reason() -> None:
    """Declaring MORE error than was achieved is conservative, not a fault.

    A held limb carries its last approved residual forward, which is routinely
    larger than the fresh solve's, and that must not fault the frame.
    """
    supervisor = preview_supervisor()
    joints, tcp = _left_arm_solution(supervisor)
    target = _residual_target(supervisor, declared=0.02, wrist=tcp, joints=joints)
    assert supervisor.evaluate(target, now_mono_ns=1_000_000_001).decision.outcome is (
        SafetyOutcome.ALLOW
    )


def test_a_half_stated_residual_claim_faults_closed() -> None:
    """A wrist target with no solution, or a solution with no target, proves nothing."""
    supervisor = preview_supervisor()
    joints, tcp = _left_arm_solution(supervisor)
    for wrist, solution in ((tcp, None), (None, joints)):
        target = _residual_target(
            supervisor, declared=0.0, wrist=wrist, joints=solution
        )
        decision = supervisor.evaluate(target, now_mono_ns=1_000_000_001).decision
        assert decision.outcome is SafetyOutcome.HOLD
        assert any("claim is incomplete" in r for r in decision.reasons)


def test_a_target_declaring_no_wrist_claim_is_unaffected() -> None:
    """Held-both-arms frames declare nothing to verify and must still pass."""
    supervisor = preview_supervisor()
    target = make_target(supervisor)
    assert target.left_ik_joints_rad is None
    assert supervisor.evaluate(target, now_mono_ns=1_000_000_001).decision.outcome is (
        SafetyOutcome.ALLOW
    )
