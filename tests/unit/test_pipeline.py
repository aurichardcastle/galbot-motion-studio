import math

import pytest

from galbot_motion_studio.contracts.core import SafetyOutcome
from galbot_motion_studio.contracts.human import IdentityState
from galbot_motion_studio.ports.command import CommandReceipt, NullCommandSink
from galbot_motion_studio.pipeline import (
    MotionStudioPipeline,
    SIM_TELEOP_START_QPOS,
    _front_head_evidence_available,
    mapping_hash_for,
)
from galbot_motion_studio.safety.profiles import MotionProfile
from galbot_motion_studio.synthetic import synthetic_observation
from galbot_motion_studio.vision.calibration import (
    CalibrationError,
    CalibrationWindowPolicy,
    create_neutral_calibration,
)
from galbot_motion_studio.vision.liveness import FrameLivenessMonitor, LivenessPolicy
from galbot_motion_studio.vision.selection import (
    OccupancyObservation,
    OperatorSelectionManager,
    SelectionPolicy,
)

from test_calibration import _pose_head_landmarks
from test_left_arm_retargeting import arm_observation


SELECTION_POLICY = SelectionPolicy(
    max_timestamp_skew_ns=1_000_000,
    max_evidence_age_ns=100_000_000,
    max_selection_grace_ns=100_000_000,
    max_cumulative_occlusion_ns=200_000_000,
    operator_zone_normalized_bounds=(0.1, 0.1, 0.9, 0.9),
    max_candidate_center_delta_normalized=0.1,
    max_shoulder_width_relative_delta=0.1,
    max_eye_span_relative_delta=0.1,
)


def _occupancy_for(observation, *, count: int = 1, center=(0.5, 0.5)) -> OccupancyObservation:
    return OccupancyObservation(
        camera_id=observation.camera_id,
        source_clock_id=observation.source_clock_id,
        source_sequence=observation.sequence,
        capture_mono_ns=observation.capture_mono_ns,
        inference_complete_mono_ns=observation.inference_complete_mono_ns,
        content_fingerprint=observation.content_fingerprint or "missing-fingerprint",
        candidate_count=count,
        candidate_center_normalized_xy=center if count == 1 else None,
    )


def test_calibrated_observation_reaches_only_the_simulator_command_boundary() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    neutral = arm_observation(sequence=1, capture_mono_ns=1_000_000_000,
                              inference_complete_mono_ns=1_000_000_000)
    pipeline.calibrate(neutral)
    first = neutral.model_copy(
        update={"sequence": 2, "capture_mono_ns": 1_033_333_333,
                "inference_complete_mono_ns": 1_033_333_333}
    )
    result = pipeline.process(first, now_mono_ns=1_033_333_334)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert result.receipt is not None and result.receipt.sink == "null"
    assert len(sink.submissions) == 1
    assert result.target is not None
    assert {joint.name for joint in result.target.joints} == {
        "leg_joint4",
        "head_joint1",
        "head_joint2",
        *(f"left_arm_joint{index}" for index in range(1, 8)),
        "left_gripper_joint",
        *(f"right_arm_joint{index}" for index in range(1, 8)),
        "right_gripper_joint",
    }


def test_gripper_governor_never_exceeds_the_active_supervisor_envelope() -> None:
    """Sparse captures must not make the gripper issue an impossible dynamic step."""
    sim = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    hardware = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=NullCommandSink(),
        motion_profile=MotionProfile.HARDWARE,
    )

    for pipeline in (sim, hardware):
        limits = pipeline.gripper_trajectory.limits
        policy = pipeline.supervisor.policy
        assert limits.max_velocity_rad_s <= policy.max_joint_rate_rad_s
        assert limits.max_acceleration_rad_s2 <= policy.max_joint_acceleration_rad_s2
        assert limits.max_position_step_rad == policy.max_joint_step_rad


def test_profile_governors_have_no_infeasible_timing_window() -> None:
    """Any source interval must have a dynamically valid endpoint command."""
    for profile in (MotionProfile.SIM, MotionProfile.HARDWARE):
        pipeline = MotionStudioPipeline(
            source_clock_id="camera-clock",
            sink=NullCommandSink(),
            motion_profile=profile,
        )
        for governor in (pipeline.trajectory, pipeline.gripper_trajectory):
            limits = governor.limits
            assert limits.max_position_step_rad is not None
            assert limits.max_velocity_rad_s < 2.0 * math.sqrt(
                limits.max_acceleration_rad_s2 * limits.max_position_step_rad
            )


def test_gripper_governor_stays_dynamically_feasible_across_a_sparse_capture_gap() -> None:
    """The profile configuration must survive the 140 ms replayed source gap."""
    for profile in (MotionProfile.SIM, MotionProfile.HARDWARE):
        pipeline = MotionStudioPipeline(
            source_clock_id="camera-clock",
            sink=NullCommandSink(),
            motion_profile=profile,
        )
        governor = pipeline.gripper_trajectory
        names = {"left_gripper_joint", "right_gripper_joint"}
        position = {name: 0.1 for name in names}
        governor.reset(position, timestamp_ns=1_000_000_000)
        timestamp_ns = 1_000_000_000
        previous_velocity = {name: 0.0 for name in names}
        policy = pipeline.supervisor.policy

        # Build momentum, then replay the exact class of sparse interval that
        # previously forced a 50 mrad cap to violate acceleration on deceleration.
        for elapsed_ns in (*([50_000_000] * 12), 139_700_000, 64_000_000):
            timestamp_ns += elapsed_ns
            next_position = governor.step(
                {name: 1.6 for name in names}, timestamp_ns=timestamp_ns
            )
            elapsed_s = elapsed_ns / 1_000_000_000
            for name in names:
                step = next_position[name] - position[name]
                velocity = step / elapsed_s
                acceleration = (velocity - previous_velocity[name]) / elapsed_s
                assert abs(step) <= policy.max_joint_step_rad + 1e-12
                assert abs(velocity) <= policy.max_joint_rate_rad_s + 1e-9
                assert abs(acceleration) <= policy.max_joint_acceleration_rad_s2 + 1e-7
                previous_velocity[name] = velocity
            position = next_position


def test_arm_governor_stays_dynamically_feasible_across_replay_p95_gaps() -> None:
    """Sparse capture must not make a full-speed arm trip the supervisor."""
    for profile in (MotionProfile.SIM, MotionProfile.HARDWARE):
        pipeline = MotionStudioPipeline(
            source_clock_id="camera-clock",
            sink=NullCommandSink(),
            motion_profile=profile,
        )
        governor = pipeline.trajectory
        governor.reset({"joint": 0.0}, timestamp_ns=1_000_000_000)
        position = 0.0
        timestamp_ns = 1_000_000_000
        previous_velocity = 0.0
        policy = pipeline.supervisor.policy

        # 203/223 ms are the retained replay p95 values. The pre-fix SIM arm
        # triple (1.5 rad/s, 2 rad/s², 100 mrad) could make them reconstruct as
        # >2 rad/s² after the position cap rewrote velocity.
        for elapsed_ns in (*([50_000_000] * 24), 203_000_000, 223_000_000):
            timestamp_ns += elapsed_ns
            next_position = governor.step({"joint": 10.0}, timestamp_ns=timestamp_ns)[
                "joint"
            ]
            elapsed_s = elapsed_ns / 1_000_000_000
            step = next_position - position
            velocity = step / elapsed_s
            acceleration = (velocity - previous_velocity) / elapsed_s
            assert abs(step) <= policy.max_joint_step_rad + 1e-12
            assert abs(velocity) <= policy.max_joint_rate_rad_s + 1e-9
            assert abs(acceleration) <= policy.max_joint_acceleration_rad_s2 + 1e-7
            position = next_position
            previous_velocity = velocity


def test_selection_enabled_pipeline_requires_matched_evidence_and_recalibration() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=sink,
        selection_manager=OperatorSelectionManager(SELECTION_POLICY),
    )
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
        content_fingerprint="frame-1",
    )
    neutral_occupancy = _occupancy_for(neutral)

    acquired = pipeline.acquire_operator(
        neutral_occupancy, neutral, now_mono_ns=1_000_000_001
    )
    assert acquired.accepted and acquired.selection_id == 1
    with pytest.raises(CalibrationError, match="matched occupancy evidence"):
        pipeline.calibrate(neutral)
    pipeline.calibrate(
        neutral,
        occupancy=neutral_occupancy,
        now_mono_ns=1_000_000_001,
    )

    first = neutral.model_copy(
        update={
            "sequence": 2,
            "capture_mono_ns": 1_033_333_333,
            "inference_complete_mono_ns": 1_033_333_333,
            "content_fingerprint": "frame-2",
        }
    )
    assert pipeline.process(
        first,
        now_mono_ns=1_033_333_334,
        occupancy=_occupancy_for(first),
    ).decision.outcome is SafetyOutcome.ALLOW

    missing = first.model_copy(
        update={
            "sequence": 3,
            "capture_mono_ns": 1_066_666_666,
            "inference_complete_mono_ns": 1_066_666_666,
            "content_fingerprint": "frame-3",
        }
    )
    held = pipeline.process(missing, now_mono_ns=1_066_666_667)
    assert held.decision.outcome is SafetyOutcome.HOLD
    assert "OCCUPANCY_MISSING" in held.decision.reasons[0]

    resumed = missing.model_copy(
        update={
            "sequence": 4,
            "capture_mono_ns": 1_100_000_000,
            "inference_complete_mono_ns": 1_100_000_000,
            "content_fingerprint": "frame-4",
        }
    )
    # The omitted reading started an occlusion gap; exact current evidence may
    # re-lock only within the policy budget and can then use the same calibration.
    assert pipeline.process(
        resumed,
        now_mono_ns=1_100_000_001,
        occupancy=_occupancy_for(resumed),
    ).decision.outcome is SafetyOutcome.ALLOW

    new_selection = resumed.model_copy(
        update={
            "sequence": 5,
            "capture_mono_ns": 1_133_333_333,
            "inference_complete_mono_ns": 1_133_333_333,
            "content_fingerprint": "frame-5",
        }
    )
    assert pipeline.acquire_operator(
        _occupancy_for(new_selection), new_selection, now_mono_ns=1_133_333_334
    ).selection_id == 2
    changed_generation = new_selection.model_copy(
        update={
            "sequence": 6,
            "capture_mono_ns": 1_166_666_666,
            "inference_complete_mono_ns": 1_166_666_666,
            "content_fingerprint": "frame-6",
        }
    )
    mismatch = pipeline.process(
        changed_generation,
        now_mono_ns=1_166_666_667,
        occupancy=_occupancy_for(changed_generation),
    )
    assert mismatch.decision.outcome is SafetyOutcome.HOLD
    assert "calibration generation mismatch" in mismatch.decision.reasons[0]


def test_selection_enabled_calibration_does_not_reject_archived_evidence_as_stale() -> None:
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=NullCommandSink(),
        selection_manager=OperatorSelectionManager(SELECTION_POLICY),
    )
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
        content_fingerprint="frame-1",
    )
    assert pipeline.acquire_operator(
        _occupancy_for(neutral), neutral, now_mono_ns=1_000_000_001
    ).accepted
    stale = neutral.model_copy(
        update={
            "sequence": 2,
            "capture_mono_ns": 1_010_000_000,
            "inference_complete_mono_ns": 1_010_000_000,
            "content_fingerprint": "frame-2",
        }
    )
    pipeline.calibrate(
        stale,
        occupancy=_occupancy_for(stale),
        now_mono_ns=1_200_000_000,
    )


def test_selection_calibration_keeps_an_established_liveness_session() -> None:
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=NullCommandSink(),
        liveness_monitor=FrameLivenessMonitor(LivenessPolicy(max_static_ns=100_000_000)),
        selection_manager=OperatorSelectionManager(SELECTION_POLICY),
    )

    def frame(sequence: int, timestamp_ns: int):
        return arm_observation(
            sequence=sequence,
            capture_mono_ns=timestamp_ns,
            inference_complete_mono_ns=timestamp_ns,
            content_fingerprint=f"frame-{sequence}",
        )

    # Establish source liveness before an explicit acquisition.
    first = frame(1, 1_000_000_000)
    pipeline.gate.evaluate(first, now_mono_ns=1_000_000_001, occupancy=_occupancy_for(first))
    second = frame(2, 1_033_333_333)
    pipeline.gate.evaluate(second, now_mono_ns=1_033_333_334, occupancy=_occupancy_for(second))
    acquired = frame(3, 1_066_666_666)
    assert pipeline.acquire_operator(
        _occupancy_for(acquired), acquired, now_mono_ns=1_066_666_667
    ).accepted

    neutral = frame(4, 1_100_000_000)
    pipeline.calibrate(
        neutral,
        occupancy=_occupancy_for(neutral),
        now_mono_ns=1_100_000_001,
    )
    live = frame(5, 1_133_333_333)
    result = pipeline.process(
        live,
        now_mono_ns=1_133_333_334,
        occupancy=_occupancy_for(live),
    )
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert pipeline.gate.selection_id == 1


def test_selection_window_can_exceed_live_frame_age_without_consuming_gate_state() -> None:
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=NullCommandSink(),
        selection_manager=OperatorSelectionManager(SELECTION_POLICY),
    )
    acquired = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
        content_fingerprint="frame-1",
    )
    assert pipeline.acquire_operator(
        _occupancy_for(acquired), acquired, now_mono_ns=1_000_000_001
    ).accepted
    samples = tuple(
        arm_observation(
            sequence=sequence,
            capture_mono_ns=1_000_000_000 + (sequence - 1) * 33_333_333,
            inference_complete_mono_ns=1_000_000_000 + (sequence - 1) * 33_333_333,
            content_fingerprint=f"frame-{sequence}",
        )
        for sequence in range(2, 22)
    )
    pipeline.calibrate_window(
        samples,
        CalibrationWindowPolicy(20, 1_000_000_000, 0.01, 0.01, 0.01),
        occupancies=tuple(
            _occupancy_for(sample, center=(0.5 + index * 0.004, 0.5))
            for index, sample in enumerate(samples)
        ),
        now_mono_ns=samples[-1].inference_complete_mono_ns + 1,
    )
    # Window calibration is an authorization transaction, not a command-frame
    # consume: a later fresh frame remains the first processable frame.
    later = arm_observation(
        sequence=22,
        capture_mono_ns=samples[-1].capture_mono_ns + 33_333_333,
        inference_complete_mono_ns=samples[-1].inference_complete_mono_ns + 33_333_333,
        content_fingerprint="frame-22",
    )
    assert pipeline.process(
        later, now_mono_ns=later.inference_complete_mono_ns + 1, occupancy=_occupancy_for(later)
    ).decision.outcome is SafetyOutcome.ALLOW


def test_invalid_calibration_does_not_consume_the_selection_frame() -> None:
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=NullCommandSink(),
        selection_manager=OperatorSelectionManager(SELECTION_POLICY),
    )
    acquired = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
        content_fingerprint="frame-1",
    )
    assert pipeline.acquire_operator(
        _occupancy_for(acquired), acquired, now_mono_ns=1_000_000_001
    ).accepted
    candidate = acquired.model_copy(
        update={
            "sequence": 2,
            "capture_mono_ns": 1_033_333_333,
            "inference_complete_mono_ns": 1_033_333_333,
            "content_fingerprint": "frame-2",
        }
    )
    invalid = candidate.model_copy(update={"identity": IdentityState.LOST})
    with pytest.raises(CalibrationError, match="stable"):
        pipeline.calibrate(
            invalid,
            occupancy=_occupancy_for(invalid),
            now_mono_ns=1_033_333_334,
        )
    # The geometry rejection happened before the gate: the corrected evidence
    # with the same frame reference remains eligible to be calibrated once.
    pipeline.calibrate(
        candidate,
        occupancy=_occupancy_for(candidate),
        now_mono_ns=1_033_333_334,
    )


def test_rejected_selection_evidence_does_not_mutate_calibration_gate_state() -> None:
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=NullCommandSink(),
        selection_manager=OperatorSelectionManager(SELECTION_POLICY),
    )
    acquired = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
        content_fingerprint="frame-1",
    )
    assert pipeline.acquire_operator(
        _occupancy_for(acquired), acquired, now_mono_ns=1_000_000_001
    ).accepted
    candidate = acquired.model_copy(
        update={
            "sequence": 2,
            "capture_mono_ns": 1_033_333_333,
            "inference_complete_mono_ns": 1_033_333_333,
            "content_fingerprint": "frame-2",
        }
    )
    with pytest.raises(CalibrationError, match="OCCUPANCY_AMBIGUOUS"):
        pipeline.calibrate(
            candidate,
            occupancy=_occupancy_for(candidate, count=2),
            now_mono_ns=1_033_333_334,
        )
    pipeline.calibrate(
        candidate,
        occupancy=_occupancy_for(candidate),
        now_mono_ns=1_033_333_334,
    )


def test_unstable_observation_holds_before_the_sink() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    neutral = arm_observation(sequence=1, capture_mono_ns=1_000_000_000,
                              inference_complete_mono_ns=1_000_000_000)
    pipeline.calibrate(neutral)
    lost = neutral.model_copy(
        update={"sequence": 2, "capture_mono_ns": 1_010_000_000,
                "inference_complete_mono_ns": 1_010_000_000,
                "aggregate_confidence": 0.1}
    )
    result = pipeline.process(lost, now_mono_ns=1_010_000_001)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert result.receipt is None
    assert sink.submissions == []


def test_perception_ingress_fault_latches_before_a_robot_target_exists() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    result = pipeline.fault_perception(
        session_id="camera-session",
        sequence=1,
        source_mono_ns=1_000_000_000,
        now_mono_ns=1_000_000_001,
        reason="capture timestamp is not strictly monotonic",
    )
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert result.target is None
    assert result.held_groups == {"head", "left_arm", "right_arm", "torso"}
    assert sink.submissions == []


def test_windowed_calibration_uses_the_validated_mean_shoulder_scale() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    policy = CalibrationWindowPolicy(
        min_samples=3,
        max_window_ns=100_000_000,
        max_center_deviation_normalized=0.01,
        max_shoulder_width_deviation_normalized=0.01,
        max_eye_span_deviation_normalized=0.01,
    )
    samples = tuple(
        arm_observation(
            sequence=sequence,
            capture_mono_ns=1_000_000_000 + (sequence - 1) * 20_000_000,
            inference_complete_mono_ns=1_000_000_000 + (sequence - 1) * 20_000_000,
        )
        for sequence in (1, 2, 3)
    )
    pipeline.calibrate_window(samples, policy)
    assert pipeline._left_arm_calibration.shoulder_width_normalized == pytest.approx(0.4)
    assert pipeline._right_arm_calibration.shoulder_width_normalized == pytest.approx(0.4)
    result = pipeline.process(
        samples[-1].model_copy(
            update={
                "sequence": 4,
                "capture_mono_ns": 1_080_000_000,
                "inference_complete_mono_ns": 1_080_000_000,
            }
        ),
        now_mono_ns=1_080_000_001,
    )
    assert result.decision.outcome is SafetyOutcome.ALLOW


def test_rejected_windowed_calibration_leaves_the_pipeline_uncalibrated() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    policy = CalibrationWindowPolicy(
        min_samples=3,
        max_window_ns=100_000_000,
        max_center_deviation_normalized=0.01,
        max_shoulder_width_deviation_normalized=0.01,
        max_eye_span_deviation_normalized=0.01,
    )
    samples = tuple(
        arm_observation(
            sequence=sequence,
            capture_mono_ns=1_000_000_000 + (sequence - 1) * 20_000_000,
            inference_complete_mono_ns=1_000_000_000 + (sequence - 1) * 20_000_000,
            camera_id="different-camera" if sequence == 3 else "webcam",
        )
        for sequence in (1, 2, 3)
    )

    with pytest.raises(CalibrationError, match="source or camera"):
        pipeline.calibrate_window(samples, policy)

    assert pipeline._face_calibration is None
    with pytest.raises(RuntimeError, match="calibrate the pipeline"):
        pipeline.process(samples[-1], now_mono_ns=samples[-1].capture_mono_ns + 1)


def test_liveness_recovery_preserves_healthy_stream_evidence() -> None:
    """A supervisor resume is not a camera/source reset.

    The first post-calibration frame cannot prove a content transition, so it
    holds. The next two genuinely different frames must both be allowed; this
    catches the previous HOLD/ALLOW livelock caused by clearing the liveness
    monitor during supervisor recovery.
    """
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=sink,
        liveness_monitor=FrameLivenessMonitor(LivenessPolicy(max_static_ns=500_000_000)),
    )
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
        content_fingerprint="calibration",
    )
    pipeline.calibrate(neutral)
    outcomes = []
    for sequence, fingerprint in ((2, "a"), (3, "b"), (4, "c")):
        timestamp = 1_000_000_000 + (sequence - 1) * 33_333_333
        result = pipeline.process(
            neutral.model_copy(
                update={
                    "sequence": sequence,
                    "capture_mono_ns": timestamp,
                    "inference_complete_mono_ns": timestamp,
                    "content_fingerprint": fingerprint,
                }
            ),
            now_mono_ns=timestamp + 1,
        )
        outcomes.append(result.decision.outcome)
    assert outcomes == [SafetyOutcome.HOLD, SafetyOutcome.ALLOW, SafetyOutcome.ALLOW]
    assert len(sink.submissions) == 2


def test_sim_confidence_floor_is_relaxed_without_relaxing_hardware() -> None:
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    outcomes = []
    for profile in (MotionProfile.SIM, MotionProfile.HARDWARE):
        pipeline = MotionStudioPipeline(
            source_clock_id="camera-clock",
            sink=NullCommandSink(),
            motion_profile=profile,
        )
        pipeline.calibrate(neutral)
        marginal = neutral.model_copy(
            update={
                "sequence": 2,
                "capture_mono_ns": 1_033_333_333,
                "inference_complete_mono_ns": 1_033_333_333,
                "aggregate_confidence": 0.4,
            }
        )
        outcomes.append(
            pipeline.process(
                marginal, now_mono_ns=1_033_333_334
            ).decision.outcome
        )
    assert outcomes == [SafetyOutcome.ALLOW, SafetyOutcome.HOLD]


def test_gate_order_rejects_confidence_before_retargeting() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    neutral = arm_observation(sequence=1, capture_mono_ns=1_000_000_000,
                              inference_complete_mono_ns=1_000_000_000)
    pipeline.calibrate(neutral)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("retargeting ran before the observation gate")

    pipeline.left_arm.map = must_not_run
    invalid = neutral.model_copy(
        update={"sequence": 2, "capture_mono_ns": 1_010_000_000,
                "inference_complete_mono_ns": 1_010_000_000,
                "aggregate_confidence": 0.1}
    )
    result = pipeline.process_fail_closed(invalid, now_mono_ns=1_010_000_001)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert "LOW_CONFIDENCE" in " ".join(result.decision.reasons)
    assert sink.submissions == []


def test_unknown_pipeline_exception_becomes_fault_not_a_command() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    neutral = arm_observation(sequence=1, capture_mono_ns=1_000_000_000,
                              inference_complete_mono_ns=1_000_000_000)
    pipeline.calibrate(neutral)

    def explode(*_args, **_kwargs):
        raise LookupError("unknown injected failure")

    pipeline.left_arm.map = explode
    first = neutral.model_copy(
        update={"sequence": 2, "capture_mono_ns": 1_033_333_333,
                "inference_complete_mono_ns": 1_033_333_333}
    )
    result = pipeline.process_fail_closed(first, now_mono_ns=1_033_333_334)
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert result.receipt is None
    assert "LookupError" in " ".join(result.decision.reasons)
    assert sink.submissions == []


def test_unknown_sink_status_becomes_fault_and_cannot_authorize_a_successor() -> None:
    class UnknownStatusSink(NullCommandSink):
        def submit(self, command):
            self.submissions.append(command)
            return CommandReceipt(
                accepted="MYSTERY",  # type: ignore[arg-type]
                command_sequence=command.target.sequence,
                sink="unknown-status",
            )

    sink = UnknownStatusSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    neutral = arm_observation(sequence=1, capture_mono_ns=1_000_000_000,
                              inference_complete_mono_ns=1_000_000_000)
    pipeline.calibrate(neutral)
    first = neutral.model_copy(
        update={"sequence": 2, "capture_mono_ns": 1_033_333_333,
                "inference_complete_mono_ns": 1_033_333_333}
    )
    result = pipeline.process_fail_closed(first, now_mono_ns=1_033_333_334)
    assert result.decision.outcome is SafetyOutcome.FAULT
    assert "MYSTERY" in " ".join(result.decision.reasons)
    assert len(sink.submissions) == 1

    successor = neutral.model_copy(
        update={"sequence": 3, "capture_mono_ns": 1_100_000_000,
                "inference_complete_mono_ns": 1_100_000_000}
    )
    later = pipeline.process_fail_closed(successor, now_mono_ns=1_100_000_001)
    assert later.decision.outcome is not SafetyOutcome.ALLOW
    assert len(sink.submissions) == 1


def test_sim_tracking_hold_recovers_on_the_next_valid_frame() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    pipeline.calibrate(neutral)
    lost = neutral.model_copy(
        update={
            "sequence": 2,
            "capture_mono_ns": 1_033_333_333,
            "inference_complete_mono_ns": 1_033_333_333,
            "aggregate_confidence": 0.1,
        }
    )
    assert pipeline.process(lost, now_mono_ns=1_033_333_334).decision.outcome is SafetyOutcome.HOLD

    outcomes = []
    for sequence in (3, 4, 5):
        timestamp = 1_000_000_000 + (sequence - 1) * 33_333_333
        recovered = neutral.model_copy(
            update={
                "sequence": sequence,
                "capture_mono_ns": timestamp,
                "inference_complete_mono_ns": timestamp,
            }
        )
        outcomes.append(
            pipeline.process(recovered, now_mono_ns=timestamp + 1).decision.outcome
        )
    assert outcomes == [SafetyOutcome.ALLOW, SafetyOutcome.ALLOW, SafetyOutcome.ALLOW]
    assert len(sink.submissions) == 3


def test_hardware_policy_tracking_hold_requires_three_valid_frames() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=sink,
        motion_profile=MotionProfile.HARDWARE,
    )
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    pipeline.calibrate(neutral)
    lost = neutral.model_copy(
        update={
            "sequence": 2,
            "capture_mono_ns": 1_033_333_333,
            "inference_complete_mono_ns": 1_033_333_333,
            "aggregate_confidence": 0.1,
        }
    )
    assert pipeline.process(
        lost, now_mono_ns=1_033_333_334
    ).decision.outcome is SafetyOutcome.HOLD

    outcomes = []
    for sequence in (3, 4, 5):
        timestamp = 1_000_000_000 + (sequence - 1) * 33_333_333
        recovered = neutral.model_copy(
            update={
                "sequence": sequence,
                "capture_mono_ns": timestamp,
                "inference_complete_mono_ns": timestamp,
            }
        )
        outcomes.append(
            pipeline.process(recovered, now_mono_ns=timestamp + 1).decision.outcome
        )
    assert outcomes == [SafetyOutcome.HOLD, SafetyOutcome.HOLD, SafetyOutcome.ALLOW]
    assert len(sink.submissions) == 1


def test_independent_hand_openness_moves_both_grippers() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="synthetic-clock")
    calibration = synthetic_observation(
        0, timestamp_ns=1_000_000_000, motion_fraction=0.5
    )
    pipeline.calibrate(calibration)
    last = None
    # Ramp from the calibrated pose instead of jumping straight to full extension.
    # The old form asked for the whole excursion on the first frame after
    # calibration; with the look-ahead horizon sized past the governor's braking
    # curve the swept check now looks as far along that abrupt path as the arm can
    # really travel, and correctly rejects part of it. That was the synthetic input
    # being unphysical, not the gripper behaviour under test -- a real operator
    # cannot teleport from neutral to full reach in 33 ms.
    for sequence in range(1, 61):
        timestamp = 1_000_000_000 + sequence * 33_333_333
        observation = synthetic_observation(
            sequence,
            timestamp_ns=timestamp,
            motion_fraction=min(1.0, 0.5 + 0.5 * sequence / 20.0),
        )
        last = pipeline.process(observation, now_mono_ns=timestamp + 1)
        assert last.decision.outcome is SafetyOutcome.ALLOW
    assert last is not None and last.target is not None
    joints = {joint.name: joint.position_rad for joint in last.target.joints}
    assert joints["left_gripper_joint"] > 0.75
    assert joints["right_gripper_joint"] < 0.2


def test_missing_hand_is_an_explicit_gripper_hold_not_a_green_success() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="synthetic-clock")
    calibration = synthetic_observation(
        0, timestamp_ns=1_000_000_000, motion_fraction=0.5
    )
    pipeline.calibrate(calibration)
    observation = synthetic_observation(
        1, timestamp_ns=1_033_333_333, motion_fraction=0.0
    ).model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in synthetic_observation(
                    1, timestamp_ns=1_033_333_333, motion_fraction=0.0
                ).landmarks
                if not landmark.name.startswith("right_hand_")
            )
        }
    )
    result = pipeline.process(observation, now_mono_ns=observation.capture_mono_ns + 1)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert result.held_grippers == {"right_gripper_joint"}
    assert result.held_gripper_reasons == (
        ("right_gripper_joint", "HAND_NOT_TRACKABLE"),
    )


def test_missing_hand_freezes_an_inflight_gripper_at_its_approved_pose() -> None:
    """Losing a hand must stop a moving gripper, not finish its stale command."""
    pipeline = MotionStudioPipeline(source_clock_id="synthetic-clock")
    pipeline.calibrate(synthetic_observation(0, timestamp_ns=1_000_000_000, motion_fraction=0.5))
    for sequence in range(1, 12):
        timestamp = 1_000_000_000 + sequence * 33_333_333
        active = synthetic_observation(sequence, timestamp_ns=timestamp, motion_fraction=1.0)
        pipeline.process(active, now_mono_ns=timestamp + 1)
    approved = pipeline.supervisor.current_pose["right_gripper_joint"]
    frozen_positions = []
    for sequence in range(12, 48):
        timestamp = 1_000_000_000 + sequence * 33_333_333
        active = synthetic_observation(sequence, timestamp_ns=timestamp, motion_fraction=1.0)
        missing_hand = active.model_copy(
            update={
                "landmarks": tuple(
                    landmark
                    for landmark in active.landmarks
                    if not landmark.name.startswith("right_hand_")
                )
            }
        )
        result = pipeline.process(missing_hand, now_mono_ns=timestamp + 1)
        assert result.held_grippers == {"right_gripper_joint"}
        assert pipeline._gripper_desired["right_gripper_joint"] == pytest.approx(approved)
        frozen_positions.append(commanded(result)["right_gripper_joint"])
    # The governor can decelerate for a few frames, but it must settle at the
    # approved pose rather than continue toward the pre-loss open/close target.
    assert frozen_positions[-1] == pytest.approx(approved, abs=1e-9)


# --------------------------------------------------------------------------
# Per-control-group holds.
#
# The operator's recorded 372-frame session produced 57 HOLDs, every one of them
# `observation: LOW_CONFIDENCE`, in runs of up to 27 consecutive frames -- 2.1 s
# of a dead robot -- because one wrist dipping under the bar stopped the head and
# both arms. A group is now held only when the landmarks that drive THAT command
# are untrustworthy. Gating the input is per-limb; judging the output never is.
# --------------------------------------------------------------------------

NEUTRAL = arm_observation(
    sequence=1,
    capture_mono_ns=1_000_000_000,
    inference_complete_mono_ns=1_000_000_000,
)
LEFT_ARM_JOINTS = tuple(f"left_arm_joint{index}" for index in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"right_arm_joint{index}" for index in range(1, 8))
HEAD_JOINTS = ("head_joint1", "head_joint2")


def moving_observation(
    sequence: int,
    *,
    left_wrist_confidence: float = 1.0,
    step: int = 1,
):
    """Both wrists and the nose move; only the left wrist's confidence varies."""
    timestamp = 1_000_000_000 + sequence * 33_333_333
    offset = 0.02 * step
    moved = []
    for landmark in NEUTRAL.landmarks:
        if landmark.name == "pose_15":
            moved.append(
                landmark.model_copy(
                    update={
                        "normalized_xyz": (0.80 + offset, 0.70 - offset, 0.0),
                        "visibility": left_wrist_confidence,
                        "presence": left_wrist_confidence,
                    }
                )
            )
        elif landmark.name == "pose_16":
            moved.append(
                landmark.model_copy(
                    update={"normalized_xyz": (0.20 - offset, 0.70 - offset, 0.0)}
                )
            )
        elif landmark.name == "face_1":
            moved.append(
                landmark.model_copy(update={"normalized_xyz": (0.50 + offset, 0.20, 0.0)})
            )
        else:
            moved.append(landmark)
    return NEUTRAL.model_copy(
        update={
            "sequence": sequence,
            "capture_mono_ns": timestamp,
            "inference_complete_mono_ns": timestamp,
            # Exactly what the MediaPipe adapter would declare for this frame.
            "aggregate_confidence": min(1.0, left_wrist_confidence),
            "landmarks": tuple(moved),
        }
    )


def torso_world_observation(
    sequence: int,
    yaw_rad: float,
    *,
    swap_shoulders: bool = False,
    capture_mono_ns: int | None = None,
) -> object:
    """Return the normal arm fixture with a metric shoulder line at ``yaw_rad``.

    The symmetric construction keeps the shoulder midpoint and width fixed, so
    a test changes exactly the torso angle and not apparent arm geometry.
    """
    timestamp = (
        1_000_000_000 + sequence * 33_333_333
        if capture_mono_ns is None
        else capture_mono_ns
    )
    half_width = 0.16
    left_world = (
        half_width * math.cos(yaw_rad),
        -0.42,
        half_width * math.sin(yaw_rad),
    )
    right_world = (-left_world[0], -0.42, -left_world[2])
    landmarks = tuple(
        landmark.model_copy(
            update={
                "world_xyz_m": (
                    (
                        right_world if swap_shoulders else left_world
                    )
                    if landmark.name == "pose_11"
                    else (
                        left_world if swap_shoulders else right_world
                    )
                )
            }
        )
        if landmark.name in {"pose_11", "pose_12"}
        else landmark
        for landmark in NEUTRAL.landmarks
    )
    return NEUTRAL.model_copy(
        update={
            "sequence": sequence,
            "capture_mono_ns": timestamp,
            "inference_complete_mono_ns": timestamp,
            "content_fingerprint": f"torso-{sequence}-{yaw_rad:.9f}-{swap_shoulders}",
            "landmarks": landmarks,
        }
    )


def commanded(result) -> dict[str, float]:
    assert result.target is not None
    return {joint.name: joint.position_rad for joint in result.target.joints}


def _continuous_torso_yaws(start_rad: float, target_rad: float) -> tuple[float, ...]:
    """A physically continuous source path, deliberately inside the 20° guard."""
    max_step = math.radians(15.0)
    steps = max(1, math.ceil(abs(target_rad - start_rad) / max_step))
    return tuple(
        start_rad + (target_rad - start_rad) * index / steps
        for index in range(1, steps + 1)
    )


def _advance_torso_continuously(
    pipeline: MotionStudioPipeline,
    *,
    neutral_rad: float,
    target_rad: float,
    first_sequence: int = 2,
) -> tuple[object, int]:
    """Feed a target through valid source steps and return its final result."""
    sequence = first_sequence
    result = None
    for yaw_rad in _continuous_torso_yaws(neutral_rad, target_rad):
        observation = torso_world_observation(sequence, yaw_rad)
        result = pipeline.process(observation, now_mono_ns=observation.capture_mono_ns + 1)
        sequence += 1
    assert result is not None
    return result, sequence


def test_torso_uses_the_calibrated_camera_yaw_as_its_zero() -> None:
    """A square pose at any permitted camera heading must command robot neutral."""
    neutral_yaw = math.radians(3.0)
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    calibration = torso_world_observation(1, neutral_yaw)
    pipeline.calibrate(calibration)

    result = pipeline.process(
        torso_world_observation(2, neutral_yaw),
        now_mono_ns=1_066_666_667,
    )
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert "torso" not in result.held_groups
    assert commanded(result)["leg_joint4"] == pytest.approx(
        SIM_TELEOP_START_QPOS["leg_joint4"], abs=1e-12
    )


def test_torso_sweeps_both_turn_directions_and_never_exceeds_its_emitted_rate() -> None:
    """Exercise every front-visible yaw cell through the real pipeline/governor.

    The outer loop covers signed left/right yaw at the deadband edge, ordinary
    range, and the exact camera boundary.  The inner stream proves that the
    torso-specific 0.35 rad/s limit constrains emitted leg_joint4 poses, rather
    than merely a raw retarget value.
    """
    for yaw_deg in (-75.0, -50.0, -10.0, 10.0, 50.0, 75.0):
        pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
        pipeline.calibrate(torso_world_observation(1, 0.0))
        previous = SIM_TELEOP_START_QPOS["leg_joint4"]
        sequence = 2
        stream = (*_continuous_torso_yaws(0.0, math.radians(yaw_deg)), *([math.radians(yaw_deg)] * 24))
        for source_yaw in stream:
            observation = torso_world_observation(sequence, source_yaw)
            result = pipeline.process(observation, now_mono_ns=observation.capture_mono_ns + 1)
            assert result.decision.outcome is SafetyOutcome.ALLOW
            assert "torso" not in result.held_groups
            current = commanded(result)["leg_joint4"]
            velocity = (current - previous) / 0.033333333
            assert abs(velocity) <= pipeline.torso.policy.max_rate_rad_s + 1e-7
            if sequence == 2:
                assert math.copysign(1.0, current - previous) == math.copysign(1.0, yaw_deg)
            previous = current
            sequence += 1


@pytest.mark.parametrize("neutral_deg", (-5.0, 0.0, 5.0))
@pytest.mark.parametrize("relative_deg", (-75.0, -74.9, 74.9, 75.0))
def test_torso_relative_observable_lattice_is_symmetric_about_every_valid_neutral(
    neutral_deg: float, relative_deg: float
) -> None:
    """The full +/-75 degree command span survives the allowed neutral offset."""
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    neutral_rad = math.radians(neutral_deg)
    pipeline.calibrate(torso_world_observation(1, neutral_rad))
    result, _ = _advance_torso_continuously(
        pipeline,
        neutral_rad=neutral_rad,
        target_rad=math.radians(neutral_deg + relative_deg),
    )
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert not {"torso", "left_arm", "right_arm"} & result.held_groups


@pytest.mark.parametrize("neutral_deg", (-5.0, 0.0, 5.0))
@pytest.mark.parametrize("relative_deg", (-75.1, 75.1))
def test_torso_relative_observable_lattice_holds_just_outside_the_safe_span(
    neutral_deg: float, relative_deg: float
) -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    neutral_rad = math.radians(neutral_deg)
    pipeline.calibrate(torso_world_observation(1, neutral_rad))
    # Reach the edge through physically bounded raw shoulder steps.  A direct
    # 75-degree one-frame fixture is a semantic discontinuity, correctly more
    # severe than a normal out-of-view hold.
    edge_rad = math.radians(neutral_deg + (75.0 if relative_deg > 0.0 else -75.0))
    _, sequence = _advance_torso_continuously(
        pipeline, neutral_rad=neutral_rad, target_rad=edge_rad
    )
    observation = torso_world_observation(
        sequence, math.radians(neutral_deg + relative_deg)
    )
    result = pipeline.process(observation, now_mono_ns=observation.capture_mono_ns + 1)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert result.held_groups == {"left_arm", "right_arm", "torso"}
    assert dict(result.held_group_reasons) == {
        "left_arm": "TORSO_YAW_OUT_OF_VIEW",
        "right_arm": "TORSO_YAW_OUT_OF_VIEW",
        "torso": "TORSO_YAW_OUT_OF_VIEW",
    }


@pytest.mark.parametrize("true_yaw_deg", (105.0, 110.0, 120.0, 150.0, 179.0))
def test_swapped_back_facing_shoulders_cannot_reenter_as_a_plausible_front_turn(
    true_yaw_deg: float,
) -> None:
    """A label swap maps the back field into the numeric front field; hold it."""
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(torso_world_observation(1, 0.0))
    swapped = torso_world_observation(
        2, math.radians(true_yaw_deg), swap_shoulders=True
    )
    observation = swapped.model_copy(
        update={
            # A back-facing operator has no independent front-face cue. The
            # shoulder line alone cannot determine an oriented tummy direction.
            "landmarks": tuple(
                landmark
                for landmark in swapped.landmarks
                if not landmark.name.startswith("face_")
            )
        }
    )
    result = pipeline.process(observation, now_mono_ns=observation.capture_mono_ns + 1)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert {"left_arm", "right_arm", "torso"} <= result.held_groups
    reasons = dict(result.held_group_reasons)
    coupled_reasons = {
        reasons[group] for group in ("left_arm", "right_arm", "torso")
    }
    # A raw jump beyond the 20-degree semantic guard is now terminal even when
    # the face is missing; a near-neutral swapped back pose remains a normal
    # front-evidence hold.  Neither case may command the coupled group.
    assert coupled_reasons <= {
        "TORSO_FRONT_UNOBSERVABLE",
        "TORSO_YAW_DISCONTINUITY",
    }
    assert len(coupled_reasons) == 1


def test_torso_out_of_view_holds_until_raw_shoulder_continuity_is_reestablished() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(torso_world_observation(1, 0.0))
    # A real continuous turn reaches the edge normally, then becomes unobservable.
    _, sequence = _advance_torso_continuously(
        pipeline, neutral_rad=0.0, target_rad=math.radians(75.0)
    )
    outside = torso_world_observation(sequence, math.radians(76.0))
    held = pipeline.process(outside, now_mono_ns=outside.capture_mono_ns + 1)
    assert dict(held.held_group_reasons)["torso"] == "TORSO_YAW_OUT_OF_VIEW"
    # A bounded front-facing return resumes all coupled controls. A source gap
    # or semantic jump would instead lock calibration (covered below).
    resumed, _ = _advance_torso_continuously(
        pipeline,
        neutral_rad=math.radians(75.0),
        target_rad=math.radians(60.0),
        first_sequence=sequence + 1,
    )
    assert not {"torso", "left_arm", "right_arm"} & resumed.held_groups


def test_torso_identity_discontinuity_requires_explicit_recalibration() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(torso_world_observation(1, 0.0))
    accepted = torso_world_observation(2, 0.0)
    pipeline.process(accepted, now_mono_ns=accepted.capture_mono_ns + 1)
    # A 30° one-frame torso step is not physically credible at 30 Hz and can be
    # a shoulder-label swap. It must not become re-entry evidence.
    discontinuity = torso_world_observation(3, math.radians(30.0))
    held = pipeline.process(
        discontinuity, now_mono_ns=discontinuity.capture_mono_ns + 1
    )
    assert dict(held.held_group_reasons)["torso"] == "TORSO_YAW_DISCONTINUITY"
    later = torso_world_observation(4, 0.0)
    locked = pipeline.process(later, now_mono_ns=later.capture_mono_ns + 1)
    assert dict(locked.held_group_reasons) == {
        "left_arm": "TORSO_YAW_RECALIBRATION_REQUIRED",
        "right_arm": "TORSO_YAW_RECALIBRATION_REQUIRED",
        "torso": "TORSO_YAW_RECALIBRATION_REQUIRED",
    }
    pipeline.calibrate(later)
    fresh = torso_world_observation(5, 0.0)
    resumed = pipeline.process(fresh, now_mono_ns=fresh.capture_mono_ns + 1)
    assert not {"torso", "left_arm", "right_arm"} & resumed.held_groups


def test_torso_front_evidence_uses_face_geometry_not_synthesized_confidence() -> None:
    front = torso_world_observation(2, 0.0)
    calibration = create_neutral_calibration(front)
    assert _front_head_evidence_available(front, calibration)
    profile = front.model_copy(
        update={
            "landmarks": tuple(
                landmark.model_copy(update={"normalized_xyz": (0.5, 0.2, 0.0)})
                if landmark.name in {"face_33", "face_263"}
                else landmark
                for landmark in front.landmarks
            )
        }
    )
    # Face-landmark visibility/presence can be adapter-synthesized at 1.0; the
    # collapsed eye span is the independent front-facing rejection.
    assert not _front_head_evidence_available(profile, calibration)


def test_face_dropout_holds_coupled_motion_but_continuous_shoulders_can_reenter() -> None:
    """Face loss never commands, but continuous raw yaw can cross neutral."""
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    start = 1_000_000_000
    neutral = torso_world_observation(1, 0.0, capture_mono_ns=start)
    pipeline.calibrate(neutral)
    tracked = torso_world_observation(
        2, math.radians(8.0), capture_mono_ns=start + 33_000_000
    )
    assert not {
        "torso", "left_arm", "right_arm"
    } & pipeline.process(tracked, now_mono_ns=tracked.capture_mono_ns + 1).held_groups

    for sequence, yaw_deg in zip(range(3, 7), (4.0, 0.0, -4.0, -6.0)):
        timestamp = start + (sequence - 1) * 33_000_000
        raw = torso_world_observation(
            sequence, math.radians(yaw_deg), capture_mono_ns=timestamp
        )
        face_lost = raw.model_copy(
            update={
                "landmarks": tuple(
                    landmark
                    for landmark in raw.landmarks
                    if not landmark.name.startswith("face_")
                )
            }
        )
        held = pipeline.process(face_lost, now_mono_ns=timestamp + 1)
        assert {
            "torso",
            "left_arm",
            "right_arm",
        } <= held.held_groups
        assert dict(held.held_group_reasons)["torso"] == "TORSO_FRONT_UNOBSERVABLE"

    reentered = torso_world_observation(
        7, math.radians(-8.0), capture_mono_ns=start + 6 * 33_000_000
    )
    result = pipeline.process(reentered, now_mono_ns=reentered.capture_mono_ns + 1)
    assert not {"torso", "left_arm", "right_arm"} & result.held_groups


def test_face_dropout_does_not_hide_a_raw_shoulder_identity_jump() -> None:
    """A face hold is not permission to skip the raw semantic-jump guard."""
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    start = 1_000_000_000
    neutral = torso_world_observation(1, 0.0, capture_mono_ns=start)
    pipeline.calibrate(neutral)
    tracked = torso_world_observation(
        2, 0.0, capture_mono_ns=start + 33_000_000
    )
    pipeline.process(tracked, now_mono_ns=tracked.capture_mono_ns + 1)
    jumped = torso_world_observation(
        3, math.radians(30.0), capture_mono_ns=start + 66_000_000
    )
    face_lost = jumped.model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in jumped.landmarks
                if not landmark.name.startswith("face_")
            )
        }
    )
    held = pipeline.process(face_lost, now_mono_ns=face_lost.capture_mono_ns + 1)
    assert dict(held.held_group_reasons)["torso"] == "TORSO_YAW_DISCONTINUITY"
    later = torso_world_observation(
        4, 0.0, capture_mono_ns=start + 99_000_000
    )
    locked = pipeline.process(later, now_mono_ns=later.capture_mono_ns + 1)
    assert dict(locked.held_group_reasons)["torso"] == "TORSO_YAW_RECALIBRATION_REQUIRED"


def test_torso_refuses_to_compare_against_an_old_yaw_observation() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(torso_world_observation(1, 0.0))
    accepted = torso_world_observation(2, 0.0)
    assert not pipeline.process(
        accepted, now_mono_ns=accepted.capture_mono_ns + 1
    ).held_groups & {"torso"}
    # Five 30-Hz captures later, even the same heading has no current continuity
    # evidence. The whole torso-coupled group holds until a fresh path exists.
    delayed = torso_world_observation(30, 0.0)
    result = pipeline.process(delayed, now_mono_ns=delayed.capture_mono_ns + 1)
    assert dict(result.held_group_reasons) == {
        "left_arm": "TORSO_YAW_CONTINUITY_GAP",
        "right_arm": "TORSO_YAW_CONTINUITY_GAP",
        "torso": "TORSO_YAW_CONTINUITY_GAP",
    }
    # The gap holds THIS frame and re-anchors continuity. While the face is
    # visible the operator cannot be turned far enough for a shoulder-label
    # swap, so the next continuous frame resumes rather than demanding a
    # recalibration the live CLI has no way to perform. Measured 2026-08-26:
    # latching here held both arms on 520 of 521 frames of a real camera run.
    resumed_seq = torso_world_observation(31, 0.0)
    resumed = pipeline.process(resumed_seq, now_mono_ns=resumed_seq.capture_mono_ns + 1)
    assert not {"torso", "left_arm", "right_arm"} & resumed.held_groups


def test_a_busy_control_worker_does_not_manufacture_a_continuity_gap() -> None:
    """Back-pressure is not a blind interval.

    Measured 2026-08-26 on a real camera run: arm IK back-pressure discarded
    ~54% of analyzed observations behind a latest-wins queue, so continuity
    measured on the surviving command stream reported worker scheduling latency
    as missing data and held both arms on 99 of 286 frames. The detector saw the
    operator throughout -- the evidence existed and was recorded, it was just
    never looked at.
    """
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(torso_world_observation(1, 0.0))
    accepted = torso_world_observation(2, 0.0)
    pipeline.observe_source_continuity(accepted)
    assert not pipeline.process(
        accepted, now_mono_ns=accepted.capture_mono_ns + 1
    ).held_groups & {"torso"}
    # The detector kept producing usable shoulders; only the worker was busy, so
    # none of these reached process().
    for sequence in range(3, 31):
        pipeline.observe_source_continuity(torso_world_observation(sequence, 0.0))
    delayed = torso_world_observation(30, 0.0)
    result = pipeline.process(delayed, now_mono_ns=delayed.capture_mono_ns + 1)
    assert not {"torso", "left_arm", "right_arm"} & result.held_groups


def test_a_genuine_blind_interval_still_holds_with_a_witness_attached() -> None:
    """The witness must not become a way to skip the gap check."""
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(torso_world_observation(1, 0.0))
    accepted = torso_world_observation(2, 0.0)
    pipeline.observe_source_continuity(accepted)
    pipeline.process(accepted, now_mono_ns=accepted.capture_mono_ns + 1)
    # Nothing usable was detected between seq 2 and seq 60 -- a real dropout,
    # not back-pressure. Identity is unprovable across it.
    reentry = torso_world_observation(60, 0.0)
    pipeline.observe_source_continuity(reentry)
    result = pipeline.process(reentry, now_mono_ns=reentry.capture_mono_ns + 1)
    assert dict(result.held_group_reasons) == {
        "left_arm": "TORSO_YAW_CONTINUITY_GAP",
        "right_arm": "TORSO_YAW_CONTINUITY_GAP",
        "torso": "TORSO_YAW_CONTINUITY_GAP",
    }


def test_a_blind_interval_is_enforced_even_when_its_reentry_frame_is_dropped() -> None:
    """The break is a remembered mark, not a skipped sample.

    The naive form of this witness -- advancing the anchor from the capture loop
    -- is unsafe precisely here: the frame that ended the dropout gets discarded
    by the worker, and the next frame is a smooth 33 ms later, so the gap
    silently disappears.
    """
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(torso_world_observation(1, 0.0))
    accepted = torso_world_observation(2, 0.0)
    pipeline.observe_source_continuity(accepted)
    pipeline.process(accepted, now_mono_ns=accepted.capture_mono_ns + 1)
    # Re-entry after a long dropout, dropped by the worker before process().
    pipeline.observe_source_continuity(torso_world_observation(60, 0.0))
    # The next frame is smoothly 33 ms after re-entry, and it is the one that
    # reaches the worker. The break must still be enforced on it.
    following = torso_world_observation(61, 0.0)
    pipeline.observe_source_continuity(following)
    result = pipeline.process(following, now_mono_ns=following.capture_mono_ns + 1)
    assert dict(result.held_group_reasons) == {
        "left_arm": "TORSO_YAW_CONTINUITY_GAP",
        "right_arm": "TORSO_YAW_CONTINUITY_GAP",
        "torso": "TORSO_YAW_CONTINUITY_GAP",
    }


def test_frozen_content_cannot_vouch_for_a_blind_interval() -> None:
    """A frozen buffer yields perfect shoulders forever.

    This is the one way a capture-cadence witness could fail open: the freeze is
    exactly the interval during which an operator could turn 180 deg unobserved,
    so the witness carries its own liveness monitor and non-live frames never
    advance it.
    """
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=NullCommandSink(),
        liveness_monitor=FrameLivenessMonitor(
            LivenessPolicy(max_static_ns=100_000_000)
        ),
    )
    pipeline.calibrate(torso_world_observation(1, 0.0))
    accepted = torso_world_observation(2, 0.0)
    pipeline.observe_source_continuity(accepted)
    pipeline.process(accepted, now_mono_ns=accepted.capture_mono_ns + 1)
    # The camera froze: usable shoulders on every frame, identical pixels.
    for sequence in range(3, 60):
        frozen = torso_world_observation(sequence, 0.0).model_copy(
            update={"content_fingerprint": "frozen-buffer"}
        )
        pipeline.observe_source_continuity(frozen)
    reentry = torso_world_observation(60, 0.0)
    pipeline.observe_source_continuity(reentry)
    result = pipeline.process(reentry, now_mono_ns=reentry.capture_mono_ns + 1)
    assert dict(result.held_group_reasons) == {
        "left_arm": "TORSO_YAW_CONTINUITY_GAP",
        "right_arm": "TORSO_YAW_CONTINUITY_GAP",
        "torso": "TORSO_YAW_CONTINUITY_GAP",
    }


def test_a_post_gap_persistent_shoulder_swap_never_reenters_without_calibration() -> None:
    """Two smooth swapped samples establish smoothness, not semantic identity."""
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(torso_world_observation(1, 0.0))
    accepted = torso_world_observation(2, 0.0)
    pipeline.process(accepted, now_mono_ns=accepted.capture_mono_ns + 1)
    # A true back-facing body with swapped labels can look near-neutral. Its
    # first reappearance is after the continuity horizon, so it cannot command.
    def _back_facing(sequence: int):
        # A body turned 179 deg presents no face mesh to the camera. Modelling
        # that is what makes the swap unprovable: front evidence is precisely
        # the observable a shoulder-label swap cannot fake.
        obs = torso_world_observation(sequence, math.radians(179.0), swap_shoulders=True)
        return obs.model_copy(
            update={
                "landmarks": tuple(
                    m for m in obs.landmarks if not m.name.startswith("face_")
                )
            }
        )

    first = _back_facing(30)
    held = pipeline.process(first, now_mono_ns=first.capture_mono_ns + 1)
    assert dict(held.held_group_reasons)["torso"] == "TORSO_YAW_CONTINUITY_GAP"
    second = _back_facing(31)
    locked = pipeline.process(second, now_mono_ns=second.capture_mono_ns + 1)
    assert dict(locked.held_group_reasons)["torso"] == "TORSO_YAW_RECALIBRATION_REQUIRED"


@pytest.mark.parametrize("interval_ms", (82, 100, 124, 150))
def test_torso_and_arms_track_inside_the_measured_continuity_budget(
    interval_ms: int,
) -> None:
    """Normal retained camera cadence must not be misread as an identity gap."""
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    start = 1_000_000_000
    calibration = torso_world_observation(1, 0.0, capture_mono_ns=start)
    pipeline.calibrate(calibration)
    for sequence in range(2, 10):
        timestamp = start + (sequence - 1) * interval_ms * 1_000_000
        observation = torso_world_observation(
            sequence,
            math.radians(8.0),
            capture_mono_ns=timestamp,
        )
        result = pipeline.process(observation, now_mono_ns=timestamp + 1)
        assert not {"torso", "left_arm", "right_arm"} & result.held_groups


def test_head_fallback_basis_switch_requires_two_consistent_frames() -> None:
    """The face mesh and pose fallback never silently share a filter history."""
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())

    def observation(sequence: int, *, face_mesh: bool):
        timestamp = 1_000_000_000 + sequence * 33_333_333
        base = arm_observation(
            sequence=sequence,
            capture_mono_ns=timestamp,
            inference_complete_mono_ns=timestamp,
        )
        landmarks = base.landmarks + _pose_head_landmarks()
        if not face_mesh:
            landmarks = tuple(
                landmark for landmark in landmarks if not landmark.name.startswith("face_")
            )
        return base.model_copy(update={"landmarks": landmarks})

    neutral = observation(1, face_mesh=True)
    pipeline.calibrate(neutral)
    mesh = observation(2, face_mesh=True)
    assert "head" not in pipeline.process(
        mesh, now_mono_ns=mesh.capture_mono_ns + 1
    ).held_groups
    first_pose = observation(3, face_mesh=False)
    transition = pipeline.process(
        first_pose, now_mono_ns=first_pose.capture_mono_ns + 1
    )
    assert dict(transition.held_group_reasons)["head"] == "HEAD_BASIS_TRANSITION"
    second_pose = observation(4, face_mesh=False)
    resumed = pipeline.process(
        second_pose, now_mono_ns=second_pose.capture_mono_ns + 1
    )
    assert "head" not in resumed.held_groups


def test_head_policy_rate_caps_the_emitted_joint_not_just_the_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Head mapper calls are stateless in live mode; the governor is binding."""
    from galbot_motion_studio.retarget.head import HeadMappingResult, HeadTarget

    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(NEUTRAL)
    monkeypatch.setattr(
        pipeline.head,
        "map",
        lambda *_args, **_kwargs: HeadMappingResult(HeadTarget(0.8, 0.0)),
    )
    previous = SIM_TELEOP_START_QPOS["head_joint1"]
    previous_timestamp_ns = NEUTRAL.capture_mono_ns
    for sequence in range(2, 34):
        observation = moving_observation(sequence, step=sequence)
        result = pipeline.process(observation, now_mono_ns=observation.capture_mono_ns + 1)
        assert result.decision.outcome is SafetyOutcome.ALLOW
        current = commanded(result)["head_joint1"]
        elapsed_s = (observation.capture_mono_ns - previous_timestamp_ns) / 1_000_000_000
        assert abs((current - previous) / elapsed_s) <= (
            pipeline.head.policy.max_rate_rad_s + 1e-7
        )
        previous = current
        previous_timestamp_ns = observation.capture_mono_ns


def test_head_soft_limit_is_explicitly_reported_without_becoming_a_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from galbot_motion_studio.retarget.head import HeadMappingResult, HeadTarget

    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(NEUTRAL)
    monkeypatch.setattr(
        pipeline.head,
        "map",
        lambda *_args, **_kwargs: HeadMappingResult(HeadTarget(0.8, 0.1), saturated=True),
    )
    observation = moving_observation(2, step=2)
    result = pipeline.process(observation, now_mono_ns=observation.capture_mono_ns + 1)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert "head" not in result.held_groups
    assert result.saturated_groups == {"head"}
    assert result.saturated_group_reasons == (("head", "HEAD_SOFT_LIMIT"),)


def test_torso_holds_rather_than_saturating_when_the_camera_view_is_ambiguous() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(torso_world_observation(1, 0.0))
    # This intentionally jumps directly into the ambiguous band.  The semantic
    # step guard must win over the normal out-of-view hold: a real 76-degree
    # turn reaches the boundary through <=20-degree samples (covered above).
    observation = torso_world_observation(2, math.radians(76.0))
    result = pipeline.process(observation, now_mono_ns=observation.capture_mono_ns + 1)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    # An arm offset is interpreted in the human torso frame.  Once the camera
    # cannot determine that frame, both arms must hold too rather than moving in
    # the robot's stale torso orientation.
    assert result.held_groups == {"left_arm", "right_arm", "torso"}
    assert result.held_group_reasons == (
        ("left_arm", "TORSO_YAW_DISCONTINUITY"),
        ("right_arm", "TORSO_YAW_DISCONTINUITY"),
        ("torso", "TORSO_YAW_DISCONTINUITY"),
    )
    # This neutral arm fixture has no hand landmarks, so both grippers are held
    # for their *own* unavailable inputs -- not spuriously by the torso yaw.
    assert result.held_grippers == {"left_gripper_joint", "right_gripper_joint"}
    assert result.held_gripper_reasons == (
        ("left_gripper_joint", "HAND_NOT_TRACKABLE"),
        ("right_gripper_joint", "HAND_NOT_TRACKABLE"),
    )
    joints = commanded(result)
    assert joints["leg_joint4"] == pytest.approx(
        SIM_TELEOP_START_QPOS["leg_joint4"], abs=1e-12
    )
    for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
        assert joints[name] == pytest.approx(SIM_TELEOP_START_QPOS[name], abs=1e-12)


def run_frames(observations) -> list:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=NullCommandSink())
    pipeline.calibrate(NEUTRAL)
    return [
        pipeline.process(observation, now_mono_ns=observation.capture_mono_ns + 1)
        for observation in observations
    ]


def test_a_dipping_wrist_holds_only_its_arm_and_the_rest_still_commands() -> None:
    tracked = run_frames([moving_observation(2)])[0]
    degraded = run_frames([moving_observation(2, left_wrist_confidence=0.2)])[0]

    # The frame that used to stop the whole robot now still produces a command.
    assert tracked.decision.outcome is SafetyOutcome.ALLOW
    assert degraded.decision.outcome is SafetyOutcome.ALLOW
    # This fixture deliberately has no pose-world coordinates.  Arms/head still
    # track, but the pipeline must report the uncalibrated torso rather than
    # presenting a static leg_joint4 as live yaw tracking.
    assert tracked.held_groups == {"torso"}
    assert degraded.held_groups == {"left_arm", "torso"}
    assert degraded.held_group_reasons == (
        ("left_arm", "LOW_CONFIDENCE"),
        ("torso", "TORSO_YAW_NEUTRAL_UNAVAILABLE"),
    )

    tracked_joints = commanded(tracked)
    degraded_joints = commanded(degraded)
    # The head and the perfectly tracked right arm are bit-for-bit unaffected by
    # the other arm being held.
    # The head is fully decoupled -- it shares no clearance geometry decision with
    # an arm, so it is bit-for-bit identical.
    for name in HEAD_JOINTS:
        assert degraded_joints[name] == tracked_joints[name]
    # The tracked arm is decoupled in COMMAND, which is the property that matters:
    # a held limb never redirects a tracked one. It is not bit-identical, because
    # self-clearance is a genuine whole-body question -- the two arms can collide,
    # so where the held arm is parked changes what the swept check sees along the
    # tracked arm's path. Measured 2026-08-26 at 4.4e-4 rad (0.025 deg), i.e. far
    # below the 0.10 rad step cap and invisible in the twin; it appeared when the
    # look-ahead horizon was sized past the governor's braking curve so the check
    # reaches as far as the arm can actually travel.
    for name in RIGHT_ARM_JOINTS:
        assert degraded_joints[name] == pytest.approx(tracked_joints[name], abs=5e-3)
    # ... and they really did move, so "unaffected" is not "everything froze".
    assert any(
        degraded_joints[name] != SIM_TELEOP_START_QPOS[name] for name in RIGHT_ARM_JOINTS
    )
    # The held arm keeps its last commanded pose instead of following the wrist.
    for name in LEFT_ARM_JOINTS:
        assert degraded_joints[name] == pytest.approx(SIM_TELEOP_START_QPOS[name], abs=1e-12)
    assert any(
        tracked_joints[name] != degraded_joints[name] for name in LEFT_ARM_JOINTS
    )
    # The whole-body pose is still commanded and still clearance-checked.
    assert set(degraded_joints) == set(tracked_joints)
    assert degraded.target.predicted_clearance_m > 0


def test_ik_nonconvergence_holds_only_its_arm_and_the_rest_still_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable left-arm solution must not freeze healthy groups.

    The retained live trial exposed long runs where a fully observed wrist was
    held because the local IK accuracy budget was not met.  Holding that arm at
    its governed pose is safe; the head and opposite arm still have trustworthy
    inputs and the complete commanded pose remains supervisor-checked.
    """
    from galbot_motion_studio.retarget.left_arm import (
        ArmMapRejection,
        IKSolveAttempt,
        IKSolveStage,
        LeftArmMappingResult,
    )

    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    pipeline.calibrate(NEUTRAL)
    monkeypatch.setattr(
        pipeline.left_arm,
        "map",
        lambda *_args, **_kwargs: LeftArmMappingResult(
            None,
            ArmMapRejection.IK_DID_NOT_CONVERGE,
            best_wrist_residual_m=0.021,
            ik_attempts=(
                IKSolveAttempt(IKSolveStage.NEUTRAL, 0.031),
                IKSolveAttempt(IKSolveStage.POSITION_PRIORITY, 0.021),
            ),
        ),
    )

    frame = moving_observation(2)
    result = pipeline.process(frame, now_mono_ns=frame.capture_mono_ns + 1)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert result.held_groups == {"left_arm", "torso"}
    assert result.held_group_reasons == (
        ("left_arm", "IK_DID_NOT_CONVERGE"),
        ("torso", "TORSO_YAW_NEUTRAL_UNAVAILABLE"),
    )
    assert result.held_group_residuals_m == (("left_arm", 0.021),)
    assert result.held_group_ik_attempts == (
        (
            "left_arm",
            (("neutral", 0.031), ("position_priority", 0.021)),
        ),
    )
    joints = commanded(result)
    assert any(
        joints[name] != SIM_TELEOP_START_QPOS[name] for name in HEAD_JOINTS + RIGHT_ARM_JOINTS
    )
    for name in LEFT_ARM_JOINTS:
        assert joints[name] == pytest.approx(SIM_TELEOP_START_QPOS[name], abs=1e-12)


def test_a_sustained_dip_no_longer_stops_the_robot_for_seconds() -> None:
    """27 consecutive dipping frames used to be 27 consecutive whole-robot HOLDs."""
    results = run_frames(
        [
            moving_observation(sequence, left_wrist_confidence=0.2, step=sequence)
            for sequence in range(2, 29)
        ]
    )
    assert [result.decision.outcome for result in results] == [SafetyOutcome.ALLOW] * 27
    assert all(result.held_groups == {"left_arm", "torso"} for result in results)
    # The held arm comes to rest and stays there; it does not creep.
    first, last = commanded(results[0]), commanded(results[-1])
    for name in LEFT_ARM_JOINTS:
        assert last[name] == pytest.approx(first[name], abs=1e-12)
    # The right arm tracked throughout.
    assert any(last[name] != first[name] for name in RIGHT_ARM_JOINTS)


def test_a_held_arm_resumes_from_where_it_stopped() -> None:
    results = run_frames(
        [
            moving_observation(2, step=1),
            moving_observation(3, left_wrist_confidence=0.2, step=2),
            moving_observation(4, step=3),
        ]
    )
    assert [result.held_groups for result in results] == [
        {"torso"},
        {"left_arm", "torso"},
        {"torso"},
    ]
    assert all(result.decision.outcome is SafetyOutcome.ALLOW for result in results)
    assert all(result.receipt is not None for result in results)


def test_every_group_low_confidence_is_the_original_whole_robot_hold() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    pipeline.calibrate(NEUTRAL)
    dim = moving_observation(2)
    dim = dim.model_copy(
        update={
            "aggregate_confidence": 0.1,
            "landmarks": tuple(
                landmark.model_copy(update={"visibility": 0.1, "presence": 0.1})
                for landmark in dim.landmarks
            ),
        }
    )
    result = pipeline.process(dim, now_mono_ns=dim.capture_mono_ns + 1)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert result.target is None
    assert result.receipt is None
    assert result.held_groups == {"head", "left_arm", "right_arm", "torso"}
    assert dict(result.held_group_reasons) == {
        "head": "LOW_CONFIDENCE",
        "torso": "LOW_CONFIDENCE",
        "left_arm": "LOW_CONFIDENCE",
        "right_arm": "LOW_CONFIDENCE",
    }
    assert "LOW_CONFIDENCE" in " ".join(result.decision.reasons)
    assert sink.submissions == []


def test_a_stale_frame_still_holds_every_group_however_clear_it_is() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    pipeline.calibrate(NEUTRAL)
    fresh = moving_observation(2)
    result = pipeline.process(fresh, now_mono_ns=fresh.capture_mono_ns + 10_000_000_000)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert "STALE" in " ".join(result.decision.reasons)
    assert result.held_groups == {"head", "left_arm", "right_arm", "torso"}
    assert sink.submissions == []


def test_an_unstable_identity_still_holds_every_group() -> None:
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock", sink=sink)
    pipeline.calibrate(NEUTRAL)
    from galbot_motion_studio.contracts.human import IdentityState

    switched = moving_observation(2).model_copy(
        update={"identity": IdentityState.SWITCHED}
    )
    result = pipeline.process(switched, now_mono_ns=switched.capture_mono_ns + 1)
    assert result.decision.outcome is SafetyOutcome.HOLD
    assert "IDENTITY_NOT_STABLE" in " ".join(result.decision.reasons)
    assert result.held_groups == {"head", "left_arm", "right_arm", "torso"}
    assert sink.submissions == []


def test_a_vanished_wrist_holds_only_its_arm_and_never_defaults_to_allowed() -> None:
    without_left_wrist = moving_observation(2).model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in moving_observation(2).landmarks
                if landmark.name != "pose_15"
            ),
            # The adapter declares zero when a command-critical landmark is gone.
            "aggregate_confidence": 0.0,
        }
    )
    result = run_frames([without_left_wrist])[0]
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert result.held_groups == {"left_arm", "torso"}
    joints = commanded(result)
    for name in LEFT_ARM_JOINTS:
        assert joints[name] == pytest.approx(SIM_TELEOP_START_QPOS[name], abs=1e-12)


def test_a_vanished_shoulder_holds_both_arms_and_leaves_the_head_tracking() -> None:
    frame = moving_observation(2)
    result = run_frames(
        [
            frame.model_copy(
                update={
                    "landmarks": tuple(
                        landmark
                        for landmark in frame.landmarks
                        if landmark.name != "pose_11"
                    ),
                    "aggregate_confidence": 0.0,
                }
            )
        ]
    )[0]
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert result.held_groups == {"left_arm", "right_arm", "torso"}
    joints = commanded(result)
    for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
        assert joints[name] == pytest.approx(SIM_TELEOP_START_QPOS[name], abs=1e-12)


def test_direction_vector_pipeline_holds_each_arm_without_a_torso_basis() -> None:
    """Candidate analysis may not quietly fall back to camera-axis directions."""
    sink = NullCommandSink()
    pipeline = MotionStudioPipeline(
        source_clock_id="camera-clock",
        sink=sink,
        arm_mapping="direction-vector",
    )
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    pipeline.calibrate(neutral)
    frame = neutral.model_copy(
        update={
            "sequence": 2,
            "capture_mono_ns": 1_033_333_333,
            "inference_complete_mono_ns": 1_033_333_333,
        }
    )
    result = pipeline.process(frame, now_mono_ns=1_033_333_334)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    assert result.target is not None
    assert result.target.mapping_hash == mapping_hash_for("direction-vector")
    assert result.held_groups == {"left_arm", "right_arm", "torso"}
    assert dict(result.held_group_reasons) == {
        "left_arm": "INCOMPLETE_OBSERVATION",
        "right_arm": "INCOMPLETE_OBSERVATION",
        "torso": "TORSO_YAW_NEUTRAL_UNAVAILABLE",
    }


def test_arm_animates_out_of_rest_without_snapping() -> None:
    # Regression for the start-pose fix: the twin begins at the commandable
    # Motion-Studio rest (arms by the sides) and animates out of it, rather than
    # snapping to the mid-range neutral on the first command. Every arm joint's
    # first commanded value must therefore sit within a single governor step of
    # rest -- the old snap jumped ~1.7 rad to the neutral on frame 1.
    pipeline = MotionStudioPipeline(source_clock_id="synthetic-clock")
    pipeline.calibrate(
        synthetic_observation(0, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    )
    first = pipeline.process(
        synthetic_observation(1, timestamp_ns=1_033_333_333, motion_fraction=1.0),
        now_mono_ns=1_033_333_334,
    )
    assert first.decision.outcome is SafetyOutcome.ALLOW
    commanded_first = commanded(first)
    for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
        assert abs(commanded_first[name] - SIM_TELEOP_START_QPOS[name]) <= 0.05
