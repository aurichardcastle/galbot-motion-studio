import pytest

from galbot_motion_studio.contracts.human import HumanObservation, IdentityState, Landmark
from galbot_motion_studio.vision.freshness import (
    COMMAND_CONTROL_GROUPS,
    ControlGroup,
    ControlGroupPolicy,
    FreshnessPolicy,
    ObservationGate,
    ObservationRejection,
)
from galbot_motion_studio.vision.liveness import FrameLivenessMonitor, LivenessPolicy
from galbot_motion_studio.vision.selection import (
    OccupancyObservation,
    OperatorSelectionManager,
    SelectionPolicy,
    SelectionRejection,
    SelectionState,
)

from test_calibration import stable_observation


def observation(**changes: object) -> HumanObservation:
    values: dict[str, object] = {
        "session_id": "camera-session",
        "sequence": 4,
        "source_clock_id": "mac-boot-uuid",
        "source_mono_ns": 1_000,
        "camera_id": "webcam-1",
        "calibration_id": "neutral-v1",
        "capture_mono_ns": 1_000,
        "inference_complete_mono_ns": 1_020,
        "image_width_px": 1280,
        "image_height_px": 720,
        "identity": IdentityState.STABLE,
        "aggregate_confidence": 0.9,
        "landmarks": (
            Landmark(
                name="face_nose",
                normalized_xyz=(0.5, 0.5, 0.0),
                visibility=1.0,
                presence=1.0,
            ),
        ),
    }
    values.update(changes)
    return HumanObservation(**values)


def test_gate_accepts_only_fresh_stable_newer_observations() -> None:
    gate = ObservationGate(
        FreshnessPolicy(
            source_clock_id="mac-boot-uuid", required_landmark_names=frozenset({"face_nose"}), max_age_ns=100
        )
    )
    assert gate.evaluate(observation(), now_mono_ns=1_050).accepted
    result = gate.evaluate(observation(sequence=4), now_mono_ns=1_060)
    assert result.reason is ObservationRejection.REORDERED


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"source_clock_id": "hpu-boot-uuid"}, ObservationRejection.CLOCK_MISMATCH),
        ({"capture_mono_ns": 900, "inference_complete_mono_ns": 920}, ObservationRejection.STALE),
        (
            {"capture_mono_ns": 1_100, "inference_complete_mono_ns": 1_120},
            ObservationRejection.FUTURE_CAPTURE,
        ),
        ({"inference_complete_mono_ns": 1_100}, ObservationRejection.FUTURE_INFERENCE),
        ({"aggregate_confidence": 0.2}, ObservationRejection.LOW_CONFIDENCE),
        ({"identity": IdentityState.SWITCHED}, ObservationRejection.IDENTITY_NOT_STABLE),
        ({"landmarks": ()}, ObservationRejection.MISSING_REQUIRED_LANDMARK),
    ],
)
def test_gate_rejects_each_unsafe_input(changes: dict[str, object], expected: ObservationRejection) -> None:
    gate = ObservationGate(
        FreshnessPolicy(
            source_clock_id="mac-boot-uuid", required_landmark_names=frozenset({"face_nose"}), max_age_ns=100
        )
    )
    result = gate.evaluate(observation(**changes), now_mono_ns=1_050)
    assert not result.accepted
    assert result.reason is expected


def test_image_reference_requires_explicit_recording() -> None:
    with pytest.raises(ValueError, match="recording"):
        observation(persisted_image_ref="video/frame-1.jpg")


def test_liveness_monitor_holds_frozen_frames_even_when_they_are_fresh_and_ordered() -> None:
    gate = ObservationGate(
        FreshnessPolicy(
            source_clock_id="mac-boot-uuid",
            required_landmark_names=frozenset({"face_nose"}),
            max_age_ns=1_000,
        ),
        liveness_monitor=FrameLivenessMonitor(LivenessPolicy(max_static_ns=20)),
    )
    first = gate.evaluate(
        observation(content_fingerprint="a"), now_mono_ns=1_050
    )
    assert first.reason is ObservationRejection.FRAME_NOT_LIVE
    # A content transition establishes liveness.
    assert gate.evaluate(
        observation(
            sequence=5,
            source_mono_ns=1_010,
            capture_mono_ns=1_010,
            inference_complete_mono_ns=1_020,
            content_fingerprint="b",
        ),
        now_mono_ns=1_050,
    ).accepted
    # Replayed content with a new timestamp is fresh but no longer live.
    frozen = gate.evaluate(
        observation(
            sequence=6,
            source_mono_ns=1_040,
            capture_mono_ns=1_040,
            inference_complete_mono_ns=1_045,
            content_fingerprint="b",
        ),
        now_mono_ns=1_050,
    )
    assert frozen.reason is ObservationRejection.FRAME_NOT_LIVE
    assert not frozen.any_group_accepted


def test_source_reset_clears_liveness_and_ordering_history() -> None:
    gate = ObservationGate(
        FreshnessPolicy(
            source_clock_id="mac-boot-uuid",
            required_landmark_names=frozenset({"face_nose"}),
            max_age_ns=1_000,
        ),
        liveness_monitor=FrameLivenessMonitor(LivenessPolicy(max_static_ns=20)),
    )
    gate.evaluate(observation(content_fingerprint="a"), now_mono_ns=1_050)
    assert gate.evaluate(
        observation(
            sequence=5,
            source_mono_ns=1_010,
            capture_mono_ns=1_010,
            inference_complete_mono_ns=1_020,
            content_fingerprint="b",
        ),
        now_mono_ns=1_050,
    ).accepted
    gate.reset_source()
    restarted = gate.evaluate(
        observation(sequence=1, content_fingerprint="a"), now_mono_ns=1_050
    )
    assert restarted.reason is ObservationRejection.FRAME_NOT_LIVE


def test_selection_reset_clears_liveness_but_not_stream_ordering() -> None:
    gate = ObservationGate(
        FreshnessPolicy(
            source_clock_id="mac-boot-uuid",
            required_landmark_names=frozenset({"face_nose"}),
            max_age_ns=1_000,
        ),
        liveness_monitor=FrameLivenessMonitor(LivenessPolicy(max_static_ns=20)),
    )
    gate.evaluate(observation(content_fingerprint="a"), now_mono_ns=1_050)
    assert gate.evaluate(
        observation(
            sequence=5,
            source_mono_ns=1_010,
            capture_mono_ns=1_010,
            inference_complete_mono_ns=1_020,
            content_fingerprint="b",
        ),
        now_mono_ns=1_050,
    ).accepted

    gate.reset_liveness()
    duplicate = gate.evaluate(
        observation(
            sequence=5,
            source_mono_ns=1_040,
            capture_mono_ns=1_040,
            inference_complete_mono_ns=1_045,
            content_fingerprint="c",
        ),
        now_mono_ns=1_050,
    )
    assert duplicate.reason is ObservationRejection.REORDERED


def test_liveness_precedes_selection_and_invalidates_it_on_frozen_source() -> None:
    selection_policy = SelectionPolicy(
        max_timestamp_skew_ns=1,
        max_evidence_age_ns=100,
        max_selection_grace_ns=50,
        max_cumulative_occlusion_ns=100,
        operator_zone_normalized_bounds=(0.1, 0.1, 0.9, 0.9),
        max_candidate_center_delta_normalized=0.1,
        max_shoulder_width_relative_delta=0.1,
        max_eye_span_relative_delta=0.1,
    )
    gate = ObservationGate(
        FreshnessPolicy(
            source_clock_id="camera-clock",
            required_landmark_names=frozenset({"face_1"}),
            max_age_ns=100,
        ),
        liveness_monitor=FrameLivenessMonitor(LivenessPolicy(max_static_ns=20)),
        selection_manager=OperatorSelectionManager(selection_policy),
    )

    def selected_observation(sequence: int, timestamp_ns: int, fingerprint: str) -> HumanObservation:
        return stable_observation(
            sequence=sequence,
            source_mono_ns=timestamp_ns,
            capture_mono_ns=timestamp_ns,
            inference_complete_mono_ns=timestamp_ns,
            content_fingerprint=fingerprint,
        )

    def occupancy_for(observation: HumanObservation) -> OccupancyObservation:
        return OccupancyObservation(
            camera_id=observation.camera_id,
            source_clock_id=observation.source_clock_id,
            source_sequence=observation.sequence,
            capture_mono_ns=observation.capture_mono_ns,
            inference_complete_mono_ns=observation.inference_complete_mono_ns,
            content_fingerprint=observation.content_fingerprint or "",
            candidate_count=1,
            candidate_center_normalized_xy=(0.5, 0.5),
        )

    # Source liveness must establish a content transition before acquisition.
    first = selected_observation(1, 1_000, "a")
    assert gate.evaluate(first, now_mono_ns=1_001, occupancy=occupancy_for(first)).reason is ObservationRejection.FRAME_NOT_LIVE
    transitioned = selected_observation(2, 1_020, "b")
    assert gate.evaluate(
        transitioned, now_mono_ns=1_021, occupancy=occupancy_for(transitioned)
    ).reason is ObservationRejection.SELECTION_NOT_LOCKED

    acquired_frame = selected_observation(3, 1_040, "c")
    acquired = gate.acquire_selection(
        occupancy_for(acquired_frame), acquired_frame, now_mono_ns=1_041
    )
    assert acquired.accepted and acquired.selection_id == 1

    frozen = selected_observation(4, 1_070, "c")
    result = gate.evaluate(frozen, now_mono_ns=1_071, occupancy=occupancy_for(frozen))
    assert result.reason is ObservationRejection.FRAME_NOT_LIVE
    assert result.selection is not None
    assert result.selection.reason is SelectionRejection.OBSERVATION_UNTRUSTED
    assert gate.selection_manager.state is SelectionState.OCCLUDED

    # A continuing frozen source exhausts the selection budget and requires a
    # new acquisition. The first liveness hold, though, is only a bounded gap.
    still_frozen = selected_observation(5, 1_130, "c")
    assert gate.evaluate(
        still_frozen, now_mono_ns=1_131, occupancy=occupancy_for(still_frozen)
    ).reason is ObservationRejection.FRAME_NOT_LIVE
    assert gate.selection_manager.state is SelectionState.LOST

    recovered = selected_observation(6, 1_140, "d")
    assert gate.evaluate(
        recovered, now_mono_ns=1_141, occupancy=occupancy_for(recovered)
    ).reason is ObservationRejection.SELECTION_NOT_LOCKED


def test_gate_allows_a_fresh_sequence_gap() -> None:
    gate = ObservationGate(
        FreshnessPolicy(
            source_clock_id="mac-boot-uuid",
            required_landmark_names=frozenset({"face_nose"}),
            max_age_ns=100,
        )
    )
    assert gate.evaluate(observation(sequence=4), now_mono_ns=1_050).accepted
    assert gate.evaluate(
        observation(
            sequence=9,
            source_mono_ns=1_030,
            capture_mono_ns=1_030,
            inference_complete_mono_ns=1_040,
        ),
        now_mono_ns=1_060,
    ).accepted


def test_nonfinite_landmark_is_rejected_at_the_contract_boundary() -> None:
    with pytest.raises(ValueError, match="finite"):
        Landmark(
            name="face_nose",
            normalized_xyz=(float("nan"), 0.5, 0.0),
            visibility=1.0,
            presence=1.0,
        )


# --------------------------------------------------------------------------
# Per-control-group confidence and landmark gating.
#
# The operator's recorded 372-frame session held on `observation: LOW_CONFIDENCE`
# for runs of up to 27 consecutive frames (2.1 s at 12.8 fps) because a single
# dipping wrist dragged the frame-wide minimum under the bar and stopped the
# head and both arms. Confidence and landmark presence are now judged per group.
# Everything below pins the safety half of that: no threshold moved, every
# whole-frame rejection still stops the whole robot, and an unevaluable group
# always holds.
# --------------------------------------------------------------------------

COMMAND_LANDMARKS = (
    "face_1", "face_33", "face_263",
    "pose_11", "pose_12", "pose_13", "pose_14", "pose_15", "pose_16",
)
#: Mirrors adapters.mediapipe_holistic.COMMAND_CRITICAL_LANDMARK_NAMES; elbows
#: are excluded there because they are a soft IK hint.
CRITICAL_LANDMARKS = frozenset(COMMAND_LANDMARKS) - {"pose_13", "pose_14"}
#: Derived from the enum on purpose: a hand-listed set silently stops covering
#: a newly added control group, and these are the whole-frame-rejection tests.
ALL_GROUPS = frozenset(str(group) for group in ControlGroup)


def tracked_observation(
    *,
    confidences: dict[str, float] | None = None,
    drop: frozenset[str] = frozenset(),
    **changes: object,
) -> HumanObservation:
    """A fully tracked nine-landmark frame with per-landmark overrides.

    ``aggregate_confidence`` is derived exactly the way the MediaPipe adapter
    derives it, so these fixtures cannot drift into being more forgiving than a
    real observation; individual tests override it deliberately.
    """
    levels = confidences or {}
    landmarks = tuple(
        Landmark(
            name=name,
            normalized_xyz=(0.5, 0.5, 0.0),
            visibility=levels.get(name, 1.0),
            presence=levels.get(name, 1.0),
        )
        for name in COMMAND_LANDMARKS
        if name not in drop
    )
    present = {landmark.name for landmark in landmarks}
    aggregate = (
        0.0
        if not CRITICAL_LANDMARKS <= present
        else min(levels.get(name, 1.0) for name in CRITICAL_LANDMARKS)
    )
    values: dict[str, object] = {
        "session_id": "camera-session",
        "sequence": 4,
        "source_clock_id": "mac-boot-uuid",
        "source_mono_ns": 1_000,
        "camera_id": "webcam-1",
        "calibration_id": "neutral-v1",
        "capture_mono_ns": 1_000,
        "inference_complete_mono_ns": 1_020,
        "image_width_px": 1280,
        "image_height_px": 720,
        "identity": IdentityState.STABLE,
        "aggregate_confidence": aggregate,
        "landmarks": landmarks,
    }
    values.update(changes)
    return HumanObservation(**values)


def command_gate(**policy_changes: object) -> ObservationGate:
    return ObservationGate(
        FreshnessPolicy(
            source_clock_id="mac-boot-uuid",
            required_landmark_names=frozenset(COMMAND_LANDMARKS),
            max_age_ns=100,
            **policy_changes,  # type: ignore[arg-type]
        ),
        control_groups=COMMAND_CONTROL_GROUPS,
    )


def test_one_dipping_wrist_holds_only_its_own_arm() -> None:
    """The measured failure: a left wrist dips and the whole robot stops."""
    result = command_gate().evaluate(
        tracked_observation(confidences={"pose_15": 0.2}), now_mono_ns=1_050
    )
    assert result.accepted_groups == {"head", "right_arm", "torso"}
    assert result.held_groups == {"left_arm"}
    assert result.group_reason(ControlGroup.LEFT_ARM) is ObservationRejection.LOW_CONFIDENCE
    assert result.any_group_accepted
    # The legacy single-verdict API keeps its original whole-frame meaning, so a
    # caller that reads only these two fields is no more permissive than before.
    assert not result.accepted
    assert result.reason is ObservationRejection.LOW_CONFIDENCE


def test_a_dipping_shoulder_holds_both_arms_because_both_are_scaled_by_it() -> None:
    """Shoulder span normalises BOTH arms, so pose_11 is not the left arm's alone."""
    result = command_gate().evaluate(
        tracked_observation(confidences={"pose_11": 0.2}), now_mono_ns=1_050
    )
    assert result.accepted_groups == {"head"}
    assert result.held_groups == {"left_arm", "right_arm", "torso"}


def test_a_dipping_elbow_holds_nothing_because_it_is_a_soft_ik_hint() -> None:
    result = command_gate().evaluate(
        tracked_observation(confidences={"pose_13": 0.2}), now_mono_ns=1_050
    )
    assert result.accepted
    assert result.held_groups == frozenset()


def test_every_group_low_confidence_is_the_original_global_hold() -> None:
    result = command_gate().evaluate(
        tracked_observation(confidences={name: 0.2 for name in COMMAND_LANDMARKS}),
        now_mono_ns=1_050,
    )
    assert not result.accepted
    assert not result.any_group_accepted
    assert result.reason is ObservationRejection.LOW_CONFIDENCE
    assert result.held_groups == ALL_GROUPS


def test_thresholds_are_unchanged_by_the_split() -> None:
    """Per-group is more precise, not more permissive: the same bar, applied narrowly."""
    policy = FreshnessPolicy(
        source_clock_id="mac-boot-uuid",
        required_landmark_names=frozenset(COMMAND_LANDMARKS),
    )
    assert policy.min_confidence == 0.5
    gate = command_gate()
    assert not gate.evaluate(
        tracked_observation(confidences={"pose_15": 0.49}), now_mono_ns=1_050
    ).group_accepted(ControlGroup.LEFT_ARM)
    assert command_gate().evaluate(
        tracked_observation(confidences={"pose_15": 0.5}), now_mono_ns=1_050
    ).group_accepted(ControlGroup.LEFT_ARM)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"source_clock_id": "hpu-boot-uuid"}, ObservationRejection.CLOCK_MISMATCH),
        ({"capture_mono_ns": 900, "inference_complete_mono_ns": 920}, ObservationRejection.STALE),
        (
            {"capture_mono_ns": 1_100, "inference_complete_mono_ns": 1_120},
            ObservationRejection.FUTURE_CAPTURE,
        ),
        ({"inference_complete_mono_ns": 1_100}, ObservationRejection.FUTURE_INFERENCE),
        ({"identity": IdentityState.SWITCHED}, ObservationRejection.IDENTITY_NOT_STABLE),
        ({"identity": IdentityState.LOST}, ObservationRejection.IDENTITY_NOT_STABLE),
    ],
)
def test_whole_frame_rejections_still_hold_the_entire_robot(
    changes: dict[str, object], expected: ObservationRejection
) -> None:
    """These are properties of the FRAME. Perfect landmarks do not rescue a group."""
    result = command_gate().evaluate(tracked_observation(**changes), now_mono_ns=1_050)
    assert result.reason is expected
    assert not result.accepted
    assert not result.any_group_accepted
    assert result.held_groups == ALL_GROUPS
    assert not result.group_accepted(ControlGroup.HEAD)


def test_a_reordered_frame_still_holds_the_entire_robot() -> None:
    gate = command_gate()
    assert gate.evaluate(tracked_observation(sequence=4), now_mono_ns=1_050).accepted
    replayed = gate.evaluate(tracked_observation(sequence=4), now_mono_ns=1_060)
    assert replayed.reason is ObservationRejection.REORDERED
    assert not replayed.any_group_accepted
    assert replayed.held_groups == ALL_GROUPS


def test_a_frame_acted_on_by_any_group_can_never_be_replayed() -> None:
    """A partially accepted frame produced a command, so its sequence is spent."""
    gate = command_gate()
    partial = gate.evaluate(
        tracked_observation(sequence=4, confidences={"pose_15": 0.2}), now_mono_ns=1_050
    )
    assert partial.any_group_accepted
    replayed = gate.evaluate(tracked_observation(sequence=4), now_mono_ns=1_060)
    assert replayed.reason is ObservationRejection.REORDERED


def test_missing_landmarks_hold_the_groups_that_need_them() -> None:
    """Fail closed: an unevaluable group HOLDS; it is never defaulted to allowed."""
    result = command_gate().evaluate(
        tracked_observation(drop=frozenset({"pose_16"})), now_mono_ns=1_050
    )
    assert result.held_groups == {"right_arm"}
    assert result.group_reason(ControlGroup.RIGHT_ARM) is (
        ObservationRejection.MISSING_REQUIRED_LANDMARK
    )
    assert result.accepted_groups == {"head", "left_arm", "torso"}

    # The elbow is not confidence-critical but it is still required to be there.
    elbow = command_gate().evaluate(
        tracked_observation(drop=frozenset({"pose_13"})), now_mono_ns=1_050
    )
    assert elbow.held_groups == {"left_arm"}
    assert elbow.group_reason(ControlGroup.LEFT_ARM) is (
        ObservationRejection.MISSING_REQUIRED_LANDMARK
    )


def test_no_landmarks_at_all_holds_everything() -> None:
    result = command_gate().evaluate(
        tracked_observation(drop=frozenset(COMMAND_LANDMARKS), landmarks=()),
        now_mono_ns=1_050,
    )
    assert not result.any_group_accepted
    assert result.held_groups == ALL_GROUPS


def test_a_group_is_never_allowed_by_default() -> None:
    """Fail-closed lookup: only an explicit per-group pass authorises a group."""
    result = command_gate().evaluate(
        tracked_observation(identity=IdentityState.LOST), now_mono_ns=1_050
    )
    assert not result.group_accepted("head")
    assert not result.group_accepted("torso-we-do-not-have")
    assert result.group_reason("torso-we-do-not-have") is None
    # A result built without any group detail authorises nothing.
    from galbot_motion_studio.vision.freshness import ObservationGateResult

    bare = ObservationGateResult(False, ObservationRejection.STALE)
    assert not bare.any_group_accepted
    assert not bare.group_accepted(ControlGroup.HEAD)


def test_unattributable_low_confidence_holds_everything() -> None:
    """A declared confidence the landmarks cannot explain is not decomposable.

    ``aggregate_confidence`` is the producer's own minimum over the
    command-critical landmarks. If a producer declares something lower than the
    landmarks it shipped account for, the shortfall belongs to information this
    gate cannot attribute to a body part, so it holds the whole robot exactly as
    it did before per-group gating.
    """
    result = command_gate().evaluate(
        tracked_observation(aggregate_confidence=0.1), now_mono_ns=1_050
    )
    assert result.reason is ObservationRejection.LOW_CONFIDENCE
    assert not result.any_group_accepted
    assert result.held_groups == ALL_GROUPS


def test_an_optimistic_aggregate_cannot_rescue_a_group_its_landmarks_condemn() -> None:
    result = command_gate().evaluate(
        tracked_observation(confidences={"pose_15": 0.1}, aggregate_confidence=1.0),
        now_mono_ns=1_050,
    )
    assert result.held_groups == {"left_arm"}
    assert result.group_reason(ControlGroup.LEFT_ARM) is ObservationRejection.LOW_CONFIDENCE


def test_a_gate_without_declared_groups_behaves_exactly_as_one_whole_robot_group() -> None:
    gate = ObservationGate(
        FreshnessPolicy(
            source_clock_id="mac-boot-uuid",
            required_landmark_names=frozenset({"face_nose"}),
            max_age_ns=100,
        )
    )
    assert gate.evaluate(observation(), now_mono_ns=1_050).accepted
    low = gate.evaluate(observation(sequence=5, aggregate_confidence=0.2), now_mono_ns=1_050)
    assert low.reason is ObservationRejection.LOW_CONFIDENCE
    assert not low.any_group_accepted


def test_a_required_landmark_no_group_claims_is_refused_at_construction() -> None:
    """The one way this could fail OPEN is a landmark nobody enforces."""
    with pytest.raises(ValueError, match="do not cover required landmarks"):
        ObservationGate(
            FreshnessPolicy(
                source_clock_id="mac-boot-uuid",
                required_landmark_names=frozenset(COMMAND_LANDMARKS) | {"pose_23"},
            ),
            control_groups=COMMAND_CONTROL_GROUPS,
        )


def test_a_control_group_cannot_gate_confidence_on_a_landmark_it_does_not_require() -> None:
    with pytest.raises(ValueError, match="confidence-critical"):
        ControlGroupPolicy(
            required=frozenset({"pose_11"}),
            confidence_critical=frozenset({"pose_11", "pose_15"}),
        )


def test_only_confidence_and_landmark_presence_are_ever_per_group() -> None:
    """Pin the safety contract: everything else is a property of the frame.

    A rejection moving out of WHOLE_FRAME_REJECTIONS would mean a bad clock, a
    replayed frame, or an unproven identity could leave part of the robot
    driving. This is the test that has to fail first if that is ever attempted.
    """
    from galbot_motion_studio.vision.freshness import WHOLE_FRAME_REJECTIONS

    per_group = set(ObservationRejection) - WHOLE_FRAME_REJECTIONS
    assert per_group == {
        ObservationRejection.LOW_CONFIDENCE,
        ObservationRejection.MISSING_REQUIRED_LANDMARK,
    }
    # Whenever any group is still driving, the reason the others were held is
    # one of those two. A whole-frame reason can only ever hold all of them.
    partial = command_gate().evaluate(
        tracked_observation(confidences={"pose_15": 0.2}, drop=frozenset({"pose_14"})),
        now_mono_ns=1_050,
    )
    assert partial.any_group_accepted
    assert {
        result.reason for result in partial.groups.values() if not result.accepted
    } == per_group


def test_the_declared_groups_partition_the_pipeline_landmark_contract() -> None:
    """Group membership is derived from the retargeters, not invented here."""
    from galbot_motion_studio.adapters.mediapipe_holistic import (
        COMMAND_CRITICAL_LANDMARK_NAMES,
    )
    from galbot_motion_studio.pipeline import MotionStudioPipeline

    required = frozenset().union(
        *(spec.required for spec in COMMAND_CONTROL_GROUPS.values())
    )
    critical = frozenset().union(
        *(spec.confidence_critical for spec in COMMAND_CONTROL_GROUPS.values())
    )
    assert required == MotionStudioPipeline.required_landmarks
    # Equality here is what makes aggregate_confidence exactly decomposable.
    assert critical == COMMAND_CRITICAL_LANDMARK_NAMES


def _with_pose_head_basis(observation, confidence: float = 0.9):
    """Drop the face mesh and supply the pose model's nose and eyes instead."""
    return observation.model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in observation.landmarks
                if not landmark.name.startswith("face_")
            )
            + tuple(
                Landmark(
                    name=name,
                    normalized_xyz=(0.5, 0.5, 0.0),
                    visibility=confidence,
                    presence=confidence,
                )
                for name in ("pose_0", "pose_2", "pose_5")
            )
        }
    )


def test_the_head_falls_back_to_the_pose_model_when_the_face_mesh_is_absent() -> None:
    """Measured: the mesh was missing on 20.4% of frames of the 2026-08-22 trial.

    Both bases produce a valid head pose against their own calibrated neutral, so
    requiring only the less reliable one froze the head for no safety benefit.
    """
    result = command_gate().evaluate(
        _with_pose_head_basis(tracked_observation()), now_mono_ns=1_050
    )
    assert "head" in result.accepted_groups, "the head held despite a usable basis"


def test_the_head_still_holds_when_neither_basis_is_present() -> None:
    observation = tracked_observation()
    neither = observation.model_copy(
        update={
            "landmarks": tuple(
                landmark
                for landmark in observation.landmarks
                if not landmark.name.startswith("face_")
            )
        }
    )
    result = command_gate().evaluate(neither, now_mono_ns=1_050)
    assert "head" in result.held_groups
    assert (
        result.groups["head"].reason is ObservationRejection.MISSING_REQUIRED_LANDMARK
    )


def test_a_low_confidence_fallback_basis_is_judged_by_the_same_bar() -> None:
    """Alternatives must not be a way in for landmarks that would fail on merit."""
    result = command_gate().evaluate(
        _with_pose_head_basis(tracked_observation(), confidence=0.2), now_mono_ns=1_050
    )
    assert "head" in result.held_groups
    assert result.groups["head"].reason is ObservationRejection.LOW_CONFIDENCE


def test_an_alternative_must_declare_a_subset_for_confidence() -> None:
    try:
        ControlGroupPolicy(
            required=frozenset({"a"}),
            confidence_critical=frozenset({"a"}),
            alternatives=((frozenset({"b"}), frozenset({"c"})),),
        )
    except ValueError as error:
        assert "must be required" in str(error)
    else:
        raise AssertionError("an alternative with a stray confidence name was accepted")
