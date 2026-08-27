import pytest

from galbot_motion_studio.contracts.human import IdentityState
from galbot_motion_studio.vision.selection import (
    OccupancyObservation,
    OperatorSelectionManager,
    SelectionPolicy,
    SelectionRejection,
    SelectionState,
)

from test_calibration import stable_observation


POLICY = SelectionPolicy(
    max_timestamp_skew_ns=1_000_000,
    max_evidence_age_ns=50_000_000,
    max_selection_grace_ns=100_000_000,
    max_cumulative_occlusion_ns=200_000_000,
    operator_zone_normalized_bounds=(0.1, 0.1, 0.9, 0.9),
    max_candidate_center_delta_normalized=0.1,
    max_shoulder_width_relative_delta=0.1,
    max_eye_span_relative_delta=0.1,
)


def _occupancy(
    timestamp_ns: int,
    *,
    sequence: int | None = None,
    fingerprint: str | None = None,
    inference_complete_mono_ns: int | None = None,
    count: int = 1,
    center=(0.5, 0.5),
) -> OccupancyObservation:
    sequence = timestamp_ns if sequence is None else sequence
    return OccupancyObservation(
        camera_id="webcam",
        source_clock_id="camera-clock",
        source_sequence=sequence,
        capture_mono_ns=timestamp_ns,
        inference_complete_mono_ns=(
            timestamp_ns if inference_complete_mono_ns is None else inference_complete_mono_ns
        ),
        content_fingerprint=f"frame-{sequence}" if fingerprint is None else fingerprint,
        candidate_count=count,
        candidate_center_normalized_xy=center if count == 1 else None,
    )


def _observation(timestamp_ns: int, **changes: object):
    changes.setdefault("sequence", timestamp_ns)
    changes.setdefault("content_fingerprint", f"frame-{changes['sequence']}")
    return stable_observation(
        capture_mono_ns=timestamp_ns,
        inference_complete_mono_ns=timestamp_ns,
        source_mono_ns=timestamp_ns,
        **changes,
    )


def _scaled_anthropometry_observation(timestamp_ns: int, *, scale: float):
    """Keep the centre fixed while scaling only continuity anthropometry."""
    base = _observation(timestamp_ns)
    coordinates = {
        "pose_11": (0.5 + 0.2 * scale, 0.4, 0.0),
        "pose_12": (0.5 - 0.2 * scale, 0.4, 0.0),
        "face_33": (0.5 - 0.1 * scale, 0.2, 0.0),
        "face_263": (0.5 + 0.1 * scale, 0.2, 0.0),
    }
    return base.model_copy(
        update={
            "landmarks": tuple(
                landmark.model_copy(update={"normalized_xyz": coordinates[landmark.name]})
                if landmark.name in coordinates
                else landmark
                for landmark in base.landmarks
            )
        }
    )


def test_selection_requires_explicit_acquisition() -> None:
    manager = OperatorSelectionManager(POLICY)
    decision = manager.observe(_occupancy(10), _observation(10), now_mono_ns=11)
    assert decision.state is SelectionState.ACQUIRING
    assert decision.reason is SelectionRejection.NOT_ACQUIRED


def test_selection_holds_on_zero_or_multiple_occupancy_candidates() -> None:
    for count in (0, 2):
        manager = OperatorSelectionManager(POLICY)
        acquired = manager.acquire(_occupancy(10), _observation(10), now_mono_ns=11)
        assert acquired.accepted
        decision = manager.observe(
            _occupancy(20, count=count), _observation(20), now_mono_ns=21
        )
        assert decision.state is SelectionState.LOST
        assert decision.reason is SelectionRejection.OCCUPANCY_AMBIGUOUS


def test_selection_rejects_detector_landmark_timestamp_skew() -> None:
    manager = OperatorSelectionManager(POLICY)
    decision = manager.acquire(
        _occupancy(10, sequence=20_000_000),
        _observation(20_000_000),
        now_mono_ns=20_000_001,
    )
    assert not decision.accepted
    assert decision.reason is SelectionRejection.TIMESTAMP_SKEW


def test_selection_relocks_only_after_continuity_check_within_grace() -> None:
    manager = OperatorSelectionManager(POLICY)
    acquired = manager.acquire(
        _occupancy(1_000_000_000),
        _observation(1_000_000_000),
        now_mono_ns=1_000_000_001,
    )
    assert acquired.accepted and acquired.selection_id == 1
    occluded = manager.observe(
        _occupancy(1_020_000_000),
        _observation(1_020_000_000, identity=IdentityState.LOST),
        now_mono_ns=1_020_000_001,
    )
    assert occluded.state is SelectionState.OCCLUDED
    assert not occluded.accepted
    relocked = manager.observe(
        _occupancy(1_050_000_000, center=(0.55, 0.5)),
        _observation(1_050_000_000),
        now_mono_ns=1_050_000_001,
    )
    assert relocked.state is SelectionState.LOCKED
    assert relocked.accepted
    assert relocked.selection_id == 1


def test_missing_occupancy_starts_a_bounded_continuity_gap() -> None:
    manager = OperatorSelectionManager(POLICY)
    assert manager.acquire(
        _occupancy(1_000_000_000),
        _observation(1_000_000_000),
        now_mono_ns=1_000_000_001,
    ).accepted

    missing = manager.observe_missing_occupancy(capture_mono_ns=1_020_000_000)
    assert missing.state is SelectionState.OCCLUDED
    assert missing.selection_id == 1
    assert missing.reason is SelectionRejection.OCCUPANCY_MISSING

    relocked = manager.observe(
        _occupancy(1_050_000_000),
        _observation(1_050_000_000),
        now_mono_ns=1_050_000_001,
    )
    assert relocked.accepted and relocked.selection_id == 1

    manager.observe_missing_occupancy(capture_mono_ns=1_070_000_000)
    expired = manager.observe_missing_occupancy(capture_mono_ns=1_171_000_000)
    assert expired.state is SelectionState.LOST
    assert expired.reason is SelectionRejection.SELECTION_GRACE_EXPIRED


def test_selection_preserves_the_occlusion_grace_across_multiple_unstable_frames() -> None:
    manager = OperatorSelectionManager(POLICY)
    assert manager.acquire(
        _occupancy(1_000_000_000),
        _observation(1_000_000_000),
        now_mono_ns=1_000_000_001,
    ).accepted
    first = manager.observe(
        _occupancy(1_020_000_000),
        _observation(1_020_000_000, identity=IdentityState.LOST),
        now_mono_ns=1_020_000_001,
    )
    second = manager.observe(
        _occupancy(1_050_000_000),
        _observation(1_050_000_000, identity=IdentityState.LOST),
        now_mono_ns=1_050_000_001,
    )
    assert first.state is second.state is SelectionState.OCCLUDED
    relocked = manager.observe(
        _occupancy(1_080_000_000),
        _observation(1_080_000_000),
        now_mono_ns=1_080_000_001,
    )
    assert relocked.accepted and relocked.selection_id == 1


def test_selection_limits_cumulative_occlusion_across_relocks() -> None:
    manager = OperatorSelectionManager(POLICY)
    assert manager.acquire(
        _occupancy(1_000_000_000), _observation(1_000_000_000), now_mono_ns=1_000_000_001
    ).accepted
    for missing_at, recovered_at in (
        (1_010_000_000, 1_060_000_000),
        (1_070_000_000, 1_120_000_000),
        (1_130_000_000, 1_180_000_000),
    ):
        manager.observe_missing_occupancy(capture_mono_ns=missing_at)
        recovered = manager.observe(
            _occupancy(recovered_at), _observation(recovered_at), now_mono_ns=recovered_at + 1
        )
        assert recovered.accepted
    manager.observe_missing_occupancy(capture_mono_ns=1_190_000_000)
    expired = manager.observe(
        _occupancy(1_241_000_000), _observation(1_241_000_000), now_mono_ns=1_241_000_001
    )
    assert expired.state is SelectionState.LOST
    assert expired.reason is SelectionRejection.CUMULATIVE_OCCLUSION_EXPIRED


def test_selection_does_not_ratchet_anthropometry_away_from_acquisition() -> None:
    manager = OperatorSelectionManager(POLICY)
    baseline = _scaled_anthropometry_observation(1_000, scale=1.0)
    assert manager.acquire(_occupancy(1_000), baseline, now_mono_ns=1_001).accepted
    for index in (1, 2):
        timestamp = 1_000 + index * 10
        assert manager.observe(
            _occupancy(timestamp),
            _scaled_anthropometry_observation(timestamp, scale=1.04**index),
            now_mono_ns=timestamp + 1,
        ).accepted
    rejected = manager.observe(
        _occupancy(1_030),
        _scaled_anthropometry_observation(1_030, scale=1.04**3),
        now_mono_ns=1_031,
    )
    assert rejected.state is SelectionState.LOST
    assert rejected.reason is SelectionRejection.CONTINUITY_FAILED


def test_selection_fails_closed_on_one_to_one_substitution_after_occlusion() -> None:
    manager = OperatorSelectionManager(POLICY)
    assert manager.acquire(
        _occupancy(1_000_000_000),
        _observation(1_000_000_000),
        now_mono_ns=1_000_000_001,
    ).accepted
    manager.observe(
        _occupancy(1_020_000_000),
        _observation(1_020_000_000, identity=IdentityState.LOST),
        now_mono_ns=1_020_000_001,
    )
    substituted = manager.observe(
        _occupancy(1_050_000_000, center=(0.8, 0.5)),
        _observation(1_050_000_000),
        now_mono_ns=1_050_000_001,
    )
    assert substituted.state is SelectionState.LOST
    assert substituted.reason is SelectionRejection.CONTINUITY_FAILED


def test_selection_fails_closed_when_grace_expires() -> None:
    manager = OperatorSelectionManager(POLICY)
    assert manager.acquire(
        _occupancy(1_000_000_000),
        _observation(1_000_000_000),
        now_mono_ns=1_000_000_001,
    ).accepted
    manager.observe(
        _occupancy(1_020_000_000),
        _observation(1_020_000_000, identity=IdentityState.LOST),
        now_mono_ns=1_020_000_001,
    )
    expired = manager.observe(
        _occupancy(1_121_000_000),
        _observation(1_121_000_000),
        now_mono_ns=1_121_000_001,
    )
    assert expired.state is SelectionState.LOST
    assert expired.reason is SelectionRejection.SELECTION_GRACE_EXPIRED


def test_reacquisition_never_reuses_a_lost_selection_generation() -> None:
    manager = OperatorSelectionManager(POLICY)
    assert manager.acquire(
        _occupancy(10), _observation(10), now_mono_ns=11
    ).selection_id == 1
    lost = manager.observe(
        _occupancy(20, count=2), _observation(20), now_mono_ns=21
    )
    assert lost.state is SelectionState.LOST
    reacquired = manager.acquire(
        _occupancy(30), _observation(30), now_mono_ns=31
    )
    assert reacquired.accepted and reacquired.selection_id == 2


def test_selection_rejects_invalid_policy_and_non_singleton_centres() -> None:
    with pytest.raises(ValueError, match="occlusion"):
        SelectionPolicy(0, 1, 0, 1, (0.1, 0.1, 0.9, 0.9), 0.1, 0.1, 0.1)
    with pytest.raises(ValueError, match="single"):
        OccupancyObservation("webcam", "camera-clock", 1, 1, 1, "frame-1", 2, (0.5, 0.5))


def test_selection_requires_exact_frame_evidence_and_fresh_occupancy() -> None:
    manager = OperatorSelectionManager(POLICY)
    mismatch = manager.acquire(
        _occupancy(10, sequence=9), _observation(10), now_mono_ns=11
    )
    assert mismatch.reason is SelectionRejection.FRAME_REFERENCE_MISMATCH

    stale = manager.acquire(
        _occupancy(10), _observation(10), now_mono_ns=60_000_011
    )
    assert stale.reason is SelectionRejection.OCCUPANCY_STALE

    future = manager.acquire(
        _occupancy(10, inference_complete_mono_ns=12), _observation(10), now_mono_ns=11
    )
    assert future.reason is SelectionRejection.OCCUPANCY_FUTURE

    outside_zone = manager.acquire(
        _occupancy(20, center=(0.95, 0.5)), _observation(20), now_mono_ns=21
    )
    assert outside_zone.reason is SelectionRejection.OCCUPANCY_OUTSIDE_OPERATOR_ZONE


def test_selection_cannot_switch_camera_after_acquisition() -> None:
    manager = OperatorSelectionManager(POLICY)
    assert manager.acquire(_occupancy(10), _observation(10), now_mono_ns=11).accepted
    switched = _observation(20, camera_id="other-webcam")
    switched_occupancy = OccupancyObservation(
        camera_id="other-webcam",
        source_clock_id="camera-clock",
        source_sequence=20,
        capture_mono_ns=20,
        inference_complete_mono_ns=20,
        content_fingerprint="frame-20",
        candidate_count=1,
        candidate_center_normalized_xy=(0.5, 0.5),
    )
    decision = manager.observe(switched_occupancy, switched, now_mono_ns=21)
    assert decision.state is SelectionState.LOST
    assert decision.reason is SelectionRejection.CLOCK_MISMATCH
