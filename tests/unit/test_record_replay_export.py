from dataclasses import replace
from pathlib import Path
from json import dumps

import pytest

from galbot_motion_studio.adapters.mujoco_preview import MujocoPreviewSink
from galbot_motion_studio.contracts.core import SafetyDecision, SafetyOutcome
from galbot_motion_studio.export_v21 import export_lerobot_v21, load_exported_source_clips
from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.pipeline import MotionStudioPipeline, SIM_TELEOP_START_QPOS
from galbot_motion_studio.recording import (
    ClipFrame,
    LivenessProvenance,
    MotionClip,
    MotionRecorder,
    SourceReplayProvenance,
)
from galbot_motion_studio.replay import replay_clip
from galbot_motion_studio.safety.clearance import ClearanceChecker, HOME_QPOS
from galbot_motion_studio.safety.profiles import clearance_kwargs_for, policy_for
from galbot_motion_studio.safety.supervisor import SafetySupervisor, SupervisorState
from galbot_motion_studio.vision.liveness import LivenessPolicy

from test_left_arm_retargeting import arm_observation


# Imported, not re-declared: this file used to carry its own copy, which silently
# drifted from the CLI's when the torso joint was added and turned a real recording
# into a validation error. There is one canonical joint order.
from galbot_motion_studio.cli import JOINT_ORDER  # noqa: E402


def recorded_clip() -> MotionClip:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock")
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    pipeline.calibrate(neutral)
    recorder = MotionRecorder(
        task="mirror",
        calibration_id="cal-1",
        joint_order=JOINT_ORDER,
        source_replay=SourceReplayProvenance(origin="live-capture"),
    )
    recorder.mark_initial_source(neutral.capture_mono_ns)
    first = neutral.model_copy(
        update={
            "sequence": 2,
            "capture_mono_ns": 1_033_333_333,
            "inference_complete_mono_ns": 1_033_333_333,
        }
    )
    recorder.append(pipeline.process(first, now_mono_ns=1_033_333_334))
    moved_landmarks = tuple(
            landmark.model_copy(update={"normalized_xyz": (0.801, 0.699, 0.0)})
        if landmark.name == "pose_15"
        else landmark
        for landmark in neutral.landmarks
    )
    moved = neutral.model_copy(
        update={
            "sequence": 3,
            "capture_mono_ns": 1_100_000_000,
            "inference_complete_mono_ns": 1_100_000_000,
            "landmarks": moved_landmarks,
        }
    )
    result = pipeline.process(moved, now_mono_ns=1_100_000_001)
    assert result.decision.outcome is SafetyOutcome.ALLOW
    recorder.append(result)
    return recorder.finish()


def test_motion_clip_json_round_trip_and_fresh_replay(tmp_path: Path) -> None:
    clip = recorded_clip()
    path = tmp_path / "clip.json"
    clip.save(path)
    assert MotionClip.load(path) == clip

    model = load_verified_fixed_base_model()
    sink = MujocoPreviewSink(model=model, initial_pose=SIM_TELEOP_START_QPOS)
    supervisor = SafetySupervisor(
        ClearanceChecker(
            model=model, home=HOME_QPOS, **clearance_kwargs_for(clip.motion_profile)
        ),
        initial_pose=SIM_TELEOP_START_QPOS,
        source_clock_id="camera-clock",
        policy=policy_for(clip.motion_profile),
    )
    replay = replay_clip(clip, supervisor=supervisor, sink=sink)
    assert len(replay.receipts) == 2
    assert replay.held_frames == 0

    model_two = load_verified_fixed_base_model()
    sink_two = MujocoPreviewSink(model=model_two, initial_pose=SIM_TELEOP_START_QPOS)
    supervisor_two = SafetySupervisor(
        ClearanceChecker(
            model=model_two, home=HOME_QPOS, **clearance_kwargs_for(clip.motion_profile)
        ),
        initial_pose=SIM_TELEOP_START_QPOS,
        source_clock_id="camera-clock",
        policy=policy_for(clip.motion_profile),
    )
    replay_two = replay_clip(clip, supervisor=supervisor_two, sink=sink_two)
    assert replay_two.joint_targets == replay.joint_targets


def test_recorder_preserves_partial_group_holds_on_an_allowed_frame() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock")
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    pipeline.calibrate(neutral)
    result = pipeline.process(
        neutral.model_copy(
            update={
                "sequence": 2,
                "capture_mono_ns": 1_033_333_333,
                "inference_complete_mono_ns": 1_033_333_333,
            }
        ),
        now_mono_ns=1_033_333_334,
    )
    assert result.decision.outcome is SafetyOutcome.ALLOW
    recorder = MotionRecorder(
        task="mirror",
        calibration_id="cal-1",
        joint_order=JOINT_ORDER,
        initial_source_mono_ns=neutral.capture_mono_ns,
        analysis_arm_generation="saved-video-analysis",
    )
    recorder.append(
        replace(
            result,
            held_groups=frozenset({"left_arm"}),
            held_group_reasons=(("left_arm", "LOW_CONFIDENCE"),),
            held_group_residuals_m=(("left_arm", 0.021),),
            held_group_ik_attempts=(
                ("left_arm", (("neutral", 0.031), ("conditioned_position_priority", 0.021))),
            ),
        )
    )

    clip = recorder.finish()

    assert clip.frames[0].decision.outcome is SafetyOutcome.ALLOW
    assert clip.frames[0].held_groups == ("left_arm",)
    assert clip.frames[0].held_group_reasons == (("left_arm", "LOW_CONFIDENCE"),)
    assert clip.frames[0].held_group_residuals_m == (("left_arm", 0.021),)
    assert clip.frames[0].held_group_ik_attempts == (
        (
            "left_arm",
            (("neutral", 0.031), ("conditioned_position_priority", 0.021)),
        ),
    )
    assert clip.frames[0].target is not None
    assert clip.frames[0].target.arm_generation == "saved-video-analysis"
    assert clip.frames[0].decision.target_fingerprint == clip.frames[0].target.fingerprint


def test_fresh_replay_rearms_dynamics_after_a_recorded_tracking_hold() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock")
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    pipeline.calibrate(neutral)
    recorder = MotionRecorder(
        task="mirror",
        calibration_id="cal-1",
        joint_order=JOINT_ORDER,
        initial_source_mono_ns=neutral.capture_mono_ns,
    )
    for sequence, confidence in ((2, 1.0), (3, 0.1), (4, 1.0)):
        timestamp = 1_000_000_000 + (sequence - 1) * 33_333_333
        observation = neutral.model_copy(
            update={
                "sequence": sequence,
                "capture_mono_ns": timestamp,
                "inference_complete_mono_ns": timestamp,
                "aggregate_confidence": confidence,
            }
        )
        recorder.append(pipeline.process(observation, now_mono_ns=timestamp + 1))
    clip = recorder.finish()
    assert [frame.decision.outcome for frame in clip.frames] == [
        SafetyOutcome.ALLOW,
        SafetyOutcome.HOLD,
        SafetyOutcome.ALLOW,
    ]

    model = load_verified_fixed_base_model()
    replay = replay_clip(
        clip,
        supervisor=SafetySupervisor(
            ClearanceChecker(
                model=model,
                home=HOME_QPOS,
                **clearance_kwargs_for(clip.motion_profile),
            ),
                initial_pose=SIM_TELEOP_START_QPOS,
            source_clock_id="camera-clock",
            policy=policy_for(clip.motion_profile),
        ),
        sink=MujocoPreviewSink(model=model, initial_pose=SIM_TELEOP_START_QPOS),
    )
    assert len(replay.receipts) == 2
    assert replay.held_frames == 1


def test_replay_rejects_a_supervisor_whose_clearance_floor_is_not_the_clips() -> None:
    """A clip records its profile; the floor is re-derived from it, like the dynamics.

    Nothing in the clip names a floor, so the only thing standing between a SIM
    recording and a replay under the 10 mm hardware floor is this check. Without
    it the mismatch is silent: frames the recording approved are simply held, and
    the "replay" is of a different envelope than the one that produced the clip.
    """
    import pytest

    from galbot_motion_studio.replay import ReplayError

    clip = recorded_clip()
    assert clip.motion_profile == "sim"
    model = load_verified_fixed_base_model()
    supervisor = SafetySupervisor(
        ClearanceChecker(model=model, home=HOME_QPOS),  # defaulted floor: 10 mm
        initial_pose=HOME_QPOS,
        source_clock_id="camera-clock",
        policy=policy_for(clip.motion_profile),
    )
    with pytest.raises(ReplayError, match="self-clearance floor"):
        replay_clip(
            clip,
            supervisor=supervisor,
            sink=MujocoPreviewSink(model=model, initial_pose=HOME_QPOS),
        )


def test_v21_export_records_the_clearance_floor_the_clips_were_approved_under(
    tmp_path: Path,
) -> None:
    """A dataset consumer cannot import this package to derive the envelope."""
    from json import loads

    clip = recorded_clip().model_copy(update={"clip_id": "one", "task": "mirror"})
    destination = export_lerobot_v21((clip,), tmp_path / "dataset", fps=10)
    provenance = loads((destination / "meta/motion_studio.json").read_text())
    assert provenance["motion_profile"] == "sim"
    assert provenance["clearance_floor_m"] == 0.005
    assert "clearance floor >= 5 mm" in provenance["motion_envelope"]


def test_clip_and_export_make_liveness_configuration_auditable(tmp_path: Path) -> None:
    clip = recorded_clip().model_copy(update={"clip_id": "liveness-disabled"})
    assert clip.liveness == LivenessProvenance()
    disabled = export_lerobot_v21((clip,), tmp_path / "disabled", fps=10)

    from json import loads

    assert loads((disabled / "meta/motion_studio.json").read_text())["liveness"] == {
        "enabled": False,
        "max_static_ns": None,
        "max_history": None,
        "require_monotonic_sequence": None,
        "fingerprint_algorithm": None,
    }

    enabled_provenance = LivenessProvenance.from_policy(
        LivenessPolicy(max_static_ns=250_000_000, max_history=128),
        fingerprint_algorithm="sha256-bgr-pixels",
    )
    enabled = clip.model_copy(
        update={"clip_id": "liveness-enabled", "liveness": enabled_provenance}
    )
    destination = export_lerobot_v21((enabled,), tmp_path / "enabled", fps=10)
    assert loads((destination / "meta/motion_studio.json").read_text())["liveness"] == (
        enabled_provenance.model_dump(mode="json")
    )


def test_export_rejects_clips_with_different_liveness_settings(tmp_path: Path) -> None:
    from galbot_motion_studio.export_v21 import ExportError

    baseline = recorded_clip().model_copy(update={"clip_id": "baseline"})
    configured = baseline.model_copy(
        update={
            "clip_id": "configured",
            "liveness": LivenessProvenance.from_policy(
                LivenessPolicy(max_static_ns=250_000_000),
                fingerprint_algorithm="sha256-bgr-pixels",
            ),
        }
    )
    with pytest.raises(ExportError, match="same liveness"):
        export_lerobot_v21((baseline, configured), tmp_path / "mixed", fps=10)


def test_export_rejects_a_clip_derived_from_diagnostic_source_material(tmp_path: Path) -> None:
    clip = recorded_clip().model_copy(
        update={
            "source_replay": SourceReplayProvenance(
                origin="recorded-video-replay",
                enabled=True,
                frame_map_schema_version=3,
                capture_outcome="never_calibrated",
                allow_failed_source=True,
            )
        }
    )
    from galbot_motion_studio.export_v21 import ExportError

    with pytest.raises(ExportError, match="unknown or diagnostic"):
        export_lerobot_v21((clip,), tmp_path / "diagnostic", fps=10)


def test_export_rejects_an_unmarked_legacy_clip_instead_of_calling_it_live(
    tmp_path: Path,
) -> None:
    legacy_document = recorded_clip().model_dump(mode="json")
    del legacy_document["source_replay"]
    clip = MotionClip.model_validate_json(dumps(legacy_document))
    assert clip.source_replay == SourceReplayProvenance()
    from galbot_motion_studio.export_v21 import ExportError

    with pytest.raises(ExportError, match="unknown or diagnostic"):
        export_lerobot_v21((clip,), tmp_path / "unknown", fps=10)


def test_synthetic_provenance_is_distinct_from_live_capture_and_exportable(
    tmp_path: Path,
) -> None:
    synthetic = recorded_clip().model_copy(
        update={"source_replay": SourceReplayProvenance(origin="synthetic")}
    )
    assert synthetic.source_replay.publishable
    destination = export_lerobot_v21((synthetic,), tmp_path / "synthetic", fps=10)
    from json import loads

    assert loads((destination / "meta/motion_studio.json").read_text())["source_replay"][
        "origin"
    ] == "synthetic"


def test_clip_and_export_make_calibration_window_auditable(tmp_path: Path) -> None:
    from galbot_motion_studio.recording import (
        CalibrationWindowEvidence,
        CalibrationWindowProvenance,
    )
    from galbot_motion_studio.vision.calibration import CalibrationWindowPolicy

    clip = recorded_clip().model_copy(update={"clip_id": "window-disabled"})
    assert clip.calibration_window == CalibrationWindowProvenance()

    configured_window = CalibrationWindowProvenance.from_policy(
        CalibrationWindowPolicy(
            min_samples=10,
            max_window_ns=500_000_000,
            max_center_deviation_normalized=0.01,
            max_shoulder_width_deviation_normalized=0.02,
            max_eye_span_deviation_normalized=0.03,
        )
    )
    configured = clip.model_copy(
        update={
            "clip_id": "window-enabled",
            "calibration_window": configured_window,
            "calibration_window_evidence": CalibrationWindowEvidence(
                samples_used=10,
                window_span_ns=400_000_000,
                first_observation_sequence=11,
                last_observation_sequence=20,
            ),
        }
    )
    destination = export_lerobot_v21((configured,), tmp_path / "configured", fps=10)

    from json import loads

    assert loads((destination / "meta/motion_studio.json").read_text())["calibration_window"] == (
        configured_window.model_dump(mode="json")
    )
    assert loads((destination / "meta/motion_studio.json").read_text())["calibration_windows"] == [
        {
            "clip_id": "window-enabled",
            "evidence": {
                "samples_used": 10,
                "window_span_ns": 400_000_000,
                "first_observation_sequence": 11,
                "last_observation_sequence": 20,
            },
        }
    ]


def test_export_rejects_clips_with_different_calibration_windows(tmp_path: Path) -> None:
    from galbot_motion_studio.export_v21 import ExportError
    from galbot_motion_studio.recording import (
        CalibrationWindowEvidence,
        CalibrationWindowProvenance,
    )
    from galbot_motion_studio.vision.calibration import CalibrationWindowPolicy

    baseline = recorded_clip().model_copy(update={"clip_id": "baseline"})
    configured = baseline.model_copy(
        update={
            "clip_id": "configured",
            "calibration_window": CalibrationWindowProvenance.from_policy(
                CalibrationWindowPolicy(
                    min_samples=10,
                    max_window_ns=500_000_000,
                    max_center_deviation_normalized=0.01,
                    max_shoulder_width_deviation_normalized=0.02,
                    max_eye_span_deviation_normalized=0.03,
                )
            ),
            "calibration_window_evidence": CalibrationWindowEvidence(
                samples_used=10,
                window_span_ns=400_000_000,
                first_observation_sequence=11,
                last_observation_sequence=20,
            ),
        }
    )
    with pytest.raises(ExportError, match="same calibration_window"):
        export_lerobot_v21((baseline, configured), tmp_path / "mixed", fps=10)


def test_enabled_calibration_window_cannot_be_saved_without_its_evidence() -> None:
    from galbot_motion_studio.recording import CalibrationWindowProvenance, MotionClip
    from galbot_motion_studio.vision.calibration import CalibrationWindowPolicy

    clip = recorded_clip().model_copy(
        update={
            "calibration_window": CalibrationWindowProvenance.from_policy(
                CalibrationWindowPolicy(10, 500_000_000, 0.01, 0.02, 0.03)
            )
        }
    )
    with pytest.raises(ValueError, match="requires evidence"):
        MotionClip.model_validate(clip.model_dump())


def test_clip_and_export_make_absent_occupancy_evidence_explicit(tmp_path: Path) -> None:
    from galbot_motion_studio.recording import OccupancyProvenance

    clip = recorded_clip().model_copy(update={"clip_id": "occupancy-disabled"})
    assert clip.occupancy == OccupancyProvenance()
    destination = export_lerobot_v21((clip,), tmp_path / "disabled", fps=10)

    from json import loads

    assert loads((destination / "meta/motion_studio.json").read_text())["occupancy"] == {
        "enabled": False,
        "provider_id": None,
        "detector_model_hash": None,
        "landmark_model_hash": None,
        "operator_zone_hash": None,
        "association_contract_version": None,
        "max_evidence_age_ns": None,
    }


def test_export_rejects_clips_with_different_occupancy_evidence(tmp_path: Path) -> None:
    from galbot_motion_studio.export_v21 import ExportError
    from galbot_motion_studio.recording import OccupancyProvenance

    baseline = recorded_clip().model_copy(update={"clip_id": "baseline"})
    configured = baseline.model_copy(
        update={
            "clip_id": "configured",
            "occupancy": OccupancyProvenance.from_configuration(
                provider_id="independent-zone-detector",
                detector_model_hash="a" * 64,
                landmark_model_hash="c" * 64,
                operator_zone_hash="b" * 64,
                association_contract_version="occupancy-selection-v1",
                max_evidence_age_ns=50_000_000,
            ),
        }
    )
    with pytest.raises(ExportError, match="same occupancy"):
        export_lerobot_v21((baseline, configured), tmp_path / "mixed", fps=10)


def test_enabled_occupancy_provenance_rejects_a_landmark_derived_detector() -> None:
    from galbot_motion_studio.recording import OccupancyProvenance

    with pytest.raises(ValueError, match="distinct hashes"):
        OccupancyProvenance.from_configuration(
            provider_id="not-independent",
            detector_model_hash="a" * 64,
            landmark_model_hash="a" * 64,
            operator_zone_hash="b" * 64,
            association_contract_version="occupancy-selection-v1",
            max_evidence_age_ns=50_000_000,
        )


def test_regressed_ingress_fault_is_retained_without_rewriting_source_time(
    tmp_path: Path,
) -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock")
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    pipeline.calibrate(neutral)
    recorder = MotionRecorder(
        task="mirror",
        calibration_id="cal-1",
        joint_order=JOINT_ORDER,
        source_replay=SourceReplayProvenance(origin="live-capture"),
    )
    assert not recorder.has_robot_target
    approved = pipeline.process(
        neutral.model_copy(
            update={
                "sequence": 2,
                "capture_mono_ns": 1_033_333_333,
                "inference_complete_mono_ns": 1_033_333_333,
            }
        ),
        now_mono_ns=1_033_333_334,
    )
    recorder.append(approved)
    assert recorder.has_robot_target
    ingress_fault = pipeline.fault_perception(
        session_id="camera-session",
        sequence=3,
        source_mono_ns=1_000_000_000,
        now_mono_ns=1_033_333_335,
        reason="capture timestamp is not strictly monotonic",
    )
    recorder.append(ingress_fault)
    clip = recorder.finish()

    assert len(clip.frames) == 1
    assert clip.terminal_fault == ingress_fault.decision
    assert clip.terminal_fault.source_mono_ns == 1_000_000_000
    assert recorder.non_monotonic_frames_dropped == 0

    destination = export_lerobot_v21((clip,), tmp_path / "faulted", fps=10)
    from json import loads

    faults = loads((destination / "meta/motion_studio.json").read_text())["terminal_faults"]
    assert faults == [
        {
            "clip_id": clip.clip_id,
            "decision": ingress_fault.decision.model_dump(mode="json"),
        }
    ]

    model = load_verified_fixed_base_model()
    supervisor = SafetySupervisor(
        ClearanceChecker(
            model=model,
            home=HOME_QPOS,
            **clearance_kwargs_for(clip.motion_profile),
        ),
        initial_pose=SIM_TELEOP_START_QPOS,
        source_clock_id="camera-clock",
        policy=policy_for(clip.motion_profile),
    )
    replay = replay_clip(
        clip,
        supervisor=supervisor,
        sink=MujocoPreviewSink(model=model, initial_pose=SIM_TELEOP_START_QPOS),
    )
    assert replay.terminal_fault == ingress_fault.decision
    assert supervisor.state is SupervisorState.FAULT


def test_first_retained_command_must_advance_past_the_calibration_seed() -> None:
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock")
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    pipeline.calibrate(neutral)
    approved = pipeline.process(
        neutral.model_copy(
            update={
                "sequence": 2,
                "capture_mono_ns": 1_033_333_333,
                "inference_complete_mono_ns": 1_033_333_333,
            }
        ),
        now_mono_ns=1_033_333_334,
    )
    recorder = MotionRecorder(
        task="mirror",
        calibration_id="cal-1",
        joint_order=JOINT_ORDER,
        initial_source_mono_ns=approved.target.source_mono_ns,
    )
    recorder.append(approved)
    assert not recorder.has_robot_target
    assert recorder.non_monotonic_frames_dropped == 1


def test_v21_export_loads_in_lerobot_passes_linter_and_round_trips(
    tmp_path: Path,
) -> None:
    clip = recorded_clip()
    clips = (clip.model_copy(update={"clip_id": "one", "task": "mirror"}),)
    destination = export_lerobot_v21(clips, tmp_path / "dataset", fps=10)
    assert load_exported_source_clips(destination) == clips

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded = LeRobotDataset(
        repo_id="local/galbot-motion-studio",
        root=destination,
        download_videos=False,
    )
    assert len(loaded) == len(clip.frames)
    assert loaded.num_episodes == 1

    import sys

    validator_root = Path(__file__).resolve().parents[3] / "lerobot-dataset-check"
    sys.path.insert(0, str(validator_root))
    try:
        from lerobot_dataset_check import checks, loaders, report

        dataset = loaders.open_dataset(destination)
        sections = [check(dataset) for check in checks.ALL_CHECKS]
        assert not report.has_failure(sections)
        assert all(
            status != report.WARN
            for section in sections
            for status, _message in section.findings
        )

        with (destination / "meta/episodes_stats.jsonl").open("a", encoding="utf-8") as file:
            file.write(dumps({"episode_index": 0, "stats": {}}) + "\n")
        corrupted = loaders.open_dataset(destination)
        assert report.has_failure([checks.check_stats_bijection(corrupted)])
    finally:
        sys.path.remove(str(validator_root))


def test_short_hold_repeats_last_approved_target_and_is_declared(tmp_path: Path) -> None:
    clip = recorded_clip()
    last = clip.frames[-1]
    hold_decision = SafetyDecision(
        session_id=last.decision.session_id,
        sequence=4,
        source_clock_id=last.decision.source_clock_id,
        source_mono_ns=1_233_333_333,
        arm_generation=last.decision.arm_generation,
        outcome=SafetyOutcome.HOLD,
        reasons=("synthetic dropout",),
    )
    held = ClipFrame(
        source_sequence=4,
        source_mono_ns=1_233_333_333,
        target=None,
        decision=hold_decision,
    )
    clip = clip.model_copy(update={"frames": clip.frames + (held,)})
    destination = export_lerobot_v21((clip,), tmp_path / "held", fps=10)

    import pandas as pd

    frame = pd.read_parquet(destination / "data/chunk-000/episode_000000.parquet")
    assert frame["held"].tolist() == [False, False, True]
    assert frame["held_head"].tolist() == [False, False, True]
    assert frame["held_left_arm"].tolist() == [False, False, True]
    assert frame["held_right_arm"].tolist() == [False, False, True]
    assert list(frame.iloc[-1]["action"]) == list(frame.iloc[-2]["action"])


def test_v21_export_preserves_partial_group_holds_without_relabeling_global_allow(
    tmp_path: Path,
) -> None:
    clip = recorded_clip()
    partial = clip.frames[0].model_copy(update={"held_groups": ("left_arm",)})
    clip = clip.model_copy(update={"frames": (partial,) + clip.frames[1:]})

    destination = export_lerobot_v21((clip,), tmp_path / "partial", fps=10)

    import pandas as pd
    from json import loads

    frame = pd.read_parquet(destination / "data/chunk-000/episode_000000.parquet")
    assert frame["held"].tolist()[0] is False
    assert frame["held_head"].tolist()[0] is False
    assert frame["held_left_arm"].tolist()[0] is True
    assert frame["held_right_arm"].tolist()[0] is False
    provenance = loads((destination / "meta/motion_studio.json").read_text())
    assert provenance["held_group_frames"] == {
        "head": 0,
        "left_arm": 1,
        "right_arm": 0,
    }


def test_a_repeated_camera_timestamp_does_not_destroy_the_whole_recording() -> None:
    """A live webcam can hand two frames the same capture time under load.

    Measured on a real 1196-frame session: every frame processed fine, the
    operator quit cleanly, and then `recorder.finish()` raised
    "clip frame timestamps must strictly increase" and the ENTIRE recording was
    lost at save time. A duplicate instant is not a distinct sample, so the
    second frame is dropped and the session survives. The validator stays strict
    because the v2.1 export depends on strictly increasing time.
    """
    pipeline = MotionStudioPipeline(source_clock_id="camera-clock")
    neutral = arm_observation(
        sequence=1,
        capture_mono_ns=1_000_000_000,
        inference_complete_mono_ns=1_000_000_000,
    )
    pipeline.calibrate(neutral)
    recorder = MotionRecorder(task="mirror", calibration_id="cal-1", joint_order=JOINT_ORDER)
    recorder.mark_initial_source(neutral.capture_mono_ns)

    first = neutral.model_copy(
        update={
            "sequence": 2,
            "capture_mono_ns": 1_033_333_333,
            "inference_complete_mono_ns": 1_033_333_333,
        }
    )
    recorder.append(pipeline.process(first, now_mono_ns=1_033_333_334))

    # Sequence advances, capture time does NOT. This is the exact shape the live
    # session produced: the sequence check passes and the timestamp check fails.
    stalled = neutral.model_copy(
        update={
            "sequence": 3,
            "capture_mono_ns": 1_033_333_333,
            "inference_complete_mono_ns": 1_033_333_333,
        }
    )
    # process_fail_closed is what the CLI runner actually calls. The duplicate
    # timestamp makes the retarget filter raise and that becomes a FAULT.
    #
    # now_mono_ns is deliberately far AHEAD of the capture time, which is the
    # real relationship: the processing clock always leads the frame it is
    # processing. A FAULT that stamped itself from now_mono_ns therefore landed
    # past every subsequent frame's capture time.
    stalled_result = pipeline.process_fail_closed(stalled, now_mono_ns=9_000_000_000)
    assert stalled_result.decision.outcome is not SafetyOutcome.ALLOW
    # The FAULT must carry the SOURCE frame's time, not the processing clock.
    assert stalled_result.decision.source_mono_ns == 1_033_333_333
    recorder.append(stalled_result)

    later = neutral.model_copy(
        update={
            "sequence": 4,
            "capture_mono_ns": 1_100_000_000,
            "inference_complete_mono_ns": 1_100_000_000,
        }
    )
    recorder.append(pipeline.process(later, now_mono_ns=1_100_000_001))

    clip = recorder.finish()

    assert recorder.non_monotonic_frames_dropped == 1
    assert [frame.source_mono_ns for frame in clip.frames] == [1_033_333_333, 1_100_000_000]
    assert [frame.source_sequence for frame in clip.frames] == [2, 4]
