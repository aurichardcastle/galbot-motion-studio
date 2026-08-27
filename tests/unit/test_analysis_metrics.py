from __future__ import annotations

from hashlib import sha256
from json import dumps, loads
from pathlib import Path

import pytest

import galbot_motion_studio.cli as cli_module
from galbot_motion_studio.analysis_metrics import (
    AnalysisProvenanceError,
    analyze_ab,
    evaluate_release_gate,
)
from galbot_motion_studio.cli import JOINT_ORDER, main
from galbot_motion_studio.contracts.core import JointTarget, RobotTarget, SafetyDecision, SafetyOutcome
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST
from galbot_motion_studio.recording import ClipFrame, MotionClip, SourceReplayProvenance
from galbot_motion_studio.safety.supervisor import SupervisorPolicy


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _input_hash(video: Path, frame_map: Path) -> str:
    return sha256((_sha(video) + _sha(frame_map)).encode("ascii")).hexdigest()


def _target(*, sequence: int, timestamp_ns: int, mapping_hash: str, offset: float) -> RobotTarget:
    target = RobotTarget(
        session_id="analysis-session",
        sequence=sequence,
        source_clock_id="raw-clock",
        source_mono_ns=timestamp_ns,
        joints=tuple(
            JointTarget(name=name, position_rad=(offset if name == "left_arm_joint2" else 0.0))
            for name in JOINT_ORDER
        ),
        arm_generation="analysis-generation",
        model_hash=CANONICAL_MANIFEST.fixed_mjcf_sha256,
        tool_hash=SupervisorPolicy().tool_hash,
        mapping_hash=mapping_hash,
        ik_residual_m=0.002,
        predicted_clearance_m=0.012,
    )
    return target


def _clip(
    *, path: Path, mapping_hash: str, input_hash: str, sequences: tuple[int, ...], offset: float
) -> None:
    frames = []
    for sequence in sequences:
        timestamp_ns = sequence * 100
        target = _target(
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            mapping_hash=mapping_hash,
            offset=offset,
        )
        decision = SafetyDecision(
            session_id="analysis-session",
            sequence=sequence,
            source_clock_id="raw-clock",
            source_mono_ns=timestamp_ns,
            outcome=SafetyOutcome.ALLOW,
            arm_generation="analysis-generation",
            target_fingerprint=target.fingerprint,
        )
        frames.append(
            ClipFrame(
                source_sequence=sequence,
                source_mono_ns=timestamp_ns,
                target=target,
                decision=decision,
            )
        )
    MotionClip(
        clip_id=f"clip-{mapping_hash[:4]}",
        task="test",
        calibration_id="neutral-test",
        model_hash=CANONICAL_MANIFEST.fixed_mjcf_sha256,
        tool_hash=SupervisorPolicy().tool_hash,
        mapping_hash=mapping_hash,
        input_hash=input_hash,
        implementation_hash="c" * 64,
        source_replay=SourceReplayProvenance(
            origin="recorded-video-replay",
            enabled=True,
            frame_map_schema_version=3,
            capture_outcome="succeeded",
        ),
        initial_source_mono_ns=1,
        joint_order=JOINT_ORDER,
        frames=tuple(frames),
    ).save(path)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"raw-camera-video")
    frame_map = tmp_path / "raw.frame-map.json"
    frame_map.write_text(
        dumps(
            {
                "schema_version": 3,
                "artifact_kind": "raw-camera",
                "video": video.name,
                "video_sha256": _sha(video),
                "video_frame_count": 2,
                "capture_outcome": "succeeded",
                "capture_provenance": {"attempt_id": "analysis-fixture"},
                "frames": [
                    {"video_frame_index": 0, "source_sequence": 1, "source_mono_ns": 100},
                    {"video_frame_index": 1, "source_sequence": 2, "source_mono_ns": 200},
                ],
            }
        )
    )
    # Both shoulders and both hips are mandatory.  The evaluator scores the
    # exact v3 torso-relative direction vectors, never raw camera axes.
    landmark = {
        "pose_11": {"world_xyz_m": [1, 0, 0], "visibility": 1, "presence": 1},
        "pose_13": {"world_xyz_m": [1, 1, 0], "visibility": 1, "presence": 1},
        "pose_15": {"world_xyz_m": [1.5, 1, 0], "visibility": 1, "presence": 1},
        "pose_12": {"world_xyz_m": [-1, 0, 0], "visibility": 1, "presence": 1},
        "pose_14": {"world_xyz_m": [-1, 1, 0], "visibility": 1, "presence": 1},
        "pose_16": {"world_xyz_m": [-1.5, 1, 0], "visibility": 1, "presence": 1},
        "pose_23": {"world_xyz_m": [1, -1, -1], "visibility": 1, "presence": 1},
        "pose_24": {"world_xyz_m": [-1, -1, -1], "visibility": 1, "presence": 1},
    }
    sidecar = tmp_path / "raw.landmarks.json"
    sidecar.write_text(
        dumps(
            {
                "schema_version": 3,
                "artifact_kind": "raw-camera-landmarks",
                "source_video": video.name,
                "source_video_sha256": _sha(video),
                "source_frame_map": frame_map.name,
                "source_frame_map_sha256": _sha(frame_map),
                "source_video_frame_count": 2,
                "frames": [
                    {"source_sequence": 1, "source_mono_ns": 100, "landmarks": landmark},
                    {"source_sequence": 2, "source_mono_ns": 200, "landmarks": landmark},
                ],
            }
        )
    )
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    identity = _input_hash(video, frame_map)
    _clip(path=baseline, mapping_hash="b" * 64, input_hash=identity, sequences=(1, 2), offset=0.0)
    _clip(path=candidate, mapping_hash="d" * 64, input_hash=identity, sequences=(1, 2), offset=0.1)
    return {
        "video": video,
        "frame_map": frame_map,
        "sidecar": sidecar,
        "baseline": baseline,
        "candidate": candidate,
    }


def _release_arm(
    *,
    direction_error_deg: float,
    elbow_error_deg: float,
    residual_p95_m: float,
    clearance_min_m: float,
    limb_holds: int = 0,
    global_holds: int = 0,
    global_faults: int = 0,
    usable_frames: int = 30,
) -> dict[str, object]:
    def summary(value: float, *, minimum: float | None = None) -> dict[str, object]:
        return {
            "count": usable_frames,
            "mean": value,
            "p95": value,
            "min": value if minimum is None else minimum,
            "max": value,
        }

    return {
        "availability": {
            "limb_held_frames": limb_holds,
            "whole_frame_hold_frames": global_holds,
            "whole_frame_fault_frames": global_faults,
            "torso_basis_unavailable_frames": 0,
        },
        "upper_arm_angle_error_deg": summary(direction_error_deg),
        "forearm_angle_error_deg": summary(direction_error_deg),
        "elbow_flexion_error_deg": summary(elbow_error_deg),
        "wrist_residual_m": summary(residual_p95_m),
        "predicted_clearance_m": summary(clearance_min_m, minimum=clearance_min_m),
    }


def _release_inputs() -> tuple[dict[str, object], dict[str, object]]:
    baseline_arm = _release_arm(
        direction_error_deg=20.0,
        elbow_error_deg=15.0,
        residual_p95_m=0.004,
        clearance_min_m=0.008,
    )
    candidate_arm = _release_arm(
        direction_error_deg=15.0,
        elbow_error_deg=14.0,
        residual_p95_m=0.0048,
        clearance_min_m=0.006,
    )
    return (
        {"left": baseline_arm, "right": baseline_arm.copy()},
        {"left": candidate_arm, "right": candidate_arm.copy()},
    )


def test_analyze_ab_reports_shape_safety_and_provenance_metrics(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    report = analyze_ab(
        baseline_path=paths["baseline"],
        candidate_path=paths["candidate"],
        source_video_path=paths["video"],
        source_frame_map_path=paths["frame_map"],
        landmark_sidecar_path=paths["sidecar"],
    )

    assert report["provenance"]["source_command_frames"] == 2
    left = report["baseline"]["left"]
    assert left["availability"]["live_command_frames"] == 2
    assert left["upper_arm_angle_error_deg"]["count"] == 2
    assert left["wrist_residual_m"]["mean"] == pytest.approx(0.002)
    assert left["predicted_clearance_m"]["min"] == pytest.approx(0.012)
    assert report["candidate_minus_baseline"]["left"]["upper_arm_angle_error_deg_mean"] is not None
    assert report["release_gate"]["verdict"] == "FAIL"


def test_analyze_ab_keeps_rejected_wrist_residuals_separate_from_targets(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    candidate = MotionClip.load(paths["candidate"])
    first, *rest = candidate.frames
    candidate = candidate.model_copy(
        update={
            "frames": (
                first.model_copy(
                    update={
                        "held_groups": ("left_arm",),
                        "held_group_reasons": (("left_arm", "IK_DID_NOT_CONVERGE"),),
                        "held_group_residuals_m": (("left_arm", 0.024),),
                    }
                ),
                *rest,
            )
        }
    )
    candidate.save(paths["candidate"])

    report = analyze_ab(
        baseline_path=paths["baseline"],
        candidate_path=paths["candidate"],
        source_video_path=paths["video"],
        source_frame_map_path=paths["frame_map"],
        landmark_sidecar_path=paths["sidecar"],
    )

    left = report["candidate"]["left"]
    assert left["wrist_residual_m"]["count"] == 2
    assert left["rejected_wrist_residual_m"] == {
        "count": 1,
        "mean": pytest.approx(0.024),
        "p95": pytest.approx(0.024),
        "min": pytest.approx(0.024),
        "max": pytest.approx(0.024),
    }


def test_release_gate_passes_only_a_complete_meaningful_both_arm_improvement() -> None:
    baseline, candidate = _release_inputs()
    gate = evaluate_release_gate(baseline=baseline, candidate=candidate)

    assert gate["verdict"] == "PASS"
    assert gate["promotion_allowed"] is True
    assert gate["failed_checks"] == []


def test_release_gate_fails_on_any_safety_or_hold_regression() -> None:
    baseline, candidate = _release_inputs()
    candidate["left"] = _release_arm(
        direction_error_deg=15.0,
        elbow_error_deg=14.0,
        residual_p95_m=0.006,
        clearance_min_m=0.004,
        limb_holds=1,
        global_holds=1,
    )
    candidate["right"] = candidate["left"].copy()

    gate = evaluate_release_gate(baseline=baseline, candidate=candidate)

    assert gate["verdict"] == "FAIL"
    assert "left.limb_holds.non_regression" in gate["failed_checks"]
    assert "global.whole_frame_hold_frames.non_regression" in gate["failed_checks"]
    assert "left.clearance.at_sim_floor" in gate["failed_checks"]
    assert "left.wrist_residual.p95_non_regression" in gate["failed_checks"]


def test_release_gate_fails_closed_when_required_metrics_are_missing() -> None:
    baseline, candidate = _release_inputs()
    candidate["left"] = candidate["left"].copy()
    candidate["left"]["forearm_angle_error_deg"] = {"count": 30, "mean": None}

    gate = evaluate_release_gate(baseline=baseline, candidate=candidate)

    assert gate["verdict"] == "FAIL"
    assert "left.forearm_angle_error_deg.meaningful_improvement" in gate["failed_checks"]


def test_analyze_ab_cli_exit_code_follows_the_saved_release_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    document: dict[str, object] = {
        "provenance": {"source_command_frames": 30},
        "release_gate": {"verdict": "PASS"},
    }
    monkeypatch.setattr(cli_module, "analyze_ab", lambda **_kwargs: document)
    arguments = [
        "analyze-ab",
        "--baseline", str(tmp_path / "baseline.json"),
        "--candidate", str(tmp_path / "candidate.json"),
        "--source-video", str(tmp_path / "raw.mp4"),
        "--source-frame-map", str(tmp_path / "raw.frame-map.json"),
        "--landmark-sidecar", str(tmp_path / "raw.landmarks.json"),
        "--output", str(output),
    ]

    assert main(arguments) == 0
    assert loads(output.read_text())["release_gate"]["verdict"] == "PASS"
    document["release_gate"] = {"verdict": "FAIL"}
    assert main(arguments + ["--force"]) == 2
    assert loads(output.read_text())["release_gate"]["verdict"] == "FAIL"


def test_analyze_ab_rejects_any_source_sequence_mismatch(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    identity = _input_hash(paths["video"], paths["frame_map"])
    _clip(
        path=paths["candidate"],
        mapping_hash="d" * 64,
        input_hash=identity,
        sequences=(1,),
        offset=0.1,
    )
    with pytest.raises(AnalysisProvenanceError, match="source sequence"):
        analyze_ab(
            baseline_path=paths["baseline"],
            candidate_path=paths["candidate"],
            source_video_path=paths["video"],
            source_frame_map_path=paths["frame_map"],
            landmark_sidecar_path=paths["sidecar"],
        )


def test_analyze_ab_rejects_a_sidecar_not_bound_to_the_exact_frame_map(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    sidecar = loads(paths["sidecar"].read_text())
    sidecar["source_frame_map_sha256"] = "0" * 64
    paths["sidecar"].write_text(dumps(sidecar))
    with pytest.raises(AnalysisProvenanceError, match="frame_map_sha256"):
        analyze_ab(
            baseline_path=paths["baseline"],
            candidate_path=paths["candidate"],
            source_video_path=paths["video"],
            source_frame_map_path=paths["frame_map"],
            landmark_sidecar_path=paths["sidecar"],
        )


@pytest.mark.parametrize("invalid_torso", ("missing_hip", "degenerate_shoulders"))
def test_analyze_ab_fails_closed_when_a_torso_basis_cannot_be_formed(
    tmp_path: Path, invalid_torso: str
) -> None:
    paths = _inputs(tmp_path)
    sidecar = loads(paths["sidecar"].read_text())
    for row in sidecar["frames"]:
        landmarks = row["landmarks"]
        if invalid_torso == "missing_hip":
            del landmarks["pose_23"]
        else:
            landmarks["pose_12"] = landmarks["pose_11"]
    paths["sidecar"].write_text(dumps(sidecar))

    report = analyze_ab(
        baseline_path=paths["baseline"],
        candidate_path=paths["candidate"],
        source_video_path=paths["video"],
        source_frame_map_path=paths["frame_map"],
        landmark_sidecar_path=paths["sidecar"],
    )

    assert report["baseline"]["left"]["upper_arm_angle_error_deg"]["count"] == 0
    assert report["candidate"]["right"]["forearm_angle_error_deg"]["count"] == 0
    assert report["candidate"]["left"]["availability"]["torso_basis_unavailable_frames"] == 2
    assert report["release_gate"]["verdict"] == "FAIL"
    assert "global.torso_basis.complete" in report["release_gate"]["failed_checks"]


def test_analyze_ab_rejects_two_clips_stamped_with_the_same_wrong_robot_model(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    for name in ("baseline", "candidate"):
        document = loads(paths[name].read_text())
        document["model_hash"] = "f" * 64
        for frame in document["frames"]:
            frame["target"]["model_hash"] = "f" * 64
        paths[name].write_text(dumps(document))
    with pytest.raises(AnalysisProvenanceError, match="pinned MuJoCo model"):
        analyze_ab(
            baseline_path=paths["baseline"],
            candidate_path=paths["candidate"],
            source_video_path=paths["video"],
            source_frame_map_path=paths["frame_map"],
            landmark_sidecar_path=paths["sidecar"],
        )


def test_analyze_ab_reports_per_limb_holds_and_hold_reasons(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    candidate = loads(paths["candidate"].read_text())
    candidate["frames"][0]["held_groups"] = ["left_arm"]
    candidate["frames"][0]["held_group_reasons"] = [["left_arm", "INCOMPLETE_OBSERVATION"]]
    paths["candidate"].write_text(dumps(candidate))
    report = analyze_ab(
        baseline_path=paths["baseline"],
        candidate_path=paths["candidate"],
        source_video_path=paths["video"],
        source_frame_map_path=paths["frame_map"],
        landmark_sidecar_path=paths["sidecar"],
    )
    left = report["candidate"]["left"]
    assert left["availability"]["limb_held_frames"] == 1
    assert left["availability"]["live_command_frames"] == 1
    assert left["holds"]["limb_reasons"] == {"INCOMPLETE_OBSERVATION": 1}


def test_analyze_ab_cli_writes_a_report_and_rejects_implicit_overwrite(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    output = tmp_path / "metrics.json"
    arguments = [
        "analyze-ab",
        "--baseline", str(paths["baseline"]),
        "--candidate", str(paths["candidate"]),
        "--source-video", str(paths["video"]),
        "--source-frame-map", str(paths["frame_map"]),
        "--landmark-sidecar", str(paths["sidecar"]),
        "--output", str(output),
    ]
    assert main(arguments) == 2
    report = loads(output.read_text())
    assert report["schema_version"] == "1.0"
    assert report["release_gate"]["verdict"] == "FAIL"
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(arguments)


def test_analyze_ab_refuses_clips_derived_from_a_diagnostic_source(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    diagnostic = SourceReplayProvenance(
        origin="recorded-video-replay",
        enabled=True,
        frame_map_schema_version=3,
        capture_outcome="perception_fault_before_calibration",
        allow_failed_source=True,
    )
    for name in ("baseline", "candidate"):
        path = paths[name]
        MotionClip.load(path).model_copy(update={"source_replay": diagnostic}).save(path)
    with pytest.raises(AnalysisProvenanceError, match="cannot support G6"):
        analyze_ab(
            baseline_path=paths["baseline"],
            candidate_path=paths["candidate"],
            source_video_path=paths["video"],
            source_frame_map_path=paths["frame_map"],
            landmark_sidecar_path=paths["sidecar"],
        )
