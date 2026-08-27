"""Fail-closed, deterministic metrics for a source-aligned teleoperation A/B.

This module intentionally does *not* try to infer timestamps from an MP4.  An
analysis is valid only when each command frame can be joined, by
``source_sequence`` and source timestamp, to the raw-video frame map and to the
raw landmark sidecar captured with that video.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from json import dumps, loads
from math import acos, degrees, isfinite, sqrt
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping

import mujoco

from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST
from galbot_motion_studio.recording import ClipFrame, MotionClip
from galbot_motion_studio.safety.profiles import SIM_CLEARANCE_FLOOR_M
from galbot_motion_studio.safety.supervisor import SupervisorPolicy


class AnalysisProvenanceError(ValueError):
    """Raised when an A/B input cannot prove an exact common source timeline."""


_SIDES = {
    "left": ("pose_11", "pose_13", "pose_15"),
    "right": ("pose_12", "pose_14", "pose_16"),
}
_TORSO = ("pose_11", "pose_12", "pose_23", "pose_24")
_MIN_LANDMARK_CONFIDENCE = 0.3
_Vector3 = tuple[float, float, float]
_TorsoBasis = tuple[_Vector3, _Vector3, _Vector3]

# Release policy is deliberately small, named, and fixed.  These are not
# optimiser knobs: changing any one changes the meaning of a PASS, so the
# complete policy is written into every report below.
MIN_USABLE_FRAMES_PER_ARM = 30
MIN_DIRECTION_ERROR_IMPROVEMENT_DEG = 2.0
MIN_DIRECTION_ERROR_IMPROVEMENT_RATIO = 0.10
ELBOW_FLEXION_MAX_REGRESSION_DEG = 0.0
WRIST_RESIDUAL_P95_TOLERANCE_M = 0.001
RELEASE_GATE_VERSION = "1.0"


@dataclass(frozen=True)
class _SourceFrame:
    sequence: int
    timestamp_ns: int


@dataclass
class _ArmMetrics:
    source_frames: int = 0
    human_segment_frames: int = 0
    live_command_frames: int = 0
    target_frames: int = 0
    limb_held_frames: int = 0
    whole_frame_nonallow_frames: int = 0
    whole_frame_hold_frames: int = 0
    whole_frame_fault_frames: int = 0
    torso_basis_unavailable_frames: int = 0
    upper_arm_error_deg: list[float] = field(default_factory=list)
    forearm_error_deg: list[float] = field(default_factory=list)
    elbow_flexion_error_deg: list[float] = field(default_factory=list)
    wrist_residual_m: list[float] = field(default_factory=list)
    rejected_wrist_residual_m: list[float] = field(default_factory=list)
    clearance_m: list[float] = field(default_factory=list)
    limb_hold_reasons: Counter[str] = field(default_factory=Counter)
    whole_frame_reasons: Counter[str] = field(default_factory=Counter)

    def document(self) -> dict[str, object]:
        return {
            "availability": {
                "source_frames": self.source_frames,
                "human_segment_frames": self.human_segment_frames,
                "live_command_frames": self.live_command_frames,
                "target_frames": self.target_frames,
                "limb_held_frames": self.limb_held_frames,
                "whole_frame_nonallow_frames": self.whole_frame_nonallow_frames,
                "whole_frame_hold_frames": self.whole_frame_hold_frames,
                "whole_frame_fault_frames": self.whole_frame_fault_frames,
                "torso_basis_unavailable_frames": self.torso_basis_unavailable_frames,
            },
            "upper_arm_angle_error_deg": _summary(self.upper_arm_error_deg),
            "forearm_angle_error_deg": _summary(self.forearm_error_deg),
            "elbow_flexion_error_deg": _summary(self.elbow_flexion_error_deg),
            # This is the mapper's recorded TCP position residual.  It is
            # intentionally reported beside each arm because RobotTarget stores a
            # single whole-body maximum, rather than inventing a per-arm value.
            "wrist_residual_m": _summary(self.wrist_residual_m),
            # A local IK rejection has no RobotTarget by design, but its finite
            # best-effort residual is diagnostic evidence. Keep this distinct
            # from accepted target residuals so an offline report cannot make a
            # rejected solution look commanded.
            "rejected_wrist_residual_m": _summary(self.rejected_wrist_residual_m),
            "predicted_clearance_m": _summary(self.clearance_m),
            "holds": {
                "limb_reasons": dict(sorted(self.limb_hold_reasons.items())),
                "whole_frame_reasons": dict(sorted(self.whole_frame_reasons.items())),
            },
        }


def analyze_ab(
    *,
    baseline_path: Path,
    candidate_path: Path,
    landmark_sidecar_path: Path,
    source_video_path: Path,
    source_frame_map_path: Path,
) -> dict[str, object]:
    """Evaluate two mappings against exactly the same captured human frames.

    The return value is JSON-serialisable and stable for byte-identical inputs.
    ``AnalysisProvenanceError`` is deliberately raised before calculating a
    metric if provenance is missing, stale, ambiguous, or only approximately
    aligned.
    """
    baseline = MotionClip.load(baseline_path)
    candidate = MotionClip.load(candidate_path)
    timeline = _load_frame_map(source_frame_map_path, source_video_path)
    sidecar = _load_landmark_sidecar(
        landmark_sidecar_path, source_video_path, source_frame_map_path, timeline
    )
    _validate_clips(baseline, candidate, source_video_path, source_frame_map_path, timeline)

    model = load_verified_fixed_base_model()
    baseline_metrics = _measure_clip(baseline, sidecar, model)
    candidate_metrics = _measure_clip(candidate, sidecar, model)
    baseline_document = {side: metric.document() for side, metric in baseline_metrics.items()}
    candidate_document = {side: metric.document() for side, metric in candidate_metrics.items()}
    source_sequences = [frame.source_sequence for frame in baseline.frames]
    return {
        "schema_version": "1.0",
        "provenance": {
            "source_video": source_video_path.name,
            "source_video_sha256": _file_sha256(source_video_path),
            "source_frame_map_sha256": _file_sha256(source_frame_map_path),
            "landmark_sidecar_sha256": _file_sha256(landmark_sidecar_path),
            "input_hash": baseline.input_hash,
            "source_sequence_sha256": _sequence_hash(source_sequences),
            "source_command_frames": len(source_sequences),
            "baseline": _clip_identity(baseline),
            "candidate": _clip_identity(candidate),
        },
        "baseline": baseline_document,
        "candidate": candidate_document,
        "candidate_minus_baseline": {
            side: _metric_delta(baseline_document[side], candidate_document[side])
            for side in _SIDES
        },
        "release_gate": evaluate_release_gate(
            baseline=baseline_document,
            candidate=candidate_document,
        ),
    }


def evaluate_release_gate(
    *, baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    """Return the deterministic, fail-closed mapping-promotion verdict.

    This is a simulation-analysis release gate, not a statement of hardware
    safety.  A malformed or partial report is a FAIL rather than a best-effort
    interpretation: without all metrics there is no evidence for promotion.
    """
    checks: list[dict[str, object]] = []
    for side in sorted(_SIDES):
        old = _arm_document(baseline, side)
        new = _arm_document(candidate, side)
        for metric_name in ("upper_arm_angle_error_deg", "forearm_angle_error_deg"):
            old_summary = _metric_summary(old, metric_name)
            new_summary = _metric_summary(new, metric_name)
            old_mean = _number(old_summary, "mean")
            new_mean = _number(new_summary, "mean")
            old_count = _integer(old_summary, "count")
            new_count = _integer(new_summary, "count")
            usable = (
                old_count is not None
                and new_count is not None
                and old_count >= MIN_USABLE_FRAMES_PER_ARM
                and new_count >= MIN_USABLE_FRAMES_PER_ARM
            )
            improvement = (
                old_mean - new_mean if old_mean is not None and new_mean is not None else None
            )
            ratio = improvement / old_mean if improvement is not None and old_mean > 0.0 else None
            passed = bool(
                usable
                and improvement is not None
                and ratio is not None
                and (
                    improvement >= MIN_DIRECTION_ERROR_IMPROVEMENT_DEG
                    or ratio >= MIN_DIRECTION_ERROR_IMPROVEMENT_RATIO
                )
            )
            checks.append(
                _check(
                    f"{side}.{metric_name}.meaningful_improvement",
                    passed,
                    baseline_count=old_count,
                    candidate_count=new_count,
                    baseline_mean_deg=old_mean,
                    candidate_mean_deg=new_mean,
                    improvement_deg=improvement,
                    improvement_ratio=ratio,
                    min_usable_frames=MIN_USABLE_FRAMES_PER_ARM,
                    min_improvement_deg=MIN_DIRECTION_ERROR_IMPROVEMENT_DEG,
                    min_improvement_ratio=MIN_DIRECTION_ERROR_IMPROVEMENT_RATIO,
                )
            )

        old_elbow = _metric_summary(old, "elbow_flexion_error_deg")
        new_elbow = _metric_summary(new, "elbow_flexion_error_deg")
        old_elbow_mean = _number(old_elbow, "mean")
        new_elbow_mean = _number(new_elbow, "mean")
        checks.append(
            _check(
                f"{side}.elbow_flexion.non_regression",
                old_elbow_mean is not None
                and new_elbow_mean is not None
                and new_elbow_mean <= old_elbow_mean + ELBOW_FLEXION_MAX_REGRESSION_DEG,
                baseline_mean_deg=old_elbow_mean,
                candidate_mean_deg=new_elbow_mean,
                max_regression_deg=ELBOW_FLEXION_MAX_REGRESSION_DEG,
            )
        )

        old_availability = _mapping(old.get("availability"))
        new_availability = _mapping(new.get("availability"))
        old_holds = _integer(old_availability, "limb_held_frames")
        new_holds = _integer(new_availability, "limb_held_frames")
        checks.append(
            _check(
                f"{side}.limb_holds.non_regression",
                old_holds is not None and new_holds is not None and new_holds <= old_holds,
                baseline_frames=old_holds,
                candidate_frames=new_holds,
            )
        )

        new_clearance = _metric_summary(new, "predicted_clearance_m")
        new_clearance_min = _number(new_clearance, "min")
        checks.append(
            _check(
                f"{side}.clearance.at_sim_floor",
                new_clearance_min is not None and new_clearance_min >= SIM_CLEARANCE_FLOOR_M,
                candidate_min_m=new_clearance_min,
                sim_floor_m=SIM_CLEARANCE_FLOOR_M,
            )
        )

        old_residual = _metric_summary(old, "wrist_residual_m")
        new_residual = _metric_summary(new, "wrist_residual_m")
        old_residual_p95 = _number(old_residual, "p95")
        new_residual_p95 = _number(new_residual, "p95")
        checks.append(
            _check(
                f"{side}.wrist_residual.p95_non_regression",
                old_residual_p95 is not None
                and new_residual_p95 is not None
                and new_residual_p95 <= old_residual_p95 + WRIST_RESIDUAL_P95_TOLERANCE_M,
                baseline_p95_m=old_residual_p95,
                candidate_p95_m=new_residual_p95,
                allowed_increase_m=WRIST_RESIDUAL_P95_TOLERANCE_M,
            )
        )

    # Whole-frame decisions are reported redundantly under each arm because the
    # arm table is the report's analysis unit.  First require exact agreement;
    # then reject any candidate HOLD or FAULT count increase.
    for outcome in ("whole_frame_hold_frames", "whole_frame_fault_frames"):
        old_counts = [
            _integer(_mapping(_arm_document(baseline, side).get("availability")), outcome)
            for side in sorted(_SIDES)
        ]
        new_counts = [
            _integer(_mapping(_arm_document(candidate, side).get("availability")), outcome)
            for side in sorted(_SIDES)
        ]
        old_count = old_counts[0] if len(set(old_counts)) == 1 else None
        new_count = new_counts[0] if len(set(new_counts)) == 1 else None
        checks.append(
            _check(
                f"global.{outcome}.non_regression",
                old_count is not None and new_count is not None and new_count <= old_count,
                baseline_frames=old_count,
                candidate_frames=new_count,
            )
        )

    torso_unavailable = [
        _integer(
            _mapping(_arm_document(candidate, side).get("availability")),
            "torso_basis_unavailable_frames",
        )
        for side in sorted(_SIDES)
    ]
    checks.append(
        _check(
            "global.torso_basis.complete",
            len(set(torso_unavailable)) == 1 and torso_unavailable[0] == 0,
            candidate_unavailable_frames=(
                torso_unavailable[0] if len(set(torso_unavailable)) == 1 else None
            ),
        )
    )

    failed = [str(check["name"]) for check in checks if not check["passed"]]
    return {
        "policy_version": RELEASE_GATE_VERSION,
        "verdict": "PASS" if not failed else "FAIL",
        "promotion_allowed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "policy": {
            "min_usable_frames_per_arm": MIN_USABLE_FRAMES_PER_ARM,
            "min_direction_error_improvement_deg": MIN_DIRECTION_ERROR_IMPROVEMENT_DEG,
            "min_direction_error_improvement_ratio": MIN_DIRECTION_ERROR_IMPROVEMENT_RATIO,
            "elbow_flexion_max_regression_deg": ELBOW_FLEXION_MAX_REGRESSION_DEG,
            "sim_clearance_floor_m": SIM_CLEARANCE_FLOOR_M,
            "wrist_residual_p95_tolerance_m": WRIST_RESIDUAL_P95_TOLERANCE_M,
        },
    }


def _check(name: str, passed: bool, **evidence: object) -> dict[str, object]:
    return {"name": name, "passed": passed, "evidence": evidence}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _arm_document(document: Mapping[str, object], side: str) -> Mapping[str, object]:
    return _mapping(document.get(side))


def _metric_summary(arm: Mapping[str, object], name: str) -> Mapping[str, object]:
    return _mapping(arm.get(name))


def _number(document: Mapping[str, object], name: str) -> float | None:
    value = document.get(name)
    return float(value) if isinstance(value, (int, float)) and isfinite(float(value)) else None


def _integer(document: Mapping[str, object], name: str) -> int | None:
    value = document.get(name)
    return value if isinstance(value, int) and value >= 0 else None


def save_analysis(document: Mapping[str, object], path: Path) -> None:
    """Write a readable stable metrics artifact without overwriting implicitly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_clips(
    baseline: MotionClip,
    candidate: MotionClip,
    source_video: Path,
    source_frame_map: Path,
    timeline: Mapping[int, _SourceFrame],
) -> None:
    # Mapping hashes must differ: passing one clip twice is not an A/B experiment.
    if baseline.mapping_hash == candidate.mapping_hash:
        raise AnalysisProvenanceError("baseline and candidate must have different mapping_hash values")
    for field_name in ("model_hash", "tool_hash", "calibration_id", "input_hash", "implementation_hash"):
        baseline_value = getattr(baseline, field_name)
        candidate_value = getattr(candidate, field_name)
        if baseline_value is None or candidate_value is None or baseline_value != candidate_value:
            raise AnalysisProvenanceError(f"baseline/candidate {field_name} does not match")
    if baseline.source_replay != candidate.source_replay:
        raise AnalysisProvenanceError("baseline/candidate source replay provenance does not match")
    if not baseline.source_replay.publishable:
        raise AnalysisProvenanceError(
            "unknown, failed, or legacy source replay cannot support G6 analysis"
        )
    # The metrics below use the *currently verified* MuJoCo asset.  Accepting two
    # mutually consistent but stale model/tool identities would silently score a
    # different robot or safety envelope, so bind the clips to that exact runtime
    # before forward kinematics are allowed to run.
    if baseline.model_hash != CANONICAL_MANIFEST.fixed_mjcf_sha256:
        raise AnalysisProvenanceError("clip model_hash does not match the pinned MuJoCo model")
    if baseline.tool_hash != SupervisorPolicy().tool_hash:
        raise AnalysisProvenanceError("clip tool_hash does not match the current simulator safety policy")
    expected_input_hash = _analysis_identity(source_video, source_frame_map)
    if baseline.input_hash != expected_input_hash:
        raise AnalysisProvenanceError("clip input_hash does not bind the supplied raw video and frame map")
    baseline_frames = tuple((frame.source_sequence, frame.source_mono_ns) for frame in baseline.frames)
    candidate_frames = tuple((frame.source_sequence, frame.source_mono_ns) for frame in candidate.frames)
    if baseline_frames != candidate_frames:
        raise AnalysisProvenanceError("baseline/candidate source sequence or timestamp differs")
    for sequence, timestamp_ns in baseline_frames:
        mapped = timeline.get(sequence)
        if mapped is None or mapped.timestamp_ns != timestamp_ns:
            raise AnalysisProvenanceError(
                f"clip frame {sequence} is not exactly represented by the raw frame map"
            )


def _measure_clip(
    clip: MotionClip,
    sidecar: Mapping[int, Mapping[str, object]],
    model: mujoco.MjModel,
) -> dict[str, _ArmMetrics]:
    metrics = {side: _ArmMetrics() for side in _SIDES}
    for frame in clip.frames:
        sidecar_row = sidecar[frame.source_sequence]
        torso_available = _torso_basis_for_row(sidecar_row) is not None
        human = {side: _human_segments(sidecar_row, side) for side in _SIDES}
        robot = _robot_segments(frame, model) if frame.target is not None else None
        for side in _SIDES:
            metric = metrics[side]
            metric.source_frames += 1
            if not torso_available:
                metric.torso_basis_unavailable_frames += 1
            if human[side] is not None:
                metric.human_segment_frames += 1
            if frame.target is not None:
                metric.target_frames += 1
                metric.wrist_residual_m.append(frame.target.ik_residual_m)
                metric.clearance_m.append(frame.target.predicted_clearance_m)
            group = f"{side}_arm"
            held = group in frame.held_groups
            if held:
                metric.limb_held_frames += 1
                reason_map = dict(frame.held_group_reasons)
                metric.limb_hold_reasons[reason_map.get(group, "UNSPECIFIED")] += 1
                residual = dict(frame.held_group_residuals_m).get(group)
                if residual is not None:
                    metric.rejected_wrist_residual_m.append(residual)
            if frame.decision.outcome.value != "ALLOW":
                metric.whole_frame_nonallow_frames += 1
                metric.whole_frame_reasons.update(frame.decision.reasons)
                if frame.decision.outcome.value == "HOLD":
                    metric.whole_frame_hold_frames += 1
                elif frame.decision.outcome.value == "FAULT":
                    metric.whole_frame_fault_frames += 1
            if (
                human[side] is None
                or robot is None
                or held
                or frame.decision.outcome.value != "ALLOW"
            ):
                continue
            metric.live_command_frames += 1
            human_upper, human_forearm, human_flexion = human[side]
            robot_upper, robot_forearm, robot_flexion = robot[side]
            metric.upper_arm_error_deg.append(_angle_deg(human_upper, robot_upper))
            metric.forearm_error_deg.append(_angle_deg(human_forearm, robot_forearm))
            metric.elbow_flexion_error_deg.append(abs(degrees(human_flexion - robot_flexion)))
    return metrics


def _robot_segments(
    frame: ClipFrame, model: mujoco.MjModel
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float], float]]:
    if frame.target is None:
        raise ValueError("robot segments require a target")
    data = mujoco.MjData(model)
    for joint in frame.target.joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        if joint_id < 0:
            raise AnalysisProvenanceError(f"clip target contains unknown model joint {joint.name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = joint.position_rad
    mujoco.mj_forward(model, data)
    measured: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], float]] = {}
    for side in _SIDES:
        shoulder = _body_position(model, data, f"{side}_arm_link1")
        elbow = _body_position(model, data, f"{side}_arm_link4")
        wrist = _body_position(model, data, f"{side}_gripper_tcp_link")
        upper = _unit(_difference(elbow, shoulder))
        forearm = _unit(_difference(wrist, elbow))
        measured[side] = (upper, forearm, _angle_rad(upper, forearm))
    return measured


def _human_segments(
    row: Mapping[str, object], side: str
) -> tuple[tuple[float, float, float], tuple[float, float, float], float] | None:
    landmarks = row.get("landmarks")
    if not isinstance(landmarks, Mapping):
        return None
    arm_entries = [landmarks.get(name) for name in _SIDES[side]]
    torso_entries = [landmarks.get(name) for name in _TORSO]
    entries = arm_entries + torso_entries
    if any(not isinstance(entry, Mapping) for entry in entries):
        return None
    typed = [entry for entry in entries if isinstance(entry, Mapping)]
    if any(_confidence(entry) < _MIN_LANDMARK_CONFIDENCE for entry in typed):
        return None
    coordinates = _common_coordinates(typed)
    if coordinates is None:
        return None
    shoulder, elbow, wrist, left_shoulder, right_shoulder, left_hip, right_hip = coordinates
    try:
        torso = _torso_basis(left_shoulder, right_shoulder, left_hip, right_hip)
        upper = _human_to_robot(_project_torso_direction(shoulder, elbow, torso))
        forearm = _human_to_robot(_project_torso_direction(elbow, wrist, torso))
    except ValueError:
        return None
    return upper, forearm, _angle_rad(upper, forearm)


def _torso_basis_for_row(
    row: Mapping[str, object],
) -> _TorsoBasis | None:
    landmarks = row.get("landmarks")
    if not isinstance(landmarks, Mapping):
        return None
    entries = [landmarks.get(name) for name in _TORSO]
    if any(not isinstance(entry, Mapping) for entry in entries):
        return None
    typed = [entry for entry in entries if isinstance(entry, Mapping)]
    if any(_confidence(entry) < _MIN_LANDMARK_CONFIDENCE for entry in typed):
        return None
    coordinates = _common_coordinates(typed)
    if coordinates is None:
        return None
    try:
        return _torso_basis(*coordinates)
    except ValueError:
        return None


def _common_coordinates(
    landmarks: list[Mapping[str, object]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    for key in ("world_xyz_m", "normalized_xyz"):
        parsed = [_xyz(landmark.get(key)) for landmark in landmarks]
        if all(value is not None for value in parsed):
            return tuple(parsed)  # type: ignore[return-value]
    return None


def _confidence(landmark: Mapping[str, object]) -> float:
    visibility = landmark.get("visibility")
    presence = landmark.get("presence")
    if not isinstance(visibility, (int, float)) or not isinstance(presence, (int, float)):
        return 0.0
    return min(float(visibility), float(presence))


def _xyz(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if not all(isinstance(component, (int, float)) and isfinite(float(component)) for component in value):
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _torso_basis(
    left_shoulder: _Vector3,
    right_shoulder: _Vector3,
    left_hip: _Vector3,
    right_hip: _Vector3,
) -> _TorsoBasis:
    """Build the candidate mapper's signed orthonormal torso basis.

    This intentionally duplicates the v3 mapping semantics rather than scoring
    in raw camera axes: lateral comes from the shoulders, inferior from the
    shoulder-to-hip midpoint after Gram-Schmidt, and the torso normal uses the
    camera depth only to resolve mirror handedness.  Any missing or degenerate
    torso leaves the frame unscored; the release gate then fails closed on its
    unavailable metric evidence and the candidate's corresponding limb hold.
    """
    lateral = _unit(_difference(left_shoulder, right_shoulder))
    shoulder_midpoint = tuple(
        (left_shoulder[index] + right_shoulder[index]) / 2.0 for index in range(3)
    )
    hip_midpoint = tuple((left_hip[index] + right_hip[index]) / 2.0 for index in range(3))
    inferior_raw = _difference(hip_midpoint, shoulder_midpoint)
    lateral_component = _dot(inferior_raw, lateral)
    inferior = _unit(
        tuple(
            inferior_raw[index] - lateral_component * lateral[index] for index in range(3)
        )
    )
    forward = _unit(_cross(lateral, inferior))
    if abs(forward[2]) < 1e-4:
        raise ValueError("torso normal has ambiguous camera-depth sign")
    if forward[2] < 0.0:
        forward = tuple(-component for component in forward)
    return lateral, inferior, forward


def _project_torso_direction(
    origin: _Vector3,
    target: _Vector3,
    torso: _TorsoBasis,
) -> _Vector3:
    direction = _unit(_difference(target, origin))
    return _unit(tuple(_dot(direction, axis) for axis in torso))


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _human_to_robot(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    # Keep the evaluator's torso-relative transform identical to the candidate:
    # human (anatomical-left, inferior, signed torso-normal) -> robot (-z, x, -y).
    return (-vector[2], vector[0], -vector[1])


def _body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[float, float, float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"pinned model does not contain {name}")
    return tuple(float(value) for value in data.xpos[body_id])


def _difference(
    right: tuple[float, float, float], left: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(right[index] - left[index] for index in range(3))  # type: ignore[return-value]


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    magnitude = sqrt(sum(component * component for component in vector))
    if not isfinite(magnitude) or magnitude <= 1e-12:
        raise ValueError("zero-length direction")
    return tuple(component / magnitude for component in vector)  # type: ignore[return-value]


def _angle_rad(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return acos(max(-1.0, min(1.0, dot)))


def _angle_deg(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return degrees(_angle_rad(left, right))


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _metric_delta(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, float | None]:
    names = (
        "upper_arm_angle_error_deg",
        "forearm_angle_error_deg",
        "elbow_flexion_error_deg",
        "wrist_residual_m",
        "predicted_clearance_m",
    )
    result: dict[str, float | None] = {}
    for name in names:
        baseline_summary = baseline[name]
        candidate_summary = candidate[name]
        if not isinstance(baseline_summary, Mapping) or not isinstance(candidate_summary, Mapping):
            raise AssertionError("metric document schema drift")
        old = baseline_summary.get("mean")
        new = candidate_summary.get("mean")
        result[f"{name}_mean"] = (
            float(new) - float(old)
            if isinstance(old, (int, float)) and isinstance(new, (int, float))
            else None
        )
    return result


def _clip_identity(clip: MotionClip) -> dict[str, str | None]:
    return {
        "clip_id": clip.clip_id,
        "mapping_hash": clip.mapping_hash,
        "implementation_hash": clip.implementation_hash,
    }


def _load_frame_map(path: Path, source_video: Path) -> dict[int, _SourceFrame]:
    document = _read_object(path, "source frame map")
    if document.get("schema_version") != 3:
        raise AnalysisProvenanceError(
            "source frame map must be a current v3 successful capture, not legacy evidence"
        )
    if document.get("artifact_kind") != "raw-camera":
        raise AnalysisProvenanceError("source frame map is not a raw-camera artifact")
    if document.get("capture_outcome") != "succeeded":
        raise AnalysisProvenanceError("source frame map is not a successful capture")
    provenance = document.get("capture_provenance")
    if not isinstance(provenance, Mapping) or not provenance.get("attempt_id"):
        raise AnalysisProvenanceError("source frame map lacks capture attempt provenance")
    if document.get("video") != source_video.name:
        raise AnalysisProvenanceError("source frame map names a different source video")
    if document.get("video_sha256") != _file_sha256(source_video):
        raise AnalysisProvenanceError("source frame map video_sha256 does not match source video")
    rows = document.get("frames")
    if not isinstance(rows, list) or not rows:
        raise AnalysisProvenanceError("source frame map must contain frames")
    if document.get("video_frame_count") != len(rows):
        raise AnalysisProvenanceError("source frame map video_frame_count does not match rows")
    result: dict[int, _SourceFrame] = {}
    previous_sequence = -1
    previous_timestamp = -1
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("video_frame_index") != index:
            raise AnalysisProvenanceError("source frame map video_frame_index must be contiguous")
        sequence = row.get("source_sequence")
        timestamp = row.get("source_mono_ns")
        if (
            not isinstance(sequence, int)
            or not isinstance(timestamp, int)
            or sequence <= previous_sequence
            or timestamp <= previous_timestamp
        ):
            raise AnalysisProvenanceError("source frame map source times must strictly increase")
        result[sequence] = _SourceFrame(sequence, timestamp)
        previous_sequence, previous_timestamp = sequence, timestamp
    return result


def _load_landmark_sidecar(
    path: Path,
    source_video: Path,
    source_frame_map: Path,
    timeline: Mapping[int, _SourceFrame],
) -> dict[int, Mapping[str, object]]:
    document = _read_object(path, "landmark sidecar")
    if document.get("schema_version") != 3:
        raise AnalysisProvenanceError(
            "landmark sidecar must be a current v3 capture, not legacy evidence"
        )
    if document.get("artifact_kind") != "raw-camera-landmarks":
        raise AnalysisProvenanceError("landmark sidecar is not a raw-camera landmark artifact")
    if document.get("source_video") != source_video.name:
        raise AnalysisProvenanceError("landmark sidecar names a different source video")
    if document.get("source_video_sha256") != _file_sha256(source_video):
        raise AnalysisProvenanceError("landmark sidecar source_video_sha256 does not match source video")
    if document.get("source_frame_map") != source_frame_map.name:
        raise AnalysisProvenanceError("landmark sidecar names a different source frame map")
    if document.get("source_frame_map_sha256") != _file_sha256(source_frame_map):
        raise AnalysisProvenanceError("landmark sidecar source_frame_map_sha256 does not match map")
    rows = document.get("frames")
    if not isinstance(rows, list) or not rows:
        raise AnalysisProvenanceError("landmark sidecar must contain frames")
    if document.get("source_video_frame_count") != len(rows):
        raise AnalysisProvenanceError("landmark sidecar source_video_frame_count does not match rows")
    result: dict[int, Mapping[str, object]] = {}
    previous_sequence = -1
    for row in rows:
        if not isinstance(row, Mapping):
            raise AnalysisProvenanceError("landmark sidecar frame must be an object")
        sequence = row.get("source_sequence")
        timestamp = row.get("source_mono_ns")
        if not isinstance(sequence, int) or not isinstance(timestamp, int) or sequence <= previous_sequence:
            raise AnalysisProvenanceError("landmark sidecar source_sequence must strictly increase")
        mapped = timeline.get(sequence)
        if mapped is None or mapped.timestamp_ns != timestamp:
            raise AnalysisProvenanceError("landmark sidecar does not exactly match source frame map")
        result[sequence] = row
        previous_sequence = sequence
    if set(result) != set(timeline):
        raise AnalysisProvenanceError("landmark sidecar and source frame map do not cover identical frames")
    return result


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        document = loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AnalysisProvenanceError(f"invalid {label}: {path}") from error
    if not isinstance(document, dict):
        raise AnalysisProvenanceError(f"{label} root must be an object")
    return document


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_identity(source_video: Path, frame_map: Path) -> str:
    digest = sha256()
    digest.update(_file_sha256(source_video).encode("ascii"))
    digest.update(_file_sha256(frame_map).encode("ascii"))
    return digest.hexdigest()


def _sequence_hash(sequences: Iterable[int]) -> str:
    return sha256(dumps(list(sequences), separators=(",", ":")).encode("utf-8")).hexdigest()
