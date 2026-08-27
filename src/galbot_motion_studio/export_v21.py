"""Honest LeRobotDataset v2.1 writer preserving one row per source event."""

from __future__ import annotations

from json import dumps
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from galbot_motion_studio.contracts.core import SafetyOutcome
from galbot_motion_studio.recording import ClipFrame, MotionClip
from galbot_motion_studio.safety.profiles import clearance_floor_for, describe


class ExportError(RuntimeError):
    pass


CONTROL_GROUP_NAMES = ("head", "left_arm", "right_arm")


def export_lerobot_v21(
    clips: Iterable[MotionClip],
    destination: Path,
    *,
    fps: int = 30,
) -> Path:
    """Export clips without inventing motion across HOLD frames.

    The latest approved command is repeated only for an explicit HOLD event; no
    interpolated motion or duplicate task is invented. Dataset timestamps use the
    declared fixed output cadence while ``source_sequence`` preserves raw ordering.
    Lossless source clips are embedded under ``meta/source_clips``.
    """
    clips = tuple(clips)
    if not clips:
        raise ExportError("at least one clip is required")
    if fps <= 0 or fps > 240:
        raise ExportError("fps must be in [1, 240]")
    if destination.exists() and any(destination.iterdir()):
        raise ExportError("destination must be absent or empty")
    joint_order = clips[0].joint_order
    if any(clip.joint_order != joint_order for clip in clips):
        raise ExportError("all clips must use the same joint order")
    if any(not clip.source_replay.publishable for clip in clips):
        raise ExportError(
            "unknown or diagnostic saved-video sources cannot be exported as dataset evidence"
        )
    provenance_fields = (
        "model_hash",
        "tool_hash",
        "mapping_hash",
        "motion_profile",
        "liveness",
        "source_replay",
        "calibration_window",
        "occupancy",
    )
    for field in provenance_fields:
        if len({getattr(clip, field) for clip in clips}) != 1:
            raise ExportError(f"all clips must use the same {field}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.rmdir()
    with TemporaryDirectory(
        prefix=f".{destination.name}-staging-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / destination.name
        _write_dataset(clips, staging, fps=fps, joint_order=joint_order)
        os.replace(staging, destination)
    return destination


def _write_dataset(
    clips: tuple[MotionClip, ...],
    destination: Path,
    *,
    fps: int,
    joint_order: tuple[str, ...],
) -> None:

    import pandas as pd

    meta = destination / "meta"
    data = destination / "data/chunk-000"
    sources = meta / "source_clips"
    data.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    task_names = tuple(dict.fromkeys(clip.task for clip in clips))
    task_indices = {task: index for index, task in enumerate(task_names)}
    episodes: list[dict[str, object]] = []
    stats: list[dict[str, object]] = []
    total_frames = 0
    held_total = 0
    held_group_totals = {name: 0 for name in CONTROL_GROUP_NAMES}

    for episode_index, clip in enumerate(clips):
        rows, held_count, held_group_counts = _resample_clip(
            clip,
            episode_index=episode_index,
            task_index=task_indices[clip.task],
            global_start=total_frames,
            fps=fps,
        )
        pd.DataFrame(rows).to_parquet(
            data / f"episode_{episode_index:06d}.parquet",
            index=False,
        )
        length = len(rows)
        episodes.append(
            {"episode_index": episode_index, "tasks": [clip.task], "length": length}
        )
        stats.append({"episode_index": episode_index, "stats": _episode_stats(rows)})
        clip.save(sources / f"clip_{episode_index:06d}.json")
        total_frames += length
        held_total += held_count
        for name, count in held_group_counts.items():
            held_group_totals[name] += count

    info = {
        "codebase_version": "v2.1",
        "total_episodes": len(clips),
        "total_frames": total_frames,
        "total_tasks": len(task_names),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "robot_type": "galbot-g1-sim",
        "fps": fps,
        "splits": {"train": f"0:{len(clips)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "action": {
                "dtype": "float32", "shape": [len(joint_order)], "names": list(joint_order)
            },
            "observation.state": {
                "dtype": "float32", "shape": [len(joint_order)], "names": list(joint_order)
            },
            "held": {"dtype": "bool", "shape": [1], "names": None},
            **{
                f"held_{name}": {"dtype": "bool", "shape": [1], "names": None}
                for name in CONTROL_GROUP_NAMES
            },
            "source_sequence": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta / "info.json").write_text(dumps(info, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(meta / "episodes.jsonl", episodes)
    _write_jsonl(meta / "episodes_stats.jsonl", stats)
    _write_jsonl(
        meta / "tasks.jsonl",
        ({"task_index": index, "task": task} for index, task in enumerate(task_names)),
    )
    provenance = {
        "source": "sim",
        "state_equals_action": True,
        "joint_order": joint_order,
        "model_hash": clips[0].model_hash,
        "tool_hash": clips[0].tool_hash,
        "mapping_hashes": sorted({clip.mapping_hash for clip in clips}),
        "motion_profile": clips[0].motion_profile,
        "liveness": clips[0].liveness.model_dump(mode="json"),
        "source_replay": clips[0].source_replay.model_dump(mode="json"),
        "calibration_window": clips[0].calibration_window.model_dump(mode="json"),
        "calibration_windows": [
            {
                "clip_id": clip.clip_id,
                "evidence": (
                    None
                    if clip.calibration_window_evidence is None
                    else clip.calibration_window_evidence.model_dump(mode="json")
                ),
            }
            for clip in clips
        ],
        "occupancy": clips[0].occupancy.model_dump(mode="json"),
        # Terminal ingress faults retain their true (potentially regressed)
        # source timestamp in the lossless clip rather than corrupting v2.1's
        # strictly increasing event table. Surface them in export provenance so
        # a consumer cannot mistake an abruptly ended episode for a clean one.
        "terminal_faults": [
            {
                "clip_id": clip.clip_id,
                "decision": clip.terminal_fault.model_dump(mode="json"),
            }
            for clip in clips
            if clip.terminal_fault is not None
        ],
        # The profile is the recorded fact; the envelope is derived from it, and
        # a dataset consumer has no import of this package to derive it with.
        # Snapshot it here so the file states the limits the clips were approved
        # under, including the self-clearance floor, which differs by profile.
        "motion_envelope": describe(clips[0].motion_profile),
        "clearance_floor_m": clearance_floor_for(clips[0].motion_profile),
        "held_frames": held_total,
        "held_group_frames": held_group_totals,
        "resampling": "one row per raw event; repeat latest approval only on explicit HOLD",
    }
    (meta / "motion_studio.json").write_text(
        dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def load_exported_source_clips(destination: Path) -> tuple[MotionClip, ...]:
    return tuple(
        MotionClip.load(path)
        for path in sorted((destination / "meta/source_clips").glob("clip_*.json"))
    )


def _resample_clip(
    clip: MotionClip,
    *,
    episode_index: int,
    task_index: int,
    global_start: int,
    fps: int,
) -> tuple[list[dict[str, object]], int, dict[str, int]]:
    approved = [
        frame
        for frame in clip.frames
        if frame.target is not None and frame.decision.outcome is SafetyOutcome.ALLOW
    ]
    if not approved:
        raise ExportError(f"clip {clip.clip_id} has no approved target")
    latest_approved: ClipFrame | None = None
    held_count = 0
    held_group_counts = {name: 0 for name in CONTROL_GROUP_NAMES}
    rows: list[dict[str, object]] = []
    for event in clip.frames:
        if event.target is not None and event.decision.outcome is SafetyOutcome.ALLOW:
            latest_approved = event
        if latest_approved is None or latest_approved.target is None:
            continue
        held = event.decision.outcome is not SafetyOutcome.ALLOW
        held_count += int(held)
        held_groups = set(event.held_groups)
        if held:
            # A whole-frame HOLD is necessarily a hold of every control group.
            # Legacy clips do not carry per-group data, but their global HOLD is
            # still unambiguous and must not look healthy in a new export.
            held_groups.update(CONTROL_GROUP_NAMES)
        held_by_group = {
            name: name in held_groups for name in CONTROL_GROUP_NAMES
        }
        for name, is_held in held_by_group.items():
            held_group_counts[name] += int(is_held)
        action_by_name = {
            joint.name: joint.position_rad for joint in latest_approved.target.joints
        }
        action = [action_by_name[name] for name in clip.joint_order]
        observed = dict(latest_approved.observed_joints_rad or ())
        state = [observed.get(name, action_by_name[name]) for name in clip.joint_order]
        rows.append(
            {
                "action": action,
                "observation.state": state,
                "held": held,
                **{f"held_{name}": value for name, value in held_by_group.items()},
                "source_sequence": event.source_sequence,
                "timestamp": len(rows) / fps,
                "frame_index": len(rows),
                "episode_index": episode_index,
                "index": global_start + len(rows),
                "task_index": task_index,
            }
        )
    if not rows:
        raise ExportError(f"clip {clip.clip_id} produced no export frames")
    return rows, held_count, held_group_counts


def _episode_stats(rows: list[dict[str, object]]) -> dict[str, dict[str, list[object]]]:
    import numpy as np

    result: dict[str, dict[str, list[object]]] = {}
    for key in (
        "action",
        "observation.state",
        "held",
        *(f"held_{name}" for name in CONTROL_GROUP_NAMES),
        "source_sequence",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ):
        values = np.asarray([row[key] for row in rows])
        if values.ndim == 1:
            values = values[:, None]
        numeric = values.astype(np.float64)
        result[key] = {
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "mean": np.mean(numeric, axis=0).tolist(),
            "std": np.std(numeric, axis=0).tolist(),
            "count": [len(rows)],
        }
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")
