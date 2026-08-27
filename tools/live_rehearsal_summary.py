#!/usr/bin/env python3
"""Summarise one retained live-preview clip without replaying or altering it.

Usage:

    PYTHONPATH=src .venv/bin/python \
      tools/live_rehearsal_summary.py artifacts/rehearsal-20260825-live/live.json \
      --terminal-log artifacts/rehearsal-20260825-live/terminal.log \
      --output artifacts/rehearsal-20260825-live/live-summary.json

The clip is authoritative for recorded decisions and groups.  When the retained
terminal log is supplied, the summary also copies its final live ``realtime:``
line as terminal-reported runner metrics; it never infers frame rate or latency
from clip timestamps.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from galbot_motion_studio.recording import MotionClip


_REALTIME_LINE = re.compile(
    r"^realtime: processed=(?P<processed>\d+) "
    r"dropped=(?P<dropped>\d+) "
    r"mean=(?P<mean_ms>\d+(?:\.\d+)?)ms "
    r"p95=(?P<p95_ms>\d+(?:\.\d+)?)ms "
    r"capacity=(?P<capacity_fps>\d+(?:\.\d+)?)fps$",
    re.MULTILINE,
)


def _nested_counts(values: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        group: dict(sorted(counts.items()))
        for group, counts in sorted(values.items())
    }


def _motion_group(joint_name: str) -> str:
    """Return the reporting group for a recorded command/readback joint."""
    if joint_name.startswith("head_"):
        return "head"
    if joint_name.startswith("left_arm_"):
        return "left_arm"
    if joint_name.startswith("right_arm_"):
        return "right_arm"
    if joint_name == "left_gripper_joint":
        return "left_gripper"
    if joint_name == "right_gripper_joint":
        return "right_gripper"
    if joint_name == "leg_joint4":
        return "torso"
    return "other"


def _joint_ranges(values: dict[str, list[float]]) -> dict[str, Any]:
    """Summarise joint excursion without turning it into a tracking threshold."""
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for name, samples in values.items():
        if samples:
            grouped[_motion_group(name)][name] = max(samples) - min(samples)
    return {
        group: {
            "sum_joint_range_rad": sum(ranges.values()),
            "per_joint_range_rad": dict(sorted(ranges.items())),
        }
        for group, ranges in sorted(grouped.items())
    }


def _terminal_realtime_metrics(path: Path) -> dict[str, int | float]:
    """Parse the final completed live-run metric line, or fail closed.

    A saved-video ``analysis-sync:`` line is intentionally not accepted here:
    it does not exercise the webcam latest-wins path or its freshness gate.
    """
    matches = list(_REALTIME_LINE.finditer(path.read_text(encoding="utf-8")))
    if not matches:
        raise ValueError(
            f"{path} has no final realtime metric line; retain the output from a "
            "successful live webcam preview"
        )
    values = matches[-1].groupdict()
    return {
        "processed": int(values["processed"]),
        "dropped_before_processing": int(values["dropped"]),
        "mean_processing_ms": float(values["mean_ms"]),
        "p95_processing_ms": float(values["p95_ms"]),
        "effective_fps": float(values["capacity_fps"]),
    }


def summarise(
    clip: MotionClip, *, terminal_realtime: dict[str, int | float] | None = None
) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    whole_frame_reasons: Counter[str] = Counter()
    held_groups: Counter[str] = Counter()
    held_group_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    source_sequences: list[int] = []
    source_times: list[int] = []
    commanded_positions: dict[str, list[float]] = defaultdict(list)
    observed_positions: dict[str, list[float]] = defaultdict(list)
    target_frames = 0
    observed_frames = 0

    for frame in clip.frames:
        decision = frame.decision
        outcomes[decision.outcome.value] += 1
        source_sequences.append(frame.source_sequence)
        source_times.append(frame.source_mono_ns)
        if frame.target is not None:
            target_frames += 1
            for joint in frame.target.joints:
                commanded_positions[joint.name].append(joint.position_rad)
        if frame.observed_joints_rad is not None:
            observed_frames += 1
            for name, position_rad in frame.observed_joints_rad:
                observed_positions[name].append(position_rad)
        if decision.outcome.value != "ALLOW":
            whole_frame_reasons.update(decision.reasons)
        reasons = dict(frame.held_group_reasons)
        for group in frame.held_groups:
            held_groups[group] += 1
            held_group_reasons[group][reasons.get(group, "UNSPECIFIED")] += 1

    source_span_ns = (
        max(source_times) - min(source_times) if len(source_times) > 1 else 0
    )
    stale_count = sum(
        count
        for reason, count in whole_frame_reasons.items()
        if "STALE" in reason.upper()
    )
    summary: dict[str, Any] = {
        "artifact": {
            "clip_id": clip.clip_id,
            "task": clip.task,
            "source": clip.source,
            "motion_profile": clip.motion_profile,
            "recorded_frames": len(clip.frames),
            "source_sequence_first": min(source_sequences, default=None),
            "source_sequence_last": max(source_sequences, default=None),
            "recorded_source_span_ms": source_span_ns / 1_000_000,
            "source_replay": (
                clip.source_replay.model_dump(mode="json")
                if hasattr(clip, "source_replay")
                else {"origin": "unknown", "note": "legacy clip lacked source provenance"}
            ),
        },
        "decisions": {
            "outcomes": dict(sorted(outcomes.items())),
            "whole_frame_reasons": dict(sorted(whole_frame_reasons.items())),
            "whole_frame_stale_count": stale_count,
        },
        "holds": {
            "per_group": dict(sorted(held_groups.items())),
            "per_group_reasons": _nested_counts(held_group_reasons),
        },
        "motion": {
            "command_target_frames": target_frames,
            "observed_readback_frames": observed_frames,
            "commanded_joint_ranges_rad": _joint_ranges(commanded_positions),
            "observed_joint_ranges_rad": _joint_ranges(observed_positions),
        },
        "interpretation": {
            "recorded_source_span_ms": (
                "Capture-time span of retained command records; not display fps or latency."
            ),
            "whole_frame_stale_count": (
                "Count of recorded whole-frame decision reasons containing STALE."
            ),
            "per_group": (
                "Per-group holds can coexist with whole-frame ALLOW; do not call an "
                "ALLOW-only clip 'zero degraded frames'."
            ),
            "source_replay": (
                "Origin/admission provenance for the clip. Only an explicit live capture "
                "or v3 successful recorded replay is publishable evidence."
            ),
            "motion": (
                "Summed joint ranges are descriptive command/readback excursion, "
                "not an accuracy threshold or a replacement for operator judgement. "
                "Read them alongside live runner metrics so a high fps result cannot "
                "hide a nearly stationary commanded arm."
            ),
        },
    }
    if terminal_realtime is not None:
        summary["realtime"] = terminal_realtime
        summary["interpretation"]["realtime"] = (
            "Final metrics emitted by the live latest-wins runner in terminal.log; "
            "not inferred from recorded source timestamps."
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="retained live MotionClip JSON")
    parser.add_argument(
        "--terminal-log",
        type=Path,
        help="retained live terminal.log; adds its final realtime runner metrics",
    )
    parser.add_argument("--output", type=Path, help="write JSON as well as stdout")
    args = parser.parse_args()

    terminal_realtime = (
        _terminal_realtime_metrics(args.terminal_log)
        if args.terminal_log is not None
        else None
    )
    summary = summarise(MotionClip.load(args.clip), terminal_realtime=terminal_realtime)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
