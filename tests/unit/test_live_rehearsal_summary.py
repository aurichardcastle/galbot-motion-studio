"""Regression coverage for the retained-live-artifact summary tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _summary_module():
    path = Path(__file__).parents[2] / "tools" / "live_rehearsal_summary.py"
    spec = importlib.util.spec_from_file_location("live_rehearsal_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_reports_command_and_readback_excursion_by_group() -> None:
    summary_tool = _summary_module()
    allow = SimpleNamespace(outcome=SimpleNamespace(value="ALLOW"), reasons=())
    clip = SimpleNamespace(
        clip_id="clip",
        task="mirror",
        source="camera",
        motion_profile="sim",
        frames=(
            SimpleNamespace(
                decision=allow,
                source_sequence=1,
                source_mono_ns=1_000,
                target=SimpleNamespace(
                    joints=(
                        SimpleNamespace(name="left_arm_joint1", position_rad=0.2),
                        SimpleNamespace(name="right_arm_joint1", position_rad=-0.4),
                        SimpleNamespace(name="leg_joint4", position_rad=0.0),
                    )
                ),
                observed_joints_rad=(
                    ("left_arm_joint1", 0.1),
                    ("right_arm_joint1", -0.3),
                    ("leg_joint4", 0.0),
                ),
                held_groups=(),
                held_group_reasons=(),
            ),
            SimpleNamespace(
                decision=allow,
                source_sequence=2,
                source_mono_ns=3_000,
                target=SimpleNamespace(
                    joints=(
                        SimpleNamespace(name="left_arm_joint1", position_rad=0.7),
                        SimpleNamespace(name="right_arm_joint1", position_rad=-0.1),
                        SimpleNamespace(name="leg_joint4", position_rad=0.2),
                    )
                ),
                observed_joints_rad=(
                    ("left_arm_joint1", 0.5),
                    ("right_arm_joint1", -0.2),
                    ("leg_joint4", 0.1),
                ),
                held_groups=(),
                held_group_reasons=(),
            ),
        ),
    )

    motion = summary_tool.summarise(clip)["motion"]

    assert motion["command_target_frames"] == 2
    assert motion["observed_readback_frames"] == 2
    assert motion["commanded_joint_ranges_rad"] == {
        "left_arm": {
            "sum_joint_range_rad": 0.49999999999999994,
            "per_joint_range_rad": {"left_arm_joint1": 0.49999999999999994},
        },
        "right_arm": {
            "sum_joint_range_rad": 0.30000000000000004,
            "per_joint_range_rad": {"right_arm_joint1": 0.30000000000000004},
        },
        "torso": {
            "sum_joint_range_rad": 0.2,
            "per_joint_range_rad": {"leg_joint4": 0.2},
        },
    }
    assert motion["observed_joint_ranges_rad"]["left_arm"]["sum_joint_range_rad"] == 0.4
