"""Name-addressed canonical G1 control groups and URDF limit extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


CONTROL_GROUPS: dict[str, tuple[str, ...]] = {
    "leg": tuple(f"leg_joint{index}" for index in range(1, 6)),
    "head": ("head_joint1", "head_joint2"),
    "left_arm": tuple(f"left_arm_joint{index}" for index in range(1, 8)),
    "left_gripper": ("left_gripper_joint",),
    "right_arm": tuple(f"right_arm_joint{index}" for index in range(1, 8)),
    "right_gripper": ("right_gripper_joint",),
}


@dataclass(frozen=True)
class JointLimit:
    lower_rad: float
    upper_rad: float
    velocity_rad_s: float
    effort_nm: float


def controlled_joint_names() -> tuple[str, ...]:
    return tuple(name for group in CONTROL_GROUPS.values() for name in group)


def parse_revolute_joint_limits(urdf_path: Path) -> dict[str, JointLimit]:
    root = ElementTree.parse(urdf_path).getroot()
    limits: dict[str, JointLimit] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        name = joint.get("name")
        if name is None:
            continue
        limits[name] = JointLimit(
            lower_rad=float(limit.attrib["lower"]),
            upper_rad=float(limit.attrib["upper"]),
            velocity_rad_s=float(limit.attrib["velocity"]),
            effort_nm=float(limit.attrib["effort"]),
        )
    return limits

