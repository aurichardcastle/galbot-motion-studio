from galbot_motion_studio.model.joint_map import (
    CONTROL_GROUPS,
    controlled_joint_names,
    parse_revolute_joint_limits,
)
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST


def test_canonical_control_surface_is_name_addressed() -> None:
    names = controlled_joint_names()
    assert len(names) == 23
    assert len(names) == len(set(names))
    limits = parse_revolute_joint_limits(CANONICAL_MANIFEST.fixed_urdf)
    assert set(names).issubset(limits)
    assert limits["head_joint2"].lower_rad == -0.2143461
    assert limits["head_joint2"].upper_rad == 0.4935988
    assert limits["left_arm_joint4"].lower_rad < 0 < limits["left_arm_joint4"].upper_rad


def test_control_groups_have_explicit_cardinality() -> None:
    assert {group: len(names) for group, names in CONTROL_GROUPS.items()} == {
        "leg": 5,
        "head": 2,
        "left_arm": 7,
        "left_gripper": 1,
        "right_arm": 7,
        "right_gripper": 1,
    }
