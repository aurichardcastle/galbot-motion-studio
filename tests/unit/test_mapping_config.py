"""MAPPING_CONFIG is hashed into every recording but drives nothing.

The pipeline builds ``LeftArmRetargeter``/``RightArmRetargeter`` with no policy
argument and ``HeadMappingPolicy`` with only its two joint limits, so the values
that actually retarget are the dataclass defaults.  If MAPPING_CONFIG drifts from
them, a dataset attests to parameters that were never used.  These tests import
both sides and compare dynamically so the two cannot separate.
"""

from dataclasses import MISSING, fields
from inspect import signature
from hashlib import sha256
from json import dumps

import pytest

from galbot_motion_studio.pipeline import (
    MAPPING_CONFIG,
    MAPPING_HASH,
    mapping_config_for,
    mapping_hash_for,
)
from galbot_motion_studio.retarget.head import HeadMappingPolicy
from galbot_motion_studio.retarget.torso import (
    TorsoMappingPolicy,
    shoulder_yaw_signal,
)
from galbot_motion_studio.retarget.left_arm import LeftArmPolicy


# Key path inside a MAPPING_CONFIG section -> the policy attribute it documents.
ARM_KEYS: dict[tuple[str, ...], str] = {
    ("forward_offset_m",): "forward_offset_m",
    ("lateral_scale_m",): "lateral_scale_m",
    ("vertical_scale_m",): "vertical_scale_m",
    ("depth_scale_m",): "depth_scale_m",
    ("depth_forward_scale_m",): "depth_forward_scale_m",
    ("elbow_scale_m",): "elbow_scale_m",
    ("elbow_hint", "weight"): "elbow_weight",
    ("tcp_orientation", "weight"): "orientation_weight",
    ("ik", "residual_tolerance_m"): "residual_tolerance_m",
    ("ik", "damping"): "damping",
    ("ik", "conditioned_retry_damping"): "conditioned_retry_damping",
    ("max_shoulder_normalized_signal",): "max_signal",
    ("ik_joint_limit_margin_rad",): "soft_margin_rad",
}
HEAD_KEYS: dict[tuple[str, ...], str] = {
    ("yaw_gain",): "yaw_gain",
    ("pitch_up_gain",): "pitch_up_gain",
    ("pitch_down_gain",): "pitch_down_gain",
    ("max_rate_rad_s",): "max_rate_rad_s",
}
TORSO_KEYS: dict[tuple[str, ...], str] = {
    ("yaw_gain",): "yaw_gain",
    ("deadband_rad",): "deadband_rad",
    ("max_camera_yaw_rad",): "max_camera_yaw_rad",
    ("max_relative_yaw_rad",): "max_relative_yaw_rad",
    ("max_neutral_camera_yaw_rad",): "max_neutral_camera_yaw_rad",
    ("max_neutral_spread_rad",): "max_neutral_spread_rad",
    ("max_observation_gap_ns",): "max_observation_gap_ns",
    ("max_observation_step_rad",): "max_observation_step_rad",
    ("max_rate_rad_s",): "max_rate_rad_s",
    # Not a TorsoMappingPolicy field: it is the shoulder_yaw_signal guard, and it
    # is pinned separately below for the same reason the others are pinned here.
}


def declared_defaults(policy_type: type) -> dict[str, object]:
    """Field defaults read without instantiating: HeadMappingPolicy needs joint limits."""
    return {
        field.name: field.default
        for field in fields(policy_type)
        if field.default is not MISSING
    }


def config_value(section: str, path: tuple[str, ...]) -> object:
    value: object = MAPPING_CONFIG[section]
    for key in path:
        value = value[key]
    return value


def numeric_paths(node: object, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Every numeric leaf under a section. Booleans are flags, not tuned values."""
    paths: set[tuple[str, ...]] = set()
    if isinstance(node, dict):
        for key, child in node.items():
            paths |= numeric_paths(child, prefix + (key,))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        paths.add(prefix)
    return paths


def fingerprint(config: object) -> str:
    """The pipeline's own MAPPING_HASH construction, re-run here."""
    return sha256(
        dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(("path", "attribute"), sorted(ARM_KEYS.items()))
def test_arm_fingerprint_values_match_live_left_arm_policy_defaults(
    path: tuple[str, ...], attribute: str
) -> None:
    defaults = declared_defaults(LeftArmPolicy)
    assert attribute in defaults, f"LeftArmPolicy has no defaulted field {attribute!r}"
    assert config_value("arms", path) == pytest.approx(defaults[attribute]), (
        f"MAPPING_CONFIG['arms']{list(path)} is hashed into recordings, but "
        f"LeftArmPolicy.{attribute} is what actually retargets"
    )


@pytest.mark.parametrize(("path", "attribute"), sorted(HEAD_KEYS.items()))
def test_head_fingerprint_values_match_live_head_policy_defaults(
    path: tuple[str, ...], attribute: str
) -> None:
    defaults = declared_defaults(HeadMappingPolicy)
    assert attribute in defaults, f"HeadMappingPolicy has no defaulted field {attribute!r}"
    assert config_value("head", path) == pytest.approx(defaults[attribute]), (
        f"MAPPING_CONFIG['head']{list(path)} is hashed into recordings, but "
        f"HeadMappingPolicy.{attribute} is what actually retargets"
    )


@pytest.mark.parametrize(("path", "attribute"), sorted(TORSO_KEYS.items()))
def test_torso_fingerprint_values_match_live_torso_policy_defaults(
    path: tuple[str, ...], attribute: str
) -> None:
    defaults = declared_defaults(TorsoMappingPolicy)
    assert attribute in defaults, f"TorsoMappingPolicy has no defaulted field {attribute!r}"
    assert config_value("torso", path) == pytest.approx(defaults[attribute]), (
        f"MAPPING_CONFIG['torso']{list(path)} is hashed into recordings, but "
        f"TorsoMappingPolicy.{attribute} is what actually retargets"
    )


def test_torso_shoulder_width_guard_is_pinned_to_the_signal_function() -> None:
    """The guard lives on the function, not the policy, so pin it where it lives."""
    guard = signature(shoulder_yaw_signal).parameters["min_shoulder_width_m"].default
    assert config_value("torso", ("min_shoulder_width_m",)) == pytest.approx(guard)


def test_pinned_attributes_exist_on_the_policy_dataclasses() -> None:
    """A field rename must break here rather than silently unpin a hashed value."""
    assert set(ARM_KEYS.values()) <= set(declared_defaults(LeftArmPolicy))
    assert set(HEAD_KEYS.values()) <= set(declared_defaults(HeadMappingPolicy))
    assert set(TORSO_KEYS.values()) <= set(declared_defaults(TorsoMappingPolicy))


def test_every_numeric_arm_and_head_value_is_pinned_to_a_dataclass_default() -> None:
    """A new tuned number in any section must be pinned before it can be hashed."""
    assert numeric_paths(MAPPING_CONFIG["arms"]) == set(ARM_KEYS)
    assert numeric_paths(MAPPING_CONFIG["head"]) == set(HEAD_KEYS)
    assert numeric_paths(MAPPING_CONFIG["torso"]) == set(TORSO_KEYS) | {
        ("min_shoulder_width_m",)
    }


def test_arm_policy_defaults_are_constructible_as_the_pipeline_builds_them() -> None:
    """The pipeline passes no policy, so the defaults must satisfy __post_init__.

    No tuned value is repeated as a literal anywhere in this file on purpose: a
    third copy of a number is a third thing that can drift.  The dataclass is the
    authority and the comparisons above are what hold MAPPING_CONFIG to it.
    """
    policy = LeftArmPolicy()
    declared = declared_defaults(LeftArmPolicy)
    for attribute in ARM_KEYS.values():
        assert getattr(policy, attribute) == pytest.approx(declared[attribute])


def test_mapping_fingerprint_is_stable_across_repeated_serialization() -> None:
    assert fingerprint(MAPPING_CONFIG) == fingerprint(MAPPING_CONFIG)
    assert fingerprint(MAPPING_CONFIG) == MAPPING_HASH


def test_mapping_config_cannot_claim_unvalidated_metric_pose_control() -> None:
    arms = MAPPING_CONFIG["arms"]
    assert arms["coordinates"] == "shoulder-relative-normalized-torso-local-offset"
    assert arms["pose_world_coordinates"] == "diagnostic-only-unvalidated"
    assert arms["human_torso_yaw_correction"] == (
        "inverse pose-world shoulder-yaw before realized robot tummy-yaw"
    )
    assert MAPPING_CONFIG["torso"]["front_evidence"] == (
        "face-mesh-eye-aperture-relative-to-neutral-required-for-torso-direction"
    )


def test_mapping_fingerprint_ignores_key_order_but_not_values() -> None:
    reordered = {key: MAPPING_CONFIG[key] for key in reversed(list(MAPPING_CONFIG))}
    reordered["head"] = {
        key: MAPPING_CONFIG["head"][key] for key in reversed(list(MAPPING_CONFIG["head"]))
    }
    assert fingerprint(reordered) == MAPPING_HASH
    drifted = {
        **MAPPING_CONFIG,
        "head": {**MAPPING_CONFIG["head"], "yaw_gain": MAPPING_CONFIG["head"]["yaw_gain"] + 0.1},
    }
    assert fingerprint(drifted) != MAPPING_HASH


def test_direction_vector_candidate_has_its_own_recording_fingerprint() -> None:
    assert mapping_hash_for("wrist-primary") == MAPPING_HASH
    assert mapping_hash_for("direction-vector") != MAPPING_HASH


def test_direction_vector_fingerprint_attests_to_the_required_torso_frame() -> None:
    candidate = mapping_config_for("direction-vector")
    assert candidate["version"] == "direction-vector-arm-shape-ab-v4-calibrated-torso-yaw"
    assert candidate["arms"]["strategy"] == (
        "torso-relative-upper-arm-and-forearm-direction-vectors"
    )
    assert candidate["arms"]["torso_frame"] == {
        "lateral": "left-shoulder-minus-right-shoulder",
        "inferior": "shoulder-midpoint-to-hip-midpoint",
        "normal": "cross-lateral-inferior-camera-depth-signed",
        "missing_or_degenerate": "hold-that-limb",
    }
