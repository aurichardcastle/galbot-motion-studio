from math import pi

import numpy as np
import pytest

from galbot_motion_studio.contracts.human import Landmark
from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.retarget.left_arm import (
    ArmRetargetError,
    ArmMapRejection,
    IKSolveStage,
    LeftArmIntent,
    LeftArmPolicy,
    LeftArmRetargeter,
    _rotation_error,
    _human_direction_to_robot,
    create_left_arm_calibration,
    create_right_arm_calibration,
    estimate_left_arm_intent,
    estimate_right_arm_intent,
)
from galbot_motion_studio.safety.clearance import HOME_QPOS
from galbot_motion_studio.safety.supervisor import SupervisorPolicy
from galbot_motion_studio.vision.calibration import (
    create_neutral_calibration,
    normalized_landmark_distance,
)

from test_calibration import stable_observation


def arm_observation(**changes: object):
    base = stable_observation()
    landmarks = base.landmarks + (
        Landmark(
            name="pose_13",
            normalized_xyz=(0.75, 0.55, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
        Landmark(
            name="pose_15",
            normalized_xyz=(0.80, 0.70, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
        Landmark(
            name="pose_14",
            normalized_xyz=(0.25, 0.55, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
        Landmark(
            name="pose_16",
            normalized_xyz=(0.20, 0.70, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
    )
    values = {"landmarks": landmarks}
    values.update(changes)
    return base.model_copy(update=values)


def test_arm_and_neutral_calibration_share_the_normalized_scale_definition() -> None:
    base = arm_observation()
    landmarks = tuple(
        landmark.model_copy(update={"normalized_xyz": (0.7, 0.4, 0.3)})
        if landmark.name == "pose_11"
        else landmark
        for landmark in base.landmarks
    )
    observation = base.model_copy(update={"landmarks": landmarks})
    by_name = {landmark.name: landmark for landmark in observation.landmarks}
    expected = normalized_landmark_distance(by_name["pose_11"], by_name["pose_12"])

    assert create_neutral_calibration(observation).shoulder_width_normalized == pytest.approx(
        expected
    )
    assert create_left_arm_calibration(observation).shoulder_width_normalized == pytest.approx(
        expected
    )
    assert create_right_arm_calibration(observation).shoulder_width_normalized == pytest.approx(
        expected
    )


def torso_arm_observation(**changes: object):
    """Arm fixture with the four landmarks required by direction-vector A/B."""
    base = arm_observation()
    landmarks = base.landmarks + (
        Landmark(
            name="pose_23",
            normalized_xyz=(0.65, 0.80, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
        Landmark(
            name="pose_24",
            normalized_xyz=(0.35, 0.80, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
    )
    values = {"landmarks": landmarks}
    values.update(changes)
    return base.model_copy(update=values)


def transform_pose_coordinates(observation, transform: np.ndarray):
    """Apply a rigid/image transform about the shoulder midpoint to pose points."""
    center = np.asarray((0.5, 0.4, 0.0), dtype=np.float64)
    return observation.model_copy(
        update={
            "landmarks": tuple(
                landmark.model_copy(
                    update={
                        "normalized_xyz": tuple(
                            center
                            + transform
                            @ (np.asarray(landmark.normalized_xyz) - center)
                        )
                    }
                )
                if landmark.name.startswith("pose_")
                else landmark
                for landmark in observation.landmarks
            )
        }
    )


def test_arm_intent_is_shoulder_normalized_and_absolute_pose_relative() -> None:
    neutral = arm_observation()
    calibration = create_left_arm_calibration(neutral)
    baseline = estimate_left_arm_intent(neutral, calibration)
    assert baseline.lateral_signal == pytest.approx(0.0, abs=1e-12)
    assert baseline.vertical_signal == pytest.approx(0.0, abs=1e-12)
    assert baseline.elbow_lateral_signal == pytest.approx(0.0, abs=1e-12)
    assert baseline.elbow_vertical_signal == pytest.approx(0.0, abs=1e-12)
    moved = tuple(
        landmark.model_copy(update={"normalized_xyz": (0.84, 0.66, 0.0)})
        if landmark.name == "pose_15"
        else landmark
        for landmark in neutral.landmarks
    )
    intent = estimate_left_arm_intent(neutral.model_copy(update={"landmarks": moved}), calibration)
    assert intent.lateral_signal == pytest.approx(0.1)
    assert intent.vertical_signal == pytest.approx(-0.1)


def test_wrist_primary_intent_is_invariant_to_a_valid_human_torso_yaw() -> None:
    """A chest-fixed hand must not acquire a false reach when the body turns."""
    observation = arm_observation()
    left_calibration = create_left_arm_calibration(observation)
    right_calibration = create_right_arm_calibration(observation)
    baseline_left = estimate_left_arm_intent(
        observation, left_calibration, human_torso_yaw_rad=0.0
    )
    baseline_right = estimate_right_arm_intent(
        observation, right_calibration, human_torso_yaw_rad=0.0
    )
    yaw = 0.62
    shoulder_midpoint_x = 0.5
    yawed = observation.model_copy(
        update={
            "landmarks": tuple(
                landmark.model_copy(
                    update={
                        "normalized_xyz": (
                            shoulder_midpoint_x
                            + np.cos(yaw)
                            * (landmark.normalized_xyz[0] - shoulder_midpoint_x)
                            - np.sin(yaw) * landmark.normalized_xyz[2],
                            landmark.normalized_xyz[1],
                            np.sin(yaw)
                            * (landmark.normalized_xyz[0] - shoulder_midpoint_x)
                            + np.cos(yaw) * landmark.normalized_xyz[2],
                        )
                    }
                )
                if landmark.name.startswith("pose_")
                else landmark
                for landmark in observation.landmarks
            )
        }
    )
    corrected_left = estimate_left_arm_intent(
        yawed, left_calibration, human_torso_yaw_rad=yaw
    )
    corrected_right = estimate_right_arm_intent(
        yawed, right_calibration, human_torso_yaw_rad=yaw
    )
    uncorrected_left = estimate_left_arm_intent(yawed, left_calibration)
    for corrected, baseline in (
        (corrected_left, baseline_left),
        (corrected_right, baseline_right),
    ):
        assert corrected.lateral_signal == pytest.approx(baseline.lateral_signal)
        assert corrected.depth_signal == pytest.approx(baseline.depth_signal)
        assert corrected.elbow_lateral_signal == pytest.approx(
            baseline.elbow_lateral_signal
        )
        assert corrected.elbow_depth_signal == pytest.approx(baseline.elbow_depth_signal)
    # Without the inverse torso transform, the exact same chest-relative arm
    # pose would falsely gain lateral/depth motion in camera axes.
    assert uncorrected_left.lateral_signal != pytest.approx(
        baseline_left.lateral_signal
    )


def test_unvalidated_pose_world_coordinates_do_not_override_normalized_v1_mapping() -> None:
    neutral = arm_observation()
    world = {
        "pose_11": (0.2, 0.0, 0.0),
        "pose_12": (-0.2, 0.0, 0.0),
        "pose_13": (0.25, 0.15, 0.0),
        "pose_15": (0.30, 0.30, 0.0),
    }
    world_landmarks = tuple(
        landmark.model_copy(update={"world_xyz_m": world[landmark.name]})
        if landmark.name in world
        else landmark
        for landmark in neutral.landmarks
    )
    neutral = neutral.model_copy(update={"landmarks": world_landmarks})
    calibration = create_left_arm_calibration(neutral)
    assert calibration.coordinate_space == "normalized"

    moved = tuple(
        landmark.model_copy(update={"world_xyz_m": (0.30, 0.30, 0.04)})
        if landmark.name == "pose_15"
        else landmark
        for landmark in neutral.landmarks
    )
    intent = estimate_left_arm_intent(
        neutral.model_copy(update={"landmarks": moved}), calibration
    )
    assert intent.depth_signal == pytest.approx(0.0)


def test_zero_arm_intent_solves_the_absolute_relaxed_arm_reference() -> None:
    mapper = LeftArmRetargeter(load_verified_fixed_base_model(), neutral_pose=HOME_QPOS)
    neutral = mapper.map(LeftArmIntent(0.0, 0.0), previous=None, elapsed_s=None)
    assert neutral.target is not None
    assert neutral.target.residual_m <= 0.01
    assert neutral.target.wrist_target_world_m[2] == pytest.approx(0.90077, abs=1e-3)

    moved = mapper.map(LeftArmIntent(-0.05, -0.05), previous=None, elapsed_s=None)
    assert moved.target is not None
    assert moved.target.residual_m <= 0.01


def test_shoulder_height_band_converges_at_the_10_mm_command_threshold() -> None:
    """The primary manipulation band is solved, not relaxed into acceptance.

    The retained replay exposed a DLS-conditioning failure around shoulder
    height.  This is intentionally an ordinary wrist-primary solve at the
    shipped 10 mm threshold: changing the acceptance bar must never be a way to
    make this regression pass.
    """
    policy = LeftArmPolicy()
    assert policy.residual_tolerance_m == pytest.approx(0.01)
    assert policy.residual_tolerance_m < SupervisorPolicy().max_ik_residual_m
    mapper = LeftArmRetargeter(
        load_verified_fixed_base_model(), neutral_pose=HOME_QPOS, policy=policy
    )

    for vertical_signal in (-0.75, -1.35, -1.10):
        result = mapper.map(
            LeftArmIntent(0.0, vertical_signal), previous=None, elapsed_s=None
        )
        assert result.target is not None
        assert result.target.residual_m <= policy.residual_tolerance_m


def test_rejected_ik_retains_the_measured_best_wrist_residual() -> None:
    """Rejected solves need an auditable residual without becoming a command."""
    policy = LeftArmPolicy(residual_tolerance_m=0.001)
    mapper = LeftArmRetargeter(
        load_verified_fixed_base_model(), neutral_pose=HOME_QPOS, policy=policy
    )

    result = mapper.map(LeftArmIntent(0.0, -1.10), previous=None, elapsed_s=None)

    assert result.target is None
    assert result.rejection is ArmMapRejection.IK_DID_NOT_CONVERGE
    assert result.best_wrist_residual_m is not None
    assert result.best_wrist_residual_m > policy.residual_tolerance_m
    assert [attempt.stage for attempt in result.ik_attempts] == [
        IKSolveStage.NEUTRAL,
        IKSolveStage.POSITION_PRIORITY,
        IKSolveStage.CONDITIONED_POSITION_PRIORITY,
    ]
    assert result.ik_attempts[-1].wrist_residual_m == pytest.approx(
        result.best_wrist_residual_m
    )


def test_joint_branch_jump_is_rejected() -> None:
    mapper = LeftArmRetargeter(load_verified_fixed_base_model(), neutral_pose=HOME_QPOS)
    result = mapper.map(
        LeftArmIntent(-0.05, -0.05),
        previous={name: HOME_QPOS[name] for name in mapper.joint_names},
        elapsed_s=0.001,
    )
    assert result.target is None
    assert result.rejection is ArmMapRejection.JOINT_CONTINUITY


def test_operator_signal_is_clamped_to_the_workspace_shell() -> None:
    mapper = LeftArmRetargeter(load_verified_fixed_base_model(), neutral_pose=HOME_QPOS)
    boundary = mapper.map(LeftArmIntent(2.0, 0.0), previous=None, elapsed_s=None)
    excessive = mapper.map(LeftArmIntent(100.0, 0.0), previous=None, elapsed_s=None)
    assert excessive == boundary


def test_unvalidated_world_landmarks_do_not_turn_the_normalized_mapper_metric() -> None:
    neutral = arm_observation()
    measured_world = {
        "pose_11": (0.127, -0.450, -0.049),
        "pose_12": (-0.189, -0.456, -0.082),
        "pose_13": (0.234, -0.453, -0.139),
        "pose_14": (-0.229, -0.473, -0.224),
        "pose_15": (0.141, -0.663, -0.308),
        "pose_16": (-0.140, -0.725, -0.310),
    }
    raised = neutral.model_copy(
        update={
            "landmarks": tuple(
                landmark.model_copy(update={"world_xyz_m": measured_world[landmark.name]})
                if landmark.name in measured_world
                else landmark
                for landmark in neutral.landmarks
            )
        }
    )
    left_intent = estimate_left_arm_intent(raised, create_left_arm_calibration(raised))
    right_intent = estimate_right_arm_intent(raised, create_right_arm_calibration(raised))
    assert left_intent.vertical_signal == pytest.approx(0.0)
    assert right_intent.vertical_signal == pytest.approx(0.0)


def test_rotation_error_is_finite_at_exactly_half_a_turn() -> None:
    target = np.diag([1.0, -1.0, -1.0])
    error = _rotation_error(target, np.eye(3))
    assert np.all(np.isfinite(error))
    assert np.linalg.norm(error) == pytest.approx(pi)


def test_low_confidence_elbow_falls_back_to_neutral_soft_hint() -> None:
    observation = arm_observation()
    calibration = create_left_arm_calibration(observation)
    dim_elbow = observation.model_copy(
        update={
            "landmarks": tuple(
                landmark.model_copy(update={"visibility": 0.1, "presence": 0.1})
                if landmark.name == "pose_13"
                else landmark
                for landmark in observation.landmarks
            )
        }
    )
    intent = estimate_left_arm_intent(dim_elbow, calibration)
    assert intent.elbow_lateral_signal == 0.0
    assert intent.elbow_vertical_signal == 0.0
    assert intent.elbow_depth_signal == 0.0
    assert intent.upper_arm_direction is None
    assert intent.forearm_direction is None


def test_direction_vector_mapper_holds_when_elbow_direction_is_not_observed() -> None:
    mapper = LeftArmRetargeter(
        load_verified_fixed_base_model(),
        neutral_pose=HOME_QPOS,
        policy=LeftArmPolicy(mapping_mode="direction-vector"),
    )
    result = mapper.map(LeftArmIntent(0.0, 0.0), previous=None, elapsed_s=None)
    assert result.target is None
    assert result.rejection is ArmMapRejection.INCOMPLETE_OBSERVATION


def test_direction_vector_axis_transform_matches_position_mapper_convention() -> None:
    assert _human_direction_to_robot((1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert _human_direction_to_robot((0.0, 1.0, 0.0)) == pytest.approx((0.0, 0.0, -1.0))
    assert _human_direction_to_robot((0.0, 0.0, 1.0)) == pytest.approx((-1.0, 0.0, 0.0))


def test_direction_vectors_are_invariant_to_camera_yaw_and_image_mirroring() -> None:
    observation = torso_arm_observation()
    left_calibration = create_left_arm_calibration(observation)
    right_calibration = create_right_arm_calibration(observation)
    baseline_left = estimate_left_arm_intent(observation, left_calibration)
    baseline_right = estimate_right_arm_intent(observation, right_calibration)
    assert baseline_left.upper_arm_direction is not None
    assert baseline_left.forearm_direction is not None
    assert baseline_right.upper_arm_direction is not None
    assert baseline_right.forearm_direction is not None

    yaw = 0.45
    yawed = transform_pose_coordinates(
        observation,
        np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ]
        ),
    )
    # A horizontal screen mirror is an improper transform.  The torso normal's
    # camera-depth sign convention must make the resulting body coordinates
    # identical for each anatomical side, rather than swapping or reversing them.
    mirrored = transform_pose_coordinates(observation, np.diag((-1.0, 1.0, 1.0)))
    for transformed in (yawed, mirrored):
        left = estimate_left_arm_intent(transformed, left_calibration)
        right = estimate_right_arm_intent(transformed, right_calibration)
        assert left.upper_arm_direction == pytest.approx(baseline_left.upper_arm_direction)
        assert left.forearm_direction == pytest.approx(baseline_left.forearm_direction)
        assert right.upper_arm_direction == pytest.approx(baseline_right.upper_arm_direction)
        assert right.forearm_direction == pytest.approx(baseline_right.forearm_direction)


def test_direction_vector_fails_closed_without_a_nondegenerate_torso_basis() -> None:
    no_hips = arm_observation()
    calibration = create_left_arm_calibration(no_hips)
    missing = estimate_left_arm_intent(no_hips, calibration)
    assert missing.upper_arm_direction is None
    assert missing.forearm_direction is None

    observation = torso_arm_observation()
    collapsed = observation.model_copy(
        update={
            "landmarks": tuple(
                landmark.model_copy(update={"normalized_xyz": (0.5, 0.4, 0.0)})
                if landmark.name in {"pose_23", "pose_24"}
                else landmark
                for landmark in observation.landmarks
            )
        }
    )
    degenerate = estimate_left_arm_intent(collapsed, create_left_arm_calibration(collapsed))
    assert degenerate.upper_arm_direction is None
    assert degenerate.forearm_direction is None
    candidate = LeftArmRetargeter(
        load_verified_fixed_base_model(),
        neutral_pose=HOME_QPOS,
        policy=LeftArmPolicy(mapping_mode="direction-vector"),
    )
    assert candidate.map(degenerate, previous=None, elapsed_s=None).rejection is (
        ArmMapRejection.INCOMPLETE_OBSERVATION
    )


def test_direction_vector_target_is_derived_from_both_segments_not_wrist_signals() -> None:
    mapper = LeftArmRetargeter(
        load_verified_fixed_base_model(),
        neutral_pose=HOME_QPOS,
        policy=LeftArmPolicy(mapping_mode="direction-vector"),
    )
    # Invert _human_direction_to_robot for the mapper's exact neutral vectors,
    # making this an immediately reachable, deterministic shape target.
    upper = mapper._neutral_elbow - mapper._neutral_shoulder
    upper /= np.linalg.norm(upper)
    forearm = mapper._neutral_forearm_direction / mapper._neutral_forearm_length
    human_upper = (upper[1], -upper[2], -upper[0])
    human_forearm = (forearm[1], -forearm[2], -forearm[0])
    first = mapper.map(
        LeftArmIntent(0.0, 0.0, upper_arm_direction=human_upper, forearm_direction=human_forearm),
        previous=None,
        elapsed_s=None,
    )
    second = mapper.map(
        LeftArmIntent(1.5, -1.5, 1.5, upper_arm_direction=human_upper, forearm_direction=human_forearm),
        previous=None,
        elapsed_s=None,
    )
    assert first.target is not None
    assert second.target is not None
    assert second.target.wrist_target_world_m == pytest.approx(first.target.wrist_target_world_m)
    assert second.target.elbow_target_world_m == pytest.approx(first.target.elbow_target_world_m)
    assert first.target.wrist_target_world_m == pytest.approx(mapper._neutral_tcp)


def test_direction_vector_reseeds_before_strict_shape_gate_but_never_relaxes_it() -> None:
    mapper = LeftArmRetargeter(
        load_verified_fixed_base_model(),
        neutral_pose=HOME_QPOS,
        policy=LeftArmPolicy(mapping_mode="direction-vector"),
    )
    upper = mapper._neutral_elbow - mapper._neutral_shoulder
    upper /= np.linalg.norm(upper)
    forearm = mapper._neutral_forearm_direction / mapper._neutral_forearm_length
    intent = LeftArmIntent(
        0.0,
        0.0,
        upper_arm_direction=(upper[1], -upper[2], -upper[0]),
        forearm_direction=(forearm[1], -forearm[2], -forearm[0]),
    )
    expected = mapper.map(intent, previous=None, elapsed_s=None)
    assert expected.target is not None
    # This valid-but-distant warm start leaves the first strict solve on a poor
    # branch.  The mapper must retry from neutral and evaluate strict segment
    # residuals there, not reject early or lower shape weights to force a wrist
    # solution. Ten seconds gives the normal rate governor ample legal distance.
    result = mapper.map(
        intent,
        previous={name: 0.0 for name in mapper.joint_names},
        elapsed_s=10.0,
    )
    assert result.target is not None
    assert result.target.wrist_target_world_m == pytest.approx(expected.target.wrist_target_world_m)
    assert result.target.elbow_target_world_m == pytest.approx(expected.target.elbow_target_world_m)


def test_command_landmark_confidence_floor_is_explicit() -> None:
    observation = arm_observation()
    calibration = create_left_arm_calibration(observation)
    marginal_wrist = observation.model_copy(
        update={
            "landmarks": tuple(
                landmark.model_copy(update={"visibility": 0.4, "presence": 0.4})
                if landmark.name == "pose_15"
                else landmark
                for landmark in observation.landmarks
            )
        }
    )
    with pytest.raises(ArmRetargetError, match="occluded"):
        estimate_left_arm_intent(marginal_wrist, calibration)
    estimate_left_arm_intent(
        marginal_wrist, calibration, min_confidence=0.35
    )


def _flip_retargeter():
    from galbot_motion_studio.model.loader import load_verified_fixed_base_model
    from galbot_motion_studio.pipeline import SIM_TELEOP_HOME_QPOS
    from galbot_motion_studio.retarget.left_arm import LeftArmRetargeter

    return LeftArmRetargeter(
        load_verified_fixed_base_model(), neutral_pose=SIM_TELEOP_HOME_QPOS
    )


def _accepted(retargeter, joints, wrist):
    from galbot_motion_studio.retarget.left_arm import (
        LeftArmMappingResult,
        LeftArmTarget,
    )

    return retargeter._accept(
        LeftArmMappingResult(
            LeftArmTarget(
                joints_rad=tuple(joints.items()),
                wrist_target_world_m=wrist,
                elbow_target_world_m=(0.0, 0.0, 0.0),
                residual_m=0.001,
            ),
            best_wrist_residual_m=0.001,
        )
    )


def _solution(**overrides):
    joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}
    joints.update(overrides)
    return joints


def test_a_branch_flip_is_rejected() -> None:
    """I5: the joint solution jumps while the wrist target barely moves."""
    from galbot_motion_studio.retarget.left_arm import ArmMapRejection

    retargeter = _flip_retargeter()
    first = _accepted(retargeter, _solution(), (0.4, 0.2, 1.0))
    assert first.target is not None
    # Wrist target moves 1 mm; joint 4 swings 1.2 rad to reach it.
    flipped = _accepted(
        retargeter, _solution(left_arm_joint4=1.2), (0.401, 0.2, 1.0)
    )
    assert flipped.target is None
    assert flipped.rejection is ArmMapRejection.JOINT_CONTINUITY


def test_a_large_move_the_operator_actually_asked_for_is_kept() -> None:
    """The same joint jump, but the target moved with it, is real tracking."""
    retargeter = _flip_retargeter()
    _accepted(retargeter, _solution(), (0.4, 0.2, 1.0))
    reached = _accepted(
        retargeter, _solution(left_arm_joint4=1.2), (0.4, 0.35, 1.0)
    )
    assert reached.target is not None, "a genuine 150 mm reach was rejected"


def test_normal_tracking_is_never_flagged() -> None:
    # Measured p50 solution delta on the 2026-08-22 trial was 0.089 rad.
    retargeter = _flip_retargeter()
    _accepted(retargeter, _solution(), (0.4, 0.2, 1.0))
    for step in range(1, 8):
        result = _accepted(
            retargeter,
            _solution(left_arm_joint4=0.089 * step),
            (0.4 + 0.001 * step, 0.2, 1.0),
        )
        assert result.target is not None, f"normal tracking flagged at step {step}"


def test_a_rejected_flip_does_not_rebaseline_the_test() -> None:
    """Otherwise one flip is refused and the next frame accepts the new branch.

    That would let the arm walk to the flipped configuration a frame at a time
    while every individual frame looks continuous.
    """
    retargeter = _flip_retargeter()
    _accepted(retargeter, _solution(), (0.4, 0.2, 1.0))
    flipped = _solution(left_arm_joint4=1.2)
    first = _accepted(retargeter, flipped, (0.401, 0.2, 1.0))
    assert first.target is None
    again = _accepted(retargeter, flipped, (0.402, 0.2, 1.0))
    assert again.target is None, "the flipped pose was accepted on the retry"


def test_the_first_solution_is_never_a_flip() -> None:
    retargeter = _flip_retargeter()
    result = _accepted(retargeter, _solution(left_arm_joint4=2.0), (0.4, 0.2, 1.0))
    assert result.target is not None
