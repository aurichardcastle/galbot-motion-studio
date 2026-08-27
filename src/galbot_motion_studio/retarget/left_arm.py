"""Shoulder-relative bilateral wrist/elbow retargeting with MuJoCo DLS IK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import acos, atan2, cos, isfinite, sin, sqrt
from typing import Mapping

import mujoco
import numpy as np

from galbot_motion_studio.contracts.human import HumanObservation, IdentityState, Landmark
from galbot_motion_studio.model.joint_map import CONTROL_GROUPS, JointLimit, parse_revolute_joint_limits
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST
from galbot_motion_studio.retarget.palm_frame import palm_frame_from_landmarks
from galbot_motion_studio.retarget.swivel_ik import swivel_geometry
from galbot_motion_studio.retarget.torso import TORSO_BODY, TORSO_YAW_JOINT
from galbot_motion_studio.vision.calibration import normalized_landmark_distance


class ArmRetargetError(ValueError):
    pass


class ArmMapRejection(StrEnum):
    NONFINITE = "NONFINITE"
    INCOMPLETE_OBSERVATION = "INCOMPLETE_OBSERVATION"
    IK_DID_NOT_CONVERGE = "IK_DID_NOT_CONVERGE"
    JOINT_CONTINUITY = "JOINT_CONTINUITY"


class IKSolveStage(StrEnum):
    """One deterministic rung of the wrist-primary DLS retry ladder."""

    WARM = "warm"
    NEUTRAL = "neutral"
    POSITION_PRIORITY = "position_priority"
    CONDITIONED_POSITION_PRIORITY = "conditioned_position_priority"


@dataclass(frozen=True)
class IKSolveAttempt:
    """A finite wrist residual measured after one DLS solve rung.

    This is diagnostic evidence only.  It is retained on a rejected mapper
    result so a replay can distinguish a target that exhausted the complete
    ladder from one rejected before the conditioned retry ran.
    """

    stage: IKSolveStage
    wrist_residual_m: float

    def __post_init__(self) -> None:
        if not isfinite(self.wrist_residual_m) or self.wrist_residual_m < 0.0:
            raise ValueError("IK attempt residual must be finite and non-negative")


MIN_SHOULDER_WIDTH_NORMALIZED = 0.05


@dataclass(frozen=True)
class LeftArmCalibration:
    source_clock_id: str
    side: str
    coordinate_space: str
    shoulder_width_normalized: float
    wrist_reference_from_shoulder_xyz: tuple[float, float, float]
    elbow_reference_from_shoulder_xyz: tuple[float, float, float]

    @property
    def wrist_from_shoulder_normalized_xy(self) -> tuple[float, float]:
        return self.wrist_reference_from_shoulder_xyz[:2]


@dataclass(frozen=True)
class LeftArmIntent:
    lateral_signal: float
    vertical_signal: float
    depth_signal: float = 0.0
    elbow_lateral_signal: float = 0.0
    elbow_vertical_signal: float = 0.0
    elbow_depth_signal: float = 0.0
    # Unit vectors in a torso-relative human frame: +X is anatomical left, +Y is
    # inferior, and +Z is the signed torso normal.  They are optional so the
    # established wrist-primary mapper retains its documented behaviour when
    # MediaPipe loses an elbow; the analysis-only direction-vector mapper refuses
    # to substitute an invented elbow pose for a missing one or a missing torso
    # coordinate frame.
    upper_arm_direction: tuple[float, float, float] | None = None
    forearm_direction: tuple[float, float, float] | None = None
    #: The operator's palm orientation, as a 3x3 rotation flattened row-major and
    #: already expressed in robot axes. ``None`` when the hand is not visible, in
    #: which case the mapper keeps its forearm-aligned guess.
    palm_rotation: tuple[float, ...] | None = None
    #: 0..1 conditioning of that measurement. An edge-on palm still yields a
    #: valid rotation but a poorly determined one, so the orientation task is
    #: weighted by this rather than trusted flat.
    palm_confidence: float = 0.0
    #: The operator's swivel angle -- the elbow's rotation about their own
    #: shoulder->wrist axis -- in radians. ``None`` when the arm's landmarks are
    #: not confident enough to define it.
    #:
    #: Measured against the SHOULDER LINE, not the torso basis. This is the whole
    #: reason the swivel mapper can run live: psi needs one body-fixed reference
    #: direction perpendicular to the shoulder->wrist axis, and the shoulder line
    #: is one. The torso basis needs hips, and hips have never once been in frame
    #: (median confidence 0.035, clearing threshold on 4 of 623 frames). Measured
    #: on the same retained captures, shoulder-line psi is available on 88.1% of
    #: left-arm frames and 90.9% of right-arm frames, never degenerate.
    #:
    #: psi is a dihedral angle between two half-planes about a shared axis, so a
    #: camera yaw rotates every input and leaves it unchanged. A mirror flips its
    #: sign, which :func:`_human_swivel_angle` handles per side.
    swivel_angle_rad: float | None = None
    #: 0..1 conditioning of the swivel measurement, faded out as the arm
    #: approaches full extension where psi stops carrying information.
    swivel_confidence: float = 0.0


@dataclass(frozen=True)
class LeftArmTarget:
    joints_rad: tuple[tuple[str, float], ...]
    wrist_target_world_m: tuple[float, float, float]
    elbow_target_world_m: tuple[float, float, float]
    residual_m: float

    @property
    def joint_map(self) -> dict[str, float]:
        return dict(self.joints_rad)


@dataclass(frozen=True)
class LeftArmMappingResult:
    """Mapper outcome, including a measured best-effort wrist residual on rejection.

    A rejected solution is never a command: ``target`` stays ``None``.  Retaining
    its finite wrist residual separately lets retained-video analysis distinguish
    an accuracy-limited workspace band from a solver/observation failure without
    loosening the command authority boundary.
    """

    target: LeftArmTarget | None
    rejection: ArmMapRejection | None = None
    best_wrist_residual_m: float | None = None
    ik_attempts: tuple[IKSolveAttempt, ...] = ()

    def __post_init__(self) -> None:
        if self.target is not None and self.rejection is not None:
            raise ValueError("a mapped arm target cannot also carry a rejection")
        if self.target is None and self.rejection is None:
            raise ValueError("an absent arm target requires a rejection")
        if self.best_wrist_residual_m is not None and not (
            isfinite(self.best_wrist_residual_m) and self.best_wrist_residual_m >= 0.0
        ):
            raise ValueError("best wrist residual must be finite and non-negative")


@dataclass(frozen=True)
class LeftArmPolicy:
    forward_offset_m: float = 0.30
    # Lateral was 0.16 against a vertical scale of 0.45, so the same physical hand
    # displacement produced ~2.8x less robot motion sideways than vertically.
    # Measured over the operator's own 372-frame session, robot TCP travel was
    # 44-48 cm vertical and 20-23 cm forward/back, against only 3-6 cm lateral: the
    # mapping distorted the motion before any safety layer saw it.
    #
    # Lateral is not raised all the way to match vertical. The crossing arm is the
    # binding constraint, and the largest inward signal that still clears the 5 mm
    # SIM floor falls off sharply with scale: 0.50 at 0.16, 0.14 at 0.30, 0.11 at
    # 0.45. Every clearance rejection is a frozen frame, so a bigger scale buys
    # reach and pays in stutter. At 0.28 the typical (vertical 0, depth 0) inward
    # limit is 0.81, against 0.50 at the old 0.16.
    lateral_scale_m: float = 0.28
    vertical_scale_m: float = 0.45
    # Depth is asymmetric because its two directions are not alike. Pulling the hand
    # CHEST-WARD (current_depth > 0) is capped by geometry: the safe chest-ward
    # distance is fixed (~0.058 m), so a bigger scale only reaches that same
    # self-collision wall on less signal (safe chest-ward signal: 0.48 at 0.12, 0.31
    # at 0.18, 0.20 at 0.28). So the chest-ward scale stays conservative. Reaching
    # FORWARD (current_depth < 0), away from the body, has open workspace -- and depth
    # was the most compressed axis, ~2.4x less than vertical, which read as poor depth
    # perception. It was held low because chest-ward over-reach used to FREEZE the
    # frame on a clearance rejection; the guarded path controller now backs that off
    # smoothly, so the forward scale can open up without paying in frozen frames.
    #: Scale for pulling the hand toward the body (current_depth > 0) -- the wall side.
    depth_scale_m: float = 0.12
    #: Scale for reaching the hand away from the body (current_depth < 0) -- open side.
    #: Raised above depth_scale_m to give forward reach the responsiveness that the
    #: single compressed 0.12 lacked, safe because there is no wall in that direction.
    depth_forward_scale_m: float = 0.24
    # The operator's core complaint for most of a day: "i cant move my shoulders at
    # all", "if i flex my lats and move my shoulders out the robot still stays super
    # stiff". Shoulder abduction is `*_arm_joint2`, the only joint that moves the
    # elbow sideways at all.
    #
    # It was never blocked. Measured from the operator's own 602-frame session, j2 was
    # COMMANDED over 19.7 deg and OBSERVED over 19.7 deg -- tracking its command
    # exactly, nowhere near its limits. Nothing was stopping the shoulder; the
    # retargeter simply never asked it for more.
    #
    # Cause: at 0.20 the elbow target travelled LESS than the wrist target (lateral
    # 0.28), so a gesture that moves the elbow outward while the wrist stays put --
    # exactly shoulder abduction -- barely moved the elbow goal, and the arm reached
    # the unchanged wrist goal without opening the shoulder. Raising `elbow_weight`
    # made it *worse* (22.0 -> 11.6 deg) for the same reason: weighting a target that
    # barely moves pins the arm still.
    #
    # Measured j2 travel on an elbow-out/wrist-still gesture, at the default weight:
    #     0.20 -> 22.0 deg     0.35 -> 69.9 deg     0.50 -> 74.7 deg     0.90 -> 74.7
    # Zero clearance rejections at every value. 0.50 is the knee of the curve.
    elbow_scale_m: float = 0.50
    elbow_weight: float = 0.02
    # Wrist position is the primary imitation task. Forearm alignment is a soft
    # secondary objective; a larger value makes ordinary hands-over-head poses
    # numerically unreachable even though the robot has the workspace for them.
    orientation_weight: float = 0.001
    #: Weight for a MEASURED palm orientation, as against the 0.001 above for the
    #: forearm-aligned guess. Two orders of magnitude higher because it is a real
    #: command rather than a tie-breaker, and affordable because wrist position
    #: (3 DOF) plus palm orientation (3) leaves exactly one redundant DOF on a
    #: 7-DOF arm -- the swivel. Scaled by palm confidence at use, so an edge-on
    #: hand decays smoothly back to the guess instead of snapping to it.
    palm_orientation_weight: float = 0.1
    #: Blend factor for the palm orientation, low because the measurement is
    #: noisy. Measured on a real capture: at 0.1 the frame-to-frame jitter falls
    #: from 40.8 deg p95 to 6.1 deg while the operator's genuine 60.7 deg median
    #: range of hand rotation survives at 60.1 deg. Higher values leak jitter,
    #: lower ones start eating the signal.
    palm_smoothing: float = 0.1
    # Shoulder-normalised absolute wrist height routinely reaches 1.5-1.8 when a
    # person raises a hand from a relaxed-arm reference. A 0.75 clamp silently
    # converted every such gesture to a below-shoulder robot target.
    max_signal: float = 2.0
    # Stay inside, rather than exactly on, the supervisor's 0.09 rad boundary.
    # The discrete trajectory integrator can otherwise overshoot an IK target by
    # a few tenths of a milliradian and latch HOLD at an otherwise safe pose.
    soft_margin_rad: float = 0.11
    #: I5, branch-flip detection. A redundant 7-DOF arm can reach the same wrist
    #: point from a different elbow branch, and the solver will happily switch
    #: between them: the task-space target moves a millimetre, the joint solution
    #: jumps most of a radian, and the arm swings through a large arc to arrive
    #: somewhere it already was. Every task-space check passes, which is why only
    #: a joint-space continuity test catches it.
    #:
    #: Measured on the 2026-08-22 trial, comparing consecutive SOLUTIONS (not
    #: commanded poses -- the governor's 0.05 rad step cap hides the jump
    #: completely, max |dq| in the commanded stream was 0.0558 rad): right-arm
    #: solution deltas reached 2.32 rad at p95 and 4.58 rad at maximum, and 3.95%
    #: of frames moved a joint more than 0.5 rad while the wrist target moved less
    #: than 10 mm. The left arm showed none. Normal motion sits at 0.089 rad p50,
    #: so 0.5 rad is roughly 6x the working delta and does not fire on tracking.
    #:
    #: 10 mm for the task-space side is `residual_tolerance_m`: below the distance
    #: the solver is allowed to miss by, the target has not meaningfully moved.
    branch_flip_joint_rad: float = 0.5
    branch_flip_target_m: float = 0.010
    max_joint_rate_rad_s: float = 0.35
    # Command acceptance stays deliberately far inside the supervisor's declared
    # 30 mm ceiling.  The retained replay's shoulder-height failures
    # were a DLS-conditioning defect, not evidence that large wrist errors were
    # useful: with the tuned solver those same targets converge below 10 mm.
    residual_tolerance_m: float = 0.01
    damping: float = 0.04
    # The primary solve and its existing position-priority retry preserve the
    # established arm/elbow behaviour. Only when both have failed their wrist
    # error gate do we use this stronger regularisation for the same reduced
    # objective. It resolves the shoulder-height conditioning band without
    # perturbing any pose the prior two-stage solver could command. Its value is
    # deliberately separate because the DLS sweep is non-monotonic.
    conditioned_retry_damping: float = 0.10
    posture_weight: float = 0.0001
    distal_posture_multiplier: float = 4.0
    # This is deliberately opt-in.  The live/simulation default remains the
    # wrist-primary mapper used by existing recordings; direction-vector is an
    # offline A/B candidate until it clears the retained-frame acceptance gate.
    mapping_mode: str = "wrist-primary"
    #: Human swivel delta -> robot swivel delta. The two psi conventions are
    #: measured in different bodies from different references, so the SIGN is an
    #: empirical fact about the two conventions, not a free tuning knob. See
    #: tools/swivel_sign_check.py for the measurement that fixes it.
    swivel_gain: float = 1.0
    max_iterations: int = 80
    max_iteration_step_rad: float = 0.04

    def __post_init__(self) -> None:
        positive = (
            self.forward_offset_m,
            self.lateral_scale_m,
            self.vertical_scale_m,
            self.depth_scale_m,
            self.elbow_scale_m,
            self.elbow_weight,
            self.orientation_weight,
            self.palm_orientation_weight,
            self.palm_smoothing,
            self.max_signal,
            self.max_joint_rate_rad_s,
            self.residual_tolerance_m,
            self.damping,
            self.conditioned_retry_damping,
            self.posture_weight,
            self.distal_posture_multiplier,
            self.max_iteration_step_rad,
        )
        if not all(isfinite(value) and value > 0 for value in positive):
            raise ValueError("left-arm policy values must be finite and positive")
        if self.soft_margin_rad < 0 or self.max_iterations < 1:
            raise ValueError("invalid left-arm margin or iteration count")
        # Signed on purpose: the two psi conventions may run opposite ways.
        if not isfinite(self.swivel_gain) or self.swivel_gain == 0:
            raise ValueError("swivel gain must be finite and non-zero")
        if self.mapping_mode not in {"wrist-primary", "direction-vector", "swivel-priority"}:
            raise ValueError("unknown arm mapping mode")


def create_left_arm_calibration(observation: HumanObservation) -> LeftArmCalibration:
    return create_arm_calibration(observation, side="left")


def create_right_arm_calibration(observation: HumanObservation) -> LeftArmCalibration:
    return create_arm_calibration(observation, side="right")


def create_arm_calibration(
    observation: HumanObservation, *, side: str
) -> LeftArmCalibration:
    if observation.identity is not IdentityState.STABLE:
        raise ArmRetargetError(f"{side}-arm calibration requires stable identity")
    shoulder_name, elbow_name, wrist_name = _arm_landmark_names(side)
    landmarks = _landmarks(
        observation, ("pose_11", "pose_12", shoulder_name, elbow_name, wrist_name)
    )
    if any(
        min(landmark.visibility, landmark.presence) < 0.5
        for name, landmark in landmarks.items()
        if name != elbow_name
    ):
        raise ArmRetargetError(f"{side}-arm calibration requires visible shoulder, elbow, and wrist")
    # V1 is deliberately a shoulder-normalized absolute-offset mapper.  MediaPipe
    # pose-world points are available, but their control error envelope has not
    # been validated. Selecting them opportunistically made identical runtime
    # behaviour change units whenever the detector supplied world values. Metric
    # mapping is an explicit, separately attested profile decision.
    coordinate_space = "normalized"
    shoulder_width = normalized_landmark_distance(
        landmarks["pose_11"], landmarks["pose_12"]
    )
    if shoulder_width < MIN_SHOULDER_WIDTH_NORMALIZED:
        raise ArmRetargetError("shoulder width is implausibly small")
    # Calibration establishes scale and the usable coordinate space, not the
    # operator's pose.  Using the captured wrist/elbow offsets as zero made an
    # arbitrary calibration pose become robot HOME: if the camera locked while
    # the operator's hands were above their head, those raised hands produced
    # exactly zero arm intent.  These canonical relaxed-arm references make pose
    # mapping absolute while still normalising for body size.
    lateral_sign = 1.0 if side == "left" else -1.0
    return LeftArmCalibration(
        source_clock_id=observation.source_clock_id,
        side=side,
        coordinate_space=coordinate_space,
        shoulder_width_normalized=shoulder_width,
        wrist_reference_from_shoulder_xyz=(0.25 * lateral_sign, 0.75, 0.0),
        elbow_reference_from_shoulder_xyz=(0.125 * lateral_sign, 0.375, 0.0),
    )


def estimate_left_arm_intent(
    observation: HumanObservation,
    calibration: LeftArmCalibration,
    *,
    min_confidence: float = 0.5,
    human_torso_yaw_rad: float | None = None,
) -> LeftArmIntent:
    return estimate_arm_intent(
        observation,
        calibration,
        min_confidence=min_confidence,
        human_torso_yaw_rad=human_torso_yaw_rad,
    )


def estimate_right_arm_intent(
    observation: HumanObservation,
    calibration: LeftArmCalibration,
    *,
    min_confidence: float = 0.5,
    human_torso_yaw_rad: float | None = None,
) -> LeftArmIntent:
    if calibration.side != "right":
        raise ArmRetargetError("right-arm estimator received a non-right calibration")
    return estimate_arm_intent(
        observation,
        calibration,
        min_confidence=min_confidence,
        human_torso_yaw_rad=human_torso_yaw_rad,
    )


def estimate_arm_intent(
    observation: HumanObservation,
    calibration: LeftArmCalibration,
    *,
    min_confidence: float = 0.5,
    human_torso_yaw_rad: float | None = None,
) -> LeftArmIntent:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    side = calibration.side
    if observation.identity is not IdentityState.STABLE:
        raise ArmRetargetError(f"{side}-arm intent requires stable identity")
    if observation.source_clock_id != calibration.source_clock_id:
        raise ArmRetargetError("observation and arm calibration clocks differ")
    if human_torso_yaw_rad is not None and not isfinite(human_torso_yaw_rad):
        raise ArmRetargetError("human torso yaw must be finite")
    shoulder_name, elbow_name, wrist_name = _arm_landmark_names(side)
    landmarks = _landmarks(
        observation, ("pose_11", "pose_12", shoulder_name, elbow_name, wrist_name)
    )
    if any(
        min(landmark.visibility, landmark.presence) < min_confidence
        for name, landmark in landmarks.items()
        if name != elbow_name
    ):
        raise ArmRetargetError(f"{side} arm is occluded")
    width = _coordinate_distance(
        landmarks["pose_11"], landmarks["pose_12"], calibration.coordinate_space
    )
    if width < MIN_SHOULDER_WIDTH_NORMALIZED:
        raise ArmRetargetError("shoulder width is implausibly small")
    shoulder = _coordinates(landmarks[shoulder_name], calibration.coordinate_space)
    wrist = _coordinates(landmarks[wrist_name], calibration.coordinate_space)
    elbow_confidence = min(
        landmarks[elbow_name].visibility,
        landmarks[elbow_name].presence,
    )
    elbow = _coordinates(landmarks[elbow_name], calibration.coordinate_space)
    current_wrist = tuple(
        (value - shoulder[index]) / width for index, value in enumerate(wrist)
    )
    current_elbow = (
        tuple((value - shoulder[index]) / width for index, value in enumerate(elbow))
        if elbow_confidence >= 0.3
        else calibration.elbow_reference_from_shoulder_xyz
    )
    if human_torso_yaw_rad is not None:
        # The baseline wrist mapper scales normalized shoulder-relative offsets,
        # but those offsets are still expressed in camera axes. A human turning
        # their chest rotates its lateral/depth components in those axes; passing
        # them straight on and then rotating the robot tummy applies that turn
        # twice. Express both limb vectors in the human torso frame first. The
        # yaw comes only from the pose-world shoulder *angle* (not a metric arm
        # length), and the caller supplies it only after its geometry gates pass.
        current_wrist = _into_human_torso_frame(current_wrist, human_torso_yaw_rad)
        current_elbow = _into_human_torso_frame(current_elbow, human_torso_yaw_rad)
    palm_rotation = None
    palm_confidence = 0.0
    # `landmarks` above holds only this arm's points; the palm needs the hand
    # landmarks, which the detector emits but nothing has read until now.
    palm = palm_frame_from_landmarks(
        {landmark.name: landmark for landmark in observation.landmarks},
        side,
        coordinate_space="normalized_xyz",
    )
    if palm is not None:
        # Same axis permutation the direction vectors use. A rotation transforms
        # as P R P^T, not P R: applying the permutation on one side only would
        # produce a matrix that is no longer a rotation.
        basis = _AXIS_PERMUTATION @ palm.as_matrix() @ _AXIS_PERMUTATION.T
        palm_rotation = tuple(float(v) for v in basis.reshape(-1))
        palm_confidence = float(palm.confidence)
    upper_arm_direction = None
    forearm_direction = None
    if elbow_confidence >= 0.3:
        # A direction in camera axes is not a body-shape measurement: rotating
        # or mirroring the camera changes it even while the operator's pose stays
        # exactly the same.  The analysis mapper therefore uses the orthonormal
        # torso basis below.  There is intentionally no camera-axis fallback;
        # missing/degenerate torso geometry becomes a candidate limb HOLD.
        torso = _torso_frame(
            observation,
            coordinate_space=calibration.coordinate_space,
            min_confidence=min_confidence,
        )
        if torso is not None:
            upper_arm_direction = torso.project(shoulder, elbow)
            forearm_direction = torso.project(elbow, wrist)
    swivel_angle_rad = None
    swivel_confidence = 0.0
    measured_swivel = _human_swivel_angle(
        observation, side=side, min_confidence=min_confidence
    )
    if measured_swivel is not None:
        swivel_angle_rad, swivel_confidence = measured_swivel
    return LeftArmIntent(
        lateral_signal=current_wrist[0] - calibration.wrist_reference_from_shoulder_xyz[0],
        vertical_signal=current_wrist[1] - calibration.wrist_reference_from_shoulder_xyz[1],
        depth_signal=current_wrist[2] - calibration.wrist_reference_from_shoulder_xyz[2],
        elbow_lateral_signal=current_elbow[0] - calibration.elbow_reference_from_shoulder_xyz[0],
        elbow_vertical_signal=current_elbow[1] - calibration.elbow_reference_from_shoulder_xyz[1],
        elbow_depth_signal=current_elbow[2] - calibration.elbow_reference_from_shoulder_xyz[2],
        upper_arm_direction=upper_arm_direction,
        forearm_direction=forearm_direction,
        palm_rotation=palm_rotation,
        palm_confidence=palm_confidence,
        swivel_angle_rad=swivel_angle_rad,
        swivel_confidence=swivel_confidence,
    )


def _into_human_torso_frame(
    vector: tuple[float, float, float], yaw_rad: float
) -> tuple[float, float, float]:
    """Undo a valid human shoulder-line yaw for a normalized limb vector.

    ``shoulder_yaw_signal`` defines +yaw as the left shoulder moving toward
    camera +Z.  Its corresponding local lateral axis in camera X/Z is
    ``(cos(yaw), sin(yaw))``. Dotting a shoulder-relative vector with that axis
    recovers anatomical lateral; the perpendicular dot recovers depth. Vertical
    image Y is unaffected by an upright yaw. At yaw zero this is exactly the
    legacy mapping, so camera-front facing inputs remain bit-for-bit unchanged.
    """
    x, y, z = vector
    if not all(isfinite(value) for value in (x, y, z, yaw_rad)):
        raise ArmRetargetError("human torso frame contains non-finite coordinates")
    return (
        cos(yaw_rad) * x + sin(yaw_rad) * z,
        y,
        -sin(yaw_rad) * x + cos(yaw_rad) * z,
    )


class LeftArmRetargeter:
    """Map a bounded 3D wrist plus elbow hint to one seven-DOF robot arm."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        neutral_pose: Mapping[str, float],
        policy: LeftArmPolicy | None = None,
        side: str = "left",
        tcp_body_name: str | None = None,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.model = model
        self.policy = policy or LeftArmPolicy()
        self.joint_names = CONTROL_GROUPS[f"{side}_arm"]
        self._joint_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        )
        if any(joint_id < 0 for joint_id in self._joint_ids):
            raise ValueError(f"{side}-arm joint is missing from model")
        self._qpos_addrs = tuple(int(model.jnt_qposadr[joint_id]) for joint_id in self._joint_ids)
        self._dof_addrs = tuple(int(model.jnt_dofadr[joint_id]) for joint_id in self._joint_ids)
        tcp_body_name = tcp_body_name or f"{side}_gripper_tcp_link"
        self._tcp_body_name = tcp_body_name
        self._tcp_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tcp_body_name)
        self._elbow_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm_link4"
        )
        self._shoulder_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm_link1"
        )
        if self._tcp_body_id < 0:
            raise ValueError(f"TCP body is missing from model: {tcp_body_name}")
        if self._elbow_body_id < 0:
            raise ValueError(f"elbow body is missing from model for {side} arm")
        if self._shoulder_body_id < 0:
            raise ValueError(f"shoulder body is missing from model for {side} arm")
        self._limits: dict[str, JointLimit] = parse_revolute_joint_limits(
            CANONICAL_MANIFEST.fixed_urdf
        )
        self._neutral_qpos = np.zeros(model.nq, dtype=np.float64)
        for name, value in neutral_pose.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"neutral pose contains unknown joint: {name}")
            self._neutral_qpos[int(model.jnt_qposadr[joint_id])] = float(value)
        self._neutral_arm = self._neutral_qpos[list(self._qpos_addrs)].copy()
        #: The torso yaw joint this arm hangs off. The G1's whole upper body --
        #: both arm mounts and the head -- rides ``leg_joint4``, so when the torso
        #: turns, this arm's shoulder moves and its target frame turns with it.
        #: See ``set_torso_yaw``.
        self._torso_yaw_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, TORSO_YAW_JOINT
        )
        self._torso_yaw_qpos_addr = (
            int(model.jnt_qposadr[self._torso_yaw_joint_id])
            if self._torso_yaw_joint_id >= 0
            else None
        )
        self._torso_yaw_home = (
            float(self._neutral_qpos[self._torso_yaw_qpos_addr])
            if self._torso_yaw_qpos_addr is not None
            else 0.0
        )
        self._torso_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY
        )
        if self._torso_body_id < 0:
            raise ValueError(f"torso body is missing from model: {TORSO_BODY}")
        #: Rotation from the home torso frame to the commanded one, read from
        #: forward kinematics rather than assumed. ``leg_joint4``'s world axis is
        #: [0.001, 0, 1.000] on the canonical model -- close to world Z but not
        #: equal to it, and assuming Z put 0.44 mm of error into an 85 deg turn.
        #: Exactly the identity until ``set_torso_yaw`` moves the joint, so a
        #: pipeline that does not drive the torso is bit-for-bit unaffected.
        self._torso_rotation = np.eye(3, dtype=np.float64)
        self._torso_rotation_home: np.ndarray | None = None
        self._recompute_neutral_frame(model)
        #: Palm orientation the first time it is confidently seen. Everything
        #: afterwards is measured RELATIVE to it, so the operator's hand and the
        #: gripper never have to agree on an absolute convention -- only on how
        #: far the hand has turned since the pose it was calibrated in.
        #: Last accepted IK solution and the wrist target it was solved for, for
        #: branch-flip detection. Only updated when a solve is ACCEPTED, so a
        #: rejected flip is compared against the last good pose rather than
        #: against the flipped one -- otherwise one flip silently rebaselines the
        #: test and the arm walks to the new branch a step at a time.
        self._previous_solution: dict[str, float] | None = None
        self._previous_wrist_target: tuple[float, float, float] | None = None
        self._reference_palm: np.ndarray | None = None
        #: Smoothed palm orientation. MediaPipe's hand landmarks are far noisier
        #: than its pose landmarks: measured on a real capture, the raw palm
        #: frame jitters 40.8 deg at p95 between consecutive frames, against a
        #: genuine 60.7 deg median range of hand rotation. Commanding that raw
        #: would shake the gripper harder than the operator moves it.
        self._smoothed_palm: np.ndarray | None = None
        #: Built on first use so importing this module never requires mink, which
        #: is an optional extra and the only borrowed dependency in the path.
        self._swivel_solver_cached = None
        self._swivel_solver_unavailable = False
        #: Why the swivel mode fell back to the baseline solver, per reason.
        #:
        #: Without this the fallback is invisible: the arm still moves, the frame
        #: is still ALLOW, and "swivel-priority" reports success while having done
        #: nothing. Measured on two real captures, the swivel target was actually
        #: reached on 2% of left-arm frames -- which no existing signal revealed.
        self.swivel_fallbacks: dict[str, int] = {}
        #: Frames where the swivel solve was accepted and used.
        self.swivel_accepted = 0
        # Allocating MjData costs a measured 799.9 us; map() ran once per arm per
        # frame, so two arms paid ~1.60 ms/frame for scratch space alone. Hold one
        # buffer per retargeter, as ClearanceChecker already does. Verified over 25
        # poses that a reused buffer reproduces a fresh one bit-for-bit across
        # xpos, xmat and the body Jacobians, so this is a pure allocation saving.
        # Safe because the pipeline maps one frame at a time and each arm owns its
        # own retargeter; this buffer must not be shared between arms or threads.
        self._data = mujoco.MjData(model)

    def _recompute_neutral_frame(self, model: mujoco.MjModel) -> None:
        """Re-derive every FK-anchored neutral quantity from ``_neutral_qpos``.

        These are the origins the operator's signals are measured from. They are
        world-frame points, so they move whenever anything BELOW this arm in the
        kinematic chain moves -- which, on the G1, means the torso yaw joint.
        """
        data = mujoco.MjData(model)
        data.qpos[:] = self._neutral_qpos
        mujoco.mj_forward(model, data)
        self._neutral_tcp = data.xpos[self._tcp_body_id].copy()
        self._neutral_shoulder = data.xpos[self._shoulder_body_id].copy()
        self._neutral_tcp_rotation = data.xmat[self._tcp_body_id].reshape(3, 3).copy()
        self._neutral_elbow = data.xpos[self._elbow_body_id].copy()
        self._neutral_forearm_direction = self._neutral_tcp - self._neutral_elbow
        self._neutral_forearm_length = float(
            np.linalg.norm(self._neutral_forearm_direction)
        )
        self._neutral_upper_arm_length = float(
            np.linalg.norm(self._neutral_elbow - self._neutral_shoulder)
        )
        torso_rotation = data.xmat[self._torso_body_id].reshape(3, 3).copy()
        if self._torso_rotation_home is None:
            # First call defines "square". Identity by construction, so the
            # untorsoed pipeline is unaffected to the last bit.
            self._torso_rotation_home = torso_rotation
            self._torso_rotation = np.eye(3, dtype=np.float64)
        else:
            self._torso_rotation = torso_rotation @ self._torso_rotation_home.T

    def set_torso_yaw(self, yaw_rad: float) -> None:
        """Carry this arm with the commanded torso yaw.

        The operator's arm signals are measured relative to their own shoulders,
        so they are already torso-relative: turning the body does not change where
        the hand is *with respect to the chest*. The robot must preserve that, and
        it takes two coupled changes, both of which happen here.

        First, the IK seed pose carries the yaw, so forward kinematics and the
        Jacobian describe the arm where it actually is rather than where it would
        be with the torso square. Second, the shoulder-relative offset is rotated
        by the same angle, because the offset is expressed in world axes and
        "forward" for the chest turns with the chest.

        Doing only the first would leave the arm reaching for a world point the
        torso has rotated away from -- which is not a cosmetic error: the wrist
        target would land outside the reachable set, and non-convergence is
        already this solver's dominant failure. Doing only the second would solve
        against the wrong Jacobian. This method is the only supported way to
        change the torso angle for exactly that reason.

        Raises ``ValueError`` on a non-finite angle rather than poisoning the
        neutral frame with a NaN that would surface later as an unexplained
        solver failure.
        """
        if not isfinite(yaw_rad):
            raise ValueError("torso yaw must be finite")
        if self._torso_yaw_qpos_addr is None:
            raise ArmRetargetError(
                f"model has no {TORSO_YAW_JOINT}; torso yaw cannot be commanded"
            )
        if yaw_rad == float(self._neutral_qpos[self._torso_yaw_qpos_addr]):
            return
        self._neutral_qpos[self._torso_yaw_qpos_addr] = float(yaw_rad)
        # Both the moved origins and the frame rotation come out of the one
        # forward-kinematics evaluation inside this call, so they cannot disagree.
        self._recompute_neutral_frame(self.model)

    @property
    def torso_yaw_rad(self) -> float:
        """The torso yaw this arm is currently solving against."""
        if self._torso_yaw_qpos_addr is None:
            return 0.0
        return float(self._neutral_qpos[self._torso_yaw_qpos_addr])

    def map(
        self,
        intent: LeftArmIntent,
        *,
        previous: Mapping[str, float] | None,
        elapsed_s: float | None,
        _position_priority_retry: bool = False,
        _conditioned_position_retry: bool = False,
        _ik_attempts: tuple[IKSolveAttempt, ...] = (),
    ) -> LeftArmMappingResult:
        values = (
            intent.lateral_signal,
            intent.vertical_signal,
            intent.depth_signal,
            intent.elbow_lateral_signal,
            intent.elbow_vertical_signal,
            intent.elbow_depth_signal,
            elapsed_s,
        )
        if any(value is not None and not isfinite(value) for value in values):
            return LeftArmMappingResult(None, ArmMapRejection.NONFINITE)
        if elapsed_s is not None and elapsed_s <= 0:
            return LeftArmMappingResult(None, ArmMapRejection.NONFINITE)

        lateral = max(-self.policy.max_signal, min(self.policy.max_signal, intent.lateral_signal))
        vertical = max(-self.policy.max_signal, min(self.policy.max_signal, intent.vertical_signal))
        depth = max(-self.policy.max_signal, min(self.policy.max_signal, intent.depth_signal))
        lateral_reference = 0.25 if self.side == "left" else -0.25
        current_lateral = lateral + lateral_reference
        # NOTE: raising this to 1.4333 (to make the neutral target coincide with the
        # arms-down rest pose) does unpin `*_arm_joint2`, but it also drops every
        # commanded wrist position by 30 cm, so the operator's hands and the robot's
        # stop corresponding -- hands raised to the head only reach the robot's chest.
        # Reverted. Fixing the shoulder this way requires moving the rest pose and the
        # target together, not the target alone.
        current_vertical = vertical + 0.75
        current_depth = depth
        elbow_signals = np.clip(
            np.array(
                [
                    intent.elbow_depth_signal,
                    intent.elbow_lateral_signal,
                    intent.elbow_vertical_signal,
                ],
                dtype=np.float64,
            ),
            -self.policy.max_signal,
            self.policy.max_signal,
        )
        # The lateral (Y) term is POSITIVE, matching the wrist target above. It was
        # negated, so the wrist and the elbow were driven to opposite sides for the
        # same sideways motion -- the solver split the difference and splayed the arm
        # into a T-pose, elbow level with the hand, while the operator's elbow was
        # tucked down by their ribs. Reported as the robot looking "like a bafoon with
        # its arms sideways", and visible in artifacts/../human_robot_preview8.mp4.
        #
        # X and Z stay negated: those match the wrist convention already (depth is
        # measured toward the torso, and MediaPipe image Y points down).
        if self.policy.mapping_mode == "direction-vector":
            if intent.upper_arm_direction is None or intent.forearm_direction is None:
                return LeftArmMappingResult(None, ArmMapRejection.INCOMPLETE_OBSERVATION)
            upper = _human_direction_to_robot(intent.upper_arm_direction)
            forearm = _human_direction_to_robot(intent.forearm_direction)
            upper = self._torso_rotation @ upper
            forearm = self._torso_rotation @ forearm
            elbow_target = self._neutral_shoulder + self._neutral_upper_arm_length * upper
            # This is a true two-segment candidate: the TCP is derived from the
            # same elbow and forearm direction rather than retaining the baseline
            # wrist-primary Cartesian target as a dominant competing objective.
            target_position = elbow_target + self._neutral_forearm_length * forearm
            target_rotation = (
                _align_vectors(self._neutral_forearm_direction, forearm)
                @ self._neutral_tcp_rotation
            )
            palm_weight = 0.0
        else:
            # Absolute shoulder-anchored wrist mapping. HOME is an unequal safe
            # start pose (the left TCP is 12.5 cm lower than the right), so using
            # each HOME TCP as an origin made equal human hands map unequally.
            # Reaching out (current_depth < 0) has open workspace, so it uses the
            # larger forward scale; pulling chest-ward (current_depth > 0) heads for
            # the self-collision wall and keeps the conservative scale.
            depth_scale = (
                self.policy.depth_scale_m
                if current_depth > 0.0
                else self.policy.depth_forward_scale_m
            )
            # The offsets are expressed in world axes, so they must turn with the
            # torso: "forward" for the chest is not world +X once the body yaws.
            # ``_torso_rotation`` is exactly the identity while the torso is
            # square, so this is a no-op for a pipeline that does not drive it.
            target_position = self._neutral_shoulder + self._torso_rotation @ np.array(
                [
                    self.policy.forward_offset_m - current_depth * depth_scale,
                    current_lateral * self.policy.lateral_scale_m,
                    -current_vertical * self.policy.vertical_scale_m,
                ],
                dtype=np.float64,
            )
            elbow_target = self._neutral_elbow + self._torso_rotation @ np.array(
                [
                    -elbow_signals[0] * self.policy.elbow_scale_m,
                    elbow_signals[1] * self.policy.elbow_scale_m,
                    -elbow_signals[2] * self.policy.elbow_scale_m,
                ],
                dtype=np.float64,
            )
            target_rotation = (
                _align_vectors(
                    self._neutral_forearm_direction,
                    target_position - elbow_target,
                )
                @ self._neutral_tcp_rotation
            )
            # A measured palm replaces that guess. Relative to the first
            # confident sighting, so only the CHANGE in hand orientation is
            # commanded and no absolute hand-to-gripper convention is assumed.
            palm_weight = 0.0
            if intent.palm_rotation is not None and intent.palm_confidence > 0.0:
                measured = np.asarray(intent.palm_rotation, dtype=np.float64).reshape(3, 3)
                palm = _smooth_rotation(
                    self._smoothed_palm, measured, self.policy.palm_smoothing
                )
                self._smoothed_palm = palm
                if self._reference_palm is None:
                    self._reference_palm = palm.copy()
                delta = palm @ self._reference_palm.T
                target_rotation = delta @ self._neutral_tcp_rotation
                palm_weight = self.policy.palm_orientation_weight * intent.palm_confidence
        qpos = self._neutral_qpos.copy()
        if previous is not None:
            for name, address in zip(self.joint_names, self._qpos_addrs, strict=True):
                if name in previous:
                    qpos[address] = previous[name]

        if self.policy.mapping_mode == "swivel-priority":
            swivel_result = self._map_swivel_priority(
                intent,
                seed_qpos=qpos,
                target_position=target_position,
                target_rotation=target_rotation,
                elbow_target=elbow_target,
                previous=previous,
                elapsed_s=elapsed_s,
            )
            if swivel_result is not None:
                return swivel_result
            # The swivel measurement was unavailable or the strict-priority solve
            # did not converge. Fall through to the wrist-primary solver below
            # rather than holding the limb: a correct wrist with a mediocre elbow
            # is strictly better than a frozen arm, and the wrist target computed
            # above is already the wrist-primary one.

        data = self._data
        stage = (
            IKSolveStage.CONDITIONED_POSITION_PRIORITY
            if _conditioned_position_retry
            else IKSolveStage.POSITION_PRIORITY
            if _position_priority_retry
            else IKSolveStage.WARM
            if previous is not None
            else IKSolveStage.NEUTRAL
        )
        wrist_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        wrist_rotation_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        elbow_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)

        for _ in range(self.policy.max_iterations):
            data.qpos[:] = qpos
            _kinematics_only(self.model, data)
            wrist_error = target_position - data.xpos[self._tcp_body_id]
            current_rotation = data.xmat[self._tcp_body_id].reshape(3, 3)
            orientation_error = _rotation_error(
                target_rotation, current_rotation
            )
            elbow_error = elbow_target - data.xpos[self._elbow_body_id]
            residual = float(np.linalg.norm(wrist_error))
            if (
                residual <= self.policy.residual_tolerance_m
                and np.linalg.norm(elbow_error) <= 0.02
                and np.linalg.norm(orientation_error) <= 0.03
            ):
                break
            wrist_jacobian.fill(0.0)
            wrist_rotation_jacobian.fill(0.0)
            elbow_jacobian.fill(0.0)
            mujoco.mj_jacBody(
                self.model,
                data,
                wrist_jacobian,
                wrist_rotation_jacobian,
                self._tcp_body_id,
            )
            mujoco.mj_jacBody(self.model, data, elbow_jacobian, None, self._elbow_body_id)
            elbow_task_weight = (
                min(self.policy.elbow_weight, 0.001)
                if _position_priority_retry
                else self.policy.elbow_weight
            )
            # A measured orientation earns a real weight; the inferred one stays a
            # tie-breaker. The position-priority retry still collapses both, so an
            # unreachable wrist target is never traded for hand alignment.
            base_orientation_weight = max(self.policy.orientation_weight, palm_weight)
            orientation_task_weight = (
                min(base_orientation_weight, 0.00001)
                if _position_priority_retry
                else base_orientation_weight
            )
            weight = elbow_task_weight**0.5
            orientation_weight = orientation_task_weight**0.5
            reduced = np.vstack(
                (
                    wrist_jacobian[:, self._dof_addrs],
                    orientation_weight * wrist_rotation_jacobian[:, self._dof_addrs],
                    weight * elbow_jacobian[:, self._dof_addrs],
                )
            )
            error = np.concatenate(
                (wrist_error, orientation_weight * orientation_error, weight * elbow_error)
            )
            # Position-only IK otherwise spends large portions of a reach in wrist
            # joints 5-7. A small posture objective keeps those unconstrained DOFs
            # near calibration while shoulders and elbows carry the hand motion.
            posture_weights = self.policy.posture_weight * np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    self.policy.distal_posture_multiplier,
                    self.policy.distal_posture_multiplier,
                    self.policy.distal_posture_multiplier,
                ],
                dtype=np.float64,
            )
            arm_qpos = qpos[list(self._qpos_addrs)]
            damping = (
                self.policy.conditioned_retry_damping
                if _conditioned_position_retry
                else self.policy.damping
            )
            lhs = (
                reduced.T @ reduced
                + (damping**2) * np.eye(len(self.joint_names))
                + np.diag(posture_weights)
            )
            rhs = reduced.T @ error + posture_weights * (self._neutral_arm - arm_qpos)
            delta = np.linalg.solve(lhs, rhs)
            delta = np.clip(
                delta,
                -self.policy.max_iteration_step_rad,
                self.policy.max_iteration_step_rad,
            )
            for index, (name, address) in enumerate(
                zip(self.joint_names, self._qpos_addrs, strict=True)
            ):
                limit = self._limits[name]
                qpos[address] = np.clip(
                    qpos[address] + delta[index],
                    limit.lower_rad + self.policy.soft_margin_rad,
                    limit.upper_rad - self.policy.soft_margin_rad,
                )
        else:
            residual = float("inf")

        data.qpos[:] = qpos
        _kinematics_only(self.model, data)
        residual = float(np.linalg.norm(target_position - data.xpos[self._tcp_body_id]))
        elbow_residual = float(np.linalg.norm(elbow_target - data.xpos[self._elbow_body_id]))
        orientation_residual = float(
            np.linalg.norm(
                _rotation_error(target_rotation, data.xmat[self._tcp_body_id].reshape(3, 3))
            )
        )
        ik_attempts = (
            (*_ik_attempts, IKSolveAttempt(stage, residual))
            if isfinite(residual)
            else _ik_attempts
        )
        shape_failed = self.policy.mapping_mode == "direction-vector" and (
            elbow_residual > 0.02 or orientation_residual > 0.03
        )
        if shape_failed:
            # A warm start may sit on the wrong IK branch.  Give the candidate
            # the same legitimate neutral-seed recovery as the baseline, then
            # judge segment residuals on that *chosen* solution.  Do not use the
            # position-priority ladder here: lowering its elbow/orientation task
            # weights would turn an arm-shape experiment into wrist-only data.
            if previous is not None:
                retry = self.map(
                    intent,
                    previous=None,
                    elapsed_s=None,
                    _ik_attempts=ik_attempts,
                )
                if retry.target is None or elapsed_s is None:
                    return retry
                allowed = self.policy.max_joint_rate_rad_s * elapsed_s
                if any(
                    abs(value - previous.get(name, value)) > allowed
                    for name, value in retry.target.joints_rad
                ):
                    return LeftArmMappingResult(
                        None,
                        ArmMapRejection.JOINT_CONTINUITY,
                        best_wrist_residual_m=residual,
                        ik_attempts=ik_attempts,
                    )
                return retry
            # A candidate result is only evidence of arm-shape tracking if both
            # segment constraints converge.  Never silently relax it to the
            # wrist-only fallback used by the production wrist-primary mapper.
            return LeftArmMappingResult(
                None,
                ArmMapRejection.IK_DID_NOT_CONVERGE,
                best_wrist_residual_m=residual,
                ik_attempts=ik_attempts,
            )
        if not isfinite(residual) or residual > self.policy.residual_tolerance_m:
            # Warm-starting from the prior branch is normally faster, but a large
            # human reach can trap DLS in a local minimum. Retry once from the
            # verified neutral seed; the downstream trajectory governor absorbs
            # any branch difference before supervision.
            if previous is not None:
                retry = self.map(
                    intent,
                    previous=None,
                    elapsed_s=None,
                    _ik_attempts=ik_attempts,
                )
                if retry.target is None or elapsed_s is None:
                    return retry
                allowed = self.policy.max_joint_rate_rad_s * elapsed_s
                if any(
                    abs(value - previous.get(name, value)) > allowed
                    for name, value in retry.target.joints_rad
                ):
                    return LeftArmMappingResult(
                        None,
                        ArmMapRejection.JOINT_CONTINUITY,
                        best_wrist_residual_m=residual,
                        ik_attempts=ik_attempts,
                    )
                return retry
            if not _position_priority_retry:
                # A high reach can make the soft elbow/orientation objectives
                # conflict with a wrist position that is itself reachable. Retry
                # once with strict wrist-position priority; downstream limits and
                # swept clearance remain unchanged and still authorize the pose.
                return self.map(
                    intent,
                    previous=None,
                    elapsed_s=None,
                    _position_priority_retry=True,
                    _ik_attempts=ik_attempts,
                )
            if not _conditioned_position_retry:
                # Preserve the established position-priority solve first.  Its
                # failed residual proves a conditioning retry is necessary; only
                # then use the separately measured stronger DLS regularisation.
                return self.map(
                    intent,
                    previous=None,
                    elapsed_s=None,
                    _position_priority_retry=True,
                    _conditioned_position_retry=True,
                    _ik_attempts=ik_attempts,
                )
            return LeftArmMappingResult(
                None,
                ArmMapRejection.IK_DID_NOT_CONVERGE,
                best_wrist_residual_m=residual,
                ik_attempts=ik_attempts,
            )

        joints = tuple(
            (name, float(qpos[address]))
            for name, address in zip(self.joint_names, self._qpos_addrs, strict=True)
        )
        if previous is not None and elapsed_s is not None:
            allowed = self.policy.max_joint_rate_rad_s * elapsed_s
            if any(abs(value - previous.get(name, value)) > allowed for name, value in joints):
                return LeftArmMappingResult(
                    None,
                    ArmMapRejection.JOINT_CONTINUITY,
                    best_wrist_residual_m=residual,
                    ik_attempts=ik_attempts,
                )
        return self._accept(
            LeftArmMappingResult(
                LeftArmTarget(
                    joints_rad=joints,
                    wrist_target_world_m=tuple(
                        float(value) for value in target_position
                    ),
                    elbow_target_world_m=tuple(float(value) for value in elbow_target),
                    residual_m=residual,
                ),
                best_wrist_residual_m=residual,
                ik_attempts=ik_attempts,
            )
        )

    def _accept(self, result: LeftArmMappingResult) -> LeftArmMappingResult:
        """Last gate before a solution becomes a target: reject a branch flip.

        I5 in docs/safety-model.md. Compares this solution with the previous
        ACCEPTED one in joint space, against how far the wrist target actually
        moved. A large joint move for a negligible task move is the arm taking a
        different route to the same place, and the correct response is to hold --
        the operator did not ask for that arc, and every task-space check the
        supervisor runs would pass it.
        """
        target = result.target
        if target is None:
            return result
        joints = dict(target.joints_rad)
        wrist = target.wrist_target_world_m
        if (
            self._previous_solution is not None
            and self._previous_wrist_target is not None
            and wrist is not None
        ):
            joint_delta = max(
                (abs(value - self._previous_solution.get(name, value))
                 for name, value in joints.items()),
                default=0.0,
            )
            target_delta = sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(wrist, self._previous_wrist_target)
                )
            )
            if (
                joint_delta > self.policy.branch_flip_joint_rad
                and target_delta < self.policy.branch_flip_target_m
            ):
                # Deliberately does NOT update the previous solution: the next
                # frame must still be judged against the last pose the operator
                # actually asked for.
                return LeftArmMappingResult(
                    None,
                    ArmMapRejection.JOINT_CONTINUITY,
                    best_wrist_residual_m=result.best_wrist_residual_m,
                    ik_attempts=result.ik_attempts,
                )
        self._previous_solution = joints
        self._previous_wrist_target = wrist
        return result

    def _map_swivel_priority(
        self,
        intent: LeftArmIntent,
        *,
        seed_qpos: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        elbow_target: np.ndarray,
        previous: Mapping[str, float] | None,
        elapsed_s: float | None,
    ) -> LeftArmMappingResult | None:
        """Hold the wrist target exactly and place the elbow with the free DOF.

        Returns ``None`` to mean "no swivel answer this frame, use the baseline
        solver", which is why every failure path below returns ``None`` rather
        than a rejection.

        The wrist target is bit-identical to the wrist-primary one -- it is passed
        in already computed. This mode changes only which arm configuration is
        chosen from the infinitely many that reach that same wrist pose.
        """
        if intent.swivel_angle_rad is None or intent.swivel_confidence <= 0.0:
            return self._swivel_fallback("no_operator_swivel")
        solver = self._swivel_solver()
        if solver is None:
            return self._swivel_fallback("solver_unavailable")

        # Both psi are now measured about their own shoulder->wrist axis from
        # their own body's outward lateral direction. That is the SAME definition
        # in both bodies, so the operator's psi can be commanded directly. psi is
        # dimensionless, so no human/robot limb-length scaling is needed either.
        #
        # An earlier relative version -- mapping only the CHANGE from the first
        # confident sighting -- is what the different references forced, and it
        # preserved the constant offset that splays the elbow. Measured on the
        # retained capture it recovered only 3.14 deg of a 38.53 deg error.
        target_swivel = self.policy.swivel_gain * float(intent.swivel_angle_rad)

        try:
            solution = solver.solve(
                seed_qpos=seed_qpos,
                wrist_position_m=target_position,
                wrist_rotation=target_rotation,
                target_swivel_rad=target_swivel,
            )
        except Exception as error:
            # mink/qpsolvers raising anything here must not take the live session
            # down mid-demo. The baseline solver still produces a pose. The
            # exception TYPE is retained: a silent except is how a mode that never
            # engages still reports success.
            return self._swivel_fallback(f"solver_raised:{type(error).__name__}")

        # The wrist is the product claim; the swivel is the improvement. Accept
        # only if the wrist is as good as the baseline is required to be. A
        # non-converged SWIVEL is fine -- the elbow simply lands partway -- but a
        # missed wrist is not.
        if (
            not isfinite(solution.wrist_residual_m)
            or solution.wrist_residual_m > self.policy.residual_tolerance_m
        ):
            return self._swivel_fallback("wrist_residual_too_large")

        joints = []
        for name, value in solution.joints_rad:
            limit = self._limits[name]
            lower = limit.lower_rad + self.policy.soft_margin_rad
            upper = limit.upper_rad - self.policy.soft_margin_rad
            if not isfinite(value) or value < lower - 1e-9 or value > upper + 1e-9:
                return self._swivel_fallback("joint_limit")
            joints.append((name, float(np.clip(value, lower, upper))))
        joints = tuple(joints)

        if previous is not None and elapsed_s is not None:
            allowed = self.policy.max_joint_rate_rad_s * elapsed_s
            if any(abs(value - previous.get(name, value)) > allowed for name, value in joints):
                return self._swivel_fallback("joint_rate_limit")

        self.swivel_accepted += 1
        if not solution.converged:
            # Accepted, but the elbow only got partway. The wrist is exact either
            # way; this counts how often the SHAPE claim is only partly delivered.
            self.swivel_fallbacks["accepted_swivel_not_converged"] = (
                self.swivel_fallbacks.get("accepted_swivel_not_converged", 0) + 1
            )
        return self._accept(
            LeftArmMappingResult(
                LeftArmTarget(
                    joints_rad=joints,
                    wrist_target_world_m=tuple(
                        float(value) for value in target_position
                    ),
                    elbow_target_world_m=tuple(float(value) for value in elbow_target),
                    residual_m=float(solution.wrist_residual_m),
                ),
                best_wrist_residual_m=float(solution.wrist_residual_m),
            )
        )

    def _swivel_fallback(self, reason: str) -> None:
        """Record why this frame did not use the swivel solve. Returns None."""
        self.swivel_fallbacks[reason] = self.swivel_fallbacks.get(reason, 0) + 1
        return None

    def _swivel_solver(self):
        """Build the mink-backed solver once, or ``None`` if mink is absent."""
        if self._swivel_solver_cached is not None:
            return self._swivel_solver_cached
        if self._swivel_solver_unavailable:
            return None
        try:
            from galbot_motion_studio.retarget.swivel_solver_mink import SwivelArmSolver

            self._swivel_solver_cached = SwivelArmSolver(
                self.model,
                joint_names=self.joint_names,
                tcp_body_id=self._tcp_body_id,
                elbow_body_id=self._elbow_body_id,
                shoulder_body_id=self._shoulder_body_id,
                tcp_frame_name=self._tcp_body_name,
                orientation_cost=self.policy.orientation_weight,
                # One authority for the band: the same number this class rejects
                # against below, so the solver cannot return a pose the caller
                # is guaranteed to discard.
                soft_margin_rad=self.policy.soft_margin_rad,
                # The robot-side twin of the operator's shoulder-line reference:
                # this arm's OUTWARD lateral direction. With both psi measured
                # from the same anatomical zero, the two are directly comparable
                # and an absolute mapping is meaningful. Measured from world-down
                # instead, they share no zero, only changes can be mapped, and the
                # large constant offset that causes the splayed elbow survives.
                psi_reference=self._outward_lateral(),
                psi_fallback_reference=np.array([0.0, 0.0, -1.0]),
            )
        except Exception:
            self._swivel_solver_unavailable = True
            return None
        return self._swivel_solver_cached

    def _outward_lateral(self) -> np.ndarray:
        """This arm's outward direction in robot axes: +Y left, -Y right.

        The operator's psi is measured from the direction pointing out of their
        torso toward THIS shoulder. This is its robot counterpart, and the two
        must agree for an absolute swivel mapping to mean anything.
        """
        return np.array([0.0, 1.0 if self.side == "left" else -1.0, 0.0])


def _landmarks(
    observation: HumanObservation,
    names: tuple[str, ...],
) -> dict[str, Landmark]:
    by_name = {landmark.name: landmark for landmark in observation.landmarks}
    try:
        return {name: by_name[name] for name in names}
    except KeyError as error:
        raise ArmRetargetError(f"arm observation is missing {error.args[0]}") from error


class RightArmRetargeter(LeftArmRetargeter):
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        neutral_pose: Mapping[str, float],
        policy: LeftArmPolicy | None = None,
    ) -> None:
        super().__init__(model, neutral_pose=neutral_pose, policy=policy, side="right")


def _arm_landmark_names(side: str) -> tuple[str, str, str]:
    if side == "left":
        return ("pose_11", "pose_13", "pose_15")
    if side == "right":
        return ("pose_12", "pose_14", "pose_16")
    raise ArmRetargetError("arm side must be left or right")


def _coordinates(landmark: Landmark, coordinate_space: str) -> tuple[float, float, float]:
    if coordinate_space == "normalized":
        return landmark.normalized_xyz
    if coordinate_space == "world_m" and landmark.world_xyz_m is not None:
        return landmark.world_xyz_m
    raise ArmRetargetError(f"arm observation is missing {coordinate_space} coordinates")


def _coordinate_distance(left: Landmark, right: Landmark, coordinate_space: str) -> float:
    left_xyz = _coordinates(left, coordinate_space)
    right_xyz = _coordinates(right, coordinate_space)
    return sqrt(sum((a - b) ** 2 for a, b in zip(left_xyz, right_xyz, strict=True)))


@dataclass(frozen=True)
class _TorsoFrame:
    """A camera-independent, anatomical coordinate system for segment vectors.

    The two shoulders establish anatomical left/right; the shoulder/hip midpoints
    establish inferior.  Their cross product is the torso normal.  The normal's
    sign is chosen with the *camera depth convention only* to resolve the
    reflection ambiguity of an image mirror.  No camera axis is projected into
    robot space: arm vectors are expressed in this basis before retargeting.
    """

    lateral: np.ndarray
    inferior: np.ndarray
    forward: np.ndarray

    def project(
        self,
        origin: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> tuple[float, float, float] | None:
        direction = _unit_direction(origin, target)
        if direction is None:
            return None
        vector = np.asarray(direction, dtype=np.float64)
        local = np.asarray(
            (
                float(np.dot(vector, self.lateral)),
                float(np.dot(vector, self.inferior)),
                float(np.dot(vector, self.forward)),
            ),
            dtype=np.float64,
        )
        length = float(np.linalg.norm(local))
        if not isfinite(length) or length < 1e-8:
            return None
        return tuple(float(value) for value in local / length)


#: Fallback psi reference, used only when the operator's arm points almost
#: exactly along their own shoulder line and the primary reference projects to
#: nothing. MediaPipe world Y points down, so this is anatomical "up".
_HUMAN_PSI_FALLBACK = np.array([0.0, -1.0, 0.0])


def _human_swivel_angle(
    observation: HumanObservation,
    *,
    side: str,
    min_confidence: float,
) -> tuple[float, float] | None:
    """Measure the operator's swivel angle from the shoulder line. No hips.

    Returns ``(angle_rad, confidence)`` or ``None`` when the arm is not
    confidently observed.

    This deliberately does NOT use :func:`_torso_frame`. That function needs
    hips to identify a full 3-D body frame, and it is right to: a direction
    vector expressed in a shoulder-only frame would turn a camera yaw into a
    robot fore/aft command. But psi is not a direction in a frame. It is the
    dihedral angle between the shoulder-line half-plane and the elbow half-plane
    about the shoulder->wrist axis -- a scalar property of four points, invariant
    to any rigid rotation of all of them, and needing only a reference direction
    rather than a complete basis.

    ``world_xyz_m`` is used rather than ``normalized_xyz`` because psi is a true
    3-D angle and image-normalised coordinates carry an aspect-ratio and depth
    scaling that would distort it. MediaPipe's world landmarks are hip-CENTRED,
    but every term here is a difference of two points, so that origin cancels
    and no hip needs to have been observed.
    """
    shoulder_name = f"pose_{11 if side == 'left' else 12}"
    elbow_name = f"pose_{13 if side == 'left' else 14}"
    wrist_name = f"pose_{15 if side == 'left' else 16}"
    opposite_name = f"pose_{12 if side == 'left' else 11}"

    try:
        landmarks = _landmarks(
            observation, (shoulder_name, elbow_name, wrist_name, opposite_name)
        )
    except ArmRetargetError:
        return None
    if any(
        min(landmark.visibility, landmark.presence) < min_confidence
        for landmark in landmarks.values()
    ):
        return None

    def world(name: str) -> np.ndarray | None:
        coordinates = landmarks[name].world_xyz_m
        if coordinates is None:
            return None
        point = np.asarray(coordinates, dtype=np.float64)
        return point if np.all(np.isfinite(point)) else None

    points = {name: world(name) for name in landmarks}
    if any(point is None for point in points.values()):
        return None

    # The reference points from the opposite shoulder to this one, so each arm
    # measures psi from its OWN outward direction. Using one shared direction for
    # both puts the right arm's psi on the +/-180 branch cut, where it wraps 360
    # deg mid-gesture: measured range 358.1 deg shared against 95.1 deg per-side.
    reference = points[shoulder_name] - points[opposite_name]

    geometry = swivel_geometry(
        points[shoulder_name],
        points[elbow_name],
        points[wrist_name],
        reference=reference,
        fallback_reference=_HUMAN_PSI_FALLBACK,
    )
    if geometry is None:
        return None
    return float(geometry.angle_rad), float(geometry.fade)


def _torso_frame(
    observation: HumanObservation,
    *,
    coordinate_space: str,
    min_confidence: float,
) -> _TorsoFrame | None:
    """Build the candidate's torso frame, or return ``None`` to hold safely.

    A shoulder pair alone cannot identify a full 3D body frame.  In particular,
    using its raw camera normal silently turns a camera yaw or mirror into a robot
    fore/aft command.  Hips are therefore mandatory for the direction-vector
    candidate; the established wrist-primary mapper never calls this function.
    """
    try:
        landmarks = _landmarks(observation, ("pose_11", "pose_12", "pose_23", "pose_24"))
        if any(
            min(landmark.visibility, landmark.presence) < min_confidence
            for landmark in landmarks.values()
        ):
            return None
        left_shoulder = np.asarray(
            _coordinates(landmarks["pose_11"], coordinate_space), dtype=np.float64
        )
        right_shoulder = np.asarray(
            _coordinates(landmarks["pose_12"], coordinate_space), dtype=np.float64
        )
        left_hip = np.asarray(
            _coordinates(landmarks["pose_23"], coordinate_space), dtype=np.float64
        )
        right_hip = np.asarray(
            _coordinates(landmarks["pose_24"], coordinate_space), dtype=np.float64
        )
    except ArmRetargetError:
        return None

    lateral = _normalise_vector(left_shoulder - right_shoulder)
    shoulder_midpoint = 0.5 * (left_shoulder + right_shoulder)
    hip_midpoint = 0.5 * (left_hip + right_hip)
    # Gram-Schmidt makes the frame well-defined when an operator is leaning or
    # one detector point is slightly asymmetric.  It must still have a genuine
    # non-collinear shoulder-to-hip component to be trusted.
    inferior = hip_midpoint - shoulder_midpoint
    if lateral is None:
        return None
    inferior = inferior - float(np.dot(inferior, lateral)) * lateral
    inferior = _normalise_vector(inferior)
    if inferior is None:
        return None
    forward = _normalise_vector(np.cross(lateral, inferior))
    if forward is None:
        return None
    # Screen mirroring is a reflection, so cross(lateral, inferior) reverses.
    # MediaPipe's camera depth convention is stable through an image mirror; use
    # it only to choose the handedness of the torso frame.  This does not map a
    # raw camera coordinate into the robot, and makes the final local vector
    # mirror-invariant.  A torso facing exactly sideways is ambiguous, so reject
    # it instead of allowing a discontinuous fore/aft flip.
    camera_depth_alignment = float(forward[2])
    if not isfinite(camera_depth_alignment) or abs(camera_depth_alignment) < 1e-4:
        return None
    if camera_depth_alignment < 0:
        forward = -forward
    return _TorsoFrame(lateral=lateral, inferior=inferior, forward=forward)


def _normalise_vector(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if not isfinite(length) or length < 1e-8:
        return None
    return vector / length


def _unit_direction(
    origin: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """Return a finite unit vector, or ``None`` for coincident landmarks."""
    vector = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not isfinite(length) or length < 1e-8:
        return None
    return tuple(float(value) for value in vector / length)


def _human_direction_to_robot(direction: tuple[float, float, float]) -> np.ndarray:
    """Map torso-relative human axes to the robot shoulder frame.

    The input is ``(anatomical-left, inferior, signed-torso-normal)``, not raw
    MediaPipe camera coordinates. Keeping this transform here makes the candidate
    explicit and independently testable rather than hiding it in a gain/sign
    tweak.
    """
    x, y, z = direction
    mapped = np.asarray((-z, x, -y), dtype=np.float64)
    length = float(np.linalg.norm(mapped))
    if not isfinite(length) or length < 1e-8:
        raise ArmRetargetError("arm direction is degenerate")
    return mapped / length


def _kinematics_only(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Position kinematics and the quantities ``mj_jacBody`` needs, nothing else.

    ``mj_forward`` also runs the dynamics pipeline, which this solver never reads:
    it only touches ``xpos``, ``xmat`` and body Jacobians. Measured on the pinned
    model, ``mj_forward`` costs 229.7 us against 6.2 us here, and the resulting
    ``xpos`` and Jacobians are bit-identical over 25 poses (max abs difference
    exactly 0.0). At up to 80 iterations per arm this is the dominant cost of a
    frame, so the saving is structural rather than cosmetic.
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


#: MediaPipe axes -> robot axes, as a matrix. Same convention as
#: ``_human_direction_to_robot``: (x, y, z) -> (-z, x, -y).
_AXIS_PERMUTATION = np.array(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def _smooth_rotation(
    previous: np.ndarray | None, measured: np.ndarray, blend: float
) -> np.ndarray:
    """Exponentially smooth a rotation, staying on SO(3).

    Blending two rotation matrices elementwise leaves SO(3) -- the result is not
    orthonormal and its determinant drifts -- so the blend is projected back with
    an SVD. That is the nearest true rotation to the blended matrix, and the
    determinant is forced positive so a noisy frame can never flip the hand
    inside out.
    """
    if previous is None:
        return measured.copy()
    blended = (1.0 - blend) * previous + blend * measured
    u, _, vt = np.linalg.svd(blended)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``current`` to ``target``."""
    difference = target @ current.T
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, difference.reshape(-1))
    # q and -q encode the same rotation. The positive-scalar representative is
    # the shortest rotation and remains well-conditioned at exactly 180 degrees,
    # where the trace/skew formula divides by sin(pi).
    if quaternion[0] < 0:
        quaternion *= -1.0
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1e-10:
        return 2.0 * quaternion[1:]
    angle = 2.0 * atan2(vector_norm, float(quaternion[0]))
    return angle * quaternion[1:] / vector_norm


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shortest rotation aligning one non-zero direction with another."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-9:
        if cosine > 0:
            return np.eye(3)
        basis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    angle = acos(cosine)
    return np.eye(3) + sin(angle) * skew + (1.0 - cosine) * (skew @ skew)
