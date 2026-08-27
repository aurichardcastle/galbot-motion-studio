"""Runnable, hardware-disabled observation-to-MuJoCo motion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from json import dumps
from math import isfinite, sqrt
from typing import Mapping, Sequence

import mujoco

from galbot_motion_studio.adapters.mujoco_preview import MujocoPreviewSink
from galbot_motion_studio.contracts.core import JointTarget, RobotTarget, SafetyDecision
from galbot_motion_studio.contracts.human import HumanObservation, IdentityState
from galbot_motion_studio.model.joint_map import parse_revolute_joint_limits
from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST
from galbot_motion_studio.ports.command import CommandReceipt, CommandSink
from galbot_motion_studio.retarget.face_pose import FacePoseError, estimate_face_pose
from galbot_motion_studio.retarget.head import (
    HeadMappingPolicy,
    HeadRetargeter,
    HeadTarget,
)
from galbot_motion_studio.retarget.filtering import IntentFilterBank
from galbot_motion_studio.retarget.hand import (
    GRIPPER_OPEN_RAD,
    estimate_hand_openness,
    openness_to_gripper_rad,
)
from galbot_motion_studio.retarget.left_arm import (
    ArmMapRejection,
    ArmRetargetError,
    LeftArmCalibration,
    LeftArmPolicy,
    LeftArmRetargeter,
    RightArmRetargeter,
    create_left_arm_calibration,
    create_right_arm_calibration,
    estimate_left_arm_intent,
    estimate_right_arm_intent,
)
from galbot_motion_studio.safety.clearance import ClearanceChecker, HOME_QPOS
from galbot_motion_studio.safety.profiles import clearance_floor_for
from galbot_motion_studio.safety.profiles import (
    MotionProfile,
    clearance_kwargs_for,
    policy_for,
)
from galbot_motion_studio.safety.supervisor import (
    SafetySupervisor,
    SupervisorPolicy,
    SupervisorState,
)
from galbot_motion_studio.retarget.torso import (
    TORSO_YAW_JOINT,
    TorsoIntent,
    TorsoMappingPolicy,
    TorsoRetargeter,
    TorsoTarget,
    shoulder_yaw_signal,
    stable_yaw_reference,
    wrapped_yaw_difference,
)
from galbot_motion_studio.retarget.trajectory import (
    JointTrajectoryGovernor,
    TrajectoryLimits,
    feasible_velocity_limit,
)
from galbot_motion_studio.retarget.guarded_controller import GuardedPathController
from galbot_motion_studio.vision.calibration import (
    CalibrationError,
    CalibrationWindowPolicy,
    NeutralCalibration,
    create_neutral_calibration,
    create_neutral_calibration_window,
)
from galbot_motion_studio.vision.freshness import (
    COMMAND_CONTROL_GROUPS,
    ControlGroup,
    FreshnessPolicy,
    ObservationGate,
    ObservationRejection,
)
from galbot_motion_studio.vision.liveness import FrameLivenessMonitor, LivenessPolicy
from galbot_motion_studio.vision.selection import (
    OccupancyObservation,
    OperatorSelectionManager,
    SelectionDecision,
)


MAPPING_CONFIG = {
    "version": "bilateral-normalized-v4-torso-local-arm-yaw",
    # These values are duplicated from LeftArmPolicy and HeadMappingPolicy and are
    # hashed into the recorded dataset fingerprint. They do not drive the retargeters
    # (the pipeline constructs those with their dataclass defaults), so drift here
    # would make a recording attest to values that were never used.
    # tests/unit/test_mapping_config.py pins this dict to those defaults.
    "head": {
        "yaw_gain": 1.4,
        "pitch_up_gain": 1.2,
        "pitch_down_gain": 0.5,
        "max_rate_rad_s": 0.25,
    },
    # The torso is the one place a target IS derived from pose-world landmarks,
    # and it is declared here rather than hidden. The reason it is admissible
    # where the arms' is not: yaw is an ANGLE of the shoulder line, so it needs no
    # metric-accuracy study to be meaningful -- a scale error in the landmarks
    # cancels in the ratio. The arms consume world coordinates as a LENGTH, which
    # is exactly the claim that is still unvalidated.
    "torso": {
        "joint": "leg_joint4",
        "axis": "upper-body-yaw",
        "input": "pose-world-shoulder-line-angle",
        "coordinates": "pose-world-metric-angle-only",
        "yaw_gain": 1.0,
        "deadband_rad": 0.05,
        "min_shoulder_width_m": 0.12,
        "max_camera_yaw_rad": 1.3962634015954636,
        "max_relative_yaw_rad": 1.3089969389957472,
        "max_neutral_camera_yaw_rad": 0.08726646259971647,
        "max_neutral_spread_rad": 0.12,
        "max_observation_gap_ns": 750000000,
        "max_observation_step_rad": 0.3490658503988659,
        "max_rate_rad_s": 0.35,
        "out_of_range": "saturate-into-soft-band",
        "outside-camera-observable-range": "hold-torso",
        "neutral_reference": "stable-circular-mean-world-shoulder-yaw-at-calibration",
        "front_evidence": "face-mesh-eye-aperture-relative-to-neutral-required-for-torso-direction",
        "arm_coupling": "arms-solve-against-realized-torso-yaw",
        "missing_or_unusable_shoulders": "hold-torso-with-reason",
    },
    "arms": {
        "sides": ["left", "right"],
        # V1 targets are scaled from shoulder-relative *normalized* pose
        # offsets. Pose-world landmarks remain diagnostic/analysis evidence until
        # a pre-registered metric-accuracy study passes; no target can quietly
        # switch units because an adapter happens to supply world coordinates.
        "coordinates": "shoulder-relative-normalized-torso-local-offset",
        "pose_world_coordinates": "diagnostic-only-unvalidated",
        "human_torso_yaw_correction": (
            "inverse pose-world shoulder-yaw before realized robot tummy-yaw"
        ),
        "wrist_origin": "robot-shoulder-pivot",
        "forward_offset_m": 0.30,
        "lateral_scale_m": 0.28,
        "vertical_scale_m": 0.45,
        "depth_scale_m": 0.12,
        "depth_forward_scale_m": 0.24,
        "elbow_scale_m": 0.50,
        "elbow_hint": {"enabled": True, "weight": 0.02},
        "pose_reference": "canonical-relaxed-arms",
        "neutral_posture_regularization": True,
        "tcp_orientation": {"mode": "forearm-aligned", "weight": 0.001},
        "unreachable_task_retry": "wrist-position-priority",
        # A second, stronger-conditioned pass is permitted only after both the
        # established primary and position-priority solves miss the 10 mm bar.
        # Every numeric value here is pinned to LeftArmPolicy below so the hash
        # describes the policy actually used to command the twin.
        "ik": {
            "residual_tolerance_m": 0.01,
            "damping": 0.04,
            "conditioned_retry_damping": 0.10,
        },
        "max_shoulder_normalized_signal": 2.0,
        "ik_joint_limit_margin_rad": 0.11,
    },
    "grippers": {
        "input": "five-finger-extension-angle",
        "mapping": "continuous-open-to-closed",
        "missing_hand": "hold-last-approved-gripper",
    },
    "filter": "one-euro",
    "trajectory": "velocity-and-acceleration-governed",
}


#: How far ABOVE the profile's hard clearance floor the guarded controller aims
#: when it has to back off. Provisional pending the standoff sweep; see
#: SESSION_2026-08-22.md. Measured motivation: with no standoff, 28.6% of allowed
#: frames on the 2026-08-22 trial sat at exactly the 5.000 mm floor and p25 was
#: 5.0000 mm, so the commanded pose carried no margin whatsoever against the
#: value that would have been rejected.
CLEARANCE_STANDOFF_M = 0.0015


@dataclass(frozen=True)
class _SourceContinuity:
    """Raw shoulder evidence gathered at detector cadence, not command cadence.

    The control worker runs behind a latest-wins queue, so an observation the
    detector fully computed can be discarded before ``process`` ever sees it.
    Continuity measured on the surviving stream therefore reports worker
    scheduling latency as missing data.  This carries the detector's own view so
    the gate can tell a genuine blind interval from a busy solver.

    ``break_capture_ns`` is a monotone mark rather than a skipped sample: a real
    dropout must stay enforced even when the frame that ended it was itself
    dropped, so the break is remembered until the gate has acted on it.
    """

    capture_ns: int
    yaw_rad: float
    #: Capture stamp of the frame that ended the most recent break, or ``None``.
    break_capture_ns: int | None
    #: ``TORSO_YAW_CONTINUITY_GAP`` or ``TORSO_YAW_DISCONTINUITY``.
    break_reason: str | None


#: Frame time the look-ahead horizon is sized for.  Covers the measured p95
#: control-worker period (~0.3 s) with margin; a slower frame only makes the
#: horizon conservative, never unsafe.
_LOOKAHEAD_NOMINAL_DT_S = 0.35


#: Consecutive frames of provable, continuous torso yaw required to clear a
#: latched `TORSO_YAW_RECALIBRATION_REQUIRED` in-run. Fifteen, deliberately the
#: same count the initial calibration window asks for -- the bar to RESUME is the
#: bar to START.
TORSO_RECOVERY_FRAMES = 15
#: A recovery frame's yaw step must be well inside the discontinuity threshold,
#: not merely under it: recovery is a claim that the chain is calm again, and a
#: run of 19-degree steps is not calm.
TORSO_RECOVERY_STEP_FRACTION = 0.5


def _lookahead_for(limits: TrajectoryLimits) -> float:
    """Swept-check horizon that is never the binding constraint on speed.

    The horizon exists to bound how far a swept check has to refine, not to
    slow the arm.  Sizing it to one governor step made it the de-facto velocity
    limit (see the call site).  Full braking distance plus one nominal frame of
    travel puts it clear of the governor's own planning, so the declared caps
    bind instead.
    """
    velocity = limits.max_velocity_rad_s
    acceleration = limits.max_acceleration_rad_s2
    braking = velocity * velocity / (2.0 * acceleration)
    step = limits.max_position_step_rad or 0.0
    return step + braking + velocity * _LOOKAHEAD_NOMINAL_DT_S


def _observed_torso_yaw(observation: HumanObservation) -> float | None:
    """Operator torso yaw from the shoulder line, or ``None`` when unusable.

    Requires the metric ``world_xyz_m`` landmarks. They are optional on the
    contract, so an adapter that supplies only normalized coordinates simply does
    not drive the torso -- it must not drive it from the normalized line instead,
    because that shortens both when the operator turns and when they lean back,
    and those are different commands.
    """
    landmarks = {mark.name: mark for mark in observation.landmarks}
    left = landmarks.get("pose_11")
    right = landmarks.get("pose_12")
    if left is None or right is None:
        return None
    if left.world_xyz_m is None or right.world_xyz_m is None:
        return None
    return shoulder_yaw_signal(left.world_xyz_m, right.world_xyz_m)


def _front_head_evidence_available(
    observation: HumanObservation, calibration: NeutralCalibration | None
) -> bool:
    """Whether shoulder-line direction has an independent front-facing cue.

    A shoulder segment is unoriented: exchanging left/right shifts its angle by
    pi. The face mesh aperture is independent evidence that the operator is
    still front-observable. Pose-head fallback can drive the head, but it shares
    the pose tracker with the shoulders and cannot resolve that ambiguity.
    """
    landmarks = {mark.name: mark for mark in observation.landmarks}
    required = tuple(
        landmarks.get(name) for name in ("face_1", "face_33", "face_263")
    )
    if not all(
        landmark is not None
        and all(isfinite(value) for value in landmark.normalized_xyz)
        for landmark in required
    ):
        return False
    if calibration is None:
        return False
    _nose, left_eye, right_eye = required
    assert left_eye is not None and right_eye is not None
    # MediaPipe face landmarks do not carry measured visibility/presence; the
    # adapter synthesizes those fields when a landmark exists. A confidence gate
    # would therefore prove only mesh presence. A profile/back view collapses
    # the horizontal eye aperture; compare it to this operator's neutral mesh
    # rather than to a global pixel threshold. Nose position is intentionally
    # not used: a real head yaw moves it outside the eye span well before a body
    # turn becomes ambiguous, which would falsely freeze tummy/arms for normal
    # head gestures.
    eye_span = abs(right_eye.normalized_xyz[0] - left_eye.normalized_xyz[0])
    return eye_span >= min(0.02, 0.5 * calibration.eye_span_normalized)


def _calibrated_torso_yaw_reference(
    observations: Sequence[HumanObservation], policy: TorsoMappingPolicy
) -> float | None:
    """Return a stable square-stance torso reference, or intentionally none.

    Head and arm calibration has useful normalized-coordinate fallbacks.  Torso
    yaw does not: accepting a partial or off-axis world-landmark window would
    create an arbitrary zero that makes every later turn wrong.  A missing
    reference therefore disables only torso yaw until a suitable recalibration.
    """
    samples = tuple(_observed_torso_yaw(observation) for observation in observations)
    if any(sample is None for sample in samples):
        return None
    return stable_yaw_reference(
        tuple(sample for sample in samples if sample is not None),
        max_absolute_yaw_rad=policy.max_neutral_camera_yaw_rad,
        max_spread_rad=policy.max_neutral_spread_rad,
    )


def _dynamically_feasible_velocity_limit(
    requested_velocity_rad_s: float,
    *,
    max_acceleration_rad_s2: float,
    max_position_step_rad: float,
) -> float:
    """Cap a profile's requested governor speed to what its envelope allows.

    The bound itself belongs to the governor it constrains and is derived there;
    see `feasible_velocity_limit`.  This only applies it to a *requested* rate,
    which for a conservative profile may already be the slower of the two.
    """
    return min(
        requested_velocity_rad_s,
        feasible_velocity_limit(
            max_acceleration_rad_s2=max_acceleration_rad_s2,
            max_position_step_rad=max_position_step_rad,
        ),
    )


MAPPING_HASH = sha256(
    dumps(MAPPING_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def mapping_config_for(arm_mapping: str) -> dict:
    """Return the exact mapping configuration attested by a recording."""
    if arm_mapping == "wrist-primary":
        return MAPPING_CONFIG
    if arm_mapping == "direction-vector":
        return {
            **MAPPING_CONFIG,
            "version": "direction-vector-arm-shape-ab-v4-calibrated-torso-yaw",
            "arms": {
                **MAPPING_CONFIG["arms"],
                "strategy": "torso-relative-upper-arm-and-forearm-direction-vectors",
                "wrist_target": "elbow-plus-neutral-forearm-vector",
                "missing_elbow": "hold-that-limb",
                "torso_frame": {
                    "lateral": "left-shoulder-minus-right-shoulder",
                    "inferior": "shoulder-midpoint-to-hip-midpoint",
                    "normal": "cross-lateral-inferior-camera-depth-signed",
                    "missing_or_degenerate": "hold-that-limb",
                },
            },
        }
    if arm_mapping == "swivel-priority":
        return {
            **MAPPING_CONFIG,
            "version": "swivel-priority-strict-wrist-v2-calibrated-torso-yaw",
            "arms": {
                **MAPPING_CONFIG["arms"],
                # The wrist target is the wrist-primary one, unchanged. Only the
                # choice of arm configuration reaching it differs.
                "strategy": "wrist-primary-target-with-swivel-resolved-redundancy",
                "wrist_target": "absolute-shoulder-anchored",
                "redundancy": "swivel-angle-about-shoulder-wrist-axis",
                "swivel_reference": "opposite-shoulder-to-this-shoulder",
                "swivel_coordinates": "pose-world-angle-only-unvalidated",
                "swivel_mapping": "relative-to-first-confident-sighting",
                "missing_swivel": "fall-back-to-wrist-primary-solver",
            },
        }
    raise ValueError(
        "arm_mapping must be wrist-primary, direction-vector or swivel-priority"
    )


def mapping_hash_for(arm_mapping: str) -> str:
    """Fingerprint the selected mapper, not merely the shared baseline gains."""
    return sha256(
        dumps(mapping_config_for(arm_mapping), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

# A minimum across seven command-driving points is much harsher than MediaPipe's
# per-detector confidence. On the recorded moving-operator regression, 0.35 keeps
# 239/245 visible-person frames while rejecting the six genuine tracking losses.
# This relaxation is simulator-only; hardware policy retains FreshnessPolicy's
# conservative 0.5 default.
SIM_PREVIEW_MIN_CONFIDENCE = 0.35
#: Arm posture the teleoperator starts from, and the seed the arm IK solves against.
#:
#: HOME is a stowed parking pose, not a pose you can move from. Measured against the
#: 0.11 rad soft margin, it leaves the shoulder-abduction joints with **1.6 deg**
#: (left) and **0.8 deg** (right) of travel -- they sit hard against their limits.
#: `left_arm_joint2` is the only joint that moves the elbow sideways at all (16.3 cm
#: per 0.5 rad; joints 3-7 move it exactly zero), so with it pinned, shoulder
#: abduction was structurally impossible. The operator reported exactly this: "i cant
#: move my shoulders at all". Their recorded session showed `*_arm_joint2` travelling
#: 5.5 deg and 2.5 deg against `joint1`'s 62.6 deg and 76.3 deg.
#:
#: It was not a gain, a scale or an elbow-weight problem: sweeping `elbow_weight` from
#: 0.02 to 2.0 left `joint2` at 0.0 deg of travel, because DLS IK takes small steps
#: and could never cross the ~170 deg to the usable side of its range.
#:
#: Solved by seeding every arm joint at the centre of its soft-limited range, biasing
#: joint2 by -0.2 rad and joint4 by -0.3 rad (mirrored per side), and solving the
#: zero-intent target. Measured result:
#:
#:     SIM       floor  5.0 mm  ->  10.55 mm  PASS
#:     HARDWARE  floor 10.0 mm  ->  10.55 mm  PASS
#:     shoulder abduction headroom  L 74.7 deg  R 74.7 deg   (HOME: 1.6 / 0.8)
#:     minimum headroom, all 14 arm joints  5.1 deg          (HOME: 0.8)
#:
#: The bias matters: the unbiased mid-range solution reaches only 8.45 mm and fails
#: the 10 mm HARDWARE floor, which is how the hardware-recovery test caught it. The
#: remaining tight joint is `left_arm_joint6` at 5.1 deg, a wrist DOF that HOME
#: already left at 5.3 deg, so this regresses nothing. Regenerate with the same
#: seed-bias-and-solve procedure, checking BOTH profile floors, if the policy's
#: neutral target changes.
#: Arms at rest: hanging down beside the torso with a slight elbow bend, which is
#: the robot's own datasheet rest pose and the pose an operator naturally starts
#: from.
#:
#: ``joint2`` sweeps the upper arm through a cone: 0 rad points it STRAIGHT OUT
#: sideways, and +-pi/2 tucks it against the body. Measured on the pinned model,
#: elbow offset from its own shoulder across that sweep:
#:
#:     joint2     0 deg  ->  lateral 0.3500 m, vertical  0.0000 m
#:     joint2   +11 deg  ->  lateral 0.3435 m, vertical +0.0627 m   (previous value)
#:     joint2   -80 deg  ->  lateral 0.0605 m, vertical -0.1728 m   (this value)
#:     joint2   -90 deg  ->  lateral 0.0002 m, vertical -0.1754 m
#:
#: The previous neutral sat 11 degrees from fully splayed, which is why the robot's
#: elbow stood a median 0.341 m out from its shoulder for an entire session while
#: the operator's hung tucked: the seed pose, the posture target and the elbow
#: reference all started there and the redundant DOF was never driven back.
#:
#: -80 rather than -90 deliberately. At -90 the pose is fine standing still, but
#: bending the elbow 30 degrees drives self-clearance to -18.55 mm, i.e. links
#: interpenetrating. At -80 the same bend holds 11.13 mm, against a 5 mm SIM floor
#: and the 6.25 mm the previous neutral managed -- so this pose has nearly double
#: the margin it replaces, measured with the project's own ClearanceChecker.
TELEOP_NEUTRAL_ARM_QPOS = {
    "left_arm_joint1": 0.669088,
    "left_arm_joint2": 0.194326,
    "left_arm_joint3": 0.785935,
    "left_arm_joint4": -1.993992,
    "left_arm_joint5": -0.306686,
    "left_arm_joint6": -0.623350,
    "left_arm_joint7": 0.026896,
    "right_arm_joint1": -0.669076,
    "right_arm_joint2": -0.193605,
    "right_arm_joint3": -0.785231,
    "right_arm_joint4": 1.994893,
    "right_arm_joint5": 0.306420,
    "right_arm_joint6": 0.621695,
    "right_arm_joint7": -0.027369,
}

#: The robot's rest posture: arms hanging beside the torso with a slight elbow
#: bend. This is the pose to LOOK at and the pose the arm-shape work is trying to
#: reach -- it is NOT the solver's seed, and the difference matters.
#:
#: Substituting it for TELEOP_NEUTRAL_ARM_QPOS was tried and reverted. Wrist
#: targets are shoulder-anchored through forward_offset_m and do not move with
#: the neutral, but the solver's seed does, so from arms-down the DLS could not
#: reach ordinary targets: two pipeline tests and the synthetic demo failed with
#: IK_DID_NOT_CONVERGE. That is the same wall the earlier joint2 re-anchor hit.
#: Adopting it as the seed needs the wrist mapping re-anchored with it, not just
#: the constant swapped.
#:
#: Measured on the pinned model, against the teleop neutral above:
#:     elbow lateral   +0.3435 m -> +0.0605 m
#:     elbow vertical  +0.0627 m -> -0.1728 m   (below the shoulder, not above)
#:     self-clearance    6.25 mm ->  11.13 mm   (SIM floor is 5 mm)
#: The robot's rest posture. Identical to the teleop neutral: the pose the
#: operator starts from IS the pose the robot rests in, so the viewer and the
#: solver cannot disagree about where 'default' is.
#: The robot's actual rest posture: arms hanging by its sides. This is what the
#: machine looks like parked, it is what the viewer opens on, and it is what any
#: picture of the robot shows.
#:
#: It is NOT the teleop neutral, and the two must not be collapsed. Measured on
#: the pinned model, with the wrist mapping anchored on each pose:
#:
#:     pose             neutral TCP height   vertical signals that converge
#:     mid-range         0.898 m             -1.2 .. +1.2   (full range)
#:     arms-by-sides     0.294 m             -1.2 ..  0.0   (no downward travel)
#:
#: Arms by the sides puts the hands 29 cm off the floor, so an operator lowering
#: their hands has nowhere to go and the solver rejects the frame. Substituting
#: this for the neutral was tried twice and reverted twice; the second attempt
#: also re-anchored the wrist mapping on the neutral TCP, which fixed the
#: geometry and still lost half the vertical range, because the range is a
#: property of where the hands start, not of the anchor.
#:
#: Against the teleop neutral this pose is nonetheless better on every static
#: measure: elbow lateral offset 0.3434 -> 0.0905 m, self-clearance 10.55 ->
#: 12.41 mm (both the 5 mm SIM and 10 mm HARDWARE floors pass), worst arm-joint
#: headroom 5.1 -> 10.8 deg. It is a good pose to rest in and a poor pose to
#: work from.
#: The robot's rest posture: arms hanging by its sides, elbows tucked, forearms
#: angled slightly forward. Posed directly in the viewer by the operator rather
#: than derived -- two automated searches for it optimised a proxy and both
#: produced the wrong shape, one scoring the elbow's lateral offset while
#: leaving the hands 0.89 m out in front.
#:
#: Mirrored: the posed left and right differed by up to 15 deg on a joint, so
#: each value is the mean of the two magnitudes and the right arm is the exact
#: negation of the left. The symmetric pose measures better than either arm as
#: posed -- elbow lateral +0.0092 m against +0.0300 and +0.0117 -- and the two
#: TCPs mirror to 0.0005 m.
#:
#: Measured on the pinned model:
#:     elbow lateral offset   L +0.0092 m   R -0.0091 m   (working neutral: 0.3434)
#:     TCP from shoulder      [+0.287, +0.010, -0.507] m, reach 0.583
#:     self-clearance         12.10 mm      SIM floor 5.0, HARDWARE floor 10.0, both PASS
#:     rest -> neutral sweep  6.25 mm minimum, never below the SIM floor
#:
#: This is what the twin opens on. It is NOT the solver's seed -- see
#: TELEOP_NEUTRAL_ARM_QPOS, where the hands start high enough to move in every
#: direction. Substituting this for the seed was tried three times and reverted
#: three times; the last attempt failed on swept clearance, not on IK.
REST_ARM_QPOS = {
    "left_arm_joint1": 1.806416,
    "left_arm_joint2": -1.544616,
    "left_arm_joint3": -0.575959,
    "left_arm_joint4": -1.624902,
    "left_arm_joint5": 0.100356,
    "left_arm_joint6": -0.632682,
    "left_arm_joint7": 0.030543,
    "right_arm_joint1": -1.806416,
    "right_arm_joint2": 1.544616,
    "right_arm_joint3": 0.575959,
    "right_arm_joint4": 1.624902,
    "right_arm_joint5": -0.100356,
    "right_arm_joint6": 0.632682,
    "right_arm_joint7": -0.030543,
}

#: Each gripper's five follower joints, and the coupling that ties them to the drive
#: joint: the model's equality reads ``qpos(drive) = c1 * qpos(follower)``, so
#: ``follower = drive / c1``.
#:
#: This table exists because the drive joint alone is not enough to *show* a grasp.
#: The gripper is a four-bar linkage whose followers are `<mimic>` in the URDF and
#: `<equality><joint>` in the MJCF (`neq=10`). MuJoCo enforces equality constraints in
#: the constraint **solver**, during `mj_step`. The preview sink writes qpos and calls
#: `mj_forward`, which computes constraint *forces* but never projects qpos onto the
#: constraint manifold -- so the followers simply stay where they were.
#:
#: Measured on the verified model, sweeping the drive joint over its full commanded
#: range (0.10 -> 1.60 rad):
#:
#:     drive joint alone   every gripper link moved   0.000 mm; jaw gap 137.3 -> 137.3 mm
#:     drive + followers   each finger moved         61.347 mm; jaw gap 134.0 ->  20.8 mm
#:
#: That is the whole defect the operator reported as "when i close my arms to grab
#: something it doesnt update the robot to do that as well": the commanded gripper
#: angle was correct and reached the sink, but nothing visible ever moved.
#:
#: Both fingers hang off follower branches, not off the drive joint's subtree, which is
#: why driving the drive joint moves literally nothing you can see.
#:
#: The followers are applied at the preview sink (`GripperLinkagePreviewSink`), NOT added
#: to `RobotTarget.joints`. They are structural consequences of the drive joint, not
#: independently commandable DOF -- MODEL_FACTS lists them as "passive / structural, not
#: independently commandable" -- so putting them in the command would add ten exactly
#: redundant columns to `JOINT_ORDER` and to every exported LeRobot dataset. The command
#: contract stays the 23 real DOF; the linkage is resolved where the geometry is drawn.
#:
#: `_verify_gripper_mimic_coupling` re-derives this table from the loaded model and
#: refuses to run if it disagrees, so it cannot silently drift from the verified MJCF.
GRIPPER_MIMIC_COUPLING: dict[str, dict[str, float]] = {
    "left_gripper_joint": {
        "left_gripper_r_inner_knuckle_joint": -1.0,
        "left_gripper_r_finger_joint": 1.0,
        "left_gripper_l_knuckle_joint": 1.0,
        "left_gripper_l_inner_knuckle_joint": 1.0,
        "left_gripper_l_finger_joint": -1.0,
    },
    "right_gripper_joint": {
        "right_gripper_r_inner_knuckle_joint": -1.0,
        "right_gripper_r_finger_joint": 1.0,
        "right_gripper_l_knuckle_joint": 1.0,
        "right_gripper_l_inner_knuckle_joint": 1.0,
        "right_gripper_l_finger_joint": -1.0,
    },
}
def expand_gripper_command(drive_positions: Mapping[str, float]) -> dict[str, float]:
    """Add the mimic followers implied by each commanded gripper drive joint.

    Returns the drives unchanged plus one entry per follower. Drives absent from the
    input are skipped, so this is safe to call on a partial pose.
    """
    expanded = dict(drive_positions)
    for drive, followers in GRIPPER_MIMIC_COUPLING.items():
        if drive not in drive_positions:
            continue
        position = drive_positions[drive]
        for follower, coupling in followers.items():
            expanded[follower] = position / coupling
    return expanded


def _verify_gripper_mimic_coupling(model) -> None:
    """Fail fast if the model's equality constraints disagree with the table above."""
    derived: dict[str, dict[str, float]] = {}
    for index in range(model.neq):
        if model.eq_type[index] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        drive = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.eq_obj1id[index])
        follower = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.eq_obj2id[index])
        if drive not in GRIPPER_MIMIC_COUPLING:
            continue
        polynomial = model.eq_data[index]
        # A non-linear or offset coupling would make `follower = drive / c1` wrong.
        if any(abs(float(term)) > 1e-9 for term in (polynomial[0], *polynomial[2:5])):
            raise RuntimeError(
                f"gripper mimic {drive}->{follower} is not a pure linear coupling"
            )
        derived.setdefault(drive, {})[follower] = float(polynomial[1])
    if derived != GRIPPER_MIMIC_COUPLING:
        raise RuntimeError(
            "model gripper mimic coupling does not match GRIPPER_MIMIC_COUPLING: "
            f"model={derived} table={GRIPPER_MIMIC_COUPLING}"
        )


class GripperLinkagePreviewSink(MujocoPreviewSink):
    """Preview sink that closes the gripper's four-bar linkage as it writes qpos.

    `MujocoPreviewSink` writes the commanded joints and calls `mj_forward`, which never
    projects qpos onto the model's equality constraints. Every follower in
    `GRIPPER_MIMIC_COUPLING` is therefore resolved here, so the rendered fingers track
    the commanded drive joint instead of standing still. See `GRIPPER_MIMIC_COUPLING`
    for the measurements that pin this down.
    """

    def __init__(
        self,
        *,
        model: "mujoco.MjModel | None" = None,
        initial_pose: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(
            model=model, initial_pose=expand_gripper_command(initial_pose or {})
        )

    def submit(self, command):  # type: ignore[override]
        receipt = super().submit(command)
        drives = {
            joint.name: joint.position_rad
            for joint in command.target.joints
            if joint.name in GRIPPER_MIMIC_COUPLING
        }
        if not drives:
            return receipt
        with self.lock:
            for name, position in expand_gripper_command(drives).items():
                if name in drives:
                    continue
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id < 0:
                    raise ValueError(f"unknown preview joint: {name}")
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = position
            mujoco.mj_forward(self.model, self.data)
        return receipt


#: What the twin shows before anything drives it: the robot at rest.
SIM_REST_QPOS = {
    **HOME_QPOS,
    **REST_ARM_QPOS,
    "left_gripper_joint": GRIPPER_OPEN_RAD,
    "right_gripper_joint": GRIPPER_OPEN_RAD,
}
SIM_TELEOP_HOME_QPOS = {
    **HOME_QPOS,
    **TELEOP_NEUTRAL_ARM_QPOS,
    "left_gripper_joint": GRIPPER_OPEN_RAD,
    "right_gripper_joint": GRIPPER_OPEN_RAD,
}


def _clamp_into_soft_band(
    pose: Mapping[str, float],
    limits: Mapping[str, object],
    margin: float,
) -> dict[str, float]:
    """Pull each joint of ``pose`` inside its soft-limit band ``[lower+margin, upper-margin]``.

    The REST posture the twin rests in has both shoulder-abduction joints ~1.5 deg
    outside the supervisor's soft band, so it is a valid pose to DISPLAY but not one
    the supervisor will ever ALLOW as a command. Nudging those joints just inside the
    band yields a commandable rest -- visually the same arms-by-the-sides posture --
    that the governor and supervisor can start from, so the arm animates out of rest
    instead of snapping to the mid-range neutral on the first command. Joints without
    a parsed limit (e.g. the gripper drives) pass through unchanged.
    """
    clamped = dict(pose)
    for name, value in pose.items():
        limit = limits.get(name)  # type: ignore[attr-defined]
        if limit is None:
            continue
        low = limit.lower_rad + margin
        high = limit.upper_rad - margin
        if low <= high:
            clamped[name] = min(high, max(low, float(value)))
    return clamped


#: The pose the twin boots and rests in, and the pose the supervisor and governor
#: start tracking from: the arms-by-the-sides Motion-Studio REST posture, nudged
#: the ~1.5 deg needed to bring both shoulder-abduction joints inside the soft-limit
#: band so it is actually commandable (raw REST is display-valid but the supervisor
#: would HOLD it). Distinct from SIM_TELEOP_HOME_QPOS, which remains the IK seed and
#: posture-regularisation anchor. Measured self-clear at 12.19 mm.
SIM_TELEOP_START_QPOS = _clamp_into_soft_band(
    SIM_REST_QPOS,
    parse_revolute_joint_limits(CANONICAL_MANIFEST.fixed_urdf),
    SupervisorPolicy().soft_margin_rad,
)


@dataclass(frozen=True)
class PipelineResult:
    observation_sequence: int
    target: RobotTarget | None
    decision: SafetyDecision
    receipt: CommandReceipt | None
    observed_joints: Mapping[str, float] | None
    #: Control groups whose input was not trustworthy on this frame and which
    #: were therefore commanded to hold their last governed pose. Empty on a
    #: fully tracked frame; every group on a whole-frame HOLD or FAULT.
    held_groups: frozenset[str] = frozenset()
    #: Per-group evidence is needed because a whole-frame ALLOW can still hold
    #: a single arm.
    held_group_reasons: tuple[tuple[str, str], ...] = ()
    #: Grippers are independently sourced from hand landmarks, not arm IK. Keep
    #: their loss separate from a ControlGroup so an intact arm is not falsely
    #: called held when only its hand/open-close input disappeared.
    held_grippers: frozenset[str] = frozenset()
    held_gripper_reasons: tuple[tuple[str, str], ...] = ()
    #: Groups whose target remains legal but is pinned in a soft-limit band.
    #: This is deliberately distinct from ``held_groups``: the group can still
    #: follow on an unconstrained axis, but the operator must see that the
    #: saturated axis is no longer following their pose.
    saturated_groups: frozenset[str] = frozenset()
    saturated_group_reasons: tuple[tuple[str, str], ...] = ()
    #: Best-effort finite wrist residuals for locally rejected arm maps.  This
    #: diagnostic is not a target and cannot authorize a command; it exists so
    #: replay analysis can measure the rejected workspace tail.
    held_group_residuals_m: tuple[tuple[str, float], ...] = ()
    #: Ordered DLS-rung residuals for locally rejected arm maps.  These are
    #: diagnostic only and deliberately live outside RobotTarget: no failed rung
    #: can look like a command or affect safety authorization.
    held_group_ik_attempts: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()


class MotionStudioPipeline:
    """Mirror head and both arms into a safety-approved digital twin.

    Construction and all methods are offline-only.  There is no Galbot SDK import,
    network transport, or physical command adapter anywhere in this path.
    """

    required_landmarks = frozenset(
        {
            "face_1", "face_33", "face_263", "pose_11", "pose_12",
            "pose_13", "pose_14", "pose_15", "pose_16",
        }
    )

    def __init__(
        self,
        *,
        source_clock_id: str,
        sink: CommandSink | None = None,
        motion_profile: MotionProfile = MotionProfile.SIM,
        # swivel-priority is the default: on the retained capture it holds the
        # same wrist target as wrist-primary while cutting upper-arm elevation
        # error by +31 deg (right) and +5.5 deg (left) via the redundant DOF.
        # It falls back to the wrist-primary solve per-frame when the strict
        # solve cannot hold the wrist, so it never does worse than wrist-primary.
        arm_mapping: str = "swivel-priority",
        liveness_monitor: FrameLivenessMonitor | None = None,
        selection_manager: OperatorSelectionManager | None = None,
    ) -> None:
        if arm_mapping not in {"wrist-primary", "direction-vector", "swivel-priority"}:
            raise ValueError(
                "arm_mapping must be wrist-primary, direction-vector or swivel-priority"
            )
        self.arm_mapping = arm_mapping
        self.mapping_hash = mapping_hash_for(arm_mapping)
        self.model = load_verified_fixed_base_model()
        # The gripper command is only correct if the followers really are coupled the
        # way GRIPPER_MIMIC_COUPLING claims. Check it against the model, once, here.
        _verify_gripper_mimic_coupling(self.model)
        # Joint limits drive the retargeters (below); parse them before those.
        limits = parse_revolute_joint_limits(CANONICAL_MANIFEST.fixed_urdf)
        # The twin, the supervisor's state estimate and the governor all start at
        # the commandable Motion-Studio rest (SIM_TELEOP_START_QPOS): the identical
        # arms-by-the-sides posture, but nudged just inside the soft-limit band so
        # the supervisor will actually ALLOW it. The arm then animates out of rest
        # into the operator's pose instead of snapping to the mid-range neutral on
        # the first command. The IK seed/posture anchor stays SIM_TELEOP_HOME.
        self._start_pose = SIM_TELEOP_START_QPOS
        self.sink = sink or GripperLinkagePreviewSink(
            model=self.model, initial_pose=self._start_pose
        )
        # Resolve the profile before the checker: the self-clearance floor is
        # profile-dependent, so the checker cannot be built until it is known.
        self.motion_profile = MotionProfile(motion_profile)
        self._recovery_required_frames = (
            1 if self.motion_profile is MotionProfile.SIM else 3
        )
        self._command_min_confidence = (
            SIM_PREVIEW_MIN_CONFIDENCE
            if self.motion_profile is MotionProfile.SIM
            else 0.5
        )
        self.checker = ClearanceChecker(
            model=self.model,
            home=HOME_QPOS,
            **clearance_kwargs_for(self.motion_profile),
        )
        supervisor_policy = policy_for(self.motion_profile)
        self.supervisor = SafetySupervisor(
            self.checker,
            # The supervisor tracks where the robot ACTUALLY is, which the twin
            # displays from boot: the commandable rest pose. Starting it at the
            # mid-range neutral instead would make the first command (a small step
            # from rest) read as a large step down and trip a rate HOLD. The IK
            # seed/posture anchor stays SIM_TELEOP_HOME (on the arm retargeters
            # below); this is the robot's state estimate, not the solver seed.
            initial_pose=self._start_pose,
            source_clock_id=source_clock_id,
            policy=supervisor_policy,
        )
        self._head_policy = HeadMappingPolicy(
            yaw_limit=limits["head_joint1"],
            pitch_limit=limits["head_joint2"],
        )
        self.head = HeadRetargeter(
            self._head_policy,
            neutral=HeadTarget(
                yaw_rad=HOME_QPOS["head_joint1"],
                pitch_rad=HOME_QPOS["head_joint2"],
            ),
        )
        # The torso carries both arm mounts and the head, so it is mapped before
        # them and the arms are told where it ended up. Neutral is HOME, matching
        # the head: the operator's square stance is the robot's square stance.
        self._torso_policy = TorsoMappingPolicy(yaw_limit=limits[TORSO_YAW_JOINT])
        self.torso = TorsoRetargeter(
            self._torso_policy,
            neutral=TorsoTarget(yaw_rad=HOME_QPOS[TORSO_YAW_JOINT]),
        )
        # A camera/world yaw is an absolute measurement.  Its zero is the
        # operator's square, still calibration pose -- never the arbitrary world
        # orientation of the camera.  ``None`` means the rest of teleop remains
        # calibrated but the torso must hold until a valid recalibration.
        self._torso_yaw_reference_rad: float | None = None
        # An edge/back-facing detector can swap the two shoulder labels. Keep
        # the last *continuous raw* shoulder observation separately from the
        # last command: face evidence may disappear while the shoulders remain
        # geometrically continuous. That evidence must never move the robot
        # during the hold, but bounded source-time/yaw continuity is sufficient
        # to prove there was no identity-sized blind interval. In particular,
        # a valid operator may cross neutral while the face cue is absent; a
        # same-side latch would reject that legitimate left/right tummy motion.
        self._last_observed_torso_yaw_rad: float | None = None
        self._last_observed_torso_capture_ns: int | None = None
        # Detector-cadence continuity evidence, fed by the capture loop through
        # observe_source_continuity. Its own liveness monitor is deliberate: a
        # frozen buffer yields perfect shoulders forever, and a witness that
        # trusted them would vouch for exactly the interval an operator could
        # turn through unobserved. Same policy, separate state -- the gate's
        # monitor is consumed by the worker and must not be double-fed.
        self._source_continuity: _SourceContinuity | None = None
        self._witness_liveness = (
            FrameLivenessMonitor(liveness_monitor.policy)
            if liveness_monitor is not None
            else None
        )
        # A source-time gap or semantic shoulder-label jump makes signed torso
        # direction ambiguous to a monocular camera. Unlike a local landmark
        # hold, no sequence of plausible frames can prove that labels did not
        # swap while it was unobserved; only calibration clears it.
        self._torso_recalibration_required = False
        #: Consecutive provable frames since the latch. See _recover_torso_yaw.
        self._torso_recovery_frames = 0
        self._torso_recovery_yaw_rad: float | None = None
        self._torso_recovery_capture_ns: int | None = None
        # SIM_TELEOP_HOME_QPOS, not HOME_QPOS: the neutral pose is the IK seed and the
        # posture-regularisation target, and HOME pins both shoulder-abduction joints
        # against their limits. See TELEOP_NEUTRAL_ARM_QPOS. The wrist target itself is
        # shoulder-anchored and unaffected by this; only the seed, the posture target
        # and the elbow reference move.
        arm_policy = LeftArmPolicy(mapping_mode=arm_mapping)
        self.left_arm = LeftArmRetargeter(
            self.model, neutral_pose=SIM_TELEOP_HOME_QPOS, policy=arm_policy
        )
        self.right_arm = RightArmRetargeter(
            self.model, neutral_pose=SIM_TELEOP_HOME_QPOS, policy=arm_policy
        )
        # Confidence and landmark presence are judged per control group, so an
        # occluded left wrist holds the left arm instead of the whole robot.
        # Every other rejection the gate can raise stays whole-frame.
        self.gate = ObservationGate(
            FreshnessPolicy(
                source_clock_id=source_clock_id,
                required_landmark_names=self.required_landmarks,
                min_confidence=self._command_min_confidence,
            ),
            control_groups=COMMAND_CONTROL_GROUPS,
            liveness_monitor=liveness_monitor,
            selection_manager=selection_manager,
        )
        arm_acceleration_limit = supervisor_policy.max_joint_acceleration_rad_s2
        arm_step_limit = supervisor_policy.max_joint_step_rad
        self.trajectory = JointTrajectoryGovernor(
            TrajectoryLimits(
                max_velocity_rad_s=_dynamically_feasible_velocity_limit(
                    supervisor_policy.max_joint_rate_rad_s,
                    max_acceleration_rad_s2=arm_acceleration_limit,
                    max_position_step_rad=arm_step_limit,
                ),
                max_acceleration_rad_s2=arm_acceleration_limit,
                max_position_step_rad=arm_step_limit,
            ),
            # The torso policy is stricter than the generic simulation envelope.
            # Put the cap in the actual integrator so every emitted leg_joint4
            # target, including one after a source gap, obeys it.
            max_velocity_rad_s_by_joint={
                "head_joint1": self._head_policy.max_rate_rad_s,
                "head_joint2": self._head_policy.max_rate_rad_s,
                TORSO_YAW_JOINT: self._torso_policy.max_rate_rad_s,
            },
        )
        # Guard-rail-aware conditioning sits between the raw retarget target and
        # the governor. It clamps into the soft-limit band and, when the straight
        # joint-space line toward the target would drive through a self-collision,
        # advances only as far along it as is clear this frame -- so the arm keeps
        # flowing toward the operator instead of ramming a wall and latching a
        # supervisor HOLD. The clearance predicate is byte-identical to the
        # supervisor's own swept check (steps=2, max_joint_step=0.01, verdict =
        # report.ok), so a path this controller accepts is one the supervisor also
        # accepts. It never relaxes a guard rail; the supervisor remains the final
        # independent authority on every emitted pose.
        self.guarded = GuardedPathController(
            joint_limits={
                name: (limit.lower_rad, limit.upper_rad)
                for name, limit in limits.items()
            },
            soft_margin_rad=supervisor_policy.soft_margin_rad,
            clearance_check=self._clearance_margin,
            # One governor step. The governor advances the pose by at most this
            # much per frame regardless, so looking further is wasted swept-check
            # cost: a far goal refines into hundreds of samples. Bounding to one
            # step keeps the common collision-APPROACH frames on the cheap
            # single-check fast path (the wall is still a step away and clear), so
            # only the frame that actually sits against the wall pays the
            # bisection.
            #
            # NOTE (2026-08-26): this horizon is ALSO the de-facto speed limit, and
            # the older claim that it "never limits real motion" is measurably
            # false. The governor plans
            # `desired_velocity = min(max_velocity, stopping_speed(error))`
            # (trajectory.py:160-179) against the distance to whatever target it is
            # handed -- which is this horizon. Live run: peak arm speed 0.345 rad/s
            # against an authorised 0.894, and 0 of 2,184 updates ever reached the
            # 0.10 step cap. Sizing the horizon past the braking curve raises peak
            # speed to exactly max_velocity_rad_s and the step to exactly
            # max_position_step_rad (measured: travel 24.17 -> 27.35 rad on the same
            # footage) -- but it also lets the swept check reach far enough that a
            # HELD limb perturbs a TRACKED one by ~4e-4 rad, breaking the decoupling
            # guarantee documented in guarded_controller.py. See _lookahead_for.
            max_lookahead_rad=_lookahead_for(self.trajectory.limits),
            # A requested tummy turn must remain an explicit, independent
            # operator command while the arm detour searches a collision-free
            # shape. If that tummy motion itself blocks with the arms held, the
            # controller still rejects it; this is priority, never an exemption.
            protected_joints=(TORSO_YAW_JOINT,),
            # Deliberately 0: the guarded controller shapes the straight-line
            # TARGET, but the clearance the supervisor sees belongs to the pose the
            # governor REALIZES, which is chosen in `_feasible_governor_command`.
            # Applying the standoff here instead was measured on the full
            # 2026-08-22 replay and moved floor-riding the wrong way, 28.6% ->
            # 34.1%, because it cost reach without touching the pose being
            # measured. The capability stays available and tested.
            preferred_clearance_m=0.0,
        )
        # Gripper speed is profile-split for the same reason the arm envelope is.
        #
        # 0.5 rad/s is the real actuator's URDF limit; hardware is deliberately
        # slower whenever its conservative supervisor envelope is tighter. SIM can
        # use that actuator ceiling because its supervisor allows more, but no
        # profile can exceed either boundary. The governor must stay inside the
        # supervisor or it will produce commands that must be rejected.
        #
        # Do not raise these limits without re-measuring on a real session.
        #
        # These three are mutually constrained, not independent knobs. A step of
        # `max_position_step_rad` taken in one ~60 ms frame implies an acceleration of
        # roughly step/dt^2; at 0.10 rad that is ~27 rad/s^2, an order of magnitude over
        # the supervisor's 2.0 limit, so the supervisor rejects the command and the
        # gripper freezes entirely.
        #
        # Measured, two attempts: 2.5 / 8.0 / 0.20 gave 365 `acceleration exceeds
        # policy` HOLDs in 602 frames; clamping only velocity and step to the SIM
        # policy (1.5 / 2.0 / 0.10) still gave 146 in 165 frames, with the grippers
        # commanded over 1.2 deg and observing 0.0 -- fully stalled.
        #
        # The gripper's own actuator envelope is still narrower than the SIM arm
        # envelope, but it must also fit *inside* every active supervisor dynamic
        # limit. In particular, a 50 mrad command cap with 0.5 rad/s velocity and
        # 1.0 rad/s² deceleration is mathematically incompatible after a sparse
        # 140 ms source interval: stopping from 0.5 rad/s within 50 mrad needs
        # slightly more than the allowed acceleration. Deriving the cap from the
        # profile makes the SIM 100 mrad allowance available while retaining the
        # hardware 50 mrad cap. The shared velocity bound also makes the
        # post-cap endpoint dynamically feasible at every source interval,
        # including custom conservative profiles. Every command remains
        # independently rechecked by the supervisor.
        gripper_acceleration_limit = min(
            1.0, supervisor_policy.max_joint_acceleration_rad_s2
        )
        gripper_step_limit = supervisor_policy.max_joint_step_rad
        self.gripper_trajectory = JointTrajectoryGovernor(
            TrajectoryLimits(
                max_velocity_rad_s=_dynamically_feasible_velocity_limit(
                    min(0.5, supervisor_policy.max_joint_rate_rad_s),
                    max_acceleration_rad_s2=gripper_acceleration_limit,
                    max_position_step_rad=gripper_step_limit,
                ),
                max_acceleration_rad_s2=gripper_acceleration_limit,
                max_position_step_rad=gripper_step_limit,
            )
        )
        self._gripper_desired = {
            "left_gripper_joint": GRIPPER_OPEN_RAD,
            "right_gripper_joint": GRIPPER_OPEN_RAD,
        }
        self._face_calibration: NeutralCalibration | None = None
        self._left_arm_calibration = None
        self._right_arm_calibration = None
        self._calibration_selection_id: int | None = None
        self._previous_head: HeadTarget | None = None
        # Face mesh and pose-head landmarks use different calibrated bases.
        # Switching them must be visible and confirmed; otherwise an adapter
        # fallback can make one frame look like a commanded head snap.
        self._last_accepted_head_basis: str | None = None
        self._head_recovery_basis: str | None = None
        self._previous_left_arm: dict[str, float] | None = None
        self._previous_right_arm: dict[str, float] | None = None
        self._previous_timestamp_ns: int | None = None
        # Last positions the trajectory governor actually emitted. A held group
        # is commanded to exactly these, so it decelerates to a standstill and
        # stays there while the other groups keep tracking.
        self._governed_joints: dict[str, float] = {}
        # Residual of each arm's most recently APPROVED IK solution. A held arm
        # is still commanded to that solution, so its residual still describes
        # the pose being sent. Only approved solutions are stored, and approval
        # requires residual <= policy, so a stale value can never latch a hold.
        self._previous_residual_m = {"left": 0.0, "right": 0.0}
        self._filters = IntentFilterBank()
        self._recovery_streak = 0

    def acquire_operator(
        self,
        occupancy: OccupancyObservation,
        observation: HumanObservation,
        *,
        now_mono_ns: int,
    ) -> SelectionDecision:
        """Perform the explicit acquisition action for a selection-enabled session."""
        return self.gate.acquire_selection(
            occupancy, observation, now_mono_ns=now_mono_ns
        )

    def calibrate(
        self,
        observation: HumanObservation,
        *,
        occupancy: OccupancyObservation | None = None,
        now_mono_ns: int | None = None,
    ) -> str:
        """Legacy single-observation calibration for deterministic simulation.

        Live webcam preview uses :meth:`calibrate_window` through the CLI. Keep
        this narrow API for synthetic fixtures and explicitly offline legacy
        playback; it must not grow a silent production fallback.
        """
        # Build all calibration artifacts before consuming a selection/gate
        # observation. An invalid neutral pose must not advance sequence,
        # liveness, or continuity state and thereby make a corrected retry look
        # reordered. Nothing is committed until both artifact and selection
        # validation have succeeded.
        face_calibration = create_neutral_calibration(observation)
        left_arm_calibration = create_left_arm_calibration(observation)
        right_arm_calibration = create_right_arm_calibration(observation)
        selection_id = self._validate_calibration_selection(
            (observation,), None if occupancy is None else (occupancy,), now_mono_ns
        )
        return self._commit_calibration(
            observation,
            face_calibration=face_calibration,
            left_arm_calibration=left_arm_calibration,
            right_arm_calibration=right_arm_calibration,
            torso_yaw_reference_rad=_calibrated_torso_yaw_reference(
                (observation,), self._torso_policy
            ),
            selection_id=selection_id,
        )

    def calibrate_window(
        self,
        observations: Sequence[HumanObservation],
        policy: CalibrationWindowPolicy,
        *,
        occupancies: Sequence[OccupancyObservation] | None = None,
        now_mono_ns: int | None = None,
    ) -> str:
        """Commit calibration only after every sample in a measured quiet window.

        The neutral face/shoulder geometry uses the validated mean. Both arm
        calibrators run on every sample so an elbow/wrist dropout cannot hide
        inside an otherwise face-valid window; their final calibration inherits
        the window-mean shoulder scale rather than a last-frame fluctuation.
        """
        samples = tuple(observations)
        # As above, reject an invalid/unstable neutral window before it can
        # mutate the stateful freshness or selection gates.
        face_calibration = create_neutral_calibration_window(samples, policy)
        left_calibrations = tuple(create_left_arm_calibration(sample) for sample in samples)
        right_calibrations = tuple(create_right_arm_calibration(sample) for sample in samples)
        selection_id = self._validate_calibration_selection(
            samples, None if occupancies is None else tuple(occupancies), now_mono_ns
        )
        latest = samples[-1]
        return self._commit_calibration(
            latest,
            face_calibration=face_calibration,
            left_arm_calibration=replace(
                left_calibrations[-1],
                shoulder_width_normalized=face_calibration.shoulder_width_normalized,
            ),
            right_arm_calibration=replace(
                right_calibrations[-1],
                shoulder_width_normalized=face_calibration.shoulder_width_normalized,
            ),
            torso_yaw_reference_rad=_calibrated_torso_yaw_reference(
                samples, self._torso_policy
            ),
            selection_id=selection_id,
        )

    def _validate_calibration_selection(
        self,
        observations: tuple[HumanObservation, ...],
        occupancies: tuple[OccupancyObservation, ...] | None,
        now_mono_ns: int | None,
    ) -> int | None:
        """Require neutral calibration to cite the currently locked operator.

        Legacy simulator/offline callers run without a selection manager. Once
        selection is enabled, though, an unassociated neutral pose is not a
        calibration at all: it could belong to a person who has not been
        explicitly acquired. Every sample therefore needs an independent
        occupancy result and must preserve the same selection generation.
        """
        if not self.gate.has_selection_manager:
            if occupancies is not None or now_mono_ns is not None:
                raise CalibrationError(
                    "occupancy evidence is only valid when operator selection is configured"
                )
            return None
        if not observations or occupancies is None or len(occupancies) != len(observations):
            raise CalibrationError(
                "selection-enabled calibration requires matched occupancy evidence"
            )
        if now_mono_ns is None:
            raise CalibrationError(
                "selection-enabled calibration requires the current monotonic time"
            )
        expected_selection_id = self.gate.selection_id
        if expected_selection_id is None:
            raise CalibrationError("calibration requires an explicitly acquired operator")
        for occupancy, observation in zip(occupancies, observations, strict=True):
            decision = self.gate.selection_manager.validate_calibration_evidence(
                occupancy, observation, now_mono_ns=now_mono_ns
            )
            if not decision.accepted or decision.selection_id != expected_selection_id:
                raise CalibrationError(
                    "calibration selection is not continuously locked: "
                    f"{decision.reason or decision.state}"
                )
        return expected_selection_id

    def _commit_calibration(
        self,
        observation: HumanObservation,
        *,
        face_calibration: NeutralCalibration,
        left_arm_calibration: LeftArmCalibration,
        right_arm_calibration: LeftArmCalibration,
        torso_yaw_reference_rad: float | None,
        selection_id: int | None,
    ) -> str:
        # Calibration binds to an existing explicit selection generation. It does
        # not restart the camera or selection source, so existing liveness
        # evidence remains valid. Resetting it here would turn the next frame
        # into a startup hold and could destroy the selection just authorized.
        self._face_calibration = face_calibration
        self._left_arm_calibration = left_arm_calibration
        self._right_arm_calibration = right_arm_calibration
        self._torso_yaw_reference_rad = torso_yaw_reference_rad
        self._last_observed_torso_yaw_rad = torso_yaw_reference_rad
        self._last_observed_torso_capture_ns = observation.capture_mono_ns
        # Seed the witness from the same accepted sample, with no break pending:
        # calibration is the epoch boundary, so evidence from before it must not
        # carry a break across into the new epoch.
        self._source_continuity = (
            None
            if torso_yaw_reference_rad is None
            else _SourceContinuity(
                capture_ns=observation.capture_mono_ns,
                yaw_rad=torso_yaw_reference_rad,
                break_capture_ns=None,
                break_reason=None,
            )
        )
        self._torso_recalibration_required = False
        self._calibration_selection_id = selection_id
        self._previous_head = None
        self._last_accepted_head_basis = None
        self._head_recovery_basis = None
        self._previous_left_arm = None
        self._previous_right_arm = None
        self._previous_timestamp_ns = None
        self._previous_residual_m = {"left": 0.0, "right": 0.0}
        self._filters = IntentFilterBank()
        self._recovery_streak = 0
        # Start the sweep from where the twin ACTUALLY is -- the pose the
        # supervisor is tracking, which the sink is displaying -- not from the
        # mid-range neutral. At boot that is the Motion-Studio REST posture (arms
        # by the sides), so the arm animates from rest into the operator's pose
        # through the governor and the guarded controller, instead of snapping to
        # neutral on the very first command. On a re-calibration it starts from
        # wherever the twin currently is, which is equally seamless. The IK
        # seed/posture anchor is unchanged (SIM_TELEOP_HOME, on the retargeters);
        # this only sets where the DISPLAYED motion begins.
        controlled_home = {
            name: self.supervisor.current_pose[name]
            for name in (
                "head_joint1",
                "head_joint2",
                # Torso yaw is governed like any other tracked joint, which is
                # what makes the swept clearance check see the torso move. Left
                # ungoverned it would be filled from HOME and the check would be
                # blind to exactly the motion the torso mapper creates.
                TORSO_YAW_JOINT,
                *(
                    name
                    for name in SIM_TELEOP_HOME_QPOS
                    if name.startswith("left_arm_joint")
                    or name.startswith("right_arm_joint")
                ),
            )
        }
        self.trajectory.reset(
            controlled_home,
            timestamp_ns=observation.capture_mono_ns,
        )
        self._governed_joints = dict(controlled_home)
        self._previous_left_arm = {
            name: controlled_home[name] for name in self.left_arm.joint_names
        }
        self._previous_right_arm = {
            name: controlled_home[name] for name in self.right_arm.joint_names
        }
        self._gripper_desired = {
            "left_gripper_joint": GRIPPER_OPEN_RAD,
            "right_gripper_joint": GRIPPER_OPEN_RAD,
        }
        self.gripper_trajectory.reset(
            self._gripper_desired,
            timestamp_ns=observation.capture_mono_ns,
        )
        return self.supervisor.start_preview(
            source_mono_ns=observation.capture_mono_ns
        )

    def _clearance_margin(
        self, start: Mapping[str, float], end: Mapping[str, float]
    ) -> float | bool:
        """Swept clearance for one segment: metres if clear, ``False`` if not.

        Byte-identical in its VERDICT to the supervisor's own check (steps=2,
        max_joint_step=0.01, verdict = report.ok), so a path this accepts is one
        the supervisor also accepts. It additionally returns the distance, which
        costs nothing -- the swept check already computed it -- and is what lets
        the guarded controller aim above the floor instead of at it.

        ``min_distance`` deliberately excludes baseline-overlapping pairs, which
        are structurally overlapped at every pose; including them would make the
        standoff permanently unreachable and silently disable it.
        """
        report = self.checker.check_path(start, end, steps=2, max_joint_step=0.01)
        if not report.ok:
            return False
        distance = report.min_distance
        if distance is None or not isfinite(distance):
            # Clear, but with no finite distance to report (no non-baseline pair
            # in range). Genuinely unconstrained, so any standoff is satisfied.
            return True
        return float(distance)

    def reset_perception_source(self) -> None:
        """Invalidate perception state at an explicit camera/source boundary.

        This deliberately does not clear a supervisor FAULT. A caller must reset
        that authority explicitly and recalibrate before it can start preview
        again; silently treating a source restart as healthy would re-authorize
        stale calibration against a new operator or camera.
        """
        self.gate.reset_source()
        self._face_calibration = None
        self._left_arm_calibration = None
        self._right_arm_calibration = None
        self._torso_yaw_reference_rad = None
        self._last_observed_torso_yaw_rad = None
        self._last_observed_torso_capture_ns = None
        self._source_continuity = None
        if self._witness_liveness is not None:
            self._witness_liveness.reset()
        self._torso_recalibration_required = False
        self._calibration_selection_id = None
        self._previous_head = None
        self._last_accepted_head_basis = None
        self._head_recovery_basis = None
        self._previous_left_arm = None
        self._previous_right_arm = None
        self._previous_timestamp_ns = None
        self._filters = IntentFilterBank()
        self._recovery_streak = 0

    def fault_perception(
        self,
        *,
        session_id: str,
        sequence: int,
        source_mono_ns: int,
        now_mono_ns: int,
        reason: str,
    ) -> PipelineResult:
        """Latch an ingress failure before a landmark observation exists."""
        decision = self.supervisor.fault(
            session_id=session_id,
            sequence=sequence,
            source_mono_ns=source_mono_ns,
            now_mono_ns=now_mono_ns,
            reason=f"perception ingress: {reason}",
        )
        return PipelineResult(
            sequence,
            None,
            decision,
            None,
            None,
            self._all_groups,
            tuple((group, decision.reasons[0]) for group in sorted(self._all_groups)),
        )

    #: Blend fractions tried, largest first, when the full guarded target's realized
    #: step breaches the clearance floor. The realized off-line direction rotates
    #: with the blend, so the safe set is not a clean prefix near the floor and a
    #: bisection could land on a false boundary; a short coarse scan that accepts the
    #: largest clear blend is robust and keeps as much tracking as the frame allows.
    #: 0.0 (hold at the current pose) is the guaranteed-safe floor and is handled
    #: separately, so it is not in this list.
    _REALIZED_BACKOFF_FRACTIONS = (0.75, 0.5, 0.25)

    def _feasible_governor_command(
        self, guarded_target: Mapping[str, float], *, timestamp_ns: int
    ) -> dict[str, float]:
        """Ease the guarded target back until the governor's realized step is clear.

        The guarded controller validates the straight ``[current, target]`` segment,
        but the governor advances each joint at its own rate, so the pose it realizes
        lands off that segment and can dip below the clearance floor even when the
        straight line is clear (a 0.19 mm breach at demo frame 90 latched a HOLD).
        The independent supervisor checks that realized swept path, so predict it here
        -- the arm/head governor step *and* the separately-governed grippers -- and,
        if it breaches, blend the arm/head target back toward the pose the governor is
        already at until the predicted realized pose clears. Blending toward the
        current pose shrinks the realized excursion back toward the already-approved,
        clear current pose, so holding is always the safe floor. This never relaxes a
        guard rail; it only declines to command motion the supervisor would reject.
        """
        current = self.supervisor.current_pose
        # The realized gripper step does not depend on the arm/head blend, so predict
        # it once and fold it into every clearance query (the supervisor sees the
        # whole-robot pose, grippers included).
        realized_gripper = self.gripper_trajectory.preview(
            self._gripper_desired, timestamp_ns=timestamp_ns
        )

        def realized_clearance(candidate: Mapping[str, float]) -> float:
            """Clearance of the pose the governor would actually realize.

            ``-inf`` when the realized path is rejected outright. Returning the
            DISTANCE rather than a yes/no costs nothing -- the swept check has
            already computed it -- and is what lets the backoff below stop with
            margin in hand instead of exactly on the floor.
            """
            realized = self.trajectory.preview(candidate, timestamp_ns=timestamp_ns)
            end_pose = {**current, **realized, **realized_gripper}
            report = self.checker.check_path(
                current, end_pose, steps=2, max_joint_step=0.01
            )
            if not report.ok:
                return float("-inf")
            distance = report.min_distance
            return float(distance) if isfinite(distance) else float("inf")

        standoff = clearance_floor_for(self.motion_profile) + CLEARANCE_STANDOFF_M
        if realized_clearance(guarded_target) >= standoff:
            return dict(guarded_target)

        start = self._governed_joints
        # This is where the floor-riding actually comes from. The loop used to
        # return the FIRST fraction that merely cleared the floor, which is by
        # construction the one sitting on it: measured on the 2026-08-22 trial,
        # 28.6% of allowed frames reported exactly 5.0000 mm and p25 was 5.0000.
        #
        # Smaller fractions sit nearer the current pose, which is already approved
        # and clear, so the fractions meeting the standoff form a suffix of this
        # list. Taking the first one that meets it therefore still takes the most
        # motion available at that quality. The first merely-clear fraction is
        # remembered on the way past and used only if the standoff is unreachable,
        # so this can never command less motion than the old code did.
        first_clear: dict[str, float] | None = None
        for fraction in self._REALIZED_BACKOFF_FRACTIONS:
            candidate = {
                name: start[name] + fraction * (guarded_target[name] - start[name])
                for name in guarded_target
            }
            clearance = realized_clearance(candidate)
            if clearance >= standoff:
                return candidate
            if first_clear is None and clearance > float("-inf"):
                first_clear = candidate
        if first_clear is not None:
            return first_clear
        # Even holding may breach if momentum from earlier frames already commits the
        # arm to cross the floor this frame -- no target can undo that, and the
        # supervisor rightly HOLDs. Commanding the current pose is the least motion
        # and bleeds the momentum off, so the next frame recovers.
        return dict(start)

    def observe_source_continuity(self, observation: HumanObservation) -> None:
        """Record raw shoulder evidence from a frame the control loop may drop.

        Never authorizes motion, never advances a retargeting filter, never
        commands the twin. It answers the one question the control loop cannot
        answer for itself: was the shoulder line actually observed throughout,
        or was there a genuine blind interval?

        Call this on every detected frame *before* handing it to the control
        worker. Feeding it is optional -- a caller that never calls it leaves
        :meth:`process` behaving exactly as it does without a witness.
        """
        if self._witness_liveness is not None:
            verdict = self._witness_liveness.evaluate(
                fingerprint=observation.content_fingerprint or "",
                capture_mono_ns=observation.capture_mono_ns,
                sequence=observation.sequence,
            )
            if not verdict.live:
                # Frozen or replayed pixels are not observation. Letting them
                # advance the anchor would vouch for a blind interval.
                return
        if observation.identity is not IdentityState.STABLE:
            return
        landmarks = {mark.name: mark for mark in observation.landmarks}
        shoulders = (landmarks.get("pose_11"), landmarks.get("pose_12"))
        if any(mark is None for mark in shoulders):
            return
        if any(
            min(mark.visibility, mark.presence) < self._command_min_confidence
            for mark in shoulders
            if mark is not None
        ):
            return
        yaw = _observed_torso_yaw(observation)
        if yaw is None:
            return
        previous = self._source_continuity
        break_ns = None if previous is None else previous.break_capture_ns
        break_reason = None if previous is None else previous.break_reason
        if previous is not None:
            step = wrapped_yaw_difference(yaw, previous.yaw_rad)
            if (
                observation.capture_mono_ns - previous.capture_ns
                > self._torso_policy.max_observation_gap_ns
            ):
                break_ns = observation.capture_mono_ns
                break_reason = "TORSO_YAW_CONTINUITY_GAP"
            elif (
                step is not None
                and abs(step) > self._torso_policy.max_observation_step_rad
            ):
                break_ns = observation.capture_mono_ns
                break_reason = "TORSO_YAW_DISCONTINUITY"
        # One immutable rebind, so the worker thread reads either the whole old
        # object or the whole new one -- never a torn (yaw, capture_ns) pair.
        self._source_continuity = _SourceContinuity(
            capture_ns=observation.capture_mono_ns,
            yaw_rad=yaw,
            break_capture_ns=break_ns,
            break_reason=break_reason,
        )

    def _recover_torso_yaw(self, observation: HumanObservation) -> None:
        """Clear a latched torso-yaw hold once identity is provable again.

        The latch exists because a >20 deg apparent shoulder step can be a
        LEFT/RIGHT LABEL SWAP, and driving arms through a swapped torso frame
        points them the wrong way in the world. That reasoning is intact. What was
        wrong was the remedy: the flag was cleared only by `calibrate_window` and
        the reset path, and the live CLI has no in-run calibration entry point --
        so both arms, both grippers and the torso were dead for the REST OF THE
        SESSION. Measured on real footage: four of five ordinary perturbations
        (stepping out of shot and back, an occluded shoulder, turning side-on, a
        light change) latched it and never recovered, while the run kept recording
        and exited 0.

        This file already made exactly this call once, one branch below: a raw
        source-time gap used to latch too, and the comment there records that it
        "ended arm tracking for the whole session on the first two held frames
        after calibration". Same disease, same cure -- hold, then let provable
        evidence resume.

        The bar to resume is the bar to start. Recovery needs
        `TORSO_RECOVERY_FRAMES` CONSECUTIVE frames in which all of the following
        hold, and any failure resets the count to zero:

        * the shoulder line is observable at all;
        * **face-front evidence is present** -- this is the identity argument the
          latch is made of. A shoulder-label swap requires the operator to be
          turned past ~105 deg, where the face is not toward the camera, so a
          face-on frame is positive evidence that the labels are the ones we
          think they are. It is the same test this method's neighbour uses to
          decide whether a re-entry after a gap is recoverable;
        * the source time advances by no more than the gap budget, so the chain is
          genuinely consecutive rather than a sample either side of a hole;
        * the yaw step stays well inside the discontinuity threshold -- half of
          it -- because a run of 19 deg steps is not a calm chain.

        Nothing is driven while this counts up: the hold is still applied on every
        one of those frames. On success the continuity witness is re-anchored to
        the frame that proved it, so the resumed chain starts clean rather than
        against a stale pre-latch yaw.

        This does NOT re-run calibration. Shoulder width, eye span and identity
        are unchanged and still hold; the only thing re-established is the yaw
        continuity chain, which is the only thing the latch invalidated.
        """
        yaw = _observed_torso_yaw(observation)
        front_ok = _front_head_evidence_available(observation, self._face_calibration)
        previous_yaw = self._torso_recovery_yaw_rad
        previous_ns = self._torso_recovery_capture_ns
        step = (
            None
            if yaw is None or previous_yaw is None
            else wrapped_yaw_difference(yaw, previous_yaw)
        )
        consecutive = (
            previous_ns is not None
            and 0
            <= observation.capture_mono_ns - previous_ns
            <= self._torso_policy.max_observation_gap_ns
        )
        calm = step is not None and abs(step) <= (
            self._torso_policy.max_observation_step_rad * TORSO_RECOVERY_STEP_FRACTION
        )

        if yaw is None or not front_ok:
            self._torso_recovery_frames = 0
            self._torso_recovery_yaw_rad = yaw
            self._torso_recovery_capture_ns = (
                observation.capture_mono_ns if yaw is not None else None
            )
            return
        if previous_yaw is None or not consecutive or not calm:
            # The first admissible frame is evidence of nothing yet; it is the
            # anchor the next frame is judged against.
            self._torso_recovery_frames = 1
        else:
            self._torso_recovery_frames += 1
        self._torso_recovery_yaw_rad = yaw
        self._torso_recovery_capture_ns = observation.capture_mono_ns
        if self._torso_recovery_frames < TORSO_RECOVERY_FRAMES:
            return

        self._torso_recalibration_required = False
        self._torso_recovery_frames = 0
        # Re-anchor every continuity reference to the frame that proved it, so the
        # first resumed frame is not compared against the pre-latch yaw and does
        # not immediately re-trip the discontinuity test it just recovered from.
        self._last_observed_torso_yaw_rad = yaw
        self._last_observed_torso_capture_ns = observation.capture_mono_ns
        self._source_continuity = _SourceContinuity(
            capture_ns=observation.capture_mono_ns,
            yaw_rad=yaw,
            break_capture_ns=None,
            break_reason=None,
        )

    def process(
        self,
        observation: HumanObservation,
        *,
        now_mono_ns: int,
        occupancy: OccupancyObservation | None = None,
    ) -> PipelineResult:
        # The clearance pose cache lives for exactly one frame: within a frame the
        # pose -> report mapping is pure, across frames the twin has moved.
        self.checker._frame_memo = {}
        try:
            return self._process(observation, now_mono_ns=now_mono_ns, occupancy=occupancy)
        finally:
            self.checker._frame_memo = None

    def _process(
        self,
        observation: HumanObservation,
        *,
        now_mono_ns: int,
        occupancy: OccupancyObservation | None = None,
    ) -> PipelineResult:
        if (
            self._face_calibration is None
            or self._left_arm_calibration is None
            or self._right_arm_calibration is None
        ):
            raise RuntimeError("calibrate the pipeline before processing observations")
        gate_result = self.gate.evaluate(
            observation, now_mono_ns=now_mono_ns, occupancy=occupancy
        )
        # Whole-frame rejections (clock, ordering, freshness, identity) and the
        # case where every group independently failed both land here, and this
        # is byte-for-byte the pre-existing global hold.
        if not gate_result.any_group_accepted:
            self._recovery_streak = 0
            selection_reason = (
                gate_result.selection.reason
                if gate_result.reason
                in {
                    ObservationRejection.OCCUPANCY_MISSING,
                    ObservationRejection.SELECTION_NOT_LOCKED,
                }
                and gate_result.selection is not None
                else None
            )
            return self._held(
                observation,
                now_mono_ns,
                "selection: "
                f"{selection_reason}"
                if selection_reason is not None
                else f"observation: {gate_result.reason}",
                held_group_reasons=tuple(
                    (
                        group,
                        (
                            selection_reason.value
                            if selection_reason is not None
                            else reason.value
                        ),
                    )
                    for group, result in sorted(gate_result.groups.items())
                    if (reason := result.reason) is not None
                ),
            )
        if (
            self.gate.has_selection_manager
            and self._calibration_selection_id != self.gate.selection_id
        ):
            self._recovery_streak = 0
            return self._held(
                observation,
                now_mono_ns,
                "selection: calibration generation mismatch",
                held_group_reasons=tuple(
                    (group, "CALIBRATION_SELECTION_MISMATCH")
                    for group in sorted(self._all_groups)
                ),
            )
        # Fail-closed: a group is driven only when the gate explicitly says its
        # own landmarks are trustworthy. An unknown or unevaluated group is not.
        head_allowed = gate_result.group_accepted(ControlGroup.HEAD)
        left_allowed = gate_result.group_accepted(ControlGroup.LEFT_ARM)
        right_allowed = gate_result.group_accepted(ControlGroup.RIGHT_ARM)
        torso_allowed = gate_result.group_accepted(ControlGroup.TORSO)
        held_groups = gate_result.held_groups
        held_group_reasons = {
            group: reason.value
            for group in held_groups
            if (reason := gate_result.group_reason(group)) is not None
        }
        held_group_residuals_m: dict[str, float] = {}
        held_group_ik_attempts: dict[str, tuple[tuple[str, float], ...]] = {}
        held_grippers: set[str] = set()
        held_gripper_reasons: dict[str, str] = {}
        saturated_groups: set[str] = set()
        saturated_group_reasons: dict[str, str] = {}

        if self._torso_recalibration_required:
            # Evaluate recovery BEFORE applying the hold, so the frame that
            # completes the evidence is also the frame the arms come back on.
            self._recover_torso_yaw(observation)
        if self._torso_recalibration_required:
            coupled_groups = frozenset(
                (
                    str(ControlGroup.TORSO),
                    str(ControlGroup.LEFT_ARM),
                    str(ControlGroup.RIGHT_ARM),
                )
            )
            held_groups = held_groups | coupled_groups
            for group in coupled_groups:
                held_group_reasons[group] = "TORSO_YAW_RECALIBRATION_REQUIRED"
            torso_allowed = False
            left_allowed = False
            right_allowed = False

        # A wrist offset is only meaningful in a human torso frame while that
        # torso frame is observable.  Letting arms continue after the camera has
        # lost left/right torso orientation makes them solve in a frozen robot
        # frame while the operator keeps rotating: visibly moving, but pointed in
        # the wrong world direction.  Gate the coupled groups *before* intent
        # estimation/filtering so none of their state advances on that frame.
        raw_torso_yaw = _observed_torso_yaw(observation) if torso_allowed else None
        torso_front_unobservable = torso_allowed and not _front_head_evidence_available(
            observation, self._face_calibration
        )
        observed_torso_yaw = raw_torso_yaw if not torso_front_unobservable else None
        relative_torso_yaw = (
            wrapped_yaw_difference(
                raw_torso_yaw, self._torso_yaw_reference_rad
            )
            if (
                raw_torso_yaw is not None
                and self._torso_yaw_reference_rad is not None
            )
            else None
        )
        torso_out_of_view = (
            torso_allowed
            and self._torso_yaw_reference_rad is not None
            and observed_torso_yaw is not None
            and (
                abs(observed_torso_yaw) > self._torso_policy.max_camera_yaw_rad
                or relative_torso_yaw is None
                or abs(relative_torso_yaw) > self._torso_policy.max_relative_yaw_rad
            )
        )
        # Continuity is deliberately evaluated from raw shoulders, not from the
        # last command.  A face-mesh dropout holds torso/arms, but continuous
        # shoulder geometry during that hold is still valid *negative evidence*
        # against a source gap.  It never authorizes a command by itself.
        continuity_yaw = self._last_observed_torso_yaw_rad
        continuity_capture_ns = self._last_observed_torso_capture_ns
        # The witness carries the detector's own continuity view. It is only
        # trusted once it has already seen this frame: a caller that calibrates
        # but never feeds it keeps the command-stream rule exactly, so the gap
        # check can never be silently disabled by forgetting to wire it up.
        witness = self._source_continuity
        witness_active = (
            witness is not None and witness.capture_ns >= observation.capture_mono_ns
        )
        witness_break = (
            witness_active
            and witness is not None
            and witness.break_capture_ns is not None
            and (
                continuity_capture_ns is None
                or witness.break_capture_ns > continuity_capture_ns
            )
        )
        torso_discontinuous = (
            torso_allowed
            and raw_torso_yaw is not None
            and (
                (witness_break and witness is not None
                 and witness.break_reason == "TORSO_YAW_DISCONTINUITY")
                or (
                    continuity_yaw is not None
                    and (yaw_step := wrapped_yaw_difference(
                        raw_torso_yaw, continuity_yaw
                    )) is not None
                    and abs(yaw_step) > self._torso_policy.max_observation_step_rad
                )
            )
        )
        torso_observation_too_old = (
            torso_allowed
            and self._torso_yaw_reference_rad is not None
            and raw_torso_yaw is not None
            and (
                (
                    witness_break
                    and witness is not None
                    and witness.break_reason == "TORSO_YAW_CONTINUITY_GAP"
                )
                if witness_active
                else (
                    continuity_capture_ns is not None
                    and observation.capture_mono_ns - continuity_capture_ns
                    > self._torso_policy.max_observation_gap_ns
                )
            )
        )
        torso_hold_reason: str | None = None
        # A raw source-time gap or a semantic shoulder jump wins over a local
        # face/view hold.  Once either occurs, later smooth samples cannot prove
        # that left/right identity remained intact while the source was absent.
        if torso_discontinuous:
            self._torso_recalibration_required = True
            torso_hold_reason = "TORSO_YAW_DISCONTINUITY"
        elif torso_observation_too_old:
            # A source-time gap is MISSING DATA, not evidence of a shoulder-label
            # swap, and the live CLI has no in-run recalibration entry point --
            # latching here ended arm tracking for the whole session on the first
            # two held frames after calibration (measured 2026-08-26: gap at frame
            # 2, arms held 520/521 thereafter). Hold this frame, re-anchor
            # continuity below, and let a continuous next frame resume.
            # The swap defences that remain are the ones that actually bear on
            # identity: required face-front evidence, the semantic 20 deg step,
            # and the out-of-view re-entry direction latch.
            #
            # The one case that stays terminal: a gap whose re-entry frame has no
            # face-front evidence. A shoulder-label swap requires the operator to
            # be turned past ~105 deg, where the face is not toward the camera --
            # so front evidence at re-entry is what makes identity provable, and
            # its absence is what makes the gap unrecoverable.
            if torso_front_unobservable:
                self._torso_recalibration_required = True
            torso_hold_reason = "TORSO_YAW_CONTINUITY_GAP"
        elif torso_front_unobservable:
            torso_hold_reason = "TORSO_FRONT_UNOBSERVABLE"
        elif torso_out_of_view:
            torso_hold_reason = "TORSO_YAW_OUT_OF_VIEW"

        # Advance continuity evidence only after accepting this raw frame as
        # physically/temporally continuous.  This applies during a face/view
        # hold as well as tracking, but never advances the retargeting filters or
        # permits torso/arm motion while a hold is active.
        if (
            torso_allowed
            and raw_torso_yaw is not None
            and not torso_discontinuous
            and not self._torso_recalibration_required
        ):
            self._last_observed_torso_yaw_rad = raw_torso_yaw
            self._last_observed_torso_capture_ns = observation.capture_mono_ns

        if torso_hold_reason is not None:
            coupled_groups = frozenset(
                (
                    str(ControlGroup.TORSO),
                    str(ControlGroup.LEFT_ARM),
                    str(ControlGroup.RIGHT_ARM),
                )
            )
            held_groups = held_groups | coupled_groups
            for group in coupled_groups:
                held_group_reasons[group] = torso_hold_reason
            torso_allowed = False
            left_allowed = False
            right_allowed = False

        # The head's basis failing is a HEAD problem. It says nothing about either
        # arm or the torso, which read entirely different landmarks, so it holds
        # that group and lets the rest keep tracking -- the same rule the gate
        # already applies to a per-group landmark or confidence failure. Before
        # this, a single unusable face geometry froze the whole robot.
        face_intent = None
        if head_allowed:
            try:
                face_intent = estimate_face_pose(observation, self._face_calibration)
            except FacePoseError as error:
                held_groups = held_groups | frozenset((str(ControlGroup.HEAD),))
                held_group_reasons[str(ControlGroup.HEAD)] = f"head input: {error}"
        if face_intent is not None and self._last_accepted_head_basis is not None:
            if face_intent.basis == self._last_accepted_head_basis:
                self._head_recovery_basis = None
            elif self._head_recovery_basis != face_intent.basis:
                # The first sample from an alternate detector establishes only
                # that it exists. Its numerical scale/placement is deliberately
                # not blended into the old detector's filter history or accepted
                # as a new robot target.
                self._head_recovery_basis = face_intent.basis
                self._filters.forget("head.yaw", "head.pitch")
                held_groups = held_groups | frozenset((str(ControlGroup.HEAD),))
                held_group_reasons[str(ControlGroup.HEAD)] = "HEAD_BASIS_TRANSITION"
                face_intent = None
        try:
            left_intent = (
                estimate_left_arm_intent(
                    observation,
                    self._left_arm_calibration,
                    min_confidence=self._command_min_confidence,
                    human_torso_yaw_rad=(
                        observed_torso_yaw
                        if self._torso_yaw_reference_rad is not None
                        and observed_torso_yaw is not None
                        and not torso_out_of_view
                        else None
                    ),
                )
                if left_allowed
                else None
            )
            right_intent = (
                estimate_right_arm_intent(
                    observation,
                    self._right_arm_calibration,
                    min_confidence=self._command_min_confidence,
                    human_torso_yaw_rad=(
                        observed_torso_yaw
                        if self._torso_yaw_reference_rad is not None
                        and observed_torso_yaw is not None
                        and not torso_out_of_view
                        else None
                    ),
                )
                if right_allowed
                else None
            )
        except (ArmRetargetError, CalibrationError) as error:
            self._recovery_streak = 0
            return self._held(observation, now_mono_ns, f"retarget input: {error}")

        timestamp = observation.capture_mono_ns
        if face_intent is not None:
            face_intent = replace(
                face_intent,
                yaw_signal=self._filters.value(
                    "head.yaw", face_intent.yaw_signal, timestamp_ns=timestamp
                ),
                pitch_signal=self._filters.value(
                    "head.pitch", face_intent.pitch_signal, timestamp_ns=timestamp
                ),
            )
        if left_intent is not None:
            left_intent = self._filter_arm_intent("left", left_intent, timestamp)
        if right_intent is not None:
            right_intent = self._filter_arm_intent("right", right_intent, timestamp)
        for side in ("left", "right"):
            gripper = f"{side}_gripper_joint"
            if gripper in held_grippers:
                continue
            openness = estimate_hand_openness(
                observation,
                side,
                min_confidence=self._command_min_confidence,
            )
            if openness is None:
                # Freeze the target at the governor's last approved position,
                # rather than at a potentially distant stale request. This lets
                # an already-moving gripper decelerate and stop instead of
                # continuing to close/open after its hand disappeared.
                held_grippers.add(gripper)
                held_gripper_reasons[gripper] = "HAND_NOT_TRACKABLE"
                self._gripper_desired[gripper] = self.supervisor.current_pose[gripper]
                continue
            filtered_openness = self._filters.value(
                f"{side}.hand_openness", openness, timestamp_ns=timestamp
            )
            self._gripper_desired[gripper] = openness_to_gripper_rad(filtered_openness)

        head_result = (
            self.head.map(face_intent, previous=self._previous_head, elapsed_s=None)
            if face_intent is not None
            else None
        )
        if head_result is not None and head_result.target is not None and head_result.saturated:
            saturated_groups.add(str(ControlGroup.HEAD))
            saturated_group_reasons[str(ControlGroup.HEAD)] = "HEAD_SOFT_LIMIT"
        torso_result = None
        if torso_allowed:
            if self._torso_yaw_reference_rad is None:
                # Supplying no metric world landmarks during calibration is
                # supported for head/arms, but must never quietly reinterpret
                # camera orientation as a torso command.
                held_groups = held_groups | frozenset((str(ControlGroup.TORSO),))
                held_group_reasons[str(ControlGroup.TORSO)] = (
                    "TORSO_YAW_NEUTRAL_UNAVAILABLE"
                )
            elif observed_torso_yaw is None:
                held_groups = held_groups | frozenset((str(ControlGroup.TORSO),))
                held_group_reasons[str(ControlGroup.TORSO)] = "TORSO_YAW_UNAVAILABLE"
            else:
                yaw_signal = relative_torso_yaw
                if yaw_signal is None:
                    held_groups = held_groups | frozenset((str(ControlGroup.TORSO),))
                    held_group_reasons[str(ControlGroup.TORSO)] = "TORSO_YAW_UNAVAILABLE"
                else:
                    torso_result = self.torso.map(
                        TorsoIntent(
                            self._filters.value(
                                "torso.yaw", yaw_signal, timestamp_ns=timestamp
                            )
                        )
                    )
        # Solve both arms against the torso yaw the robot is ACTUALLY at, not the
        # one just commanded. The governor rate-limits the torso like any other
        # joint, so the commanded angle leads the realized one; anchoring the arms
        # to the command would aim them at a chest orientation the robot has not
        # reached, and the clearance check and supervisor both evaluate the pose
        # the robot is really in. One frame of lag is the physically honest answer.
        realized_torso_yaw = self._governed_joints.get(
            TORSO_YAW_JOINT, HOME_QPOS[TORSO_YAW_JOINT]
        )
        self.left_arm.set_torso_yaw(realized_torso_yaw)
        self.right_arm.set_torso_yaw(realized_torso_yaw)
        left_result = (
            self.left_arm.map(
                left_intent, previous=self._previous_left_arm, elapsed_s=None
            )
            if left_intent is not None
            else None
        )
        right_result = (
            self.right_arm.map(
                right_intent, previous=self._previous_right_arm, elapsed_s=None
            )
            if right_intent is not None
            else None
        )
        # A local arm-estimation failure means that arm has no new target; it is
        # not evidence that the head or independently tracked opposite arm changed.
        # Keep that limb at its last governed pose while the complete resulting
        # pose still goes through the same clearance and supervisor checks.  This
        # covers the direction-vector candidate's missing geometry and the
        # wrist-primary solver's bounded non-convergence.  NONFINITE and joint
        # continuity rejections remain whole-frame failures: those indicate a
        # malformed input or an unsafe discontinuity rather than an unavailable
        # local solution.
        limb_local_rejections = frozenset(
            (
                ArmMapRejection.INCOMPLETE_OBSERVATION,
                ArmMapRejection.IK_DID_NOT_CONVERGE,
                # A detected branch flip (I5) is local by nature: one arm found a
                # different route to the same wrist point, which says nothing
                # about the other arm or the head. Holding that limb is the
                # matrix's prescribed response; holding the whole robot would
                # freeze two healthy limbs for it.
                ArmMapRejection.JOINT_CONTINUITY,
            )
        )
        if (
            left_result is not None
            and left_result.rejection in limb_local_rejections
        ):
            assert left_result.rejection is not None
            held_groups = held_groups | frozenset((str(ControlGroup.LEFT_ARM),))
            held_group_reasons[str(ControlGroup.LEFT_ARM)] = left_result.rejection.value
            if left_result.best_wrist_residual_m is not None:
                held_group_residuals_m[str(ControlGroup.LEFT_ARM)] = (
                    left_result.best_wrist_residual_m
                )
            if left_result.ik_attempts:
                held_group_ik_attempts[str(ControlGroup.LEFT_ARM)] = tuple(
                    (attempt.stage.value, attempt.wrist_residual_m)
                    for attempt in left_result.ik_attempts
                )
            left_result = None
        if (
            right_result is not None
            and right_result.rejection in limb_local_rejections
        ):
            assert right_result.rejection is not None
            held_groups = held_groups | frozenset((str(ControlGroup.RIGHT_ARM),))
            held_group_reasons[str(ControlGroup.RIGHT_ARM)] = right_result.rejection.value
            if right_result.best_wrist_residual_m is not None:
                held_group_residuals_m[str(ControlGroup.RIGHT_ARM)] = (
                    right_result.best_wrist_residual_m
                )
            if right_result.ik_attempts:
                held_group_ik_attempts[str(ControlGroup.RIGHT_ARM)] = tuple(
                    (attempt.stage.value, attempt.wrist_residual_m)
                    for attempt in right_result.ik_attempts
                )
            right_result = None
        # A retargeter that rejects its own well-conditioned input is not a
        # per-landmark trust problem, so it still holds the whole robot.
        if head_result is not None and head_result.target is None:
            self._recovery_streak = 0
            return self._held(observation, now_mono_ns, f"head map: {head_result.rejection}")
        if torso_result is not None and torso_result.target is None:
            # Torso-specific mapping rejection is no more evidence against the
            # independently gated head or arms than a single-limb IK failure.
            held_groups = held_groups | frozenset((str(ControlGroup.TORSO),))
            held_group_reasons[str(ControlGroup.TORSO)] = (
                f"torso map: {torso_result.rejection}"
            )
            torso_result = None
        if left_result is not None and left_result.target is None:
            self._recovery_streak = 0
            return self._held(observation, now_mono_ns, f"left arm map: {left_result.rejection}")
        if right_result is not None and right_result.target is None:
            self._recovery_streak = 0
            return self._held(observation, now_mono_ns, f"right arm map: {right_result.rejection}")

        # A transient perception HOLD must stop output immediately, but it must
        # not wedge a live preview forever. Re-arm only after the profile's
        # consecutive-good-frame requirement is met. Rotation of the
        # supervisor generation invalidates every pre-HOLD target, and both the
        # supervisor and governor restart from the last approved simulator pose.
        if self.supervisor.state is SupervisorState.HOLD:
            self._recovery_streak += 1
            if self._recovery_streak < self._recovery_required_frames:
                return self._held(
                    observation,
                    now_mono_ns,
                    "tracking recovery "
                    f"{self._recovery_streak}/{self._recovery_required_frames}",
                )
            # Seed one nominal 30 Hz interval behind the recovery frame. A 1 ns
            # interval is mathematically valid but below useful qpos precision;
            # reconstructing velocity from the rounded position can then appear
            # to violate acceleration by orders of magnitude.
            seed_timestamp_ns = max(0, observation.capture_mono_ns - 33_333_333)
            self.supervisor.resume_preview(source_mono_ns=seed_timestamp_ns)
            controlled_current = {
                name: self.supervisor.current_pose[name]
                for name in self.trajectory.velocities
            }
            self.trajectory.reset(controlled_current, timestamp_ns=seed_timestamp_ns)
            self._governed_joints = dict(controlled_current)
            self._previous_left_arm = {
                name: controlled_current[name] for name in self.left_arm.joint_names
            }
            self._previous_right_arm = {
                name: controlled_current[name] for name in self.right_arm.joint_names
            }
            controlled_grippers = {
                name: self.supervisor.current_pose[name]
                for name in self.gripper_trajectory.velocities
            }
            self.gripper_trajectory.reset(
                controlled_grippers, timestamp_ns=seed_timestamp_ns
            )
            # Recovery changes only robot-command authority. The camera stream,
            # capture clock, ordering baseline, and liveness evidence remain
            # continuous; resetting them here would cause a healthy source to
            # alternate NOT_EVALUATED holds and resumed previews indefinitely.
            self._recovery_streak = 0

        # A held group is commanded to the position the governor last emitted
        # for it, so it decelerates to a standstill and stays there while the
        # tracked groups keep moving. Seeding from the governor rather than from
        # the last desired pose is what makes "held" mean stopped: a stale
        # desired pose would keep pulling the frozen limb toward it.
        desired_joints = dict(self._governed_joints)
        if head_result is not None:
            desired_joints["head_joint1"] = head_result.target.yaw_rad
            desired_joints["head_joint2"] = head_result.target.pitch_rad
        if torso_result is not None and torso_result.target is not None:
            desired_joints[TORSO_YAW_JOINT] = torso_result.target.yaw_rad
        if left_result is not None:
            desired_joints.update(left_result.target.joint_map)
        if right_result is not None:
            desired_joints.update(right_result.target.joint_map)
        # Bend the raw target so it is feasible before the governor chases it:
        # clamp into the soft-limit band and, if the straight joint-space line
        # would drive through a self-collision, advance only as far along it as is
        # clear this frame (continuing next frame) instead of ramming the wall and
        # freezing. Grippers are governed separately and passed only as clearance
        # context so the swept check sees the whole-robot pose.
        guarded = self.guarded.solve(
            self._governed_joints,
            desired_joints,
            context=self._gripper_desired,
        )
        # The guarded target is clearance-safe as a straight [current, target]
        # segment, but the governor rate-limits each joint independently, so the
        # pose it actually realizes lands off that segment and can dip below the
        # clearance floor -- which the supervisor checks and would HOLD. Ease the
        # command back toward the current pose until the governor's *realized* next
        # pose is clear, so the arm keeps flowing instead of latching a HOLD.
        command = self._feasible_governor_command(
            guarded.target, timestamp_ns=observation.capture_mono_ns
        )
        governed_joints = self.trajectory.step(
            command,
            timestamp_ns=observation.capture_mono_ns,
        )
        self._governed_joints = dict(governed_joints)
        governed_joints.update(
            self.gripper_trajectory.step(
                self._gripper_desired,
                timestamp_ns=observation.capture_mono_ns,
            )
        )
        joints = tuple(
            JointTarget(
                name=name,
                position_rad=position,
                max_velocity_rad_s=0.5 if name.endswith("_gripper_joint") else None,
                max_acceleration_rad_s2=1.0
                if name.endswith("_gripper_joint")
                else None,
            )
            for name, position in governed_joints.items()
        )
        joint_map = {joint.name: joint.position_rad for joint in joints}
        # Gating the INPUT is per-limb; judging the OUTPUT never is. The
        # clearance sweep and everything the supervisor checks below run on the
        # complete resulting pose, held joints included, exactly as before.
        clearance = self.supervisor.predict_clearance(joint_map)
        residual_m = max(
            left_result.target.residual_m
            if left_result is not None
            else self._previous_residual_m["left"],
            right_result.target.residual_m
            if right_result is not None
            else self._previous_residual_m["right"],
        )
        target = RobotTarget(
            session_id=observation.session_id,
            sequence=observation.sequence,
            source_clock_id=observation.source_clock_id,
            source_mono_ns=observation.capture_mono_ns,
            arm_generation=self.supervisor.arm_generation,
            joints=joints,
            model_hash=CANONICAL_MANIFEST.fixed_mjcf_sha256,
            tool_hash=SupervisorPolicy().tool_hash,
            mapping_hash=self.mapping_hash,
            ik_residual_m=residual_m,
            predicted_clearance_m=clearance.min_distance,
            # Only for a limb that actually retargeted this frame. A held limb is
            # commanded to where the governor last left it, which is not a claim
            # about reaching any wrist target, so declaring one would invite the
            # supervisor to check an assertion nobody made.
            left_wrist_target_world_m=(
                left_result.target.wrist_target_world_m
                if left_result is not None
                else None
            ),
            right_wrist_target_world_m=(
                right_result.target.wrist_target_world_m
                if right_result is not None
                else None
            ),
            left_ik_joints_rad=(
                tuple(left_result.target.joints_rad)
                if left_result is not None
                else None
            ),
            right_ik_joints_rad=(
                tuple(right_result.target.joints_rad)
                if right_result is not None
                else None
            ),
        )
        supervision = self.supervisor.evaluate(target, now_mono_ns=now_mono_ns)
        receipt = None
        observed = None
        if supervision.approved is not None:
            receipt = self.sink.submit(supervision.approved)
            if type(receipt.accepted) is not bool or receipt.accepted is not True:
                decision = self.supervisor.fault(
                    session_id=observation.session_id,
                    sequence=observation.sequence,
                    now_mono_ns=now_mono_ns,
                    source_mono_ns=observation.capture_mono_ns,
                    reason=f"unknown or rejected sink status: {receipt.accepted!r}",
                )
                return PipelineResult(
                    observation_sequence=observation.sequence,
                    target=target,
                    decision=decision,
                    receipt=receipt,
                    observed_joints=None,
                    held_groups=held_groups,
                    held_group_reasons=tuple(sorted(held_group_reasons.items())),
                    held_grippers=frozenset(held_grippers),
                    held_gripper_reasons=tuple(sorted(held_gripper_reasons.items())),
                    saturated_groups=frozenset(saturated_groups),
                    saturated_group_reasons=tuple(sorted(saturated_group_reasons.items())),
                    held_group_residuals_m=tuple(sorted(held_group_residuals_m.items())),
                    held_group_ik_attempts=tuple(sorted(held_group_ik_attempts.items())),
                )
            if hasattr(self.sink, "joint_positions"):
                observed = self.sink.joint_positions()  # type: ignore[attr-defined]
            # Only a group that actually retargeted this frame updates its
            # continuity reference; a held group keeps the last state it was
            # approved in, so it resumes from there rather than from a gap.
            if head_result is not None:
                self._previous_head = head_result.target
                self._last_accepted_head_basis = face_intent.basis
                self._head_recovery_basis = None
            # The mapper's warm-start/continuity reference must be the pose that
            # actually reached the twin, not its ungoverned IK wish.  Otherwise
            # each new solve starts ahead of the clearance-conditioned trajectory
            # and keeps asking the governor to chase an unreachable moving branch.
            # Update both arms even when one was held: a held governor decelerates
            # toward its emitted pose, which is still the only truthful next seed.
            self._previous_left_arm = {
                name: governed_joints[name] for name in self.left_arm.joint_names
            }
            self._previous_right_arm = {
                name: governed_joints[name] for name in self.right_arm.joint_names
            }
            if left_result is not None:
                self._previous_residual_m["left"] = left_result.target.residual_m
            if right_result is not None:
                self._previous_residual_m["right"] = right_result.target.residual_m
            self._previous_timestamp_ns = observation.capture_mono_ns
            self._recovery_streak = 0
        return PipelineResult(
            observation_sequence=observation.sequence,
            target=target,
            decision=supervision.decision,
            receipt=receipt,
            observed_joints=observed,
            held_groups=held_groups,
            held_group_reasons=tuple(sorted(held_group_reasons.items())),
            held_grippers=frozenset(held_grippers),
            held_gripper_reasons=tuple(sorted(held_gripper_reasons.items())),
            saturated_groups=frozenset(saturated_groups),
            saturated_group_reasons=tuple(sorted(saturated_group_reasons.items())),
            held_group_residuals_m=tuple(sorted(held_group_residuals_m.items())),
            held_group_ik_attempts=tuple(sorted(held_group_ik_attempts.items())),
        )

    def process_fail_closed(
        self,
        observation: HumanObservation,
        *,
        now_mono_ns: int,
        occupancy: OccupancyObservation | None = None,
    ) -> PipelineResult:
        """Public runner boundary: unknown failures become FAULT, never a command."""
        try:
            return self.process(
                observation, now_mono_ns=now_mono_ns, occupancy=occupancy
            )
        except Exception as error:
            decision = self.supervisor.fault(
                session_id=observation.session_id,
                sequence=observation.sequence,
                now_mono_ns=now_mono_ns,
                source_mono_ns=observation.capture_mono_ns,
                reason=f"unexpected pipeline failure: {type(error).__name__}: {error}",
            )
            return PipelineResult(
                observation.sequence,
                None,
                decision,
                None,
                None,
                self._all_groups,
                tuple((group, decision.reasons[0]) for group in sorted(self._all_groups)),
            )

    @property
    def _all_groups(self) -> frozenset[str]:
        return frozenset(str(name) for name in self.gate.control_groups)

    @property
    def liveness_policy(self) -> LivenessPolicy | None:
        """The explicitly configured frame-liveness policy, if any."""
        return self.gate.liveness_policy

    def _held(
        self,
        observation: HumanObservation,
        now_mono_ns: int,
        reason: str,
        *,
        held_group_reasons: tuple[tuple[str, str], ...] | None = None,
    ) -> PipelineResult:
        """Whole-robot hold: no group is commanded, so every group is held."""
        decision = self.supervisor.hold(
            session_id=observation.session_id,
            sequence=observation.sequence,
            source_mono_ns=observation.capture_mono_ns,
            now_mono_ns=now_mono_ns,
            reason=reason,
        )
        return PipelineResult(
            observation.sequence,
            None,
            decision,
            None,
            None,
            self._all_groups,
            held_group_reasons
            if held_group_reasons is not None
            else tuple((group, reason) for group in sorted(self._all_groups)),
        )

    def _filter_arm_intent(self, side: str, intent, timestamp_ns: int):
        updates = {
            field: self._filters.value(
                f"{side}.{field}", getattr(intent, field), timestamp_ns=timestamp_ns
            )
            for field in (
                "lateral_signal",
                "vertical_signal",
                "depth_signal",
                "elbow_lateral_signal",
                "elbow_vertical_signal",
                "elbow_depth_signal",
            )
        }
        # Filter direction vectors on the same source timestamps as the baseline
        # wrist/elbow signals. Re-normalise after component filtering so the
        # direction-vector candidate cannot introduce a raw-noise confound.
        for field in ("upper_arm_direction", "forearm_direction"):
            direction = getattr(intent, field)
            if direction is None:
                continue
            filtered = tuple(
                self._filters.value(
                    f"{side}.{field}.{axis}", component, timestamp_ns=timestamp_ns
                )
                for axis, component in enumerate(direction)
            )
            magnitude = sqrt(sum(component * component for component in filtered))
            updates[field] = (
                tuple(component / magnitude for component in filtered)
                if magnitude > 1e-9
                else None
            )
        return replace(intent, **updates)
