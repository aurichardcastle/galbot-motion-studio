"""Normalized, neutral-relative face pose estimate for the two-DOF head mapper."""

from __future__ import annotations

from galbot_motion_studio.contracts.human import HumanObservation, IdentityState
from galbot_motion_studio.retarget.head import HeadPoseIntent
from galbot_motion_studio.vision.calibration import NeutralCalibration


class FacePoseError(ValueError):
    pass


def estimate_face_pose(
    observation: HumanObservation, calibration: NeutralCalibration
) -> HeadPoseIntent:
    """Return relative, dimensionless yaw/pitch signals; gains are applied downstream.

    This deliberately estimates no metric 3D depth and does not use webcam image
    orientation as a physical robot frame. Both estimates are relative to the
    explicit neutral capture and normalized by eye span.
    """
    if observation.identity is not IdentityState.STABLE:
        raise FacePoseError("face pose requires stable identity")
    if observation.source_clock_id != calibration.source_clock_id:
        raise FacePoseError("observation and calibration clocks differ")
    landmarks = {landmark.name: landmark for landmark in observation.landmarks}
    # Prefer the face mesh, which is the more precise basis, but fall back to the
    # pose model's own nose and eyes when the mesh is absent -- which it often is.
    # Measured on the 2026-08-22 trial: the head was held on 20.4% of frames for a
    # missing `face_1/33/263` while the pose landmarks were present on 94.9%.
    #
    # Each basis is measured against the neutral captured in THAT basis. Mixing
    # them would be a silent scale error: `face_33`/`face_263` are the outer eye
    # corners and `pose_2`/`pose_5` are eye centres, so the two eye spans differ
    # by a person-dependent ratio and the normalisation would not cancel.
    face_names = ("face_1", "face_33", "face_263")
    pose_names = ("pose_0", "pose_2", "pose_5")
    if all(name in landmarks for name in face_names):
        source = face_names
        basis = "face_mesh"
        neutral_nose = calibration.face_nose_normalized_xyz
        neutral_center = calibration.eye_center_normalized_xyz
        neutral_span = calibration.eye_span_normalized
    elif all(name in landmarks for name in pose_names) and (
        calibration.pose_nose_normalized_xyz is not None
    ):
        source = pose_names
        basis = "pose"
        neutral_nose = calibration.pose_nose_normalized_xyz
        neutral_center = calibration.pose_eye_center_normalized_xyz
        neutral_span = calibration.pose_eye_span_normalized
    else:
        missing = [name for name in face_names if name not in landmarks]
        raise FacePoseError(
            f"face pose is missing {missing[0] if missing else face_names[0]}"
            " and has no calibrated pose-landmark fallback"
        )
    nose = landmarks[source[0]].normalized_xyz
    left_eye = landmarks[source[1]].normalized_xyz
    right_eye = landmarks[source[2]].normalized_xyz
    eye_center = tuple((left + right) / 2 for left, right in zip(left_eye, right_eye, strict=True))
    eye_span = abs(right_eye[0] - left_eye[0])
    # The floor scales with the basis. 0.02 was sized for the face mesh, whose
    # `face_33`/`face_263` are the OUTER eye corners; the pose model's `pose_2`/
    # `pose_5` are eye CENTRES and are systematically closer together, so the same
    # absolute floor rejects a perfectly good pose-basis frame. Measured: applying
    # the face floor to the fallback turned 7 frames of the 2026-08-22 replay into
    # whole-frame holds.
    #
    # `min` rather than a plain proportion so the face path is bit-for-bit
    # unchanged: for a face neutral span of ~0.08 this is still 0.02. Half the
    # calibrated span is the meaningful test either way -- it says the eyes have
    # collapsed toward each other, which is what a lost or near-side-on face looks
    # like, and it is measured against THIS operator rather than a global constant.
    minimum_span = min(0.02, 0.5 * neutral_span)
    if eye_span < minimum_span:
        raise FacePoseError("current eye span is too small")
    neutral_yaw = (neutral_nose[0] - neutral_center[0]) / neutral_span
    neutral_pitch = (neutral_nose[1] - neutral_center[1]) / neutral_span
    # Yaw is negated. The operator watches a MIRRORED preview (display.py flips the
    # camera horizontally by default, so the panel behaves like a mirror), but this
    # signal is computed from the RAW, unflipped frame. Without the negation the two
    # disagree in sign and the robot turns its head the opposite way to the operator:
    # reported directly as "when i turn my head left the robot turns it the opposite
    # way". Pitch is unaffected -- a horizontal flip does not change image Y.
    return HeadPoseIntent(
        yaw_signal=-((nose[0] - eye_center[0]) / eye_span - neutral_yaw),
        pitch_signal=(nose[1] - eye_center[1]) / eye_span - neutral_pitch,
        basis=basis,
    )
