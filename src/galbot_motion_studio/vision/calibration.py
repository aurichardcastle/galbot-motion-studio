"""Explicit neutral-pose calibration for shoulder-relative retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, isfinite
from typing import Iterable, Sequence

from galbot_motion_studio.contracts.human import HumanObservation, IdentityState, Landmark


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class CalibrationWindowPolicy:
    """Explicit, deployment-owned limits for neutral-pose calibration.

    There are no defaults because sample count, duration, and motion tolerance
    are safety/ergonomics measurements for a particular camera placement. A
    caller that has not supplied them has not supplied a production calibration
    policy and cannot accidentally fall back to one good-looking frame.
    """

    min_samples: int
    max_window_ns: int
    max_center_deviation_normalized: float
    max_shoulder_width_deviation_normalized: float
    max_eye_span_deviation_normalized: float

    def __post_init__(self) -> None:
        if self.min_samples < 2 or self.max_window_ns <= 0:
            raise ValueError("calibration needs at least two samples and a positive window")
        tolerances = (
            self.max_center_deviation_normalized,
            self.max_shoulder_width_deviation_normalized,
            self.max_eye_span_deviation_normalized,
        )
        if any(not isfinite(value) or value < 0 for value in tolerances):
            raise ValueError("calibration window tolerances must be finite and non-negative")


@dataclass(frozen=True)
class NeutralCalibration:
    source_clock_id: str
    observation_sequence: int
    shoulder_center_normalized_xyz: tuple[float, float, float]
    shoulder_width_normalized: float
    face_nose_normalized_xyz: tuple[float, float, float]
    eye_center_normalized_xyz: tuple[float, float, float]
    eye_span_normalized: float
    #: The same three quantities measured on the POSE model's own nose and eye
    #: landmarks (`pose_0`, `pose_2`, `pose_5`) rather than the face mesh's.
    #:
    #: They are a separate basis, not a substitute: `face_33`/`face_263` are the
    #: OUTER eye corners while `pose_2`/`pose_5` are eye centres, so the two eye
    #: spans differ by a person-dependent ratio and a yaw normalised by one cannot
    #: be compared against a neutral captured with the other. Recording both means
    #: head pose can be estimated from whichever basis the frame actually carries,
    #: each against its own neutral.
    #:
    #: This exists because the face mesh is the unreliable one. On the 2026-08-22
    #: trial the head was held on 20.4% of frames for a missing `face_1/33/263`,
    #: and calibration spent 816 frames waiting for a face that MediaPipe was not
    #: producing, while the pose landmarks were present on 94.9%.
    pose_nose_normalized_xyz: tuple[float, float, float] | None = None
    pose_eye_center_normalized_xyz: tuple[float, float, float] | None = None
    pose_eye_span_normalized: float | None = None


def create_neutral_calibration(observation: HumanObservation) -> NeutralCalibration:
    """Create calibration only from a stable, complete frame; never guess missing anatomy."""
    if observation.identity is not IdentityState.STABLE:
        raise CalibrationError("neutral calibration requires a stable tracked person")
    landmarks = {landmark.name: landmark for landmark in observation.landmarks}
    try:
        left_shoulder = landmarks["pose_11"]
        right_shoulder = landmarks["pose_12"]
        face_nose = landmarks["face_1"]
        left_eye = landmarks["face_33"]
        right_eye = landmarks["face_263"]
    except KeyError as error:
        raise CalibrationError(f"neutral calibration is missing {error.args[0]}") from error
    if min(
        left_shoulder.visibility,
        right_shoulder.visibility,
        face_nose.visibility,
        left_eye.visibility,
        right_eye.visibility,
    ) < 0.5:
        raise CalibrationError("neutral calibration requires high-visibility shoulders and face")
    # The pose-model basis, when the frame carries it. Recorded, never required:
    # a calibration without it simply has no fallback and behaves exactly as it
    # did before. Its own visibility gate is the same 0.5 the rest of the frame
    # is held to -- unlike the face landmarks, these carry real visibility values.
    pose_basis: dict[str, object] = {}
    pose_nose = landmarks.get("pose_0")
    pose_left_eye = landmarks.get("pose_2")
    pose_right_eye = landmarks.get("pose_5")
    if pose_nose is not None and pose_left_eye is not None and pose_right_eye is not None:
        pose_eye_span = normalized_landmark_distance(pose_left_eye, pose_right_eye)
        if (
            min(pose_nose.visibility, pose_left_eye.visibility, pose_right_eye.visibility)
            >= 0.5
            and pose_eye_span >= 0.02
        ):
            pose_basis = {
                "pose_nose_normalized_xyz": pose_nose.normalized_xyz,
                "pose_eye_center_normalized_xyz": tuple(
                    (left + right) / 2
                    for left, right in zip(
                        pose_left_eye.normalized_xyz,
                        pose_right_eye.normalized_xyz,
                        strict=True,
                    )
                ),
                "pose_eye_span_normalized": pose_eye_span,
            }
    center = tuple(
        (left + right) / 2 for left, right in zip(left_shoulder.normalized_xyz, right_shoulder.normalized_xyz, strict=True)
    )
    width = normalized_landmark_distance(left_shoulder, right_shoulder)
    if width < 0.05:
        raise CalibrationError("shoulder width is implausibly small")
    eye_span = normalized_landmark_distance(left_eye, right_eye)
    if eye_span < 0.02:
        raise CalibrationError("eye span is implausibly small")
    return NeutralCalibration(
        source_clock_id=observation.source_clock_id,
        observation_sequence=observation.sequence,
        shoulder_center_normalized_xyz=center,
        shoulder_width_normalized=width,
        face_nose_normalized_xyz=face_nose.normalized_xyz,
        eye_center_normalized_xyz=tuple(
            (left + right) / 2
            for left, right in zip(left_eye.normalized_xyz, right_eye.normalized_xyz, strict=True)
        ),
        eye_span_normalized=eye_span,
        **pose_basis,
    )


def create_neutral_calibration_window(
    observations: Sequence[HumanObservation], policy: CalibrationWindowPolicy
) -> NeutralCalibration:
    """Create a calibration only from a quiet, ordered stable-observation window.

    The returned neutral geometry is the arithmetic mean of independently valid
    samples.  More importantly, this function is a validator: a moving operator,
    changed source/camera, time jump, duplicate/out-of-order sequence, or too
    short window never produces a calibration artifact.
    """
    samples = tuple(observations)
    if len(samples) < policy.min_samples:
        raise CalibrationError("calibration window has too few samples")
    first = samples[0]
    previous = first
    for observation in samples[1:]:
        if (
            observation.source_clock_id != first.source_clock_id
            or observation.camera_id != first.camera_id
        ):
            raise CalibrationError("calibration window source or camera changed")
        if (
            observation.capture_mono_ns <= previous.capture_mono_ns
            or observation.sequence <= previous.sequence
        ):
            raise CalibrationError("calibration window is not strictly ordered")
        previous = observation
    if samples[-1].capture_mono_ns - first.capture_mono_ns > policy.max_window_ns:
        raise CalibrationError("calibration window exceeded its duration")

    calibrations = tuple(create_neutral_calibration(observation) for observation in samples)
    center = _mean_xyz(calibration.shoulder_center_normalized_xyz for calibration in calibrations)
    shoulder_width = _mean_scalar(
        calibration.shoulder_width_normalized for calibration in calibrations
    )
    face_nose = _mean_xyz(calibration.face_nose_normalized_xyz for calibration in calibrations)
    eye_center = _mean_xyz(calibration.eye_center_normalized_xyz for calibration in calibrations)
    eye_span = _mean_scalar(calibration.eye_span_normalized for calibration in calibrations)

    if max(
        dist(calibration.shoulder_center_normalized_xyz, center) for calibration in calibrations
    ) > policy.max_center_deviation_normalized:
        raise CalibrationError("calibration window shoulder centre moved too much")
    if max(
        abs(calibration.shoulder_width_normalized - shoulder_width) for calibration in calibrations
    ) > policy.max_shoulder_width_deviation_normalized:
        raise CalibrationError("calibration window shoulder width varied too much")
    if max(
        abs(calibration.eye_span_normalized - eye_span) for calibration in calibrations
    ) > policy.max_eye_span_deviation_normalized:
        raise CalibrationError("calibration window eye span varied too much")

    # The face mesh and pose model are separate head-pose bases.  Preserve the
    # pose basis only when it was complete and valid in *every* sample: averaging
    # a subset of the window would turn a stable-window calibration into a
    # silent mid-window source switch.  This keeps a face-less live frame able
    # to use the documented fallback after ``calibrate_window``.
    pose_basis: dict[str, object] = {}
    if all(
        calibration.pose_nose_normalized_xyz is not None
        and calibration.pose_eye_center_normalized_xyz is not None
        and calibration.pose_eye_span_normalized is not None
        for calibration in calibrations
    ):
        pose_noses = tuple(
            calibration.pose_nose_normalized_xyz
            for calibration in calibrations
            if calibration.pose_nose_normalized_xyz is not None
        )
        pose_eye_centers = tuple(
            calibration.pose_eye_center_normalized_xyz
            for calibration in calibrations
            if calibration.pose_eye_center_normalized_xyz is not None
        )
        pose_eye_spans = tuple(
            calibration.pose_eye_span_normalized
            for calibration in calibrations
            if calibration.pose_eye_span_normalized is not None
        )
        pose_basis = {
            "pose_nose_normalized_xyz": _mean_xyz(pose_noses),
            "pose_eye_center_normalized_xyz": _mean_xyz(pose_eye_centers),
            "pose_eye_span_normalized": _mean_scalar(pose_eye_spans),
        }

    return NeutralCalibration(
        source_clock_id=first.source_clock_id,
        observation_sequence=samples[-1].sequence,
        shoulder_center_normalized_xyz=center,
        shoulder_width_normalized=shoulder_width,
        face_nose_normalized_xyz=face_nose,
        eye_center_normalized_xyz=eye_center,
        eye_span_normalized=eye_span,
        **pose_basis,
    )


def normalized_landmark_distance(left: Landmark, right: Landmark) -> float:
    """The one normalized-space Euclidean distance used for calibration scale."""
    return dist(left.normalized_xyz, right.normalized_xyz)


def _mean_xyz(values: Iterable[tuple[float, float, float]]) -> tuple[float, float, float]:
    points = tuple(values)
    if not points:
        raise ValueError("cannot average empty coordinate collection")
    return tuple(
        sum(point[index] for point in points) / len(points) for index in range(3)
    )


def _mean_scalar(values: Iterable[float]) -> float:
    numbers = tuple(values)
    if not numbers:
        raise ValueError("cannot average empty scalar collection")
    return sum(numbers) / len(numbers)
