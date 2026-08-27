"""Deterministic landmark trajectory used to dogfood the complete offline product."""

from __future__ import annotations

from galbot_motion_studio.contracts.human import HumanObservation, IdentityState, Landmark


#: How far the synthetic operator's head moves, as a fraction of frame width.
NOSE_SHIFT_FRACTION = 0.14

#: How far the synthetic operator's wrists move, as a fraction of the frame.
#: Kept below 0.35 so the wrists stay inside the frame at motion_fraction = 1.0
#: (they start at x = 0.20 and x = 0.80).
WRIST_SHIFT_FRACTION = 0.20

#: Elbows follow the wrists at lower amplitude so the gesture is an upper-arm
#: sweep rather than a fixed-elbow wrist nudge.
ELBOW_SHIFT_FRACTION = 0.04


def _hand_landmarks(side: str, openness: float) -> tuple[Landmark, ...]:
    center_x = 0.80 if side == "left" else 0.20
    points = {index: (center_x, 0.50, 0.0) for index in range(21)}
    chains = ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    for finger, (proximal, joint, tip) in enumerate(chains):
        x = center_x + (finger - 2) * 0.015
        points[proximal] = (x, 0.50, 0.0)
        points[joint] = (x, 0.43, 0.0)
        # Straight/open points away from the proximal segment; folded/closed
        # points back toward it. Interpolation exercises continuous aperture.
        points[tip] = (x, 0.50 - 0.14 * openness, 0.0)
    return tuple(
        Landmark(
            name=f"{side}_hand_{index}",
            normalized_xyz=points[index],
            visibility=1.0,
            presence=1.0,
        )
        for index in range(21)
    )

def synthetic_observation(
    sequence: int,
    *,
    timestamp_ns: int,
    motion_fraction: float,
) -> HumanObservation:
    if not 0.0 <= motion_fraction <= 1.0:
        raise ValueError("motion_fraction must be in [0, 1]")
    # Amplitudes are a fraction of frame width/height, so they describe how far the
    # synthetic operator actually moves. They were 0.08 / 0.105 -- a nose sliding 8%
    # of the frame and a wrist 10.5% -- which is a nudge, not a wave, and produced a
    # robot that measurably moved while looking motionless (14 deg peak excursion
    # across the whole clip).
    #
    # A person waving moves a wrist across a third of the frame and turns their head
    # noticeably. These values are bounded by the landmark coordinate space, not by
    # the safety envelope: the SIM motion profile permits 1.5 rad/s (the model's own
    # URDF limit) and the clearance gate passes a 46 deg shoulder swing with 11.4 mm
    # to spare, so the gate is no longer what bounds expressiveness.
    nose_shift = NOSE_SHIFT_FRACTION * motion_fraction
    wrist_shift = WRIST_SHIFT_FRACTION * motion_fraction
    elbow_shift = ELBOW_SHIFT_FRACTION * motion_fraction
    landmarks = (
        # MediaPipe names anatomy from the person's perspective. In an
        # unmirrored camera image the person's left side therefore appears on
        # screen-right (pose_11), and the right side on screen-left (pose_12).
        Landmark(name="pose_11", normalized_xyz=(0.70, 0.40, 0.0), visibility=1.0, presence=1.0),
        Landmark(name="pose_12", normalized_xyz=(0.30, 0.40, 0.0), visibility=1.0, presence=1.0),
        Landmark(
            name="pose_13",
            normalized_xyz=(0.75 + elbow_shift, 0.55 - elbow_shift, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
        Landmark(
            name="pose_15",
            normalized_xyz=(0.80 + wrist_shift, 0.70 - wrist_shift, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
        Landmark(
            name="pose_14",
            normalized_xyz=(0.25 - elbow_shift, 0.55 - elbow_shift, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
        Landmark(
            name="pose_16",
            normalized_xyz=(0.20 - wrist_shift, 0.70 - wrist_shift, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
        Landmark(
            name="face_1",
            normalized_xyz=(0.50 + nose_shift, 0.20 + nose_shift / 2, 0.0),
            visibility=1.0,
            presence=1.0,
        ),
        Landmark(name="face_33", normalized_xyz=(0.40, 0.20, 0.0), visibility=1.0, presence=1.0),
        Landmark(name="face_263", normalized_xyz=(0.60, 0.20, 0.0), visibility=1.0, presence=1.0),
    ) + _hand_landmarks("left", 1.0 - motion_fraction) + _hand_landmarks(
        "right", motion_fraction
    )
    return HumanObservation(
        session_id="synthetic-demo",
        sequence=sequence,
        source_clock_id="synthetic-clock",
        source_mono_ns=timestamp_ns,
        camera_id="synthetic",
        calibration_id="synthetic-neutral-v1",
        capture_mono_ns=timestamp_ns,
        inference_complete_mono_ns=timestamp_ns,
        image_width_px=640,
        image_height_px=480,
        track_id="synthetic-operator",
        identity=IdentityState.STABLE,
        aggregate_confidence=1.0,
        landmarks=landmarks,
    )
