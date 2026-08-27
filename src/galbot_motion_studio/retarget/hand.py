"""Scale-independent MediaPipe hand openness to parallel-gripper targets."""

from __future__ import annotations

from math import isfinite, sqrt

from galbot_motion_studio.contracts.human import HumanObservation


FINGER_CHAINS = (
    (2, 3, 4),
    (5, 6, 8),
    (9, 10, 12),
    (13, 14, 16),
    (17, 18, 20),
)
# Keep the model drive joint inside the supervisor's 0.09 rad soft-limit margin.
# These still render as fully open/closed to the eye while avoiding commands on a
# hard mechanical stop.
GRIPPER_OPEN_RAD = 0.10
GRIPPER_CLOSED_RAD = 1.60
# Fitted to MEASURED operator hands, not chosen. Running the detector over two real
# sessions gave a left-hand openness range of 0.249..0.997:
#     operator_crop.mp4   min 0.249  p10 0.291  median 0.448  p90 0.963  max 0.992
#     preview7 (operator) min 0.280  p10 0.518  median 0.823  p90 0.991  max 0.997
# The lowest-scoring frame was visually confirmed to be a curled hand and the highest
# an open one, so the metric's direction is correct.
#
# The previous 0.72/0.38 were fitted to nothing. 0.72 sat BELOW one session's median
# (0.823) and ABOVE the other's (0.448), so the same gesture read as fully open in one
# and 80% closed in the other -- the operator saw an open hand command a shut gripper.
# 0.95/0.32 spans the measured range with a little saturation at each end.
OPEN_HAND_SCORE = 0.95
CLOSED_HAND_SCORE = 0.32


def estimate_hand_openness(
    observation: HumanObservation,
    side: str,
    *,
    min_confidence: float = 0.35,
) -> float | None:
    """Return 0..1 hand openness, or ``None`` when the hand is not trackable.

    The score averages five scale- and rotation-independent finger extension
    angles. Missing hands do not guess a close command; the pipeline retains the
    last approved gripper pose until the dedicated hand tracker returns.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    by_name = {landmark.name: landmark for landmark in observation.landmarks}
    prefix = f"{side}_hand_"
    required = {index for chain in FINGER_CHAINS for index in chain}
    landmarks = []
    for index in sorted(required):
        landmark = by_name.get(f"{prefix}{index}")
        if landmark is None:
            return None
        if min(landmark.visibility, landmark.presence) < min_confidence:
            return None
        landmarks.append(landmark)

    points = {
        index: by_name[f"{prefix}{index}"].normalized_xyz
        for index in required
    }
    scores = []
    for proximal, joint, tip in FINGER_CHAINS:
        toward_proximal = tuple(
            points[proximal][axis] - points[joint][axis] for axis in range(3)
        )
        toward_tip = tuple(
            points[tip][axis] - points[joint][axis] for axis in range(3)
        )
        norm_a = sqrt(sum(value * value for value in toward_proximal))
        norm_b = sqrt(sum(value * value for value in toward_tip))
        if norm_a < 1e-6 or norm_b < 1e-6:
            return None
        cosine = sum(
            toward_proximal[axis] * toward_tip[axis] for axis in range(3)
        ) / (norm_a * norm_b)
        cosine = max(-1.0, min(1.0, cosine))
        # A straight finger has opposing vectors at the middle joint (cos=-1).
        # A fully folded finger points both vectors the same way (cos=+1).
        scores.append((1.0 - cosine) * 0.5)
    openness = sum(scores) / len(scores)
    return openness if isfinite(openness) else None


def openness_to_gripper_rad(openness: float) -> float:
    """Map continuous human openness to safe near-open/near-closed joint poses."""
    if not isfinite(openness):
        raise ValueError("hand openness must be finite")
    close_fraction = (OPEN_HAND_SCORE - openness) / (
        OPEN_HAND_SCORE - CLOSED_HAND_SCORE
    )
    close_fraction = max(0.0, min(1.0, close_fraction))
    return GRIPPER_OPEN_RAD + close_fraction * (
        GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD
    )
