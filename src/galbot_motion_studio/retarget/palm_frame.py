"""A measured palm orientation, from the hand landmarks the detector already gives.

Why this exists: the arm's TCP orientation has never been driven by the operator.
It was *inferred* from the forearm direction and weighted at 0.001 against the
wrist position task -- effectively a tie-breaker, not a command. So the robot put
its hand in the right place pointing whichever way the solver preferred, which is
the single most visible way the mirroring looks wrong.

The detector emits 21 landmarks per hand. Three of them define the palm:

    0   wrist
    5   index finger MCP  (knuckle)
    17  pinky finger MCP  (knuckle)

Those three points are rigid with respect to each other -- the palm is the one
part of the hand that does not articulate -- so they give a stable frame even
while the fingers move:

    across   = index_mcp - pinky_mcp          (thumb-ward across the knuckles)
    forward  = mid(index_mcp, pinky_mcp) - wrist   (wrist toward the knuckles)
    normal   = across x forward               (out of the back of the hand)

Orthonormalised with Gram-Schmidt, keeping ``forward`` exact because it is the
best-conditioned of the three: it spans the length of the palm, whereas
``across`` spans its width and is the first to degenerate when the hand turns
edge-on to the camera.

Kinematically this is the missing half of the specification. Wrist position is
3 DOF and palm orientation is 3 more; on a 7-DOF arm that leaves exactly one
redundant DOF, which is the swivel angle :mod:`swivel_ik` resolves. Before this,
3 of the 7 were driven by nothing in particular.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

#: MediaPipe hand landmark indices for the three rigid palm points.
WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17

#: Below this the palm points are collinear or coincident and the frame is not
#: defined. Palm width across the knuckles is ~8 cm, so this is a real collapse
#: of the measurement rather than a small hand.
_DEGENERATE_M = 1e-3


@dataclass(frozen=True)
class PalmFrame:
    """Right-handed orthonormal frame fixed to the palm.

    Columns are ``across``, ``forward``, ``normal``; as a matrix it maps palm
    coordinates into whatever space the input points were expressed in.
    """

    across: tuple[float, float, float]
    forward: tuple[float, float, float]
    normal: tuple[float, float, float]
    #: Palm width in the input units. Small values mean the hand is edge-on and
    #: the frame is poorly conditioned even when it is not degenerate.
    width: float

    def as_matrix(self) -> np.ndarray:
        return np.column_stack(
            (
                np.asarray(self.across, dtype=np.float64),
                np.asarray(self.forward, dtype=np.float64),
                np.asarray(self.normal, dtype=np.float64),
            )
        )

    @property
    def confidence(self) -> float:
        """0..1 conditioning, ramping in over the first 4 cm of palm width."""
        return float(min(1.0, max(0.0, (self.width - _DEGENERATE_M) / 0.04)))


def palm_frame(
    wrist_xyz: np.ndarray | tuple[float, float, float],
    index_mcp_xyz: np.ndarray | tuple[float, float, float],
    pinky_mcp_xyz: np.ndarray | tuple[float, float, float],
    *,
    side: str,
) -> PalmFrame | None:
    """Palm frame from three landmarks, or ``None`` when it is not defined.

    ``side`` matters: the knuckle order runs index-to-pinky on one hand and
    pinky-to-index on the other, so without the flip the two palms would produce
    mirror-image frames and one hand would be commanded inside out.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    wrist = np.asarray(wrist_xyz, dtype=np.float64)
    index = np.asarray(index_mcp_xyz, dtype=np.float64)
    pinky = np.asarray(pinky_mcp_xyz, dtype=np.float64)
    if not all(np.all(np.isfinite(point)) for point in (wrist, index, pinky)):
        return None

    across = index - pinky
    if side == "right":
        across = -across
    width = float(np.linalg.norm(across))
    if not isfinite(width) or width < _DEGENERATE_M:
        return None

    forward = 0.5 * (index + pinky) - wrist
    length = float(np.linalg.norm(forward))
    if not isfinite(length) or length < _DEGENERATE_M:
        return None
    forward = forward / length

    # Keep `forward` exact and project `across` off it: the palm is longer than
    # it is wide, so `forward` is the better-conditioned of the two.
    across = across - float(np.dot(across, forward)) * forward
    residual = float(np.linalg.norm(across))
    if not isfinite(residual) or residual < _DEGENERATE_M:
        return None
    across = across / residual

    normal = np.cross(across, forward)
    normal_norm = float(np.linalg.norm(normal))
    if not isfinite(normal_norm) or normal_norm < _DEGENERATE_M:
        return None
    normal = normal / normal_norm

    return PalmFrame(
        across=tuple(float(v) for v in across),
        forward=tuple(float(v) for v in forward),
        normal=tuple(float(v) for v in normal),
        width=width,
    )


def palm_frame_from_landmarks(
    landmarks: dict[str, object],
    side: str,
    *,
    coordinate_space: str = "world_xyz_m",
) -> PalmFrame | None:
    """Palm frame from a ``{name: landmark}`` map, or ``None`` if unavailable.

    Reads only the three rigid palm points, so a partly occluded hand with
    unreliable fingertips still yields an orientation.
    """
    points = []
    for index in (WRIST, INDEX_MCP, PINKY_MCP):
        landmark = landmarks.get(f"{side}_hand_{index}")
        if landmark is None:
            return None
        value = getattr(landmark, coordinate_space, None)
        if value is None:
            value = getattr(landmark, "normalized_xyz", None)
        if value is None:
            return None
        points.append(np.asarray(value, dtype=np.float64))
    return palm_frame(points[0], points[1], points[2], side=side)
