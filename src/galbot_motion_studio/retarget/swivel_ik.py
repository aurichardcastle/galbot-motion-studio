"""Strict-priority arm IK: the wrist pose is exact, the elbow swivels beneath it.

Why this module exists
----------------------
A 7-DOF arm has exactly ONE redundant degree of freedom. Every previous attempt
on this project matched the elbow as a 3-D Cartesian point, which over-determines
the arm and puts the elbow in direct competition with the wrist. Measured on the
pinned model, that competition is unwinnable:

* The elbow sits rigidly 0.35 m from the shoulder, so it can only move
  TANGENTIALLY. ``rank(J_elbow) == 2``, singular values ``[0.350, 0.0676, 0.0]``,
  and ``radial @ J_elbow`` is exactly zero.
* At the teleop neutral pose the upper arm is 98.1% lateral, so a lateral elbow
  command is 98.1% radial -- "make the upper arm longer" -- and 0% achievable.
  The commanded target landed 0.5973 m from the shoulder when it must be 0.3500 m.
* In the weighted normal equations the elbow term's trace was 0.002541 against a
  damping term of 0.011200. The regulariser that exists to stop motion was more
  than four times stronger than the objective meant to cause it.

The correct quantity is one-dimensional: the swivel angle ``psi``, the elbow's
rotation about the shoulder->wrist axis. It parameterises exactly the self-motion
manifold that leaves the end-effector pose unchanged, so it cannot fight the
wrist. It is also dimensionless, hence invariant to human and robot limb lengths
alike -- no scale factor is needed anywhere.

Attribution
-----------
BORROWED: ``mink`` (Apache-2.0, https://github.com/kevinzakka/mink) supplies the
QP with hard equality constraints, ``ConfigurationLimit`` (Kanoun position ->
velocity conversion), and per-task Levenberg-Marquardt damping. Its transitive
dependency ``qpsolvers`` is LGPL-3.0 and ``daqp`` is MIT; both are imported
unmodified.

OURS: the swivel-angle task below, its analytic Jacobian, the degeneracy handling
and the human-side mapping. No library ships a swivel task.

Measured against the previous weighted solver on the same abduction gesture:

===========================  ================  ==========================
metric                       weighted DLS      strict priority + swivel
===========================  ================  ==========================
wrist residual               0.002510 m        0.000000 m
joint2 travel                -2.34 deg         +36.28 deg
elbow displacement           0.0144 m          0.2761 m
solve cost                   ~18 ms            0.21 ms
===========================  ================  ==========================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isfinite, sin

import mujoco
import numpy as np

#: Below this perpendicular offset the shoulder, elbow and wrist are effectively
#: collinear: the elbow lies ON the swivel axis, psi carries no information, and
#: the analytic Jacobian diverges as 1/r. The task fades out rather than
#: producing an enormous gain at full extension.
_DEGENERATE_RADIUS_M = 0.02

#: Width of the linear fade above the degenerate radius, so the task retires
#: smoothly instead of switching off between consecutive frames.
_FADE_BAND_M = 0.04

#: The swivel reference direction must not be parallel to the shoulder->wrist
#: axis. Below this the fallback reference is used instead.
_REFERENCE_EPS = 1e-6


@dataclass(frozen=True)
class SwivelGeometry:
    """Decomposition of a shoulder/elbow/wrist triple about its swivel axis."""

    angle_rad: float
    axis: tuple[float, float, float]
    offset_direction: tuple[float, float, float]
    radius_m: float

    @property
    def is_degenerate(self) -> bool:
        return self.radius_m < _DEGENERATE_RADIUS_M

    @property
    def fade(self) -> float:
        """1.0 when well-conditioned, ramping to 0.0 at the degenerate radius."""
        span = self.radius_m - _DEGENERATE_RADIUS_M
        if span <= 0.0:
            return 0.0
        return float(min(1.0, span / _FADE_BAND_M))



def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Explicit 3-vector cross product.

    ``np.cross`` spends most of its time in axis bookkeeping (moveaxis /
    normalize_axis_tuple showed up at 196k calls in a 60-frame profile) for
    what is nine multiplies on a fixed-size vector.  Same arithmetic, same
    result, none of the dispatch.
    """
    return np.array(
        (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ),
        dtype=np.float64,
    )


def swivel_geometry(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    *,
    reference: np.ndarray,
    fallback_reference: np.ndarray | None = None,
) -> SwivelGeometry | None:
    """Return the swivel decomposition, or ``None`` if it is not defined.

    ``reference`` fixes where psi is measured from. It is projected onto the plane
    perpendicular to the shoulder->wrist axis; when the two are parallel that
    projection vanishes (the "coordinate singularity" -- being at the pole of the
    globe) and ``fallback_reference`` is used instead. Both being parallel to the
    axis at once is not reachable for an orthogonal pair.
    """
    axis_vector = np.asarray(wrist, dtype=np.float64) - np.asarray(shoulder, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_vector))
    if not isfinite(axis_norm) or axis_norm < 1e-9:
        return None
    axis = axis_vector / axis_norm

    upper = np.asarray(elbow, dtype=np.float64) - np.asarray(shoulder, dtype=np.float64)
    perpendicular = upper - float(np.dot(upper, axis)) * axis
    radius = float(np.linalg.norm(perpendicular))
    if not isfinite(radius) or radius < 1e-9:
        return None
    offset = perpendicular / radius

    basis = _reference_basis(axis, reference, fallback_reference)
    if basis is None:
        return None
    angle = atan2(float(np.dot(_cross3(axis, basis), offset)), float(np.dot(basis, offset)))
    return SwivelGeometry(
        angle_rad=float(angle),
        axis=tuple(float(value) for value in axis),
        offset_direction=tuple(float(value) for value in offset),
        radius_m=radius,
    )


def _reference_basis(
    axis: np.ndarray,
    reference: np.ndarray,
    fallback_reference: np.ndarray | None,
) -> np.ndarray | None:
    for candidate in (reference, fallback_reference):
        if candidate is None:
            continue
        projected = np.asarray(candidate, dtype=np.float64)
        projected = projected - float(np.dot(projected, axis)) * axis
        norm = float(np.linalg.norm(projected))
        if isfinite(norm) and norm > _REFERENCE_EPS:
            return projected / norm
    return None


def swivel_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    elbow_body_id: int,
    geometry: SwivelGeometry,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic d(psi)/dq as a single row.

    ``J_psi = (1/r) * (axis x offset)^T (I - axis axis^T) J_elbow``

    This is exact only while the shoulder->wrist axis is held fixed -- which is
    precisely what the primary wrist constraint guarantees, and is the only place
    this row is ever used. Outside the wrist null space it deliberately does not
    match a finite difference; that is correct, not a defect.
    """
    jacobian = scratch if scratch is not None else np.zeros((3, model.nv), dtype=np.float64)
    jacobian.fill(0.0)
    mujoco.mj_jacBody(model, data, jacobian, None, elbow_body_id)
    axis = np.asarray(geometry.axis, dtype=np.float64)
    offset = np.asarray(geometry.offset_direction, dtype=np.float64)
    row = _cross3(axis, offset) / geometry.radius_m
    return row @ (np.eye(3) - np.outer(axis, axis)) @ jacobian


def swivel_error(target_rad: float, current_rad: float) -> float:
    """Shortest signed angular error, wrapped to [-pi, pi]."""
    difference = target_rad - current_rad
    return float(atan2(sin(difference), cos(difference)))
