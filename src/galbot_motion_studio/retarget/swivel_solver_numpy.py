"""Strict-priority arm IK with no third-party solver. The own-work version.

Same architecture as :mod:`swivel_solver_mink` -- the wrist pose is exact and the
elbow swivels beneath it -- but the quadratic program is replaced by a KKT system
solved directly with numpy, so this file adds no dependency at all.

The equality-constrained least squares being solved each step is

    minimise   1/2 ||J2 dq - v2||^2 + 1/2 lambda ||dq||^2
    subject to J1 dq = v1

whose stationarity conditions are the KKT system

    [ J2^T J2 + lambda I   J1^T ] [ dq ]   [ J2^T v2 ]
    [ J1                   0    ] [ nu ] = [ v1      ]

J1 is the wrist position Jacobian (3x7) and J2 the scalar swivel row (1x7), so
this is one dense 10x10 solve. The constraint row is what makes the priority
strict: any dq the solver returns satisfies the wrist task exactly, so the swivel
term provably cannot trade wrist accuracy for elbow position at any gain. That is
the property a weighted stack cannot have, and the reason every previous attempt
on this arm failed -- with weights, the secondary's disturbance of the primary
only goes to zero as its weight does, at which point it stops doing anything.

Mathematically identical to the null-space form
``dq = J1^+ v1 + N1 (J2 N1)^+ (v2 - J2 J1^+ v1)`` with ``N1 = I - J1^+ J1``, but
better conditioned and without forming a projector.

Damping note, easy to get wrong: the damped inverse is used for the STEP, never
to build a projector. ``I - J_damped^+ J`` is not idempotent, so a projector made
from it leaks primary-task error into the secondary. Here the constraint row
carries the priority, so no projector is formed at all and the question does not
arise.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from galbot_motion_studio.retarget.swivel_ik import (
    swivel_error,
    swivel_geometry,
    swivel_jacobian,
)

_PSI_REFERENCE = np.array([0.0, 0.0, -1.0])
_PSI_FALLBACK = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class SwivelSolution:
    joints_rad: tuple[tuple[str, float], ...]
    wrist_residual_m: float
    swivel_residual_rad: float
    iterations: int
    converged: bool


class SwivelArmSolverNumpy:
    """Hold the wrist exactly; place the elbow with what is left over."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        joint_names: tuple[str, ...],
        tcp_body_id: int,
        elbow_body_id: int,
        shoulder_body_id: int,
        max_iterations: int = 24,
        wrist_tolerance_m: float = 1e-4,
        swivel_tolerance_rad: float = 0.02,
        wrist_gain: float = 0.8,
        swivel_gain: float = 0.5,
        damping: float = 1e-4,
        max_step_rad: float = 0.15,
        soft_margin_rad: float = 0.11,
    ) -> None:
        self._model = model
        self._joint_names = joint_names
        self._tcp = tcp_body_id
        self._elbow = elbow_body_id
        self._shoulder = shoulder_body_id
        self._max_iterations = max_iterations
        self._wrist_tolerance_m = wrist_tolerance_m
        self._swivel_tolerance_rad = swivel_tolerance_rad
        self._wrist_gain = wrist_gain
        self._swivel_gain = swivel_gain
        self._damping = damping
        self._max_step_rad = max_step_rad

        self._data = mujoco.MjData(model)
        self._qadr = tuple(
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in joint_names
        )
        self._dof = tuple(
            int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in joint_names
        )
        limits = []
        for name in joint_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            low, high = model.jnt_range[joint]
            if not bool(model.jnt_limited[joint]):
                low, high = -np.pi, np.pi
            limits.append((float(low) + soft_margin_rad, float(high) - soft_margin_rad))
        self._limits = np.asarray(limits, dtype=np.float64)
        self._jac_wrist = np.zeros((3, model.nv), dtype=np.float64)
        self._jac_scratch = np.zeros((3, model.nv), dtype=np.float64)

    def _kinematics(self, qpos: np.ndarray) -> None:
        self._data.qpos[:] = qpos
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)

    def _geometry(self):
        data = self._data
        return swivel_geometry(
            data.xpos[self._shoulder],
            data.xpos[self._elbow],
            data.xpos[self._tcp],
            reference=_PSI_REFERENCE,
            fallback_reference=_PSI_FALLBACK,
        )

    def solve(
        self,
        *,
        seed_qpos: np.ndarray,
        wrist_position_m: np.ndarray,
        target_swivel_rad: float,
    ) -> SwivelSolution:
        qpos = np.asarray(seed_qpos, dtype=np.float64).copy()
        target = np.asarray(wrist_position_m, dtype=np.float64)
        arm = np.asarray(self._dof, dtype=int)
        identity = np.eye(len(self._joint_names))

        iterations = 0
        converged = False
        for iterations in range(1, self._max_iterations + 1):
            self._kinematics(qpos)
            wrist_error = target - self._data.xpos[self._tcp]
            geometry = self._geometry()
            swivel_residual = (
                0.0 if geometry is None
                else swivel_error(target_swivel_rad, geometry.angle_rad)
            )
            if (
                float(np.linalg.norm(wrist_error)) <= self._wrist_tolerance_m
                and abs(swivel_residual) <= self._swivel_tolerance_rad
            ):
                converged = True
                break

            self._jac_wrist.fill(0.0)
            mujoco.mj_jacBody(self._model, self._data, self._jac_wrist, None, self._tcp)
            primary = self._jac_wrist[:, arm]

            if geometry is None or geometry.is_degenerate:
                secondary = np.zeros((1, len(arm)))
                secondary_target = np.zeros(1)
            else:
                row = swivel_jacobian(
                    self._model, self._data, self._elbow, geometry,
                    scratch=self._jac_scratch,
                )
                # Fade the term out as the arm approaches full extension, where
                # psi carries no information and the Jacobian diverges as 1/r.
                secondary = (row[arm] * geometry.fade).reshape(1, -1)
                secondary_target = np.array([self._swivel_gain * swivel_residual * geometry.fade])

            # [ J2^T J2 + lambda I   J1^T ] [ dq ]   [ J2^T v2 ]
            # [ J1                   0    ] [ nu ] = [ v1      ]
            size = len(arm)
            kkt = np.zeros((size + 3, size + 3))
            kkt[:size, :size] = secondary.T @ secondary + self._damping * identity
            kkt[:size, size:] = primary.T
            kkt[size:, :size] = primary
            rhs = np.concatenate(
                (secondary.T @ secondary_target, self._wrist_gain * wrist_error)
            )
            try:
                step = np.linalg.solve(kkt, rhs)[:size]
            except np.linalg.LinAlgError:
                # Singular at this configuration: fall back to a damped primary-
                # only step rather than returning a bogus direction.
                step = np.linalg.lstsq(
                    primary, self._wrist_gain * wrist_error, rcond=None
                )[0]

            step = np.clip(step, -self._max_step_rad, self._max_step_rad)
            current = qpos[list(self._qadr)]
            qpos[list(self._qadr)] = np.clip(
                current + step, self._limits[:, 0], self._limits[:, 1]
            )

        self._kinematics(qpos)
        geometry = self._geometry()
        wrist_residual = float(np.linalg.norm(target - self._data.xpos[self._tcp]))
        swivel_residual = (
            0.0 if geometry is None
            else swivel_error(target_swivel_rad, geometry.angle_rad)
        )
        return SwivelSolution(
            joints_rad=tuple(
                (name, float(qpos[address]))
                for name, address in zip(self._joint_names, self._qadr, strict=True)
            ),
            wrist_residual_m=wrist_residual,
            swivel_residual_rad=swivel_residual,
            iterations=iterations,
            converged=converged,
        )
