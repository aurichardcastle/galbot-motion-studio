"""mink-backed strict-priority solver for one arm.

BORROWED (this file only): ``mink`` (Apache-2.0) provides the QP with hard
equality constraints, ``ConfigurationLimit`` and per-task Levenberg-Marquardt
damping. Transitively ``qpsolvers`` (LGPL-3.0) and ``daqp`` (MIT), both imported
unmodified. The pure swivel mathematics lives in :mod:`swivel_ik` and has no
third-party dependency, so ``version/only-mine`` reuses it unchanged.

The whole point of this file is the ``constraints=`` argument. Tasks passed there
are satisfied EXACTLY rather than in a least-squares sense, so the wrist pose is
a hard constraint and the swivel task can only ever act inside its null space.
That is strict priority by construction: no weight can make the elbow disturb the
wrist, because the elbow is confined to motions that provably do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from galbot_motion_studio.retarget.swivel_ik import (
    SwivelGeometry,
    swivel_error,
    swivel_geometry,
    swivel_jacobian,
)

try:  # pragma: no cover - exercised by the import guard test
    import mink
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "the swivel-priority mapping mode requires mink; install the 'swivel' extra"
    ) from error


#: Reference direction for psi, and the fallback used when the shoulder->wrist
#: axis is parallel to it. World -Z points into the floor, deliberately away from
#: the reachable workspace, so the coordinate singularity sits where the arm does
#: not operate.
_PSI_REFERENCE = np.array([0.0, 0.0, -1.0])
_PSI_FALLBACK = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class SwivelSolution:
    joints_rad: tuple[tuple[str, float], ...]
    wrist_residual_m: float
    swivel_residual_rad: float
    iterations: int
    converged: bool


class SwivelTask(mink.Task):
    """Drive the arm's swivel angle to a target. OURS, not borrowed.

    ``compute_error`` returns ``target - psi``; the Jacobian must therefore be
    ``-d(psi)/dq``. Returning ``+d(psi)/dq`` drives the solver AWAY from the
    target: it converges cleanly, reports a small residual, and lands on the
    wrong side. Measured on the abduction gesture, the wrong sign reached 26.83
    deg against a 148.38 deg target while still holding the wrist to 0.000000 m.
    A confident, converged, completely wrong answer -- hence this note.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        elbow_body_id: int,
        shoulder_body_id: int,
        tcp_body_id: int,
        *,
        cost: float = 1.0,
        gain: float = 1.0,
        lm_damping: float = 1e-3,
        reference: np.ndarray | None = None,
        fallback_reference: np.ndarray | None = None,
    ) -> None:
        super().__init__(cost=np.array([cost]), gain=gain, lm_damping=lm_damping)
        self._model = model
        self._elbow_body_id = elbow_body_id
        self._shoulder_body_id = shoulder_body_id
        self._tcp_body_id = tcp_body_id
        self._target_rad: float | None = None
        self._scratch = np.zeros((3, model.nv), dtype=np.float64)
        # psi is only meaningful relative to a reference direction. Callers that
        # compare the robot's psi against a HUMAN's must supply the robot-side
        # equivalent of the human's reference, or the two angles are measured
        # from different zeroes and only their changes are comparable.
        self._reference = _PSI_REFERENCE if reference is None else np.asarray(
            reference, dtype=np.float64
        )
        self._fallback_reference = (
            _PSI_FALLBACK if fallback_reference is None else np.asarray(
                fallback_reference, dtype=np.float64
            )
        )

    def set_target_rad(self, target_rad: float) -> None:
        self._target_rad = float(target_rad)

    def _geometry(self, configuration) -> SwivelGeometry | None:
        data = configuration.data
        return swivel_geometry(
            data.xpos[self._shoulder_body_id],
            data.xpos[self._elbow_body_id],
            data.xpos[self._tcp_body_id],
            reference=self._reference,
            fallback_reference=self._fallback_reference,
        )

    def compute_error(self, configuration) -> np.ndarray:
        geometry = self._geometry(configuration)
        if geometry is None or self._target_rad is None:
            return np.zeros(1)
        error = swivel_error(self._target_rad, geometry.angle_rad)
        return np.array([error * geometry.fade])

    def compute_jacobian(self, configuration) -> np.ndarray:
        geometry = self._geometry(configuration)
        if geometry is None or self._target_rad is None or geometry.is_degenerate:
            return np.zeros((1, self._model.nv))
        row = swivel_jacobian(
            self._model,
            configuration.data,
            self._elbow_body_id,
            geometry,
            scratch=self._scratch,
        )
        # error = target - psi  =>  d(error)/dq = -d(psi)/dq. See the class note.
        return -(row * geometry.fade).reshape(1, -1)


class SwivelArmSolver:
    """Hold the wrist pose exactly; place the elbow with the remaining freedom."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        joint_names: tuple[str, ...],
        tcp_body_id: int,
        elbow_body_id: int,
        shoulder_body_id: int,
        tcp_frame_name: str,
        solver: str = "daqp",
        #: Matches LeftArmPolicy.orientation_weight. Orientation has always been a
        #: weak preference on this arm, not a requirement.
        orientation_cost: float = 0.001,
        # Measured on the pinned model: a 60 deg swivel request converges in 16
        # iterations and a 45 deg one plateaus around 0.7 deg, so the budget is
        # about honest worst-case cost rather than reachability.
        max_iterations: int = 24,
        wrist_tolerance_m: float = 1e-4,
        # 1.15 deg. Tighter than this the solver plateaus without the answer
        # getting usefully better, and "converged" would stop meaning anything.
        swivel_tolerance_rad: float = 0.02,
        #: Kept in step with ``LeftArmPolicy.soft_margin_rad`` by the caller. Zero
        #: reproduces the pre-2026-08-22 behaviour of solving against the hard
        #: limits only.
        soft_margin_rad: float = 0.0,
        dt: float = 1.0,
        psi_reference: np.ndarray | None = None,
        psi_fallback_reference: np.ndarray | None = None,
    ) -> None:
        self._psi_reference = _PSI_REFERENCE if psi_reference is None else np.asarray(
            psi_reference, dtype=np.float64
        )
        self._psi_fallback_reference = (
            _PSI_FALLBACK if psi_fallback_reference is None else np.asarray(
                psi_fallback_reference, dtype=np.float64
            )
        )
        self._model = model
        self._joint_names = joint_names
        self._tcp_body_id = tcp_body_id
        self._elbow_body_id = elbow_body_id
        self._shoulder_body_id = shoulder_body_id
        self._solver = solver
        self._max_iterations = max_iterations
        self._wrist_tolerance_m = wrist_tolerance_m
        self._swivel_tolerance_rad = swivel_tolerance_rad
        self._dt = dt

        self._configuration = mink.Configuration(model)
        self._qpos_addrs = tuple(
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in joint_names
        )
        arm_dofs = {
            int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in joint_names
        }
        # Everything outside this arm is pinned exactly, so a single-arm solve can
        # never quietly move the head, the torso or the opposite arm.
        self._frozen = mink.DofFreezingTask(
            model=model,
            dof_indices=[index for index in range(model.nv) if index not in arm_dofs],
        )
        # POSITION is the hard constraint; ORIENTATION is a soft task carrying the
        # same weak weight the wrist-primary mapper has always used (0.001).
        #
        # The arm has 7 DOF. Constraining a full 6-DOF wrist pose would leave
        # exactly one free DOF, which the swivel task would then have to fight the
        # orientation for -- reintroducing the competition this design removes.
        # Constraining position alone leaves four, so the swivel gets real
        # authority and orientation takes what is left.
        self._wrist = mink.FrameTask(
            frame_name=tcp_frame_name,
            frame_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-4,
        )
        self._orientation = mink.FrameTask(
            frame_name=tcp_frame_name,
            frame_type="body",
            position_cost=0.0,
            orientation_cost=orientation_cost,
            lm_damping=1e-4,
        )
        self._swivel = SwivelTask(
            model,
            elbow_body_id,
            shoulder_body_id,
            tcp_body_id,
            lm_damping=1e-3,
            reference=self._psi_reference,
            fallback_reference=self._psi_fallback_reference,
        )
        # The caller rejects any solution whose joints land outside a soft band
        # `soft_margin_rad` inside the hard limits, so the solver has to respect
        # that band too -- otherwise it happily parks a joint 0.05 rad from the
        # hard stop, returns a perfectly good wrist, and has the whole solve
        # thrown away. Measured on the 2026-08-22 trial once the approach step
        # landed: `joint_limit` became the dominant rejection at 319 (left) and
        # 371 (right) per 250 frames, having been 233 and 14 before.
        #
        # The band is applied ONLY to this arm's joints, not through
        # `min_distance_from_limits`, which would apply it to every joint in the
        # model. That would be actively wrong here: the grippers are frozen DOFs
        # sitting 0.1 rad from their lower stop, so a global 0.11 rad band would
        # demand motion from a joint the frozen equality pins at zero, and every
        # step would be infeasible.
        configuration_limit = mink.ConfigurationLimit(model, gain=0.95)
        if soft_margin_rad > 0.0:
            for name in joint_names:
                jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                address = int(model.jnt_qposadr[jnt])
                lower, upper = model.jnt_range[jnt]
                if upper - lower <= 2 * soft_margin_rad:
                    continue
                configuration_limit.lower[address] = lower + soft_margin_rad
                configuration_limit.upper[address] = upper - soft_margin_rad
        self._limits = [configuration_limit]
        # Same frame, same target, but a heavily weighted TASK rather than an
        # equality. Used only when the exact-wrist equality is infeasible for a
        # step; see `_step_with_backoff`. The cost is large enough that the wrist
        # dominates the swivel and orientation by three orders of magnitude, so
        # the step still goes essentially straight at the wrist target.
        self._wrist_soft = mink.FrameTask(
            frame_name=tcp_frame_name,
            frame_type="body",
            position_cost=1000.0,
            orientation_cost=0.0,
            lm_damping=1e-4,
        )

    def solve(
        self,
        *,
        seed_qpos: np.ndarray,
        wrist_position_m: np.ndarray,
        wrist_rotation: np.ndarray,
        target_swivel_rad: float,
    ) -> SwivelSolution:
        self._configuration.update(np.asarray(seed_qpos, dtype=np.float64).copy())
        target_pose = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(np.asarray(wrist_rotation, dtype=np.float64)),
            np.asarray(wrist_position_m, dtype=np.float64),
        )
        self._wrist.set_target(target_pose)
        self._wrist_soft.set_target(target_pose)
        self._orientation.set_target(target_pose)
        self._swivel.set_target_rad(target_swivel_rad)

        iterations = 0
        converged = False
        # Hard equality constraints can render the QP infeasible -- measured: the
        # frozen-DOF equalities conflict with ConfigurationLimit's inequalities
        # when a non-arm DOF already sits at a limit, and daqp then reports no
        # solution at all.
        #
        # Retrying that step WITHOUT the joint limits looks tempting and is wrong.
        # Measured on a -30 deg swivel request: joints integrate past their range,
        # the configuration walks somewhere the velocity-level wrist equality no
        # longer holds at position level, and the solve returns a confident answer
        # with the wrist 0.1448 m off target. Silently trading the primary task for
        # a secondary one is precisely the failure this module exists to remove.
        #
        # The infeasibility is transient, not a reachability boundary: measured on
        # the pinned model, -10 deg and -40 deg swivel requests both converge while
        # -20 deg fails. So back off the SECONDARY demand for that step -- the
        # swivel gain -- leaving the wrist constraint and the joint limits fully
        # intact. If even a near-zero swivel ask is infeasible the step really is
        # blocked, and the solve ends with honest residuals for the caller to
        # reject.
        for iterations in range(1, self._max_iterations + 1):
            velocity = self._step_with_backoff()
            if velocity is None:
                break
            self._configuration.integrate_inplace(velocity, self._dt)
            wrist_residual, swivel_residual = self._residuals(
                wrist_position_m, target_swivel_rad
            )
            if (
                wrist_residual <= self._wrist_tolerance_m
                and abs(swivel_residual) <= self._swivel_tolerance_rad
            ):
                converged = True
                break

        wrist_residual, swivel_residual = self._residuals(wrist_position_m, target_swivel_rad)
        qpos = self._configuration.q
        return SwivelSolution(
            joints_rad=tuple(
                (name, float(qpos[address]))
                for name, address in zip(self._joint_names, self._qpos_addrs, strict=True)
            ),
            wrist_residual_m=wrist_residual,
            swivel_residual_rad=swivel_residual,
            iterations=iterations,
            converged=converged,
        )

    #: Successive scalings of the swivel gain tried when a step is infeasible.
    #: The wrist constraint and the joint limits are never relaxed.
    _BACKOFF = (1.0, 0.5, 0.2, 0.05)

    def _step_with_backoff(self) -> np.ndarray | None:
        """One QP step, easing off the secondary task until the step is feasible."""
        original_gain = self._swivel.gain
        try:
            # Census over 150 recorded frames: 6,116 calls, 286 (4.7%) feasible at
            # scale 1.0, 5,830 (95.3%) exhausted all four scales; _approach_step then
            # succeeded on 5,830/5,830.  Scales 0.5/0.2/0.05 produced a feasible step
            # ZERO times, at ~28.6 ms a frame.  So try full gain, then go straight to
            # the approach step -- the rungs are KEPT as a fallback after it, so no
            # capability is removed, only the order changes.
            self._swivel.gain = original_gain
            velocity = self._step()
            if velocity is not None:
                return velocity
            self._swivel.gain = original_gain * self._BACKOFF[-1]
            velocity = self._approach_step()
            if velocity is not None:
                return velocity
            for scale in self._BACKOFF[1:]:
                self._swivel.gain = original_gain * scale
                velocity = self._step()
                if velocity is not None:
                    return velocity
            return None
        finally:
            self._swivel.gain = original_gain

    def _approach_step(self) -> np.ndarray | None:
        """Best legal step TOWARD the wrist when landing on it exactly is infeasible.

        Measured on the 2026-08-22 trial: the exact-wrist equality and the
        frozen-DOF equality are each satisfiable inside the configuration limits,
        and both together are satisfiable without them -- but not both together
        *with* them. With everything below the arm held still, the seven arm
        joints alone cannot cover the whole wrist error in one limit-bounded step.

        Before this existed the solver treated that as "no answer" and broke out
        at iteration 1, which is why 97.6% of right-arm solves never moved at all
        and why no amount of extra iterations, damping or dt changed anything: the
        very first step was the one being refused. Walking toward the target over
        several steps is not available when step one is rejected outright.

        So for that step only, the wrist is demoted from an equality to a task
        weighted 1000x above the others. This does NOT relax anything that makes
        the answer safe: the joint limits stay hard, the frozen DOFs stay hard, so
        the pose cannot integrate out of range or quietly borrow motion from the
        torso, the head or the opposite arm. And the caller still rejects the
        whole solve unless the FINAL wrist residual is inside
        ``LeftArmPolicy.residual_tolerance_m`` -- a partial approach that never
        arrives is still refused, exactly as before.
        """
        return self._step(constraints=[self._frozen], tasks_override=True)

    def _step(
        self,
        constraints: list | None = None,
        *,
        tasks_override: bool = False,
    ) -> np.ndarray | None:
        """One QP step, or ``None`` when the solver reports no feasible solution.

        By default the wrist is a hard equality. ``tasks_override`` moves it into
        the weighted task set instead, for the approach step described in
        ``_approach_step``; the joint limits are unaffected either way.
        """
        tasks = (
            [self._wrist_soft, self._swivel, self._orientation]
            if tasks_override
            else [self._swivel, self._orientation]
        )
        try:
            return mink.solve_ik(
                self._configuration,
                tasks,
                self._dt,
                self._solver,
                damping=1e-12,
                limits=self._limits,
                constraints=[self._wrist, self._frozen]
                if constraints is None
                else constraints,
            )
        except mink.exceptions.NoSolutionFound:
            return None

    def _residuals(
        self, wrist_position_m: np.ndarray, target_swivel_rad: float
    ) -> tuple[float, float]:
        data = self._configuration.data
        wrist_residual = float(
            np.linalg.norm(np.asarray(wrist_position_m) - data.xpos[self._tcp_body_id])
        )
        geometry = swivel_geometry(
            data.xpos[self._shoulder_body_id],
            data.xpos[self._elbow_body_id],
            data.xpos[self._tcp_body_id],
            reference=self._psi_reference,
            fallback_reference=self._psi_fallback_reference,
        )
        if geometry is None:
            return wrist_residual, 0.0
        return wrist_residual, swivel_error(target_swivel_rad, geometry.angle_rad)
