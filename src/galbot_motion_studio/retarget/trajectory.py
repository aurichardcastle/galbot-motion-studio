"""Stateful velocity/acceleration conditioning for realtime joint targets."""

from __future__ import annotations

from dataclasses import dataclass
from math import copysign, isfinite, sqrt
from typing import Mapping


def feasible_velocity_limit(
    *, max_acceleration_rad_s2: float, max_position_step_rad: float
) -> float:
    """Fastest speed a step-capped governor can hold and still stop legally.

    ``JointTrajectoryGovernor`` applies its position-discontinuity cap *after* the
    acceleration clamp, and rewrites velocity to ``step / dt`` when that cap binds.
    The supervisor reconstructs endpoint acceleration the same way, so a speed
    ``v`` is dynamically feasible only if, for every source interval ``dt``::

        v - step/dt <= a*dt      i.e.      v <= a*dt + step/dt

    The right-hand side is minimised at ``dt = sqrt(step/a)``, where it equals
    ``2*sqrt(a*step)``.  Declared at or below that speed, a governor can never be
    forced into a deceleration its own limiter did not authorize, at any frame
    rate.  Declared above it, there is a window of ``dt`` -- always a *sparse*
    interval, so it appears exactly when the source stutters -- in which the
    supervisor must reject the command.  This is not a tuning knob: the three
    limits are mutually constrained, and this is the constraint.

    The inward margin keeps a declared limit off the exact knife-edge.  At
    equality the reconstructed acceleration *equals* ``a``, and the supervisor
    compares with only a 1e-12 absolute tolerance, so float rounding alone could
    tip an otherwise legal command into a HOLD.
    """
    return 2.0 * sqrt(max_acceleration_rad_s2 * max_position_step_rad) * (1.0 - 1e-6)


@dataclass(frozen=True)
class TrajectoryLimits:
    max_velocity_rad_s: float
    max_acceleration_rad_s2: float
    max_position_step_rad: float | None = None

    def __post_init__(self) -> None:
        if not (
            isfinite(self.max_velocity_rad_s)
            and self.max_velocity_rad_s > 0
            and isfinite(self.max_acceleration_rad_s2)
            and self.max_acceleration_rad_s2 > 0
        ):
            raise ValueError("trajectory limits must be finite and positive")
        if self.max_position_step_rad is None:
            return
        if not (
            isfinite(self.max_position_step_rad) and self.max_position_step_rad > 0
        ):
            raise ValueError("max position step must be finite and positive when set")
        # Refuse an internally inconsistent triple at construction rather than
        # discovering it as a supervisor HOLD on whichever sparse frame happens to
        # land in the infeasible window.  Callers that want the fastest legal
        # governor for a profile should derive it with `feasible_velocity_limit`
        # instead of hand-picking a speed.
        feasible = feasible_velocity_limit(
            max_acceleration_rad_s2=self.max_acceleration_rad_s2,
            max_position_step_rad=self.max_position_step_rad,
        )
        if self.max_velocity_rad_s > feasible:
            worst_dt = sqrt(self.max_position_step_rad / self.max_acceleration_rad_s2)
            raise ValueError(
                f"max_velocity_rad_s={self.max_velocity_rad_s} is dynamically "
                f"infeasible with max_acceleration_rad_s2="
                f"{self.max_acceleration_rad_s2} and max_position_step_rad="
                f"{self.max_position_step_rad}: when the step cap binds near "
                f"dt={worst_dt:.4g}s it forces a deceleration above the "
                f"acceleration limit. Cap velocity at {feasible} "
                f"(2*sqrt(a*step)) or widen the other two."
            )


class JointTrajectoryGovernor:
    """Follow desired joints without exceeding velocity or acceleration limits.

    The desired pose may jump because of vision noise or IK branch changes.  This
    class turns it into a physically continuous stream.  It does not authorize the
    stream; the independent safety supervisor still checks every emitted target.
    """

    def __init__(
        self,
        limits: TrajectoryLimits,
        *,
        max_velocity_rad_s_by_joint: Mapping[str, float] | None = None,
    ) -> None:
        """Create a governor with optional stricter per-joint velocity caps.

        The shared profile remains the outer physical envelope.  A mapping may
        impose a *stricter* measured/validated rate for one joint (the torso is
        the current case), but it can never raise a joint above the profile's
        global cap.  Keeping this at the governor makes the cap apply to the
        emitted trajectory rather than merely to a raw vision target.
        """
        self.limits = limits
        overrides = dict(max_velocity_rad_s_by_joint or {})
        for name, value in overrides.items():
            if not (
                isinstance(name, str)
                and name
                and isfinite(value)
                and value > 0
                and value <= limits.max_velocity_rad_s
            ):
                raise ValueError(
                    "per-joint velocity overrides must be named, finite, positive, "
                    "and no greater than the profile maximum"
                )
        self._max_velocity_rad_s_by_joint = overrides
        self._positions: dict[str, float] | None = None
        self._velocities: dict[str, float] = {}
        self._timestamp_ns: int | None = None

    def reset(self, positions: Mapping[str, float], *, timestamp_ns: int) -> None:
        if timestamp_ns < 0 or not positions:
            raise ValueError("trajectory reset needs positions and a non-negative timestamp")
        if any(not isfinite(float(value)) for value in positions.values()):
            raise ValueError("trajectory reset positions must be finite")
        self._positions = {name: float(value) for name, value in positions.items()}
        self._velocities = {name: 0.0 for name in positions}
        self._timestamp_ns = timestamp_ns

    def _elapsed_s(self, desired: Mapping[str, float], timestamp_ns: int) -> float:
        if self._positions is None or self._timestamp_ns is None:
            raise RuntimeError("reset the trajectory governor before stepping")
        if set(desired) != set(self._positions):
            raise ValueError("desired joints must exactly match the governed joint set")
        if timestamp_ns <= self._timestamp_ns:
            raise ValueError("trajectory timestamps must strictly increase")
        if any(not isfinite(float(value)) for value in desired.values()):
            raise ValueError("desired joint positions must be finite")
        return (timestamp_ns - self._timestamp_ns) / 1_000_000_000

    def _advance(
        self, desired: Mapping[str, float], elapsed_s: float
    ) -> tuple[dict[str, float], dict[str, float]]:
        """One rate/accel-limited integration step. Pure: reads state, writes none.

        ``step`` commits the result and ``preview`` discards it, so a previewed pose
        is byte-identical to the one ``step`` would emit for the same arguments.
        """
        assert self._positions is not None  # guaranteed by _elapsed_s
        next_positions: dict[str, float] = {}
        next_velocities: dict[str, float] = {}
        # Stay infinitesimally inside the declared envelope so reconstructing
        # acceleration from successive float positions cannot round above it.
        acceleration_step = self.limits.max_acceleration_rad_s2 * elapsed_s * (1.0 - 1e-9)

        for name, position in self._positions.items():
            max_velocity = self._max_velocity_rad_s_by_joint.get(
                name, self.limits.max_velocity_rad_s
            )
            error = float(desired[name]) - position
            previous_velocity = self._velocities[name]
            if error == 0.0:
                desired_velocity = 0.0
            else:
                # Discrete-time stopping bound. The continuous sqrt(2*a*x)
                # formula ignores the distance travelled during this sample and
                # can overshoot a target/joint margin by ~20 mrad at 30 Hz.
                acceleration = self.limits.max_acceleration_rad_s2
                stopping_speed = max(
                    0.0,
                    sqrt(
                        (acceleration * elapsed_s) ** 2
                        + 2.0 * acceleration * abs(error)
                    )
                    - acceleration * elapsed_s,
                )
                desired_velocity = copysign(
                    min(max_velocity, stopping_speed), error
                )
            velocity_delta = max(
                -acceleration_step,
                min(acceleration_step, desired_velocity - previous_velocity),
            )
            velocity = previous_velocity + velocity_delta
            velocity = max(
                -max_velocity,
                min(max_velocity, velocity),
            )
            position_step = velocity * elapsed_s
            if self.limits.max_position_step_rad is not None:
                position_step = max(
                    -self.limits.max_position_step_rad,
                    min(self.limits.max_position_step_rad, position_step),
                )
                velocity = position_step / elapsed_s
            next_positions[name] = position + position_step
            next_velocities[name] = velocity

        return next_positions, next_velocities

    def step(self, desired: Mapping[str, float], *, timestamp_ns: int) -> dict[str, float]:
        elapsed_s = self._elapsed_s(desired, timestamp_ns)
        next_positions, next_velocities = self._advance(desired, elapsed_s)
        self._positions = next_positions
        self._velocities = next_velocities
        self._timestamp_ns = timestamp_ns
        return dict(next_positions)

    def preview(
        self, desired: Mapping[str, float], *, timestamp_ns: int
    ) -> dict[str, float]:
        """Return the positions ``step`` would emit, without mutating state.

        The governor rate-limits each joint on its own, so the pose it realizes for
        a given target lands *off* the straight ``[current, desired]`` segment. A
        caller that must validate the realized pose (e.g. against a clearance floor)
        before committing to it can preview it here and only then ``step``.
        """
        elapsed_s = self._elapsed_s(desired, timestamp_ns)
        next_positions, _ = self._advance(desired, elapsed_s)
        return next_positions

    @property
    def velocities(self) -> dict[str, float]:
        return dict(self._velocities)
