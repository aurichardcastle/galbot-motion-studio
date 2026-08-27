"""Guard-rail-aware path conditioning for realtime joint targets.

The retargeter maps the operator's pose to a robot joint target every frame.
That raw target can sit past a joint's soft limit or drive the arm through a
self-collision.  The only thing between it and the robot today is
``JointTrajectoryGovernor``, which smooths velocity/acceleration but is blind to
limits and collision: it drives the straight joint-space line at the target, and
when that line crosses a guard rail the independent ``SafetySupervisor`` rejects
the whole frame and the limb latches into a frozen HOLD.  That is the "blindly
mirror the human, hit a wall, freeze" failure.

``GuardedPathController`` sits between the raw target and the governor.  It does
not replace any safety check -- the supervisor remains the final independent
authority and still vets every emitted pose.  Its job is to make the target
*feasible* so the supervisor allows it, and so the arm keeps flowing toward the
operator instead of stalling:

  1. **Limit clamp.**  Clamp each joint of the goal inside its soft-limit band so
     the governor never chases a target sitting on a limit.
  2. **Collision backoff.**  Move as far along the (clamped) straight line toward
     the goal as is collision-free *this* frame, and continue next frame --
     instead of driving into the wall and freezing.

The backoff is exact.  The injected clearance check sweeps the whole segment
*from the current pose*, so a longer step is a superset of a shorter one: once a
prefix of the line collides, every larger fraction does too.  The safe fractions
therefore form a prefix ``[0, alpha_max]`` and a bisection converges on the
boundary.  All joints are scaled by the same ``alpha``, so the arm moves *along
the straight line toward the operator's pose* -- the shape/direction of the
motion is preserved, only its extent is limited.

The controller emits a target over the exact same joint set it was handed (the
governed head+arm set); grippers are governed separately upstream and are passed
in only as ``context`` so the clearance query sees the true whole-robot pose.

This module carries no MuJoCo dependency: the clearance check is injected, which
also keeps it unit-testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable, Iterable, List, Mapping, Optional

# (start_pose, end_pose) -> True if the swept straight-line path between two full
# joint maps is collision-free.  The caller injects the real ClearanceChecker so
# the controller's notion of "safe" is identical to the supervisor's.
#: Either a plain "is this path clear?" predicate, or one that additionally
#: reports the smallest clearance it saw, in metres, so the controller can aim
#: for a standoff without paying for a second swept check. ``True``/``False`` are
#: read as "clear, distance unknown" / "not clear".
ClearancePathCheck = Callable[
    [Mapping[str, float], Mapping[str, float]], "bool | float"
]

#: Bisection steps for the proportional straight backoff. Four resolves the wall to
#: within one-sixteenth of a look-ahead step (~0.006 rad at a 0.10 rad look-ahead);
#: finer is wasted because the governor caps the actual per-frame motion.
_STRAIGHT_BISECTION_ITERS = 4

#: The per-joint detour is only worth its extra swept checks when the straight
#: backoff is essentially cornered. If the proportional step already advances any
#: joint by more than this (~0.6 deg), the arm is easing toward a wall, not trapped,
#: and the cheap bisection alone is the answer. With the repaired swivel solver,
#: retained replay shows the detour is often needed; disabling it severely reduces
#: arm travel, so it remains the reach-preserving path rather than a cheap fallback.
_DETOUR_TRIGGER_RAD = 0.01

# ``leg_joint4`` carries the complete upper body. It is a direct, separately
# gated tummy command; it is never a collision-detour degree of freedom for an
# arm. Letting alphabetical ordering try it before the arm joints can make a
# cornered wrist look as if the operator asked to swivel their chest. It remains
# part of the proportional straight-line move when the operator actually turns,
# and every such move is still swept; it is only excluded from the opportunistic
# one-joint detour phase.
_DETOUR_EXCLUDED_JOINTS = frozenset({"leg_joint4"})


@dataclass(frozen=True)
class GuardedStep:
    """The feasible per-frame target and why it differs from the raw goal."""

    target: dict[str, float]
    #: L2 fraction of the look-ahead step actually achieved this frame, in
    #: ``[0, 1]``: 1.0 when the whole step was clear, 0.0 when the arm is fully
    #: cornered and holds, in between when a detour advanced only the free joints.
    alpha: float
    #: Joints whose goal was clamped into the soft-limit band this frame.
    clamped_joints: tuple[str, ...]
    #: True when a self-collision held the step short of the full look-ahead --
    #: a per-joint detour, or (fully cornered) a hold at the current pose.
    backed_off: bool

    @property
    def limited(self) -> bool:
        """True when a guard rail changed the raw goal this frame."""
        return bool(self.clamped_joints) or self.backed_off


class GuardedPathController:
    def __init__(
        self,
        *,
        joint_limits: Mapping[str, tuple[float, float]],
        soft_margin_rad: float,
        clearance_check: ClearancePathCheck,
        max_lookahead_rad: Optional[float] = None,
        preferred_clearance_m: float = 0.0,
        protected_joints: Iterable[str] = (),
    ) -> None:
        if not isfinite(soft_margin_rad) or soft_margin_rad < 0:
            raise ValueError("soft_margin_rad must be finite and non-negative")
        if max_lookahead_rad is not None and not (
            isfinite(max_lookahead_rad) and max_lookahead_rad > 0
        ):
            raise ValueError("max_lookahead_rad must be finite and positive when set")
        self._limits = {
            name: (float(lo), float(hi)) for name, (lo, hi) in joint_limits.items()
        }
        self._margin = float(soft_margin_rad)
        self._clearance_check = clearance_check
        # Bounds how far along the ray to the goal a single frame's swept check
        # looks. The downstream governor caps per-frame motion to one small step
        # anyway, so checking the whole path to a distant goal is wasted cost (and
        # a big jump makes check_path refine into hundreds of samples). As long as
        # the look-ahead is >= one governor step it never limits real motion --
        # the governor's output is identical -- it only bounds the check cost.
        self._max_lookahead = (
            None if max_lookahead_rad is None else float(max_lookahead_rad)
        )
        # Clearance the BACKED-OFF target aims for, above the supervisor's hard
        # floor. Without it the bisection below converges on the boundary of
        # "clear", so every backed-off frame parks exactly ON the floor: measured
        # on the 2026-08-22 trial, 28.6% of allowed frames sat at 5.0000 mm and
        # p25 was 5.0000 mm, i.e. no margin at all between the commanded pose and
        # the value that would have been rejected.
        #
        # This is deliberately NOT applied to the free-space fast path, which is
        # already far from anything and would only lose tracking for nothing, and
        # it never overrides the floor: if no point on the ray achieves the
        # standoff, the controller falls back to the plain clearance predicate and
        # behaves exactly as it did before. It can therefore only add margin, and
        # never costs reach that was previously available.
        if not isfinite(preferred_clearance_m) or preferred_clearance_m < 0:
            raise ValueError("preferred_clearance_m must be finite and non-negative")
        self._preferred_clearance_m = float(preferred_clearance_m)
        # A protected joint stays at its explicit horizon while collision
        # backoff searches the other DOFs. If its own requested motion collides
        # even with every other joint held, alpha=0 is rejected and it holds
        # normally. This is a priority rule, not a clearance exemption.
        self._protected_joints = frozenset(str(name) for name in protected_joints)

    def _clamp_goal(
        self, goal: Mapping[str, float], clamped_out: List[str]
    ) -> dict[str, float]:
        clamped: dict[str, float] = {}
        for name, raw in goal.items():
            value = float(raw)
            limit = self._limits.get(name)
            if limit is None:
                clamped[name] = value
                continue
            lo, hi = limit
            low = lo + self._margin
            high = hi - self._margin
            if low > high:
                # Band narrower than twice the margin: no strictly-safe value
                # exists, so fall back to the raw midpoint rather than emit
                # low > high nonsense. Should not happen for the arm/head joints.
                clamped[name] = 0.5 * (lo + hi)
                clamped_out.append(name)
                continue
            bounded = min(high, max(low, value))
            clamped[name] = bounded
            if bounded != value:
                clamped_out.append(name)
        return clamped

    def solve(
        self,
        start: Mapping[str, float],
        goal: Mapping[str, float],
        *,
        context: Optional[Mapping[str, float]] = None,
    ) -> GuardedStep:
        """Return the furthest feasible point along ``start -> goal`` this frame.

        ``start`` is the pose the governor last emitted (already approved, so
        assumed clearance-safe); ``goal`` is the raw retargeted target.  Both must
        cover the exact same joint set.  ``context`` (e.g. the current gripper
        command) is folded into every clearance query so the swept check sees the
        whole-robot pose, but is not part of the emitted target.
        """
        if set(start) != set(goal):
            raise ValueError("start and goal must cover the same joint set")
        for name, value in start.items():
            if not isfinite(float(value)) or not isfinite(float(goal[name])):
                raise ValueError("start and goal joint positions must be finite")
        ctx = {} if context is None else {k: float(v) for k, v in context.items()}

        clamped_joints: List[str] = []
        clamped_goal = self._clamp_goal(goal, clamped_joints)
        clamped_tuple = tuple(sorted(set(clamped_joints)))

        # Per-joint look-ahead horizon: bound EACH joint's step on its own, not by
        # a single global scale. A global `lookahead / max-delta` scale would
        # couple unrelated limbs -- a held or fast-moving left arm would shrink the
        # right arm's commanded step -- and the pipeline guarantees a held limb
        # never disturbs a tracked one. Bounding each joint independently keeps the
        # common free-space frame decoupled while still capping how far the swept
        # check reaches (a distant goal would otherwise refine into a
        # frame-dropping sweep). The governor's own per-frame cap is the binding
        # constraint on real motion, so this never limits it.
        if self._max_lookahead is None:
            horizon = dict(clamped_goal)
        else:
            reach = self._max_lookahead
            horizon = {
                name: start[name]
                + max(-reach, min(reach, clamped_goal[name] - start[name]))
                for name in start
            }

        def candidate_at(alpha: float) -> dict[str, float]:
            return {
                name: (
                    horizon[name]
                    if name in self._protected_joints
                    else start[name] + alpha * (horizon[name] - start[name])
                )
                for name in start
            }

        def measure(candidate: Mapping[str, float]) -> float:
            """Clearance in metres for ``start -> candidate``; ``-inf`` if blocked.

            A predicate that only answers yes/no reports ``inf`` for a clear path,
            which makes every clear path satisfy any standoff -- the correct
            reading of "clear, distance unknown" for a caller that cannot measure.
            """
            verdict = self._clearance_check({**ctx, **start}, {**ctx, **candidate})
            if verdict is True:
                return float("inf")
            if verdict is False or verdict is None:
                return float("-inf")
            return float(verdict)

        def path_ok(candidate: Mapping[str, float]) -> bool:
            return measure(candidate) > float("-inf")

        def path_preferred(candidate: Mapping[str, float]) -> bool:
            return measure(candidate) >= self._preferred_clearance_m

        # Fast path: the whole look-ahead step is clear. The common case, one swept
        # check, and the emitted target is a per-joint function of the goal only --
        # unrelated limbs stay decoupled.
        if path_ok(horizon):
            return GuardedStep(
                target=dict(horizon),
                alpha=1.0,
                clamped_joints=clamped_tuple,
                backed_off=False,
            )

        # Step 1 -- proportional straight backoff. Bisect for the furthest point
        # along the straight line [start, horizon] that is collision-free. Because
        # the swept check from `start` makes the safe fractions a prefix
        # [0, alpha_max], bisection converges on it. This is deliberately the FIRST
        # move because it is momentum-friendly: it eases every joint proportionally
        # and shrinks the step as the arm nears a wall, so a joint moving fast
        # toward a collision is decelerated rather than frozen mid-flight -- the
        # downstream governor carries real velocity, and freezing a fast joint
        # would make it coast straight into the collision this avoids.
        def bisect(accept) -> tuple[dict[str, float], float]:
            lo, hi = 0.0, 1.0
            # ``alpha=0`` can still carry protected joints at their requested
            # horizon. Verify it; returning ``start`` remains the safe result
            # when the protected motion itself is the thing that blocks.
            found = candidate_at(0.0)
            if not accept(found):
                return dict(start), 0.0
            for _ in range(_STRAIGHT_BISECTION_ITERS):
                mid = 0.5 * (lo + hi)
                candidate = candidate_at(mid)
                if accept(candidate):
                    lo = mid
                    found = candidate
                else:
                    hi = mid
            return found, lo

        # Aim for the standoff first. Only if the ray cannot deliver it anywhere
        # does this fall back to the bare floor -- so the frames that lose reach
        # are exactly the frames that would otherwise have had zero margin, and
        # the second bisection is paid for only in that case.
        if self._preferred_clearance_m > 0.0:
            base, advanced = bisect(path_preferred)
            guard = path_preferred if advanced > 0.0 else path_ok
            if advanced <= 0.0:
                base, _ = bisect(path_ok)
        else:
            base, _ = bisect(path_ok)
            guard = path_ok

        # Step 2 -- per-joint detour. From that proportional base, advance the
        # joints that individually still make free progress toward the target,
        # holding back only the ones driving the collision. This is what a
        # straight-line-only backoff cannot do: when the target maps to a
        # self-colliding pose (a fully flexed arm folding its own forearm) the
        # straight step is ~0 and the arm would park where it first blocked and
        # never track the operator again. Advancing the free DOFs lets the arm
        # ride the collision boundary and, crucially, unfold and come back down
        # when the operator does. Arms are tried proximal-first (sorted names put
        # *_joint1 before *_joint7) because a fold is distal-driven, so holding the
        # distal joints while the proximal ones advance is what unfolds the reach.
        # The tummy joint is deliberately excluded: collision conditioning must
        # never turn a wrist blockage into an unrequested whole-upper-body turn.
        # A blocked joint keeps its momentum-friendly proportional value from
        # step 1, never a hard freeze. Every accepted move is re-checked against the
        # whole cumulative path from `start`, so the emitted pose is collision-free
        # by construction.
        #
        # This is a purely geometric target: it says nothing about how fast the
        # governor will realize it. The governor carries velocity and rate-limits
        # each joint on its own, so its realized pose lands off this straight
        # target and can dip below the clearance floor even though this path is
        # clear. That dynamics feasibility is enforced downstream, where the real
        # governor lives (see MotionStudioPipeline._feasible_governor_command),
        # which validates the governor's actual next pose and eases it back toward
        # the current pose if momentum would carry it into a wall. Keeping the two
        # concerns apart -- shape here, dynamics there -- is what lets this stay a
        # pure, MuJoCo-free, governor-free geometry solver.
        accepted = dict(base)
        base_advance = max(
            (abs(base[name] - start[name]) for name in start), default=0.0
        )
        if base_advance < _DETOUR_TRIGGER_RAD:
            for name in sorted(start):
                if name in _DETOUR_EXCLUDED_JOINTS:
                    continue
                if horizon[name] == accepted[name] or horizon[name] == start[name]:
                    continue
                trial = dict(accepted)
                trial[name] = horizon[name]
                # Same guard the straight backoff settled on, so the detour cannot
                # spend the standoff the bisection just bought.
                if guard(trial):
                    accepted = trial

        moved_sq = sum((accepted[name] - start[name]) ** 2 for name in start)
        span_sq = sum((horizon[name] - start[name]) ** 2 for name in start)
        alpha = sqrt(moved_sq / span_sq) if span_sq > 0.0 else 0.0
        return GuardedStep(
            target=accepted,
            alpha=alpha,
            clamped_joints=clamped_tuple,
            backed_off=True,
        )
