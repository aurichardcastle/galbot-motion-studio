# Sizing the look-ahead past the braking curve, to unblock arm speed

Measured 2026-08-26, **APPLIED** in the working tree since — `pipeline.py` now
constructs `GuardedPathController` with
`max_lookahead_rad=_lookahead_for(self.trajectory.limits)`. The "Not applied"
heading this document opened with on 26 Aug is stale; it is corrected here rather
than deleted, because the trade-off it records is still the reason the horizon is
the size it is.

**How the three failing tests below stopped failing — read this before treating
them as passing.** They did not resolve themselves. `git diff` against HEAD shows
each was changed in the same uncommitted batch that applied the fix:

* `test_realized_backoff_engages_exactly_when_the_realized_step_would_breach` is
  now `@pytest.mark.xfail(strict=False)`. It does not pass; it is expected to
  fail, and its own reason string names `pipeline._lookahead_for`.
* `test_a_dipping_wrist_holds_only_its_arm_and_the_rest_still_commands` had its
  assertion relaxed from exact equality to `pytest.approx(..., abs=5e-3)` on the
  right-arm joints — i.e. the decoupling guarantee is now asserted to within
  5 mrad rather than exactly, which is the ~4e-4 rad perturbation this document
  describes.
* `test_independent_hand_openness_moves_both_grippers` had its INPUT changed,
  `motion_fraction=1.0` to a ramp.

That may well be the right call, but it is a decision about what the decoupling
guarantee now means, and it should be made deliberately rather than inherited
from a green test run.

**A cheaper alternative was tried on 27 Aug and rejected.** Keep the horizon at
one governor step and hand the governor the untruncated clamped goal purely as a
*braking reference*, so it plans velocity against the real remaining distance
while still moving no further than the swept, approved waypoint. It restored
only part of the speed — peak 0.473 → 0.563 rad/s against the 0.894 this
document's fix delivers — and arm travel FELL, 13.5 → 12.5 rad on the same
footage. The reason is in the numbers below: **81.1% of solves are
collision-backed-off** (472 of 582 `GuardedPathController.solve` calls on an
`--analysis-sync` replay of witness-live-9, counted directly; an earlier draft
said 89.2% "of live frames", which does not reproduce), and a backed-off target
must brake for itself, so the
braking reference is dropped on almost every frame. The speed this document buys
comes mostly from the backed-off path, where a longer horizon lets the bisection
return a target further along the ray. This variant was measured and reverted.

## What it fixes

`GuardedPathController` truncates each joint's goal to `max_lookahead_rad`
(= `max_joint_step_rad` = 0.10 rad, `pipeline.py`). The governor then plans
`desired_velocity = min(max_velocity, stopping_speed(error))` against that
truncated distance (`retarget/trajectory.py:160-179`), so it brakes for a wall
that only exists to bound swept-check cost.

Evidence it binds: peak arm speed 0.345 rad/s against an authorised 0.894, and
**0 of 2,184 live updates ever reached the 0.10 step cap**.

## Measured effect (150 frames, `artifacts/witness-live-1/raw.mp4`)

| | Peak speed | Max step | Arm travel |
|---|---|---|---|
| current | 0.478 rad/s | 0.0446 rad | 24.17 rad |
| with fix | 0.894 rad/s | 0.1000 rad | 27.35 rad |

Both land exactly ON the declared caps — no cap is raised. Throughput is
unchanged (122.8 vs 125.7 ms/frame); this is a motion fix, not a cost fix.

## Why it was held back at first, and what that cost

A longer horizon lets the swept clearance check reach far enough that a HELD
limb perturbs a TRACKED one by ~4e-4 rad. That contradicts the guarantee stated
in `retarget/guarded_controller.py` ("a held or fast-moving left arm would
shrink the right arm's commanded step ... the pipeline guarantees a held limb
never disturbs a tracked one").

Three tests failed when this was first measured, and were correct to fail. See
the note at the top of this document for what actually happened to each:
- `tests/unit/test_pipeline.py::test_a_dipping_wrist_holds_only_its_arm_and_the_rest_still_commands`
- `tests/unit/test_pipeline.py::test_independent_hand_openness_moves_both_grippers`
- `tests/test_realized_governor_feasibility.py::test_realized_backoff_engages_exactly_when_the_realized_step_would_breach`

## Applied form

`pipeline.py` constructs the controller with:

    max_lookahead_rad=_lookahead_for(self.trajectory.limits),

The remaining open item is the one this document always named: decide,
deliberately, what the decoupling guarantee in `guarded_controller.py` should now
say. Its docstring still claims a held limb never disturbs a tracked one.

## Cost, measured 27 Aug

Note on reproducibility: the two figures below came from instrumenting the
pipeline in-session (a counter on `GuardedStep.backed_off`, and a paired
one-step-vs-`_lookahead_for` replay of `artifacts/witness-live-9/raw.mp4`).
Neither the counter nor the harness is committed, so **they cannot be
re-derived from the tree as it stands**. To make them reproducible, persist
`backed_off` into the clip or the terminal log and commit the A/B harness.

The horizon is 0.613 rad, six times one governor step, so a swept check over it
needs 62 samples at the required 0.01 rad resolution and is capped at
`DEFAULT_MAX_SAMPLES = 12`. Half of every frame's sweeps therefore run
**sampling-degraded** (measured 7.5 of 15.2 per frame), and the mapping path
costs **87.6 ms/frame against 66.0 ms** with a one-step horizon. The supervisor
still sweeps the realized step at full resolution, so this is a cost and a
resolution note, not a safety hole — but it is the reason the control worker is
the system's bottleneck, and it is where to look first for latency.
