from math import sqrt

from galbot_motion_studio.retarget.trajectory import (
    JointTrajectoryGovernor,
    TrajectoryLimits,
    feasible_velocity_limit,
)


def test_governor_bounds_velocity_and_acceleration_through_reversals() -> None:
    governor = JointTrajectoryGovernor(
        TrajectoryLimits(max_velocity_rad_s=1.5, max_acceleration_rad_s2=2.0)
    )
    governor.reset({"joint": 0.0}, timestamp_ns=1_000_000_000)
    previous_position = 0.0
    previous_velocity = 0.0
    period_ns = 33_333_333

    for frame in range(1, 181):
        desired = 0.7 if (frame // 30) % 2 == 0 else -0.7
        position = governor.step(
            {"joint": desired}, timestamp_ns=1_000_000_000 + frame * period_ns
        )["joint"]
        elapsed_s = period_ns / 1_000_000_000
        velocity = (position - previous_position) / elapsed_s
        acceleration = (velocity - previous_velocity) / elapsed_s
        assert abs(velocity) <= 1.5 + 1e-8
        assert abs(acceleration) <= 2.0 + 1e-6
        previous_position = position
        previous_velocity = velocity


def test_governor_applies_a_stricter_named_velocity_cap_to_emitted_positions() -> None:
    governor = JointTrajectoryGovernor(
        TrajectoryLimits(
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=10.0,
            max_position_step_rad=0.1,
        ),
        max_velocity_rad_s_by_joint={"torso": 0.35},
    )
    governor.reset({"torso": 0.0, "arm": 0.0}, timestamp_ns=1_000_000_000)
    timestamp_ns = 1_000_000_000
    previous = {"torso": 0.0, "arm": 0.0}
    for _ in range(20):
        timestamp_ns += 100_000_000
        emitted = governor.step({"torso": 2.0, "arm": 2.0}, timestamp_ns=timestamp_ns)
        assert abs((emitted["torso"] - previous["torso"]) / 0.1) <= 0.35 + 1e-12
        assert abs((emitted["arm"] - previous["arm"]) / 0.1) <= 1.0 + 1e-12
        previous = emitted


def test_governor_requires_strict_time_and_exact_joint_set() -> None:
    governor = JointTrajectoryGovernor(
        TrajectoryLimits(max_velocity_rad_s=1.0, max_acceleration_rad_s2=1.0)
    )
    governor.reset({"joint": 0.0}, timestamp_ns=10)

    try:
        governor.step({"other": 0.0}, timestamp_ns=11)
    except ValueError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("mismatched joints were accepted")

    try:
        governor.step({"joint": 0.0}, timestamp_ns=10)
    except ValueError as error:
        assert "strictly increase" in str(error)
    else:
        raise AssertionError("duplicate timestamp was accepted")


def test_governor_caps_a_sparse_frame_step_for_the_supervisor() -> None:
    governor = JointTrajectoryGovernor(
        TrajectoryLimits(
            max_velocity_rad_s=0.6,
            max_acceleration_rad_s2=2.0,
            max_position_step_rad=0.05,
        )
    )
    governor.reset({"joint": 0.0}, timestamp_ns=1_000_000_000)
    position = governor.step({"joint": 1.0}, timestamp_ns=2_000_000_000)["joint"]
    assert position == 0.05


def _governor_with_momentum() -> JointTrajectoryGovernor:
    # Build up real velocity so preview and step exercise the accel/velocity path,
    # not just a from-rest step.
    governor = JointTrajectoryGovernor(
        TrajectoryLimits(
            max_velocity_rad_s=0.8,
            max_acceleration_rad_s2=2.0,
            max_position_step_rad=0.1,
        )
    )
    governor.reset({"a": 0.0, "b": 0.0}, timestamp_ns=1_000_000_000)
    ts = 1_000_000_000
    for _ in range(5):
        ts += 33_000_000
        governor.step({"a": 1.0, "b": -1.0}, timestamp_ns=ts)
    return governor


def test_preview_matches_step_exactly() -> None:
    # preview must return byte-identical positions to what step would emit, so a
    # caller can validate the realized pose before committing to it.
    a = _governor_with_momentum()
    b = _governor_with_momentum()
    ts = a._timestamp_ns + 33_000_000
    desired = {"a": 0.3, "b": 0.2}
    previewed = a.preview(desired, timestamp_ns=ts)
    stepped = b.step(desired, timestamp_ns=ts)
    assert previewed == stepped


def test_preview_does_not_mutate_state() -> None:
    governor = _governor_with_momentum()
    ts = governor._timestamp_ns + 33_000_000
    before_pos = dict(governor._positions)
    before_vel = dict(governor._velocities)
    before_ts = governor._timestamp_ns
    # Two previews with different targets, then a step: none of the previews may
    # perturb the governor, and the step must land where a fresh governor would.
    governor.preview({"a": 5.0, "b": 5.0}, timestamp_ns=ts)
    governor.preview({"a": -5.0, "b": -5.0}, timestamp_ns=ts)
    assert governor._positions == before_pos
    assert governor._velocities == before_vel
    assert governor._timestamp_ns == before_ts

    reference = _governor_with_momentum()
    desired = {"a": 0.4, "b": -0.4}
    assert governor.step(desired, timestamp_ns=ts) == reference.step(
        desired, timestamp_ns=ts
    )


def test_preview_validates_time_and_joint_set() -> None:
    governor = JointTrajectoryGovernor(
        TrajectoryLimits(max_velocity_rad_s=1.0, max_acceleration_rad_s2=1.0)
    )
    governor.reset({"joint": 0.0}, timestamp_ns=10)
    try:
        governor.preview({"other": 0.0}, timestamp_ns=11)
    except ValueError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("preview accepted a mismatched joint set")
    try:
        governor.preview({"joint": 0.0}, timestamp_ns=10)
    except ValueError as error:
        assert "strictly increase" in str(error)
    else:
        raise AssertionError("preview accepted a non-increasing timestamp")


def _reconstructed_accelerations(
    limits: TrajectoryLimits, intervals_ns: list[int], target: float
) -> list[float]:
    """Drive one joint and rebuild acceleration the way the supervisor does.

    The supervisor never sees the governor's internal velocity. It differentiates
    the emitted positions twice, so that is what this measures.

    Intervals are nanoseconds, not seconds, and the elapsed time is derived from
    the integer timestamps -- the same arithmetic the governor and the supervisor
    both perform. Passing a float interval and dividing by it instead disagrees
    with the governor by the nanosecond truncation, which is ~5e-9 relative and
    large enough on its own to fake an acceleration violation.
    """
    governor = JointTrajectoryGovernor(limits)
    timestamp_ns = 1_000_000_000
    governor.reset({"joint": 0.0}, timestamp_ns=timestamp_ns)
    position = 0.0
    previous_velocity = 0.0
    accelerations: list[float] = []
    for interval_ns in intervals_ns:
        timestamp_ns += interval_ns
        elapsed_s = interval_ns / 1_000_000_000
        emitted = governor.step({"joint": target}, timestamp_ns=timestamp_ns)["joint"]
        velocity = (emitted - position) / elapsed_s
        accelerations.append(abs(velocity - previous_velocity) / elapsed_s)
        position, previous_velocity = emitted, velocity
    return accelerations


def test_limits_reject_a_dynamically_infeasible_triple() -> None:
    # 1.5 rad/s cannot stop inside a 0.05 rad step cap at 1.0 rad/s^2: the cap
    # rewrites velocity and the implied deceleration exceeds the limiter's own.
    try:
        TrajectoryLimits(
            max_velocity_rad_s=1.5,
            max_acceleration_rad_s2=1.0,
            max_position_step_rad=0.05,
        )
    except ValueError as error:
        assert "dynamically" in str(error)
        # The message must name the fix, not just the fault.
        assert "0.44" in str(error)
    else:
        raise AssertionError("an infeasible limit triple was accepted")


def test_feasible_velocity_limit_is_itself_constructible() -> None:
    # The derived bound is the fastest admissible speed, so it must round-trip
    # through the invariant it was derived to satisfy -- otherwise every caller
    # deriving a limit would be rejected by the check meant to bless it.
    for acceleration, step in ((1.0, 0.05), (2.0, 0.1), (0.35, 0.02), (8.0, 0.2)):
        bound = feasible_velocity_limit(
            max_acceleration_rad_s2=acceleration, max_position_step_rad=step
        )
        limits = TrajectoryLimits(
            max_velocity_rad_s=bound,
            max_acceleration_rad_s2=acceleration,
            max_position_step_rad=step,
        )
        assert limits.max_velocity_rad_s == bound
        # And nothing faster is: the bound is tight, not merely sufficient.
        try:
            TrajectoryLimits(
                max_velocity_rad_s=bound * 1.001,
                max_acceleration_rad_s2=acceleration,
                max_position_step_rad=step,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"a speed above the bound was accepted for {step}")


def test_no_constructible_governor_exceeds_its_acceleration_limit() -> None:
    """The safety property the invariant exists to buy.

    Build velocity at a dense frame rate, then hand the governor a sparse
    interval -- including the worst one, ``sqrt(step/a)``, where the step cap
    binds hardest. No emitted pair of positions may reconstruct an acceleration
    above the declared limit, at any frame rate.
    """
    for acceleration, step in ((1.0, 0.05), (2.0, 0.1), (0.35, 0.02)):
        velocity = feasible_velocity_limit(
            max_acceleration_rad_s2=acceleration, max_position_step_rad=step
        )
        limits = TrajectoryLimits(
            max_velocity_rad_s=velocity,
            max_acceleration_rad_s2=acceleration,
            max_position_step_rad=step,
        )
        worst_s = sqrt(step / acceleration)
        # 60 dense frames saturate velocity; then sweep sparse intervals around
        # the worst case, which is where the old code produced its HOLDs.
        sparse = [
            int(worst_s * scale * 1_000_000_000)
            for scale in (0.25, 0.5, 0.9, 1.0, 1.1, 2.0, 4.0)
        ]
        for interval_ns in sparse:
            intervals = [8_000_000] * 60 + [interval_ns] + [8_000_000] * 5
            accelerations = _reconstructed_accelerations(limits, intervals, target=10.0)
            worst = max(accelerations)
            assert worst <= acceleration + 1e-12, (
                f"a={acceleration} step={step} dt={interval_ns / 1e9:.4f}: "
                f"reconstructed {worst} exceeds the limit"
            )


def test_the_feasibility_bound_is_the_real_constraint_not_a_safety_factor() -> None:
    # Necessity, proved on the inequality itself rather than by constructing an
    # object the invariant now forbids: just above 2*sqrt(a*step) there is always
    # an interval where the step cap demands more deceleration than is allowed.
    acceleration, step = 1.0, 0.05
    exact_bound = 2.0 * sqrt(acceleration * step)
    worst_s = sqrt(step / acceleration)
    allowed = acceleration * worst_s + step / worst_s
    assert abs(allowed - exact_bound) < 1e-12
    assert exact_bound * 1.01 > allowed  # infeasible at the worst interval
    assert exact_bound * 0.99 < allowed  # feasible everywhere below it
