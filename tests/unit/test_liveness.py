"""Frame liveness: the gate that a frozen camera passes on every other check.

A frozen source keeps timestamps advancing and confidence high, so freshness and
confidence gates both approve it.  These tests pin the two properties that make
this component worth having -- it detects a freeze whose timestamps advance, and
it does not fire on an operator who is simply holding still -- plus the
fail-closed behaviour on every malformed or out-of-order input.
"""

from __future__ import annotations

import pytest

from galbot_motion_studio.vision.liveness import (
    FrameLivenessMonitor,
    LivenessPolicy,
    LivenessRejection,
    LivenessResult,
)

BUDGET_NS = 400_000_000
FPS30_NS = 33_333_333
FPS8_NS = 125_000_000
START_NS = 1_000_000_000


def monitor(**changes: object) -> FrameLivenessMonitor:
    values: dict[str, object] = {"max_static_ns": BUDGET_NS}
    values.update(changes)
    return FrameLivenessMonitor(LivenessPolicy(**values))  # type: ignore[arg-type]


def drive(
    subject: FrameLivenessMonitor,
    fingerprints: list[str],
    *,
    interval_ns: int = FPS30_NS,
    start_ns: int = START_NS,
    start_sequence: int = 1,
) -> list[LivenessResult]:
    """Feed a fingerprint stream at a fixed frame interval."""
    return [
        subject.evaluate(
            fingerprint=fingerprint,
            capture_mono_ns=start_ns + index * interval_ns,
            sequence=start_sequence + index,
        )
        for index, fingerprint in enumerate(fingerprints)
    ]


def test_first_frame_is_never_live_because_a_transition_needs_a_predecessor() -> None:
    result = drive(monitor(), ["a"])[0]
    assert result.live is False
    assert result.reason is LivenessRejection.NOT_EVALUATED
    assert result.accepted is True
    assert result.stalled_for_ns == 0


def test_a_changing_source_goes_live_on_the_first_transition() -> None:
    results = drive(monitor(), ["a", "b", "c"])
    assert [r.live for r in results] == [False, True, True]
    assert [r.reason for r in results] == [LivenessRejection.NOT_EVALUATED, None, None]
    assert all(r.stalled_for_ns == 0 for r in results[1:])


def test_a_source_frozen_from_the_first_frame_never_goes_live() -> None:
    """The startup case a confidence or freshness gate cannot see at all.

    Every frame is well-formed, fresh, and in order; only the pixels never move.
    """
    results = drive(monitor(), ["a"] * 40)
    assert not any(r.live for r in results)
    assert results[0].reason is LivenessRejection.NOT_EVALUATED
    assert results[-1].reason is LivenessRejection.SOURCE_STALLED
    assert all(r.repeated for r in results[1:])


def test_a_freeze_is_detected_although_capture_timestamps_keep_advancing() -> None:
    subject = monitor()
    live = drive(subject, ["a", "b", "c"])
    assert live[-1].live is True

    # The source freezes: same content, timestamps and sequence still advancing.
    frozen = drive(
        subject,
        ["c"] * 30,
        start_ns=START_NS + 3 * FPS30_NS,
        start_sequence=4,
    )
    assert frozen[0].live is True, "one repeated frame is a duplicate, not yet a stall"
    stalled = [r for r in frozen if r.reason is LivenessRejection.SOURCE_STALLED]
    assert stalled, "a frozen source with advancing timestamps was never detected"
    # The detection came from the content, not from a timing gate: every frozen
    # frame was accepted, and their capture times strictly increased throughout.
    assert all(r.accepted for r in frozen)
    captures = [r.capture_mono_ns for r in frozen]
    assert captures == sorted(set(captures))


def test_a_repeated_frame_after_a_capture_gap_is_still_a_freeze() -> None:
    """The first frame after a gap is the first chance to report its source stall.

    Pruning a fingerprint before this comparison would label the repeated image
    novel and incorrectly restart the liveness timer at the detection boundary.
    """
    subject = monitor()
    drive(subject, ["a", "b"])
    result = subject.evaluate(
        fingerprint="b",
        capture_mono_ns=START_NS + FPS30_NS + BUDGET_NS + 1,
        sequence=3,
    )
    assert result.reason is LivenessRejection.SOURCE_STALLED
    assert result.repeated is True


@pytest.mark.parametrize("interval_ns", [FPS30_NS, FPS8_NS])
def test_detection_time_is_bounded_by_the_budget_at_any_frame_rate(interval_ns: int) -> None:
    """The budget is a duration, so detection latency does not follow the frame rate.

    Measured frame rates in this project span roughly 8-25 fps depending on the
    operator's pose; a frame-count rule would make detection ~3x slower exactly
    when the pipeline is most loaded.
    """
    subject = monitor()
    drive(subject, ["a", "b"], interval_ns=interval_ns)
    frozen = drive(
        subject,
        ["b"] * 40,
        interval_ns=interval_ns,
        start_ns=START_NS + 2 * interval_ns,
        start_sequence=3,
    )
    first_stall = next(r for r in frozen if r.reason is LivenessRejection.SOURCE_STALLED)
    assert first_stall.stalled_for_ns is not None
    assert BUDGET_NS < first_stall.stalled_for_ns <= BUDGET_NS + interval_ns


def test_a_motionless_operator_never_stalls() -> None:
    """The false-positive case: a person holding still still produces sensor noise.

    Distinct fingerprints every frame is what a real camera gives even on a static
    scene, and none of it may read as a freeze.
    """
    results = drive(monitor(), [f"noise-{index}" for index in range(600)])
    assert all(r.live for r in results[1:])
    assert not any(r.repeated for r in results)


def test_an_alternating_driver_buffer_stalls_although_each_frame_differs() -> None:
    """A, B, A, B ... defeats a compare-with-the-predecessor rule entirely.

    Every frame differs from the one before it while no new imagery exists, which
    is why novelty is judged against a window rather than against the last frame.
    """
    results = drive(monitor(), ["a", "b"] * 30)
    assert results[1].live is True, "the first A->B transition is genuine evidence"
    assert results[-1].live is False
    assert results[-1].reason is LivenessRejection.SOURCE_STALLED


def test_a_regressed_capture_clock_cannot_restart_the_stall_timer() -> None:
    """A frozen camera behind a jittery clock must stay held.

    If a rejected frame advanced the state, a single backwards timestamp would
    reset the stall measurement and rescue a stalled source every time it glitched.
    """
    subject = monitor()
    drive(subject, ["a", "b"])
    frozen = drive(subject, ["b"] * 30, start_ns=START_NS + 2 * FPS30_NS, start_sequence=3)
    assert frozen[-1].reason is LivenessRejection.SOURCE_STALLED
    stalled_before = frozen[-1].stalled_for_ns
    assert stalled_before is not None

    regressed = subject.evaluate(
        fingerprint="b", capture_mono_ns=START_NS, sequence=999
    )
    assert regressed.live is False
    assert regressed.reason is LivenessRejection.CLOCK_NOT_MONOTONIC
    assert regressed.accepted is False

    # The next in-order frame is still stalled by at least as much as before.
    after = subject.evaluate(
        fingerprint="b",
        capture_mono_ns=START_NS + 32 * FPS30_NS,
        sequence=1_000,
    )
    assert after.reason is LivenessRejection.SOURCE_STALLED
    assert after.stalled_for_ns is not None and after.stalled_for_ns > stalled_before


def test_a_duplicate_capture_timestamp_is_rejected() -> None:
    subject = monitor()
    drive(subject, ["a", "b"])
    repeat = subject.evaluate(
        fingerprint="c", capture_mono_ns=START_NS + FPS30_NS, sequence=3
    )
    assert repeat.reason is LivenessRejection.CLOCK_NOT_MONOTONIC
    assert repeat.accepted is False


def test_sequence_regression_is_rejected_unless_the_policy_disables_the_check() -> None:
    strict = monitor()
    drive(strict, ["a", "b"])
    out_of_order = strict.evaluate(
        fingerprint="c", capture_mono_ns=START_NS + 2 * FPS30_NS, sequence=1
    )
    assert out_of_order.reason is LivenessRejection.SEQUENCE_NOT_MONOTONIC
    assert out_of_order.accepted is False

    relaxed = monitor(require_monotonic_sequence=False)
    drive(relaxed, ["a", "b"])
    accepted = relaxed.evaluate(
        fingerprint="c", capture_mono_ns=START_NS + 2 * FPS30_NS, sequence=1
    )
    assert accepted.accepted is True
    assert accepted.live is True


@pytest.mark.parametrize("fingerprint", ["", "   "])
def test_a_missing_fingerprint_fails_closed_without_touching_state(fingerprint: str) -> None:
    subject = monitor()
    drive(subject, ["a", "b"])
    before = subject.last_change_mono_ns

    result = subject.evaluate(
        fingerprint=fingerprint, capture_mono_ns=START_NS + 2 * FPS30_NS, sequence=3
    )
    assert result.live is False
    assert result.reason is LivenessRejection.MISSING_FINGERPRINT
    assert result.accepted is False
    assert subject.last_change_mono_ns == before

    # The state really is untouched: the rejected frame's timestamp is still free.
    replayed = subject.evaluate(
        fingerprint="c", capture_mono_ns=START_NS + 2 * FPS30_NS, sequence=3
    )
    assert replayed.accepted is True


@pytest.mark.parametrize(
    ("capture_mono_ns", "sequence"),
    [(-1, 3), (START_NS + 2 * FPS30_NS, -1)],
)
def test_a_negative_timestamp_or_sequence_is_an_invalid_frame(
    capture_mono_ns: int, sequence: int
) -> None:
    subject = monitor()
    drive(subject, ["a", "b"])
    result = subject.evaluate(
        fingerprint="c", capture_mono_ns=capture_mono_ns, sequence=sequence
    )
    assert result.reason is LivenessRejection.INVALID_FRAME
    assert result.accepted is False


def test_the_stall_boundary_is_inclusive() -> None:
    """Exactly at the budget is still live; one nanosecond past it is not."""
    subject = monitor()
    subject.evaluate(fingerprint="a", capture_mono_ns=START_NS, sequence=1)
    subject.evaluate(fingerprint="b", capture_mono_ns=START_NS + 1, sequence=2)

    at_budget = subject.evaluate(
        fingerprint="b", capture_mono_ns=START_NS + 1 + BUDGET_NS, sequence=3
    )
    assert at_budget.live is True
    assert at_budget.stalled_for_ns == BUDGET_NS

    past_budget = subject.evaluate(
        fingerprint="b", capture_mono_ns=START_NS + 2 + BUDGET_NS, sequence=4
    )
    assert past_budget.live is False
    assert past_budget.reason is LivenessRejection.SOURCE_STALLED


def test_new_content_after_a_stall_restores_liveness() -> None:
    subject = monitor()
    drive(subject, ["a", "b"])
    frozen = drive(subject, ["b"] * 30, start_ns=START_NS + 2 * FPS30_NS, start_sequence=3)
    assert frozen[-1].reason is LivenessRejection.SOURCE_STALLED

    recovered = subject.evaluate(
        fingerprint="fresh",
        capture_mono_ns=START_NS + 32 * FPS30_NS,
        sequence=100,
    )
    assert recovered.live is True
    assert recovered.reason is None
    assert recovered.stalled_for_ns == 0
    assert recovered.frames_since_change == 0


def test_frames_since_change_counts_repeats() -> None:
    results = drive(monitor(), ["a", "b", "b", "b", "c"])
    assert [r.frames_since_change for r in results] == [0, 0, 1, 2, 0]
    assert [r.repeated for r in results] == [False, False, True, True, False]


def test_reset_returns_the_monitor_to_the_fail_closed_startup_state() -> None:
    subject = monitor()
    drive(subject, ["a", "b", "c"])
    assert subject.last_change_mono_ns is not None

    subject.reset()
    assert subject.last_change_mono_ns is None
    assert subject.tracked_fingerprints == 0

    # A reset monitor accepts an earlier clock (a reopened camera) and is not live
    # until content changes again.
    first = subject.evaluate(fingerprint="a", capture_mono_ns=1, sequence=1)
    assert first.live is False
    assert first.reason is LivenessRejection.NOT_EVALUATED


def test_the_novelty_window_stays_bounded() -> None:
    subject = monitor(max_history=8, max_static_ns=10**15)
    drive(subject, [f"unique-{index}" for index in range(200)])
    assert subject.tracked_fingerprints <= 8


def test_the_same_input_sequence_replays_to_identical_verdicts() -> None:
    """Determinism: no wall clock, no randomness, no I/O."""
    stream = ["a", "b", "b", "c", "c", "c", "c", "c", "c", "c", "c", "d", "d", "e"]
    assert drive(monitor(), stream) == drive(monitor(), stream)


@pytest.mark.parametrize(
    "changes",
    [{"max_static_ns": 0}, {"max_static_ns": -1}, {"max_history": 0}],
)
def test_the_policy_rejects_configuration_that_cannot_be_safe(changes: dict[str, int]) -> None:
    values: dict[str, int] = {"max_static_ns": BUDGET_NS}
    values.update(changes)
    with pytest.raises(ValueError):
        LivenessPolicy(**values)  # type: ignore[arg-type]


def test_the_budget_has_no_default_so_a_deployment_must_state_it() -> None:
    """The detection budget is a safety-owned deployment value, not a library constant."""
    with pytest.raises(TypeError):
        LivenessPolicy()  # type: ignore[call-arg]
