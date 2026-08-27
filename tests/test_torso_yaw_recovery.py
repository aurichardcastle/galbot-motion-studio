"""A latched torso-yaw hold must be recoverable without restarting the session.

`TORSO_YAW_RECALIBRATION_REQUIRED` freezes BOTH arms, both grippers and the torso.
It used to be cleared only by `calibrate_window` and the reset path, and the live
CLI has no in-run calibration entry point -- so once it latched, the arms were
dead for the rest of the session while the run kept recording and exited 0. On
real footage four of five ordinary perturbations reached that state: stepping out
of shot and back, an occluded shoulder, turning side-on, a light change.

The reason for the latch is sound and is unchanged: a >20 deg apparent shoulder
step can be a LEFT/RIGHT LABEL SWAP, and arms driven through a swapped torso frame
point the wrong way in the world. What changed is the remedy. Recovery requires
the same evidence the initial calibration requires -- fifteen consecutive frames,
face-on, continuous in time, with a calm yaw chain -- and nothing is driven while
that evidence accumulates.
"""

from __future__ import annotations

from galbot_motion_studio.pipeline import (
    TORSO_RECOVERY_FRAMES,
    MotionStudioPipeline,
)
from galbot_motion_studio.synthetic import synthetic_observation
from galbot_motion_studio.vision.freshness import ControlGroup

FRAME_NS = 33_000_000
LEFT_ARM = str(ControlGroup.LEFT_ARM)
RIGHT_ARM = str(ControlGroup.RIGHT_ARM)
TORSO = str(ControlGroup.TORSO)


def _calibrated() -> MotionStudioPipeline:
    pipeline = MotionStudioPipeline(source_clock_id="synthetic-clock")
    pipeline.calibrate(
        synthetic_observation(0, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    )
    return pipeline


def _observation(sequence: int):
    return synthetic_observation(
        sequence,
        timestamp_ns=1_000_000_000 + sequence * FRAME_NS,
        motion_fraction=0.5,
    )


def _run(pipeline, first, count, *, yaw=0.0, front=True, monkeypatch=None):
    """Feed `count` calm, face-on frames and return each frame's held groups.

    The yaw and the face-front verdict are injected rather than inferred from the
    frames: `synthetic_observation` carries no metric `world_xyz_m`, so
    `_observed_torso_yaw` returns None for it and no synthetic clip can exercise
    this path at all. What is under test is the recovery POLICY -- how many
    consecutive admissible frames are required, what breaks the chain, and that
    nothing is driven until it is earned -- so the evidence is supplied directly
    and the policy is what gets measured.
    """
    held = []
    for index in range(count):
        sequence = first + index
        observation = _observation(sequence)
        held_groups = _step(pipeline, observation, yaw, front, monkeypatch)
        held.append(held_groups)
    return held


def _step(pipeline, observation, yaw, front, monkeypatch):
    import galbot_motion_studio.pipeline as module

    monkeypatch.setattr(module, "_observed_torso_yaw", lambda _obs: yaw)
    monkeypatch.setattr(
        module, "_front_head_evidence_available", lambda _obs, _cal: front
    )
    result = pipeline.process_fail_closed(
        observation, now_mono_ns=observation.capture_mono_ns
    )
    return frozenset(result.held_groups)


def test_a_latched_torso_yaw_hold_clears_itself_once_the_operator_is_face_on(
    monkeypatch,
) -> None:
    pipeline = _calibrated()
    pipeline._torso_recalibration_required = True
    held = _run(pipeline, 6, TORSO_RECOVERY_FRAMES + 4, monkeypatch=monkeypatch)

    assert {LEFT_ARM, RIGHT_ARM, TORSO} <= held[0], "the latch never held"
    assert not pipeline._torso_recalibration_required, (
        "the arms are still dead after "
        f"{TORSO_RECOVERY_FRAMES + 4} provable frames -- the latch never cleared"
    )
    freed = next(
        (index for index, groups in enumerate(held) if LEFT_ARM not in groups), None
    )
    assert freed is not None, "the arms never came back"
    assert freed >= TORSO_RECOVERY_FRAMES - 1, (
        f"the arms resumed after only {freed + 1} frames; recovery must require "
        f"{TORSO_RECOVERY_FRAMES}"
    )


def test_nothing_is_driven_while_the_evidence_is_still_accumulating(
    monkeypatch,
) -> None:
    """The hold is real until it is earned -- no partial credit."""
    pipeline = _calibrated()
    pipeline._torso_recalibration_required = True
    held = _run(pipeline, 6, TORSO_RECOVERY_FRAMES - 2, monkeypatch=monkeypatch)

    assert pipeline._torso_recalibration_required, "cleared too early"
    for index, groups in enumerate(held):
        assert {LEFT_ARM, RIGHT_ARM, TORSO} <= groups, (
            f"frame {index} drove a coupled group before recovery was complete"
        )


def test_a_face_turned_away_is_not_evidence_of_anything(monkeypatch) -> None:
    """Face-front evidence IS the identity argument the latch is made of.

    A shoulder-label swap needs the operator turned past ~105 deg, where the face
    is not toward the camera -- so a face-on frame is what makes the labels
    provable. Without it, no number of calm frames may clear the latch.
    """
    pipeline = _calibrated()
    pipeline._torso_recalibration_required = True
    _run(pipeline, 6, TORSO_RECOVERY_FRAMES * 3, front=False, monkeypatch=monkeypatch)

    assert pipeline._torso_recalibration_required, (
        "the latch cleared without a single face-on frame"
    )
    assert pipeline._torso_recovery_frames == 0


def test_a_violent_yaw_chain_is_not_a_calm_one(monkeypatch) -> None:
    """A run of near-threshold steps is not evidence the chain has settled."""
    from math import radians

    pipeline = _calibrated()
    pipeline._torso_recalibration_required = True
    for index in range(TORSO_RECOVERY_FRAMES * 3):
        observation = _observation(6 + index)
        _step(
            pipeline,
            observation,
            radians(19.0) * (index % 2),   # 0, 19, 0, 19 ... all under the 20 limit
            True,
            monkeypatch,
        )
    assert pipeline._torso_recalibration_required, (
        "19-degree alternating steps counted as a calm chain"
    )


def test_a_break_in_the_evidence_restarts_the_count(monkeypatch) -> None:
    """Fifteen CONSECUTIVE frames. A gap is not a chain."""
    pipeline = _calibrated()
    pipeline._torso_recalibration_required = True
    _run(pipeline, 6, TORSO_RECOVERY_FRAMES - 3, monkeypatch=monkeypatch)
    assert pipeline._torso_recalibration_required

    stale = synthetic_observation(
        900, timestamp_ns=1_000_000_000 + 900 * FRAME_NS * 40, motion_fraction=0.5
    )
    _step(pipeline, stale, 0.0, True, monkeypatch)
    assert pipeline._torso_recalibration_required, "a time hole still counted"
    assert pipeline._torso_recovery_frames <= 1, (
        f"the count survived a break: {pipeline._torso_recovery_frames}"
    )
