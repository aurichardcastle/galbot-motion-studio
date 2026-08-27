"""A latched FAULT must stop the session on every path, with or without a window.

`docs/safety-model.md` states the contract: HOLD is normal and must be quiet, FAULT
is latched and requires human inspection and an explicit reset. Concretely, once
a FAULT decision exists:

* no further joint target may reach the sink,
* no further frame may be appended to the published MotionClip,
* the operator has to be able to SEE it -- on screen, in the terminal, and in the
  exit status,
* and the clip must say what actually stopped the session.

This file exists because that contract has been broken three times, each time
only on the paths the previous test did not run:

1. Enforcement moved, during a refactor, into the function that draws the window
   -- which returns immediately when there is no window. Measured then: a FAULT
   injected at solve 20 of a headless replay stopped nothing, and the run
   published 398 further frames and exited 0.
2. Back on the capture loop, it still only ran when a result surfaced through
   `poll_latest`. The latest-wins pipeline is two deep, so in every LIVE mode two
   more frames were solved, submitted to the sink and appended to the clip after
   the fault. `--analysis-sync` is synchronous and structurally could not show it.
3. Latching on the worker fixed that and broke the DISPLAY: the declined result
   overwrote the FAULT in the one-slot results queue, so the window, the status
   line and the composite all ended on the last ALLOW while the clip ended in
   FAULT. And a perception ingress fault never set the latch at all, so a solve
   already in flight appended out of order and became the clip's `terminal_fault`
   -- making a replay blame "supervisor is FAULT" instead of the camera.

So the assertions are made against the CLIP, the SINK, the EXIT STATUS and the
HUD, in every mode: synchronous, live, live with a composite writer, and the
perception-ingress path, which is a different fault route entirely.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
from itertools import pairwise
from pathlib import Path

import pytest

import galbot_motion_studio.pipeline as pipeline_module
from galbot_motion_studio.adapters.mediapipe_holistic import (
    HolisticDetectorError,
    MediaPipeHolisticDetector,
)
from galbot_motion_studio.adapters.mujoco_preview import MujocoPreviewSink
from galbot_motion_studio.cli import main
from galbot_motion_studio.contracts.core import SafetyOutcome
from galbot_motion_studio.display import StudioDisplay

WITNESS = Path(__file__).resolve().parents[1] / "artifacts" / "witness-live-9" / "raw.mp4"
INJECT_AT_SOLVE = 20
#: Detector call to fail on for the perception path. Calibration completes around
#: call 170 on this capture, so this is comfortably into the commanded session.
INJECT_AT_DETECT = 250

pytestmark = pytest.mark.skipif(
    not WITNESS.exists(),
    reason=(
        f"no witness capture at {WITNESS} -- the FAULT latch is UNTESTED in this "
        "checkout; record one with `preview --camera builtin --source-video ...`"
    ),
)

CALIBRATION_ARGS = [
    "--calibration-window-ms", "1500",
    "--calibration-min-samples", "15",
    "--calibration-max-center-deviation-normalized", "0.03",
    "--calibration-max-shoulder-width-deviation-normalized", "0.03",
    "--calibration-max-eye-span-deviation-normalized", "0.02",
]

# `--analysis-sync` solves inline on the capture loop; without it the solve runs
# on the latest-wins control worker, which is where two of the three regressions
# lived. `--preview-video` additionally builds a StudioDisplay and a composite
# writer (still headless: `show_canvas` needs --display).
LIVE_MODES = ("live", "live+composite", "perception")
MODES = ("analysis-sync", *LIVE_MODES)


@pytest.fixture
def watched(monkeypatch):
    """Inject one fault, and watch everything that must stop when it lands.

    Counts solves, sink submissions and the statuses the HUD was asked to draw,
    so the test can say what reached the robot, the clip and the operator -- not
    merely what was printed.
    """
    state = {
        "solves": 0,
        "solves_after_fault": 0,
        "submits": 0,
        "submits_at_fault": None,
        "faulted": False,
        "hud_statuses": [],
    }
    original = pipeline_module.MotionStudioPipeline.process_fail_closed
    original_submit = MujocoPreviewSink.submit
    original_detect = MediaPipeHolisticDetector.detect
    original_render = StudioDisplay.render

    def counted_submit(self, approved):
        state["submits"] += 1
        return original_submit(self, approved)

    def watched_render(self, *args, **kwargs):
        state["hud_statuses"].append(kwargs.get("status"))
        return original_render(self, *args, **kwargs)

    def injected_solve(self, observation, **kwargs):
        if state["faulted"]:
            state["solves_after_fault"] += 1
        result = original(self, observation, **kwargs)
        state["solves"] += 1
        if state["mode"] != "perception" and state["solves"] == INJECT_AT_SOLVE:
            state["faulted"] = True
            state["submits_at_fault"] = state["submits"]
            decision = result.decision.model_copy(
                update={
                    "outcome": SafetyOutcome.FAULT,
                    "reasons": ("injected fault: latch regression test",),
                }
            )
            return dataclasses.replace(result, decision=decision)
        return result

    def injected_detect(self, frame, **kwargs):
        state["detects"] = state.get("detects", 0) + 1
        if state["mode"] == "perception" and state["detects"] == INJECT_AT_DETECT:
            state["faulted"] = True
            state["submits_at_fault"] = state["submits"]
            raise HolisticDetectorError("injected fault: perception ingress")
        return original_detect(self, frame, **kwargs)

    monkeypatch.setattr(MujocoPreviewSink, "submit", counted_submit)
    monkeypatch.setattr(StudioDisplay, "render", watched_render)
    monkeypatch.setattr(MediaPipeHolisticDetector, "detect", injected_detect)
    monkeypatch.setattr(
        pipeline_module.MotionStudioPipeline, "process_fail_closed", injected_solve
    )
    return state


def _run(tmp_path: Path, mode: str, watched: dict) -> tuple[dict, str, object]:
    watched["mode"] = mode
    output = tmp_path / "live.json"
    argv = [
        "preview",
        "--video", str(WITNESS),
        *(["--analysis-sync"] if mode == "analysis-sync" else []),
        *CALIBRATION_ARGS,
        "--output", str(output),
        "--force",
    ]
    if mode == "live+composite":
        argv += ["--preview-video", str(tmp_path / "composite.mp4")]
    buffer = io.StringIO()
    stdout = sys.stdout
    sys.stdout = buffer
    exit_code: object = 0
    try:
        main(argv)
    except SystemExit as stopped:
        exit_code = stopped.code
    finally:
        sys.stdout = stdout
    return json.loads(output.read_text()), buffer.getvalue(), exit_code


@pytest.mark.parametrize("mode", MODES)
def test_a_latched_fault_stops_the_session(tmp_path: Path, watched, mode) -> None:
    clip, printed, exit_code = _run(tmp_path, mode, watched)
    frames = clip["frames"]
    outcomes = [frame["decision"]["outcome"] for frame in frames]
    where = f"[{mode}]"

    # 1. Nothing is solved, and nothing reaches the robot, after the fault.
    assert watched["solves_after_fault"] == 0, (
        f"{where} the pipeline solved {watched['solves_after_fault']} frame(s) past "
        "a latched FAULT -- the latch is not reachable on this path"
    )
    assert watched["submits"] == watched["submits_at_fault"], (
        f"{where} {watched['submits'] - watched['submits_at_fault']} joint target(s) "
        "were submitted to the sink after the fault latched"
    )

    # 2. The published clip ends AT the fault, in order, and says so itself.
    assert outcomes[-1] == "FAULT", f"{where} clip does not end in FAULT: {outcomes[-5:]}"
    first_fault = outcomes.index("FAULT")
    assert set(outcomes[first_fault:]) == {"FAULT"}, (
        f"{where} {len(outcomes) - 1 - first_fault} commanded frame(s) were published "
        f"after the first FAULT: {outcomes[first_fault:][:6]}"
    )
    sequences = [frame["source_sequence"] for frame in frames]
    assert all(b > a for a, b in pairwise(sequences)), (
        f"{where} clip sequences are not strictly increasing: {sequences[-5:]}"
    )
    assert clip["terminal_fault"] is None, (
        f"{where} the fault was diverted to terminal_fault "
        f"(sequence {clip['terminal_fault'].get('sequence')}) instead of ending the "
        f"frames, whose last sequence is {sequences[-1]} -- a replay will blame "
        f"{clip['terminal_fault'].get('reasons')} rather than the real cause"
    )

    # 3. The operator can see it: on the terminal, and in the exit status.
    expected = (
        "perception ingress FAULT" if mode == "perception"
        else "preview entered FAULT"
    )
    assert expected in printed, printed[-400:]
    assert exit_code != 0, (
        f"{where} a latched FAULT exited 0 -- indistinguishable from a clean run "
        "to close_gates.sh or any other caller that reads the status"
    )
    assert "latched FAULT" in str(exit_code), exit_code

    # 4. ...and on screen, wherever there is a screen.
    if mode == "live+composite":
        assert "FAULT" in watched["hud_statuses"], (
            f"{where} the HUD was never asked to draw FAULT (last statuses: "
            f"{watched['hud_statuses'][-5:]}) -- the window latches shut still "
            "showing the operator green"
        )
