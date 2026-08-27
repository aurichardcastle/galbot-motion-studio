"""The live `preview` command, run end to end against a fake camera.

This is the demo path and the one path CI cannot normally reach: macOS denies
camera access to a non-interactive process, so `cv2.VideoCapture(0)` fails and
every live-only branch -- the raw-capture evidence path, the landmark sidecar,
the continuity witness, the calibration window against a real moving operator,
and the publication checks -- would otherwise ship having never been executed
once. Everything the repository could previously test ran through `--video`
replay, which takes a different branch at almost every one of those points.

`WebcamSource` takes its `cv2` module by injection, so replacing the capture with
one that replays a real recorded clip at the real 30 fps cadence drives the
production code path exactly. The pacing is not cosmetic: calibration wants 15
consecutive frames of a nearly-still operator, so an unthrottled replay hands the
loop several seconds of motion per "consecutive" frame and never calibrates.

The assertions are about what the loop must never lose: the raw video, its frame
map and the landmark sidecar are "no gaps" records of every frame the detector
saw, while the UI deliberately drops frames it cannot keep up with. If those
three still agree frame-for-frame after a live session, the drop happened in the
right place.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import pytest

from galbot_motion_studio import cli

WITNESS = Path(__file__).resolve().parents[1] / "artifacts" / "witness-live-9" / "raw.mp4"
# The witness clip opens with the operator walking into shot, and calibration
# only locked on its frame 356. Start the replay where a person is already framed
# so the test spends its time on the live loop rather than on that walk-in.
FIRST_FRAME = 300
FRAMES = 420


# The clip is operator-identifying camera footage and `.gitignore` keeps every
# `*.mp4` and all of `artifacts/` out of the repository on purpose, so this file
# runs ONLY on a machine that has recorded one. That is a real limitation and it
# is stated loudly rather than hidden behind a green run: on any other checkout
# these tests SKIP, and a skipped test is not evidence. Recording a fresh one is
# `preview --camera builtin --source-video ... --landmark-sidecar ...` per
# docs/operating-guide.md.
pytestmark = pytest.mark.skipif(
    not WITNESS.exists(),
    reason=(
        f"no witness capture at {WITNESS} -- the live path is UNTESTED in this "
        "checkout; record one with `preview --camera builtin --source-video ...`"
    ),
)


class ReplayCapture:
    """A `cv2.VideoCapture` stand-in that loops a recorded clip forever.

    Endless on purpose: a real camera does not end, and `WebcamSource` correctly
    raises `WebcamError` when `read()` fails. The run is bounded by `--max-frames`
    exactly as a live session is bounded by the operator pressing q.
    """

    #: Real capture cadence of the witness session. Pacing matters: the
    #: calibration window asks for 15 CONSECUTIVE samples whose shoulder centre,
    #: shoulder width and eye span all stay inside measured deviation limits.
    #: An unthrottled replay hands the loop frames as fast as it can decode them,
    #: so 15 consecutive delivered frames span several seconds of operator motion
    #: instead of half a second, and the window is correctly rejected every time.
    #: The camera is the clock here, so the fake one has to keep it.
    FRAME_PERIOD_S = 1.0 / 30.0

    def __init__(self, path: Path) -> None:
        self._path = str(path)
        self._capture = self._open()
        self._next_due = time.monotonic()
        self.released = False

    def _open(self):
        capture = cv2.VideoCapture(self._path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, FIRST_FRAME)
        return capture

    def isOpened(self) -> bool:  # cv2 spelling, not ours
        return self._capture.isOpened()

    def read(self):
        delay = self._next_due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_due = max(time.monotonic(), self._next_due + self.FRAME_PERIOD_S)
        ok, image = self._capture.read()
        if not ok:
            self._capture.release()
            self._capture = self._open()
            ok, image = self._capture.read()
        return ok, image

    def release(self) -> None:
        self.released = True
        self._capture.release()


class ReplayCv2:
    def __init__(self, path: Path) -> None:
        self._path = path

    def VideoCapture(self, device):  # cv2 spelling, not ours
        return ReplayCapture(self._path)


@pytest.fixture
def fake_camera(monkeypatch):
    """Swap the camera for a paced replay of a real recorded session."""
    real = cli.WebcamSource

    def build(device, *, source_clock_id, cv2_module=None):
        return real(device, source_clock_id=source_clock_id, cv2_module=ReplayCv2(WITNESS))

    monkeypatch.setattr(cli, "WebcamSource", build)


def run_preview(tmp_path: Path, *extra: str) -> int:
    return cli.main(
        [
            "preview",
            "--camera", "0",
            "--max-frames", str(FRAMES),
            "--calibration-window-ms", "1500",
            "--calibration-min-samples", "15",
            "--calibration-max-center-deviation-normalized", "0.03",
            "--calibration-max-shoulder-width-deviation-normalized", "0.03",
            "--calibration-max-eye-span-deviation-normalized", "0.02",
            "--liveness-max-static-ms", "500",
            "--output", str(tmp_path / "live.json"),
            # Turns the studio on: `cli.py` builds a `StudioDisplay` only when
            # one of --display/--fullscreen/--preview-video is set, so without
            # this the end-to-end test for a session whose headline change is the
            # HUD would exercise none of it. Writing to tmp_path costs nothing
            # extra and additionally checks the composite video and its frame map.
            "--preview-video", str(tmp_path / "composite.mp4"),
            *extra,
        ]
    )


def test_a_live_session_records_a_publishable_clip(
    tmp_path: Path, fake_camera
) -> None:
    """The whole live loop runs: capture, calibrate, command, publish."""
    assert run_preview(tmp_path) == 0
    clip = json.loads((tmp_path / "live.json").read_text())
    assert clip["frames"], "a live session produced no commanded frames"
    outcomes = {frame["decision"]["outcome"] for frame in clip["frames"]}
    assert "FAULT" not in outcomes, sorted(outcomes)

    # The HUD ran for every one of those frames: the composite exists, decodes,
    # and its frame map lines up with it.
    composite = tmp_path / "composite.mp4"
    assert composite.exists() and composite.stat().st_size > 0
    composite_map = json.loads((tmp_path / "composite.frame-map.json").read_text())
    decoded = cv2.VideoCapture(str(composite))
    frames = int(decoded.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(decoded.get(cv2.CAP_PROP_FRAME_WIDTH))
    decoded.release()
    assert frames == len(composite_map["frames"]), (frames, len(composite_map["frames"]))
    # The canvas is two panels wide and they are NOT equal, so assert the contract
    # against the display's own split rather than against "even and positive",
    # which `cv2.VideoWriter` would have refused to open on anyway.
    from galbot_motion_studio.display import _fitted_panel_size

    panel_width, _panel_height = _fitted_panel_size()
    assert width == 2 * panel_width, (
        f"composite is {width} px wide; the canvas contract is 2 * {panel_width}"
    )


def test_the_raw_capture_and_sidecar_keep_every_detected_frame(
    tmp_path: Path, fake_camera
) -> None:
    """The evidence invariant threading could break silently.

    The UI takes only the newest detection, so the clip is shorter than the raw
    capture -- that is the point of latest-wins. The raw video, its frame map and
    the landmark sidecar must nonetheless agree with each other exactly.
    """
    raw = tmp_path / "raw.mp4"
    sidecar = tmp_path / "raw.landmarks.json"
    assert run_preview(
        tmp_path, "--source-video", str(raw), "--landmark-sidecar", str(sidecar)
    ) == 0

    frame_map = json.loads((tmp_path / "raw.frame-map.json").read_text())
    rows = frame_map["frames"]
    assert frame_map["video_frame_count"] == len(rows)
    assert [r["video_frame_index"] for r in rows] == list(range(len(rows)))
    sequences = [r["source_sequence"] for r in rows]
    assert sequences == sorted(sequences), "the frame map is out of capture order"
    assert len(set(sequences)) == len(sequences), "a frame was recorded twice"

    landmarks = json.loads(sidecar.read_text())["frames"]
    assert len(landmarks) == len(rows), (
        "the landmark sidecar and the raw frame map disagree on how many frames "
        "were detected -- the evidence hook missed one"
    )
    assert [r["source_sequence"] for r in landmarks] == sequences

    decoded = cv2.VideoCapture(str(raw))
    decoded_count = int(decoded.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded.release()
    assert decoded_count == len(rows), (decoded_count, len(rows))

    clip = json.loads((tmp_path / "live.json").read_text())
    commanded = {f["source_sequence"] for f in clip["frames"]}
    assert commanded, "no frame was commanded"
    assert commanded <= set(sequences), (
        "the clip references a frame the raw capture never recorded"
    )
