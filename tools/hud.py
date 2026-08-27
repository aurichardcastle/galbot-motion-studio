"""Local control HUD: run a simulation, watch it, browse what it stored.

    python tools/hud.py            # then open http://127.0.0.1:8765

Binds to loopback only and never to a routable interface. It executes this
project's own CLI as a subprocess; it is a control surface for work you could
already run by hand, not a new capability, and it is not a network service.

Simulation only, like everything else here: no Galbot SDK, no transport, no
hardware command path. The HUD cannot reach a robot because nothing in this
repository can.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
CAPTURES = ROOT / "artifacts" / "live-camera-input"
POSES = ROOT / "artifacts" / "hud-poses.jsonl"
JOB_LOGS = ROOT / "artifacts" / "hud-jobs"
CAMERA_CHOICE = ROOT / "artifacts" / "hud-camera.json"

#: Which capture device the capture job uses. Held in a dict so the JOBS table,
#: which is built once at import, reads the current value rather than baking in
#: whatever was selected at startup.
CAMERA: dict[str, object] = {"index": 0, "override": None, "why": ""}
if CAMERA_CHOICE.exists():
    try:
        CAMERA["override"] = json.loads(CAMERA_CHOICE.read_text()).get("override")
    except Exception:  # noqa: BLE001
        pass
PYTHON = sys.executable
HOST, PORT = "127.0.0.1", 8765

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The robot view is optional: the run/log/gate half of the HUD is useful on a
#: box that cannot open a GL context, so a failure here degrades the page
#: instead of taking the whole tool down.
try:
    from camera_select import builtin_camera_index, enumerate_cameras
    from hud_robot import RobotView

    ROBOT: "RobotView | None" = RobotView()
    ROBOT_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    ROBOT, ROBOT_ERROR = None, f"{type(error).__name__}: {error}"

#: Every job the HUD can start. Nothing else is runnable from the page: the
#: browser sends a job id and a capture name, never a command line, so the page
#: cannot be talked into executing something that is not on this list.
JOBS: dict[str, dict[str, object]] = {
    "capture": {
        "label": "New capture (webcam)",
        "blurb": "Opens the live preview. Press q in that window to finish. Needs camera access: start this HUD from a terminal that has it.",
        "args": lambda d: [
            "preview", "--camera", str(_capture_camera_index()), "--display",
            "--output", str(d / "live.json"),
            "--source-video", str(d / "raw.mp4"),
            "--landmark-sidecar", str(d / "raw.landmarks.json"),
        ],
        # A capture is irreplaceable: it is the only record of the operator
        # moving. Never overwrite one -- take the next free directory instead.
        "fresh": True,
    },
    "baseline": {
        "label": "Run baseline arm",
        "blurb": "Reprocesses the capture with the wrist-primary mapper.",
        "args": lambda d: _analysis_args(d, "wrist-primary", d / "baseline.json"),
        "fresh": False,
    },
    "candidate": {
        "label": "Run candidate arm",
        "blurb": "Reprocesses the same capture with the direction-vector mapper.",
        "args": lambda d: _analysis_args(d, "direction-vector", d / "candidate.json"),
        "fresh": False,
    },
    "analyze": {
        "label": "Analyze A/B + release gate",
        "blurb": "Scores both arms against the capture. Fails closed on provenance.",
        "args": lambda d: [
            "analyze-ab",
            "--baseline", str(d / "baseline.json"),
            "--candidate", str(d / "candidate.json"),
            "--source-video", str(d / "raw.mp4"),
            "--source-frame-map", str(d / "raw.frame-map.json"),
            "--landmark-sidecar", str(d / "raw.landmarks.json"),
            "--output", str(d / "ab.metrics.json"),
            "--force",
        ],
        "fresh": False,
    },
    "cameras": {
        "label": "Probe cameras",
        "blurb": "Lists which capture devices actually open. Opens no window.",
        "args": lambda d: [],
        "fresh": False,
        "script": "tools/probe_cameras.py",
    },
    "bench": {
        "label": "Bench both solvers (own vs borrowed)",
        "blurb": "Sweeps swivel targets through each solver and appends the run "
                 "to artifacts/solver-bench.jsonl. No capture needed.",
        "args": lambda d: [],
        "fresh": False,
        "script": "tools/solver_bench.py",
    },
    "demo": {
        "label": "Headless demo",
        "blurb": "Deterministic synthetic run. No camera needed.",
        "args": lambda d: ["demo", "--output", str(d / "demo")],
    },
}


#: Whether this process can actually open the camera. Probed once, in the
#: background, because macOS grants camera access per APPLICATION: a HUD started
#: from a sandbox or an editor inherits no grant and never will, however many
#: times the button is pressed. Knowing up front lets the page say so instead of
#: handing back a traceback.
CAMERA_OK: dict[str, object] = {"checked": False, "usable": False, "detail": ""}


def _probe_camera() -> None:
    index = _capture_camera_index()
    try:
        import cv2

        capture = cv2.VideoCapture(index)
        usable = bool(capture.isOpened()) and bool(capture.read()[0])
        capture.release()
    except Exception as error:  # noqa: BLE001
        usable, error_text = False, f"{type(error).__name__}: {error}"
    else:
        error_text = "" if usable else "macOS did not grant camera access to this process"
    CAMERA_OK.update(checked=True, usable=usable, detail=error_text, index=index)


def _capture_camera_index() -> int:
    """Built-in camera, resolved fresh each run.

    An explicit override wins; otherwise the device is chosen by TYPE rather
    than by position, because a Continuity Camera entering the list shifts every
    index below it and silently changes which physical camera "0" means.
    """
    if CAMERA.get("override") is not None:
        return int(CAMERA["override"])
    index, why = builtin_camera_index(default=CAMERA["index"])
    CAMERA["why"] = why
    return index


def _cameras() -> dict[str, object]:
    """Named capture devices, so the page can offer a real choice.

    OpenCV addresses cameras by index and cannot name them, and macOS will hand
    a nearby iPhone out as index 0 -- so an index on its own is not enough to
    know what you are about to record from.
    """
    import subprocess

    found, exact = enumerate_cameras()
    index, why = builtin_camera_index(default=CAMERA["index"])
    return {
        "devices": [
            {"index": d.index, "name": d.name, "model": d.model,
             "phone": d.phone, "builtin": d.builtin}
            for d in found
        ],
        "selected": CAMERA["override"] if CAMERA.get("override") is not None else index,
        "auto": CAMERA.get("override") is None,
        "exact": exact,
        "why": why,
        "usable": CAMERA_OK.get("usable"),
        "checked": CAMERA_OK.get("checked"),
        "detail": CAMERA_OK.get("detail"),
        "terminal_hint": (
            "cd " + str(ROOT) + " && PYTHONPATH=src "
            ".venv/bin/python tools/hud.py"
        ),
    }


def _has_capture(directory: Path) -> bool:
    return any((directory / name).exists() for name in ("live.json", "raw.mp4"))


def _next_free(directory: Path) -> Path:
    """First unused sibling: name-2, name-3, ... so no capture is ever clobbered."""
    for index in range(2, 200):
        candidate = directory.with_name(f"{directory.name}-{index}")
        if not _has_capture(candidate):
            return candidate
    return directory


def _analysis_args(directory: Path, mapping: str, output: Path) -> list[str]:
    frame_map = directory / "raw.frame-map.json"
    legacy_override: list[str] = []
    try:
        if json.loads(frame_map.read_text(encoding="utf-8")).get("schema_version") != 3:
            # The CLI records this override in every derived clip and refuses it
            # for A/B/export acceptance.  Keep the HUD useful for inspecting an
            # old local capture without silently treating it as current evidence.
            legacy_override = ["--allow-legacy-source"]
    except (OSError, ValueError, TypeError):
        # Let the CLI surface the authoritative malformed-map failure.
        pass
    return [
        "preview",
        "--video", str(directory / "raw.mp4"),
        "--analysis-sync",
        "--source-frame-map", str(frame_map),
        "--source-landmark-sidecar", str(directory / "raw.landmarks.json"),
        *legacy_override,
        "--arm-mapping", mapping,
        "--output", str(output),
        # Derived from the capture and exactly reproducible, so replacing an
        # earlier run is the intended behaviour rather than data loss.
        "--force",
    ]


class Runner:
    """One job at a time, with a bounded log the page can poll."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._log: deque[str] = deque(maxlen=400)
        self._job: str | None = None
        self._started: str | None = None
        self._exit: int | None = None
        self._logfile: Path | None = None
        self._sink = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self, job: str, directory: Path) -> tuple[bool, str]:
        requested = directory.name
        if job not in JOBS:
            return False, f"unknown job {job!r}"
        if self.busy:
            return False, "a job is already running"
        if job == "capture" and CAMERA_OK.get("checked") and not CAMERA_OK.get("usable"):
            return False, (
                "no camera access in this process. macOS grants it per application, "
                "so start the HUD from a terminal that has it."
            )
        if JOBS[job].get("fresh") and _has_capture(directory):
            directory = _next_free(directory)
        directory.mkdir(parents=True, exist_ok=True)
        args = JOBS[job]["args"](directory)  # type: ignore[operator]
        script = JOBS[job].get("script")
        command = (
            [PYTHON, str(ROOT / str(script)), "--label", directory.name]
            if script
            else [PYTHON, "-m", "galbot_motion_studio.cli", *args]
        )
        # Inherit the real environment. A stripped one breaks the webcam: the
        # capture needs the user's window-server session to open a preview window
        # and to sit under the same camera authorisation as the terminal that
        # started this HUD.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        # OpenCV's AVFoundation backend tries to raise the camera permission
        # prompt itself, which it can only do from the main thread. From a
        # subprocess it fails with "can not spin main run loop from other thread"
        # and then reports the camera as broken, which reads like a hardware
        # fault rather than a permissions one. Skipping its prompt makes it use
        # the grant the parent process already holds, and a genuine denial then
        # surfaces as a clean authorisation error instead.
        env.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
        with self._lock:
            self._log.clear()
            self._job, self._exit = job, None
            self._started = datetime.now(timezone.utc).strftime("%H:%M:%S")
            if directory.name != requested:
                self._log.append(
                    f"{requested} already holds a capture; writing to {directory.name}"
                )
            JOB_LOGS.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._logfile = JOB_LOGS / f"{stamp}-{job}.log"
            self._sink = self._logfile.open("w", encoding="utf-8")
            if job == "capture":
                self._sink.write(f"camera: {CAMERA.get('why') or 'default'}\n")
                self._log.append(f"camera: {CAMERA.get('why') or 'default'}")
            self._sink.write(f"$ {shlex.join(command)}\n")
            self._sink.flush()
            self._log.append(f"log: artifacts/hud-jobs/{self._logfile.name}")
            self._log.append(f"$ {shlex.join(command)}")
            self._process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
        threading.Thread(target=self._drain, daemon=True).start()
        return True, f"started {job}"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return False, "nothing running"
            self._process.terminate()
        return True, "terminate sent"

    def _drain(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            with self._lock:
                self._log.append(line.rstrip())
                if self._sink is not None:
                    self._sink.write(line)
                    self._sink.flush()
        code = process.wait()
        with self._lock:
            self._exit = code
            if code != 0 and any(
                "not authorized to capture video" in line
                or "camera failed to properly initialize" in line
                for line in self._log
            ):
                self._log.append("")
                self._log.append(
                    "macOS is refusing camera access to the app that launched this HUD. "
                    "Grant it in System Settings > Privacy & Security > Camera, then "
                    "restart the HUD from that same app. This is a permission the OS "
                    "only lets you grant yourself."
                )
            self._log.append(f"— exit {code} —")
            if self._sink is not None:
                self._sink.write(f"— exit {code} —\n")
                self._sink.close()
                self._sink = None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "busy": self._process is not None and self._process.poll() is None,
                "job": self._job,
                "started": self._started,
                "exit": self._exit,
                "log": list(self._log),
            }


RUNNER = Runner()


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _describe(directory: Path) -> dict[str, object]:
    """Everything the page needs about one capture, read fresh off disk."""
    files = []
    for name in ("raw.mp4", "raw.frame-map.json", "raw.landmarks.json",
                 "live.json", "baseline.json", "candidate.json", "ab.metrics.json"):
        path = directory / name
        files.append({
            "name": name,
            "present": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
            "mtime": (
                datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
                if path.exists() else ""
            ),
        })

    bound = {"video": None, "map": None, "sidecar": None, "consistent": None}
    video, frame_map = directory / "raw.mp4", directory / "raw.frame-map.json"
    if video.exists() and frame_map.exists():
        try:
            document = json.loads(frame_map.read_text())
            bound["video"] = _digest(video)
            bound["map"] = (document.get("video_sha256") or "")[:12]
            bound["consistent"] = bound["video"] == bound["map"]
            bound["frames"] = document.get("video_frame_count")
        except (OSError, json.JSONDecodeError, ValueError):
            bound["consistent"] = False

    gate: dict[str, object] | None = None
    metrics = directory / "ab.metrics.json"
    if metrics.exists():
        try:
            document = json.loads(metrics.read_text())
            release = document.get("release_gate", {})
            gate = {
                "verdict": release.get("verdict"),
                "promotion_allowed": release.get("promotion_allowed"),
                "failed": release.get("failed_checks", []),
                "checks": [
                    {"name": check.get("name"), "passed": bool(check.get("passed"))}
                    for check in release.get("checks", [])
                ],
            }
        except (OSError, json.JSONDecodeError):
            gate = None

    return {"name": directory.name, "files": files, "bound": bound, "gate": gate}


def _fill_default_axes(body: bytes) -> bytes:
    """Give bare ``<axis/>`` nodes the URDF default of ``1 0 0``.

    This robot has four joints whose axis element carries no xyz attribute. The
    URDF spec defaults that to (1, 0, 0); urdf-loader instead calls .split() on
    the missing attribute and dies mid-parse, which surfaces as the whole robot
    silently failing to load. Fixed on the way out rather than by editing the
    vendored loader or the vendored robot description, so neither third-party
    tree carries a local modification.
    """
    return re.sub(rb"<axis\s*/>", b'<axis xyz="1 0 0"/>', body)


def _home_degrees() -> tuple[dict[str, float], dict[str, float]]:
    """The teleop neutral pose, in degrees, for the page's HOME preset."""
    try:
        from galbot_motion_studio.pipeline import REST_ARM_QPOS, SIM_TELEOP_HOME_QPOS
    except Exception:  # noqa: BLE001
        return {}, {}
    import math

    def _deg(pose):
        return {n: round(math.degrees(float(v)), 2) for n, v in pose.items()}

    # REST is what the page opens on: arms down, the posture worth looking at.
    # NEUTRAL is the solver's seed, which is a different thing.
    return _deg({**SIM_TELEOP_HOME_QPOS, **REST_ARM_QPOS}), _deg(SIM_TELEOP_HOME_QPOS)


def _saved_pose_count() -> int:
    if not POSES.exists():
        return 0
    with POSES.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _save_pose(joints: dict[str, float], payload: dict[str, object]) -> dict[str, object]:
    """Append a posed configuration with its measurements, one JSON object a line.

    This is the data-gathering half of the HUD: every pose you stop on is a
    labelled sample of what the arm can actually do, recorded with the model hash
    so it stays attributable to the robot it was measured on.
    """
    assert ROBOT is not None
    ROBOT.apply(joints)
    from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST

    record = {
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": str(payload.get("label") or "").strip()[:120],
        "note": str(payload.get("note") or "").strip()[:500],
        "model_hash": CANONICAL_MANIFEST.fixed_mjcf_sha256,
        "source": "hud-manual-pose",
        "measurements": ROBOT.measurements(),
    }
    POSES.parent.mkdir(parents=True, exist_ok=True)
    with POSES.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return {"ok": True, "saved": _saved_pose_count(), "path": str(POSES.relative_to(ROOT))}


def _state() -> dict[str, object]:
    captures = []
    if CAPTURES.exists():
        for directory in sorted(
            (path for path in CAPTURES.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime, reverse=True,
        ):
            captures.append(_describe(directory))
    return {
        "captures": captures,
        "jobs": [{"id": key, "label": value["label"], "blurb": value["blurb"]}
                 for key, value in JOBS.items()],
        "runner": RUNNER.snapshot(),
    }


from hud_page import PAGE


class Handler(BaseHTTPRequestHandler):
    #: Keep-alive. A page load fetches ~190 meshes; without it each one opens a
    #: fresh connection and a reload can overrun the listen backlog, which the
    #: browser reports as a failed load rather than a slow one.
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, kind: str, *, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            # The robot description and the vendored libraries do not change while
            # the server runs. Re-downloading 190 meshes on every reload was the
            # reload failure: no-store on static assets turned a refresh into a
            # cold load of the entire robot.
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("ETag", f'"{sha256(body).hexdigest()[:16]}"')
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A reload cancels in-flight mesh requests. Normal, not an error.
            pass

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif route == "/api/state":
            self._send(200, json.dumps(_state()).encode(), "application/json")
        elif route == "/api/cameras":
            self._send(200, json.dumps(_cameras()).encode(), "application/json")
        elif route == "/api/robot/meta":
            payload: dict[str, object] = {"available": ROBOT is not None, "error": ROBOT_ERROR}
            if ROBOT is not None:
                payload["joints"] = ROBOT.describe_joints()
                rest, neutral = _home_degrees()
                payload["home_deg"] = rest
                payload["neutral_deg"] = neutral
                payload["saved"] = _saved_pose_count()
            self._send(200, json.dumps(payload).encode(), "application/json")
        elif route.startswith("/vendor/"):
            self._static(ROOT / "tools" / "vendor", route[len("/vendor/"):])
        elif route.startswith("/loaders/"):
            self._static(ROOT / "tools" / "vendor", Path(route).name)
        elif route.startswith("/utils/"):
            # three.js addons import siblings relatively: GLTFLoader.js asks for
            # ../utils/BufferGeometryUtils.js, which resolves above /vendor/.
            self._static(ROOT / "tools" / "vendor", Path(route).name)
        elif route.startswith("/robot/"):
            self._static(
                ROOT / "third_party" / "galbot_one_golf_description", route[len("/robot/"):]
            )
        else:
            self._send(404, b"not found", "text/plain")

    #: Only these are served, so a crafted path cannot pull arbitrary files off
    #: the machine even if it escapes the directory check below.
    _TYPES = {
        ".js": "text/javascript", ".urdf": "application/xml", ".xml": "application/xml",
        ".glb": "model/gltf-binary", ".gltf": "model/gltf+json", ".stl": "model/stl",
        ".obj": "text/plain", ".png": "image/png", ".jpg": "image/jpeg",
        ".bin": "application/octet-stream", ".md": "text/plain",
    }

    def _static(self, base: Path, relative: str) -> None:
        suffix = Path(relative).suffix.lower()
        if suffix not in self._TYPES:
            self._send(404, b"not found", "text/plain")
            return
        try:
            path = (base / relative).resolve()
            path.relative_to(base.resolve())          # refuse anything outside base
        except (ValueError, OSError):
            self._send(403, b"forbidden", "text/plain")
            return
        if not path.is_file() and base.name == "vendor":
            # three.js addons are published under loaders/ and utils/ but are
            # vendored flat, so fall back to the bare filename.
            flat = base / Path(relative).name
            if flat.is_file():
                path = flat
        if not path.is_file():
            self._send(404, b"not found", "text/plain")
            return
        body = path.read_bytes()
        if suffix == ".urdf":
            body = _fill_default_axes(body)
        if self.headers.get("If-None-Match") == f'"{sha256(body).hexdigest()[:16]}"':
            self.send_response(304)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            return
        self._send(200, body, self._TYPES[suffix], cache=True)

    def _json_body(self) -> dict[str, object] | None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return None

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/camera":
            payload = self._json_body() or {}
            try:
                index = int(payload.get("index", 0))
            except (TypeError, ValueError):
                self._send(400, b'{"error":"bad index"}', "application/json")
                return
            CAMERA["override"] = None if index < 0 else max(0, min(16, index))
            CAMERA["index"] = CAMERA["override"] or 0
            CAMERA_CHOICE.parent.mkdir(parents=True, exist_ok=True)
            CAMERA_CHOICE.write_text(json.dumps({"override": CAMERA["override"]}))
            self._send(200, json.dumps({"ok": True, "index": CAMERA["index"]}).encode(),
                       "application/json")
            return
        if route in ("/api/pose/save", "/api/clearance"):
            if ROBOT is None:
                self._send(503, json.dumps({"error": ROBOT_ERROR}).encode(), "application/json")
                return
            payload = self._json_body()
            if payload is None:
                self._send(400, b'{"error":"bad json"}', "application/json")
                return
            joints = {
                str(key): float(value)
                for key, value in dict(payload.get("joints") or {}).items()
                if isinstance(value, (int, float))
            }
            result = (
                ROBOT.clearance(joints)
                if route == "/api/clearance"
                else _save_pose(joints, payload)
            )
            self._send(200, json.dumps(result).encode(), "application/json")
            return
        if route == "/api/stop":
            ok, message = RUNNER.stop()
            self._send(200, json.dumps({"ok": ok, "message": message}).encode(), "application/json")
            return
        if route != "/api/run":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"ok":false,"message":"bad json"}', "application/json")
            return
        # The capture name is used to build a path, so it must be a single plain
        # directory name -- never a separator, never a parent reference.
        name = str(payload.get("capture") or "").strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            self._send(400, b'{"ok":false,"message":"bad capture name"}', "application/json")
            return
        ok, message = RUNNER.start(str(payload.get("job") or ""), CAPTURES / name)
        self._send(200, json.dumps({"ok": ok, "message": message}).encode(), "application/json")

    def log_message(self, *args: object) -> None:
        """Quiet: the page has its own log and this would drown the terminal."""


class _Server(ThreadingHTTPServer):
    #: A page load opens many sockets at once; the stdlib default of 5 is far too
    #: small and excess connections are refused outright.
    request_queue_size = 128
    #: Without this a restart inside the TIME_WAIT window fails with EADDRINUSE
    #: even though nothing is listening.
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    # A stale instance holding the port used to abort startup with a bare
    # OSError traceback, which reads like a bug in the tool. Say what is
    # happening and move to the next free port instead.
    threading.Thread(target=_probe_camera, daemon=True).start()
    server = None
    port = PORT
    for port in range(PORT, PORT + 12):
        try:
            server = _Server((HOST, port), Handler)
            break
        except OSError as error:
            if error.errno != 48:
                raise
            print(f"port {port} is in use, trying {port + 1}")
    if server is None:
        print(
            f"ports {PORT}-{PORT + 11} are all in use. Stop the other instance with:\n"
            f"    pkill -f tools/hud.py",
            file=sys.stderr,
        )
        return 2
    url = f"http://{HOST}:{port}"
    print(f"HUD on {url}  (loopback only, ctrl-c to stop)")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - a headless box is fine, the URL is printed
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
