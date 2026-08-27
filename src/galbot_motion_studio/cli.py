"""Command-line entry point for the hardware-disabled Motion Studio MVP."""

from __future__ import annotations

from dataclasses import asdict
from math import pi, sin

import argparse
import sys
from datetime import datetime
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from threading import Event, Lock
from time import monotonic_ns
from typing import Iterable
from uuid import uuid4

from galbot_motion_studio.adapters.mediapipe_holistic import (
    HolisticDetectorError,
    MediaPipeHolisticDetector,
)
from galbot_motion_studio.adapters.mujoco_preview import MujocoPreviewSink
from galbot_motion_studio.adapters.recorded_video import (
    RecordedVideoSource,
    SourceTimelineFrame,
)
from galbot_motion_studio.adapters.webcam import (
    WebcamError,
    WebcamSource,
    macos_cameras,
    resolve_builtin_camera,
    resolve_builtin_camera_avf,
)
from galbot_motion_studio.analysis_metrics import (
    AnalysisProvenanceError,
    analyze_ab,
    save_analysis,
)
from galbot_motion_studio.contracts.core import SafetyOutcome
from galbot_motion_studio.contracts.human import HumanObservation, IdentityState
from galbot_motion_studio.display import CalibrationStatus, LANDMARK_LABELS, StudioDisplay
from galbot_motion_studio.export_v21 import export_lerobot_v21
from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.offline import SynchronousProcessor
from galbot_motion_studio.pipeline import MotionStudioPipeline, SIM_TELEOP_START_QPOS
from galbot_motion_studio.ports.frames import CapturedFrame
from galbot_motion_studio.recording import (
    CalibrationWindowProvenance,
    LivenessProvenance,
    MotionClip,
    MotionRecorder,
    SourceReplayProvenance,
)
from galbot_motion_studio.realtime import (
    DECLINED,
    LatestWinsProcessor,
    RealtimeWorkerError,
)
from galbot_motion_studio.replay import replay_clip
from galbot_motion_studio.retarget.left_arm import ArmRetargetError
from galbot_motion_studio.retarget.torso import TORSO_YAW_JOINT
from galbot_motion_studio.safety.clearance import ClearanceChecker, HOME_QPOS
from galbot_motion_studio.safety.profiles import clearance_kwargs_for, policy_for
from galbot_motion_studio.safety.supervisor import SafetySupervisor
from galbot_motion_studio.synthetic import synthetic_observation
from galbot_motion_studio.vision.calibration import CalibrationError, CalibrationWindowPolicy
from galbot_motion_studio.vision.liveness import FrameLivenessMonitor, LivenessPolicy


JOINT_ORDER = (
    # Torso first: it carries both arm mounts and the head, so a reader of the
    # recorded row sees the frame the arm angles are relative to before it sees
    # the arm angles. Appending it instead would keep the old prefix stable, but
    # nothing consumes this order positionally across schema versions and a
    # kinematically ordered row is worth more than that.
    (TORSO_YAW_JOINT, "head_joint1", "head_joint2")
    + tuple(f"left_arm_joint{index}" for index in range(1, 8))
    + ("left_gripper_joint",)
    + tuple(f"right_arm_joint{index}" for index in range(1, 8))
    + ("right_gripper_joint",)
)

# A retained session is intentionally bounded. At the measured ~15 Hz live rate
# this permits roughly 33 minutes, while preventing recorder, frame-map, and
# landmark-sidecar lists from growing without limit during an unattended run.
MAX_SESSION_FRAMES = 30_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="galbot-motion-studio",
        description="Hardware-disabled webcam/recorded-video to Galbot MuJoCo studio",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preview = commands.add_parser("preview", help="capture, retarget, simulate, and record")
    source = preview.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument(
        "--webcam",
        "--camera",
        dest="webcam",
        type=_camera_device,
        metavar="DEVICE",
        help=(
            "camera index, or 'builtin' (also accepted: 'auto', 'facetime', "
            "'built-in') to resolve the first non-Continuity device by name. The "
            "argument is REQUIRED -- there is no implied default -- because a "
            "Continuity Camera can be handed out as index 0, so a bare index is "
            "not a stable way to mean 'the built-in webcam'."
        ),
    )
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--task", default="mirror head and both arms")
    preview.add_argument("--calibration-id", default="neutral-live-v1")
    preview.add_argument(
        "--calibration-confidence",
        type=float,
        default=CALIBRATION_MIN_CONFIDENCE,
        help="minimum per-landmark confidence required to lock calibration",
    )
    preview.add_argument(
        "--calibration-window-ms",
        type=int,
        metavar="MS",
        help="measured maximum capture-time span for a live neutral calibration window",
    )
    preview.add_argument(
        "--calibration-min-samples",
        type=int,
        metavar="N",
        help="measured minimum consecutive good samples for a live neutral window",
    )
    preview.add_argument(
        "--calibration-max-center-deviation-normalized",
        type=float,
        metavar="DELTA",
        help="measured maximum shoulder-centre deviation within the calibration window",
    )
    preview.add_argument(
        "--calibration-max-shoulder-width-deviation-normalized",
        type=float,
        metavar="DELTA",
        help="measured maximum shoulder-width deviation within the calibration window",
    )
    preview.add_argument(
        "--calibration-max-eye-span-deviation-normalized",
        type=float,
        metavar="DELTA",
        help="measured maximum eye-span deviation within the calibration window",
    )
    preview.add_argument("--max-frames", type=int)
    preview.add_argument(
        "--liveness-max-static-ms",
        type=int,
        metavar="MS",
        help=(
            "fail-closed camera-freeze detection budget; required for live webcams "
            "and deliberately has no implicit deployment default"
        ),
    )
    preview.add_argument(
        "--liveness-max-history",
        type=int,
        default=256,
        help="fingerprint replay-window size recorded with the liveness configuration",
    )
    preview.add_argument(
        "--source-frame-map",
        type=Path,
        help="exact rendered-video frame to original source-clock map for offline analysis",
    )
    preview.add_argument(
        "--analysis-sync",
        action="store_true",
        help="process every saved-video frame synchronously for reproducible analysis",
    )
    preview.add_argument(
        "--arm-mapping",
        choices=("wrist-primary", "direction-vector", "swivel-priority"),
        default="swivel-priority",
        help=(
            "arm-mapper selection (default swivel-priority). direction-vector is "
            "analysis-only; swivel-priority runs live and holds the same wrist "
            "target as wrist-primary while placing the elbow with the redundant "
            "DOF; wrist-primary is the legacy splayed-elbow mapper"
        ),
    )
    preview.add_argument("--display", action="store_true")
    preview.add_argument(
        "--fullscreen",
        action="store_true",
        help="open the preview window fullscreen (implies --display)",
    )
    preview.add_argument(
        "--mirror-camera",
        action="store_true",
        help=(
            "show the operator a mirrored selfie view. Mirrors BOTH panels so "
            "they keep agreeing about which side is which -- which also mirrors "
            "the robot's chassis, so the GALBOT wordmark reads backwards. Off by "
            "default for that reason; useful when operating alone."
        ),
    )
    preview.add_argument(
        "--preview-video",
        type=Path,
        help="save the human-skeleton + robot digital-twin composite as MP4",
    )
    preview.add_argument(
        "--source-video",
        type=Path,
        help="save raw webcam frames plus a cryptographically bound source-time map",
    )
    preview.add_argument(
        "--landmark-sidecar",
        type=Path,
        help="save raw shoulder/elbow/wrist/hip landmarks beside --source-video for A/B metrics",
    )
    preview.add_argument(
        "--source-landmark-sidecar",
        type=Path,
        help="cryptographically bound raw-landmark sidecar for --video A/B analysis",
    )
    preview.add_argument(
        "--allow-failed-source",
        action="store_true",
        help=(
            "permit diagnostic replay of a raw source explicitly marked as a failed "
            "capture; it is never G5/G6 evidence"
        ),
    )
    preview.add_argument(
        "--allow-legacy-source",
        action="store_true",
        help=(
            "permit diagnostic replay of a pre-v3 source map with no capture-outcome "
            "attestation; it is never G5/G6 or export evidence"
        ),
    )
    preview.add_argument("--preview-fps", type=float, default=30.0)
    preview.add_argument("--force", action="store_true")

    replay = commands.add_parser("replay", help="re-run a clip through fresh safety checks")
    replay.add_argument("clip", type=Path)

    export = commands.add_parser("export-v21", help="write a checked LeRobot v2.1 dataset")
    export.add_argument("clips", type=Path, nargs="+")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--fps", type=int, default=30)

    analyze = commands.add_parser(
        "analyze-ab",
        help="fail-closed source-aligned offline metrics for two recorded mappings",
    )
    analyze.add_argument("--baseline", type=Path, required=True)
    analyze.add_argument("--candidate", type=Path, required=True)
    analyze.add_argument("--source-video", type=Path, required=True)
    analyze.add_argument("--source-frame-map", type=Path, required=True)
    analyze.add_argument("--landmark-sidecar", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--force", action="store_true")

    demo = commands.add_parser("demo", help="run the complete deterministic headless demo")
    demo.add_argument("--output", type=Path)
    return parser


def _camera_help(device: int | None) -> str:
    """What to actually do about an unavailable camera, in order of likelihood.

    Every line here is a failure that has really happened on this project, and
    the order is how often. Written out in full because the moment it appears is
    the moment nobody wants to be reading source.
    """
    index = 0 if device is None else device
    return "\n".join(
        (
            "",
            "  Most likely, in order:",
            "  1. Another process still holds the camera. A previous preview or",
            "     HUD run that did not exit cleanly is the usual culprit:",
            "         pkill -f galbot_motion_studio.cli",
            "         pkill -f tools/hud.py",
            "  2. This terminal has no camera permission. macOS grants it per",
            "     APP, and child processes inherit from the terminal. Check",
            "     System Settings > Privacy & Security > Camera, then restart",
            "     the terminal app -- a running app does not pick up a new grant.",
            f"  3. Device {index} is not the camera you meant. List what is present:",
            "         PYTHONPATH=src python tools/probe_cameras.py",
            "     then pass the right one with --camera N.",
            "",
            "  The simulation path does not need a camera at all:",
            "      python -m galbot_motion_studio.cli demo --output /tmp/demo",
            "",
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preview":
        try:
            return _preview(args)
        except WebcamError as error:
            # A camera that is busy, unplugged or not yet permitted is an
            # operating condition, not a defect, and it happens in front of an
            # audience. Print what to DO about it instead of a traceback.
            print(f"\ncamera unavailable: {error}", file=sys.stderr)
            print(_camera_help(getattr(args, "webcam", None)), file=sys.stderr)
            return 2
    if args.command == "replay":
        return _replay(args.clip)
    if args.command == "export-v21":
        export_lerobot_v21(
            (MotionClip.load(path) for path in args.clips),
            args.output,
            fps=args.fps,
        )
        print(f"exported {len(args.clips)} episode(s) to {args.output}")
        return 0
    if args.command == "analyze-ab":
        return _analyze_ab(args)
    if args.command == "demo":
        return _demo(args.output)
    raise AssertionError(args.command)


def _analyze_ab(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    try:
        document = analyze_ab(
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            source_video_path=args.source_video,
            source_frame_map_path=args.source_frame_map,
            landmark_sidecar_path=args.landmark_sidecar,
        )
    except AnalysisProvenanceError as error:
        raise SystemExit(f"A/B analysis rejected: {error}") from error
    save_analysis(document, args.output)
    frame_count = document["provenance"]["source_command_frames"]
    gate = document["release_gate"]
    assert isinstance(gate, dict)
    verdict = gate["verdict"]
    print(
        f"saved source-aligned A/B metrics for {frame_count} command frames to {args.output}; "
        f"release gate: {verdict}"
    )
    # A failed candidate is a valid measurement, so the JSON must still be
    # saved.  Exit 2 makes a production pipeline fail closed without erasing the
    # audit trail that explains why.
    return 0 if verdict == "PASS" else 2


#: Confidence required to LOCK CALIBRATION (not the runtime safety gate).
#:
#: `aggregate_confidence` is a minimum across command-critical landmarks: both
#: wrists, both shoulders and three face points. Elbows are used as a soft IK hint
#: and checked independently at 0.3 because MediaPipe routinely marks a correctly
#: tracked elbow below 0.5 when a forearm crosses the face or torso.
#:
#: This was 0.75 and is empirically unreachable: measured over 1,785 consecutive
#: frames of a correctly-tracked person on an M1 Max, the minimum peaked at 0.66 and
#: calibration never locked, while the drawn skeleton was visibly correct throughout.
#: 0.5 matches MediaPipe's own default detection/presence/tracking confidence.
#:
#: This lowers the bar for STARTING a session, not for safety during one: all nine
#: landmarks must still be present and the tracked identity must still be STABLE,
#: and the runtime FreshnessPolicy gate is unchanged.
CALIBRATION_MIN_CONFIDENCE = 0.5


def _preview(args: argparse.Namespace) -> int:
    resolved_camera_label = ""
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    # Reject before constructing WebcamSource/OpenCV. On macOS, passing a
    # negative index through AVFoundation can abort the interpreter while the
    # native capture object is released instead of reporting an unopened device.
    if (
        args.webcam is not None
        and not isinstance(args.webcam, str)
        and args.webcam < 0
    ):
        raise SystemExit("--camera must be a non-negative device index")
    if args.analysis_sync and args.video is None:
        raise SystemExit("--analysis-sync requires --video")
    if args.source_frame_map is not None and args.video is None:
        raise SystemExit("--source-frame-map requires --video")
    if args.source_video is not None and args.video is not None:
        raise SystemExit("--source-video is for a new webcam capture, not --video replay")
    if args.source_video is not None and args.landmark_sidecar is None:
        raise SystemExit(
            "--source-video requires --landmark-sidecar; raw A/B input is a bound "
            "video/map/landmark set"
        )
    if args.landmark_sidecar is not None and args.source_video is None:
        raise SystemExit("--landmark-sidecar requires --source-video")
    if (
        args.landmark_sidecar is not None
        and not args.landmark_sidecar.name.endswith(".landmarks.json")
    ):
        raise SystemExit(
            "--landmark-sidecar must end in .landmarks.json so private body data "
            "is excluded by the repository ignore policy"
        )
    if args.source_landmark_sidecar is not None and args.video is None:
        raise SystemExit("--source-landmark-sidecar requires --video")
    if args.source_landmark_sidecar is not None and args.source_frame_map is None:
        # This used to be a bare `assert` deep inside _preview, so the operator got
        # a traceback instead of the sentence. The sidecar is bound to the frame
        # map by hash; without one there is nothing to validate it against.
        raise SystemExit(
            "--source-landmark-sidecar requires --source-frame-map; the sidecar is "
            "bound to the map by hash and cannot be checked without it"
        )
    if args.allow_failed_source and (args.video is None or args.source_frame_map is None):
        raise SystemExit("--allow-failed-source requires --video and --source-frame-map")
    if args.allow_legacy_source and (args.video is None or args.source_frame_map is None):
        raise SystemExit("--allow-legacy-source requires --video and --source-frame-map")
    if args.allow_failed_source and args.allow_legacy_source:
        raise SystemExit("choose only one of --allow-failed-source or --allow-legacy-source")
    if args.fullscreen:
        # `--fullscreen` has always advertised "implies --display", and four
        # separate sites re-derived `args.display or args.fullscreen` instead. Do
        # it once, here, so the implication is a fact rather than a convention.
        args.display = True
    if args.mirror_camera and not (args.display or args.preview_video is not None):
        # A presentation-only flag with no surface to present on is a silent
        # no-op, and this one is reached for under demo pressure ("the mirror is
        # the wrong way round") where a silent no-op reads as a broken flag.
        raise SystemExit(
            "--mirror-camera requires --display, --fullscreen or --preview-video; "
            "it changes only what is drawn"
        )
    if args.arm_mapping == "direction-vector" and not args.analysis_sync:
        raise SystemExit("direction-vector mapping is available only with --analysis-sync")
    if args.arm_mapping == "direction-vector" and args.source_frame_map is None:
        raise SystemExit(
            "direction-vector analysis requires --source-frame-map; "
            "unmapped video is not publishable A/B evidence"
        )
    if args.max_frames is not None and args.max_frames > MAX_SESSION_FRAMES:
        raise SystemExit(
            f"--max-frames cannot exceed the bounded session limit {MAX_SESSION_FRAMES}"
        )
    if args.liveness_max_static_ms is not None and args.liveness_max_static_ms <= 0:
        raise SystemExit("--liveness-max-static-ms must be positive")
    if args.liveness_max_history < 1:
        raise SystemExit("--liveness-max-history must be at least 1")
    if args.webcam is not None and args.liveness_max_static_ms is None:
        raise SystemExit(
            "--liveness-max-static-ms is required for a live webcam; measure and "
            "approve the camera-freeze detection budget before preview"
        )
    calibration_values = (
        args.calibration_window_ms,
        args.calibration_min_samples,
        args.calibration_max_center_deviation_normalized,
        args.calibration_max_shoulder_width_deviation_normalized,
        args.calibration_max_eye_span_deviation_normalized,
    )
    if any(value is not None for value in calibration_values) and any(
        value is None for value in calibration_values
    ):
        raise SystemExit(
            "calibration window configuration is all-or-nothing; provide its "
            "duration, sample count, and all three measured deviation limits"
        )
    if args.webcam is not None and all(value is None for value in calibration_values):
        raise SystemExit(
            "a complete calibration-window policy is required for a live webcam; "
            "measure and approve all limits before preview"
        )
    if args.arm_mapping == "direction-vector" and args.source_landmark_sidecar is None:
        raise SystemExit(
            "direction-vector analysis requires --source-landmark-sidecar; "
            "raw landmarks are required A/B evidence"
        )
    requested_outputs = tuple(
        output
        for output in (
            args.output,
            args.preview_video,
            args.source_video,
            args.landmark_sidecar,
        )
        if output is not None
    )
    if len({output.resolve() for output in requested_outputs}) != len(requested_outputs):
        raise SystemExit("output, preview, raw video, and landmark sidecar paths must differ")
    generated_maps = tuple(
        video.with_suffix(".frame-map.json")
        for video in (args.preview_video, args.source_video)
        if video is not None
    )
    all_artifacts = requested_outputs + generated_maps
    if len({output.resolve() for output in all_artifacts}) != len(all_artifacts):
        raise SystemExit("capture artifact paths must not overlap generated frame maps")
    for output in all_artifacts:
        if output is not None and output.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {output}; pass --force")
    analysis_input_hash = (
        _analysis_identity(args.video, args.source_frame_map)
        if args.analysis_sync
        else None
    )
    source_clock_id = "recorded-video-clock" if args.video else "local-monotonic"
    frames: Iterable[CapturedFrame]
    source_replay_provenance = SourceReplayProvenance(origin="live-capture")
    if args.video:
        source_replay_provenance = SourceReplayProvenance(
            origin="recorded-video-replay",
            enabled=True,
            capture_outcome="unmapped-recorded-video",
        )
        timeline = (
            _load_source_frame_map(
                args.source_frame_map,
                args.video,
                required_artifact_kind=(
                    "raw-camera" if args.arm_mapping == "direction-vector" else None
                ),
                allow_failed_source=args.allow_failed_source,
                allow_legacy_source=args.allow_legacy_source,
            )
            if args.source_frame_map is not None
            else None
        )
        if args.source_frame_map is not None:
            source_replay_provenance = _source_replay_provenance(
                args.source_frame_map,
                allow_failed_source=args.allow_failed_source,
                allow_legacy_source=args.allow_legacy_source,
            )
        if args.source_landmark_sidecar is not None:
            _load_landmark_sidecar(
                args.source_landmark_sidecar,
                video=args.video,
                frame_map=args.source_frame_map,
                timeline=timeline,
            )
        frames = RecordedVideoSource(
            args.video, source_clock_id=source_clock_id, source_timeline=timeline
        ).frames()
    else:
        # A Continuity Camera (iPhone/iPad) can be offered as index 0, so an index
        # alone does not reliably mean "the built-in webcam". Resolve by name when
        # asked, and ALWAYS say which device was chosen -- an operator who cannot
        # see this has no way to tell a phone hand-off from the built-in camera.
        # AVFoundation first: it enumerates in the SAME order OpenCV indexes.
        # system_profiler does not -- resolving from it picked a Continuity iPhone
        # while claiming the built-in camera (observed 2026-08-26).
        _avf_index, _avf_name, _avf_devices = resolve_builtin_camera_avf()
        if _avf_devices:
            print("cameras (AVFoundation order, as OpenCV indexes them):")
            for _n, _dev in enumerate(_avf_devices):
                _kind = (
                    "built-in"
                    if _dev.get("device_type")
                    == "AVCaptureDeviceTypeBuiltInWideAngleCamera"
                    else "EXTERNAL/CONTINUITY (phone?)"
                )
                print(f"  {_n}  {_dev.get('name','?'):<28} {_kind}")
        if args.webcam == "builtin":
            if _avf_index is None:
                _fallback, _description = resolve_builtin_camera(macos_cameras())
                if _fallback is None:
                    raise SystemExit(
                        f"--camera builtin could not resolve a built-in camera: "
                        f"{_avf_name}. Run tools/probe_cameras.py and pass an index."
                    )
                # Reached only without pyobjc (the `camera-select` extra), and
                # this is the resolver the comment above records picking a
                # Continuity iPhone while calling it the built-in camera. Use it
                # rather than refuse the session, but never print the confident
                # word: on this path the ordering is genuinely unverified.
                _avf_index, _avf_name = _fallback, _description
                args.webcam = _avf_index
                print(
                    f"camera: using index {_avf_index} -> {_avf_name} "
                    "*** ORDERING UNVERIFIED: install the 'camera-select' extra "
                    "(pyobjc-framework-AVFoundation) to resolve by AVFoundation, "
                    "or pass an explicit index ***"
                )
                resolved_camera_label = _avf_name
            else:
                args.webcam = _avf_index
                print(f"camera: using index {_avf_index} -> {_avf_name} (built-in)")
                resolved_camera_label = _avf_name
        elif _avf_devices and 0 <= args.webcam < len(_avf_devices):
            _chosen = _avf_devices[args.webcam]
            if _chosen.get("device_type") != "AVCaptureDeviceTypeBuiltInWideAngleCamera":
                print(
                    f"camera: index {args.webcam} -> {_chosen.get('name','?')} "
                    "*** NOT the built-in camera *** pass --camera builtin instead"
                )
            else:
                print(f"camera: using index {args.webcam} -> {_chosen.get('name','?')}")
            resolved_camera_label = str(_chosen.get("name", "?"))
        frames = WebcamSource(args.webcam, source_clock_id=source_clock_id).frames(
            max_frames=args.max_frames
        )
    liveness_policy = (
        None
        if args.liveness_max_static_ms is None
        else LivenessPolicy(
            max_static_ns=args.liveness_max_static_ms * 1_000_000,
            max_history=args.liveness_max_history,
        )
    )
    if calibration_values[0] is None:
        calibration_window_policy = None
    else:
        try:
            calibration_window_policy = CalibrationWindowPolicy(
                min_samples=args.calibration_min_samples,
                max_window_ns=args.calibration_window_ms * 1_000_000,
                max_center_deviation_normalized=(
                    args.calibration_max_center_deviation_normalized
                ),
                max_shoulder_width_deviation_normalized=(
                    args.calibration_max_shoulder_width_deviation_normalized
                ),
                max_eye_span_deviation_normalized=(
                    args.calibration_max_eye_span_deviation_normalized
                ),
            )
        except ValueError as error:
            raise SystemExit(f"invalid calibration-window policy: {error}") from error
    liveness_provenance = LivenessProvenance.from_policy(
        liveness_policy,
        # This CLI's only capture adapters leave ``content_fingerprint`` unset,
        # so MediaPipe hashes their raw BGR pixels. A different producer must
        # name its own fingerprint algorithm rather than inheriting this claim.
        fingerprint_algorithm=(
            None if liveness_policy is None else "sha256-bgr-pixels"
        ),
    )
    calibration_window_provenance = CalibrationWindowProvenance.from_policy(
        calibration_window_policy
    )
    pipeline = MotionStudioPipeline(
        source_clock_id=source_clock_id,
        arm_mapping=args.arm_mapping,
        liveness_monitor=(
            None if liveness_policy is None else FrameLivenessMonitor(liveness_policy)
        ),
    )
    capture_attempt_id = uuid4().hex if args.source_video is not None else None
    capture_provenance = (
        {
            "attempt_id": capture_attempt_id,
            "mapping_hash": pipeline.mapping_hash,
            "arm_mapping": args.arm_mapping,
            "liveness": liveness_provenance.model_dump(mode="json"),
            "calibration_window": calibration_window_provenance.model_dump(mode="json"),
        }
        if capture_attempt_id is not None
        else None
    )
    analysis_artifact_hash = (
        _analysis_artifact_identity(
            input_hash=analysis_input_hash,
            mapping_hash=pipeline.mapping_hash,
            arm_mapping=args.arm_mapping,
            calibration_confidence=args.calibration_confidence,
            calibration_window=calibration_window_policy,
            max_frames=args.max_frames,
            liveness=liveness_provenance,
        )
        if analysis_input_hash
        else None
    )
    recorder = MotionRecorder(
        task=args.task,
        calibration_id=args.calibration_id,
        joint_order=JOINT_ORDER,
        motion_profile=pipeline.motion_profile.value,
        # A normal recording needs a fresh UUID.  A saved-video analysis needs a
        # stable identifier so equal input/config/output runs can be compared
        # byte-for-byte rather than merely after stripping metadata.
        clip_id=(f"analysis-{analysis_artifact_hash}" if analysis_artifact_hash else None),
        analysis_arm_generation=(
            f"analysis-{analysis_artifact_hash[:32]}" if analysis_artifact_hash else None
        ),
        input_hash=analysis_input_hash,
        implementation_hash=_implementation_hash() if analysis_input_hash else None,
        liveness=liveness_provenance,
        source_replay=source_replay_provenance,
        calibration_window=calibration_window_provenance,
    )
    if args.preview_fps <= 0:
        raise SystemExit("--preview-fps must be positive")
    studio = (
        StudioDisplay(pipeline.model)
        if args.display or args.preview_video is not None
        else None
    )
    if studio is not None:
        studio.fullscreen = bool(args.fullscreen)
        studio.mirror_camera = bool(args.mirror_camera)
        if resolved_camera_label:
            studio.camera_label = resolved_camera_label
    video_writer = None
    source_video_writer = None
    source_video_staging = (
        _staging_path(args.source_video) if args.source_video is not None else None
    )
    source_map_path = (
        args.source_video.with_suffix(".frame-map.json")
        if args.source_video is not None
        else None
    )
    recording_output_staging = (
        _staging_path(args.output) if args.source_video is not None else args.output
    )
    # A composite frame is not necessarily every decoded source frame: live mode
    # deliberately drops stale work, and calibration frames have no command.  Keep
    # the exact rendered-frame -> source-command identity beside every new preview
    # video so a future A/B never has to guess from MP4 frame ordinal or FPS.
    preview_frame_manifest: list[dict[str, int]] = []
    source_frame_manifest: list[dict[str, int]] = []
    landmark_sidecar_rows: list[dict[str, object]] = []
    calibrated = False
    hold_counter = {"n": 0}
    calib_frames = 0
    calibration_samples: list[HumanObservation] = []
    processed = 0
    runner = None
    stop_requested = False
    perception_fault_reason: str | None = None
    # Set by whichever thread first produces a FAULT, read by both. See
    # process_work.
    fault_latched = Event()
    # `recorder.append` is written from two places -- the control worker for every
    # solved frame, the capture loop for a perception ingress fault -- and it is
    # check-then-act on the last recorded frame and on the terminal-fault slot.
    # Interleaving them appends out of order, which `MotionClip`'s
    # strictly-increasing validator rejects at save time: the recording is lost at
    # the very end, which is how a real 1196-frame session was lost once already.
    #
    # What actually makes the sequence monotonic today is the `close(drain=True)`
    # in the ingress handler: it JOINS the worker, so by the time the capture loop
    # appends its fault the other writer no longer exists. This lock does not earn
    # its keep against that ordering -- it is kept as the cheap, uncontended guard
    # that survives someone moving or removing the join, which is exactly the edit
    # that would otherwise reintroduce the lost-recording bug silently.
    recorder_lock = Lock()

    def process_work(work):
        """Solve one observation on the control worker -- and latch a FAULT HERE.

        This function is where a frame is acted on: `process_fail_closed` submits
        the approved target to the sink, and `recorder.append` puts the frame into
        the clip that gets published. So this is also where a latched FAULT has to
        stop, and the reason is a race, not a preference. `present` runs on the
        capture loop and only sees an outcome once `poll_latest` hands it back;
        the latest-wins pipeline is two deep, so by then the next two frames have
        already been solved, commanded and recorded. Measured with the latch in
        `present` alone: two trailing ALLOWs reached the sink and the published
        clip after a latched FAULT, in every live mode, exit 0.

        The worker is a single thread, so a flag set at the end of one call is
        already visible at the start of the next: the stop is exact, not
        best-effort. `--analysis-sync` runs this same function inline, which is
        why one latch covers both.
        """
        observation, _frame = work
        if fault_latched.is_set():
            # Work that was already in flight when the fault latched. Nothing is
            # solved, commanded or recorded for it. DECLINED rather than a real
            # result, so the worker publishes nothing: the results queue holds one
            # slot and overwrites, and the one result that must survive it is the
            # FAULT the operator's window still has to show.
            return DECLINED
        now_ns = (
            monotonic_ns()
            if _frame.source_kind.startswith("webcam:")
            else max(observation.inference_complete_mono_ns, observation.capture_mono_ns)
        )
        result = pipeline.process_fail_closed(observation, now_mono_ns=now_ns)
        if result.decision.outcome is SafetyOutcome.FAULT:
            fault_latched.set()
            print(
                "preview entered FAULT; recording stopped: "
                f"{list(result.decision.reasons)[:2]}"
            )
        with recorder_lock:
            recorder.append(result)
        return result

    # Overlay state from the most recent completed solve.  The screen repaints at
    # CAPTURE cadence (see repaint below), which is 2-3x the solve cadence, so the
    # labels drawn between solves are the last decided ones -- which is the truth:
    # the twin's pose has not changed either.
    view: dict = {}

    def present(timed):
        nonlocal processed, stop_requested
        result = timed.result
        observation = timed.item[0]
        processed += 1
        status = result.decision.outcome.value
        clearance = None if result.target is None else result.target.predicted_clearance_m
        print(
            f"frame={result.observation_sequence} status={status} "
            f"processing={timed.processing_ms:.1f}ms queue={timed.queue_latency_ms:.1f}ms"
            + ("" if clearance is None else f" clearance={clearance:.4f}")
        )
        # HOLD is normal, not fatal. docs/safety-model.md is explicit about this --
        # P1 "no person in frame" is marked *"Normal, not exceptional. Must be quiet
        # in the UI or operators learn to ignore alerts."* A HOLD means "do not send
        # a target for THIS frame": blink, turn away, move a wrist out of shot, or
        # let one landmark drop below confidence and you get one. Ending the session
        # on the first one made a live preview survive exactly one frame.
        #
        # FAULT is the latched state and does stop the session, which is correct:
        # it requires human inspection and an explicit reset.
        #
        # The HOLD count lives in `present`, once per SOLVE, and deliberately not
        # in `repaint`: `repaint` returns immediately when there is no window, so
        # on a headless run -- every gate in `close_gates.sh`, every `analyze-ab`
        # input from `tools/hud.py`, any `--video` replay -- the counter would
        # never advance. Once per solve rather than once per repaint also means it
        # counts holds in the retained terminal log, not screen refreshes.
        #
        # FAULT is latched and announced by `process_work`, which is the only
        # place that can stop the session before the next frame is commanded. All
        # that is left here is ending the capture loop.
        if result.decision.outcome is SafetyOutcome.FAULT:
            stop_requested = True
        elif result.decision.outcome is SafetyOutcome.HOLD:
            hold_counter["n"] += 1
            # Report the first HOLD and then every 30th, so a persistent problem is
            # visible without a wall of text for ordinary transient ones.
            if hold_counter["n"] == 1 or hold_counter["n"] % 30 == 0:
                print(
                    f"HOLD x{hold_counter['n']} (not fatal, still tracking): "
                    f"{list(result.decision.reasons)[:2]}"
                )
        if studio is not None:
            metrics = runner.metrics
            view.update(
                pending_video=True,
                status=status,
                observation=observation,
                sequence=result.observation_sequence,
                telemetry=(
                    f"{metrics.effective_fps:.1f} FPS | process "
                    f"{timed.processing_ms:.0f} ms | "
                    f"dropped {metrics.dropped_before_processing}"
                ),
                result=result,
            )

    def repaint(frame, observation):
        """Draw the window from the CURRENT camera frame at capture cadence.

        Previously this ran only when a solve landed, so the whole window --
        including the operator's own mirror -- repainted at the solve rate
        (measured 4.83 fps against a 15 fps camera) and showed the camera frame
        that solve had consumed, mean 191.7 ms stale.  Presentation only: no
        command is produced, gated or modified here, and the robot panel still
        shows whichever supervisor-approved pose the sink currently holds.
        """
        nonlocal stop_requested, video_writer
        if studio is None or not view:
            return
        result = view["result"]
        status = view["status"]
        telemetry = view["telemetry"]
        sink = pipeline.sink
        if not isinstance(sink, MujocoPreviewSink):
            raise TypeError("display requires the MuJoCo preview sink")
        with sink.lock:
            # A HOLD frame carries no readback, but the twin still HAS a
            # pose: read it so the per-joint activity bars fall to zero
            # (which is the truth) rather than going blank.
            joints = result.observed_joints or sink.joint_positions()
            canvas = studio.render(
                frame.image_bgr,
                sink.data,
                status=status,
                telemetry=telemetry,
                observation=observation,
                held_groups=result.held_groups,
                held_grippers=result.held_grippers,
                saturated_groups=result.saturated_groups,
                # A partial hold can still be a supervisor ALLOW, so its
                # reason does not live in SafetyDecision.  Put it on the
                # canvas alongside global safety reasons; otherwise a
                # stopped tummy/arm is indistinguishable from a deliberate
                # still pose during the live demo.
                reasons=(
                    *result.decision.reasons,
                    *(
                        f"{group}: {reason}"
                        for group, reason in result.held_group_reasons
                    ),
                    *(
                        f"{gripper}: {reason}"
                        for gripper, reason in result.held_gripper_reasons
                    ),
                    *(
                        f"{group}: {reason}"
                        for group, reason in result.saturated_group_reasons
                    ),
                ),
                joints=joints,
                # The live safety margin, on the twin's own panel. It is the
                # number that says the supervisor is actually running, and it
                # was previously only visible by reading the terminal behind
                # the window.
                clearance_m=(
                    None if result.target is None
                    else result.target.predicted_clearance_m
                ),
            )
        if args.preview_video is not None and view.get("pending_video"):
            view["pending_video"] = False
            if video_writer is None:
                import cv2

                args.preview_video.parent.mkdir(parents=True, exist_ok=True)
                video_writer = cv2.VideoWriter(
                    str(args.preview_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.preview_fps,
                    (int(canvas.shape[1]), int(canvas.shape[0])),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(
                        f"could not open preview video: {args.preview_video}"
                    )
            video_writer.write(canvas)
            # Keyed to the observation that produced the POSE, not the newer
            # camera image now drawn beside it.  The composite is an operator
            # review aid; the runbook already states it is not a replay source
            # or a latency proof.  The raw triple remains the source of truth.
            preview_frame_manifest.append(
                {
                    "video_frame_index": len(preview_frame_manifest),
                    "source_sequence": view["sequence"],
                    "source_mono_ns": view["observation"].capture_mono_ns,
                }
            )
        if args.display and not studio.show_canvas(canvas):
            stop_requested = True

    try:
        with MediaPipeHolisticDetector() as detector:
            for frame in frames:
                if (
                    args.source_video is not None
                    and len(source_frame_manifest) >= MAX_SESSION_FRAMES
                ):
                    print(
                        f"raw capture reached the bounded session limit "
                        f"({MAX_SESSION_FRAMES} frames); saving cleanly"
                    )
                    break
                if args.source_video is not None:
                    if source_video_writer is None:
                        import cv2

                        assert source_video_staging is not None
                        source_video_staging.parent.mkdir(parents=True, exist_ok=True)
                        source_video_writer = cv2.VideoWriter(
                            str(source_video_staging),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.preview_fps,
                            (int(frame.image_bgr.shape[1]), int(frame.image_bgr.shape[0])),
                        )
                        if not source_video_writer.isOpened():
                            raise RuntimeError(
                                f"could not open source video staging file: {args.source_video}"
                            )
                    source_video_writer.write(frame.image_bgr)
                    source_frame_manifest.append(
                        {
                            "video_frame_index": len(source_frame_manifest),
                            "source_sequence": frame.sequence,
                            "source_mono_ns": frame.capture_mono_ns,
                        }
                    )
                # Detection stays on this thread deliberately.  Moving it to its own
                # thread so it could overlap `cv2.waitKey` was built, measured, and
                # rejected: MediaPipe then runs at the full camera rate instead of
                # the loop rate, and the CPU it takes comes straight out of the
                # control worker.  Three paired runs against a paced 30 fps replay,
                # display on -- photon-to-twin-pose mean 97.5 / 100.4 / 97.7 ms
                # synchronous against 208.8 / 213.5 / 237.1 ms threaded, with the
                # worker's own mean rising 44-47 ms -> 126-149 ms.  The bottleneck
                # here is the control worker, and starving it to feed perception
                # more frames makes the operator wait longer, not less.
                #
                # Those figures come from a WINDOWED harness with equal panels and
                # no recording flags. They are not comparable with the fullscreen
                # numbers in docs/operating-guide.md, which are a different
                # configuration measured a different way; the A/B above is only
                # valid against itself.
                #
                # The implementation is gone, so this cannot be re-run from the
                # tree: the run conditions, the harness and the full numbers are
                # in the project validation notes, where the threaded variant
                # was measured and rejected.
                try:
                    observation = detector.detect(frame, calibration_id=args.calibration_id)
                except HolisticDetectorError as error:
                    perception_fault_reason = str(error)
                    if args.landmark_sidecar is not None:
                        # The raw frame was already retained above.  Preserve an
                        # explicit no-landmarks row for it, rather than leaving a
                        # shorter sidecar that looks like accidental corruption.
                        landmark_sidecar_rows.append(
                            _failed_landmark_sidecar_row(frame, str(error))
                        )
                    now_ns = (
                        monotonic_ns()
                        if frame.source_kind.startswith("webcam:")
                        else frame.capture_mono_ns
                    )
                    result = pipeline.fault_perception(
                        session_id=frame.source_clock_id,
                        sequence=frame.sequence,
                        source_mono_ns=frame.capture_mono_ns,
                        now_mono_ns=now_ns,
                        reason=str(error),
                    )
                    # Latch FIRST. Without this the control worker was still
                    # accepting work, and `close(drain=True)` in the `finally`
                    # let one already-queued frame solve, latch and append AFTER
                    # this fault was recorded -- leaving the published clip with
                    # a terminal fault whose sequence is not among its own
                    # frames, and a replay that blamed the wrong thing.
                    fault_latched.set()
                    if isinstance(runner, LatestWinsProcessor):
                        # Let the solve already in flight finish and append BEFORE
                        # this fault. It carries an EARLIER sequence, and an
                        # out-of-order FAULT is diverted into `terminal_fault`
                        # (recording.py) -- where it displaces the real reason the
                        # session stopped, and a replay then blames "supervisor is
                        # FAULT" instead of the camera. Measured: terminal_fault
                        # sequence 245 while the clip's own last frame was 249.
                        #
                        # `close` re-raises a worker failure, and this fault must
                        # be recorded either way: letting that escape from here
                        # would throw past the whole publication block and lose
                        # the ingress FAULT entirely. The worker's own failure is
                        # re-raised by the `close` in the `finally`, which is
                        # where it belongs. `close` is safe to call twice.
                        try:
                            runner.close(drain=True)
                        except RealtimeWorkerError as worker_error:
                            print(f"control worker had already failed: {worker_error}")
                    with recorder_lock:
                        recorder.append(result)
                    print(f"perception ingress FAULT at frame {frame.sequence}: {error}")
                    stop_requested = True
                    break
                if args.landmark_sidecar is not None:
                    landmark_sidecar_rows.append(_arm_landmark_sidecar_row(observation))
                if not calibrated:
                    # Show the camera + tracked skeleton from frame ONE, before
                    # calibration succeeds. This is pure visualisation -- it issues no
                    # command and passes no safety gate -- and gating it behind
                    # calibration meant a user staring at nothing while frames were
                    # silently rejected, with no way to see that (say) a wrist was out
                    # of shot. Seeing your own skeleton makes calibration
                    # self-diagnosing instead of a hang.
                    present_names = {landmark.name for landmark in observation.landmarks}
                    missing = pipeline.required_landmarks - present_names
                    identity_ok = observation.identity is IdentityState.STABLE
                    confidence_ok = (
                        observation.aggregate_confidence
                        >= args.calibration_confidence
                    )
                    if identity_ok and confidence_ok and not missing:
                        if calibration_window_policy is None:
                            calibration_window_samples = 0
                        else:
                            _advance_calibration_window(
                                calibration_samples,
                                observation,
                                calibration_window_policy,
                            )
                            calibration_window_samples = len(calibration_samples)
                    else:
                        # Samples must be consecutive eligible observations. A
                        # disqualifying frame is a hard boundary, never a gap
                        # over which stillness may be inferred.
                        calibration_samples.clear()
                        calibration_window_samples = 0
                    calib_frames += 1
                    if missing:
                        # The most common cause by far: the operator is framed head
                        # and shoulders, so elbows and wrists are outside the camera.
                        # Naming the body parts and the remedy beats printing
                        # "pose_15" to a terminal hidden behind this window.
                        # "Step back" is the right remedy only when the body is
                        # too CLOSE. If the detector is placing landmarks outside
                        # the frame, the camera is aimed wrong and stepping back
                        # will not help -- observed 2026-08-26 with a laptop lid
                        # tilted up: the operator's eyes sat on the bottom edge,
                        # the skeleton was estimated below the image entirely, and
                        # the HUD kept saying STEP BACK while confidence stayed at
                        # 0.00. Name the direction to move the CAMERA instead.
                        aim = _framing_advice(observation)
                        hint = (aim or "STEP BACK") + " - can't see: " + ", ".join(
                            sorted(LANDMARK_LABELS.get(n, n) for n in missing)
                        )
                    elif not identity_ok:
                        hint = f"hold still - tracking {observation.identity.name}"
                    elif not confidence_ok:
                        hint = (
                            f"need clearer view - confidence "
                            f"{observation.aggregate_confidence:.2f} of "
                            f"{args.calibration_confidence}"
                        )
                    elif calibration_window_policy is None:
                        hint = "ready - calibrating"
                    elif calibration_window_samples < calibration_window_policy.min_samples:
                        hint = (
                            "collecting neutral window - "
                            f"{calibration_window_samples}/"
                            f"{calibration_window_policy.min_samples} samples"
                        )
                    else:
                        hint = "validating neutral window"
                    if studio is not None:
                        with pipeline.sink.lock:
                            canvas = studio.render(
                                frame.image_bgr,
                                pipeline.sink.data,
                                status="CALIBRATING",
                                # The hint rides in the calibration panel, under
                                # the checklist it explains, rather than on the
                                # telemetry line far above it.
                                telemetry="",
                                observation=observation,
                                joints=pipeline.sink.joint_positions(),
                                calibration=CalibrationStatus(
                                    frames=calib_frames,
                                    missing_landmarks=tuple(sorted(missing)),
                                    required_landmarks=len(pipeline.required_landmarks),
                                    identity=observation.identity.name,
                                    identity_ok=identity_ok,
                                    confidence=observation.aggregate_confidence,
                                    required_confidence=args.calibration_confidence,
                                    hint=hint,
                                ),
                            )
                        if args.display and not studio.show_canvas(canvas):
                            stop_requested = True
                            break
                    # Calibration silently rejecting frames is indistinguishable from
                    # a hang: it took 122 frames once and never converged the next
                    # time, with nothing printed either way. Report which condition
                    # is blocking, so a user can act on it (step back, raise arms,
                    # improve lighting) instead of staring at a still window.
                    if calib_frames % 15 == 0:
                        blockers = []
                        if not identity_ok:
                            blockers.append(f"identity={observation.identity.name}")
                        if not confidence_ok:
                            blockers.append(
                                f"confidence={observation.aggregate_confidence:.2f}"
                                f"<{args.calibration_confidence}"
                            )
                        if missing:
                            blockers.append(f"missing={sorted(missing)}")
                        progress = (
                            f"window={calibration_window_samples}/"
                            f"{calibration_window_policy.min_samples}"
                            if calibration_window_policy is not None and not blockers
                            else "ready"
                        )
                        print(
                            f"calibrating... frame {calib_frames}: "
                            + (", ".join(blockers) if blockers else progress)
                        )
                    if identity_ok and confidence_ok and not missing:
                        try:
                            if calibration_window_policy is None:
                                # Offline/synthetic callers may retain the legacy
                                # single-frame path. Live webcam validation above
                                # rejects this configuration before capture opens.
                                pipeline.calibrate(observation)
                            else:
                                if (
                                    len(calibration_samples)
                                    < calibration_window_policy.min_samples
                                ):
                                    continue
                                pipeline.calibrate_window(
                                    calibration_samples, calibration_window_policy
                                )
                        except (CalibrationError, ArmRetargetError) as error:
                            # A quality failure starts a fresh consecutive window
                            # from this frame; previous moving/invalid evidence can
                            # never be mixed into a later successful calibration.
                            calibration_samples[:] = [observation]
                            if calib_frames % 15 == 0:
                                print(f"calibration rejected: {error}")
                            continue
                        if calibration_window_policy is not None:
                            recorder.record_calibration_window(
                                tuple(calibration_samples)
                            )
                        recorder.mark_initial_source(observation.capture_mono_ns)
                        calibrated = True
                        if args.analysis_sync:
                            runner = SynchronousProcessor(process_work)
                        else:
                            runner = LatestWinsProcessor(
                                process_work, name="motion-studio-control"
                            )
                            runner.start()
                        print(f"calibrated on frame {observation.sequence}")
                        # Calibration seeds the trajectory clock and is not itself a
                        # command frame. The next observation supplies a positive dt.
                        continue
                    else:
                        continue
                # Raw perception continuity is measured HERE, on the capture and
                # detection loop, not inside the control worker: the latest-wins
                # submit below discards observations the detector fully computed,
                # and continuity measured after that discard reports worker
                # scheduling latency as missing data.
                pipeline.observe_source_continuity(observation)
                if args.analysis_sync:
                    timed = runner.process((observation, frame))
                else:
                    runner.submit((observation, frame))
                    timed = runner.poll_latest()
                if timed is not None:
                    present(timed)
                # Repaint EVERY iteration, not only when a solve lands.  The window
                # is the operator's feedback loop; locking it to the solve rate is
                # what made a 15 fps camera look like a 5 fps slideshow.  Also the
                # only place cv2.waitKey pumps the macOS event loop.
                repaint(frame, observation)
                # Deliberately NOT `if fault_latched.is_set(): break` here. The
                # latched FAULT is still in the results queue at that moment, and
                # breaking on the flag left the window, the status line and the
                # composite ending on the last ALLOW -- the operator's only
                # on-screen evidence of the fault, gone. `present` stops the loop
                # when the FAULT itself surfaces, one poll later; the frames
                # captured in between are declined by the worker, so nothing is
                # solved, commanded or recorded for them.
                if stop_requested:
                    break
                command_frame_limit = args.max_frames or MAX_SESSION_FRAMES
                if processed >= command_frame_limit:
                    break
    except KeyboardInterrupt:
        # A human interrupt is a request to finish the session, not permission to
        # lose its evidence.  In particular, the live runner's final metrics are
        # emitted below the cleanup block; letting SIGINT propagate from here used
        # to skip them and abandon an otherwise valid retained capture.  The
        # finally block still closes/drains the latest-wins worker and releases
        # staged video before the normal publication checks run.
        stop_requested = True
        print("preview interrupted; finalizing retained artifacts")
    finally:
        # `close(drain=True)` re-raises a worker failure, and it used to be the
        # first statement here -- so a worker that died took the video releases
        # with it, and the session left behind a half-megabyte composite.mp4 that
        # OpenCV could not open at all. Whatever the worker does, the writers are
        # released and the window is closed.
        try:
            if isinstance(runner, LatestWinsProcessor):
                runner.close(drain=True)
                timed = runner.poll_latest()
                if timed is not None and not stop_requested:
                    present(timed)
        finally:
            if video_writer is not None:
                video_writer.release()
            if source_video_writer is not None:
                source_video_writer.release()
            if studio is not None:
                studio.close()
    if runner is not None:
        processed = runner.metrics.processed
    if perception_fault_reason is not None and not recorder.has_robot_target:
        failure_outcome = (
            RAW_CAPTURE_PERCEPTION_FAULT_AFTER_CALIBRATION
            if calibrated
            else RAW_CAPTURE_PERCEPTION_FAULT_BEFORE_CALIBRATION
        )
        retained = _retain_diagnostic_capture(
            staged_video=source_video_staging,
            final_video=args.source_video,
            frame_map=source_map_path,
            source_frames=source_frame_manifest,
            sidecar=args.landmark_sidecar,
            sidecar_rows=landmark_sidecar_rows,
            capture_outcome=failure_outcome,
            capture_provenance=capture_provenance,
        )
        _discard_staging(recording_output_staging)
        # --force may have admitted an older successful clip.  A failed run must
        # never leave that stale JSON beside its new diagnostic raw video.
        args.output.unlink(missing_ok=True)
        phase = "after" if calibrated else "before"
        raise SystemExit(
            f"perception ingress FAULT {phase} calibration: {perception_fault_reason}; "
            f"{retained}; no live MotionClip was published"
        )
    if not calibrated or not recorder.has_robot_target:
        failure_outcome = (
            RAW_CAPTURE_NEVER_CALIBRATED
            if not calibrated
            else RAW_CAPTURE_ZERO_APPROVED_AFTER_CALIBRATION
        )
        retained = _retain_diagnostic_capture(
            staged_video=source_video_staging,
            final_video=args.source_video,
            frame_map=source_map_path,
            source_frames=source_frame_manifest,
            sidecar=args.landmark_sidecar,
            sidecar_rows=landmark_sidecar_rows,
            capture_outcome=failure_outcome,
            capture_provenance=capture_provenance,
        )
        _discard_staging(recording_output_staging)
        args.output.unlink(missing_ok=True)
        raise SystemExit(
            "no approved robot command frame was captured; "
            f"{retained}; no live MotionClip was published"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clip = recorder.finish()
    clip.save(recording_output_staging)
    saved_frame_count = len(clip.frames)
    if args.arm_mapping == "swivel-priority":
        # A mapping mode that silently falls back is indistinguishable from one
        # that works: the arm still moves and the frame is still ALLOW. Print how
        # often the swivel solve was actually used, so "ran with swivel-priority"
        # can never again be mistaken for "the swivel drove the arm".
        for side, arm in (("left", pipeline.left_arm), ("right", pipeline.right_arm)):
            rejected = sum(
                count
                for reason, count in arm.swivel_fallbacks.items()
                if reason != "accepted_swivel_not_converged"
            )
            attempts = arm.swivel_accepted + rejected
            if not attempts:
                continue
            share = 100.0 * arm.swivel_accepted / attempts
            print(
                f"swivel {side:5s}: used on {arm.swivel_accepted}/{attempts} "
                f"solve attempts ({share:.1f}%)"
            )
            for reason, count in sorted(
                arm.swivel_fallbacks.items(), key=lambda item: -item[1]
            ):
                print(f"    {reason}: {count}")
    if recorder.non_monotonic_frames_dropped:
        # Never silent: a dropped frame is a real gap in the recording, and the
        # reason this is survivable at all is that it used to lose the session.
        print(
            f"note: dropped {recorder.non_monotonic_frames_dropped} frame(s) whose "
            "source timestamp did not advance past the previous frame; the camera "
            "handed back duplicate capture times. The clip is otherwise complete."
        )
    if runner is not None:
        metrics = runner.metrics
        print(
            f"{'analysis-sync' if args.analysis_sync else 'realtime'}: "
            f"processed={metrics.processed} dropped={metrics.dropped_before_processing} "
            f"mean={metrics.mean_processing_ms:.1f}ms p95={metrics.p95_processing_ms:.1f}ms "
            f"capacity={metrics.effective_fps:.1f}fps"
        )
    if args.source_video is None:
        print(f"saved {saved_frame_count} frame(s) to {args.output}")
    if args.preview_video is not None:
        print(f"saved composite preview to {args.preview_video}")
        manifest_path = args.preview_video.with_suffix(".frame-map.json")
        _write_json_document(
            manifest_path,
            _frame_map_document(
                video=args.preview_video,
                artifact_kind="composite-preview",
                frames=preview_frame_manifest,
            ),
        )
        print(f"saved preview frame map to {manifest_path}")
    if args.source_video is not None:
        assert source_video_staging is not None
        assert source_map_path is not None
        assert args.landmark_sidecar is not None
        frame_map_document = _frame_map_document(
            video=args.source_video,
            video_for_hash=source_video_staging,
            artifact_kind="raw-camera",
            frames=source_frame_manifest,
            capture_outcome=RAW_CAPTURE_SUCCEEDED,
            capture_provenance=capture_provenance,
        )
        sidecar_document = _landmark_sidecar_document(
            source_video=args.source_video,
            source_video_for_hash=source_video_staging,
            source_frame_map=source_map_path,
            frame_map_document=frame_map_document,
            frames=landmark_sidecar_rows,
        )
        _validate_raw_capture_documents(
            video=source_video_staging,
            final_video_name=args.source_video.name,
            final_frame_map_name=source_map_path.name,
            frame_map_document=frame_map_document,
            sidecar_document=sidecar_document,
        )
        # Publish the complete raw evidence set before the MotionClip.  A clip
        # is the session's success claim and therefore is always the final
        # commit; an interruption can leave a diagnostic source set, never a
        # success clip with its human evidence missing.
        _publish_raw_capture(
            staged_video=source_video_staging,
            final_video=args.source_video,
            frame_map=source_map_path,
            frame_map_document=frame_map_document,
            sidecar=args.landmark_sidecar,
            sidecar_document=sidecar_document,
        )
        recording_output_staging.replace(args.output)
        print(f"saved {saved_frame_count} frame(s) to {args.output}")
        print(f"saved raw source video to {args.source_video}")
        print(f"saved source frame map to {source_map_path}")
        print(f"saved landmark sidecar to {args.landmark_sidecar}")
    if fault_latched.is_set():
        # Everything above still published: the clip up to the fault IS the
        # evidence, and the raw capture is unaffected -- which is why its
        # provenance still reads `succeeded`. But docs/safety-model.md says a FAULT
        # is latched and needs human inspection before the next session, and a
        # caller that reads only the exit status cannot tell a safety stop from a
        # clean run. `close_gates.sh` is exactly such a caller.
        raise SystemExit(
            f"session ended in a latched FAULT after {saved_frame_count} frame(s); "
            f"the clip and its evidence were published to {args.output} and the "
            "fault is the clip's terminal fault. Inspect before running again."
        )
    return 0


def _file_sha256(path: Path) -> str:
    """Return a stable input identity without loading a recording into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_identity(video: Path, frame_map: Path | None) -> str:
    """Fingerprint every input that affects deterministic recorded-video output."""
    digest = sha256()
    digest.update(_file_sha256(video).encode("ascii"))
    if frame_map is not None:
        digest.update(_file_sha256(frame_map).encode("ascii"))
    return digest.hexdigest()


def _arm_landmark_sidecar_row(observation) -> dict[str, object]:
    """Persist the human evidence required for a later shape metric.

    This sidecar is deliberately capture-only: it records the detector's raw
    shoulder/elbow/wrist/hip evidence and never changes the simulated command path.
    """
    # Hips are required to reconstruct the same torso-relative frame used by
    # the direction-vector mapper and A/B evaluator. Excluding them would force
    # analysis back onto raw camera axes.
    names = (
        "pose_11",
        "pose_12",
        "pose_13",
        "pose_14",
        "pose_15",
        "pose_16",
        "pose_23",
        "pose_24",
    )
    by_name = {landmark.name: landmark for landmark in observation.landmarks}
    return {
        "source_sequence": observation.sequence,
        "source_mono_ns": observation.capture_mono_ns,
        "detector_status": "ok",
        "landmarks": {
            name: {
                "normalized_xyz": landmark.normalized_xyz,
                "world_xyz_m": landmark.world_xyz_m,
                "visibility": landmark.visibility,
                "presence": landmark.presence,
            }
            for name in names
            if (landmark := by_name.get(name)) is not None
        },
    }


def _failed_landmark_sidecar_row(frame: CapturedFrame, reason: str) -> dict[str, object]:
    """Record that MediaPipe failed on a raw frame without fabricating landmarks."""
    return {
        "source_sequence": frame.sequence,
        "source_mono_ns": frame.capture_mono_ns,
        "detector_status": "fault",
        "detector_error": reason,
        "landmarks": {},
    }


def _implementation_hash() -> str:
    """Fingerprint the local Python implementation, including a dirty worktree."""
    package_root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _analysis_artifact_identity(
    *,
    input_hash: str,
    mapping_hash: str,
    arm_mapping: str,
    calibration_confidence: float,
    calibration_window: CalibrationWindowPolicy | None,
    max_frames: int | None,
    liveness: LivenessProvenance,
) -> str:
    """Fingerprint every deliberate choice in an offline analysis artifact."""
    document = {
        "input_hash": input_hash,
        "mapping_hash": mapping_hash,
        "arm_mapping": arm_mapping,
        "calibration_confidence": calibration_confidence,
        "calibration_window": (
            None if calibration_window is None else asdict(calibration_window)
        ),
        "max_frames": max_frames,
        "liveness": liveness.model_dump(mode="json"),
        "implementation_hash": _implementation_hash(),
    }
    return sha256(dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _advance_calibration_window(
    samples: list[HumanObservation],
    observation: HumanObservation,
    policy: CalibrationWindowPolicy,
) -> int:
    """Append one eligible observation, retaining its maximal quiet time window.

    This is deliberately *sliding*, not tumbling. A slow but otherwise healthy
    camera must be able to form a valid window from its newest consecutive
    samples; clearing every sample on a span overrun can leave the operator at
    a misleading perpetual ``ready`` status. Ineligible frames are cleared by
    the caller and therefore remain a hard contiguity boundary.
    """
    dropped = 0
    while (
        samples
        and observation.capture_mono_ns - samples[0].capture_mono_ns
        > policy.max_window_ns
    ):
        samples.pop(0)
        dropped += 1
    samples.append(observation)
    return dropped


CAPTURE_SCHEMA_VERSION = 3
RAW_CAMERA_ARTIFACT = "raw-camera"
COMPOSITE_PREVIEW_ARTIFACT = "composite-preview"
RAW_LANDMARK_SIDECAR_ARTIFACT = "raw-camera-landmarks"
RAW_CAPTURE_SUCCEEDED = "succeeded"
RAW_CAPTURE_NEVER_CALIBRATED = "never_calibrated"
RAW_CAPTURE_ZERO_APPROVED_AFTER_CALIBRATION = "zero_approved_after_calibration"
RAW_CAPTURE_PERCEPTION_FAULT_BEFORE_CALIBRATION = "perception_fault_before_calibration"
RAW_CAPTURE_PERCEPTION_FAULT_AFTER_CALIBRATION = "perception_fault_after_calibration"
RAW_CAPTURE_FAILURES = frozenset(
    {
        RAW_CAPTURE_NEVER_CALIBRATED,
        RAW_CAPTURE_ZERO_APPROVED_AFTER_CALIBRATION,
        RAW_CAPTURE_PERCEPTION_FAULT_BEFORE_CALIBRATION,
        RAW_CAPTURE_PERCEPTION_FAULT_AFTER_CALIBRATION,
    }
)
RAW_CAPTURE_OUTCOMES = RAW_CAPTURE_FAILURES | {RAW_CAPTURE_SUCCEEDED}


def _json_document_bytes(document: dict[str, object]) -> bytes:
    """The one serialization used for manifests and their content hashes."""
    return (dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _document_sha256(document: dict[str, object]) -> str:
    return sha256(_json_document_bytes(document)).hexdigest()


def _write_json_document(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_document_bytes(document))




def _framing_advice(observation) -> str | None:
    """Which way to move the CAMERA, from where landmarks land relative to frame.

    MediaPipe reports normalized coordinates that can fall outside [0, 1] when it
    infers a joint beyond the image, so the estimated body centroid says whether
    the operator is below, above, or beside the view rather than merely too close.
    Returns None when the body is inside the frame and distance is the real issue.
    """
    ys: list[float] = []
    xs: list[float] = []
    for mark in observation.landmarks:
        if not mark.name.startswith("pose_"):
            continue
        xyz = getattr(mark, "normalized_xyz", None)
        if not xyz or len(xyz) < 2:
            continue
        xs.append(float(xyz[0]))
        ys.append(float(xyz[1]))
    if not ys:
        return None
    centre_y = sum(ys) / len(ys)
    centre_x = sum(xs) / len(xs)
    if centre_y > 1.02:
        return "TILT CAMERA DOWN - you are below the view"
    if centre_y < -0.02:
        return "TILT CAMERA UP - you are above the view"
    if centre_x > 1.02:
        return "MOVE LEFT / PAN CAMERA RIGHT"
    if centre_x < -0.02:
        return "MOVE RIGHT / PAN CAMERA LEFT"
    return None


def _camera_device(value: str) -> int | str:
    """Accept an explicit index, or 'builtin'/'auto' to resolve one by name."""
    text = str(value).strip().lower()
    if text in {"builtin", "built-in", "auto", "facetime"}:
        return "builtin"
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--camera takes a device index or 'builtin'"
        ) from None


def _staging_path(path: Path) -> Path:
    """Return a same-directory, deliberately non-publishable capture path."""
    # OpenCV selects its encoder from the final suffix, so the staging path must
    # retain ``.mp4`` rather than appending a suffix after it.
    return path.with_name(f".{path.stem}.partial-{uuid4().hex}{path.suffix}")


def _discard_staging(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def _decoded_video_frame_count(path: Path) -> int:
    """Count decoded frames rather than trusting mutable container metadata."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - production dependency
        raise RuntimeError("OpenCV is required to verify captured video") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not reopen captured video for verification: {path}")
        count = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                return count
            count += 1
    finally:
        capture.release()


def _frame_map_document(
    *,
    video: Path,
    artifact_kind: str,
    frames: list[dict[str, int]],
    video_for_hash: Path | None = None,
    capture_outcome: str | None = None,
    capture_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create a self-describing map for either raw frames or a rendered composite."""
    if artifact_kind not in {RAW_CAMERA_ARTIFACT, COMPOSITE_PREVIEW_ARTIFACT}:
        raise ValueError(f"unsupported frame-map artifact kind: {artifact_kind}")
    hashed_video = video if video_for_hash is None else video_for_hash
    decoded_count = _decoded_video_frame_count(hashed_video)
    if decoded_count != len(frames):
        raise RuntimeError(
            "refusing to publish frame map: decoded video frame count "
            f"{decoded_count} does not equal manifest rows {len(frames)}"
        )
    document: dict[str, object] = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "video": video.name,
        "video_sha256": _file_sha256(hashed_video),
        "video_frame_count": decoded_count,
        "frames": frames,
    }
    if artifact_kind == RAW_CAMERA_ARTIFACT:
        if capture_outcome not in RAW_CAPTURE_OUTCOMES:
            raise ValueError("raw capture frame maps require a known capture outcome")
        if not isinstance(capture_provenance, dict) or not capture_provenance.get("attempt_id"):
            raise ValueError("raw capture frame maps require a capture attempt provenance")
        document["capture_outcome"] = capture_outcome
        document["capture_provenance"] = capture_provenance
    elif capture_outcome is not None or capture_provenance is not None:
        raise ValueError("composite preview maps cannot carry raw-capture outcome metadata")
    return document


def _landmark_sidecar_document(
    *,
    source_video: Path,
    source_video_for_hash: Path,
    source_frame_map: Path,
    frame_map_document: dict[str, object],
    frames: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "artifact_kind": RAW_LANDMARK_SIDECAR_ARTIFACT,
        "source_video": source_video.name,
        "source_video_sha256": _file_sha256(source_video_for_hash),
        "source_frame_map": source_frame_map.name,
        "source_frame_map_sha256": _document_sha256(frame_map_document),
        "source_video_frame_count": frame_map_document["video_frame_count"],
        "frames": frames,
    }


def _require_valid_frame_map_document(
    document: object,
    *,
    video: Path,
    expected_video_name: str | None = None,
    required_artifact_kind: str | None = None,
) -> tuple[list[dict[str, object]], str | None]:
    if not isinstance(document, dict):
        raise SystemExit("source frame map must be a JSON object")
    rows = document.get("frames")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("source frame map must contain at least one frame")
    schema_version = document.get("schema_version")
    if schema_version not in {1, 2, CAPTURE_SCHEMA_VERSION}:
        raise SystemExit("source frame map schema_version is unsupported")
    artifact_kind: str | None = None
    if schema_version in {2, CAPTURE_SCHEMA_VERSION}:
        artifact_kind = document.get("artifact_kind")
        if artifact_kind not in {RAW_CAMERA_ARTIFACT, COMPOSITE_PREVIEW_ARTIFACT}:
            raise SystemExit("source frame map has an unknown artifact_kind")
        if document.get("video_frame_count") != _decoded_video_frame_count(video):
            raise SystemExit("source frame map video_frame_count does not match decoded video")
        if document["video_frame_count"] != len(rows):
            raise SystemExit("source frame map row count does not match video_frame_count")
        if schema_version == CAPTURE_SCHEMA_VERSION and artifact_kind == RAW_CAMERA_ARTIFACT:
            outcome = document.get("capture_outcome")
            provenance = document.get("capture_provenance")
            if outcome not in RAW_CAPTURE_OUTCOMES:
                raise SystemExit("raw capture frame map has an unknown capture_outcome")
            if not isinstance(provenance, dict) or not provenance.get("attempt_id"):
                raise SystemExit("raw capture frame map is missing capture attempt provenance")
    if required_artifact_kind is not None and artifact_kind != required_artifact_kind:
        raise SystemExit(
            f"source frame map must be a {required_artifact_kind} artifact, not "
            f"{artifact_kind or 'legacy/unspecified'}"
        )
    if document.get("video") != (video.name if expected_video_name is None else expected_video_name):
        raise SystemExit("source frame map names a different video")
    if document.get("video_sha256") != _file_sha256(video):
        raise SystemExit("source frame map video_sha256 does not match the supplied video")
    return rows, artifact_kind


def _validate_raw_capture_documents(
    *,
    video: Path,
    final_video_name: str,
    final_frame_map_name: str,
    frame_map_document: dict[str, object],
    sidecar_document: dict[str, object],
) -> None:
    """Fail closed unless all raw-capture evidence has one exact row per frame."""
    rows, artifact_kind = _require_valid_frame_map_document(
        frame_map_document,
        video=video,
        expected_video_name=final_video_name,
        required_artifact_kind=RAW_CAMERA_ARTIFACT,
    )
    if frame_map_document.get("video") != final_video_name:
        raise RuntimeError("raw frame map final video name does not match capture target")
    if not isinstance(sidecar_document.get("frames"), list):
        raise RuntimeError("landmark sidecar frames must be a list")
    if sidecar_document.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise RuntimeError("landmark sidecar schema_version is invalid")
    if sidecar_document.get("artifact_kind") != RAW_LANDMARK_SIDECAR_ARTIFACT:
        raise RuntimeError("landmark sidecar artifact_kind is invalid")
    if sidecar_document.get("source_video") != final_video_name:
        raise RuntimeError("landmark sidecar names a different source video")
    if sidecar_document.get("source_video_sha256") != _file_sha256(video):
        raise RuntimeError("landmark sidecar source video hash does not match capture")
    if sidecar_document.get("source_frame_map") != final_frame_map_name:
        raise RuntimeError("landmark sidecar names a different frame map")
    if sidecar_document.get("source_frame_map_sha256") != _document_sha256(frame_map_document):
        raise RuntimeError("landmark sidecar frame-map hash does not match capture")
    if sidecar_document.get("source_video_frame_count") != len(rows):
        raise RuntimeError("landmark sidecar frame count does not match raw frame map")
    sidecar_rows = sidecar_document["frames"]
    if len(sidecar_rows) != len(rows):
        raise RuntimeError("landmark sidecar row count does not match raw video")
    for index, (map_row, sidecar_row) in enumerate(zip(rows, sidecar_rows, strict=True)):
        if not isinstance(sidecar_row, dict):
            raise RuntimeError("landmark sidecar row must be an object")
        if sidecar_row.get("source_sequence") != map_row.get("source_sequence"):
            raise RuntimeError(f"landmark sidecar source_sequence mismatch at frame {index}")
        if sidecar_row.get("source_mono_ns") != map_row.get("source_mono_ns"):
            raise RuntimeError(f"landmark sidecar source_mono_ns mismatch at frame {index}")
    assert artifact_kind == RAW_CAMERA_ARTIFACT


def _publish_raw_capture(
    *,
    staged_video: Path,
    final_video: Path,
    frame_map: Path,
    frame_map_document: dict[str, object],
    sidecar: Path,
    sidecar_document: dict[str, object],
) -> None:
    """Write all metadata first, then expose a fully validated raw capture set."""
    staged_map = _staging_path(frame_map)
    staged_sidecar = _staging_path(sidecar)
    try:
        _write_json_document(staged_map, frame_map_document)
        _write_json_document(staged_sidecar, sidecar_document)
        # A map/sidecar without its video is not analyzable.  Publishing it first
        # therefore never creates a misleading usable recording; the video is the
        # commit point and all output paths were verified absent before capture.
        staged_map.replace(frame_map)
        staged_sidecar.replace(sidecar)
        staged_video.replace(final_video)
    except BaseException:
        _discard_staging(staged_map)
        _discard_staging(staged_sidecar)
        _discard_staging(staged_video)
        raise


def _retain_diagnostic_capture(
    *,
    staged_video: Path | None,
    final_video: Path | None,
    frame_map: Path | None,
    source_frames: list[dict[str, int]],
    sidecar: Path | None,
    sidecar_rows: list[dict[str, object]],
    capture_outcome: str,
    capture_provenance: dict[str, object] | None,
) -> str:
    """Retain raw evidence from a failed live attempt without minting a clip.

    A no-command capture cannot produce a valid ``MotionClip``: it has neither
    an approved target nor calibration evidence.  It can still be the most
    valuable diagnostic artifact of the rehearsal, particularly when neutral
    calibration or perception ingress fails.  Preserve the raw video and its
    exact frame map after the writers close, but never create a file that could
    be mistaken for a successful ``live.json``.

    The landmark sidecar is published when it has exactly one declared detector
    result for every raw frame.  A detector fault has an explicit ``fault`` row
    with no fabricated landmarks; it is complete provenance, not successful
    human tracking.
    """
    if (
        staged_video is None
        or final_video is None
        or frame_map is None
        or not source_frames
    ):
        return "no raw diagnostic frames were captured"
    frame_map_document = _frame_map_document(
        video=final_video,
        video_for_hash=staged_video,
        artifact_kind=RAW_CAMERA_ARTIFACT,
        frames=source_frames,
        capture_outcome=capture_outcome,
        capture_provenance=capture_provenance,
    )
    if sidecar is not None and len(sidecar_rows) == len(source_frames):
        sidecar_document = _landmark_sidecar_document(
            source_video=final_video,
            source_video_for_hash=staged_video,
            source_frame_map=frame_map,
            frame_map_document=frame_map_document,
            frames=sidecar_rows,
        )
        _validate_raw_capture_documents(
            video=staged_video,
            final_video_name=final_video.name,
            final_frame_map_name=frame_map.name,
            frame_map_document=frame_map_document,
            sidecar_document=sidecar_document,
        )
        _publish_raw_capture(
            staged_video=staged_video,
            final_video=final_video,
            frame_map=frame_map,
            frame_map_document=frame_map_document,
            sidecar=sidecar,
            sidecar_document=sidecar_document,
        )
        return "diagnostic raw video, frame map, and complete landmark sidecar retained"

    # A raw/map pair remains valuable, but no incomplete sidecar may survive a
    # forced rerun and be mistaken for a full source triple.  The caller chose
    # --force before an existing target could be removed.
    _require_valid_frame_map_document(
        frame_map_document,
        video=staged_video,
        expected_video_name=final_video.name,
        required_artifact_kind=RAW_CAMERA_ARTIFACT,
    )
    staged_map = _staging_path(frame_map)
    try:
        _write_json_document(staged_map, frame_map_document)
        staged_map.replace(frame_map)
        staged_video.replace(final_video)
        if sidecar is not None:
            sidecar.unlink(missing_ok=True)
    except BaseException:
        _discard_staging(staged_map)
        _discard_staging(staged_video)
        raise
    return "diagnostic raw video and frame map retained; landmark sidecar incomplete"


def _load_landmark_sidecar(
    path: Path,
    *,
    video: Path,
    frame_map: Path,
    timeline: tuple[SourceTimelineFrame, ...] | None,
) -> None:
    """Verify that a candidate A/B run owns raw human evidence for every frame."""
    try:
        document = loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"invalid landmark sidecar: {path}") from error
    if timeline is None:
        raise SystemExit("landmark sidecar requires a source frame map")
    if not isinstance(document, dict) or document.get("schema_version") not in {2, CAPTURE_SCHEMA_VERSION}:
        raise SystemExit("landmark sidecar schema_version is unsupported")
    if document.get("artifact_kind") != RAW_LANDMARK_SIDECAR_ARTIFACT:
        raise SystemExit("landmark sidecar is not a raw-camera landmark artifact")
    if document.get("source_video") != video.name:
        raise SystemExit("landmark sidecar names a different source video")
    if document.get("source_video_sha256") != _file_sha256(video):
        raise SystemExit("landmark sidecar source_video_sha256 does not match video")
    if document.get("source_frame_map") != frame_map.name:
        raise SystemExit("landmark sidecar names a different source frame map")
    if document.get("source_frame_map_sha256") != _file_sha256(frame_map):
        raise SystemExit("landmark sidecar source_frame_map_sha256 does not match map")
    rows = document.get("frames")
    if document.get("source_video_frame_count") != len(timeline) or not isinstance(rows, list):
        raise SystemExit("landmark sidecar frame count does not match source map")
    if len(rows) != len(timeline):
        raise SystemExit("landmark sidecar row count does not match source map")
    for index, (row, mapped) in enumerate(zip(rows, timeline, strict=True)):
        if not isinstance(row, dict):
            raise SystemExit(f"landmark sidecar row {index} is not an object")
        if row.get("source_sequence") != mapped.source_sequence:
            raise SystemExit(f"landmark sidecar source_sequence mismatch at frame {index}")
        if row.get("source_mono_ns") != mapped.source_mono_ns:
            raise SystemExit(f"landmark sidecar source_mono_ns mismatch at frame {index}")


def _load_source_frame_map(
    path: Path,
    video: Path,
    *,
    required_artifact_kind: str | None = None,
    allow_failed_source: bool = False,
    allow_legacy_source: bool = False,
) -> tuple[SourceTimelineFrame, ...]:
    """Load a frame map emitted beside a preview MP4, rejecting ambiguous joins."""
    try:
        document = loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"invalid source frame map: {path}") from error
    rows, artifact_kind = _require_valid_frame_map_document(
        document, video=video, required_artifact_kind=required_artifact_kind
    )
    schema_version = document.get("schema_version")
    if schema_version != CAPTURE_SCHEMA_VERSION:
        if not allow_legacy_source:
            raise SystemExit(
                "source frame map is legacy and has no v3 capture-outcome attestation; pass "
                "--allow-legacy-source only for diagnosis, never G5/G6 evidence"
            )
    elif allow_legacy_source:
        raise SystemExit("--allow-legacy-source is only valid for a pre-v3 source frame map")
    if allow_failed_source and not (
        artifact_kind == RAW_CAMERA_ARTIFACT
        and schema_version == CAPTURE_SCHEMA_VERSION
        and document.get("capture_outcome") in RAW_CAPTURE_FAILURES
    ):
        raise SystemExit(
            "--allow-failed-source is only valid for an unsuccessful v3 raw capture"
        )
    if (
        artifact_kind == RAW_CAMERA_ARTIFACT
        and schema_version == CAPTURE_SCHEMA_VERSION
        and document.get("capture_outcome") != RAW_CAPTURE_SUCCEEDED
        and not allow_failed_source
    ):
        raise SystemExit(
            "source frame map records an unsuccessful capture; pass "
            "--allow-failed-source only for diagnosis, never G5/G6 evidence"
        )
    mapped: list[SourceTimelineFrame] = []
    previous_sequence = -1
    previous_timestamp = -1
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("video_frame_index") != index:
            raise SystemExit("source frame map video_frame_index must be contiguous from zero")
        sequence = row.get("source_sequence")
        timestamp = row.get("source_mono_ns")
        if (
            not isinstance(sequence, int)
            or not isinstance(timestamp, int)
            or sequence <= previous_sequence
            or timestamp <= previous_timestamp
        ):
            raise SystemExit("source frame map source sequence/timestamps must strictly increase")
        mapped.append(SourceTimelineFrame(sequence, timestamp))
        previous_sequence, previous_timestamp = sequence, timestamp
    return tuple(mapped)


def _source_replay_provenance(
    frame_map: Path,
    *,
    allow_failed_source: bool,
    allow_legacy_source: bool,
) -> SourceReplayProvenance:
    """Copy the source map's admission decision into every derived MotionClip."""
    try:
        document = loads(frame_map.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:  # loader already reports this normally
        raise RuntimeError(f"could not read validated source frame map: {frame_map}") from error
    schema_version = document.get("schema_version")
    if not isinstance(schema_version, int):  # defensive: loader should have rejected it
        raise RuntimeError("validated source frame map did not carry an integer schema version")
    raw_v3 = (
        schema_version == CAPTURE_SCHEMA_VERSION
        and document.get("artifact_kind") == RAW_CAMERA_ARTIFACT
    )
    capture_outcome = (
        document.get("capture_outcome") if raw_v3 else "legacy-or-nonraw-source-map"
    )
    if not isinstance(capture_outcome, str):
        raise RuntimeError("validated source frame map did not carry a capture outcome")
    return SourceReplayProvenance(
        origin="recorded-video-replay",
        enabled=True,
        frame_map_schema_version=schema_version,
        capture_outcome=capture_outcome,
        allow_failed_source=allow_failed_source,
        allow_legacy_source=allow_legacy_source,
    )


def _replay(path: Path) -> int:
    clip = MotionClip.load(path)
    model = load_verified_fixed_base_model()
    sink = MujocoPreviewSink(model=model, initial_pose=SIM_TELEOP_START_QPOS)
    supervisor = SafetySupervisor(
        ClearanceChecker(
            model=model, home=HOME_QPOS, **clearance_kwargs_for(clip.motion_profile)
        ),
        initial_pose=SIM_TELEOP_START_QPOS,
        source_clock_id=clip.frames[0].decision.source_clock_id,
        policy=policy_for(clip.motion_profile),
    )
    result = replay_clip(clip, supervisor=supervisor, sink=sink)
    print(f"replayed {len(result.receipts)} approved frame(s); {result.held_frames} held")
    if result.terminal_fault is not None:
        print(
            "replay ended in terminal FAULT: "
            + "; ".join(result.terminal_fault.reasons)
        )
    return 0


#: Dogfood demo shape. 30 Hz is the rate a webcam pipeline actually runs at, and
#: sampling finely is what keeps each step inside the limiter while still covering
#: a large excursion.
DEMO_FPS = 30
DEMO_SECONDS = 6
DEMO_FRAMES = DEMO_FPS * DEMO_SECONDS
DEMO_FRAME_NS = 1_000_000_000 // DEMO_FPS
DEMO_WAVE_CYCLES = 2.0
DEMO_WAVE_AMPLITUDE = 0.30

def _demo(output: Path | None) -> int:
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = Path("artifacts") / f"demo-{stamp}"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty demo directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pipeline = MotionStudioPipeline(source_clock_id="synthetic-clock")
    if not isinstance(pipeline.sink, MujocoPreviewSink):
        raise RuntimeError("demo requires the MuJoCo preview sink")
    # Calibrate at the middle of the intended range, then begin command frames one
    # cadence later. This exercises both directions without biasing an IK branch.
    neutral = synthetic_observation(
        0, timestamp_ns=1_000_000_000, motion_fraction=0.5
    )
    pipeline.calibrate(neutral)
    recorder = MotionRecorder(
        task="synthetic bilateral upper-body mirror",
        calibration_id="synthetic-neutral-v1",
        joint_order=JOINT_ORDER,
        motion_profile=pipeline.motion_profile.value,
        initial_source_mono_ns=neutral.capture_mono_ns,
        source_replay=SourceReplayProvenance(origin="synthetic"),
    )
    frames_dir = output / "frames"
    frames_dir.mkdir()
    import mujoco

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 1.1]
    camera.distance = 2.2
    camera.azimuth = 150.0
    camera.elevation = -10.0
    print(f"rendering {DEMO_FRAMES} synthetic frames...", flush=True)
    # 30 Hz for DEMO_SECONDS, tracing a wave rather than a one-way ramp.
    #
    # This used to be 11 frames at 5 Hz, with a comment saying the low rate existed
    # to keep the conservative acceleration gate satisfied. That had two costs: the
    # per-frame jump was large enough to trip the step limiter on `left_arm_joint6`,
    # and 2.2 seconds of one-way ramp reads as a robot standing still. Sampling the
    # SAME total excursion 6x more finely makes each step 6x smaller -- it passes the
    # gate comfortably AND looks like motion, because it is motion.
    #
    # motion_fraction follows a raised sine so the arms sweep out and back twice
    # instead of drifting to a stop, which is what "mirroring someone waving" is.
    with mujoco.Renderer(pipeline.sink.model, height=480, width=640) as renderer:
        for frame_index in range(DEMO_FRAMES):
            sequence = frame_index + 1
            timestamp = 1_000_000_000 + sequence * DEMO_FRAME_NS
            phase = sequence / DEMO_FRAMES
            motion_fraction = 0.5 + DEMO_WAVE_AMPLITUDE * sin(
                2.0 * pi * DEMO_WAVE_CYCLES * phase
            )
            observation = synthetic_observation(
                sequence,
                timestamp_ns=timestamp,
                motion_fraction=motion_fraction,
            )
            result = pipeline.process_fail_closed(observation, now_mono_ns=timestamp)
            recorder.append(result)
            if result.decision.outcome is not SafetyOutcome.ALLOW:
                raise RuntimeError(
                    f"synthetic frame {sequence} was not allowed: {result.decision.reasons}"
                )
            _render_png(pipeline.sink, frames_dir / f"frame_{frame_index:03d}.png", renderer, camera)
            if sequence % 30 == 0:
                print(f"  rendered {sequence}/{DEMO_FRAMES}", flush=True)
    clip = recorder.finish()
    clip_path = output / "motion_clip.json"
    clip.save(clip_path)

    poses = [dict(frame.observed_joints_rad or ()) for frame in clip.frames]
    start = poses[0]

    def pose_deltas(pose: dict[str, float]) -> tuple[float, float, float, float, float]:
        return (
            max(abs(pose[f"head_joint{index}"] - start[f"head_joint{index}"]) for index in (1, 2)),
            max(
                abs(pose[f"left_arm_joint{index}"] - start[f"left_arm_joint{index}"])
                for index in range(1, 8)
            ),
            max(
                abs(pose[f"right_arm_joint{index}"] - start[f"right_arm_joint{index}"])
                for index in range(1, 8)
            ),
            abs(pose["left_gripper_joint"] - start["left_gripper_joint"]),
            abs(pose["right_gripper_joint"] - start["right_gripper_joint"]),
        )

    deltas = [pose_deltas(pose) for pose in poses]
    peak_index = max(range(len(deltas)), key=lambda index: sum(deltas[index]))
    head_delta = max(delta[0] for delta in deltas)
    left_arm_delta = max(delta[1] for delta in deltas)
    right_arm_delta = max(delta[2] for delta in deltas)
    left_gripper_delta = max(delta[3] for delta in deltas)
    right_gripper_delta = max(delta[4] for delta in deltas)
    if min(
        head_delta,
        left_arm_delta,
        right_arm_delta,
        left_gripper_delta,
        right_gripper_delta,
    ) <= 1e-4:
        raise RuntimeError("dogfood trajectory did not move head, arms, and grippers")
    _write_comparison(
        frames_dir / "frame_000.png",
        frames_dir / f"frame_{peak_index:03d}.png",
        output / "motion_comparison.png",
        head_delta=head_delta,
        left_arm_delta=left_arm_delta,
        right_arm_delta=right_arm_delta,
        left_gripper_delta=left_gripper_delta,
        right_gripper_delta=right_gripper_delta,
    )
    _write_video(frames_dir, output / "motion_preview.mp4", fps=DEMO_FPS)

    model = load_verified_fixed_base_model()
    replay_sink = MujocoPreviewSink(model=model, initial_pose=SIM_TELEOP_START_QPOS)
    replay_supervisor = SafetySupervisor(
        ClearanceChecker(
            model=model, home=HOME_QPOS, **clearance_kwargs_for(clip.motion_profile)
        ),
        initial_pose=SIM_TELEOP_START_QPOS,
        source_clock_id="synthetic-clock",
        policy=policy_for(clip.motion_profile),
    )
    replayed = replay_clip(clip, supervisor=replay_supervisor, sink=replay_sink)
    expected = tuple(
        tuple((joint.name, joint.position_rad) for joint in frame.target.joints)
        for frame in clip.frames
        if frame.target is not None and frame.decision.outcome is SafetyOutcome.ALLOW
    )
    if replayed.joint_targets != expected:
        raise RuntimeError("replay target sequence differs from the canonical clip")

    clips = (clip.model_copy(update={"clip_id": "demo-0", "task": "mirror"}),)
    dataset = export_lerobot_v21(clips, output / "lerobot-v21", fps=DEMO_FPS)
    validator_summary = _validate_v21(dataset)
    print("Galbot Motion Studio dogfood: PASS")
    print(f"  approved frames: {len(clip.frames)}")
    print(f"  head qpos delta:  {head_delta:.6f} rad")
    print(f"  left-arm delta:   {left_arm_delta:.6f} rad")
    print(f"  right-arm delta:  {right_arm_delta:.6f} rad")
    print(f"  left-grip delta:  {left_gripper_delta:.6f} rad")
    print(f"  right-grip delta: {right_gripper_delta:.6f} rad")
    print(f"  rendered motion:  {output / 'motion_comparison.png'}")
    print(f"  motion video:     {output / 'motion_preview.mp4'} ({DEMO_FRAMES} poses at {DEMO_FPS} FPS)")
    print(f"  clip + replay:    {clip_path} ({len(replayed.receipts)} fresh approvals)")
    print(f"  v2.1 dataset:     {dataset} (1 honest episode, 1 task; {validator_summary})")
    print("  hardware imports: none")
    return 0


def _validate_v21(dataset: Path) -> str:
    import sys

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as error:
        if error.name == "lerobot":
            return "optional LeRobot validation unavailable; install .[export-validation]"
        raise

    loaded = LeRobotDataset(
        repo_id="local/galbot-motion-studio",
        root=dataset,
        download_videos=False,
    )
    if len(loaded) <= 0:
        raise RuntimeError("LeRobot loaded an empty dataset")

    validator_root = Path(__file__).resolve().parents[3] / "lerobot-dataset-check"
    if not validator_root.is_dir():
        return f"LeRobot loaded {len(loaded)} frames; optional secondary linter unavailable"
    sys.path.insert(0, str(validator_root))
    try:
        from lerobot_dataset_check import checks, loaders, report

        opened = loaders.open_dataset(dataset)
        sections = [check(opened) for check in checks.ALL_CHECKS]
        findings = [finding for section in sections for finding in section.findings]
        failures = sum(status == report.FAIL for status, _ in findings)
        warnings = sum(status == report.WARN for status, _ in findings)
        passes = sum(status == report.PASS for status, _ in findings)
        if failures or warnings:
            raise RuntimeError(
                f"v2.1 validation failed: {passes} PASS, {warnings} WARN, {failures} FAIL"
            )
        return f"{passes} PASS, 0 WARN, 0 FAIL"
    finally:
        sys.path.remove(str(validator_root))


def _render_png(sink: MujocoPreviewSink, path: Path, renderer, camera) -> None:
    import cv2

    with sink.lock:
        renderer.update_scene(sink.data, camera=camera)
        rgb = renderer.render()
    if not cv2.imwrite(str(path), rgb[:, :, ::-1]):
        raise RuntimeError(f"could not write rendered frame: {path}")


def _write_comparison(
    start_path: Path,
    end_path: Path,
    destination: Path,
    *,
    head_delta: float,
    left_arm_delta: float,
    right_arm_delta: float,
    left_gripper_delta: float,
    right_gripper_delta: float,
) -> None:
    import cv2
    import numpy as np

    start = cv2.imread(str(start_path))
    end = cv2.imread(str(end_path))
    if start is None or end is None:
        raise RuntimeError("could not read rendered frames for comparison")
    cv2.putText(start, "START", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(end, "PEAK", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    canvas = np.concatenate((start, end), axis=1)
    cv2.putText(
        canvas,
        f"head {head_delta:.3f} | arms {left_arm_delta:.3f}/{right_arm_delta:.3f} | "
        f"grips {left_gripper_delta:.3f}/{right_gripper_delta:.3f} rad",
        (20, canvas.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"could not write comparison: {destination}")


def _write_video(frames_dir: Path, destination: Path, *, fps: int) -> None:
    import cv2

    paths = sorted(frames_dir.glob("frame_*.png"))
    if len(paths) != DEMO_FRAMES:
        raise RuntimeError(
            f"expected {DEMO_FRAMES} rendered poses, found {len(paths)}"
        )
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise RuntimeError(f"could not read rendered frame: {paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        if not writer.isOpened():
            raise RuntimeError("could not open the motion-preview video writer")
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError(f"invalid rendered frame: {path}")
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
