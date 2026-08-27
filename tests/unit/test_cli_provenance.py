from json import dumps, loads
from pathlib import Path
from types import SimpleNamespace

import pytest

from galbot_motion_studio.cli import (
    RAW_CAMERA_ARTIFACT,
    _advance_calibration_window,
    _arm_landmark_sidecar_row,
    _analysis_artifact_identity,
    _frame_map_document,
    _file_sha256,
    _landmark_sidecar_document,
    _load_landmark_sidecar,
    _load_source_frame_map,
    _publish_raw_capture,
    _staging_path,
    _validate_raw_capture_documents,
    _write_json_document,
)
from galbot_motion_studio.vision.calibration import CalibrationWindowPolicy


def test_landmark_sidecar_retains_hips_for_torso_relative_metrics() -> None:
    landmarks = tuple(
        SimpleNamespace(
            name=f"pose_{index}",
            normalized_xyz=(0.1, 0.2, 0.3),
            world_xyz_m=(0.4, 0.5, 0.6),
            visibility=1.0,
            presence=1.0,
        )
        for index in (11, 12, 13, 14, 15, 16, 23, 24)
    )
    row = _arm_landmark_sidecar_row(
        SimpleNamespace(sequence=7, capture_mono_ns=99, landmarks=landmarks)
    )
    assert set(row["landmarks"]) == {
        "pose_11",
        "pose_12",
        "pose_13",
        "pose_14",
        "pose_15",
        "pose_16",
        "pose_23",
        "pose_24",
    }


def test_private_landmark_sidecar_requires_the_ignored_suffix(tmp_path: Path) -> None:
    from galbot_motion_studio.cli import main

    with pytest.raises(SystemExit, match="must end in .landmarks.json"):
        main(
            [
                "preview",
                "--camera",
                "0",
                "--output",
                str(tmp_path / "capture.json"),
                "--source-video",
                str(tmp_path / "raw.mp4"),
                "--landmark-sidecar",
                str(tmp_path / "private-body-data.json"),
            ]
        )


def test_live_preview_requires_an_explicit_liveness_budget(tmp_path: Path) -> None:
    from galbot_motion_studio.cli import main

    with pytest.raises(SystemExit, match="liveness-max-static-ms is required"):
        main(
            [
                "preview",
                "--camera",
                "0",
                "--output",
                str(tmp_path / "capture.json"),
            ]
        )


def test_live_preview_rejects_negative_camera_before_native_open(tmp_path: Path) -> None:
    """Invalid device indices must never reach AVFoundation/OpenCV."""
    from galbot_motion_studio.cli import main

    with pytest.raises(SystemExit, match="non-negative device index"):
        main(
            [
                "preview",
                "--camera",
                "-1",
                "--output",
                str(tmp_path / "capture.json"),
            ]
        )


def test_preview_keyboard_interrupt_finalizes_instead_of_propagating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C after an approved frame must retain the clip and final metrics."""
    import galbot_motion_studio.cli as cli
    from galbot_motion_studio.ports.frames import CapturedFrame
    from galbot_motion_studio.synthetic import synthetic_observation

    numpy = pytest.importorskip("numpy")
    observations = iter(
        (
            synthetic_observation(
                1, timestamp_ns=1_000_000_000, motion_fraction=0.5
            ).model_copy(
                update={
                    "source_clock_id": "recorded-video-clock",
                    "session_id": "interrupt-regression",
                }
            ),
            synthetic_observation(
                2, timestamp_ns=1_100_000_000, motion_fraction=0.7
            ).model_copy(
                update={
                    "source_clock_id": "recorded-video-clock",
                    "session_id": "interrupt-regression",
                }
            ),
        )
    )

    class InterruptingSource:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def frames(self):
            for sequence, timestamp_ns in ((1, 1_000_000_000), (2, 1_100_000_000)):
                yield CapturedFrame(
                    sequence=sequence,
                    source_clock_id="recorded-video-clock",
                    capture_mono_ns=timestamp_ns,
                    image_bgr=numpy.zeros((480, 640, 3), dtype=numpy.uint8),
                    source_kind="recorded-video",
                )
            raise KeyboardInterrupt("operator requested stop")

    class NoopDetector:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def detect(self, _frame, *, calibration_id: str):
            return next(observations).model_copy(
                update={"calibration_id": calibration_id}
            )

    monkeypatch.setattr(cli, "RecordedVideoSource", InterruptingSource)
    monkeypatch.setattr(cli, "MediaPipeHolisticDetector", NoopDetector)

    output = tmp_path / "capture.json"
    assert (
        cli.main(
            [
                "preview",
                "--video",
                str(tmp_path / "ignored.mp4"),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    terminal = capsys.readouterr().out
    assert output.exists()
    assert "preview interrupted; finalizing retained artifacts" in terminal
    assert "realtime: processed=1" in terminal


def test_uncalibrated_live_attempt_retains_diagnostic_raw_triple(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed neutral window must not erase the only useful rehearsal evidence."""
    import galbot_motion_studio.cli as cli
    from galbot_motion_studio.ports.frames import CapturedFrame
    from galbot_motion_studio.synthetic import synthetic_observation

    numpy = pytest.importorskip("numpy")
    observations = iter(
        (
            synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.0),
            synthetic_observation(2, timestamp_ns=1_100_000_000, motion_fraction=0.0),
        )
    )

    class TwoFrameWebcam:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def frames(self, *, max_frames: int | None = None):
            for sequence, timestamp_ns in ((1, 1_000_000_000), (2, 1_100_000_000)):
                yield CapturedFrame(
                    sequence=sequence,
                    source_clock_id="local-monotonic",
                    capture_mono_ns=timestamp_ns,
                    image_bgr=numpy.zeros((16, 16, 3), dtype=numpy.uint8),
                    source_kind="webcam:0",
                )

    class NoopDetector:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def detect(self, _frame, *, calibration_id: str):
            return next(observations).model_copy(
                update={
                    "source_clock_id": "local-monotonic",
                    "session_id": "diagnostic-capture",
                    "camera_id": "webcam:0",
                    "calibration_id": calibration_id,
                }
            )

    monkeypatch.setattr(cli, "WebcamSource", TwoFrameWebcam)
    monkeypatch.setattr(cli, "MediaPipeHolisticDetector", NoopDetector)

    output = tmp_path / "live.json"
    raw_video = tmp_path / "raw.mp4"
    sidecar = tmp_path / "raw.landmarks.json"
    with pytest.raises(SystemExit, match="no approved robot command frame") as error:
        cli.main(
            [
                "preview",
                "--camera",
                "0",
                "--output",
                str(output),
                "--source-video",
                str(raw_video),
                "--landmark-sidecar",
                str(sidecar),
                "--liveness-max-static-ms",
                "500",
                "--liveness-max-history",
                "256",
                "--calibration-window-ms",
                "1500",
                "--calibration-min-samples",
                "15",
                "--calibration-max-center-deviation-normalized",
                "0.03",
                "--calibration-max-shoulder-width-deviation-normalized",
                "0.03",
                "--calibration-max-eye-span-deviation-normalized",
                "0.02",
            ]
        )

    frame_map = raw_video.with_suffix(".frame-map.json")
    assert not output.exists()
    assert raw_video.is_file()
    assert frame_map.is_file()
    assert sidecar.is_file()
    with pytest.raises(SystemExit, match="unsuccessful capture"):
        _load_source_frame_map(
            frame_map, raw_video, required_artifact_kind=RAW_CAMERA_ARTIFACT
        )
    timeline = _load_source_frame_map(
        frame_map,
        raw_video,
        required_artifact_kind=RAW_CAMERA_ARTIFACT,
        allow_failed_source=True,
    )
    _load_landmark_sidecar(sidecar, video=raw_video, frame_map=frame_map, timeline=timeline)
    assert "diagnostic raw video, frame map, and complete landmark sidecar retained" in str(
        error.value
    )


def test_perception_fault_retains_explicit_fault_sidecar_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A detector failure is retained as a failed source, never an omitted sidecar."""
    import galbot_motion_studio.cli as cli
    from galbot_motion_studio.ports.frames import CapturedFrame

    numpy = pytest.importorskip("numpy")

    class OneFrameWebcam:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def frames(self, *, max_frames: int | None = None):
            yield CapturedFrame(
                sequence=1,
                source_clock_id="local-monotonic",
                capture_mono_ns=1_000_000_000,
                image_bgr=numpy.zeros((16, 16, 3), dtype=numpy.uint8),
                source_kind="webcam:0",
            )

    class FailingDetector:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def detect(self, _frame, *, calibration_id: str):
            raise cli.HolisticDetectorError("injected detector fault")

    monkeypatch.setattr(cli, "WebcamSource", OneFrameWebcam)
    monkeypatch.setattr(cli, "MediaPipeHolisticDetector", FailingDetector)

    output = tmp_path / "live.json"
    raw_video = tmp_path / "raw.mp4"
    sidecar = tmp_path / "raw.landmarks.json"
    with pytest.raises(SystemExit, match="perception ingress FAULT before calibration"):
        cli.main(
            [
                "preview",
                "--camera",
                "0",
                "--output",
                str(output),
                "--source-video",
                str(raw_video),
                "--landmark-sidecar",
                str(sidecar),
                "--liveness-max-static-ms",
                "500",
                "--calibration-window-ms",
                "1500",
                "--calibration-min-samples",
                "15",
                "--calibration-max-center-deviation-normalized",
                "0.03",
                "--calibration-max-shoulder-width-deviation-normalized",
                "0.03",
                "--calibration-max-eye-span-deviation-normalized",
                "0.02",
            ]
        )

    frame_map = raw_video.with_suffix(".frame-map.json")
    document = loads(frame_map.read_text(encoding="utf-8"))
    assert document["capture_outcome"] == "perception_fault_before_calibration"
    timeline = _load_source_frame_map(
        frame_map,
        raw_video,
        required_artifact_kind=RAW_CAMERA_ARTIFACT,
        allow_failed_source=True,
    )
    _load_landmark_sidecar(sidecar, video=raw_video, frame_map=frame_map, timeline=timeline)
    sidecar_document = loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_document["frames"] == [
        {
            "detector_error": "injected detector fault",
            "detector_status": "fault",
            "landmarks": {},
            "source_mono_ns": 1_000_000_000,
            "source_sequence": 1,
        }
    ]


def test_failed_source_override_is_persisted_and_cannot_be_exported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A diagnostic replay must carry its failed-source admission into its clip."""
    import galbot_motion_studio.cli as cli
    from galbot_motion_studio.export_v21 import ExportError, export_lerobot_v21
    from galbot_motion_studio.ports.frames import CapturedFrame
    from galbot_motion_studio.recording import MotionClip
    from galbot_motion_studio.synthetic import synthetic_observation

    numpy = pytest.importorskip("numpy")
    raw_video = tmp_path / "raw.mp4"
    _write_small_video(raw_video)
    timestamps = (1_000_000_000, 1_100_000_000, 1_200_000_000)
    frame_map = raw_video.with_suffix(".frame-map.json")
    _write_json_document(
        frame_map,
        _frame_map_document(
            video=raw_video,
            artifact_kind=RAW_CAMERA_ARTIFACT,
            frames=[
                {
                    "video_frame_index": index,
                    "source_sequence": index + 1,
                    "source_mono_ns": timestamp,
                }
                for index, timestamp in enumerate(timestamps)
            ],
            capture_outcome="never_calibrated",
            capture_provenance={"attempt_id": "failed-source-diagnostic"},
        ),
    )
    observations = iter(
        synthetic_observation(index + 1, timestamp_ns=timestamp, motion_fraction=0.5)
        .model_copy(
            update={
                "source_clock_id": "recorded-video-clock",
                "session_id": "failed-source-diagnostic",
            }
        )
        for index, timestamp in enumerate(timestamps)
    )

    class FakeRecordedVideoSource:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def frames(self):
            for index, timestamp in enumerate(timestamps):
                yield CapturedFrame(
                    sequence=index + 1,
                    source_clock_id="recorded-video-clock",
                    capture_mono_ns=timestamp,
                    image_bgr=numpy.zeros((16, 16, 3), dtype=numpy.uint8),
                    source_kind="video:raw.mp4",
                )

    class FakeDetector:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def detect(self, _frame, *, calibration_id: str):
            return next(observations).model_copy(update={"calibration_id": calibration_id})

    monkeypatch.setattr(cli, "RecordedVideoSource", FakeRecordedVideoSource)
    monkeypatch.setattr(cli, "MediaPipeHolisticDetector", FakeDetector)
    output = tmp_path / "diagnostic.json"
    assert cli.main(
        [
            "preview",
            "--video",
            str(raw_video),
            "--source-frame-map",
            str(frame_map),
            "--allow-failed-source",
            "--analysis-sync",
            "--max-frames",
            "2",
            "--output",
            str(output),
        ]
    ) == 0
    clip = MotionClip.load(output)
    assert clip.source_replay.model_dump(mode="json") == {
        "origin": "recorded-video-replay",
        "enabled": True,
        "frame_map_schema_version": 3,
        "capture_outcome": "never_calibrated",
        "allow_failed_source": True,
        "allow_legacy_source": False,
    }
    with pytest.raises(ExportError, match="unknown or diagnostic"):
        export_lerobot_v21((clip,), tmp_path / "diagnostic-export")


def test_live_preview_requires_a_complete_calibration_window_policy(tmp_path: Path) -> None:
    from galbot_motion_studio.cli import main

    with pytest.raises(SystemExit, match="calibration-window policy is required"):
        main(
            [
                "preview",
                "--camera",
                "0",
                "--output",
                str(tmp_path / "capture.json"),
                "--liveness-max-static-ms",
                "250",
            ]
        )


def test_calibration_window_configuration_cannot_be_partial(tmp_path: Path) -> None:
    from galbot_motion_studio.cli import main

    with pytest.raises(SystemExit, match="all-or-nothing"):
        main(
            [
                "preview",
                "--video",
                str(tmp_path / "input.mp4"),
                "--output",
                str(tmp_path / "capture.json"),
                "--calibration-window-ms",
                "200",
            ]
        )


def test_invalid_calibration_window_policy_exits_before_capture(tmp_path: Path) -> None:
    from galbot_motion_studio.cli import main

    with pytest.raises(SystemExit, match="invalid calibration-window policy"):
        main(
            [
                "preview",
                "--video",
                str(tmp_path / "input.mp4"),
                "--output",
                str(tmp_path / "capture.json"),
                "--calibration-window-ms",
                "200",
                "--calibration-min-samples",
                "1",
                "--calibration-max-center-deviation-normalized",
                "0.01",
                "--calibration-max-shoulder-width-deviation-normalized",
                "0.01",
                "--calibration-max-eye-span-deviation-normalized",
                "0.01",
            ]
        )


def test_calibration_accumulator_slides_over_old_eligible_samples() -> None:
    policy = CalibrationWindowPolicy(
        min_samples=2,
        max_window_ns=100,
        max_center_deviation_normalized=0.01,
        max_shoulder_width_deviation_normalized=0.01,
        max_eye_span_deviation_normalized=0.01,
    )
    samples = [
        SimpleNamespace(capture_mono_ns=0),
        SimpleNamespace(capture_mono_ns=50),
    ]

    dropped = _advance_calibration_window(
        samples, SimpleNamespace(capture_mono_ns=120), policy
    )

    assert dropped == 1
    assert [sample.capture_mono_ns for sample in samples] == [50, 120]


def test_source_frame_map_is_cryptographically_bound_to_its_video(tmp_path: Path) -> None:
    video = tmp_path / "operator.mp4"
    video.write_bytes(b"first-video")
    frame_map = tmp_path / "operator.frame-map.json"
    frame_map.write_text(
        dumps(
            {
                "schema_version": 1,
                "video": video.name,
                "video_sha256": _file_sha256(video),
                "frames": [
                    {
                        "video_frame_index": 0,
                        "source_sequence": 9,
                        "source_mono_ns": 123,
                    }
                ],
            }
        )
    )
    with pytest.raises(SystemExit, match="legacy"):
        _load_source_frame_map(frame_map, video)
    assert _load_source_frame_map(
        frame_map, video, allow_legacy_source=True
    )[0].source_sequence == 9

    video.write_bytes(b"different-video")
    with pytest.raises(SystemExit, match="video_sha256"):
        _load_source_frame_map(frame_map, video, allow_legacy_source=True)


def test_analysis_artifact_identity_separates_candidate_from_baseline() -> None:
    from galbot_motion_studio.recording import LivenessProvenance

    common = dict(
        input_hash="a" * 64,
        calibration_confidence=0.5,
        calibration_window=None,
        max_frames=None,
        liveness=LivenessProvenance(),
    )
    baseline = _analysis_artifact_identity(
        **common, mapping_hash="b" * 64, arm_mapping="wrist-primary"
    )
    candidate = _analysis_artifact_identity(
        **common, mapping_hash="c" * 64, arm_mapping="direction-vector"
    )
    assert baseline != candidate


def test_analysis_artifact_identity_includes_liveness_configuration() -> None:
    from galbot_motion_studio.recording import LivenessProvenance
    from galbot_motion_studio.vision.liveness import LivenessPolicy

    common = dict(
        input_hash="a" * 64,
        mapping_hash="b" * 64,
        arm_mapping="wrist-primary",
        calibration_confidence=0.5,
        calibration_window=None,
        max_frames=None,
    )
    disabled = _analysis_artifact_identity(**common, liveness=LivenessProvenance())
    enabled = _analysis_artifact_identity(
        **common,
        liveness=LivenessProvenance.from_policy(
            LivenessPolicy(max_static_ns=250_000_000),
            fingerprint_algorithm="sha256-bgr-pixels",
        ),
    )
    assert enabled != disabled


def test_analysis_artifact_identity_includes_calibration_window() -> None:
    from galbot_motion_studio.recording import LivenessProvenance
    from galbot_motion_studio.vision.calibration import CalibrationWindowPolicy

    common = dict(
        input_hash="a" * 64,
        mapping_hash="b" * 64,
        arm_mapping="wrist-primary",
        calibration_confidence=0.5,
        max_frames=None,
        liveness=LivenessProvenance(),
    )
    narrow = _analysis_artifact_identity(
        **common,
        calibration_window=CalibrationWindowPolicy(
            min_samples=10,
            max_window_ns=500_000_000,
            max_center_deviation_normalized=0.01,
            max_shoulder_width_deviation_normalized=0.01,
            max_eye_span_deviation_normalized=0.01,
        ),
    )
    wide = _analysis_artifact_identity(
        **common,
        calibration_window=CalibrationWindowPolicy(
            min_samples=10,
            max_window_ns=500_000_000,
            max_center_deviation_normalized=0.02,
            max_shoulder_width_deviation_normalized=0.01,
            max_eye_span_deviation_normalized=0.01,
        ),
    )
    assert narrow != wide


def _write_small_video(path: Path, frames: int = 3) -> None:
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16)
    )
    assert writer.isOpened()
    for value in range(frames):
        writer.write(numpy.full((16, 16, 3), value, dtype=numpy.uint8))
    writer.release()


def _raw_capture_documents(tmp_path: Path):
    video = tmp_path / "raw.mp4"
    _write_small_video(video)
    rows = [
        {
            "video_frame_index": index,
            "source_sequence": index + 10,
            "source_mono_ns": (index + 1) * 100,
        }
        for index in range(3)
    ]
    frame_map = tmp_path / "raw.frame-map.json"
    frame_map_document = _frame_map_document(
        video=video,
        artifact_kind=RAW_CAMERA_ARTIFACT,
        frames=rows,
        capture_outcome="succeeded",
        capture_provenance={"attempt_id": "test-raw-capture"},
    )
    sidecar = tmp_path / "raw.landmarks.json"
    sidecar_rows = [
        {
            "source_sequence": row["source_sequence"],
            "source_mono_ns": row["source_mono_ns"],
            "landmarks": {},
        }
        for row in rows
    ]
    sidecar_document = _landmark_sidecar_document(
        source_video=video,
        source_video_for_hash=video,
        source_frame_map=frame_map,
        frame_map_document=frame_map_document,
        frames=sidecar_rows,
    )
    return video, frame_map, frame_map_document, sidecar, sidecar_document


def test_raw_capture_requires_exact_video_map_and_sidecar_rows(tmp_path: Path) -> None:
    video, frame_map, frame_map_document, sidecar, sidecar_document = _raw_capture_documents(
        tmp_path
    )
    _validate_raw_capture_documents(
        video=video,
        final_video_name=video.name,
        final_frame_map_name=frame_map.name,
        frame_map_document=frame_map_document,
        sidecar_document=sidecar_document,
    )
    _write_json_document(frame_map, frame_map_document)
    _write_json_document(sidecar, sidecar_document)
    timeline = _load_source_frame_map(
        frame_map, video, required_artifact_kind=RAW_CAMERA_ARTIFACT
    )
    _load_landmark_sidecar(sidecar, video=video, frame_map=frame_map, timeline=timeline)

    short_map = dict(frame_map_document)
    short_map["frames"] = frame_map_document["frames"][:-1]
    with pytest.raises(SystemExit, match="row count"):
        _validate_raw_capture_documents(
            video=video,
            final_video_name=video.name,
            final_frame_map_name=frame_map.name,
            frame_map_document=short_map,
            sidecar_document=sidecar_document,
        )


def test_landmark_sidecar_is_bound_to_exact_map_bytes(tmp_path: Path) -> None:
    video, frame_map, frame_map_document, sidecar, sidecar_document = _raw_capture_documents(
        tmp_path
    )
    _write_json_document(frame_map, frame_map_document)
    _write_json_document(sidecar, sidecar_document)
    timeline = _load_source_frame_map(
        frame_map, video, required_artifact_kind=RAW_CAMERA_ARTIFACT
    )
    frame_map.write_text(frame_map.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(SystemExit, match="source_frame_map_sha256"):
        _load_landmark_sidecar(sidecar, video=video, frame_map=frame_map, timeline=timeline)


def test_raw_capture_publishes_only_after_staged_bytes_validate(tmp_path: Path) -> None:
    staged_video = tmp_path / ".raw.partial-test.mp4"
    _write_small_video(staged_video)
    final_video = tmp_path / "raw.mp4"
    frame_map = tmp_path / "raw.frame-map.json"
    rows = [
        {
            "video_frame_index": index,
            "source_sequence": index + 1,
            "source_mono_ns": (index + 1) * 100,
        }
        for index in range(3)
    ]
    frame_map_document = _frame_map_document(
        video=final_video,
        video_for_hash=staged_video,
        artifact_kind=RAW_CAMERA_ARTIFACT,
        frames=rows,
        capture_outcome="succeeded",
        capture_provenance={"attempt_id": "test-staged-capture"},
    )
    sidecar = tmp_path / "raw.landmarks.json"
    sidecar_document = _landmark_sidecar_document(
        source_video=final_video,
        source_video_for_hash=staged_video,
        source_frame_map=frame_map,
        frame_map_document=frame_map_document,
        frames=[
            {
                "source_sequence": row["source_sequence"],
                "source_mono_ns": row["source_mono_ns"],
                "landmarks": {},
            }
            for row in rows
        ],
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
    assert final_video.is_file()
    assert not staged_video.exists()
    timeline = _load_source_frame_map(
        frame_map, final_video, required_artifact_kind=RAW_CAMERA_ARTIFACT
    )
    _load_landmark_sidecar(sidecar, video=final_video, frame_map=frame_map, timeline=timeline)


def test_staging_path_keeps_video_suffix_for_opencv_encoder(tmp_path: Path) -> None:
    staged = _staging_path(tmp_path / "raw.mp4")
    assert staged.suffix == ".mp4"
    assert staged.parent == tmp_path
    assert staged.name.startswith(".raw.partial-")
