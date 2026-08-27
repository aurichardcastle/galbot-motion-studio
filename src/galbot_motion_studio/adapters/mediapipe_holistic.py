"""Pinned MediaPipe Holistic adapter for simulator-only video and webcam inputs."""

from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from pathlib import Path
from time import monotonic_ns
from typing import Any

from galbot_motion_studio.contracts.human import HumanObservation, IdentityState, Landmark
from galbot_motion_studio.ports.frames import CapturedFrame, content_fingerprint_for


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGED_MODEL_ASSET = (
    Path(__file__).resolve().parents[1] / "_assets/mediapipe/holistic_landmarker.task"
)
DEFAULT_MODEL_ASSET = (
    PACKAGED_MODEL_ASSET
    if PACKAGED_MODEL_ASSET.is_file()
    else PROJECT_ROOT / "assets/mediapipe/holistic_landmarker.task"
)
DEFAULT_MODEL_SHA256 = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8"
# Elbows are deliberately excluded from the hard confidence aggregate. MediaPipe
# reports low elbow visibility whenever a forearm crosses the torso or face even
# while its tracked position remains stable. Elbows are a soft IK hint and have a
# separate lower threshold in the retargeter; wrists, shoulders and face remain
# the command-critical confidence gate.
COMMAND_CRITICAL_LANDMARK_NAMES = frozenset(
    {
        "face_1",
        "face_33",
        "face_263",
        "pose_11",
        "pose_12",
        "pose_15",
        "pose_16",
    }
)


class HolisticDetectorError(RuntimeError):
    pass


#: Consecutive frames without a usable pose that are absorbed before identity is
#: declared LOST.
#:
#: Identity answers one question -- "is this still the same person?" -- and nobody is
#: replaced within a fifth of a second. At the 15-25 FPS this loop actually achieves,
#: five frames is 200-340 ms: long enough to ride out MediaPipe's single-frame
#: flicker, short enough that a person genuinely leaving frame is caught almost
#: immediately.
IDENTITY_GRACE_FRAMES = 5


class MediaPipeHolisticDetector:
    """Converts a BGR frame to an immutable raw landmark observation.

    Holistic detects one person but does not prove continuity after an occlusion.
    The first complete frame after construction, or after identity was genuinely
    lost, is marked AMBIGUOUS; continuous tracking thereafter is STABLE.

    Identity is keyed on the **pose**, never on the face. Keying it on the face made
    turning your head destroy identity -- the exact gesture head tracking exists to
    serve. Measured on a real 1486-frame operator session, that produced 180+
    consecutive-run holds, every one ``IDENTITY_NOT_STABLE`` and not a single
    ``LOW_CONFIDENCE``. The terminal trace showed HOLD, HOLD, ALLOW repeating: a
    one-frame dropout cost two rejected frames, because recovery had to pass through
    AMBIGUOUS before reaching STABLE, so alternate-frame flicker never re-stabilised.

    Losing the face is still handled -- but as what it actually is, a *head* problem.
    The face landmarks are required by ``ControlGroup.head`` alone, so a head turn now
    holds the head while both arms keep tracking, instead of freezing the whole robot.
    """

    def __init__(self, model_asset: Path = DEFAULT_MODEL_ASSET) -> None:
        if not model_asset.is_file() or self._sha256_file(model_asset) != DEFAULT_MODEL_SHA256:
            raise HolisticDetectorError("MediaPipe model asset is missing or has an unapproved hash")
        try:
            self._mp = import_module("mediapipe")
        except ImportError as error:
            raise HolisticDetectorError("install the optional MediaPipe vision dependency") from error
        base_options = self._mp.tasks.BaseOptions(model_asset_path=str(model_asset))
        options = self._mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=self._mp.tasks.vision.RunningMode.VIDEO,
        )
        self._detector = self._mp.tasks.vision.HolisticLandmarker.create_from_options(options)
        # Held rather than re-resolved per frame, the same way `self._mp` is.
        # OpenCV is a hard dependency of this package; MediaPipe is the optional
        # one, which is why only that import is guarded above.
        self._cv2 = import_module("cv2")
        self._last_timestamp_ms = -1
        self._last_capture_mono_ns: int | None = None
        self._continuous_identity = False
        self._missing_pose_streak = 0

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "MediaPipeHolisticDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def detect(self, frame: CapturedFrame, *, calibration_id: str) -> HumanObservation:
        image = frame.image_bgr
        if not hasattr(image, "shape") or len(image.shape) != 3 or image.shape[2] != 3:
            raise HolisticDetectorError("expected a BGR image with three channels")
        if (
            self._last_capture_mono_ns is not None
            and frame.capture_mono_ns <= self._last_capture_mono_ns
        ):
            # MediaPipe requires increasing millisecond timestamps.  Its former
            # ``max(capture, last + 1)`` conversion silently hid a source-clock
            # regression, which made a stale/replayed camera frame appear fresh.
            # Reject at the capture-clock boundary before converting units; the
            # millisecond bump below is now only quantisation for distinct,
            # already-monotonic nanosecond captures.
            raise HolisticDetectorError("capture timestamp is not strictly monotonic")
        timestamp_ms = max(frame.capture_mono_ns // 1_000_000, self._last_timestamp_ms + 1)
        # `image[:, :, ::-1].copy()` walks a negative-stride view one byte at a
        # time: measured on a 1920x1080 frame, 200 reps each, interleaved, it
        # costs **p50 7.28 ms** (min 6.91, mean 7.51) against **p50 0.54 ms**
        # (min 0.42, mean 0.54) for the SIMD channel swap below -- and the two
        # outputs compare bitwise equal (np.array_equal True, both C-contiguous).
        # 7 ms is a fifth of this detector's whole per-frame cost and it sat on
        # the capture thread, so it was pure latency between the operator moving
        # and the twin hearing about it.
        rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        self._last_capture_mono_ns = frame.capture_mono_ns
        landmarks = self._to_landmarks(result)
        # Pose only. See the class docstring: a missing face is a head-group problem,
        # not evidence that the operator changed.
        has_pose = len(result.pose_landmarks) >= 33
        if has_pose:
            self._missing_pose_streak = 0
            if self._continuous_identity:
                identity = IdentityState.STABLE
            else:
                identity = IdentityState.AMBIGUOUS
                self._continuous_identity = True
        else:
            self._missing_pose_streak += 1
            if self._continuous_identity and self._missing_pose_streak <= IDENTITY_GRACE_FRAMES:
                # Coast. The person did not change in a few tens of milliseconds, so
                # STABLE is the truthful answer to the question identity asks. This
                # authorises nothing on its own: the landmarks really are absent, so
                # the observation gate still rejects the affected control groups on
                # MISSING_REQUIRED_LANDMARK. Each gate answers its own question.
                identity = IdentityState.STABLE
            else:
                identity = IdentityState.LOST
                self._continuous_identity = False
        inference_complete_ns = (
            monotonic_ns()
            if frame.source_kind.startswith("webcam:")
            else max(frame.capture_mono_ns, timestamp_ms * 1_000_000)
        )
        return HumanObservation(
            session_id=frame.source_clock_id,
            sequence=frame.sequence,
            source_clock_id=frame.source_clock_id,
            source_mono_ns=frame.capture_mono_ns,
            camera_id=frame.source_kind,
            calibration_id=calibration_id,
            capture_mono_ns=frame.capture_mono_ns,
            inference_complete_mono_ns=inference_complete_ns,
            image_width_px=int(image.shape[1]),
            image_height_px=int(image.shape[0]),
            content_fingerprint=content_fingerprint_for(frame),
            # A track exists whenever identity is being carried, including while
            # coasting through a brief dropout -- that is what "still the same
            # subject" means. It is None only once identity is genuinely LOST.
            track_id=(
                "holistic-single-subject"
                if identity is not IdentityState.LOST
                else None
            ),
            identity=identity,
            aggregate_confidence=self._aggregate_confidence(landmarks),
            landmarks=landmarks,
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _to_landmarks(result: Any) -> tuple[Landmark, ...]:
        output: list[Landmark] = []
        groups = (
            ("face", result.face_landmarks, None),
            ("pose", result.pose_landmarks, result.pose_world_landmarks),
            ("left_hand", result.left_hand_landmarks, result.left_hand_world_landmarks),
            ("right_hand", result.right_hand_landmarks, result.right_hand_world_landmarks),
        )
        for prefix, normalized, world in groups:
            for index, landmark in enumerate(normalized):
                world_landmark = world[index] if world is not None and index < len(world) else None
                output.append(
                    Landmark(
                        name=f"{prefix}_{index}",
                        normalized_xyz=(landmark.x, landmark.y, landmark.z),
                        world_xyz_m=(world_landmark.x, world_landmark.y, world_landmark.z)
                        if world_landmark is not None
                        else None,
                        visibility=MediaPipeHolisticDetector._optional_confidence(
                            landmark, "visibility"
                        ),
                        presence=MediaPipeHolisticDetector._optional_confidence(
                            landmark, "presence"
                        ),
                    )
                )
        # Holistic runs a dedicated hand tracker in addition to pose. When a hand
        # is present its wrist (landmark 0) is more precise and often remains at
        # full confidence while pose_15/16 briefly falls near zero. Fuse its
        # *image-normalized* wrist measurement into the pose wrist so the arm can
        # follow it. Never fuse ``hand_world_landmarks`` here: those coordinates
        # are hand-centred, whereas ``pose_world_landmarks`` are body-centred.
        # Mixing the two makes shoulder/elbow/wrist subtraction physically
        # meaningless. Dedicated hand-world points remain available under their
        # own ``left_hand_*``/``right_hand_*`` names for hand-only features.
        indexes = {landmark.name: index for index, landmark in enumerate(output)}
        for side, pose_name in (("left", "pose_15"), ("right", "pose_16")):
            hand_name = f"{side}_hand_0"
            if pose_name not in indexes or hand_name not in indexes:
                continue
            pose = output[indexes[pose_name]]
            hand = output[indexes[hand_name]]
            output[indexes[pose_name]] = pose.model_copy(
                update={
                    "normalized_xyz": hand.normalized_xyz,
                    "world_xyz_m": pose.world_xyz_m,
                    "visibility": max(pose.visibility, hand.visibility),
                    "presence": max(pose.presence, hand.presence),
                }
            )
        return tuple(output)

    @staticmethod
    def _optional_confidence(landmark: Any, field: str) -> float:
        """Treat MediaPipe's unsupported optional confidence fields as unknown/high.

        Hand and face landmarks expose ``visibility`` and ``presence`` attributes whose
        value is ``None`` rather than omitting the attribute.  Converting that directly
        with ``float(None)`` crashes the first real-person frame.
        """
        value = getattr(landmark, field, None)
        return 1.0 if value is None else float(value)

    @staticmethod
    def _aggregate_confidence(landmarks: tuple[Landmark, ...]) -> float:
        driving = [
            landmark
            for landmark in landmarks
            if landmark.name in COMMAND_CRITICAL_LANDMARK_NAMES
        ]
        if len(driving) != len(COMMAND_CRITICAL_LANDMARK_NAMES):
            return 0.0
        return min(min(landmark.visibility, landmark.presence) for landmark in driving)
