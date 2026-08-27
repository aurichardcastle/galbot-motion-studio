"""OpenCV webcam frame source; it has no robot or command dependency."""

from __future__ import annotations

from importlib import import_module
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic_ns
from typing import Any, Iterator

from galbot_motion_studio.ports.frames import CapturedFrame


class WebcamError(RuntimeError):
    pass


class WebcamSource:
    def __init__(
        self,
        device: int = 0,
        *,
        source_clock_id: str = "local-monotonic",
        cv2_module: Any | None = None,
    ) -> None:
        # AVFoundation-backed OpenCV can terminate the process while tearing
        # down an invalid negative device index (observed as exit 139 on macOS),
        # rather than returning an unopened capture. Reject it before native
        # camera code runs so an operator gets a recoverable error instead.
        if device < 0:
            raise ValueError("webcam device must be non-negative")
        if not source_clock_id:
            raise ValueError("source_clock_id is required")
        self.device = device
        self.source_clock_id = source_clock_id
        self._cv2 = cv2_module

    def frames(self, *, max_frames: int | None = None) -> Iterator[CapturedFrame]:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        cv2 = self._cv2 or import_module("cv2")
        latest: Queue[CapturedFrame | BaseException] = Queue(maxsize=1)
        stop = Event()
        finished = Event()

        def publish(value: CapturedFrame | BaseException) -> None:
            try:
                latest.put_nowait(value)
                return
            except Full:
                pass
            try:
                latest.get_nowait()
            except Empty:
                pass
            latest.put_nowait(value)

        def capture_latest() -> None:
            capture = cv2.VideoCapture(self.device)
            try:
                if not capture.isOpened():
                    publish(WebcamError(f"cannot open webcam device {self.device}"))
                    return
                sequence = 0
                while not stop.is_set():
                    ok, image = capture.read()
                    timestamp = monotonic_ns()
                    if not ok:
                        publish(WebcamError("webcam frame capture failed"))
                        return
                    publish(
                        CapturedFrame(
                            sequence=sequence,
                            source_clock_id=self.source_clock_id,
                            capture_mono_ns=timestamp,
                            image_bgr=image,
                            source_kind=f"webcam:{self.device}",
                        )
                    )
                    sequence += 1
            finally:
                capture.release()
                finished.set()

        thread = Thread(
            target=capture_latest,
            name=f"webcam-{self.device}-latest",
            daemon=True,
        )
        thread.start()
        yielded = 0
        try:
            while max_frames is None or yielded < max_frames:
                try:
                    value = latest.get(timeout=0.25)
                except Empty:
                    if finished.is_set():
                        raise WebcamError("webcam capture stopped without a frame")
                    continue
                if isinstance(value, BaseException):
                    raise value
                yield value
                yielded += 1
        finally:
            stop.set()
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise WebcamError("webcam capture worker did not stop")


def macos_cameras() -> list[dict[str, str]]:
    """macOS camera list with names, which OpenCV's AVFoundation backend hides.

    OpenCV addresses cameras by index only, so an index cannot tell an operator
    whether they are about to record from the built-in camera or from a phone
    that wandered into Continuity range.  Returns [] on any non-macOS system or
    if system_profiler is unavailable -- callers must treat that as "unknown",
    never as "no cameras".
    """
    import json
    import subprocess

    try:
        raw = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
        devices = json.loads(raw).get("SPCameraDataType", [])
    except Exception:  # noqa: BLE001
        return []
    return [device for device in devices if isinstance(device, dict)]


def is_continuity_camera(device: dict[str, str]) -> bool:
    """Whether this device is an iPhone/iPad handed over by Continuity."""
    model = str(device.get("spcamera_model-id", ""))
    return model.startswith(("iPhone", "iPad"))


def resolve_builtin_camera(devices: list[dict[str, str]] | None = None) -> tuple[int | None, str]:
    """Index of the first non-Continuity camera, plus a human description.

    A Continuity Camera can be offered as index 0, so "0" is not a stable way
    to mean "the built-in webcam".  Pick the first device macOS does not report
    as a phone or tablet.  Returns (None, reason) when the list is unavailable
    or every device is a phone, so the caller can fall back rather than guess.
    """
    if devices is None:
        devices = macos_cameras()
    if not devices:
        return None, "macOS did not report a camera list"
    for index, device in enumerate(devices):
        if not is_continuity_camera(device):
            return index, str(device.get("_name", f"device {index}"))
    return None, "every reported camera is a Continuity (phone/tablet) device"


def avfoundation_cameras() -> list[dict[str, str]]:
    """Cameras in AVFoundation's own order -- the order OpenCV indexes.

    ``system_profiler`` lists devices in a DIFFERENT order from AVFoundation, so
    resolving an index from it silently picks the wrong camera (observed
    2026-08-26: system_profiler said FaceTime HD was first, while OpenCV index 0
    was a Continuity iPhone).  OpenCV's AVFoundation backend enumerates through
    the deprecated external-device path, so a Continuity Camera is included here
    and must be filtered by device TYPE, not by name.

    Returns [] when PyObjC is unavailable, so callers fall back rather than guess.
    """
    try:
        import AVFoundation as AVF  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return []
    devices = None
    try:
        discover = (
            AVF.AVCaptureDeviceDiscoverySession
            .discoverySessionWithDeviceTypes_mediaType_position_
        )
        session = discover(
            [
                "AVCaptureDeviceTypeBuiltInWideAngleCamera",
                "AVCaptureDeviceTypeExternal",
                "AVCaptureDeviceTypeContinuityCamera",
            ],
            "vide",
            0,
        )
        devices = list(session.devices())
    except Exception:  # noqa: BLE001
        devices = None
    if not devices:
        try:
            devices = list(AVF.AVCaptureDevice.devicesWithMediaType_("vide"))
        except Exception:  # noqa: BLE001
            return []
    listed: list[dict[str, str]] = []
    for device in devices:
        try:
            listed.append(
                {
                    "name": str(device.localizedName()),
                    "unique_id": str(device.uniqueID()),
                    "device_type": str(device.deviceType()),
                }
            )
        except Exception as error:  # noqa: BLE001
            # A device that will not describe itself cannot be matched by name or
            # unique id later, so it is skipped -- but never silently. A camera
            # missing from this list is precisely the confusing failure the
            # enumeration exists to prevent.
            print(f"skipping a camera that would not describe itself: {error!r}")
            continue
    return listed


def resolve_builtin_camera_avf() -> tuple[int | None, str, list[dict[str, str]]]:
    """Index of the built-in wide-angle camera in AVFoundation's own ordering.

    Matches on device TYPE rather than name: a Continuity Camera can carry a
    personal device name, so a name test cannot separate them reliably, while
    the type is unambiguous.
    """
    devices = avfoundation_cameras()
    if not devices:
        return None, "AVFoundation device list unavailable (PyObjC missing?)", []
    for index, device in enumerate(devices):
        if device.get("device_type") == "AVCaptureDeviceTypeBuiltInWideAngleCamera":
            return index, device.get("name", f"device {index}"), devices
    return None, "no built-in wide-angle camera reported by AVFoundation", devices
