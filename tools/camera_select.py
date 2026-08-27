"""Resolve the built-in camera's OpenCV index, and never hand back a phone.

Why this is not just "use index 0": OpenCV addresses cameras by position in
AVFoundation's device list and cannot name them. A Continuity Camera enters that
list only when the iPhone is actually available to the process, so the index of
the built-in camera is not stable -- it moves depending on whether a phone
happens to be nearby and awake. Selecting "index 0" therefore selects a
different physical camera at different times, which is exactly the failure this
project hit: captures kept coming from the phone.

Enumerating the same list OpenCV does, and matching on device TYPE rather than
position, makes the choice stable. Continuity and external devices are excluded
outright rather than deprioritised, because silently recording an operator from
the wrong camera is worse than failing.

Uses pyobjc when it is installed (exact, same ordering as OpenCV). Without it,
falls back to system_profiler, which lists the same devices but whose ordering
is only a good guess -- so the fallback says so rather than pretending.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

#: Model-id prefixes macOS reports for a Continuity Camera.
_PHONE_PREFIXES = ("iPhone", "iPad")


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    model: str
    builtin: bool
    phone: bool

    def describe(self) -> str:
        kind = "phone" if self.phone else ("built-in" if self.builtin else "external")
        return f"{self.index}  {self.name}  [{kind}]"


def _from_pyobjc() -> tuple[list[CameraDevice], bool]:
    try:
        import AVFoundation as AVF
    except ImportError:
        return [], False
    try:
        devices = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)
    except Exception:  # noqa: BLE001
        return [], False
    builtin_type = str(AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera)
    resolved = []
    for index, device in enumerate(devices):
        model = str(device.modelID())
        resolved.append(CameraDevice(
            index=index,
            name=str(device.localizedName()),
            model=model,
            builtin=str(device.deviceType()) == builtin_type,
            phone=model.startswith(_PHONE_PREFIXES),
        ))
    return resolved, True


def _from_system_profiler() -> tuple[list[CameraDevice], bool]:
    try:
        raw = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
        entries = json.loads(raw).get("SPCameraDataType", [])
    except Exception:  # noqa: BLE001
        return [], False
    resolved = []
    for index, entry in enumerate(entries):
        model = str(entry.get("spcamera_model-id", ""))
        phone = model.startswith(_PHONE_PREFIXES)
        resolved.append(CameraDevice(
            index=index,
            name=str(entry.get("_name", f"camera {index}")),
            model=model,
            builtin=not phone,
            phone=phone,
        ))
    return resolved, False


def enumerate_cameras() -> tuple[list[CameraDevice], bool]:
    """Devices plus whether the index mapping is exact rather than inferred."""
    devices, exact = _from_pyobjc()
    if devices:
        return devices, exact
    return _from_system_profiler()


def builtin_camera_index(default: int = 0) -> tuple[int, str]:
    """Index of the built-in camera, and a line explaining the choice."""
    devices, exact = enumerate_cameras()
    if not devices:
        return default, (
            f"no camera list available; falling back to index {default}. "
            "If this records from a phone, install pyobjc-framework-AVFoundation "
            "so the device can be identified by type."
        )

    listing = "; ".join(device.describe() for device in devices)
    builtin = [device for device in devices if device.builtin and not device.phone]
    if not builtin:
        phones = [device for device in devices if device.phone]
        if phones and len(devices) == len(phones):
            return default, (
                f"only phone cameras are available ({listing}). Refusing to pick one: "
                "a capture recorded from a Continuity Camera is not the operator's "
                "webcam and will drop out mid-run. Disconnect it or disable "
                "Continuity Camera on the phone."
            )
        return default, f"no built-in camera identified among: {listing}"

    chosen = builtin[0]
    exactness = "exact" if exact else "inferred from system_profiler order"
    return chosen.index, f"using index {chosen.index} ({chosen.name}); {exactness}. Seen: {listing}"


if __name__ == "__main__":
    index, why = builtin_camera_index()
    print(why)
    print(f"camera index: {index}")
