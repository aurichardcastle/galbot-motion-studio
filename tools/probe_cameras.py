"""Which capture devices actually open, and what they report.

Opens no preview window and writes no capture. Its whole purpose is to turn
"the webcam does not work" into a specific device index, because the failures
seen on this machine look identical from the outside: a Continuity Camera that
drops out, a permission denial, and a genuinely absent device all surface as
"cannot open webcam device N".
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2  # noqa: E402


def _macos_cameras() -> list[dict[str, str]]:
    """Device names from system_profiler, which OpenCV cannot give us.

    OpenCV's AVFoundation backend addresses cameras by index and exposes no
    name, so an index alone cannot tell you whether you are about to record
    from the built-in camera or from a phone that wandered into range.
    """
    import json
    import subprocess

    try:
        raw = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
        return json.loads(raw).get("SPCameraDataType", [])
    except Exception:  # noqa: BLE001
        return []


def _is_phone(device: dict[str, str]) -> bool:
    model = str(device.get("spcamera_model-id", ""))
    return model.startswith(("iPhone", "iPad"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    parser.add_argument("--max-index", type=int, default=4)
    args = parser.parse_args()

    print(f"opencv {cv2.__version__}")
    devices = _macos_cameras()
    if devices:
        print("\nmacOS reports:")
        for order, device in enumerate(devices):
            kind = "CONTINUITY (phone)" if _is_phone(device) else "built-in / wired"
            print(f"  {order}  {device.get('_name','?'):<28} "
                  f"{device.get('spcamera_model-id','?'):<22} {kind}")
        if any(_is_phone(d) for d in devices):
            print("  a Continuity Camera is present: macOS can hand it out as index 0.")
        print()
    print(f"{'idx':>4} {'opens':>6} {'reads':>6} {'size':>11} {'fps':>6}")
    working = []
    for index in range(args.max_index + 1):
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            print(f"{index:>4} {'no':>6} {'-':>6} {'-':>11} {'-':>6}")
            capture.release()
            continue
        ok, frame = capture.read()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = f"{width}x{height}" if ok else "-"
        print(f"{index:>4} {'yes':>6} {'yes' if ok else 'no':>6} {size:>11} {fps:>6.1f}")
        if ok and frame is not None:
            working.append(index)
        capture.release()

    if not working:
        print(
            "\nNo device produced a frame. On macOS this is almost always the "
            "camera privacy grant: it is given per application, and a child "
            "process can only inherit it. Grant it to the app that launched "
            "this, in System Settings > Privacy & Security > Camera, then start "
            "it again from that same app.",
            file=sys.stderr,
        )
        return 1
    print(f"\nusable indices: {working}")
    print("Pick the one that is NOT your phone and set it in the HUD camera "
          "selector, or pass --camera N to preview directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
