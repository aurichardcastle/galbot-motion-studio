"""Camera-frame boundary shared by recorded-video and future webcam sources."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    source_clock_id: str
    capture_mono_ns: int
    image_bgr: Any
    source_kind: str
    #: Opaque digest of camera content, if the capture adapter can provide one.
    #: It is metadata, never retained image content. The MediaPipe adapter derives
    #: a deterministic fallback from ``image_bgr`` when this is absent.
    content_fingerprint: str | None = None


def content_fingerprint_for(frame: CapturedFrame) -> str:
    """Return the canonical opaque digest shared by all same-frame consumers.

    A capture adapter may supply a digest when it has a hardware-backed source
    of truth.  Otherwise every independent inference provider must derive the
    same SHA-256 digest from the unmodified BGR payload.  Keeping the fallback
    here avoids an implicit, fragile agreement between the landmark adapter and
    occupancy provider about how a frame is identified.
    """
    if frame.content_fingerprint is not None:
        return frame.content_fingerprint
    image = frame.image_bgr
    if not hasattr(image, "tobytes"):
        raise ValueError("captured frame image cannot produce a content fingerprint")
    return sha256(image.tobytes()).hexdigest()
