"""Independent, same-frame occupancy boundary for production perception."""

from __future__ import annotations

from typing import Protocol

from galbot_motion_studio.ports.frames import CapturedFrame
from galbot_motion_studio.vision.selection import OccupancyObservation


class OccupancyProvider(Protocol):
    """Report operator-zone occupancy from an independent model.

    The returned observation must cite the supplied frame's sequence, capture
    clock, and canonical capture-frame fingerprint value. Implementations may
    obtain that fallback through ``ports.frames.content_fingerprint_for``; they
    may not derive the count from the landmark estimator, because it cannot
    observe a second person.
    """

    def observe(self, frame: CapturedFrame) -> OccupancyObservation: ...
