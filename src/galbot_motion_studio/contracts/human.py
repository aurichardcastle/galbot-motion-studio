"""Raw-cadence webcam-observation contracts; no camera SDK is imported here."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

from pydantic import Field, model_validator

from galbot_motion_studio.contracts.core import Contract, Envelope


class IdentityState(StrEnum):
    STABLE = "STABLE"
    LOST = "LOST"
    SWITCHED = "SWITCHED"
    AMBIGUOUS = "AMBIGUOUS"


class Landmark(Contract):
    name: str = Field(min_length=1)
    normalized_xyz: tuple[float, float, float]
    world_xyz_m: tuple[float, float, float] | None = None
    visibility: float = Field(ge=0, le=1)
    presence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def coordinates_are_finite(self) -> "Landmark":
        coordinates = self.normalized_xyz + (self.world_xyz_m or ())
        if not all(isfinite(value) for value in coordinates):
            raise ValueError("landmark coordinates must be finite")
        return self


class HumanObservation(Envelope):
    """One raw camera result. Sequence/timestamps are never resampled here."""

    camera_id: str = Field(min_length=1)
    calibration_id: str = Field(min_length=1)
    capture_mono_ns: int = Field(ge=0)
    inference_complete_mono_ns: int = Field(ge=0)
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    #: Opaque per-frame content evidence for a liveness monitor.  It is not an
    #: image reference and therefore does not relax the explicit recording gate.
    content_fingerprint: str | None = Field(default=None, min_length=1)
    track_id: str | None = None
    identity: IdentityState
    aggregate_confidence: float = Field(ge=0, le=1)
    landmarks: tuple[Landmark, ...]
    recording_enabled: bool = False
    persisted_image_ref: str | None = None

    @model_validator(mode="after")
    def observation_is_consistent(self) -> "HumanObservation":
        if self.inference_complete_mono_ns < self.capture_mono_ns:
            raise ValueError("inference time precedes capture time")
        names = tuple(landmark.name for landmark in self.landmarks)
        if len(names) != len(set(names)):
            raise ValueError("landmark names must be unique")
        if self.persisted_image_ref is not None and not self.recording_enabled:
            raise ValueError("image persistence requires explicit recording enable")
        return self
