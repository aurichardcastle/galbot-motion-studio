"""Canonical, lossless MotionClip recording for simulator sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping
from uuid import uuid4

from pydantic import Field, model_validator

from galbot_motion_studio.contracts.core import (
    Contract,
    RobotTarget,
    SafetyDecision,
    SafetyOutcome,
)
from galbot_motion_studio.contracts.human import HumanObservation
from galbot_motion_studio.pipeline import PipelineResult
from galbot_motion_studio.vision.calibration import CalibrationWindowPolicy
from galbot_motion_studio.vision.liveness import LivenessPolicy


class ClipFrame(Contract):
    source_sequence: int = Field(ge=0)
    source_mono_ns: int = Field(ge=0)
    target: RobotTarget | None
    decision: SafetyDecision
    observed_joints_rad: tuple[tuple[str, float], ...] | None = None
    sink: str | None = None
    # SafetyDecision describes whether a whole-frame command was permitted.  A
    # group can still be held while the remaining groups are safely commanded,
    # so persist that orthogonal fact rather than teaching offline analysis that
    # ALLOW means every limb was live.
    held_groups: tuple[str, ...] = ()
    held_group_reasons: tuple[tuple[str, str], ...] = ()
    #: Hand-derived gripper commands can be held even while the corresponding
    #: arm IK is healthy.  Preserve that separate truth instead of relabeling a
    #: healthy arm as failed in downstream analysis.
    held_grippers: tuple[str, ...] = ()
    held_gripper_reasons: tuple[tuple[str, str], ...] = ()
    #: A constrained group still has a legal command, unlike a held group, but
    #: must remain visible in replays because its saturated axis has stopped
    #: following the source pose.
    saturated_groups: tuple[str, ...] = ()
    saturated_group_reasons: tuple[tuple[str, str], ...] = ()
    #: Best-effort mapper diagnostics for local arm holds. These values are
    #: deliberately outside ``RobotTarget``: a rejected solve must never look
    #: like an authorized residual-bearing command.
    held_group_residuals_m: tuple[tuple[str, float], ...] = ()
    #: Ordered DLS-rung residuals for a locally rejected arm.  Kept alongside
    #: hold evidence, never in ``RobotTarget``, because an exhausted solver
    #: ladder is analysis data rather than command authority.
    held_group_ik_attempts: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()


class LivenessProvenance(Contract):
    """The liveness-gate configuration used for a recorded session.

    A missing monitor is a safety fact, not an omitted implementation detail.
    Disabled provenance carries no phantom setting; enabled provenance carries
    every decision that changes the monitor's verdict.
    """

    enabled: bool = False
    max_static_ns: int | None = Field(default=None, gt=0)
    max_history: int | None = Field(default=None, ge=1)
    require_monotonic_sequence: bool | None = None
    fingerprint_algorithm: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def configuration_is_complete_when_enabled(self) -> "LivenessProvenance":
        configuration = (
            self.max_static_ns,
            self.max_history,
            self.require_monotonic_sequence,
            self.fingerprint_algorithm,
        )
        if self.enabled:
            if any(value is None for value in configuration):
                raise ValueError("enabled liveness provenance requires complete configuration")
        elif any(value is not None for value in configuration):
            raise ValueError("disabled liveness provenance must not carry configuration")
        return self

    @classmethod
    def from_policy(
        cls,
        policy: LivenessPolicy | None,
        *,
        fingerprint_algorithm: str | None = None,
    ) -> "LivenessProvenance":
        if policy is None:
            if fingerprint_algorithm is not None:
                raise ValueError("disabled liveness must not name a fingerprint algorithm")
            return cls()
        if fingerprint_algorithm is None:
            raise ValueError("enabled liveness provenance requires a fingerprint algorithm")
        return cls(
            enabled=True,
            max_static_ns=policy.max_static_ns,
            max_history=policy.max_history,
            require_monotonic_sequence=policy.require_monotonic_sequence,
            fingerprint_algorithm=fingerprint_algorithm,
        )


class SourceReplayProvenance(Contract):
    """The origin and admission status of the material that produced a clip.

    ``unknown`` is the only safe interpretation for a legacy clip that lacks
    this field: absence cannot silently become a camera or synthetic attestation.
    ``recorded-video-replay`` records whether a source map was current and
    successful or deliberately admitted as diagnostic legacy/failed material.
    """

    origin: Literal[
        "unknown", "live-capture", "synthetic", "recorded-video-replay"
    ] = "unknown"
    enabled: bool = False
    frame_map_schema_version: int | None = Field(default=None, ge=1)
    capture_outcome: str | None = Field(default=None, min_length=1)
    allow_failed_source: bool = False
    allow_legacy_source: bool = False

    @model_validator(mode="after")
    def configuration_is_complete_and_honest(self) -> "SourceReplayProvenance":
        configuration_present = (
            self.frame_map_schema_version is not None
            or self.capture_outcome is not None
            or self.allow_failed_source
            or self.allow_legacy_source
        )
        if self.origin == "unknown":
            if (
                self.enabled
                or configuration_present
            ):
                raise ValueError("unknown source provenance must not carry an attestation")
            return self
        if self.origin in {"live-capture", "synthetic"}:
            if self.enabled or configuration_present:
                raise ValueError(f"{self.origin} provenance must not carry replay configuration")
            return self
        if not self.enabled or self.capture_outcome is None:
            raise ValueError("recorded-video replay provenance requires its admission outcome")
        if (
            self.frame_map_schema_version is None
            and self.capture_outcome != "unmapped-recorded-video"
        ):
            raise ValueError("mapped recorded-video replay requires a frame-map schema")
        if self.allow_failed_source and self.capture_outcome == "succeeded":
            raise ValueError("allow_failed_source cannot attest a succeeded capture")
        if (
            self.allow_legacy_source
            and self.frame_map_schema_version is not None
            and self.frame_map_schema_version >= 3
        ):
            raise ValueError("allow_legacy_source requires a pre-v3 source map")
        return self

    @property
    def publishable(self) -> bool:
        """Whether this clip has an explicit, publishable source attestation."""
        if self.origin in {"live-capture", "synthetic"}:
            return True
        return (
            self.origin == "recorded-video-replay"
            and self.enabled
            and self.frame_map_schema_version is not None
            and self.frame_map_schema_version >= 3
            and self.capture_outcome == "succeeded"
            and not self.allow_failed_source
            and not self.allow_legacy_source
        )


class CalibrationWindowProvenance(Contract):
    """The neutral-window policy used to authorize a recorded calibration.

    An absent policy is a meaningful fact for legacy/offline clips. A live
    session that claims quality-gated calibration must retain every threshold
    that made that claim true, not just a human-readable calibration ID.
    """

    enabled: bool = False
    min_samples: int | None = Field(default=None, ge=2)
    max_window_ns: int | None = Field(default=None, gt=0)
    max_center_deviation_normalized: float | None = Field(default=None, ge=0)
    max_shoulder_width_deviation_normalized: float | None = Field(default=None, ge=0)
    max_eye_span_deviation_normalized: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def configuration_is_complete_when_enabled(self) -> "CalibrationWindowProvenance":
        configuration = (
            self.min_samples,
            self.max_window_ns,
            self.max_center_deviation_normalized,
            self.max_shoulder_width_deviation_normalized,
            self.max_eye_span_deviation_normalized,
        )
        if self.enabled:
            if any(value is None for value in configuration):
                raise ValueError(
                    "enabled calibration-window provenance requires complete configuration"
                )
        elif any(value is not None for value in configuration):
            raise ValueError("disabled calibration-window provenance must not carry configuration")
        return self

    @classmethod
    def from_policy(
        cls, policy: CalibrationWindowPolicy | None
    ) -> "CalibrationWindowProvenance":
        if policy is None:
            return cls()
        return cls(
            enabled=True,
            min_samples=policy.min_samples,
            max_window_ns=policy.max_window_ns,
            max_center_deviation_normalized=policy.max_center_deviation_normalized,
            max_shoulder_width_deviation_normalized=(
                policy.max_shoulder_width_deviation_normalized
            ),
            max_eye_span_deviation_normalized=policy.max_eye_span_deviation_normalized,
        )


class CalibrationWindowEvidence(Contract):
    """Observed evidence for the specific neutral window that was accepted."""

    samples_used: int = Field(ge=2)
    window_span_ns: int = Field(ge=0)
    first_observation_sequence: int = Field(ge=0)
    last_observation_sequence: int = Field(ge=0)

    @model_validator(mode="after")
    def observations_are_ordered(self) -> "CalibrationWindowEvidence":
        if self.last_observation_sequence <= self.first_observation_sequence:
            raise ValueError("calibration-window evidence must span ordered observations")
        return self

    @classmethod
    def from_observations(
        cls, observations: tuple[HumanObservation, ...]
    ) -> "CalibrationWindowEvidence":
        if len(observations) < 2:
            raise ValueError("calibration-window evidence requires at least two observations")
        return cls(
            samples_used=len(observations),
            window_span_ns=(
                observations[-1].capture_mono_ns - observations[0].capture_mono_ns
            ),
            first_observation_sequence=observations[0].sequence,
            last_observation_sequence=observations[-1].sequence,
        )


class OccupancyProvenance(Contract):
    """Attestation for an independent operator-zone occupancy input.

    This reference implementation has no such adapter, so its default is an
    explicit disabled state. A future deployment cannot make a clip appear
    single-operator-safe merely by omitting the detector metadata.
    """

    enabled: bool = False
    provider_id: str | None = Field(default=None, min_length=1)
    detector_model_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    landmark_model_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    operator_zone_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    association_contract_version: str | None = Field(default=None, min_length=1)
    max_evidence_age_ns: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def configuration_is_complete_when_enabled(self) -> "OccupancyProvenance":
        configuration = (
            self.provider_id,
            self.detector_model_hash,
            self.landmark_model_hash,
            self.operator_zone_hash,
            self.association_contract_version,
            self.max_evidence_age_ns,
        )
        if self.enabled:
            if any(value is None for value in configuration):
                raise ValueError("enabled occupancy provenance requires complete configuration")
            if self.detector_model_hash == self.landmark_model_hash:
                raise ValueError(
                    "occupancy detector and landmark model must have distinct hashes"
                )
        elif any(value is not None for value in configuration):
            raise ValueError("disabled occupancy provenance must not carry configuration")
        return self

    @classmethod
    def from_configuration(
        cls,
        *,
        provider_id: str,
        detector_model_hash: str,
        landmark_model_hash: str,
        operator_zone_hash: str,
        association_contract_version: str,
        max_evidence_age_ns: int,
    ) -> "OccupancyProvenance":
        return cls(
            enabled=True,
            provider_id=provider_id,
            detector_model_hash=detector_model_hash,
            landmark_model_hash=landmark_model_hash,
            operator_zone_hash=operator_zone_hash,
            association_contract_version=association_contract_version,
            max_evidence_age_ns=max_evidence_age_ns,
        )


class MotionClip(Contract):
    schema_version: str = "1.0"
    clip_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    calibration_id: str = Field(min_length=1)
    model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_hash: str = Field(min_length=1)
    mapping_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: For deterministic retained-video analysis, distinguish the shared input
    #: from the exact mapper/configuration and implementation that consumed it.
    #: Live recordings intentionally leave these unset.
    input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    implementation_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source: Literal["sim"] = "sim"
    motion_profile: Literal["sim", "hardware"] = "sim"
    control_period_ns: int = Field(default=33_333_333, gt=0)
    initial_source_mono_ns: int | None = Field(default=None, ge=0)
    #: Explicitly distinguishes a clip that passed liveness from one recorded
    #: while that gate was absent. Defaults preserve legacy clip readability.
    liveness: LivenessProvenance = Field(default_factory=LivenessProvenance)
    source_replay: SourceReplayProvenance = Field(default_factory=SourceReplayProvenance)
    calibration_window: CalibrationWindowProvenance = Field(
        default_factory=CalibrationWindowProvenance
    )
    calibration_window_evidence: CalibrationWindowEvidence | None = None
    #: Explicitly records whether an independent occupancy detector gated this
    #: clip. Legacy and simulator clips are deliberately disabled, not unknown.
    occupancy: OccupancyProvenance = Field(default_factory=OccupancyProvenance)
    #: A terminal safety fault may originate from a rejected/reordered source
    #: frame. Keeping it outside the strictly monotonic event stream preserves
    #: the actual bad capture timestamp instead of dropping or rewriting it.
    terminal_fault: SafetyDecision | None = None
    state_equals_action: Literal[True] = True
    joint_order: tuple[str, ...] = Field(min_length=1)
    frames: tuple[ClipFrame, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def clip_is_ordered_and_consistent(self) -> "MotionClip":
        sequences = [frame.source_sequence for frame in self.frames]
        timestamps = [frame.source_mono_ns for frame in self.frames]
        if any(right <= left for left, right in zip(sequences, sequences[1:])):
            raise ValueError("clip frame sequences must strictly increase")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("clip frame timestamps must strictly increase")
        if (
            self.initial_source_mono_ns is not None
            and self.initial_source_mono_ns >= timestamps[0]
        ):
            raise ValueError("initial source timestamp must precede every clip frame")
        order = set(self.joint_order)
        for frame in self.frames:
            if frame.target is not None:
                names = {joint.name for joint in frame.target.joints}
                if names != order:
                    raise ValueError("every target must contain the clip joint order exactly")
                if frame.target.model_hash != self.model_hash:
                    raise ValueError("target model hash differs from clip")
                if frame.target.tool_hash != self.tool_hash:
                    raise ValueError("target tool hash differs from clip")
                if frame.target.mapping_hash != self.mapping_hash:
                    raise ValueError("target mapping hash differs from clip")
        if self.terminal_fault is not None and self.terminal_fault.outcome is not SafetyOutcome.FAULT:
            raise ValueError("terminal fault must have FAULT outcome")
        evidence = self.calibration_window_evidence
        if self.calibration_window.enabled:
            if evidence is None:
                raise ValueError("enabled calibration-window provenance requires evidence")
            if evidence.samples_used < self.calibration_window.min_samples:
                raise ValueError("calibration-window evidence has too few samples")
            if evidence.window_span_ns > self.calibration_window.max_window_ns:
                raise ValueError("calibration-window evidence exceeded the approved duration")
        elif evidence is not None:
            raise ValueError("disabled calibration-window provenance must not carry evidence")
        return self

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MotionClip":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class MotionRecorder:
    def __init__(
        self,
        *,
        task: str,
        calibration_id: str,
        joint_order: tuple[str, ...],
        motion_profile: Literal["sim", "hardware"] = "sim",
        control_period_ns: int = 33_333_333,
        initial_source_mono_ns: int | None = None,
        clip_id: str | None = None,
        analysis_arm_generation: str | None = None,
        input_hash: str | None = None,
        implementation_hash: str | None = None,
        liveness: LivenessProvenance | None = None,
        source_replay: SourceReplayProvenance | None = None,
        calibration_window: CalibrationWindowProvenance | None = None,
        occupancy: OccupancyProvenance | None = None,
    ) -> None:
        if not task or not calibration_id or not joint_order:
            raise ValueError("task, calibration_id, and joint_order are required")
        self.task = task
        self.calibration_id = calibration_id
        self.joint_order = joint_order
        self.motion_profile = motion_profile
        if control_period_ns <= 0:
            raise ValueError("control_period_ns must be positive")
        self.control_period_ns = control_period_ns
        if initial_source_mono_ns is not None and initial_source_mono_ns < 0:
            raise ValueError("initial source timestamp must be non-negative")
        self.initial_source_mono_ns = initial_source_mono_ns
        if clip_id is not None and not clip_id:
            raise ValueError("clip_id must be non-empty when provided")
        self.clip_id = clip_id
        if analysis_arm_generation is not None and not analysis_arm_generation:
            raise ValueError("analysis_arm_generation must be non-empty when provided")
        self.analysis_arm_generation = analysis_arm_generation
        for name, value in (
            ("input_hash", input_hash),
            ("implementation_hash", implementation_hash),
        ):
            if value is not None and len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest when provided")
        self.input_hash = input_hash
        self.implementation_hash = implementation_hash
        self.liveness = liveness or LivenessProvenance()
        self.source_replay = source_replay or SourceReplayProvenance()
        self.calibration_window = calibration_window or CalibrationWindowProvenance()
        self._calibration_window_evidence: CalibrationWindowEvidence | None = None
        self.occupancy = occupancy or OccupancyProvenance()
        self._frames: list[ClipFrame] = []
        self._terminal_fault: SafetyDecision | None = None
        #: Frames whose source timestamp did not advance past the previous one.
        #:
        #: A live webcam can hand two frames the same capture time under load --
        #: measured on a real 1196-frame session, which then lost the ENTIRE
        #: recording at save time to `MotionClip`'s strict-increase validator.
        #: Two frames stamped at the same instant are not distinct samples in
        #: time, so the second is dropped here rather than being allowed to
        #: invalidate the whole clip. The validator is deliberately not relaxed:
        #: strictly increasing time is what the v2.1 export depends on.
        self.non_monotonic_frames_dropped = 0

    def mark_initial_source(self, timestamp_ns: int) -> None:
        """Record the exact calibration/control seed used before frame one."""
        if timestamp_ns < 0:
            raise ValueError("initial source timestamp must be non-negative")
        if self._frames:
            raise ValueError("initial source timestamp must be recorded before frames")
        if self.initial_source_mono_ns is not None:
            raise ValueError("initial source timestamp is already recorded")
        self.initial_source_mono_ns = timestamp_ns

    def record_calibration_window(
        self, observations: tuple[HumanObservation, ...]
    ) -> None:
        """Bind an enabled policy to the exact neutral samples it approved."""
        if not self.calibration_window.enabled:
            raise ValueError("cannot record window evidence without an enabled policy")
        if self._frames:
            raise ValueError("calibration-window evidence must precede command frames")
        if self._calibration_window_evidence is not None:
            raise ValueError("calibration-window evidence is already recorded")
        evidence = CalibrationWindowEvidence.from_observations(observations)
        if evidence.samples_used < self.calibration_window.min_samples:
            raise ValueError("calibration-window evidence has too few samples")
        if evidence.window_span_ns > self.calibration_window.max_window_ns:
            raise ValueError("calibration-window evidence exceeded the approved duration")
        self._calibration_window_evidence = evidence

    @property
    def has_robot_target(self) -> bool:
        """Whether this recorder can produce a command-bearing MotionClip.

        Worker completion is not recording completion in latest-wins mode: a
        result can finish after the main thread has elected to stop and never be
        presented/recorded. Finalization must inspect this retained evidence,
        not a processor metric from another thread.
        """
        return any(frame.target is not None for frame in self._frames)

    def append(self, result: PipelineResult) -> None:
        observed = _ordered_items(result.observed_joints, self.joint_order)
        target = result.target
        decision = result.decision
        if self.analysis_arm_generation is not None:
            # The supervisor's arm generation is an intentionally random
            # capability token in a live session.  It is not an observation or
            # a model result, so retain neither its randomness nor the derived
            # fingerprint in a deterministic saved-video analysis artifact.
            if target is not None:
                target = target.model_copy(
                    update={"arm_generation": self.analysis_arm_generation}
                )
                decision = decision.model_copy(
                    update={
                        "arm_generation": self.analysis_arm_generation,
                        "target_fingerprint": (
                            target.fingerprint
                            if decision.outcome is SafetyOutcome.ALLOW
                            else None
                        ),
                    }
                )
            else:
                decision = decision.model_copy(
                    update={"arm_generation": self.analysis_arm_generation}
                )
        source_mono_ns = (
            result.target.source_mono_ns
            if result.target is not None
            else result.decision.source_mono_ns
        )
        if (
            not self._frames
            and self.initial_source_mono_ns is not None
            and source_mono_ns <= self.initial_source_mono_ns
        ):
            # The calibration seed must precede the first retained command. A
            # terminal ingress fault can legitimately expose this invalid source
            # time, so preserve it separately; any non-fault event is dropped
            # rather than deferred to MotionClip.finish() to fail much later.
            if decision.outcome is SafetyOutcome.FAULT:
                if self._terminal_fault is not None:
                    raise ValueError("recording already has a terminal fault")
                self._terminal_fault = decision
                return
            self.non_monotonic_frames_dropped += 1
            return
        # A terminal FAULT from a bad ingress frame must be auditable even when
        # that frame's source time is regressed (the very reason for the fault).
        # It cannot join the monotonic motion stream without rewriting its
        # timestamp, so retain it separately and preserve the original decision.
        if self._frames and (
            source_mono_ns <= self._frames[-1].source_mono_ns
            or result.observation_sequence <= self._frames[-1].source_sequence
        ):
            if decision.outcome is SafetyOutcome.FAULT:
                if self._terminal_fault is not None:
                    raise ValueError("recording already has a terminal fault")
                self._terminal_fault = decision
                return
            # Drop rather than corrupt. See non_monotonic_frames_dropped.
            self.non_monotonic_frames_dropped += 1
            return
        self._frames.append(
            ClipFrame(
                source_sequence=result.observation_sequence,
                source_mono_ns=source_mono_ns,
                target=target,
                decision=decision,
                observed_joints_rad=observed,
                sink=None if result.receipt is None else result.receipt.sink,
                held_groups=tuple(sorted(result.held_groups)),
                held_group_reasons=tuple(sorted(result.held_group_reasons)),
                held_grippers=tuple(sorted(result.held_grippers)),
                held_gripper_reasons=tuple(sorted(result.held_gripper_reasons)),
                saturated_groups=tuple(sorted(result.saturated_groups)),
                saturated_group_reasons=tuple(sorted(result.saturated_group_reasons)),
                held_group_residuals_m=tuple(sorted(result.held_group_residuals_m)),
                held_group_ik_attempts=tuple(sorted(result.held_group_ik_attempts)),
            )
        )

    def finish(self) -> MotionClip:
        if not self._frames:
            raise ValueError("cannot finish an empty recording")
        first_target = next((frame.target for frame in self._frames if frame.target is not None), None)
        if first_target is None:
            raise ValueError("recording contains no robot target")
        return MotionClip(
            clip_id=self.clip_id or str(uuid4()),
            task=self.task,
            calibration_id=self.calibration_id,
            model_hash=first_target.model_hash,
            tool_hash=first_target.tool_hash,
            # The recorder must attest to the mapper that produced the target.
            # Analysis candidates intentionally have a different fingerprint from
            # the wrist-primary production mapper.
            mapping_hash=first_target.mapping_hash,
            input_hash=self.input_hash,
            implementation_hash=self.implementation_hash,
            motion_profile=self.motion_profile,
            control_period_ns=self.control_period_ns,
            initial_source_mono_ns=self.initial_source_mono_ns,
            liveness=self.liveness,
            source_replay=self.source_replay,
            calibration_window=self.calibration_window,
            calibration_window_evidence=self._calibration_window_evidence,
            occupancy=self.occupancy,
            terminal_fault=self._terminal_fault,
            joint_order=self.joint_order,
            frames=tuple(self._frames),
        )


def _ordered_items(
    values: Mapping[str, float] | None,
    order: tuple[str, ...],
) -> tuple[tuple[str, float], ...] | None:
    if values is None:
        return None
    missing = [name for name in order if name not in values]
    if missing:
        raise ValueError(f"simulator readback is missing joints: {missing}")
    return tuple((name, float(values[name])) for name in order)
