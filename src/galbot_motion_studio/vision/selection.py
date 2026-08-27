"""Fail-closed single-operator acquisition and continuity checks.

This module deliberately does not detect people.  It is the policy boundary for
an *independent* occupancy detector: the detector supplies a same-frame person
count and candidate centre, while this module decides whether an explicitly
acquired selection may continue to use its calibration.  A landmark-only
detector cannot make that decision because it is incapable of seeing a second
person by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import dist, isfinite

from galbot_motion_studio.contracts.human import HumanObservation, IdentityState


class SelectionState(StrEnum):
    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"


class SelectionRejection(StrEnum):
    NOT_ACQUIRED = "NOT_ACQUIRED"
    OBSERVATION_UNTRUSTED = "OBSERVATION_UNTRUSTED"
    OCCUPANCY_MISSING = "OCCUPANCY_MISSING"
    OCCUPANCY_AMBIGUOUS = "OCCUPANCY_AMBIGUOUS"
    OCCUPANCY_OUTSIDE_OPERATOR_ZONE = "OCCUPANCY_OUTSIDE_OPERATOR_ZONE"
    OCCUPANCY_LANDMARK_DISAGREEMENT = "OCCUPANCY_LANDMARK_DISAGREEMENT"
    CLOCK_MISMATCH = "CLOCK_MISMATCH"
    FRAME_REFERENCE_MISMATCH = "FRAME_REFERENCE_MISMATCH"
    TIMESTAMP_SKEW = "TIMESTAMP_SKEW"
    OCCUPANCY_FUTURE = "OCCUPANCY_FUTURE"
    OCCUPANCY_STALE = "OCCUPANCY_STALE"
    IDENTITY_NOT_STABLE = "IDENTITY_NOT_STABLE"
    SELECTION_GRACE_EXPIRED = "SELECTION_GRACE_EXPIRED"
    CUMULATIVE_OCCLUSION_EXPIRED = "CUMULATIVE_OCCLUSION_EXPIRED"
    CONTINUITY_FAILED = "CONTINUITY_FAILED"


@dataclass(frozen=True)
class OccupancyObservation:
    """Independent detector output associated with one capture instant.

    ``candidate_count`` is deliberately scoped to the configured operator zone,
    not the entire room.  A non-singleton result carries no candidate centre;
    using a stale centre from a previous frame would be an unsafe association.
    """

    camera_id: str
    source_clock_id: str
    source_sequence: int
    capture_mono_ns: int
    inference_complete_mono_ns: int
    content_fingerprint: str
    candidate_count: int
    candidate_center_normalized_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.camera_id or not self.source_clock_id:
            raise ValueError("occupancy camera and source clock IDs are required")
        if (
            self.source_sequence < 0
            or self.capture_mono_ns < 0
            or self.inference_complete_mono_ns < self.capture_mono_ns
            or self.candidate_count < 0
        ):
            raise ValueError(
                "occupancy sequence, timestamps, and candidate count must be valid"
            )
        if not self.content_fingerprint:
            raise ValueError("occupancy evidence requires a content fingerprint")
        center = self.candidate_center_normalized_xy
        if self.candidate_count == 1:
            if center is None or len(center) != 2 or not all(isfinite(value) for value in center):
                raise ValueError("a single occupancy candidate requires a finite centre")
            if not all(0 <= value <= 1 for value in center):
                raise ValueError("occupancy candidate centre must be normalized to [0, 1]")
        elif center is not None:
            raise ValueError("only a single occupancy candidate may have a centre")


@dataclass(frozen=True)
class SelectionPolicy:
    """Deployment-owned bounds for selection continuity.

    There are intentionally no defaults: these values must be measured and
    approved for the deployed camera zone rather than copied as source constants.
    """

    max_timestamp_skew_ns: int
    max_evidence_age_ns: int
    max_selection_grace_ns: int
    max_cumulative_occlusion_ns: int
    operator_zone_normalized_bounds: tuple[float, float, float, float]
    max_candidate_center_delta_normalized: float
    max_shoulder_width_relative_delta: float
    max_eye_span_relative_delta: float

    def __post_init__(self) -> None:
        if (
            self.max_timestamp_skew_ns < 0
            or self.max_evidence_age_ns <= 0
            or self.max_selection_grace_ns <= 0
            or self.max_cumulative_occlusion_ns <= 0
        ):
            raise ValueError(
                "selection timestamp skew, evidence age, and occlusion budgets must be non-negative/positive"
            )
        if self.max_cumulative_occlusion_ns < self.max_selection_grace_ns:
            raise ValueError("cumulative occlusion budget must cover one selection grace interval")
        x_min, y_min, x_max, y_max = self.operator_zone_normalized_bounds
        if (
            not all(isfinite(value) for value in self.operator_zone_normalized_bounds)
            or not 0 <= x_min < x_max <= 1
            or not 0 <= y_min < y_max <= 1
        ):
            raise ValueError("operator zone must be ordered normalized bounds inside [0, 1]")
        values = (
            self.max_candidate_center_delta_normalized,
            self.max_shoulder_width_relative_delta,
            self.max_eye_span_relative_delta,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("selection tolerances must be finite and non-negative")


@dataclass(frozen=True)
class SelectionDecision:
    state: SelectionState
    selection_id: int | None
    accepted: bool
    reason: SelectionRejection | None = None


@dataclass(frozen=True)
class _ContinuitySignature:
    candidate_center_normalized_xy: tuple[float, float]
    shoulder_width_normalized: float
    eye_span_normalized: float


class OperatorSelectionManager:
    """State machine for one explicitly acquired, non-biometric selection.

    A loss is always a whole-frame hold.  Re-locking is permitted only within the
    measured grace interval and only after positional plus anthropometric
    continuity checks.  A failure ends the generation; callers must explicitly
    acquire a new selection and calibration rather than inheriting the old one.
    """

    def __init__(self, policy: SelectionPolicy) -> None:
        self._policy = policy
        self._state = SelectionState.ACQUIRING
        self._selection_id: int | None = None
        self._next_selection_id = 1
        self._acquisition_signature: _ContinuitySignature | None = None
        self._last_signature: _ContinuitySignature | None = None
        self._camera_id: str | None = None
        self._source_clock_id: str | None = None
        self._occluded_since_ns: int | None = None
        self._cumulative_occluded_ns = 0

    @property
    def state(self) -> SelectionState:
        return self._state

    @property
    def selection_id(self) -> int | None:
        return self._selection_id

    @property
    def is_locked(self) -> bool:
        return self._state is SelectionState.LOCKED and self._selection_id is not None

    def reset(self) -> None:
        """Forget selection at a verified source boundary or explicit re-acquire."""
        self._state = SelectionState.ACQUIRING
        self._selection_id = None
        self._acquisition_signature = None
        self._last_signature = None
        self._camera_id = None
        self._source_clock_id = None
        self._occluded_since_ns = None
        self._cumulative_occluded_ns = 0

    def reject_acquisition(self) -> SelectionDecision:
        """Fail an explicit acquire without preserving a prior authorization."""
        self.reset()
        return SelectionDecision(
            SelectionState.ACQUIRING,
            None,
            False,
            SelectionRejection.OBSERVATION_UNTRUSTED,
        )

    def observe_untrusted(self) -> SelectionDecision:
        """End a selection when the shared frame gate cannot trust a frame.

        Base gate failures such as stale timing or frozen imagery occur before
        there is safe occupancy/landmark association to evaluate.  Preserving a
        LOCKED selection across that gap would let a recovered source reuse an
        unbounded old authorization, so recovery requires explicit acquisition.
        """
        if self._selection_id is None:
            return SelectionDecision(
                SelectionState.ACQUIRING,
                None,
                False,
                SelectionRejection.OBSERVATION_UNTRUSTED,
            )
        return self._lose(SelectionRejection.OBSERVATION_UNTRUSTED)

    def observe_liveness_gap(self, *, capture_mono_ns: int) -> SelectionDecision:
        """Hold through bounded liveness warm-up or loss without preserving motion.

        ``FRAME_NOT_LIVE`` cannot prove a content transition, but it is not by
        itself evidence that the acquired operator changed. It therefore starts
        the same bounded gap as an occupancy omission: no frame is accepted, a
        sustained source stall expires the generation, and a later live frame
        still has to pass exact association and continuity before re-locking.
        """
        if capture_mono_ns < 0:
            raise ValueError("capture timestamp must be non-negative")
        if (
            self._selection_id is None
            or self._acquisition_signature is None
            or self._last_signature is None
        ):
            return SelectionDecision(
                SelectionState.ACQUIRING,
                None,
                False,
                SelectionRejection.OBSERVATION_UNTRUSTED,
            )
        return self._occlude(capture_mono_ns, SelectionRejection.OBSERVATION_UNTRUSTED)

    def validate_calibration_evidence(
        self,
        occupancy: OccupancyObservation,
        observation: HumanObservation,
        *,
        now_mono_ns: int,
    ) -> SelectionDecision:
        """Validate an archived neutral sample without consuming live state.

        Window duration is checked by the calibration policy. Applying the live
        frame-age budget when the finished window is committed would reject the
        earliest valid sample solely because it was retained in that window.
        This preserves exact association, zone, identity, continuity, and
        future-inference checks without advancing the selection or liveness
        state machine during validation.
        """
        if not self.is_locked:
            return SelectionDecision(
                self._state,
                self._selection_id,
                False,
                SelectionRejection.NOT_ACQUIRED,
            )
        rejection = self._frame_rejection(
            occupancy,
            observation,
            now_mono_ns,
            enforce_evidence_age=False,
        )
        if rejection is not None:
            return SelectionDecision(self._state, self._selection_id, False, rejection)
        try:
            signature = _signature(occupancy, observation)
        except ValueError:
            return SelectionDecision(
                self._state,
                self._selection_id,
                False,
                SelectionRejection.OCCUPANCY_LANDMARK_DISAGREEMENT,
            )
        # Candidate-centre displacement over a retained neutral window is owned
        # by CalibrationWindowPolicy.max_center_deviation_normalized. Applying
        # the live per-frame centre delta here would silently turn it into a
        # total-window tolerance as this pure validation intentionally does not
        # advance _last_signature. Anthropometry remains acquisition-anchored:
        # it is the continuity evidence selection owns.
        if not self._has_matching_anthropometry(signature):
            return SelectionDecision(
                self._state,
                self._selection_id,
                False,
                SelectionRejection.CONTINUITY_FAILED,
            )
        return SelectionDecision(self._state, self._selection_id, True)

    def acquire(
        self,
        occupancy: OccupancyObservation,
        observation: HumanObservation,
        *,
        now_mono_ns: int,
    ) -> SelectionDecision:
        """Start a new generation after an explicit acquisition action."""
        rejection = self._frame_rejection(occupancy, observation, now_mono_ns)
        if rejection is not None:
            self.reset()
            return SelectionDecision(self._state, None, False, rejection)
        self._selection_id = self._next_selection_id
        self._next_selection_id += 1
        signature = _signature(occupancy, observation)
        self._acquisition_signature = signature
        self._last_signature = signature
        self._camera_id = observation.camera_id
        self._source_clock_id = observation.source_clock_id
        self._occluded_since_ns = None
        self._cumulative_occluded_ns = 0
        self._state = SelectionState.LOCKED
        return SelectionDecision(self._state, self._selection_id, True)

    def observe(
        self,
        occupancy: OccupancyObservation,
        observation: HumanObservation,
        *,
        now_mono_ns: int,
    ) -> SelectionDecision:
        """Evaluate a new frame without creating a selection implicitly."""
        if (
            self._selection_id is None
            or self._acquisition_signature is None
            or self._last_signature is None
        ):
            return SelectionDecision(
                SelectionState.ACQUIRING,
                None,
                False,
                SelectionRejection.NOT_ACQUIRED,
            )

        rejection = self._frame_rejection(occupancy, observation, now_mono_ns)
        if rejection is not None:
            if rejection is SelectionRejection.IDENTITY_NOT_STABLE:
                return self._occlude(observation.capture_mono_ns, rejection)
            return self._lose(rejection)

        signature = _signature(occupancy, observation)
        if self._state is SelectionState.OCCLUDED:
            if self._occluded_since_ns is None:
                return self._lose(SelectionRejection.CONTINUITY_FAILED)
            if observation.capture_mono_ns - self._occluded_since_ns > self._policy.max_selection_grace_ns:
                return self._lose(SelectionRejection.SELECTION_GRACE_EXPIRED)
            if not self._is_continuous(signature):
                return self._lose(SelectionRejection.CONTINUITY_FAILED)
            gap_ns = observation.capture_mono_ns - self._occluded_since_ns
            if self._cumulative_occluded_ns + gap_ns > self._policy.max_cumulative_occlusion_ns:
                return self._lose(SelectionRejection.CUMULATIVE_OCCLUSION_EXPIRED)
            self._cumulative_occluded_ns += gap_ns
            self._state = SelectionState.LOCKED
            self._occluded_since_ns = None
            self._last_signature = signature
            return SelectionDecision(self._state, self._selection_id, True)

        if self._state is not SelectionState.LOCKED:
            return self._lose(SelectionRejection.NOT_ACQUIRED)
        if not self._is_continuous(signature):
            return self._lose(SelectionRejection.CONTINUITY_FAILED)
        self._last_signature = signature
        return SelectionDecision(self._state, self._selection_id, True)

    def observe_missing_occupancy(self, *, capture_mono_ns: int) -> SelectionDecision:
        """Enter a bounded continuity gap when independent evidence is absent.

        A missing occupancy result is not neutral evidence.  It suspends the
        selection immediately, starts the same grace budget used for landmark
        occlusion, and requires a later exact-frame continuity check before the
        lock may resume.  This prevents a transient detector failure from
        silently extending an old authorization indefinitely.
        """
        if capture_mono_ns < 0:
            raise ValueError("capture timestamp must be non-negative")
        if (
            self._selection_id is None
            or self._acquisition_signature is None
            or self._last_signature is None
        ):
            return SelectionDecision(
                SelectionState.ACQUIRING,
                None,
                False,
                SelectionRejection.OCCUPANCY_MISSING,
            )
        return self._occlude(capture_mono_ns, SelectionRejection.OCCUPANCY_MISSING)

    def _frame_rejection(
        self,
        occupancy: OccupancyObservation,
        observation: HumanObservation,
        now_mono_ns: int,
        *,
        enforce_evidence_age: bool = True,
    ) -> SelectionRejection | None:
        if occupancy.camera_id != observation.camera_id or occupancy.source_clock_id != observation.source_clock_id:
            return SelectionRejection.CLOCK_MISMATCH
        if (
            self._camera_id is not None
            and (
                observation.camera_id != self._camera_id
                or observation.source_clock_id != self._source_clock_id
            )
        ):
            return SelectionRejection.CLOCK_MISMATCH
        if (
            occupancy.source_sequence != observation.sequence
            or occupancy.content_fingerprint != observation.content_fingerprint
            or observation.content_fingerprint is None
        ):
            return SelectionRejection.FRAME_REFERENCE_MISMATCH
        if abs(occupancy.capture_mono_ns - observation.capture_mono_ns) > self._policy.max_timestamp_skew_ns:
            return SelectionRejection.TIMESTAMP_SKEW
        if occupancy.inference_complete_mono_ns > now_mono_ns:
            return SelectionRejection.OCCUPANCY_FUTURE
        if (
            enforce_evidence_age
            and now_mono_ns - occupancy.inference_complete_mono_ns
            > self._policy.max_evidence_age_ns
        ):
            return SelectionRejection.OCCUPANCY_STALE
        if occupancy.candidate_count != 1:
            return SelectionRejection.OCCUPANCY_AMBIGUOUS
        if not self._in_operator_zone(occupancy.candidate_center_normalized_xy):
            return SelectionRejection.OCCUPANCY_OUTSIDE_OPERATOR_ZONE
        if observation.identity is not IdentityState.STABLE:
            return SelectionRejection.IDENTITY_NOT_STABLE
        try:
            _signature(occupancy, observation)
        except ValueError:
            return SelectionRejection.OCCUPANCY_LANDMARK_DISAGREEMENT
        return None

    def _is_continuous(self, current: _ContinuitySignature) -> bool:
        """Check motion locally but anthropometry against the acquired operator.

        Candidate position legitimately moves throughout a session, so it is
        compared with the immediately prior accepted frame.  Shoulder width and
        eye span are the non-biometric continuity evidence: comparing them only
        to a sliding prior frame would let a gradual substitution ratchet through
        one small legal change at a time.
        """
        assert self._last_signature is not None
        assert self._acquisition_signature is not None
        if (
            dist(current.candidate_center_normalized_xy, self._last_signature.candidate_center_normalized_xy)
            > self._policy.max_candidate_center_delta_normalized
        ):
            return False
        return self._has_matching_anthropometry(current)

    def _has_matching_anthropometry(self, current: _ContinuitySignature) -> bool:
        """Compare identity-bearing body scale only with the acquisition frame."""
        assert self._acquisition_signature is not None
        return (
            _relative_delta(
                current.shoulder_width_normalized,
                self._acquisition_signature.shoulder_width_normalized,
            )
            <= self._policy.max_shoulder_width_relative_delta
            and _relative_delta(
                current.eye_span_normalized,
                self._acquisition_signature.eye_span_normalized,
            )
            <= self._policy.max_eye_span_relative_delta
        )

    def _in_operator_zone(self, center: tuple[float, float] | None) -> bool:
        if center is None:
            return False
        x_min, y_min, x_max, y_max = self._policy.operator_zone_normalized_bounds
        x, y = center
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _occlude(
        self, capture_mono_ns: int, reason: SelectionRejection
    ) -> SelectionDecision:
        """Hold through a bounded evidence gap without accepting a frame."""
        if self._state is SelectionState.LOCKED:
            self._state = SelectionState.OCCLUDED
            self._occluded_since_ns = capture_mono_ns
        elif self._state is not SelectionState.OCCLUDED or self._occluded_since_ns is None:
            return self._lose(SelectionRejection.CONTINUITY_FAILED)
        if capture_mono_ns < self._occluded_since_ns:
            return self._lose(SelectionRejection.CONTINUITY_FAILED)
        if capture_mono_ns - self._occluded_since_ns > self._policy.max_selection_grace_ns:
            return self._lose(SelectionRejection.SELECTION_GRACE_EXPIRED)
        return SelectionDecision(self._state, self._selection_id, False, reason)

    def _lose(self, reason: SelectionRejection) -> SelectionDecision:
        self._state = SelectionState.LOST
        self._selection_id = None
        self._acquisition_signature = None
        self._last_signature = None
        self._camera_id = None
        self._source_clock_id = None
        self._occluded_since_ns = None
        return SelectionDecision(self._state, None, False, reason)


def _signature(
    occupancy: OccupancyObservation, observation: HumanObservation
) -> _ContinuitySignature:
    if occupancy.candidate_center_normalized_xy is None:
        raise ValueError("no single occupancy candidate")
    landmarks = {landmark.name: landmark for landmark in observation.landmarks}
    try:
        left_shoulder = landmarks["pose_11"]
        right_shoulder = landmarks["pose_12"]
        left_eye = landmarks["face_33"]
        right_eye = landmarks["face_263"]
    except KeyError as error:
        raise ValueError(f"missing continuity landmark {error.args[0]}") from error
    if min(
        left_shoulder.visibility,
        right_shoulder.visibility,
        left_eye.visibility,
        right_eye.visibility,
    ) < 0.5:
        raise ValueError("low-confidence continuity landmark")
    shoulder_width = dist(left_shoulder.normalized_xyz, right_shoulder.normalized_xyz)
    eye_span = dist(left_eye.normalized_xyz, right_eye.normalized_xyz)
    if shoulder_width <= 0 or eye_span <= 0:
        raise ValueError("degenerate continuity anthropometry")
    return _ContinuitySignature(
        occupancy.candidate_center_normalized_xy,
        shoulder_width,
        eye_span,
    )


def _relative_delta(current: float, reference: float) -> float:
    if reference <= 0:
        return float("inf")
    return abs(current - reference) / reference
