"""Fail-closed freshness, sequence, confidence, and identity gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

from galbot_motion_studio.contracts.human import HumanObservation, IdentityState
from galbot_motion_studio.vision.liveness import FrameLivenessMonitor, LivenessPolicy
from galbot_motion_studio.vision.selection import (
    OccupancyObservation,
    OperatorSelectionManager,
    SelectionDecision,
)


class ObservationRejection(StrEnum):
    CLOCK_MISMATCH = "CLOCK_MISMATCH"
    REORDERED = "REORDERED"
    FUTURE_CAPTURE = "FUTURE_CAPTURE"
    FUTURE_INFERENCE = "FUTURE_INFERENCE"
    STALE = "STALE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    IDENTITY_NOT_STABLE = "IDENTITY_NOT_STABLE"
    FRAME_NOT_LIVE = "FRAME_NOT_LIVE"
    OCCUPANCY_MISSING = "OCCUPANCY_MISSING"
    SELECTION_NOT_LOCKED = "SELECTION_NOT_LOCKED"
    MISSING_REQUIRED_LANDMARK = "MISSING_REQUIRED_LANDMARK"


#: Rejections that are properties of the FRAME, not of any one body part.  A
#: mismatched clock, a replayed or reordered frame, a timestamp from the future,
#: a stale frame, or an identity that is not provably the same person says
#: nothing about which landmark is visible: the whole observation is
#: untrustworthy, so every control group holds.  These are never per-group.
WHOLE_FRAME_REJECTIONS = frozenset(
    {
        ObservationRejection.CLOCK_MISMATCH,
        ObservationRejection.REORDERED,
        ObservationRejection.FUTURE_CAPTURE,
        ObservationRejection.FUTURE_INFERENCE,
        ObservationRejection.STALE,
        ObservationRejection.IDENTITY_NOT_STABLE,
        ObservationRejection.FRAME_NOT_LIVE,
        ObservationRejection.OCCUPANCY_MISSING,
        ObservationRejection.SELECTION_NOT_LOCKED,
    }
)

#: Precedence used when several groups fail for different reasons, and for the
#: single ``reason`` this gate reports to legacy callers.  It matches the order
#: the checks were written in before gating became per-group, so a caller that
#: only reads ``ObservationGateResult.reason`` sees exactly what it saw before.
_GROUP_REJECTION_PRECEDENCE = (
    ObservationRejection.LOW_CONFIDENCE,
    ObservationRejection.MISSING_REQUIRED_LANDMARK,
)


class ControlGroup(StrEnum):
    """Independently commandable parts of the robot.

    Confidence and landmark presence are judged per group.  A command is
    untrustworthy only when the landmarks that DRIVE THAT COMMAND are
    untrustworthy, so one occluded wrist holds one arm instead of holding the
    head and the other, perfectly tracked arm with it.  This is more precise
    than a single frame-wide minimum, not more permissive: no threshold moves,
    and every group is still judged against the same ``min_confidence``.
    """

    HEAD = "head"
    TORSO = "torso"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"


@dataclass(frozen=True)
class ControlGroupPolicy:
    """The landmarks one control group depends on.

    ``required`` must all be present for the group to be evaluable at all;
    a group whose landmarks are missing is unevaluable and therefore HOLDS.

    ``confidence_critical`` is the subset whose confidence actually drives the
    command.  It is a strict subset for the arms because the retargeter treats
    the elbow as a soft IK hint with its own, lower threshold
    (``retarget/left_arm.py::estimate_arm_intent`` skips the elbow in its
    occlusion check), and MediaPipe routinely reports a correctly tracked elbow
    below 0.5 whenever a forearm crosses the torso or face.
    """

    required: frozenset[str]
    confidence_critical: frozenset[str]
    #: Additional landmark sets that can stand in for ``required``, each paired
    #: with its own confidence-critical subset. The group is evaluable when
    #: ``required`` is satisfied OR any alternative is, and the confidence that is
    #: then judged is the one belonging to the set that was actually used.
    #:
    #: This exists for the head. Its natural basis is the face mesh, but the mesh
    #: is frequently absent -- measured on the 2026-08-22 trial the head was held
    #: on 20.4% of frames for a missing ``face_1/33/263`` while the pose model's
    #: own nose and eyes were present on 94.9%. Both bases produce a valid head
    #: pose against their own calibrated neutral, so requiring only the less
    #: reliable one froze the head for no safety benefit.
    #:
    #: Alternatives are not a relaxation: each set is judged against the same
    #: ``min_confidence`` as ``required``, and a group with no satisfied set still
    #: holds.
    alternatives: tuple[tuple[frozenset[str], frozenset[str]], ...] = ()

    def satisfied_by(
        self, present: "frozenset[str] | set[str] | Mapping[str, object]"
    ) -> "frozenset[str] | None":
        """The confidence-critical set to judge, or ``None`` if nothing is usable."""
        names = present.keys() if hasattr(present, "keys") else present
        if self.required <= names:
            return self.confidence_critical
        for required, critical in self.alternatives:
            if required <= names:
                return critical
        return None

    def __post_init__(self) -> None:
        if not self.required or not self.confidence_critical:
            raise ValueError("a control group needs required and confidence-critical landmarks")
        for required, critical in self.alternatives:
            if not required or not critical:
                raise ValueError("an alternative needs required and confidence-critical landmarks")
            if not critical <= required:
                raise ValueError("alternative confidence-critical names must be required")
        if any(not name for name in self.required | self.confidence_critical):
            raise ValueError("control group landmark names must be non-empty")
        if not self.confidence_critical <= self.required:
            raise ValueError("confidence-critical landmarks must also be required landmarks")


#: Landmark -> control group mapping, read off the code that consumes them.
#:
#: * head: ``retarget/face_pose.py`` reads face_1 (nose) and face_33/face_263
#:   (eyes) and nothing else.
#: * arms: ``retarget/left_arm.py::_arm_landmark_names`` gives (shoulder, elbow,
#:   wrist) = pose_11/13/15 for the left and pose_12/14/16 for the right, and
#:   ``estimate_arm_intent`` additionally reads BOTH shoulders on every frame
#:   because shoulder span is the scale normaliser.  pose_11 and pose_12 are
#:   therefore required by both arms, not one each.
#:
#: The union of ``confidence_critical`` here is exactly
#: ``adapters/mediapipe_holistic.COMMAND_CRITICAL_LANDMARK_NAMES``, the set the
#: producer takes its ``aggregate_confidence`` minimum over.  That equality is
#: what makes the frame-wide aggregate exactly decomposable into these groups;
#: see ``ObservationGate._aggregate_is_attributable``.
COMMAND_CONTROL_GROUPS: Mapping[ControlGroup, ControlGroupPolicy] = {
    # Gated on the POSE model's nose and eyes, not the face mesh's. Two reasons,
    # both measured on the 2026-08-22 trial: the mesh is absent far more often
    # (the head was held on 20.4% of frames for a missing `face_1/33/263` while
    # these were present on 94.9%), and face-mesh landmarks carry no visibility or
    # presence value at all -- the adapter reports 1.0 for an unset field, so
    # gating the head's CONFIDENCE on them was never a real check. These carry
    # real values. `estimate_face_pose` still prefers the mesh when the frame has
    # it; this only decides when the head is allowed to be commanded.
    ControlGroup.HEAD: ControlGroupPolicy(
        required=frozenset({"face_1", "face_33", "face_263"}),
        confidence_critical=frozenset({"face_1", "face_33", "face_263"}),
        alternatives=(
            (
                frozenset({"pose_0", "pose_2", "pose_5"}),
                frozenset({"pose_0", "pose_2", "pose_5"}),
            ),
        ),
    ),
    # Torso yaw is read from the shoulder line alone, so it survives frames where
    # a wrist or the face is occluded -- the body keeps turning while a limb holds.
    ControlGroup.TORSO: ControlGroupPolicy(
        required=frozenset({"pose_11", "pose_12"}),
        confidence_critical=frozenset({"pose_11", "pose_12"}),
    ),
    ControlGroup.LEFT_ARM: ControlGroupPolicy(
        required=frozenset({"pose_11", "pose_12", "pose_13", "pose_15"}),
        confidence_critical=frozenset({"pose_11", "pose_12", "pose_15"}),
    ),
    ControlGroup.RIGHT_ARM: ControlGroupPolicy(
        required=frozenset({"pose_11", "pose_12", "pose_14", "pose_16"}),
        confidence_critical=frozenset({"pose_11", "pose_12", "pose_16"}),
    ),
}

#: Name of the implicit single group used when a caller builds a gate without
#: declaring control groups.  One group covering every required landmark makes
#: the gate behave exactly as it did before per-group gating existed.
WHOLE_ROBOT_GROUP = "all"


@dataclass(frozen=True)
class FreshnessPolicy:
    source_clock_id: str
    required_landmark_names: frozenset[str]
    max_age_ns: int = 150_000_000
    # 0.5, matching MediaPipe's own default detection/tracking confidence.
    #
    # This bar is applied per control group (see ControlGroup): the wrist,
    # shoulders, and face points that drive a given command must clear it. It
    # used to be applied to aggregate_confidence, a MINIMUM across every
    # command-critical landmark at once, which demanded both wrists, both
    # shoulders, and three face points independently clear the bar in the same
    # frame -- so a single occluded wrist held the entire robot. At 0.75 that
    # frame-wide aggregate was unreachable for a real person: measured on a live
    # camera, a correctly tracked operator calibrated at frame 24 and then held
    # LOW_CONFIDENCE for 780 consecutive frames without a single approved
    # target. The skeleton was drawn correctly throughout; the bar was simply
    # never met. Elbows are a soft IK hint and have their own lower retargeting
    # threshold. The value itself is unchanged; only its scope narrowed.
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if not self.required_landmark_names or any(not name for name in self.required_landmark_names):
            raise ValueError("at least one required landmark name is required")


@dataclass(frozen=True)
class ControlGroupGateResult:
    """Per-group verdict.  ``confidence`` is None when it was not measurable."""

    group: str
    accepted: bool
    reason: ObservationRejection | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class ObservationGateResult:
    """Additive result.

    ``accepted`` and ``reason`` keep their original whole-frame meaning:
    ``accepted`` is True only when every control group is trustworthy, so a
    caller that reads nothing else behaves exactly as it did before.  A caller
    that wants per-limb resolution reads ``groups`` / ``group_accepted``.
    """

    accepted: bool
    reason: ObservationRejection | None = None
    groups: Mapping[str, ControlGroupGateResult] = field(default_factory=dict)
    selection: SelectionDecision | None = None

    @property
    def any_group_accepted(self) -> bool:
        return any(result.accepted for result in self.groups.values())

    @property
    def accepted_groups(self) -> frozenset[str]:
        return frozenset(str(name) for name, result in self.groups.items() if result.accepted)

    @property
    def held_groups(self) -> frozenset[str]:
        return frozenset(str(name) for name, result in self.groups.items() if not result.accepted)

    def group_accepted(self, group: str) -> bool:
        """Fail-closed lookup: an unknown or unevaluated group is never allowed."""
        result = self.groups.get(group)
        return result is not None and result.accepted

    def group_reason(self, group: str) -> ObservationRejection | None:
        result = self.groups.get(group)
        return None if result is None else result.reason


class ObservationGate:
    """Stateful, same-host gate. It never compares cross-host monotonic clocks."""

    def __init__(
        self,
        policy: FreshnessPolicy,
        *,
        control_groups: Mapping[str, ControlGroupPolicy] | None = None,
        liveness_monitor: FrameLivenessMonitor | None = None,
        selection_manager: OperatorSelectionManager | None = None,
    ) -> None:
        self._policy = policy
        groups: dict[str, ControlGroupPolicy] = (
            dict(control_groups)
            if control_groups is not None
            else {
                WHOLE_ROBOT_GROUP: ControlGroupPolicy(
                    required=policy.required_landmark_names,
                    confidence_critical=policy.required_landmark_names,
                )
            }
        )
        if not groups:
            raise ValueError("at least one control group is required")
        # A required landmark that no group claims would be enforced by nobody,
        # which is the one way this refactor could silently fail OPEN. Refuse to
        # build such a gate rather than discover it in a session.
        covered: frozenset[str] = frozenset().union(*(spec.required for spec in groups.values()))
        unclaimed = policy.required_landmark_names - covered
        if unclaimed:
            raise ValueError(
                f"control groups do not cover required landmarks: {sorted(unclaimed)}"
            )
        self._control_groups = groups
        self._confidence_critical: frozenset[str] = frozenset().union(
            *(spec.confidence_critical for spec in groups.values())
        )
        self._last_sequence: int | None = None
        self._liveness_monitor = liveness_monitor
        self._selection_manager = selection_manager

    @property
    def control_groups(self) -> Mapping[str, ControlGroupPolicy]:
        return dict(self._control_groups)

    @property
    def has_liveness_monitor(self) -> bool:
        return self._liveness_monitor is not None

    @property
    def liveness_policy(self) -> LivenessPolicy | None:
        """Configured policy, if liveness was explicitly enabled for this source."""
        return None if self._liveness_monitor is None else self._liveness_monitor.policy

    @property
    def selection_id(self) -> int | None:
        """The currently locked session generation, if independent selection is enabled."""
        return None if self._selection_manager is None else self._selection_manager.selection_id

    @property
    def has_selection_manager(self) -> bool:
        return self._selection_manager is not None

    @property
    def selection_manager(self) -> OperatorSelectionManager:
        """Configured manager; callers must first establish that it exists."""
        if self._selection_manager is None:
            raise RuntimeError("operator selection is not configured")
        return self._selection_manager

    def acquire_selection(
        self,
        occupancy: OccupancyObservation,
        observation: HumanObservation,
        *,
        now_mono_ns: int,
    ) -> SelectionDecision:
        """Explicitly acquire an operator only from a healthy current frame."""
        if self._selection_manager is None:
            raise RuntimeError("operator selection is not configured")
        if self._base_frame_rejection(observation, now_mono_ns) is not None:
            return self._selection_manager.reject_acquisition()
        return self._selection_manager.acquire(
            occupancy, observation, now_mono_ns=now_mono_ns
        )

    def reset_liveness(self) -> None:
        """Forget content evidence at an externally verified source boundary.

        Normal calibration and operator acquisition preserve the camera stream,
        so they must not reset this monitor. ``reset_source`` uses this helper
        only after a camera reopen, clip restart, or other verified boundary;
        stream ordering remains independently controlled by that caller.
        """
        if self._liveness_monitor is not None:
            self._liveness_monitor.reset()

    def reset_source(self) -> None:
        """Forget stream ordering and liveness at a real capture-source break.

        Camera reopen, clip restart, and source switch restart the capture clock
        and sequence space. This method is deliberately reserved for those
        externally verified boundaries, never supervisor recovery.
        """
        self._last_sequence = None
        self.reset_liveness()
        if self._selection_manager is not None:
            self._selection_manager.reset()

    def evaluate(
        self,
        observation: HumanObservation,
        *,
        now_mono_ns: int,
        occupancy: OccupancyObservation | None = None,
    ) -> ObservationGateResult:
        # Whole-frame rejections, in their original order. Each one holds the
        # ENTIRE robot; none of them is per-group.
        if (rejection := self._base_frame_rejection(observation, now_mono_ns)) is not None:
            if self._selection_manager is None:
                selection = None
            elif rejection is ObservationRejection.FRAME_NOT_LIVE:
                selection = self._selection_manager.observe_liveness_gap(
                    capture_mono_ns=observation.capture_mono_ns
                )
            else:
                selection = self._selection_manager.observe_untrusted()
            return self._reject_everything(rejection, selection=selection)
        if self._selection_manager is not None:
            if occupancy is None:
                selection = self._selection_manager.observe_missing_occupancy(
                    capture_mono_ns=observation.capture_mono_ns
                )
                return self._reject_everything(
                    ObservationRejection.OCCUPANCY_MISSING, selection=selection
                )
            selection = self._selection_manager.observe(
                occupancy, observation, now_mono_ns=now_mono_ns
            )
            if not selection.accepted:
                return self._reject_everything(
                    ObservationRejection.SELECTION_NOT_LOCKED, selection=selection
                )
        else:
            selection = None

        groups = self._evaluate_groups(observation)

        if observation.aggregate_confidence < self._policy.min_confidence and not (
            self._aggregate_is_attributable(observation)
        ):
            # The producer declared a frame-wide confidence below the bar that
            # this gate cannot pin on any group's own landmarks. The shortfall
            # is therefore information we cannot attribute, and an unattributable
            # low confidence holds the whole robot exactly as it did before.
            return self._reject_everything(
                ObservationRejection.LOW_CONFIDENCE, groups, selection=selection
            )
        if observation.identity is not IdentityState.STABLE:
            return self._reject_everything(
                ObservationRejection.IDENTITY_NOT_STABLE, groups, selection=selection
            )

        accepted_count = sum(1 for result in groups.values() if result.accepted)
        if accepted_count == 0:
            # Every group failed: identical to the original global hold.
            return ObservationGateResult(False, self._summary_reason(groups), groups, selection)
        # At least one group produces a command from this frame, so the frame
        # has been acted on and must never be replayable.
        self._last_sequence = observation.sequence
        if accepted_count == len(groups):
            return ObservationGateResult(True, None, groups, selection)
        return ObservationGateResult(False, self._summary_reason(groups), groups, selection)

    def _base_frame_rejection(
        self, observation: HumanObservation, now_mono_ns: int
    ) -> ObservationRejection | None:
        if observation.source_clock_id != self._policy.source_clock_id:
            return ObservationRejection.CLOCK_MISMATCH
        if self._last_sequence is not None and observation.sequence <= self._last_sequence:
            return ObservationRejection.REORDERED
        if observation.capture_mono_ns > now_mono_ns:
            return ObservationRejection.FUTURE_CAPTURE
        if observation.inference_complete_mono_ns > now_mono_ns:
            return ObservationRejection.FUTURE_INFERENCE
        if now_mono_ns - observation.inference_complete_mono_ns > self._policy.max_age_ns:
            return ObservationRejection.STALE
        if self._liveness_monitor is not None:
            liveness = self._liveness_monitor.evaluate(
                fingerprint=observation.content_fingerprint or "",
                capture_mono_ns=observation.capture_mono_ns,
                sequence=observation.sequence,
            )
            if not liveness.live:
                return ObservationRejection.FRAME_NOT_LIVE
        return None

    def _evaluate_groups(
        self, observation: HumanObservation
    ) -> dict[str, ControlGroupGateResult]:
        by_name = {landmark.name: landmark for landmark in observation.landmarks}
        results: dict[str, ControlGroupGateResult] = {}
        for name, spec in self._control_groups.items():
            critical = spec.satisfied_by(by_name)
            if critical is None:
                # Unevaluable, so it holds. No confidence is claimed for it.
                results[name] = ControlGroupGateResult(
                    str(name), False, ObservationRejection.MISSING_REQUIRED_LANDMARK
                )
                continue
            confidence = min(
                min(by_name[landmark].visibility, by_name[landmark].presence)
                for landmark in critical
            )
            if confidence < self._policy.min_confidence:
                results[name] = ControlGroupGateResult(
                    str(name), False, ObservationRejection.LOW_CONFIDENCE, confidence
                )
                continue
            results[name] = ControlGroupGateResult(str(name), True, None, confidence)
        return results

    def _aggregate_is_attributable(self, observation: HumanObservation) -> bool:
        """Can the declared frame-wide confidence be explained by the landmarks?

        ``aggregate_confidence`` is defined by its producer as the minimum of
        ``min(visibility, presence)`` across the command-critical landmarks, and
        zero if any of them is absent (see
        ``adapters/mediapipe_holistic.MediaPipeHolisticDetector._aggregate_confidence``).
        The union of the groups' ``confidence_critical`` sets is exactly that
        same set, so this reconstructs the producer's own number from the
        landmarks it shipped.

        A declared value at or above the reconstruction is fully explained by
        per-landmark evidence, and the groups can be judged on their own
        landmarks.  A declared value BELOW it carries information this gate
        cannot attribute to any body part -- a producer that folds in something
        the landmarks do not show, or a hand-built observation -- and the frame
        falls back to the original whole-robot hold.  Trusting the landmarks
        over an optimistic scalar is the safe direction: a group is still held
        whenever its own landmarks are below the bar.
        """
        by_name = {landmark.name: landmark for landmark in observation.landmarks}
        if not self._confidence_critical <= by_name.keys():
            expected = 0.0
        else:
            expected = min(
                min(by_name[name].visibility, by_name[name].presence)
                for name in self._confidence_critical
            )
        return observation.aggregate_confidence >= expected

    def _reject_everything(
        self,
        reason: ObservationRejection,
        groups: Mapping[str, ControlGroupGateResult] | None = None,
        selection: SelectionDecision | None = None,
    ) -> ObservationGateResult:
        """Hold every configured group, preserving any confidence measured."""
        measured = groups or {}
        held = {
            name: ControlGroupGateResult(
                str(name),
                False,
                reason,
                measured[name].confidence if name in measured else None,
            )
            for name in self._control_groups
        }
        return ObservationGateResult(False, reason, held, selection)

    @staticmethod
    def _summary_reason(
        groups: Mapping[str, ControlGroupGateResult],
    ) -> ObservationRejection | None:
        reasons = {result.reason for result in groups.values() if not result.accepted}
        for candidate in _GROUP_REJECTION_PRECEDENCE:
            if candidate in reasons:
                return candidate
        return next(iter(reasons), None)
