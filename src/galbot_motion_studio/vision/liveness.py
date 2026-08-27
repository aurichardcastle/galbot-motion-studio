"""Fail-closed frame-liveness monitor: is the camera still producing new imagery?

Freshness answers "did this frame arrive recently?".  That is a statement about
*timestamps*, and a frozen source keeps producing perfectly fresh ones: a stalled
capture pipeline, a driver replaying its last buffer, or a paused virtual camera
all advance ``capture_mono_ns`` normally while the pixels stand still.  Confidence
does not help either -- a frozen frame of a well-lit operator scores *high*
confidence forever, because the landmarks really are there.  Every downstream gate
in this package therefore passes a frozen source, and the robot would keep
tracking an image of a person who has already walked away.

This module answers the other question: **is the content changing?**  It compares
an opaque per-frame content fingerprint supplied by the capture adapter, and holds
when no new content has appeared for longer than a configured budget.

Three design choices carry the safety argument:

1. **The budget is a DURATION, not a frame count.**  Measured frame rates in this
   project range from about 8 fps in difficult near-body poses to 25-30 fps in
   open space, so a fixed "N identical frames" rule would give a detection time
   that varies by roughly 3x with the operator's pose -- exactly when it matters
   least.  A duration bounds detection time to ``max_static_ns`` plus one
   inter-frame interval at any frame rate.

2. **Novelty is judged against a bounded HISTORY of fingerprints, not against the
   immediately preceding frame.**  A driver that replays a short buffer emits
   A, B, A, B ... in which every frame differs from its predecessor while no new
   imagery exists at all.  Comparing only to the predecessor reports that source
   as live forever.

3. **A frame that fails validation never advances the state.**  In particular a
   regressed or duplicated capture timestamp cannot restart the stall timer, so a
   frozen camera behind a jittery clock stays held instead of being rescued by the
   glitch.

The monitor never touches pixels.  It takes an opaque fingerprint string, which
keeps it dependency-free and unit-testable without a camera -- the same injection
pattern ``GuardedPathController`` uses for its clearance check.  Computing the
fingerprint, and choosing what region of the image it covers, belongs to the
capture adapter; see ``FrameLivenessMonitor.evaluate``.

Nothing here authorizes motion.  It reports a verdict and the evidence behind it;
the caller decides what to hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LivenessRejection(StrEnum):
    """Why a frame is not evidence of a live source.

    ``NOT_EVALUATED`` is the fail-closed startup state, not an error: until the
    monitor has seen content actually change at least once it cannot distinguish a
    healthy camera from one that was frozen before the session began, so it
    declines to call either one live.  A novel fingerprint on the very first frame
    is not evidence of change -- everything is novel to an empty window -- so
    liveness needs a *transition*, which needs two frames.  Once the budget has
    elapsed with no transition, the verdict escalates from ``NOT_EVALUATED`` to
    ``SOURCE_STALLED``: a source that has never moved is a stalled source.
    """

    NOT_EVALUATED = "NOT_EVALUATED"
    MISSING_FINGERPRINT = "MISSING_FINGERPRINT"
    INVALID_FRAME = "INVALID_FRAME"
    CLOCK_NOT_MONOTONIC = "CLOCK_NOT_MONOTONIC"
    SEQUENCE_NOT_MONOTONIC = "SEQUENCE_NOT_MONOTONIC"
    SOURCE_STALLED = "SOURCE_STALLED"


@dataclass(frozen=True)
class LivenessPolicy:
    """Liveness configuration.

    ``max_static_ns`` has **no default**, deliberately.  It is a detection-time
    budget, and a detection-time budget is a deployment decision owned by the
    safety authority for a given camera and task -- not a constant to inherit
    silently from a library.  Requiring it at construction makes the number
    visible in the deployment profile where it can be reviewed.

    ``max_history`` bounds the fingerprint history's memory. A known fingerprint
    must not be discarded merely because the next frame arrives after the
    detection budget: that next frame is exactly the first opportunity to report
    a freeze. Deployments therefore size this bound for the largest expected
    short-cycle driver buffer; the least-recently-seen entry is evicted only when
    the explicit count cap is reached.

    ``require_monotonic_sequence`` can be turned off for callers whose sequence
    numbers are not meaningful (a decoded file, a synthetic corpus).  Capture-time
    monotonicity is always enforced and is not optional: the whole stall
    measurement is expressed in capture time.
    """

    max_static_ns: int
    max_history: int = 256
    require_monotonic_sequence: bool = True

    def __post_init__(self) -> None:
        if self.max_static_ns <= 0:
            raise ValueError("max_static_ns must be positive")
        if self.max_history < 1:
            raise ValueError("max_history must be at least 1")


@dataclass(frozen=True)
class LivenessResult:
    """One frame's verdict plus the evidence a trace needs to reconstruct it."""

    live: bool
    reason: LivenessRejection | None
    sequence: int
    capture_mono_ns: int
    #: Capture-time span since content last changed.  ``None`` before any change
    #: has been observed, which is the fail-closed startup state.
    stalled_for_ns: int | None
    #: Frames accepted since content last changed.  Telemetry only: the verdict is
    #: a function of ``stalled_for_ns``, so it does not vary with frame rate.
    frames_since_change: int
    #: This frame's fingerprint was already seen inside the novelty window.  A
    #: single repeat is normal (a duplicated delivery); a sustained run is a stall.
    repeated: bool
    #: The frame passed validation and advanced the monitor's state.  A rejected
    #: frame leaves the monitor exactly as it was.
    accepted: bool


class FrameLivenessMonitor:
    """Stateful, deterministic frame-liveness state machine.

    Deterministic in the strict sense: the verdict is a pure function of the
    fingerprint/timestamp/sequence sequence handed to :meth:`evaluate`.  There is
    no wall clock, no randomness, and no I/O, so a recorded session replays to
    byte-identical verdicts.
    """

    def __init__(self, policy: LivenessPolicy) -> None:
        self._policy = policy
        self._seen: dict[str, int] = {}
        self._first_capture_ns: int | None = None
        self._last_capture_ns: int | None = None
        self._last_sequence: int | None = None
        self._last_change_ns: int | None = None
        self._frames_since_change = 0

    @property
    def policy(self) -> LivenessPolicy:
        return self._policy

    @property
    def last_change_mono_ns(self) -> int | None:
        """Capture time of the most recent frame that carried new content."""
        return self._last_change_ns

    @property
    def _reference_ns(self) -> int | None:
        """The instant the stall span is measured from.

        The last observed change once one exists; before that, the first accepted
        frame -- so a source that has never moved accrues a stall span from the
        moment it was first seen rather than being permanently unmeasured.
        """
        return self._last_change_ns if self._last_change_ns is not None else self._first_capture_ns

    @property
    def tracked_fingerprints(self) -> int:
        """Size of the novelty window.  Bounded by ``policy.max_history``."""
        return len(self._seen)

    def reset(self) -> None:
        """Forget everything and return to the fail-closed startup state.

        Call this whenever continuity of the *source* is broken -- camera reopened,
        a new selection generation, a resumed preview.  Carrying a previous
        source's novelty window across that boundary would let content from before
        the break vouch for content after it.
        """
        self._seen.clear()
        self._first_capture_ns = None
        self._last_capture_ns = None
        self._last_sequence = None
        self._last_change_ns = None
        self._frames_since_change = 0

    def evaluate(
        self, *, fingerprint: str, capture_mono_ns: int, sequence: int
    ) -> LivenessResult:
        """Judge one captured frame.

        ``fingerprint`` is an opaque content digest produced by the capture
        adapter.  Two frames are "the same content" exactly when their
        fingerprints compare equal, so the adapter's choice of digest defines what
        this monitor can see: it must cover a region of the image that genuinely
        varies with the scene, and must not include anything the driver stamps in
        per frame (an overlaid clock, a frame counter) or every frozen frame will
        look novel.

        ``capture_mono_ns`` must come from the capture clock, not from processing
        time: the stall span is measured in the same clock the caller's freshness
        gate uses, so the two numbers can be compared in a trace.

        Rejected frames leave the monitor unchanged and report ``accepted=False``.
        """
        if not fingerprint or not fingerprint.strip():
            return self._rejected(LivenessRejection.MISSING_FINGERPRINT, capture_mono_ns, sequence)
        if capture_mono_ns < 0 or sequence < 0:
            return self._rejected(LivenessRejection.INVALID_FRAME, capture_mono_ns, sequence)
        if self._last_capture_ns is not None and capture_mono_ns <= self._last_capture_ns:
            # Equal counts as regressed: two frames cannot be captured at the same
            # instant, and treating a duplicate timestamp as progress would let a
            # replayed frame advance the stall measurement.
            return self._rejected(LivenessRejection.CLOCK_NOT_MONOTONIC, capture_mono_ns, sequence)
        if (
            self._policy.require_monotonic_sequence
            and self._last_sequence is not None
            and sequence <= self._last_sequence
        ):
            return self._rejected(
                LivenessRejection.SEQUENCE_NOT_MONOTONIC, capture_mono_ns, sequence
            )

        first_frame = self._last_capture_ns is None
        repeated = fingerprint in self._seen
        self._seen[fingerprint] = capture_mono_ns
        while len(self._seen) > self._policy.max_history:
            oldest = min(self._seen, key=lambda key: self._seen[key])
            del self._seen[oldest]

        if first_frame:
            # Novel to an empty window is not evidence of change: liveness needs a
            # transition, and a transition needs a predecessor.
            self._first_capture_ns = capture_mono_ns
            self._frames_since_change = 0
        elif repeated:
            self._frames_since_change += 1
        else:
            self._last_change_ns = capture_mono_ns
            self._frames_since_change = 0

        self._last_capture_ns = capture_mono_ns
        self._last_sequence = sequence

        reference_ns = self._reference_ns
        assert reference_ns is not None  # set above on the first accepted frame
        stalled_for_ns = capture_mono_ns - reference_ns
        if stalled_for_ns > self._policy.max_static_ns:
            # Escalates over NOT_EVALUATED: a source that has produced no new
            # content for longer than the budget is stalled whether or not it ever
            # established liveness.
            reason: LivenessRejection | None = LivenessRejection.SOURCE_STALLED
        elif self._last_change_ns is None:
            reason = LivenessRejection.NOT_EVALUATED
        else:
            reason = None
        return LivenessResult(
            live=reason is None,
            reason=reason,
            sequence=sequence,
            capture_mono_ns=capture_mono_ns,
            stalled_for_ns=stalled_for_ns,
            frames_since_change=self._frames_since_change,
            repeated=repeated,
            accepted=True,
        )

    def _rejected(
        self, reason: LivenessRejection, capture_mono_ns: int, sequence: int
    ) -> LivenessResult:
        """A verdict that reports the rejection without advancing any state."""
        reference_ns = self._reference_ns
        stalled_for_ns = (
            None
            if reference_ns is None or capture_mono_ns < reference_ns
            else capture_mono_ns - reference_ns
        )
        return LivenessResult(
            live=False,
            reason=reason,
            sequence=sequence,
            capture_mono_ns=capture_mono_ns,
            stalled_for_ns=stalled_for_ns,
            frames_since_change=self._frames_since_change,
            repeated=False,
            accepted=False,
        )
