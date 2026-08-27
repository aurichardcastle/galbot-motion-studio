"""A Galbot G1 test double that fails the way the real robot was measured to fail.

Read this first
---------------
The teleop stack develops offline against a simulator that ALWAYS completes and
ALWAYS returns SUCCESS. Every line of the safety layer that handles failure is
therefore dead code offline, and would be exercised for the first time on real
hardware standing next to a person. This module exists to close that gap on a
laptop.

Every fault below corresponds to a documented simulator safety case. The rows
they map to in the safety spec are ``docs/safety-model.md`` R1-R9.

The nine faults
---------------
=== ==================================== ============ ==================================
 #   fault                                matrix row   evidence
=== ==================================== ============ ==================================
 1   SUCCESS without reaching             R3           finding 1f, transcript 2483-2484
 2   clamp-to-limit then report           R3           mechanism behind finding 1f
 3   TIMEOUT with partial motion          R4           transcript 2975-2998
 4   bare FAULT, zero motion              R5           transcript 3049-3051
 5   latched FAULT surviving restart      R6           finding 1g, 19:42
 6   stalled blocking call                R7           abort analysis, ~4 min of motion
 7   swallowed SIGINT                     (A1)         0 KeyboardInterrupt in 3060 lines
 8   segfault on missing teardown         --           Retractions, "the 17:35 segfault"
 9   getters unguarded after failed init  R2           reportable item 1
=== ==================================== ============ ==================================

What this module refuses to do
------------------------------
It does not invent numbers. The single most important unknown is the
acceptance tolerance behind faults 1 and 2: two hardware observations bound it
to the open interval (0.0062, 0.0359) rad and **no observation sits inside
it**. The default behaviour for a judgement landing in that band is to raise
``ToleranceUnknownError`` rather than guess -- see :class:`UnknownBandPolicy`.
Carried-forward item 12 in the field log ("bisect where SUCCESS turns into
TIMEOUT") is still OPEN.

Determinism
-----------
Every stochastic decision is drawn from a seeded ``random.Random``. Exactly two
draws are consumed per motion call, unconditionally, so the random stream is a
function of the call count alone and never of which faults happen to be
configured. A failing test reproduces from ``(seed, call sequence)``.

Usage sketch::

    fake = FakeG1Robot(state_path=tmp / "g1.json", seed=7)
    assert fake.init() is True
    fake.faults.set_group("head", FaultConfig(clamp_to_limit=True))
    status = fake.set_joint_positions([0.0, 0.4998], ["head"])
    fake.request_shutdown(); fake.wait_for_shutdown(); fake.destroy()
"""

from __future__ import annotations

import atexit
import enum
import json
import math
import os
import random
import signal
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .base import (
    DEFAULT_CONTROL_GROUPS,
    GROUP_JOINTS,
    JOINT_LIMITS,
    MEASURED_POSE_2026_08_07_1806,
    AdapterError,
    ControlStatus,
    MissingTeardownError,
    ToleranceUnknownError,
    UninitializedGetterError,
)

__all__ = [
    "ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD",
    "ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD",
    "ACCEPTANCE_TOLERANCE_IS_UNKNOWN_WITHIN",
    "MEASURED_BOW_COMMANDED_RAD",
    "MEASURED_DOWN_COMMANDED_RAD",
    "MEASURED_TIMEOUT_PARTIAL_STALL_FRACTION",
    "MISSING_TEARDOWN_EXIT_CODE",
    "SDK_DEFAULT_BLOCKING_TIMEOUT_S",
    "UnknownBandPolicy",
    "LatchCause",
    "FaultConfig",
    "FaultInjector",
    "MotionRecord",
    "FakeG1Robot",
    "SigintSwallower",
    "swallow_sigint",
    "judge_reach",
]


# ==========================================================================
# Measured constants. Changing any of these means claiming a new measurement.
# ==========================================================================

# MEASURED, transcript 2483-2484 and 2377 (VISIT2_FIELD_LOG.md finding 1f).
# head_joint2 sat at -0.0002 and two out-of-range deltas were commanded:
_MEASURED_HEAD_JOINT2_START_RAD = -0.0002
_MEASURED_HEAD_JOINT2_UPPER_LIMIT_RAD = 0.4935988   # URDF
_MEASURED_HEAD_JOINT2_LOWER_LIMIT_RAD = -0.2143461  # URDF

#: MEASURED: ``start + 0.50``. Returned SUCCESS, readback ``+0.4936``.
MEASURED_BOW_COMMANDED_RAD = 0.4998
#: MEASURED: ``start - 0.25``. Returned TIMEOUT.
MEASURED_DOWN_COMMANDED_RAD = -0.2502

#: The largest position error ever observed to return SUCCESS.
#:
#: Derived, not typed in: ``|0.4998 - 0.4935988|``. The field log rounds this to
#: 0.0062 rad because the hardware readback printed four decimals (``+0.4936``).
#: The bound is computed from the commanded value and the URDF limit instead, so
#: that "clamp to limit, then judge" reproduces the measured SUCCESS bit-exactly
#: rather than missing it by 1.2e-6 rad.
ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD = abs(
    MEASURED_BOW_COMMANDED_RAD - _MEASURED_HEAD_JOINT2_UPPER_LIMIT_RAD
)

#: The smallest position error ever observed to return TIMEOUT.
#: Derived as ``|-0.2502 - (-0.2143461)|``; the field log rounds it to 0.0359.
ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD = abs(
    MEASURED_DOWN_COMMANDED_RAD - _MEASURED_HEAD_JOINT2_LOWER_LIMIT_RAD
)

#: The honest statement of what is NOT known.
#:
#: The SDK's acceptance tolerance lies strictly inside this open interval. Two
#: data points bound it; zero data points sit inside it. Both came from ONE
#: joint (head_joint2), on ONE robot, at ONE speed, on ONE evening. It is not
#: known whether the tolerance is per-joint, per-group, speed-dependent,
#: timeout-dependent, or a single global constant. Any code that needs a
#: number here is under-specified and must say so.
ACCEPTANCE_TOLERANCE_IS_UNKNOWN_WITHIN = (
    ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD,
    ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD,
)

#: MEASURED: left_arm_joint3 was commanded -0.5625 -> -0.2625 (0.3 rad) and
#: stalled at -0.3477, having travelled 0.2147 rad -- 71.6% of the way, 0.0853
#: rad (4.9 deg) short. Used as the default stall fraction for fault 3.
MEASURED_TIMEOUT_PARTIAL_STALL_FRACTION = 0.716

#: MEASURED: six of seven joints still reached home while the call returned
#: TIMEOUT. That is the shape of fault 3: partial, not zero.
MEASURED_TIMEOUT_PARTIAL_REACHED_COUNT = 6
MEASURED_TIMEOUT_PARTIAL_TOTAL_COUNT = 7

#: The blocking timeout the 7 Aug scripts passed. NOTE: the SDK 1.9.0 docs give
#: ``timeout_s = 15.0`` as the default. 20.0 is used here because it is the
#: value the measured behaviour was produced under -- six printed timeouts at
#: 20 s is the 120 s floor in the field log's abort analysis. The two numbers
#: genuinely disagree and the docs are not wrong; the scripts overrode them.
SDK_DEFAULT_BLOCKING_TIMEOUT_S = 20.0

#: MEASURED driver-log lines, 19:42 and 19:51. These never reach the Python
#: API -- the whole of finding 1g is that all of this resolution exists inside
#: the SDK and the caller sees a bare FAULT.
_DRIVER_LOG_ESTOP = (
    "left_arm_driver_state   , code: 70 , description is DriverEstop",
    "right_arm_driver_state  , code: 70 , DriverEstop",
    "leg_driver_state        , code: 70 , DriverEstop",
    "head_driver_state       , code: 70 , DriverEstop",
    "chassis_driver_state    , code: 70 , DriverEstop",
    "left_arm_max_error_level, code: 327680 , ErrorSafetyCritical",
)
_DRIVER_LOG_DRIVER_ERROR = (
    "left_arm_joint4_custom_error , code: 15 , description is Not yet identified",
    "left_arm_driver_state        , code: 69 , DriverError",
    "left_arm_max_error_level     , code: 196608 , ErrorSafetyMedium",
    "left_arm , code: 533 , Not ready to start controller left_arm_pvt_ctrl for device left_arm",
)

_STATE_SCHEMA_VERSION = 1
_ATEXIT_BANNER = "GALBOT-FAKE: MISSING TEARDOWN"

#: Process exit status used when ``register_atexit_check=True`` catches a
#: missing teardown at interpreter exit.
#:
#: This exists because of a measured property of CPython, not of the robot:
#: **an exception raised inside an ``atexit`` callback does not affect the exit
#: code.** Verified on CPython 3.14.6 -- both a plain exception and an explicit
#: ``SystemExit(70)`` print "Exception ignored in atexit callback" to stderr and
#: the process still exits 0. So raising alone would make this fault loud to a
#: human reading a terminal and completely invisible to CI, which is the wrong
#: way round for the one fault whose entire purpose is to be observable.
#:
#: 86 is chosen to be distinguishable at a glance from 0, from 1 (an ordinary
#: Python traceback), and from 139 / -11 (the SIGSEGV the real SDK produces).
MISSING_TEARDOWN_EXIT_CODE = 86


# ==========================================================================
# Policy for the unmeasured band
# ==========================================================================
class UnknownBandPolicy(enum.Enum):
    """What to do when a reach error lands inside the unmeasured tolerance band.

    There is no correct answer, because the answer was never measured. Pick
    deliberately and record which you picked.
    """

    #: Refuse to guess. Raises ``ToleranceUnknownError``. This is the default
    #: because a test that silently depends on an invented threshold is worse
    #: than a test that fails loudly at the boundary of the evidence.
    RAISE = "raise"
    #: Assume the SDK reports TIMEOUT. Safe direction for a safety layer: it
    #: makes the caller handle failure more often than hardware might.
    PESSIMISTIC = "pessimistic"
    #: Assume the SDK reports SUCCESS. The dangerous direction; useful only to
    #: prove the safety layer catches an unreached target from readback rather
    #: than trusting the status.
    OPTIMISTIC = "optimistic"
    #: Seeded coin flip. For fuzzing the safety layer against both possible
    #: worlds in one run. Deterministic given the seed.
    RANDOM = "random"


class LatchCause(enum.Enum):
    """Why the controller latched. Both variants are MEASURED (19:42, 19:51).

    The codes are genuinely distinct and informative inside the SDK -- 70/327680
    is an engaged e-stop, 69/196608 is a drive error. **Neither reaches the
    Python API**; both collapse to a bare ``ControlStatus.FAULT``. That is
    finding 1g, and it is why a caller cannot tell "somebody pressed a button"
    from "a motor overloaded" without reading a log file off the robot.
    """

    ESTOP = "estop"
    DRIVER_ERROR = "driver_error"

    @property
    def driver_log(self) -> tuple[str, ...]:
        return _DRIVER_LOG_ESTOP if self is LatchCause.ESTOP else _DRIVER_LOG_DRIVER_ERROR


# ==========================================================================
# Judgement
# ==========================================================================
def judge_reach(
    error_rad: float,
    policy: UnknownBandPolicy = UnknownBandPolicy.RAISE,
    band_roll: float = 0.0,
) -> ControlStatus:
    """Map a position error to the status the SDK was measured to return.

    ``error_rad`` is ``|achieved - ORIGINAL commanded|`` -- the comparison is
    against what the caller asked for, never against the clamped value the
    robot actually executed. That asymmetry is the whole of fault 2.

    Bounds are inclusive on the measured side: an error of exactly
    ``ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD`` was observed to return SUCCESS,
    and exactly ``ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD`` to return TIMEOUT.
    """
    if error_rad <= ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD:
        return ControlStatus.SUCCESS
    if error_rad >= ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD:
        return ControlStatus.TIMEOUT

    if policy is UnknownBandPolicy.PESSIMISTIC:
        return ControlStatus.TIMEOUT
    if policy is UnknownBandPolicy.OPTIMISTIC:
        return ControlStatus.SUCCESS
    if policy is UnknownBandPolicy.RANDOM:
        return ControlStatus.TIMEOUT if band_roll < 0.5 else ControlStatus.SUCCESS

    raise ToleranceUnknownError(
        f"reach error {error_rad!r} rad falls inside the UNMEASURED acceptance band "
        f"({ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD!r}, {ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD!r}). "
        "Hardware returned SUCCESS at the lower bound and TIMEOUT at the upper bound; "
        "nothing was ever measured in between, so this test double will not guess. "
        "Choose a FaultConfig.unknown_band_policy explicitly, or bisect the real "
        "threshold on hardware (field log carried-forward item 12, still OPEN)."
    )


# ==========================================================================
# Fault configuration
# ==========================================================================
@dataclass(frozen=True, slots=True)
class FaultConfig:
    """One switchable fault profile. Every fault is independent.

    Resolution order for a motion call, highest first:

    1. a config queued with :meth:`FaultInjector.queue_for_next_call` (per-call)
    2. the config registered for the first named group that has one (per-group)
    3. the injector default

    Whether the resolved config *fires* on a given call is controlled by
    ``on_calls`` (explicit, fully deterministic) or ``probability`` (seeded).
    """

    # -- fault 9: init fails silently -------------------------------------
    #: ``init()`` returns a bare False, nothing on stdout or stderr, and every
    #: subsequent getter raises instead of returning plausible data.
    #: Read from the injector *default* config only; a per-group init is
    #: meaningless. MEASURED: reportable items 1 and 2.
    fail_init: bool = False

    # -- fault 2: clamp to limit, then judge against the original ---------
    #: Silently clamp each target to its URDF limit, execute the clamped value,
    #: then judge the result against the ORIGINAL command. MEASURED mechanism
    #: behind finding 1f.
    clamp_to_limit: bool = False

    # -- fault 1: SUCCESS without reaching --------------------------------
    #: Force the joint to stop this many radians short of its (possibly
    #: clamped) target, moving back toward the start pose. ``None`` disables.
    #: Set to ``ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD`` to reproduce the
    #: measured SUCCESS-without-reaching directly, without needing a limit
    #: violation to cause it. A zero-delta command cannot stop short.
    stop_short_rad: float | None = None
    #: Which joint indices within the group stop short. ``None`` means all.
    stop_short_indices: tuple[int, ...] | None = None

    # -- fault 3: TIMEOUT with partial motion -----------------------------
    #: Return TIMEOUT while SOME joints still reach their target. This is the
    #: measured shape of TIMEOUT and the reason TIMEOUT must never be treated
    #: as "nothing happened".
    timeout_partial: bool = False
    #: Indices that fail to reach. ``None`` -> :func:`default_unreached_indices`.
    timeout_partial_unreached_indices: tuple[int, ...] | None = None
    #: How far the unreached joints get, as a fraction of commanded travel.
    timeout_partial_stall_fraction: float = MEASURED_TIMEOUT_PARTIAL_STALL_FRACTION

    # -- fault 4: bare FAULT ----------------------------------------------
    #: Reject the command outright with ZERO motion. Distinct from
    #: ``timeout_partial``: nothing moves at all, not even a little.
    bare_fault: bool = False

    # -- fault 5: latched FAULT surviving process restart -----------------
    #: Latch the controller. The latch is written to the state file, so it
    #: survives ``destroy()``, a new adapter instance, and a genuinely new
    #: interpreter. It takes effect from the NEXT motion call onward; combine
    #: with ``bare_fault=True`` if the latching call should itself return FAULT.
    latch_fault: bool = False
    latch_cause: LatchCause = LatchCause.ESTOP

    # -- fault 6: stalled blocking call -----------------------------------
    #: Block for this many seconds inside ``set_joint_positions`` before the
    #: motion is applied. ``None`` disables. The motion still lands afterwards
    #: even if an abort was requested during the stall -- that is the point.
    stall_seconds: float | None = None

    # -- judgement --------------------------------------------------------
    unknown_band_policy: UnknownBandPolicy = UnknownBandPolicy.RAISE

    # -- firing schedule --------------------------------------------------
    #: Fire only on these 1-based motion-call indices. Fully deterministic and
    #: independent of the seed. Takes precedence over ``probability``.
    on_calls: tuple[int, ...] | None = None
    #: Probability of firing on any given call, drawn from the seeded RNG.
    #: 1.0 always fires, 0.0 never does.
    probability: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability!r}")
        if not 0.0 <= self.timeout_partial_stall_fraction <= 1.0:
            raise ValueError("timeout_partial_stall_fraction must be in [0, 1]")
        if self.stall_seconds is not None and self.stall_seconds < 0:
            raise ValueError("stall_seconds must be >= 0")

    @property
    def any_motion_fault_enabled(self) -> bool:
        return bool(
            self.clamp_to_limit
            or self.stop_short_rad is not None
            or self.timeout_partial
            or self.bare_fault
            or self.latch_fault
            or self.stall_seconds is not None
        )


NOMINAL = FaultConfig()
"""A config with every fault off: the always-SUCCESS simulator, for contrast."""


def default_unreached_indices(joint_count: int) -> tuple[int, ...]:
    """Which joint fails to reach, by default, under fault 3.

    MEASURED: on 7 Aug it was ``left_arm_joint3`` -- index 2 of the 7-joint arm
    -- that stalled while the other six reached. For groups with fewer than
    three joints there is no measurement at all, so the last joint is used;
    that choice is LOCAL and arbitrary.
    """
    if joint_count <= 0:
        return ()
    return (2,) if joint_count > 2 else (joint_count - 1,)


# ==========================================================================
# Injector
# ==========================================================================
class FaultInjector:
    """Holds the per-call / per-group / default fault configuration."""

    def __init__(self, default: FaultConfig | None = None) -> None:
        self.default: FaultConfig = default or NOMINAL
        self._per_group: dict[str, FaultConfig] = {}
        self._queue: list[FaultConfig] = []

    def set_default(self, config: FaultConfig) -> None:
        self.default = config

    def set_group(self, group: str, config: FaultConfig) -> None:
        """Register a config for one joint group."""
        self._per_group[group] = config

    def clear_group(self, group: str) -> None:
        self._per_group.pop(group, None)

    def queue_for_next_call(self, config: FaultConfig) -> None:
        """Apply ``config`` to the next motion call only. FIFO; highest precedence."""
        self._queue.append(config)

    def clear(self) -> None:
        self.default = NOMINAL
        self._per_group.clear()
        self._queue.clear()

    @property
    def queued_count(self) -> int:
        return len(self._queue)

    def resolve(self, groups: Sequence[str]) -> tuple[FaultConfig, str]:
        """Resolve the config for a call touching ``groups``.

        Returns ``(config, source)`` where source is ``"queued"``, ``"group:X"``
        or ``"default"``. When a command names several groups, the FIRST named
        group with a registered config wins -- deterministic, and it matches the
        SDK's own rule that groups are expanded in the order given.
        """
        if self._queue:
            return self._queue.pop(0), "queued"
        for group in groups:
            if group in self._per_group:
                return self._per_group[group], f"group:{group}"
        return self.default, "default"


# ==========================================================================
# Call record
# ==========================================================================
@dataclass(frozen=True, slots=True)
class MotionRecord:
    """One motion call, for assertions and for reproducing a failing run."""

    call_index: int
    groups: tuple[str, ...]
    joint_names: tuple[str, ...]
    commanded: tuple[float, ...]
    achieved: tuple[float, ...]
    status: ControlStatus
    config_source: str
    fired: bool
    stalled_seconds: float
    max_error_rad: float
    abort_was_requested: bool
    is_blocking: bool
    speed: float
    timeout: float


# ==========================================================================
# SIGINT swallowing -- fault 7. DANGEROUS BY DESIGN.
# ==========================================================================
class SigintSwallower:
    """Install a SIGINT handler that eats the signal, like the real SDK's.

    .. warning::

       **This makes Ctrl-C stop working for as long as it is installed.** It is
       opt-in, it is never enabled by any default, and it must always be used
       as a context manager so the previous handler is restored.

    MEASURED (VISIT2_FIELD_LOG.md, "The abort that wasn't"): on 7 Aug a Ctrl-C
    landed at transcript line 2982. The loop headers for joints 4, 5, 6 and 7
    all printed *after* it, each returning TIMEOUT, then the final home command
    returned TIMEOUT. ``Traceback`` and ``KeyboardInterrupt`` appear **zero
    times** in the whole 3,060-line file. Python never saw the interrupt. With
    a 20 s timeout that is a 120 s floor of continued blocking motion commands
    after the abort; with the unprinted per-iteration restores and sleeps,
    roughly four minutes. Every script that evening printed "ctrl-C to abort".
    That instruction was false on every one of them.

    The *mechanism* is inference, not measurement: the field log is explicit
    that the SDK installing its own handler is the best explanation, and that
    the swallowing is measured while the cause is not. What this class
    reproduces is the observable behaviour.

    Point: any abort path that depends on ``KeyboardInterrupt`` must fail here,
    in a test, rather than on hardware next to a person.
    """

    def __init__(self) -> None:
        self.caught_count = 0
        self._previous: object = None
        self._installed = False

    def _handler(self, signum: int, frame: object) -> None:  # noqa: ARG002
        # Deliberately does nothing else. No exception, no flag the caller
        # would naturally check -- exactly the behaviour that made Ctrl-C a lie.
        self.caught_count += 1

    def install(self) -> "SigintSwallower":
        if self._installed:
            raise AdapterError("SigintSwallower already installed")
        if threading.current_thread() is not threading.main_thread():
            raise AdapterError(
                "SIGINT handlers can only be installed from the main thread; "
                "signal.signal() would raise ValueError here."
            )
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        self._installed = True
        return self

    def restore(self) -> None:
        if not self._installed:
            return
        signal.signal(signal.SIGINT, self._previous)  # type: ignore[arg-type]
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    def __enter__(self) -> "SigintSwallower":
        return self.install()

    def __exit__(self, *exc: object) -> None:
        self.restore()


def swallow_sigint() -> SigintSwallower:
    """Context manager form of :class:`SigintSwallower`. Opt-in and dangerous."""
    return SigintSwallower()


# ==========================================================================
# Teardown tracking -- fault 8
# ==========================================================================
class _TeardownStage(enum.IntEnum):
    NOT_STARTED = 0
    REQUESTED = 1
    WAITED = 2
    DESTROYED = 3


# ==========================================================================
# The fake
# ==========================================================================
class FakeG1Robot:
    """A G1 adapter that misbehaves on demand. Implements ``base.RobotAdapter``.

    Persistence
    -----------
    Joint positions AND any latched fault live in a JSON state file, not on the
    instance. That is not an implementation convenience -- it is the fault. The
    real latch lives in the robot's controller, so it survived
    ``request_shutdown() -> wait_for_shutdown() -> destroy()`` and a brand new
    Python interpreter. Positions persist for the same reason: a robot does not
    teleport to a home pose because your process exited. A mock that resets
    state in ``__init__`` cannot reproduce either, and that is precisely the
    bug this module exists to catch.

    To simulate a process restart in-process: tear the instance down, then
    construct a new ``FakeG1Robot`` with the same ``state_path``. To simulate it
    for real: run a new interpreter against the same file.
    """

    def __init__(
        self,
        state_path: str | os.PathLike[str] | None = None,
        seed: int = 0,
        config: FaultConfig | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        register_atexit_check: bool = False,
    ) -> None:
        self.state_path = Path(
            state_path
            if state_path is not None
            else Path(tempfile.gettempdir()) / "galbot_fake_g1_state.json"
        )
        self.seed = seed
        self.faults = FaultInjector(config)
        self._rng = random.Random(seed)
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep

        self._init_called = False
        self._init_ok = False
        self._stage = _TeardownStage.NOT_STARTED
        self._motion_calls = 0
        self.call_log: list[MotionRecord] = []

        #: Set by :meth:`request_abort`. The fake deliberately IGNORES it for an
        #: in-flight blocking call. "Stop commanding" does not equal "stop
        #: moving", and this attribute is how a test proves it.
        self.abort_requested = False
        self.motions_applied_after_abort = 0

        self._positions: dict[str, float] = {}
        self._latch: dict[str, object] | None = None
        self._load_state()

        self._atexit_registered = False
        if register_atexit_check:
            atexit.register(self._atexit_check)
            self._atexit_registered = True

    # ---------------------------------------------------------------- state
    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                raise AdapterError(f"corrupt fake-robot state file {self.state_path}: {exc}") from exc
            if raw.get("schema_version") != _STATE_SCHEMA_VERSION:
                raise AdapterError(
                    f"state file {self.state_path} has schema_version "
                    f"{raw.get('schema_version')!r}, expected {_STATE_SCHEMA_VERSION}"
                )
            self._positions = {str(k): float(v) for k, v in raw.get("positions", {}).items()}
            self._latch = raw.get("latch")
        else:
            self._positions = dict(MEASURED_POSE_2026_08_07_1806)
            self._latch = None
            self._save_state()

    def _save_state(self) -> None:
        payload = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "positions": self._positions,
            "latch": self._latch,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------- lifecycle
    def init(self, sensor_set: Sequence[object] | None = None) -> bool:
        """Initialise. Returns a bare bool; never raises, never prints.

        Two measured behaviours combine here and both matter:

        * ``fail_init`` makes this return False **silently** -- no exception,
          nothing on stdout or stderr. The reason would be in
          ``~/galbot_sdk_log/g1/`` on real hardware and nothing tells you to look.
        * A latched controller fault does NOT make this return False. It returns
          True, and so does every getter. That is finding 1g and it is the
          single most dangerous thing in this module.
        """
        del sensor_set  # accepted for signature fidelity; unused by the fake
        self._init_called = True
        if self._stage is _TeardownStage.DESTROYED:
            # DOCUMENTED: "After destroy(), the SDK enters a terminal state and
            # cannot be re-initialized in the same process." The exact return
            # value in that situation is LOCAL -- never observed.
            self._init_ok = False
            return False
        if self.faults.default.fail_init:
            self._init_ok = False
            return False
        self._init_ok = True
        return True

    def request_shutdown(self) -> None:
        if self._stage is not _TeardownStage.NOT_STARTED:
            raise MissingTeardownError(
                f"request_shutdown() called at stage {self._stage.name}; it must be first."
            )
        self._stage = _TeardownStage.REQUESTED

    def wait_for_shutdown(self) -> None:
        if self._stage is not _TeardownStage.REQUESTED:
            raise MissingTeardownError(
                f"wait_for_shutdown() called at stage {self._stage.name}; "
                "it must follow request_shutdown()."
            )
        self._stage = _TeardownStage.WAITED

    def destroy(self) -> None:
        if self._stage is not _TeardownStage.WAITED:
            raise MissingTeardownError(
                f"destroy() called at stage {self._stage.name}. The documented sequence is "
                "request_shutdown() -> wait_for_shutdown() -> destroy(); on real hardware "
                "omitting it SIGSEGVs the interpreter with no Python traceback."
            )
        self._stage = _TeardownStage.DESTROYED
        self._init_ok = False
        self._save_state()

    @property
    def teardown_complete(self) -> bool:
        return self._stage is _TeardownStage.DESTROYED

    def check_teardown(self) -> None:
        """Raise if the documented shutdown sequence has not completed."""
        if self._stage is not _TeardownStage.DESTROYED:
            raise MissingTeardownError(
                f"{_ATEXIT_BANNER}: process is exiting at stage {self._stage.name}. "
                "Required: request_shutdown() -> wait_for_shutdown() -> destroy(). "
                "On real hardware this omission produced SIGSEGV on exit (A/B confirmed: "
                "omission -> SIGSEGV, documented sequence -> clean exit 0)."
            )

    def _atexit_check(self) -> None:
        """Interpreter-exit hook. Opt-in via ``register_atexit_check=True``.

        Prints a distinctive banner and then terminates the process with
        :data:`MISSING_TEARDOWN_EXIT_CODE`. It uses ``os._exit`` deliberately:
        an exception raised here would be swallowed by CPython (see the
        constant's docstring) and the omission would not be detectable by
        anything except a human reading stderr.

        This is a hard exit -- remaining ``atexit`` callbacks will not run.
        That is a real cost, accepted because the flag is opt-in and its only
        purpose is to catch exactly this. It is still strictly gentler than the
        SIGSEGV the real SDK produces, and unlike a segfault it leaves a
        readable explanation behind.
        """
        if self._stage is _TeardownStage.DESTROYED:
            return
        try:
            self.check_teardown()
        except MissingTeardownError as exc:
            detail = f"{type(exc).__name__}: {exc}"
        else:  # pragma: no cover - unreachable; check_teardown always raises here
            return
        print(
            f"\n{'=' * 70}\n{_ATEXIT_BANNER}\n"
            f"  stage at exit : {self._stage.name}\n"
            f"  state file    : {self.state_path}\n"
            f"  {detail}\n"
            "  On real hardware this is where the process SIGSEGVs.\n"
            f"  Exiting {MISSING_TEARDOWN_EXIT_CODE} so CI can see it.\n"
            f"{'=' * 70}",
            file=sys.stderr,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(MISSING_TEARDOWN_EXIT_CODE)

    def __enter__(self) -> "FakeG1Robot":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            # Do not mask the real failure; still make the omission visible.
            if self._stage is not _TeardownStage.DESTROYED:
                warnings.warn(
                    f"{_ATEXIT_BANNER} (suppressed: another exception is propagating). "
                    f"stage={self._stage.name}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return
        self.check_teardown()

    # --------------------------------------------------------------- getters
    def _guard_getter(self, what: str) -> None:
        """Fault 9. On hardware this path is a SIGSEGV, not an exception."""
        if not self._init_ok:
            reason = (
                "init() was never called"
                if not self._init_called
                else (
                    "destroy() has been called"
                    if self._stage is _TeardownStage.DESTROYED
                    else "init() returned False"
                )
            )
            raise UninitializedGetterError(
                f"{what}() called while uninitialised ({reason}). On real hardware the SDK "
                "does not guard its getters against this: it dereferences null and kills the "
                "process with SIGSEGV and no Python traceback. A test double must be "
                "observable instead, so this raises."
            )

    def get_joint_group_names(self) -> list[str]:
        self._guard_getter("get_joint_group_names")
        return list(GROUP_JOINTS)

    def get_joint_positions(self, groups: Sequence[str] = ()) -> dict[str, list[float]]:
        """Live joint positions, keyed by group.

        MEASURED: these keep returning correct live values while the controller
        is latched. There is no health check anywhere in the API that reveals
        the fault -- you discover it only by commanding motion and having it
        refused.
        """
        self._guard_getter("get_joint_positions")
        wanted = tuple(groups) if groups else DEFAULT_CONTROL_GROUPS
        out: dict[str, list[float]] = {}
        for group in wanted:
            # Unknown group -> empty list, no exception. MEASURED via the
            # documented-but-wrong "legs" plural returning 0 joints (finding 1b).
            out[group] = [self._positions[name] for name in GROUP_JOINTS.get(group, ())]
        return out

    def get_joint_positions_flat(self, groups: Sequence[str] = ()) -> list[float]:
        """The real SDK's flat return shape, for code that must match it exactly."""
        return [v for group in (groups or DEFAULT_CONTROL_GROUPS) for v in self.get_joint_positions([group])[group]]

    # ------------------------------------------------------- out-of-band ops
    # None of these exist on the real API. They model things an operator does
    # with a log file, a button, or their hands.

    def debug_read_driver_log(self) -> list[str]:
        """Model ``tail ~/galbot_sdk_log/g1/``. NOT part of ``RobotAdapter``.

        This is deliberately not a method the safety layer may call over the
        wire: on hardware the operator had to SSH in and read a file to
        discover that a button had been pressed. If your recovery design needs
        this, your recovery design needs a human.
        """
        if self._latch is None:
            return []
        cause = LatchCause(str(self._latch["cause"]))
        return list(cause.driver_log)

    @property
    def is_latched(self) -> bool:
        """Ground truth. **No equivalent exists on the real API** -- that is
        finding 1g. Tests may read this; production code may not pretend to."""
        return self._latch is not None

    @property
    def latch_record(self) -> Mapping[str, object] | None:
        return dict(self._latch) if self._latch is not None else None

    def power_cycle(self) -> None:
        """Simulate a power cycle. MEASURED: this does NOT clear the latch.

        "19:42 -- every chain returned ControlStatus.FAULT. A power cycle did
        not clear it." Reloads state from disk and resets process-local
        lifecycle, exactly like a real restart would.
        """
        self._init_called = False
        self._init_ok = False
        self._stage = _TeardownStage.NOT_STARTED
        self._load_state()

    def clear_latch(self) -> None:
        """Release the e-stop / reposition the arm by hand. Clears the latch.

        MEASURED recovery path (19:51 -> ~20:10): releasing the e-stop let the
        head move but the left arm still refused; the arm had to be repositioned
        by hand before the driver error cleared.
        """
        self._latch = None
        self._save_state()

    def request_abort(self) -> None:
        """Ask the teleop layer to stop commanding.

        Note what this does NOT do: it does not stop an in-flight blocking call.
        There is no known cancel API on the real SDK. Setting this while a
        stalled call is running is how a test proves that ``t_motion`` is not
        bounded by ``t_cmd``.
        """
        self.abort_requested = True

    # ---------------------------------------------------------------- motion
    def set_joint_positions(
        self,
        positions: Sequence[float],
        groups: Sequence[str] = (),
        is_blocking: bool = True,
        speed: float = 0.2,
        timeout: float = SDK_DEFAULT_BLOCKING_TIMEOUT_S,
        *,
        joint_names: Sequence[str] = (),
    ) -> ControlStatus:
        """Command joint positions, with whatever faults are configured."""
        if not self._init_ok:
            # Motion IS guarded on hardware -- the SDK prints "PublishTarget
            # failed: target_dispatcher_ is nullptr, please init." The specific
            # enum member returned was never observed, so INIT_FAILED here is
            # INFERRED, not measured. Contrast the getters, which are unguarded.
            return ControlStatus.INIT_FAILED

        target_names = self._expand(groups, joint_names)
        invalid = self._validate(positions, target_names, speed, timeout)
        if invalid is not None:
            return invalid

        self._motion_calls += 1
        call_index = self._motion_calls
        # Exactly two draws per motion call, unconditionally, so the RNG stream
        # depends only on how many motion calls have happened -- never on which
        # faults are configured. This is what makes a seed reproduce a run.
        fire_roll = self._rng.random()
        band_roll = self._rng.random()

        commanded = tuple(float(p) for p in positions)
        start = tuple(self._positions[n] for n in target_names)
        group_tuple = tuple(groups) if groups else DEFAULT_CONTROL_GROUPS

        # --- latched fault wins over everything (fault 5) -----------------
        if self._latch is not None:
            return self._record(
                call_index, group_tuple, target_names, commanded, start,
                ControlStatus.FAULT, "latched", True, 0.0, is_blocking, speed, timeout,
            )

        config, source = self.faults.resolve(group_tuple)
        fired = (
            call_index in config.on_calls
            if config.on_calls is not None
            else fire_roll < config.probability
        )

        # --- nominal: the always-SUCCESS simulator ------------------------
        if not fired or not config.any_motion_fault_enabled:
            self._apply(target_names, commanded)
            return self._record(
                call_index, group_tuple, target_names, commanded, commanded,
                ControlStatus.SUCCESS, source, fired, 0.0, is_blocking, speed, timeout,
            )

        # --- fault 6: stall the blocking call -----------------------------
        stalled = 0.0
        if config.stall_seconds is not None:
            stalled = float(config.stall_seconds)
            self._sleep(stalled)

        # --- fault 4: bare FAULT, zero motion -----------------------------
        if config.bare_fault:
            if config.latch_fault:
                self._set_latch(config.latch_cause, call_index)
            return self._record(
                call_index, group_tuple, target_names, commanded, start,
                ControlStatus.FAULT, source, True, stalled, is_blocking, speed, timeout,
            )

        achieved = self._compute_achieved(config, target_names, start, commanded)
        self._apply(target_names, achieved)
        if self.abort_requested:
            # Motion landed after somebody asked us to stop. Counted, not prevented.
            self.motions_applied_after_abort += 1

        if config.timeout_partial:
            status = ControlStatus.TIMEOUT
        else:
            error = max((abs(a - c) for a, c in zip(achieved, commanded)), default=0.0)
            status = judge_reach(error, config.unknown_band_policy, band_roll)

        if config.latch_fault:
            self._set_latch(config.latch_cause, call_index)

        return self._record(
            call_index, group_tuple, target_names, commanded, achieved,
            status, source, True, stalled, is_blocking, speed, timeout,
        )

    # ------------------------------------------------------------- internals
    def _expand(self, groups: Sequence[str], joint_names: Sequence[str]) -> tuple[str, ...]:
        """DOCUMENTED SDK rule: joint_names takes precedence over joint_groups."""
        if joint_names:
            return tuple(joint_names)
        wanted = tuple(groups) if groups else DEFAULT_CONTROL_GROUPS
        return tuple(name for group in wanted for name in GROUP_JOINTS.get(group, ()))

    def _validate(
        self,
        positions: Sequence[float],
        target_names: Sequence[str],
        speed: float,
        timeout: float,
    ) -> ControlStatus | None:
        # LOCAL: none of these validations were observed on hardware. The SDK
        # documents INVALID_INPUT as "input parameters invalid or not meeting
        # interface requirements" and the only INVALID_INPUT this project ever
        # saw came from forward_kinematics with a bad frame name. Treat these
        # as a defensible contract for the adapter, not as measured behaviour.
        if not target_names:
            return ControlStatus.INVALID_INPUT
        if len(positions) != len(target_names):
            return ControlStatus.INVALID_INPUT
        if any(name not in self._positions for name in target_names):
            return ControlStatus.INVALID_INPUT
        if any(not math.isfinite(float(p)) for p in positions):
            return ControlStatus.INVALID_INPUT
        if speed <= 0 or timeout <= 0:
            return ControlStatus.INVALID_INPUT
        return None

    def _compute_achieved(
        self,
        config: FaultConfig,
        names: Sequence[str],
        start: Sequence[float],
        commanded: Sequence[float],
    ) -> tuple[float, ...]:
        achieved = list(commanded)

        # fault 2: silent clamp to the URDF limit.
        if config.clamp_to_limit:
            for i, name in enumerate(names):
                limit = JOINT_LIMITS.get(name)
                if limit is not None:
                    achieved[i] = limit.clamp(achieved[i])

        # fault 1: stop short of the (possibly clamped) target.
        if config.stop_short_rad is not None:
            short = abs(float(config.stop_short_rad))
            indices = (
                range(len(names))
                if config.stop_short_indices is None
                else config.stop_short_indices
            )
            for i in indices:
                if not 0 <= i < len(names):
                    continue
                delta = achieved[i] - start[i]
                if delta == 0.0:
                    continue  # a zero-delta command cannot stop short
                step = math.copysign(min(short, abs(delta)), delta)
                achieved[i] -= step

        # fault 3: some joints reach, some stall part-way.
        if config.timeout_partial:
            indices = (
                config.timeout_partial_unreached_indices
                if config.timeout_partial_unreached_indices is not None
                else default_unreached_indices(len(names))
            )
            for i in indices:
                if not 0 <= i < len(names):
                    continue
                achieved[i] = start[i] + config.timeout_partial_stall_fraction * (
                    commanded[i] - start[i]
                )

        return tuple(achieved)

    def _apply(self, names: Sequence[str], values: Sequence[float]) -> None:
        for name, value in zip(names, values):
            self._positions[name] = float(value)
        self._save_state()

    def _set_latch(self, cause: LatchCause, call_index: int) -> None:
        self._latch = {
            "cause": cause.value,
            "latched_on_call": call_index,
            # MEASURED: robot-wide, not per-chain. head and leg were untouched
            # by the incident and still returned FAULT.
            "scope": "robot_wide",
        }
        self._save_state()

    def _record(
        self,
        call_index: int,
        groups: tuple[str, ...],
        names: tuple[str, ...],
        commanded: tuple[float, ...],
        achieved: tuple[float, ...],
        status: ControlStatus,
        source: str,
        fired: bool,
        stalled: float,
        is_blocking: bool,
        speed: float,
        timeout: float,
    ) -> ControlStatus:
        record = MotionRecord(
            call_index=call_index,
            groups=groups,
            joint_names=names,
            commanded=commanded,
            achieved=achieved,
            status=status,
            config_source=source,
            fired=fired,
            stalled_seconds=stalled,
            max_error_rad=max((abs(a - c) for a, c in zip(achieved, commanded)), default=0.0),
            abort_was_requested=self.abort_requested,
            is_blocking=is_blocking,
            speed=speed,
            timeout=timeout,
        )
        self.call_log.append(record)
        return status
