"""Robot adapters for the teleop stack.

``base``            -- the ``RobotAdapter`` protocol and ``ControlStatus``.
``fault_injection`` -- a test double that reproduces measured G1 misbehaviour.

Nothing above this package may import a concrete adapter directly. Depend on
``base.RobotAdapter`` so the safety layer can be driven by a fake that fails.
"""

from __future__ import annotations

from .base import (
    DEFAULT_CONTROL_GROUPS,
    G1_JOINT_GROUPS,
    GROUP_JOINTS,
    JOINT_LIMITS,
    MEASURED_POSE_2026_08_07_1806,
    MOTION_FAILURE_STATUSES,
    AdapterError,
    ControlStatus,
    JointLimit,
    MissingTeardownError,
    RobotAdapter,
    ToleranceUnknownError,
    UninitializedGetterError,
    clamp_to_limits,
    limits_for_group,
)
from .fault_injection import (
    ACCEPTANCE_TOLERANCE_IS_UNKNOWN_WITHIN,
    ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD,
    ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD,
    MISSING_TEARDOWN_EXIT_CODE,
    NOMINAL,
    SDK_DEFAULT_BLOCKING_TIMEOUT_S,
    FakeG1Robot,
    FaultConfig,
    FaultInjector,
    LatchCause,
    MotionRecord,
    SigintSwallower,
    UnknownBandPolicy,
    judge_reach,
    swallow_sigint,
)

__all__ = [
    "ACCEPTANCE_TOLERANCE_IS_UNKNOWN_WITHIN",
    "ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD",
    "ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD",
    "DEFAULT_CONTROL_GROUPS",
    "G1_JOINT_GROUPS",
    "GROUP_JOINTS",
    "JOINT_LIMITS",
    "MEASURED_POSE_2026_08_07_1806",
    "MISSING_TEARDOWN_EXIT_CODE",
    "MOTION_FAILURE_STATUSES",
    "NOMINAL",
    "SDK_DEFAULT_BLOCKING_TIMEOUT_S",
    "AdapterError",
    "ControlStatus",
    "FakeG1Robot",
    "FaultConfig",
    "FaultInjector",
    "JointLimit",
    "LatchCause",
    "MissingTeardownError",
    "MotionRecord",
    "RobotAdapter",
    "SigintSwallower",
    "ToleranceUnknownError",
    "UninitializedGetterError",
    "UnknownBandPolicy",
    "clamp_to_limits",
    "judge_reach",
    "limits_for_group",
    "swallow_sigint",
]
