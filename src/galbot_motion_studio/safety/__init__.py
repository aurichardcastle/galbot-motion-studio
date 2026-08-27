"""Safety gates: geometric clearance checking.

Owner: safety lane. Consumes the pinned fixed-base model read-only.
"""
"""Safety gates and simulator-only command authority."""

from galbot_motion_studio.safety.supervisor import (
    SafetySupervisor,
    SupervisionResult,
    SupervisorPolicy,
    SupervisorState,
)

__all__ = [
    "SafetySupervisor",
    "SupervisionResult",
    "SupervisorPolicy",
    "SupervisorState",
]
