"""Command boundary that makes hardware motion impossible in the core package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from galbot_motion_studio.contracts.core import ApprovedCommand


@dataclass(frozen=True)
class CommandReceipt:
    accepted: bool
    command_sequence: int
    sink: str


class CommandSink(Protocol):
    def submit(self, command: ApprovedCommand) -> CommandReceipt: ...


class NullCommandSink:
    """Records approved commands for tests; deliberately performs no motion."""

    def __init__(self) -> None:
        self.submissions: list[ApprovedCommand] = []

    def submit(self, command: ApprovedCommand) -> CommandReceipt:
        self.submissions.append(command)
        return CommandReceipt(accepted=True, command_sequence=command.target.sequence, sink="null")

