"""Verified, offline-only MuJoCo digital-twin command sink."""

from __future__ import annotations

import mujoco

from galbot_motion_studio.contracts.core import ApprovedCommand
from typing import Mapping
from threading import RLock

from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.ports.command import CommandReceipt


class MujocoPreviewSink:
    """Applies named approved targets to the digital twin and never contacts hardware."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel | None = None,
        initial_pose: Mapping[str, float] | None = None,
    ) -> None:
        self.model = model or load_verified_fixed_base_model()
        self.data = mujoco.MjData(self.model)
        self.lock = RLock()
        for name, position in (initial_pose or {}).items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"unknown preview joint: {name}")
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = position
        mujoco.mj_forward(self.model, self.data)

    def submit(self, command: ApprovedCommand) -> CommandReceipt:
        if not isinstance(command, ApprovedCommand):
            raise TypeError("MuJoCo preview accepts only an ApprovedCommand capability")
        with self.lock:
            for target in command.target.joints:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, target.name)
                if joint_id < 0:
                    raise ValueError(f"unknown preview joint: {target.name}")
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = target.position_rad
            mujoco.mj_forward(self.model, self.data)
        return CommandReceipt(accepted=True, command_sequence=command.target.sequence, sink="mujoco-preview")

    def joint_positions(self) -> dict[str, float]:
        """Return a name-addressed digital-twin readback for recording."""
        positions: dict[str, float] = {}
        with self.lock:
            for joint_id in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if name:
                    positions[name] = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        return positions
