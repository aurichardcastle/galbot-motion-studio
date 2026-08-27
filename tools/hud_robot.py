"""Model-side numbers for the HUD.

The browser draws the robot from the same URDF the model is compiled from; this
module supplies the joint table, the home pose and the measurements, so the
figures on the page come from the same kinematics the solver uses.

Simulation only. Posing the model here computes forward kinematics and draws a
picture. Nothing is commanded anywhere; there is no transport in this project.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from threading import Lock

import mujoco
import numpy as np
from PIL import Image

from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.retarget.swivel_ik import swivel_geometry

_PSI_REFERENCE = np.array([0.0, 0.0, -1.0])
_PSI_FALLBACK = np.array([1.0, 0.0, 0.0])

ARMS = {
    "left": {
        "joints": tuple(f"left_arm_joint{i}" for i in range(1, 8)),
        "shoulder": "left_arm_link1",
        "elbow": "left_arm_link4",
        "tcp": "left_gripper_tcp_link",
    },
    "right": {
        "joints": tuple(f"right_arm_joint{i}" for i in range(1, 8)),
        "shoulder": "right_arm_link1",
        "elbow": "right_arm_link4",
        "tcp": "right_gripper_tcp_link",
    },
}
HEAD_JOINTS = ("head_joint1", "head_joint2")


@dataclass
class Camera:
    azimuth: float = 135.0
    elevation: float = -18.0
    distance: float = 2.0
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.9)


@dataclass
class RobotView:
    width: int = 640
    height: int = 480
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        self.model = load_verified_fixed_base_model()
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self._checker = None
        self._floor = 0.005
        self._ids: dict[str, int] = {}
        for arm in ARMS.values():
            for key in ("shoulder", "elbow", "tcp"):
                name = arm[key]
                self._ids[name] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
        self.joint_limits = {}
        for name in self.all_joints():
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            low, high = self.model.jnt_range[joint]
            if not bool(self.model.jnt_limited[joint]):
                low, high = -np.pi, np.pi
            self.joint_limits[name] = (float(low), float(high))
        self._qadr = {
            name: int(
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in self.all_joints()
        }

    @staticmethod
    def all_joints() -> tuple[str, ...]:
        return (*ARMS["left"]["joints"], *ARMS["right"]["joints"], *HEAD_JOINTS)

    def _renderer_for(self, width: int, height: int) -> mujoco.Renderer:
        if (
            self._renderer is None
            or self._renderer.width != width
            or self._renderer.height != height
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderer

    def apply(self, joints: dict[str, float]) -> None:
        """Set the pose, clamped to the model's own limits, and run kinematics."""
        for name, value in joints.items():
            if name not in self._qadr:
                continue
            low, high = self.joint_limits[name]
            self.data.qpos[self._qadr[name]] = float(np.clip(float(value), low, high))
        mujoco.mj_forward(self.model, self.data)

    def measurements(self) -> dict[str, object]:
        """The numbers worth watching while posing the arm."""
        out: dict[str, object] = {}
        for side, arm in ARMS.items():
            shoulder = self.data.xpos[self._ids[arm["shoulder"]]]
            elbow = self.data.xpos[self._ids[arm["elbow"]]]
            tcp = self.data.xpos[self._ids[arm["tcp"]]]
            geometry = swivel_geometry(
                shoulder, elbow, tcp,
                reference=_PSI_REFERENCE, fallback_reference=_PSI_FALLBACK,
            )
            out[side] = {
                "shoulder_m": [round(float(v), 4) for v in shoulder],
                "elbow_m": [round(float(v), 4) for v in elbow],
                "tcp_m": [round(float(v), 4) for v in tcp],
                # The defect this project exists to fix: the robot elbow sat a
                # median 0.341 m lateral of its own shoulder all session while a
                # seated operator's elbow hangs tucked. Watch this number.
                "elbow_lateral_m": round(float(elbow[1] - shoulder[1]), 4),
                "elbow_vertical_m": round(float(elbow[2] - shoulder[2]), 4),
                "upper_arm_m": round(float(np.linalg.norm(elbow - shoulder)), 4),
                "forearm_m": round(float(np.linalg.norm(tcp - elbow)), 4),
                "reach_m": round(float(np.linalg.norm(tcp - shoulder)), 4),
                "swivel_deg": (
                    round(float(np.degrees(geometry.angle_rad)), 2) if geometry else None
                ),
                "swivel_radius_m": round(float(geometry.radius_m), 4) if geometry else None,
            }
        out["joints_rad"] = {
            name: round(float(self.data.qpos[self._qadr[name]]), 5)
            for name in self.all_joints()
        }
        out["joints_deg"] = {
            name: round(float(np.degrees(self.data.qpos[self._qadr[name]])), 2)
            for name in self.all_joints()
        }
        return out

    def render_png(self, camera: Camera, width: int, height: int) -> str:
        renderer = self._renderer_for(width, height)
        view = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, view)
        view.azimuth = camera.azimuth
        view.elevation = camera.elevation
        view.distance = camera.distance
        view.lookat[:] = camera.lookat
        renderer.update_scene(self.data, camera=view)
        pixels = renderer.render()
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG", optimize=False, compress_level=1)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def snapshot(self, joints: dict[str, float], camera: Camera, width: int, height: int):
        with self._lock:
            self.apply(joints)
            return {
                "png": self.render_png(camera, width, height),
                "measurements": self.measurements(),
            }

    def clearance(self, joints: dict[str, float]) -> dict[str, object]:
        """Self-clearance for a posed configuration, from the project's own checker.

        The same ClearanceChecker the pipeline gates on, at the same SIM floor, so
        a pose the HUD calls unreachable is a pose the supervisor would reject.
        Baseline pairs are already excluded from the headline minimum by the
        checker itself -- this does not re-implement that judgement.
        """
        with self._lock:
            if self._checker is None:
                from galbot_motion_studio.safety.clearance import (
                    HOME_QPOS,
                    ClearanceChecker,
                )
                from galbot_motion_studio.safety.profiles import (
                    MotionProfile,
                    clearance_kwargs_for,
                    clearance_floor_for,
                )

                self._checker = ClearanceChecker(
                    model=self.model, home=HOME_QPOS,
                    **clearance_kwargs_for(MotionProfile.SIM),
                )
                self._floor = float(clearance_floor_for(MotionProfile.SIM))

            pose = {
                name: float(value)
                for name, value in joints.items()
                if name in self._qadr
            }
            report = self._checker.check_pose(pose)
            pair = getattr(report, "min_pair", None)
            key = getattr(pair, "key", None)
            names = (
                f"{key.body1} | {key.body2}"
                if key is not None and hasattr(key, "body1")
                else (None if pair is None else str(pair)[:80])
            )
            return {
                "min_distance_m": round(float(report.min_distance), 5),
                "floor_m": self._floor,
                "blocked": bool(report.min_distance < self._floor),
                "pair": names,
                "violations": len(getattr(report, "violations", []) or []),
            }

    def describe_joints(self) -> list[dict[str, object]]:
        groups = [
            ("left arm", ARMS["left"]["joints"]),
            ("right arm", ARMS["right"]["joints"]),
            ("head", HEAD_JOINTS),
        ]
        described = []
        for group, names in groups:
            for name in names:
                low, high = self.joint_limits[name]
                described.append({
                    "name": name, "group": group,
                    "min_deg": round(float(np.degrees(low)), 1),
                    "max_deg": round(float(np.degrees(high)), 1),
                })
        return described
