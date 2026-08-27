"""Fix the sign of LeftArmPolicy.swivel_gain by measurement, not by guessing.

The operator's psi and the robot's psi are measured in different bodies from
different reference directions. Whether they run the SAME way or OPPOSITE ways
is a fact about those two conventions. Getting it wrong does not fail loudly --
it drives the elbow smoothly in the wrong direction, which is exactly the class
of error this project has been bitten by before.

Method: drive the robot's psi a known amount either side of its neutral, and
report where the elbow actually goes. Then compare against what the operator's
psi does anatomically over the retained capture.

    PYTHONPATH=src .venv/bin/python tools/swivel_sign_check.py
"""

from __future__ import annotations

import numpy as np

from galbot_motion_studio.model.joint_map import CONTROL_GROUPS
from galbot_motion_studio.pipeline import SIM_TELEOP_HOME_QPOS as SIM_NEUTRAL_QPOS
from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.retarget.swivel_ik import swivel_geometry
from galbot_motion_studio.retarget.swivel_solver_mink import (
    _PSI_FALLBACK,
    _PSI_REFERENCE,
    SwivelArmSolver,
)

import mujoco


def main() -> None:
    model = load_verified_fixed_base_model()
    data = mujoco.MjData(model)

    for side in ("left", "right"):
        joint_names = CONTROL_GROUPS[f"{side}_arm"]
        tcp_name = f"{side}_gripper_tcp_link"
        tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tcp_name)
        elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm_link4")
        shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm_link1")

        qpos = np.zeros(model.nq, dtype=np.float64)
        for name, value in SIM_NEUTRAL_QPOS.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos[int(model.jnt_qposadr[joint_id])] = float(value)

        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        neutral_tcp = data.xpos[tcp_id].copy()
        neutral_rot = data.xmat[tcp_id].reshape(3, 3).copy()
        neutral_shoulder = data.xpos[shoulder_id].copy()
        neutral_elbow = data.xpos[elbow_id].copy()

        geometry = swivel_geometry(
            neutral_shoulder,
            neutral_elbow,
            neutral_tcp,
            reference=_PSI_REFERENCE,
            fallback_reference=_PSI_FALLBACK,
        )
        assert geometry is not None
        neutral_psi = geometry.angle_rad

        # The elbow's height and lateral offset relative to the shoulder are what
        # an observer in the room actually sees. "Tucked" is low and close.
        def describe(elbow: np.ndarray) -> tuple[float, float]:
            offset = elbow - neutral_shoulder
            lateral = abs(float(offset[1]))
            height = float(offset[2])
            return lateral, height

        base_lateral, base_height = describe(neutral_elbow)
        print(f"=== {side.upper()} arm")
        print(f"  robot neutral psi = {np.degrees(neutral_psi):+.2f} deg")
        print(f"  neutral elbow: lateral={base_lateral:.4f} m  height={base_height:+.4f} m")

        solver = SwivelArmSolver(
            model,
            joint_names=joint_names,
            tcp_body_id=tcp_id,
            elbow_body_id=elbow_id,
            shoulder_body_id=shoulder_id,
            tcp_frame_name=tcp_name,
        )

        for delta_deg in (-30.0, -15.0, +15.0, +30.0):
            solution = solver.solve(
                seed_qpos=qpos,
                wrist_position_m=neutral_tcp,
                wrist_rotation=neutral_rot,
                target_swivel_rad=neutral_psi + np.radians(delta_deg),
            )
            probe = np.zeros(model.nq, dtype=np.float64)
            probe[:] = qpos
            for name, value in solution.joints_rad:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                probe[int(model.jnt_qposadr[joint_id])] = value
            data.qpos[:] = probe
            mujoco.mj_forward(model, data)
            lateral, height = describe(data.xpos[elbow_id])
            print(
                f"  psi {delta_deg:+6.1f} deg -> elbow lateral {lateral:.4f} m "
                f"({lateral - base_lateral:+.4f})  height {height:+.4f} m "
                f"({height - base_height:+.4f})  wrist_res={solution.wrist_residual_m:.6f} m "
                f"{'converged' if solution.converged else 'PARTIAL'}"
            )
        print()


if __name__ == "__main__":
    main()
