"""Compare human and robot ARM SHAPE on the same frames, without hips.

The quantity compared is the upper arm's elevation away from the operator's own
shoulder line: the angle between the shoulder->elbow segment and the outward
lateral direction. 0 deg is an elbow straight out sideways (the "T-pose" failure
this project has chased since preview8), 90 deg is an elbow hanging tucked.

Why this metric
---------------
It is NOT the swivel angle. Commanding psi and then scoring psi would be
circular -- the solver is told to match it, so of course it does. The upper arm's
elevation is a different, independently observable quantity: it is what a person
in the room actually sees, and neither mapper optimises it directly.

It also needs no hips. The lateral direction comes from the shoulder line, and
the elevation is an angle between two directions, so it is invariant to any
rigid rotation of the operator relative to the camera.

    PYTHONPATH=src .venv/bin/python tools/arm_shape_ab.py \
        artifacts/live-camera-input/production-ab-v3/raw.landmarks.json \
        artifacts/swivel-ab/baseline.json artifacts/swivel-ab/candidate.json
"""

from __future__ import annotations

import json
import statistics as st
import sys

import mujoco
import numpy as np

from galbot_motion_studio.model.loader import load_verified_fixed_base_model

MIN_CONF = 0.35
SIDES = {
    "left": {"sho": "pose_11", "elb": "pose_13", "wri": "pose_15", "opp": "pose_12"},
    "right": {"sho": "pose_12", "elb": "pose_14", "wri": "pose_16", "opp": "pose_11"},
}


def unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    return vector / norm


def human_elevation_deg(landmarks: dict, side: str) -> float | None:
    """Angle between the operator's upper arm and their own outward lateral."""
    names = SIDES[side]
    if not all(
        key in landmarks and min(landmarks[key]["visibility"], landmarks[key]["presence"]) >= MIN_CONF
        for key in names.values()
    ):
        return None
    points = {}
    for role, key in names.items():
        coordinates = landmarks[key].get("world_xyz_m")
        if coordinates is None:
            return None
        points[role] = np.asarray(coordinates, dtype=np.float64)

    lateral = unit(points["sho"] - points["opp"])
    upper = unit(points["elb"] - points["sho"])
    if lateral is None or upper is None:
        return None
    return float(np.degrees(np.arccos(np.clip(float(np.dot(upper, lateral)), -1.0, 1.0))))


class RobotShape:
    """Forward-kinematic elevation of the robot's upper arm, same definition."""

    def __init__(self) -> None:
        self.model = load_verified_fixed_base_model()
        self.data = mujoco.MjData(self.model)
        self.body = {}
        for side in SIDES:
            self.body[side] = (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm_link1"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm_link4"),
            )

    def elevation_deg(self, joints: dict[str, float], side: str) -> float | None:
        self.data.qpos[:] = 0.0
        for name, value in joints.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = float(value)
        mujoco.mj_forward(self.model, self.data)
        shoulder_id, elbow_id = self.body[side]
        upper = unit(self.data.xpos[elbow_id] - self.data.xpos[shoulder_id])
        if upper is None:
            return None
        # Outward lateral in robot axes: +Y for the left arm, -Y for the right.
        lateral = np.array([0.0, 1.0 if side == "left" else -1.0, 0.0])
        return float(np.degrees(np.arccos(np.clip(float(np.dot(upper, lateral)), -1.0, 1.0))))


def load_clip(path: str) -> dict[int, dict[str, float]]:
    """Robot pose at EVERY frame, forward-filling between commands.

    ``observed_joints_rad`` appears only on frames that issued a new command, and
    the two mappers do not issue on the same frames -- rate limiting and solver
    rejection differ. Intersecting the commanded frames alone would score the two
    on different subsets of the gesture, which is the "same ordinal frame"
    mistake this project has already been bitten by once.

    The robot holds its last commanded pose between commands, so forward-filling
    is not an interpolation: it is what the twin was actually showing.
    """
    document = json.load(open(path))
    out: dict[int, dict[str, float]] = {}
    current: dict[str, float] | None = None
    for frame in sorted(document["frames"], key=lambda item: int(item["source_sequence"])):
        joints = frame.get("observed_joints_rad")
        if joints:
            current = {name: float(value) for name, value in joints}
        if current is not None:
            out[int(frame["source_sequence"])] = current
    return out


def main() -> None:
    sidecar_path, baseline_path, candidate_path = sys.argv[1:4]
    sidecar = json.load(open(sidecar_path))
    human = {
        int(frame["source_sequence"]): frame["landmarks"]
        for frame in sidecar["frames"]
        if frame["landmarks"]
    }
    baseline = load_clip(baseline_path)
    candidate = load_clip(candidate_path)
    robot = RobotShape()

    common = sorted(set(human) & set(baseline) & set(candidate))
    print(f"common frames: {len(common)}\n")

    for side in SIDES:
        rows = []
        for sequence in common:
            operator = human_elevation_deg(human[sequence], side)
            if operator is None:
                continue
            base = robot.elevation_deg(baseline[sequence], side)
            cand = robot.elevation_deg(candidate[sequence], side)
            if base is None or cand is None:
                continue
            rows.append((operator, base, cand))

        if not rows:
            print(f"{side.upper()}: no scorable frames")
            continue

        operator_deg = [row[0] for row in rows]
        base_err = [abs(row[1] - row[0]) for row in rows]
        cand_err = [abs(row[2] - row[0]) for row in rows]
        improved = sum(1 for b, c in zip(base_err, cand_err) if c < b)

        print(f"{side.upper()}: scored on {len(rows)} frames")
        print(f"  operator elevation  median={st.median(operator_deg):6.2f} deg  "
              f"range=[{min(operator_deg):.1f}, {max(operator_deg):.1f}]")
        print(f"  wrist-primary   error median={st.median(base_err):6.2f} deg  "
              f"mean={st.mean(base_err):6.2f}  p95={sorted(base_err)[int(0.95*(len(base_err)-1))]:6.2f}")
        print(f"  swivel-priority error median={st.median(cand_err):6.2f} deg  "
              f"mean={st.mean(cand_err):6.2f}  p95={sorted(cand_err)[int(0.95*(len(cand_err)-1))]:6.2f}")
        delta = st.median(base_err) - st.median(cand_err)
        print(f"  --> median improvement {delta:+.2f} deg; "
              f"candidate closer on {improved}/{len(rows)} frames "
              f"({100*improved/len(rows):.1f}%)\n")


if __name__ == "__main__":
    main()
