"""Frame-by-frame: what the operator did against what the robot did.

Every quantity here is measured on both bodies with the SAME definition, and
none of them is a solver residual. The operator's side comes from MediaPipe
world landmarks; the robot's side comes from forward kinematics on the joints
that were actually commanded.

Emits a per-frame CSV plus a segmented summary.

    PYTHONPATH=src .venv/bin/python tools/rehearsal_forensics.py \
        <raw.mp4> <wrist-primary.json> <swivel-priority.json> <out.csv>
"""

from __future__ import annotations

import json
import statistics as st
import sys

import cv2
import mujoco
import numpy as np

from galbot_motion_studio.adapters.mediapipe_holistic import MediaPipeHolisticDetector
from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.ports.frames import CapturedFrame
from galbot_motion_studio.retarget.swivel_ik import swivel_geometry

MIN_CONF = 0.35
HEAD_CHAIN = ("face_1", "face_33", "face_263")
SIDES = {
    "left": dict(sho="pose_11", opp="pose_12", elb="pose_13", wri="pose_15"),
    "right": dict(sho="pose_12", opp="pose_11", elb="pose_14", wri="pose_16"),
}


def unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if (not np.isfinite(n) or n < 1e-9) else v / n


class Human:
    """Operator-side geometry, hip-free."""

    @staticmethod
    def measure(lm: dict, side: str) -> dict | None:
        k = SIDES[side]
        if not all(
            n in lm and min(lm[n]["visibility"], lm[n]["presence"]) >= MIN_CONF
            for n in k.values()
        ):
            return None
        p = {}
        for role, name in k.items():
            w = lm[name].get("world_xyz_m")
            if w is None:
                return None
            p[role] = np.asarray(w, dtype=np.float64)

        lateral = unit(p["sho"] - p["opp"])
        upper = unit(p["elb"] - p["sho"])
        fore = unit(p["wri"] - p["elb"])
        if lateral is None or upper is None or fore is None:
            return None

        geom = swivel_geometry(
            p["sho"], p["elb"], p["wri"],
            reference=(p["sho"] - p["opp"]),
            fallback_reference=np.array([0.0, -1.0, 0.0]),
        )
        return {
            "elevation_deg": float(np.degrees(np.arccos(np.clip(np.dot(upper, lateral), -1, 1)))),
            "elbow_flex_deg": float(np.degrees(np.arccos(np.clip(np.dot(upper, fore), -1, 1)))),
            "psi_deg": None if geom is None else float(np.degrees(geom.angle_rad)),
            # Wrist offset from shoulder, normalised by the operator's own upper-arm
            # length so it is comparable to the robot without a scale factor.
            "reach": float(np.linalg.norm(p["wri"] - p["sho"]))
            / max(float(np.linalg.norm(p["elb"] - p["sho"])), 1e-9),
        }


class Robot:
    """Robot-side geometry from the joints that were actually commanded."""

    def __init__(self) -> None:
        self.model = load_verified_fixed_base_model()
        self.data = mujoco.MjData(self.model)
        self.ids = {}
        for side in SIDES:
            self.ids[side] = tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in (
                    f"{side}_arm_link1", f"{side}_arm_link4", f"{side}_gripper_tcp_link"
                )
            )

    def measure(self, joints: dict[str, float], side: str) -> dict | None:
        self.data.qpos[:] = 0.0
        for name, value in joints.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[int(self.model.jnt_qposadr[jid])] = float(value)
        mujoco.mj_forward(self.model, self.data)
        sho_id, elb_id, tcp_id = self.ids[side]
        sho, elb, tcp = (
            self.data.xpos[sho_id].copy(),
            self.data.xpos[elb_id].copy(),
            self.data.xpos[tcp_id].copy(),
        )
        lateral = np.array([0.0, 1.0 if side == "left" else -1.0, 0.0])
        upper = unit(elb - sho)
        fore = unit(tcp - elb)
        if upper is None or fore is None:
            return None
        geom = swivel_geometry(
            sho, elb, tcp, reference=lateral,
            fallback_reference=np.array([0.0, 0.0, -1.0]),
        )
        return {
            "elevation_deg": float(np.degrees(np.arccos(np.clip(np.dot(upper, lateral), -1, 1)))),
            "elbow_flex_deg": float(np.degrees(np.arccos(np.clip(np.dot(upper, fore), -1, 1)))),
            "psi_deg": None if geom is None else float(np.degrees(geom.angle_rad)),
            "reach": float(np.linalg.norm(tcp - sho))
            / max(float(np.linalg.norm(elb - sho)), 1e-9),
        }


def load_clip(path: str) -> tuple[dict[int, dict[str, float]], dict[int, dict]]:
    doc = json.load(open(path))
    poses: dict[int, dict[str, float]] = {}
    meta: dict[int, dict] = {}
    current: dict[str, float] | None = None
    for f in sorted(doc["frames"], key=lambda x: int(x["source_sequence"])):
        seq = int(f["source_sequence"])
        if f.get("observed_joints_rad"):
            current = {n: float(v) for n, v in f["observed_joints_rad"]}
        if current is not None:
            poses[seq] = current
        meta[seq] = {
            "outcome": f["decision"]["outcome"],
            "held": tuple(f.get("held_groups") or ()),
        }
    return poses, meta


def main() -> None:
    video, base_path, cand_path, out_csv = sys.argv[1:5]
    base_poses, base_meta = load_clip(base_path)
    cand_poses, cand_meta = load_clip(cand_path)
    robot = Robot()

    rows = []
    with MediaPipeHolisticDetector() as det:
        cap = cv2.VideoCapture(video)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            obs = det.detect(
                CapturedFrame(
                    image_bgr=frame, sequence=i, capture_mono_ns=int(i * 1e9 / 30.0),
                    source_clock_id="forensics", source_kind="recorded-video",
                ),
                calibration_id="forensics",
            )
            lm = {l.name: {
                "visibility": l.visibility, "presence": l.presence,
                "world_xyz_m": l.world_xyz_m,
            } for l in obs.landmarks}
            head_ok = all(
                n in lm and min(lm[n]["visibility"], lm[n]["presence"]) >= MIN_CONF
                for n in HEAD_CHAIN
            )
            row = {"frame": i, "head_seen": head_ok}
            for side in SIDES:
                h = Human.measure(lm, side)
                b = robot.measure(base_poses[i], side) if i in base_poses else None
                c = robot.measure(cand_poses[i], side) if i in cand_poses else None
                row[f"{side}_human_elev"] = None if h is None else h["elevation_deg"]
                row[f"{side}_human_flex"] = None if h is None else h["elbow_flex_deg"]
                row[f"{side}_human_psi"] = None if h is None else h["psi_deg"]
                row[f"{side}_human_reach"] = None if h is None else h["reach"]
                for tag, r in (("base", b), ("cand", c)):
                    row[f"{side}_{tag}_elev"] = None if r is None else r["elevation_deg"]
                    row[f"{side}_{tag}_flex"] = None if r is None else r["elbow_flex_deg"]
                    row[f"{side}_{tag}_psi"] = None if r is None else r["psi_deg"]
                    row[f"{side}_{tag}_reach"] = None if r is None else r["reach"]
                row[f"{side}_held_cand"] = f"{side}_arm" in (
                    cand_meta.get(i, {}).get("held") or ()
                )
            row["outcome_cand"] = cand_meta.get(i, {}).get("outcome")
            rows.append(row)

    keys = list(rows[0].keys())
    with open(out_csv, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")

    # ---- summary ----
    n = len(rows)
    print(f"frames: {n}\n")
    print(f"head seen: {sum(1 for r in rows if r['head_seen'])}/{n} "
          f"({100*sum(1 for r in rows if r['head_seen'])/n:.1f}%)")
    print()
    for side in SIDES:
        scored = [r for r in rows
                  if r[f"{side}_human_elev"] is not None
                  and r[f"{side}_base_elev"] is not None
                  and r[f"{side}_cand_elev"] is not None]
        print(f"=== {side.upper()} ARM  (scored on {len(scored)}/{n} frames)")
        if not scored:
            continue
        for label, quantity in (("upper-arm elevation", "elev"), ("elbow flexion", "flex")):
            be = [abs(r[f"{side}_base_{quantity}"] - r[f"{side}_human_{quantity}"]) for r in scored]
            ce = [abs(r[f"{side}_cand_{quantity}"] - r[f"{side}_human_{quantity}"]) for r in scored]
            better = sum(1 for b, c in zip(be, ce) if c < b)
            hv = [r[f"{side}_human_{quantity}"] for r in scored]
            print(f"  {label}:")
            print(f"    operator range      {min(hv):6.1f} .. {max(hv):6.1f} deg "
                  f"(median {st.median(hv):.1f})")
            print(f"    wrist-primary  err  median {st.median(be):6.2f}  mean {st.mean(be):6.2f}")
            print(f"    swivel-priority err median {st.median(ce):6.2f}  mean {st.mean(ce):6.2f}"
                  f"   better on {better}/{len(scored)} ({100*better/len(scored):.0f}%)")
        held = sum(1 for r in rows if r[f"{side}_held_cand"])
        print(f"  limb held: {held}/{n} ({100*held/n:.1f}%)")
        print()

    print("=== worst 8 frames for swivel-priority, left arm (elevation)")
    worst = sorted(
        (r for r in rows
         if r["left_human_elev"] is not None and r["left_cand_elev"] is not None),
        key=lambda r: -abs(r["left_cand_elev"] - r["left_human_elev"]),
    )[:8]
    for r in worst:
        print(f"  frame {r['frame']:4d}  human {r['left_human_elev']:6.1f}  "
              f"robot {r['left_cand_elev']:6.1f}  err {abs(r['left_cand_elev']-r['left_human_elev']):6.1f}"
              f"  reach h={r['left_human_reach']:.2f} r={r['left_cand_reach']:.2f}"
              f"  held={r['left_held_cand']}")


if __name__ == "__main__":
    main()
