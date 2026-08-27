"""Run both swivel solvers over identical targets and append the results to disk.

    python tools/solver_bench.py [--out artifacts/solver-bench.jsonl]

BORROWED  mink (Apache-2.0) QP with hard equality constraints, plus qpsolvers
          (LGPL-3.0) and daqp (MIT) underneath it.
OWN       the same architecture as a 10x10 KKT system solved with numpy, no
          third-party solver at all.

Both are driven from the same swivel mathematics in retarget/swivel_ik.py, which
has no dependency either, so the comparison isolates the solver rather than the
formulation.

One JSON object per line, appended, never rewritten -- so runs accumulate and an
older result stays readable after the code that produced it has changed. Every
row carries the model hash and the implementation hash, so a row can always be
attributed to the code that produced it.

Simulation only: this poses a model and measures it. Nothing is commanded.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from galbot_motion_studio.model.loader import load_verified_fixed_base_model  # noqa: E402
from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST  # noqa: E402
from galbot_motion_studio.pipeline import SIM_TELEOP_HOME_QPOS  # noqa: E402
from galbot_motion_studio.retarget.swivel_ik import swivel_geometry  # noqa: E402
from galbot_motion_studio.retarget.swivel_solver_numpy import (  # noqa: E402
    SwivelArmSolverNumpy,
)

REFERENCE = np.array([0.0, 0.0, -1.0])
FALLBACK = np.array([1.0, 0.0, 0.0])
ARM = tuple(f"left_arm_joint{i}" for i in range(1, 8))


def _implementation_hash() -> str:
    digest = sha256()
    for path in sorted((ROOT / "src" / "galbot_motion_studio").rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _bodies(model):
    return {
        key: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for key, name in (
            ("shoulder", "left_arm_link1"),
            ("elbow", "left_arm_link4"),
            ("tcp", "left_gripper_tcp_link"),
        )
    }


def _measure(model, data, ids, qpos) -> dict[str, float | None]:
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    shoulder, elbow, tcp = (data.xpos[ids[k]] for k in ("shoulder", "elbow", "tcp"))
    geometry = swivel_geometry(
        shoulder, elbow, tcp, reference=REFERENCE, fallback_reference=FALLBACK
    )
    return {
        "elbow_lateral_m": round(float(elbow[1] - shoulder[1]), 6),
        "elbow_vertical_m": round(float(elbow[2] - shoulder[2]), 6),
        "swivel_deg": None if geometry is None else round(float(np.degrees(geometry.angle_rad)), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "solver-bench.jsonl")
    parser.add_argument("--label", default="", help="free-text note stored with the run")
    parser.add_argument("--sweep", type=int, default=19, help="swivel targets across +-45 deg")
    args = parser.parse_args()

    model = load_verified_fixed_base_model()
    data = mujoco.MjData(model)
    ids = _bodies(model)

    seed = np.zeros(model.nq)
    for name, value in SIM_TELEOP_HOME_QPOS.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0:
            seed[int(model.jnt_qposadr[joint])] = float(value)

    data.qpos[:] = seed
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    tcp0 = data.xpos[ids["tcp"]].copy()
    rot0 = data.xmat[ids["tcp"]].reshape(3, 3).copy()
    neutral = swivel_geometry(
        data.xpos[ids["shoulder"]], data.xpos[ids["elbow"]], tcp0,
        reference=REFERENCE, fallback_reference=FALLBACK,
    )
    if neutral is None:
        print("neutral pose has no swivel geometry", file=sys.stderr)
        return 2

    solvers: dict[str, object] = {}
    solvers["own"] = SwivelArmSolverNumpy(
        model, joint_names=ARM, tcp_body_id=ids["tcp"],
        elbow_body_id=ids["elbow"], shoulder_body_id=ids["shoulder"],
    )
    borrowed_error = None
    try:
        from galbot_motion_studio.retarget.swivel_solver_mink import SwivelArmSolver

        solvers["borrowed"] = SwivelArmSolver(
            model, joint_names=ARM, tcp_body_id=ids["tcp"],
            elbow_body_id=ids["elbow"], shoulder_body_id=ids["shoulder"],
            tcp_frame_name="left_gripper_tcp_link",
        )
    except Exception as error:  # noqa: BLE001
        borrowed_error = f"{type(error).__name__}: {error}"
        print(f"borrowed solver unavailable ({borrowed_error}); benching own only")

    targets = np.linspace(-45.0, 45.0, args.sweep)
    rows = []
    for variant, solver in solvers.items():
        for delta in targets:
            want = neutral.angle_rad + np.radians(float(delta))
            started = time.perf_counter()
            if variant == "borrowed":
                solution = solver.solve(  # type: ignore[union-attr]
                    seed_qpos=seed, wrist_position_m=tcp0,
                    wrist_rotation=rot0, target_swivel_rad=want,
                )
            else:
                solution = solver.solve(  # type: ignore[union-attr]
                    seed_qpos=seed, wrist_position_m=tcp0, target_swivel_rad=want,
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            qpos = seed.copy()
            for (name, value), address in zip(
                solution.joints_rad,
                [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                 for n in ARM],
                strict=True,
            ):
                qpos[address] = value
            rows.append({
                "variant": variant,
                "delta_deg": round(float(delta), 3),
                "converged": bool(solution.converged),
                "iterations": int(solution.iterations),
                "solve_ms": round(elapsed_ms, 3),
                "wrist_residual_m": round(float(solution.wrist_residual_m), 8),
                "swivel_residual_deg": round(float(np.degrees(solution.swivel_residual_rad)), 3),
                **_measure(model, data, ids, qpos),
            })

    record = {
        "schema_version": "1.0",
        "kind": "solver-bench",
        "label": args.label,
        "model_hash": CANONICAL_MANIFEST.fixed_mjcf_sha256,
        "implementation_hash": _implementation_hash(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "neutral_swivel_deg": round(float(np.degrees(neutral.angle_rad)), 3),
        "neutral": _measure(model, data, ids, seed),
        "borrowed_error": borrowed_error,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"neutral swivel {record['neutral_swivel_deg']:+.2f} deg, "
          f"elbow lateral {record['neutral']['elbow_lateral_m']:+.4f} m")
    # Statistics over CONVERGED rows only. A failed solve leaves the arm wherever
    # it gave up, so mixing those into a residual percentile reports the failures
    # rather than the accuracy -- p95 across all rows read 0.14 m while converged
    # rows sat at 26 micrometres.
    print(f"\n{'variant':>9} {'conv':>7} {'wrist max (m)':>15} "
          f"{'|swivel| mean':>14} {'ms mean':>9} {'best tuck (m)':>14}")
    for variant in solvers:
        subset = [r for r in rows if r["variant"] == variant]
        converged = [r for r in subset if r["converged"]]
        if not converged:
            print(f"{variant:>9} {0:>3}/{len(subset):<3} {'-':>15} {'-':>14} {'-':>9} {'-':>14}")
            continue
        worst = max(r["wrist_residual_m"] for r in converged)
        swivel = np.mean([abs(r["swivel_residual_deg"]) for r in converged])
        tuck = min(r["elbow_lateral_m"] for r in converged)
        print(f"{variant:>9} {len(converged):>3}/{len(subset):<3} {worst:>15.8f} "
              f"{swivel:>14.3f} {np.mean([r['solve_ms'] for r in subset]):>9.2f} {tuck:>14.4f}")
    print("  conv = targets reached; stats are over converged rows only.")
    print(f"\nappended to {args.out.relative_to(ROOT)} "
          f"({sum(1 for _ in args.out.open()) } runs on file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
