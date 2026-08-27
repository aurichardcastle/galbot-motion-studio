# Workspace envelope — measured reach and clearance limits

Owner: control lane. Companion to `docs/model-notes.md` (model ground truth) and
`docs/safety-model.md` (failure modes). This file records what the arm mapping can
*safely and reachably* express, measured against the SIM profile
(`ClearanceChecker` floor **5 mm**) on the verified fixed-base model.

Everything here is measured, not derived. Reproduce with the sweeps in §6.

---

## 1. Why this file exists

The operator reported that the robot "only tracks my arms moving back and
forth" and that tracking felt bad. Both are measurable, and neither was a
missing feature — lateral and head tracking were already implemented.

Measured from the operator's own 372-frame session (`my_clip.json`), robot
TCP travel per axis:

| body | fwd/back (X) | **sideways (Y)** | up/down (Z) |
|---|---|---|---|
| `left_arm_link7` | 19.9 cm | **6.0 cm** | 43.9 cm |
| `right_arm_link7` | 23.3 cm | **3.3 cm** | 47.5 cm |

Sideways travel is **7–14× smaller** than vertical. The cause is that the
mapping is **anisotropic**: `LeftArmPolicy` scales lateral at 0.16 m and depth
at 0.12 m against vertical at 0.45 m. Identical physical hand displacement
therefore produces ~2.8× less robot motion sideways than vertically. The
mapping distorts the operator's motion before any safety layer is involved.

## 2. Coordinate conventions (easy to get backwards)

- **MediaPipe image Y points down.** A *negative* `vertical_signal` means the
  operator raised their hands.
- **Positive `depth_signal` moves the target toward the torso.**
  `target_x = forward_offset_m - current_depth * depth_scale_m`. Hands pulled
  in toward the chest are what drive the robot's hands into its own body.
- **Lateral signal is raw image-X for both arms.** A single positive signal
  moves *both* wrists in world +Y, so one arm travels outward and the other
  crosses the body. The crossing arm is always the binding constraint.
  `lateral_reference = +0.25` (left) / `−0.25` (right) offsets each arm to its
  own side before scaling.

## 3. Single-axis envelope at the current scales (0.16 / 0.45 / 0.12)

Both arms driven, elbow signals at half the wrist signal, cold-start IK.

| axis | result |
|---|---|
| depth, ±2.0 | safe throughout, ~11.4–12.4 mm clearance. **Unconstrained.** |
| lateral, outward to signal +2.0 | safe throughout, 11.9–12.3 mm. **Unconstrained** (reaches world Y = 0.5685). |
| lateral, crossing | safe to world Y ≈ −0.048 (10.89 mm); **blocked** at −0.008 (4.64 mm). |
| vertical | usable band only **signal −0.5 … +1.2**; outside it IK does not converge. |

The vertical band is narrower than `max_signal = 2.0` allows, so the extremes
of the configured signal range are not reachable at all.

## 4. Position alone does not predict safety

A tempting fix is to clamp the wrist target to keep it on its own side of the
centreline. **That rule is wrong**, and the measurement says so directly:

| lateral scale | wrist target Y | clearance | verdict |
|---|---|---|---|
| 0.16 | −0.0480 | +10.89 mm | ok |
| 0.45 | −0.0509 | **−1.58 mm** | BLOCKED |

Nearly the same wrist position, opposite verdict. The elbow target scales with
the same policy, so the whole arm *configuration* differs between the two
scales. Any clamp must be expressed in signal space for a specific scale, or be
validated by the clearance gate itself — never by wrist position alone.

## 5. Safe inward (body-crossing) signal limit vs scale

Largest inward signal for the crossing arm that still clears 5 mm, bisected to
7 iterations over a vertical × depth grid. `unsafe@0` means the pose already
violates the floor at zero lateral signal.

**scale 0.16 (current)** — most conservative limit **0.50**

| vertical | depth −1 | depth 0 | depth +1 |
|---|---|---|---|
| −0.4 | 1.98 | 1.98 | `unsafe@0` |
| 0.0 | 1.98 | 1.48 | 0.50 |
| 0.6 | 1.98 | 1.56 | 1.09 |
| 1.2 | `unsafe@0` | `unsafe@0` | 0.92 |

**scale 0.30** — most conservative limit **0.14**

| vertical | depth −1 | depth 0 | depth +1 |
|---|---|---|---|
| −0.4 | 1.31 | 1.08 | `unsafe@0` |
| 0.0 | 1.25 | 0.77 | `unsafe@0` |
| 0.6 | 0.61 | 0.77 | 0.22 |
| 1.2 | `unsafe@0` | 0.14 | 0.50 |

**scale 0.45 (isotropic)** — most conservative limit **0.11**

| vertical | depth −1 | depth 0 | depth +1 |
|---|---|---|---|
| −0.4 | 0.58 | 0.77 | `unsafe@0` |
| 0.0 | 0.45 | 0.56 | `unsafe@0` |
| 0.6 | 0.11 | 0.55 | 0.22 |
| 1.2 | `unsafe@0` | 0.19 | 0.36 |

**Two conclusions.**

1. **The current configuration already has unsafe regions**, at zero lateral
   signal — hands pulled toward the chest at a vertical extreme. It has not
   bitten because the operator has not produced those signal combinations, not
   because the gate prevents it. The gate correctly rejects them; a rejection is
   a frozen frame.
2. **Raising the scale trades expressiveness for freezing.** Going 0.16 → 0.45
   shrinks the worst-case inward limit from 0.50 to 0.11. More reach per unit of
   hand motion, but a much larger share of the workspace rejected.

## 5a. Decision taken (2026-08-10)

`lateral_scale_m` **0.16 → 0.28**. `depth_scale_m` and `vertical_scale_m` unchanged.

Measured at the new scale, largest safe inward signal for the crossing arm:

| vertical | depth −1 | depth 0 | depth +1 |
|---|---|---|---|
| −0.4 | 1.77 | 1.16 | `unsafe@0` |
| 0.0 | 1.58 | **0.81** | 0.61 |
| 0.6 | 1.36 | 0.81 | 0.64 |
| 1.2 | `unsafe@0` | 0.12 | 0.27 |

- Typical pose (vertical 0, depth 0) inward limit **1.48 → 0.81**. Reduced, but the
  operator's recorded session produced lateral signals spanning only ≈0.2, so this
  keeps roughly 4× margin over observed use.
- `unsafe@0` cells **3 → 2**. Counter-intuitive but real: at zero lateral signal the
  neutral offset is `±0.25 × scale`, so each arm now parks 7 cm out from the shoulder
  rather than 4 cm — further from the torso. Raising lateral scale moved two
  previously-unsafe cells into the safe region.

**Depth was deliberately left at 0.12.** Raising it to 0.28 made `depth +1` at
neutral vertical `unsafe@0` — hands drawn toward the chest, an ordinary motion.
The chest-ward limit is a fixed *distance*, not a signal: `scale × safe-signal`
is near-constant at ≈0.058 (0.48 at scale 0.12, 0.36 at 0.16, 0.31 at 0.18,
0.20 at 0.28), so a larger depth scale reaches the same wall on less signal and
buys nothing. Forward/back was also already the second-largest measured axis
(19.9–23.3 cm), i.e. the axis that was already working.

Further headroom exists: lateral 0.45 would give 2.8× the original range with a
typical-pose inward limit of 0.56. Not taken, because the margin over observed
operator signals gets thin and every clearance rejection is a frozen frame.

## 6. Reproduction

All sweeps used the SIM profile and the verified fixed-base model:

```python
m  = load_verified_fixed_base_model()
ck = ClearanceChecker(**clearance_kwargs_for(MotionProfile.SIM), model=m, home=HOME_QPOS)
r  = RightArmRetargeter(m, neutral_pose=HOME_QPOS, policy=LeftArmPolicy(...)).map(
         LeftArmIntent(lat, vert, dep, lat*0.5, vert*0.5, dep*0.5),
         previous=None, elapsed_s=None)
ck.check_pose({**HOME_QPOS, **r.target.joint_map})
```

## 7. A synthetic-sweep result that does NOT hold live

Driving `LeftArmRetargeter.map` directly with a warm start and a raw ramped
signal produces heavy `JOINT_CONTINUITY` and `IK_DID_NOT_CONVERGE` rejection —
at a 0.05 signal step, 40 of 41 frames. **This does not reproduce in the live
pipeline**, which filters the intent before retargeting: the operator's real
372-frame session recorded **zero** IK failures and **zero** continuity
rejections. The sweep drives a layer the live path never drives raw.

Recorded here so the next person does not re-derive it and mistake it for a
live defect. The live rejection reasons in that session were exclusively
`observation: LOW_CONFIDENCE`.

## 8. What actually froze the operator's session

| measure | value |
|---|---|
| frames | 372 (315 ALLOW / 57 HOLD) |
| holds | 57, all `observation: LOW_CONFIDENCE` |
| distinct runs | 8 |
| longest run | **27 frames ≈ 2.11 s at 12.8 fps** |
| second longest | 14 frames ≈ 1.1 s |
| frames whose target equalled the previous frame's | 49 / 371 |

The holds are not scattered stutter; they are a few long freezes. The cause is
structural: `FreshnessPolicy.min_confidence` is compared against
`observation.aggregate_confidence`, which is a **minimum across both wrists,
both shoulders and three face points**. One landmark dipping — a wrist
occluded as the hands cross — holds the **entire** robot, head included.

The clearance gate has the same global property: it evaluates a whole pose, so
one unsafe arm freezes every joint.

Making both gates **per control group** (left arm / right arm / head) would
mean a dropped left wrist holds only the left arm while the head and right arm
keep tracking. This is more precise rather than more permissive: a command is
untrustworthy only when the landmark that drives *that* command is
untrustworthy. Both gates live in files owned by the perception and safety lanes
(`vision/freshness.py`, `safety/supervisor.py`, `pipeline.py`); this file
records the measurement and the argument, not an implementation.
