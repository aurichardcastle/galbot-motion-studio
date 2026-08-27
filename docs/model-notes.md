# Galbot One Golf — model notes

**Status:** REFERENCE. This repository loads the hash-pinned description package
vendored under `third_party/galbot_one_golf_description/`. These notes describe
the simulator model, not a hardware-control contract.

> ### Verification status — read before citing anything here
>
> The two claims that change the simulator design were independently re-run:
>
> | claim | re-run result | verdict |
> |---|---|---|
> | In-memory `geom_contype` alone is insufficient, but `geom_contype` **+ `body_contype`** equals the on-disk patch | `A_none: 0 · B_geom: 0 · C_geom+body: 7 · D_ondisk: 7` | **CONFIRMED** |
> | `d.contact` is blind to many penetrating pairs, including arm-into-torso | `d.contact` → 4 body pairs; `mj_geomDistance` → **21** penetrating, **17 hidden**, incl. `left_arm_link1\|torso_base_link` at −10.720 mm | **CONFIRMED** |
>
> Everything else is single-sourced from the audit. Sections marked **SINGLE-SOURCED** have not
> been independently re-run; treat them as well-evidenced but not yet double-checked.
>
> **Package-copy caveat:** earlier source copies differed by 180° roll on the TCP
> links. **Every §4 figure must be re-checked against the canonical copy before use.** §6 (collision
> geometry) is unaffected — TCP links carry no collision geoms.

---

## 0 · Premises that were WRONG — read first

| # | Premise | Verdict |
|---|---|---|
| P1 | Model ships a suction cup on the right | **WRONG.** `suction` appears **zero times** in the package. The model has an identical parallel-jaw gripper on **both** arms. |
| P2 | SDK groups map onto model joints | **WRONG.** None of `left_gripper_joint1`, `right_suction_cup_joint1`, `chassis_joint1..4` exist in either the URDF or the MJCF. |
| P3 | The URDF contains `wheel_N_passive_*` rollers | **True of one URDF only.** `galbot_one_golf_fixed_base.urdf` has **zero**. |
| P4 | Patching `geom_contype` after load fails, so the XML must be patched on disk | **Half right — and the half that is wrong matters.** The stated fact is true; the conclusion is false. See §6.4. **independently re-verified.** |
| P5 | Arms mirrored — joint4 and joint6 have bounds swapped | **Confirmed, but the framing misleads.** Negation is the correct mirror for **all seven** joints. §3.2. |
| P6 | Home pose is hardware-measured | **Correct — and it disagrees with two files in this project.** §3.4. |

---

## 1 · Controlled joints

### 1.1 Contents *(single-sourced)*

| | count |
|---|---|
| `galbot_one_golf.urdf` joints / links | 103 / 104 |
| `galbot_one_golf_fixed_base.urdf` joints / links | 63 / 64 |
| `wheel_N_passive_*` in `galbot_one_golf.urdf` | 40 (4 wheels × 10 rollers), type `continuous` |
| `wheel_N_passive_*` in `galbot_one_golf_fixed_base.urdf` | **0** — the four `wheelN_joint` are type `fixed`, rollers absent |

Package `README.md` says *"Passive omni-wheel roller joints are omitted from preset URDFs"* — accurate
for the fixed-base preset, **inaccurate for `galbot_one_golf.urdf`**, which carries all 40.

### 1.2 Definitive controlled list

Model declaration order, via `mj_id2name` over `mjOBJ_JOINT` on `galbot_one_golf_fixed_base.xml`:

```
leg        (5)  leg_joint1..5                        [jnt idx  0– 4]
head       (2)  head_joint1..2                       [jnt idx  5– 6]
left_arm   (7)  left_arm_joint1..7                   [jnt idx  7–13]
left_gripper    left_gripper_joint   (drive)         [jnt idx 14]
right_arm  (7)  right_arm_joint1..7                  [jnt idx 20–26]
right_gripper   right_gripper_joint  (drive)         [jnt idx 27]
```

**Passive / structural, present but not independently commandable:** 10 gripper follower joints
(5 per hand, `<mimic>` in URDF / `<equality><joint>` in MJCF, `neq=10`, no actuator); 4 wheel drives
(wheeled MJCF only, velocity actuators, `ctrlrange="-25 25"`); 40 omni rollers; 3 planar-base joints
(planar MJCF only); 1 `freejoint` (wheeled MJCF only).

### 1.3 The SDK's joint names do not exist in the model ⭐

Hardware observations retained outside this repository were checked against both
model formats — **all zero matches**: `left_gripper_joint1`, `right_gripper_joint1`,
`right_suction_cup_joint1`, `chassis_joint1`–`chassis_joint4`.

**The 21 arm/leg/head names are byte-identical between SDK and model** and need no translation. The
other three groups do:

| SDK group | SDK name | Model equivalent | Status |
|---|---|---|---|
| `left_gripper` | `left_gripper_joint1` | `left_gripper_joint` | Rename only — but **units differ**. SDK read `+0.0500` against a model range of `[0, 1.703]` rad. The SDK value is probably metres of jaw width. **UNKNOWN.** |
| `right_suction_cup` | `right_suction_cup_joint1` | *nothing* | **No counterpart exists.** §4. |
| `chassis` | `chassis_joint1..4` | `wheel1_joint..wheel4_joint` (wheeled MJCF only) | **Plausible but UNVERIFIED** — nothing states the correspondence or the wheel ordering. |

Note "passive" means two different things: the SDK reports `chassis active=0`, while the wheeled MJCF
gives all four wheels velocity actuators. **Passive-in-SDK ≠ unactuated-in-model.**

---

## 2 · Model variants *(single-sourced except ncon, §6.4)*

Read off `mjModel` after load, not grepped.

| file | nq | nv | nu | ngeom | nbody | njnt | neq | base |
|---|---|---|---|---|---|---|---|---|
| `galbot_one_golf.xml` | 84 | 83 | 27 | 268 | 105 | 78 | 10 | **free** |
| `galbot_one_golf_fixed_base.xml` | **33** | **33** | **23** | **268** | 65 | 33 | 10 | **fixed** |
| `galbot_one_golf_planar_base.xml` | 36 | 36 | 26 | 268 | 65 | 36 | 10 | **planar** |

Cross-checks: `nq − nv = 1` for the free joint only. `84 − 33 = 51 = 7 (free) + 4 (wheel) + 40 (rollers)`.
`nbody 105 − 65 = 40` roller bodies. `ngeom` identical across all three: **184 collision** (`group==3`),
83 visual, 1 floor plane. All 23 robot actuators are `<position>`; wheel and planar-base actuators are
`<velocity>`.

*(Entry C-11 of the independent review reports `nq=33, nv=33, nu=23, ngeom=268` and 184 `contype=0` geoms for
the fixed-base model. **Two independent derivations agree.**)*

### 2.1 Use `galbot_one_golf_fixed_base.xml`

1. Only variant with **no base DOF**, so `jnt_qposadr[j] == j` for all 33 joints — the others need an
   offset of 7 or 3.
2. Identical 268 geoms / 184 collision geoms, so nothing is lost for clearance work.
3. `docs/mjcf.md` designates it for display/planning/fixed-base workflows.
4. `tools/galbot_preview.py:50` already targets it.

Planar is the fallback if base motion is ever modelled — note it is generated from the fixed-base
URDF, so it also has no rollers.

---

## 3 · Limits *(single-sourced)*

### 3.1 All three sources agree exactly

`jnt_range` vs `actuator_ctrlrange` vs URDF `<limit>` across all 23 actuated joints: **0 mismatches**,
`0.000e+00` on every bound. The checked-in manifest was machine-compared row by row: max error
`0.000e+00` on all 21 consistency rows and all 7 arm rows.

> **A retraction the audit made on itself, kept because the method matters.** A first pass printed
> joint ranges at `%.10g` and ctrlranges at `%.16g` and appeared to show a precision discrepancy on
> `left_arm_joint4` — a clean, plausible "MJCF rounds `<joint range>`" story. It was a **print-format
> artifact**; matched `%.17g` gave bit-identical values. This is the fourth time on this project a
> satisfying explanation has died on one more check.

**One cosmetic rounding:** `G1_JOINT_LIMITS.md:73` lists `head_joint2` range as `0.7079`; the true
value is `0.7079449`. Bounds themselves are exact.

**One genuine cross-source disagreement — torque.** URDF `<limit effort>` and MJCF
`actuator_forcerange` do not correspond, by 2–6.7× (and **0.10×** the other way on the grippers:
URDF 50 N·m vs MJCF ±5). The MJCF values come from `config/mjcf/joint_data.json`, a separate
hand-authored input. **Neither is a hardware torque limit.** `G1_JOINT_LIMITS.md` quotes the URDF column.

`G1_JOINT_LIMITS.md:78` (arms all have `acceleration=2, jerk=12, velocity=1.5`) is **confirmed** —
across all 14 arm joints the tuple set is exactly `{('2','12','1.5')}`. These are non-standard URDF
attributes, dropped by the MJCF conversion.

### 3.2 Mirroring — confirmed, with a correction to the framing ⭐

The bound swap is real: `right_lower = −left_upper`, `right_upper = −left_lower`. But machine check
over all 7 pairs gives `mirror(R = −L) maxerr = 0.000e+00` for **every** joint 1–7. Joints 1, 2, 3, 5,
7 have ranges symmetric about zero, so negation maps each range onto itself and the swap is invisible.
**Joints 4 and 6 are the only two with asymmetric ranges, which is why they are the only two where it
shows.**

Verified kinematically, not just from limits: setting `right_jointN = −left_jointN` for N=1..7 at four
random poses and running `mj_forward` gives `R_pos = (−L_x, L_y, L_z)` with **max error 0.000000 m**.
Competing hypotheses (mirror in y, mirror in z) were off by ~1.9 m.

**Practical rule: to mirror a whole-arm pose, negate all seven values.** Do not negate only 4 and 6.
The model's mirrored joint limits imply 4 and 6 are a special case; they are a *consequence* of
the mirror, not a separate rule.

### 3.3 Hardware validation stands

All 21 measured joints from the 18:06 capture fall strictly inside their model ranges. Independent
confirmation that this description matches this physical unit.

### 3.4 Three transcription errors in derived files ⭐

| joint | `VISIT2_FIELD_LOG.md` L563ff (primary capture) | `G1_JOINT_LIMITS.md` | `galbot_preview.py` |
|---|---|---|---|
| `left_arm_joint2` | **−1.4697** | −1.4698 (L33) | −1.4698 (L59) |
| `left_arm_joint6` | **−0.2749** | −0.2748 (L37) | −0.2748 (L60) |
| `left_arm_joint7` | **+0.1115** | +0.1114 (L38) | +0.1114 (L61) |

The other 18 agree everywhere. The field log is the primary capture, so **the two derived files carry
three last-digit transcription errors.**

**Impact: none.** Both poses give an identical contact set — same 7 contacts, same 4 body pairs,
`max |Δ| = 0.000e+00 m`. The three joints differ by 1e-4 rad and none of the affected links are in
contact. Worth fixing so the files stop diverging from the capture, not urgent.

---

## 4 · End effectors — the model is wrong on **both** arms ⭐

> ⚠️ Re-check against the independent review's canonical upstream package copy before use — the two copies differ by
> 180° roll on the TCP links (Entry C-11). Collision-geom counts below are unaffected: TCP links carry
> no collision geometry.

### 4.1 The finding

**The model carries an identical parallel-jaw gripper on both arms.** `grep -rni suction` over the
whole package returns **0 matches**. There is no suction cup, geometry, joint or link anywhere.

Hardware ground truth, from the teach pendant (`VISIT2_FIELD_LOG.md:410`):
> 当前场景 `base` · 左臂工具 `HUILING_GRAPPER` · 右臂工具 `GSUCKER`

So **both** end effectors are wrong: the left is a generic `galbot_gripper` standing in for a
HUILING_GRAPPER, and the right is that same generic gripper standing in for a **fundamentally
different device**.

`xacro/robot.xacro:7-9` shows this is build-time configuration — `left_ee_type` / `right_ee_type`
default to `galbot_gripper`, and the shipped URDFs were generated at the defaults. **No suction-cup
xacro exists** in `xacro/components/`, so **the correct model cannot be regenerated from these files.**

Corroborating: the package ships
`docs/.images/galbot_one_golf_left_hitbot_gripper_right_suction_cup_urdf.png` — a rendering of a
configuration the URDFs do not produce. Note that filename says **hitbot** while the pendant says
**HUILING** — a third name for the left tool. **UNKNOWN** which is correct.

### 4.2 Link inventory (per side, identical both sides)

10 links; collision-geom counts in brackets:
`*_arm_end_effector_mount_link` [0] → `*_gripper_flange_link` [0] → `*_gripper_base_link` [3] →
`*_gripper_r_knuckle_link` [8] ← **the driven DOF**, joint `*_gripper_joint` →
`*_gripper_r_inner_knuckle_link` [8] → `*_gripper_r_finger_link` [3] →
`*_gripper_l_knuckle_link` [8] → `*_gripper_l_inner_knuckle_link` [8] →
`*_gripper_l_finger_link` [3] → `*_gripper_tcp_link` [0] (`xyz="0.13996 0 0"`).

Adjacent but **not** end effector, also mounted off `*_arm_link7`: `*_arm_wrist_camera_stand` [0],
`*_wrist_camera_link` [1].

**Naming trap:** there is no `*_gripper_r_knuckle_joint`. The drive joint is `*_gripper_joint`.

### 4.3 How much of the collision model is untrustworthy

- **82 of 184 collision geoms — 44.6% — sit on the two end effectors** (41 per side).
- The wrong geometry extends **0.1814 m** from `*_arm_end_effector_mount_link`. TCP sits at 0.14646 m.
- Every one of the 41 right-side geoms models a device the robot does not have.

**This justifies `docs/safety-model.md` K6 — reject outright, do not downgrade to advisory.** It is not a
small correction at the fingertips: it is 45% of the collision model and up to 18 cm of geometry.

---

## 5 · Adjacency — 63 link pairs joined by a joint

Derived programmatically from `galbot_one_golf_fixed_base.urdf` by walking every `<joint>` and reading
`<parent>`/`<child>`. Verified complete against the MJCF body tree: `URDF pairs not in MJCF: []`,
`MJCF pairs not in URDF: [('base_link','world')]`.

**These 63 pairs must be excluded from any self-collision check.** Not reproduced here — derive them,
do not copy them, since a hand-maintained list drifts from the model. `galbot_one_golf.urdf` adds 44
more (4 wheel drives + 40 rollers), all inside the wheel subtrees.

**Adjacency is necessary but not sufficient.** Only 22 of the 63 pairs involve two links that both
carry collision geoms, and excluding by adjacency alone still leaves structurally-overlapping
non-adjacent pairs — §6.3.

---

## 6 · Baseline overlaps at the home pose

Model `mjcf/galbot_selfcollide.xml` (already on disk, created 2026-08-07 by `galbot_preview.py`;
identical to `galbot_one_golf_fixed_base.xml` except line 8, `contype="0"` → `"1"`).

### 6.1 What `d.contact` reports — 7 contacts, 4 body pairs **(independently re-verified)**

| body A | body B | penetration |
|---|---|---|
| `omni_chassis_base_link` | `leg_link1` | −172.062 mm |
| `torso_base_link` | `head_link2` | −73.665 mm |
| `leg_link1` | `wheel_1` / `wheel_4` | −1.935 mm |

Matches the three known examples in `galbot_preview.py:24-26` and is the baseline that script subtracts.

### 6.2 ⭐⭐ `d.contact` is blind to 17 further penetrating pairs — **independently re-verified**

Re-derived with `mj_geomDistance` over all collision-geom pairs on distinct bodies, bypassing MuJoCo's
contact filtering. The independent re-run:

```
d.contact                    →  4 body pairs
mj_geomDistance              → 21 penetrating body pairs
HIDDEN from d.contact        → 17
    −132.432 mm  omni_chassis_base_link | wheel_1/2/3/4     (same weld)
     −23.955 mm  leg_link4              | torso_base_link   (filterparent)
     −10.723 mm  right_arm_link1        | torso_base_link   (filterparent)
     −10.720 mm  left_arm_link1         | torso_base_link   (filterparent)
      −8.735 mm  right_arm_link6        | right_arm_link7   (filterparent)
```

> ⚠️ **Caveat added 16:40 — the "21 penetrating" figure is a floor, not an exact count.**
> `mj_geomDistance` has a confirmed failure mode in which it returns spurious `0.0` for mesh pairs
> (§6.5). The scan above classified a pair as penetrating on `dist < 0`, so a genuinely penetrating
> pair collapsed to `0.0` would have been **missed**. All three spurious zeros observed at this
> `distmax` were on adjacent or near-adjacent pairs that adjacency exclusion removes anyway, so the
> conclusion is unaffected in practice — but **21 is a lower bound.** `teleop/clearance.py` defends
> against this with a monotonicity ladder; this ad-hoc scan did not.

**The rule:** MuJoCo excludes a geom pair iff (a) the bodies share a `body_weldid`, or (b)
`filterparent` is on (it is — `opt.disableflags == 0`), neither weld is the world weld, and one body's
weld-parent is the other's weld. The world exemption is why `leg_link1 ↔ omni_chassis_base_link` *does*
appear despite being weld-parent/child.

Weld groups are large here because of the many `fixed` joints — notably `torso_base_link` is welded to
`leg_link5`, and `left_arm_link1` / `right_arm_link1` are its weld-children.

> ### ⚠️ Safety consequence — the most actionable finding in this document
>
> **`d.contact` can never report an arm-link1-into-torso collision, at any pose.** Same for
> `leg_link4 ↔ torso_base_link`.
>
> `tools/galbot_preview.py` builds its verdict purely from `d.ncon` / `d.contact` (lines 111–121), so
> **it inherits every one of these blind spots** — including all `*_arm_linkN ↔ linkN+1` pairs and the
> whole `*_arm_link7`/gripper weld. That tool is described in the project README as the way to *"check
> a pose before sending it to a robot."*
>
> **Precision matters here and the finding must not be overstated:** the 2026-08-07 incident pair
> (`left_arm_link5 ↔ leg_link2`) is **not** weld-related, so the tool *did* catch that one, and its
> reconstruction of the incident stands. **The blind spot is the torso, not the leg.**

Two sufficient mitigations:
1. Set `opt.disableflags |= mjDSBL_FILTERPARENT` and maintain the §5 adjacency exclusions yourself.
2. **Skip `d.contact` entirely; use `mj_geomDistance` over the geom-pair list**, subtracting the §6.3
   baseline. This also yields a *signed clearance* rather than a boolean — which is what a teleop loop
   needs, and what `teleop/clearance.py` is being built to do.

### 6.3 Pairs at or within 10 mm *(single-sourced)*

53 body pairs; 21 interpenetrate. **Non-adjacent, non-weld pairs a naive checker will false-positive
on, which must join the baseline even though §5 does not cover them:**
`leg_link1↔omni_chassis_base_link`, `torso_base_link↔head_link2`, `leg_link1↔wheel_1/4`,
`leg_link4↔torso_base_link`, `torso_base_link↔left/right_arm_link1`,
`*_arm_link7↔*_gripper_base_link`, `*_arm_link5↔*_arm_link7`, and the intra-gripper knuckle pairs.

Only **42 of 65 bodies carry collision geoms**. `head_link1` and `head_base_link` have **none** — head
yaw moves no collision geometry until `head_link2`.

### 6.4 ⭐ The on-disk XML patch is avoidable — **independently re-verified**

Controlled experiment at the home pose, `galbot_one_golf_fixed_base.xml`:

| # | setup | `ncon` |
|---|---|---|
| A | as loaded | **0** |
| B | `geom_contype[geom_group==3] = 1` after load | **0** |
| C | B **plus** `body_contype[:] = 1` after load | **7** |
| D | XML patched on disk (`galbot_selfcollide.xml`) | **7** |

**"Patching `geom_contype` after load does not work" is CORRECT (row B). The conclusion that the XML
must be patched on disk is FALSE (row C ≡ row D).** MuJoCo precomputes per-body aggregates at compile
time exactly as `galbot_preview.py:20-22` says — but those aggregates are exposed as writable
`body_contype` / `body_conaffinity`, so a two-line in-memory patch is equivalent.

Worth adopting: `galbot_preview.py`'s `ensure_patched()` (lines 68–82) **writes a file into the
vendored description package on every run**, which is a side effect a read-only checker should not have.

`grep -c contype` returns 2 in every MJCF (the two default classes only) — **no geom overrides it**, so
"every collision geom" is exact. `config/mjcf/appendix.xml` has an empty `<contact>` block, so there
are zero `<exclude>` pairs and MuJoCo's automatic filtering is the only filtering in play.

### 6.5 ⭐⭐ `mj_geomDistance` returns spurious zeros — **independently re-verified**

Found by the clearance-gate subagent, independently reproduced. The API contract is
`f(distmax) = min(true_distance, distmax)`, so `f` **must be non-decreasing in `distmax`**. It is not:

```
leg_link1_collision_1 | leg_link2_collision_0     (geom centres 223.2 mm apart)
   distmax=0.001 -> +0.00100     saturates correctly: proves true distance > 1 mm
   distmax=0.005 -> +0.00000     impossible — a decrease
   distmax=0.02  -> +0.00000
   distmax=0.1   -> +0.00000
```

The convex narrowphase collapses on mesh/mesh pairs and returns exactly `0.0`. **The breaking
threshold is pair-dependent**, and lower than one might assume — this pair breaks at 5 mm.

*(The subagent's own cited example — `leg_link1_collision_2` vs
`left_gripper_r_knuckle_link_collision_2` breaking at `distmax=0.5` — did **not** reproduce here; it
returned a sensible `0.443798`. The phenomenon is real and confirmed on other pairs; that particular
illustration is not reproducible and should not be cited.)*

**Direction of the error matters and is the reason this is survivable.** Across the subagent's
220,590 (pair, pose) evaluations the error was always a spurious `0.0` — a **false alarm**, never a
false clear. A checker that treats `0.0` as "touching" therefore fails safe.

**Defence, implemented in `teleop/clearance.py`:** a ladder of increasing `distmax`, exploiting the
contract — a rung that saturates *proves* the true distance exceeds it, so any later rung returning
less than an already-proven lower bound is rejected and the bound kept. Affected pairs are marked
`exact=False` and surfaced in `report.suspect_pairs`. A naive single query at `distmax=0.5` reports
20 of 30 test poses as "0.00 mm clearance", which is unusable.

### 6.6 Distance queries ignore `contype` entirely — **independently re-verified**

`mj_geomDistance` calls the narrowphase directly and does not consult `contype`/`conaffinity`. All
16 339 non-same-body collision-geom pairs at the home pose, shipped model vs on-disk-patched:

```
max |unpatched − patched| = 0.000e+00   (IDENTICAL)
penetrating pairs: 62 unpatched · 62 patched
```

**So a distance-based checker needs no patch at all — neither on disk nor in memory.** §6.4's
in-memory `body_contype` fix applies only to `d.contact`-based checking. `teleop/clearance.py`
consequently opens the shipped MJCF **read-only and writes nothing**, and a test asserts the model
directory's mtimes are unchanged.

**Caveat on all distances:** MuJoCo collides meshes as **convex hulls**. The collision meshes are
pre-decomposed (`leg_link1` → 3 pieces, `*_gripper_r_knuckle_link` → 8), so hull error is bounded but
nonzero. Treat these as approximations — and see §4.3 for why anything involving an end-effector link
is worse than approximate.

---

## 7 · Zero-pose offsets — a viewer artifact, not the model ⭐

**Verdict: ARTIFACT.** The values observed in URDF Studio (≈ −0.00159 on continuous wheel joints,
milliradian offsets on head/arm joints — Entry 5) are **not** properties of these files:

| check | result |
|---|---|
| `qpos0` nonzero, fixed-base | **0** (min = max = 0) |
| `qpos0` nonzero, planar-base | **0** |
| `qpos0` nonzero, wheeled | 2 — indices 2, 3 = `0.034366`, `1.0`: the **free joint's base height and quaternion w**, matching `<body name="base_link" pos="0 0 0.034366">`. All 34 hinge joints are 0. |
| `nkey` (keyframes) | **0** in all three |
| `ref=` / `springref=` in MJCF | none |
| `offset=` in URDF | **0** — all 10 `<mimic>` tags carry `multiplier` only |
| `calibration` / `<safety_controller>` | none |

URDF has no concept of a default joint position; absent any of the above, every joint is exactly 0 at
nominal zero. **A correct viewer shows 0.00000000.**

Two details pointing at the viewer's arithmetic: the offsets differed in magnitude between continuous
and limited joints — the signature of a slider deriving its quantum from the declared range, with a
substituted default span where no range exists. And "continuous wheel joints" pins down which file was
loaded: the wheeled URDF has 44 continuous joints, the fixed-base has 0.

Exact arithmetic is **UNKNOWN** (lives in the site's JavaScript, out of scope for a local audit). The
magnitude ~1.6e-3 rad ≈ 0.09° is consistent with slider quantization and is three orders of magnitude
below any joint's range.

**Do not compensate for these offsets anywhere in the teleop code. They are a display bug.**
The Entry 5 conclusion — never treat browser slider state as canonical — was right; the offsets
themselves are not real.

---

## 8 · Open / UNKNOWN

| item | status |
|---|---|
| Units of SDK `left_gripper_joint1` (`+0.0500`) vs model `left_gripper_joint` range `[0, 1.703]` rad | **UNKNOWN.** Likely metres of jaw width vs radians of drive angle. Needs a hardware sweep or a Galbot answer. |
| Mapping of SDK `chassis_joint1..4` → model `wheel1..4_joint`, and wheel index ordering | **UNKNOWN.** Nothing in either package states it. |
| Real left tool: `HUILING_GRAPPER` (pendant) or `hitbot` (package image filename)? | **UNKNOWN.** Two vendor-sourced names disagree. |
| Geometry of the real GSUCKER and HUILING_GRAPPER | **NOT IN THIS PACKAGE.** No suction xacro exists; the correct model cannot be regenerated. Tracks field-log carried-forward #15. |
| Whether URDF `effort` or MJCF `forcerange` reflects real actuator torque | **UNKNOWN.** They disagree by 2–6.7×, and 10× the other way on the grippers. |
| Whether alternate upstream package copies differ beyond TCP roll | **OPEN** — use the vendored pinned package as canonical. |

---

## 9 · Shareable-project boundary

The repository's pinned model and checked-in joint map are the source of truth
for this simulator. Hardware-specific observations and operational tooling are
intentionally kept outside the shareable project tree.
