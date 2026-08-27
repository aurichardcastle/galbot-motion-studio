# Failure matrix — safety supervisor

**Status:** IMPLEMENTED. The supervisor lives at `src/galbot_motion_studio/safety/supervisor.py`. **Row-by-row coverage, including honest DEFERRED entries, is in `docs/safety-coverage.md` — read that for what is actually enforced.** This file is the specification it is measured against.

Every way this system can fail, what detects it, where it lands, and the test that proves it.
Derived from `docs/architecture.md` §Safety supervisor plus attacks A1–A7 in `DAY3_SHARED_CHAT.md`
Entry O-3. The implementation must satisfy every row; a row without a passing test is not done.

---

## The two latencies

Most safety writing tracks one number — "how fast do we stop." This system has **two**, and
conflating them is the mistake attack A1 identifies:

| | meaning |
|---|---|
| **t_cmd** | time from fault onset to **no new commands issued**. Bounded by our own code. |
| **t_motion** | time from fault onset to **the robot is not moving**. Bounded by `t_cmd` **plus the remaining duration of the in-flight blocking call**. |

`set_joint_positions` is blocking with a 20 s default timeout, and there is no known cancel API
(open question 4 to Galbot). So **`t_motion` is not bounded by anything we control**, and its
worst case today is `t_cmd + 20 s`. In simulation `t_motion ≈ t_cmd`, which is exactly why this
must be written down now rather than discovered on hardware.

**Rule: no row below may claim a `t_motion` bound until it has been measured on hardware.**
Until then the honest value is `t_cmd + UNKNOWN`, and the plan must say so.

`HOLD` = no new targets, in-flight command runs to completion or timeout.
`FAULT` = latched, requires human inspection and explicit reset.

---

## 1 · Perception

| # | Failure | Detection | → | Notes |
|---|---|---|---|---|
| P1 | No person in frame | landmark count / presence below threshold | HOLD | Normal, not exceptional. Must be quiet in the UI or operators learn to ignore alerts. |
| P2 | Person leaves mid-motion | presence lost while `LIVE` | HOLD | Must not complete the in-flight target's *successors*. |
| P3 | Second person enters | >1 detected body | HOLD | Never silently pick one. |
| P4 | **Identity switch** | tracked-body continuity check across frames | HOLD | The dangerous one: landmarks stay valid and plausible while belonging to a different human. Continuity must be positional + temporal, not "a person is present". |
| P5 | Partial body / occluded arm | per-landmark visibility below threshold on any driving landmark | HOLD | Wrist occluded is the common case; shoulder occluded destroys the normalisation frame. |
| P6 | **Hands cross / left-right swap** | handedness assignment flips between frames | HOLD | Produces a large, *smooth*, plausible target jump — the rate limiter may not catch it because it is not a teleport, it is a mirror. Needs its own check. |
| P7 | Back-facing pose | face landmarks absent while pose present | HOLD | Head mapping is meaningless; arm mapping inverts. |
| P8 | Low light / overexposure | aggregate confidence below threshold | HOLD | |
| P9 | **Frozen camera** | identical consecutive frames, or capture timestamp not advancing | HOLD | Confidence stays *high* on a frozen frame. A confidence gate alone cannot see this. |
| P10 | Camera disconnect | capture error / no frame within timeout | HOLD | |
| P11 | NaN / Inf landmark | non-finite check on every value before use | HOLD | Must be checked at the boundary, never propagated. |
| P12 | Impossible anthropometry | shoulder width ≈ 0, limb length outside human range, ratios inconsistent | HOLD | Shoulder width is the normalisation denominator — near-zero produces unbounded targets from tiny motions. This is a division-by-near-zero with a physical consequence. |

## 2 · Timing and transport

| # | Failure | Detection | → | Notes |
|---|---|---|---|---|
| T1 | Stale frame | `now − capture_ts` > max age | HOLD | Age measured from **capture**, not receipt. |
| T2 | Reordered packets | monotonic sequence number | drop | Latest-wins: a lower sequence than the last accepted is discarded, not applied. |
| T3 | Duplicate packets | sequence already seen | drop | |
| T4 | Gap / packet loss | sequence discontinuity | continue if fresh | A gap is not itself unsafe; a *stale* frame is. Do not conflate. |
| T5 | **Clock reset / non-monotonic timestamp** | monotonic source only; reject regression | FAULT | A clock going backwards makes every freshness check meaningless — this is a protocol violation, not a hiccup. |
| T6 | Burst traffic | rate cap on accepted frames | drop excess | |
| T7 | Missed heartbeat | no heartbeat within threshold | HOLD | **Never auto-return-home.** Disconnection must not create new autonomous motion (independent review, Entry 3 — accepted). |
| T8 | Network partition | heartbeat loss + transport error | HOLD | Same as T7; no distinct handling, deliberately. |
| T9 | Vision process death | heartbeat loss | HOLD | Indistinguishable from T7/T8 at the receiver, and must be. |
| T10 | Malformed / oversize datagram | strict schema + size check + source allowlist | drop, log | Repeated occurrences → FAULT (probe or corruption). |

## 3 · Intent and retargeting

| # | Failure | Detection | → | Notes |
|---|---|---|---|---|
| I1 | Teleport | implied velocity > cap | **drop and hold** | Drop, never clamp. A clamped teleport is still a full-speed move in a direction nobody asked for. |
| I2 | Reference-frame discontinuity | calibration frame changed mid-session | HOLD | |
| I3 | IK no solution | solver status | hold last valid | Never fall through to "send it anyway". |
| I4 | IK residual above threshold | residual norm | hold last valid | A converged-but-poor solution is a silently wrong pose. |
| I5 | **IK branch flip (elbow flip)** | joint-space distance between consecutive solutions >> task-space distance | **drop and hold** | The nastiest one here. Task-space target moves 1 cm; the solver picks a different redundancy branch; the arm swings through a large joint-space arc to reach a nearly identical hand position. Every task-space check passes. Only a joint-space continuity check catches it. |
| I6 | Target outside workspace shell | shell test before IK | clamp to shell | Clamping is correct here — the shell is a design envelope, not a fault. |
| I7 | **Operator/robot desync** | divergence between operator-implied pose and last readback > threshold | HOLD + require explicit re-sync | Attack A7. After repeated I1 rejections the robot is somewhere the operator does not believe it is. Harmless until they command a large deliberate motion from a wrong mental model. |
| I8 | Torso basis lost | shoulder line missing, below the minimum width, or non-finite | **hold the torso only** | The shoulder line is the torso's whole input. A collapsed line reports a wildly wrong angle rather than no angle, which is why the width guard exists and why this is a hold rather than a clamp. Holding the torso alone is correct: an occluded wrist says nothing about which way the chest is facing. |
| I9 | **Torso/arm frame desync** | arm target solved against a torso yaw the robot is not at | **must be structurally impossible** | The arms hang off the torso, so a stale torso angle aims them at a chest orientation that no longer exists -- silently, because every task-space check still passes. Not detectable after the fact: the only defence is that the arm's target frame is derived from the realized torso pose in the same forward-kinematics evaluation. |

## 4 · Kinematics and geometry

| # | Failure | Detection | → | Notes |
|---|---|---|---|---|
| K1 | Joint limit breach | limits from `G1_JOINT_LIMITS.md`, cross-checked against MJCF `ctrlrange` | reject | |
| K2 | Soft margin breach | ≥0.09 rad (5°) inside the hard limit | reject | `left_arm_joint4` faulted into `DriverError` at **2.1° from its stop** under gravity load. The margin must exceed where the drive gives up, not merely the limit. |
| K3 | Velocity / acceleration breach | per-step and per-second caps | reject | |
| K4 | **Self-collision on the path** | minimum distance over **every interpolated sample** | reject | The 7 Aug collision had a clear start pose *and* a clear end pose. Endpoint checking would have passed it. |
| K5 | Clearance below floor | conservative minimum distance vs configured floor | reject | Distance, never contact count. A 2 mm miss and a 0.5 m miss both report "0 collisions". |
| K6 | **Closest pair involves a tool link** | link-name classification | **reject outright** | Attack A5. The model ships a generic gripper on *both* arms; the real robot has a HUILING gripper left and a suction cup right. Tool-link distances are computed against geometry that is not on the robot. "Advisory" is not enough — advisory warnings in a real-time loop are read by nobody. |
| K7 | Baseline pair worsening | per-pair allowable depth + worsening check | reject | Structural overlaps (head/torso, leg/chassis, leg/wheels) may not be whitelisted forever. |
| K8 | Unevaluable pair | any evaluation error | **reject (fail closed)** | |
| K9 | Model / tool hash mismatch | hash compared at ARM and at replay | FAULT | A matching hash proves consistency, not correctness — see K6. |

## 5 · Robot and adapter

| # | Failure | Detection | → | Notes |
|---|---|---|---|---|
| R1 | `init()` returns False | return value checked — **never assumed** | FAULT | On 7 Aug a script printed `init OK` unconditionally and the "successful init then segfault" finding was an artifact of that print. Check the value. |
| R2 | Getter after failed init | guard in the adapter | FAULT | Unguarded on real hardware: segfaults the process, no Python exception. |
| R3 | **`SUCCESS` without reaching** | compare readback against commanded every time | reject next target until reconciled | Measured: SUCCESS returned 0.0062 rad short. Tolerance lies in (0.0062, 0.0359) rad and is **not** known more precisely. |
| R4 | `TIMEOUT` (partial motion) | status | HOLD | Six of seven joints still reached home in the observed case — partial, not zero. |
| R5 | Bare `FAULT` | status | FAULT | Zero motion. Distinct from R4. |
| R6 | **Latched `FAULT` surviving restart** | probe command at ARM; tail `~/galbot_sdk_log/g1/` | FAULT | Attack A3. The latch lives in the controller. `init()` returns **True**, every getter returns correct live values, and only a motion command reveals it. `ARMED` cannot be asserted without a probe. |
| R7 | Blocking call stalls | wall-clock watchdog around the call | FAULT | Detects the A1 gap. Cannot shorten it — only observe it. |
| R8 | Readback drift from commanded | ‖readback − commanded‖ > threshold | force re-sync | Attack A2: the next path must be validated from **readback**, never from the last command, or every R3 silently invalidates the next clearance check. |
| R9 | **Readback frozen** | identical readback across N commands with nonzero commanded delta | FAULT | Dead telemetry looks exactly like a perfectly still robot. |
| R10 | Start pose outside approved envelope | preflight | refuse ARM | |
| R11 | E-stop engaged | `~/galbot_sdk_log/g1/` code 70 `DriverEstop` vs 69 `DriverError` | FAULT, distinguish in UI | The API collapses both to `FAULT`. The distinction exists only in the log file. An operator should not have to read a log to learn a button is pressed. |

## 6 · Operator

| # | Failure | Detection | → | Notes |
|---|---|---|---|---|
| O1 | Dead-man released | key/pedal state, continuously sampled | HOLD | `t_cmd` = heartbeat threshold. `t_motion` = that **plus in-flight remainder** (A1). |
| O2 | **Dead-man held but operator absent** | dead-man held while perception reports no person, beyond a short grace | HOLD | A taped-down key is a defeated interlock. Two independent signals must agree that a human is present and engaged. |
| O3 | Operator commands from a wrong mental model | see I7 | HOLD + re-sync | |
| O4 | **Ctrl-C** | — | **not an abort path** | The SDK installs its own SIGINT handler. On 7 Aug `KeyboardInterrupt` appeared **zero** times in a 3,060-line transcript while motion continued ~4 minutes after the operator aborted. Every script that evening printed "ctrl-C to abort" and every one was lying. **No UI affordance may imply Ctrl-C stops anything.** |

---

## Test obligations

Each row needs a test that fires the failure and asserts the transition **before any command reaches
the adapter**. Beyond per-row tests:

1. **Fuzz.** ≥100,000 arbitrary finite and non-finite observations produce zero unchecked commands.
2. **Ordering.** The gate order is itself a property: a NaN must be rejected before it reaches IK; a
   teleport before clearance; clearance before the adapter. Assert the order, not just the outcome.
3. **Soak.** 30 minutes of continuous simulated teleop, zero forbidden commands, bounded memory.
4. **Determinism.** Same clip + config + model hash → byte-identical target sequence.
5. **Fail-closed by construction.** A new failure mode nobody enumerated should land in HOLD by
   default. Test this by injecting an unknown status code and an unknown exception type — the
   supervisor must not treat "I do not recognise this" as "proceed."
6. **Latency.** Measure and record `t_cmd` for every HOLD trigger. Record `t_motion` as
   `t_cmd + UNKNOWN` until hardware measurement exists.

## Known gaps

- **`t_motion` is unmeasured and currently unbounded.** Everything else is architecture around this
  one number. It is the first thing to measure if hardware time ever happens.
- **Tool geometry is wrong on both arms**, so K5/K6 margins are optimistic at the fingertips by an
  unknown amount. Mitigated by K6's outright rejection, not solved. Solved only by Galbot supplying
  the real end-effector descriptions.
- **Controller behaviour on command expiry, client death and network loss is unknown** — questions
  1–5 to Galbot. Until answered, T7/T8/T9 assume the worst: that the robot completes whatever it was
  last told to do.
