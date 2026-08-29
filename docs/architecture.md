# Galbot Motion Studio — architecture

**Status:** Historical architecture context. The implemented product boundary is
simulation-only; no robot-control code is included in this repository.

## Product decision

Build **Galbot Motion Studio**: a simulation-first webcam motion-capture, robot-retargeting, sequence-authoring, and supervised teleoperation system for the G1.

The highest-value product is not merely “move the robot with a webcam.” It is one pipeline with four progressively unlocked modes:

1. **Mirror:** face, shoulders, arms, and hands animate the supplied G1 model locally.
2. **Studio:** record, trim, smooth, slow, inspect, and replay motion clips.
3. **Dataset:** export commanded and observed trajectories for imitation-learning workflows.
4. **Supervised robot:** send safety-approved intent to a G1 through a separate robot-side supervisor.

The first shippable release is Mirror + Studio + Dataset. Physical teleoperation is a later gated capability, never required for the offline product to be useful.

## Why this version is valuable

- It creates an immediately understandable demo: the digital G1 mirrors the operator.
- It replaces special capture hardware with a laptop webcam for early demonstration authoring.
- Recorded performances remain valuable even if the G1 control API is too slow for convincing live control.
- It produces a replayable motion artifact that can be validated independently.
- It produces a complete artifact without depending on another robot visit.

The internal clip format is version-neutral. Export adapters should target current LeRobotDataset v3 and Galbot-compatible v2.1; the v2.1 export is verified with `lerobot-dataset-check`.

## Scope

### Version 1

- One operator, one webcam, one tracked body.
- Head yaw/pitch.
- Left arm task-space wrist position; partial wrist orientation only after position is stable.
- Left gripper gesture mapping when the real tool interface is confirmed.
- Simulation-only omnidirectional chassis driving and five-joint torso-column crouch/lean mapping. The G1 drives on a wheeled base; it does not have a walking gait.
- Simulation, recording, editing, deterministic replay, validation, and dataset export.
- Query-only robot inspection and staged head-only/left-arm hardware trials after all gates pass.

### Explicitly deferred

- Physical chassis driving, autonomous navigation, and any claim of walking or whole-body balance. Hardware chassis control is excluded unless Galbot supplies and supervises its own safety architecture.
- Unsupervised execution.
- Multiple people or remote Internet teleoperation.
- Full finger imitation; the G1 tools are not anthropomorphic hands.
- Right suction control until tool assignment and semantics are confirmed on the target robot.
- Claims of safety certification. This is a supervised research prototype.

## Architecture

```text
Mac / development machine

webcam
  -> frame capture (monotonic timestamp)
  -> holistic landmarks (face + pose + hands)
  -> confidence, identity, latency, and occlusion gate
  -> calibrated operator-space intent
  -> retargeter (head + wrist + elbow hint + grip)
  -> model IK
  -> joint margin / speed / acceleration / step limits
  -> collision and minimum-clearance prediction
  -> digital twin + recorder + UI
  -> canonical motion clip

Optional supervised hardware mode

approved intent + sequence/session metadata
  -> wired, latest-wins transport
  -> Linux/HPU robot supervisor
  -> independent schema, freshness, sequence, rate, limit, and state gates
  -> Galbot SDK IK/collision check
  -> short-horizon nonblocking command (only if hardware measurements support it)
  -> readback + status + fault telemetry
  -> Mac recorder/UI
```

Vision output never writes directly to joint commands. The robot-side process receives intent, independently solves and validates it, and owns the only hardware adapter.

## Model provenance

The vendored description package is the canonical source package. It corrects
both gripper TCP frames from `+pi/2` to `-pi/2` roll. Reference hashes are:

- floating URDF: `8e7722191495d8b96ca8d64cc5dff918fb9cd673c74b70728641856d757e3b48`
- fixed URDF: `b340d2491fcbb53ec19bf9df3779f12b10585e7130847386f18cb6f476bbdfce`
- fixed MJCF: `6a92e1bca62507c0d5e3ffbfaddf9fff50d8e1bc2b0a3bacd2c320f36159eea7`

The package's `scripts/validate_tcp_frames.py` passes across Xacro, both URDFs, all three MJCF variants, and USD. P0 must pin or copy the exact package into a repository-controlled location, store its hashes in configuration, and fail preflight on a model/tool/TCP mismatch. Wrist orientation is disabled until this check passes. A folder name or version label alone is not sufficient provenance.

The fixed-base MJCF is the pinned arm/head preview model: it loads as 33 generalized coordinates, 23 actuators, and 268 total MuJoCo geometries. Of those, 184 are robot collision geometries with shipped `contype=0` / `conaffinity=1`, so self-collision is disabled until the clearance implementation explicitly enables and audits them. The fixed model has 23 name-addressed control joints: leg 5, head 2, left arm 7, left gripper 1, right arm 7, right gripper 1. Four fixed mount joints contain empty URDF `<axis/>` tags, which are valid XML but a strict-import compatibility risk. Neither XML traversal order nor viewer order is a valid command ordering.

## Technology choices

- **Vision:** MediaPipe Tasks `HolisticLandmarker` in live-stream mode. The current API exposes face, pose, and both-hand landmarks, including pose/hand world landmarks. Live mode may drop frames to minimize latency, which fits a latest-wins design.
- **Camera:** OpenCV `VideoCapture`; optional checkerboard/ChArUco calibration with stored reprojection error.
- **Model, FK, IK, and simulation:** the supplied Golf MJCF/URDF with MuJoCo. Implement damped least-squares task-space IK using MuJoCo Jacobians; do not depend on the Galbot SDK for offline operation.
- **UI:** local application boundary with camera overlay, human skeleton, digital twin, safety state, clearance, latency, recording timeline, and explicit arm/disarm controls. Choose the thinnest UI implementation after a one-day spike; keep the core independent of it.
- **Contracts/config:** typed, versioned data objects and checked-in YAML/TOML configuration. No magic joint ordering.
- **Transport:** latest-wins datagrams are acceptable only on the isolated wired link, with a session nonce, monotonic sequence number, monotonic capture time, strict schema/size checks, source allowlist, and receiver-side TTL. Do not expose the receiver on Wi-Fi or the Internet.

`galbot-sim` is not the preview engine or a P0–P3 dependency. Its current G1 load fails because its S1-oriented registry expects `torso_lift_joint1`, which the Golf model does not contain. A separately scoped G1 model-profile port may later become a fault-injectable `galbot_sdk` test double, but only after its API and failure fidelity are verified.

Official references checked during planning:

- MediaPipe Holistic live-stream and result APIs: <https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HolisticLandmarker>
- OpenCV camera calibration: <https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html>
- MuJoCo Python viewer: <https://mujoco.readthedocs.io/en/latest/python.html>
- MuJoCo geometry-distance API: <https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html>

## Core data contracts

Every message contains `schema_version`, `session_id`, `sequence`, `source_clock_id`, and a source monotonic timestamp. Monotonic clocks are host-local: a Mac timestamp is never subtracted from an HPU timestamp. Network age/TTL uses an HPU-stamped ingress time; ordering uses sequence numbers. Every transition into `ARMED` creates a new unpredictable `arm_generation`, and targets from earlier generations are invalid.

### `HumanObservation`

- Capture and inference timestamps.
- Normalized and world landmarks.
- Per-landmark presence/visibility and aggregate confidence.
- Tracked-person identity continuity result.
- Camera/calibration identifiers.

### `OperatorIntent`

- Neutral-calibrated head yaw/pitch.
- Left/right wrist position relative to shoulder frame.
- Wrist orientation confidence.
- Elbow-plane hint.
- Grip scalar or discrete tool request.
- Source confidence, age, and filter state.

### `RobotTarget`

- Named joint targets; never positional arrays without names.
- Desired execution time and maximum speed/acceleration.
- IK residual and solver status.
- Minimum joint-limit margin.
- Predicted minimum clearance and offending link pair.
- Provenance back to the observation/intent sequence.

### `SafetyDecision`

- `ALLOW`, `HOLD`, or `FAULT`; never silent clipping.
- All failed rules with measured values and thresholds.
- State-machine state before/after the decision.

### `MotionClip`

- Model hash, tool-description hash, calibration ID, mapping-config hash.
- Explicit left/right TCP transform hashes and frame names.
- Raw operator intent, approved targets, and (on hardware) observed joint states.
- Safety decisions, clearance, statuses, and timestamps.
- Deterministic playback rate and start-pose requirements.

## Retargeting strategy

### Calibration

1. Calibrate camera intrinsics when metric projection is required.
2. Capture a neutral operator pose for head orientation, shoulder centre, shoulder width, wrist offsets, and handedness.
3. Read the robot’s current pose as the robot neutral pose; never assume zeros or a preset.
4. Store calibration and model hashes with every clip.

Monocular “world” coordinates are treated as model estimates, not ground-truth metric depth. Version 1 normalizes motion by shoulder width and maps relative motion into a bounded robot workspace.

### Head

- Face orientation maps to `head_joint1` yaw and `head_joint2` pitch.
- Roll is discarded because the robot has no matching DOF.
- Use asymmetric pitch gains because the measured joint range is asymmetric.
- Apply deadband, One-Euro filtering, conservative soft limits, and hard rate limits.

### Left arm

- Track wrist relative to the operator’s shoulder frame.
- Begin with lateral and vertical movement only; freeze webcam depth.
- Scale into a small task-space shell around the robot’s measured starting end-effector pose.
- Solve robot IK with wrist position as the primary objective and elbow direction as a secondary redundancy objective.
- Add low-gain depth only after recorded tests demonstrate stable depth under occlusion, distance changes, and crossed arms.
- Add wrist orientation only after position tracking meets its acceptance criteria.

Human joint angles are never copied directly to robot joints.

### Hands/tools

- A pinch ratio can control a continuous gripper only after the installed gripper API/range is confirmed.
- A suction tool uses a deliberate discrete gesture with dwell and explicit UI confirmation, not a noisy continuous mapping.
- The current URDF models generic grippers on both sides and does not match the gripper and suction tools actually installed. Until the installed tool identity, TCP, mount, payload, and conservative collision envelope are physically attested, all physical arm motion—not merely tool actuation—remains disabled. A measured conservative bounding volume may temporarily substitute only if its uncertainty is added to every clearance calculation.

### Simulation locomotion

- Operator step/lean intent may map to bounded simulated chassis `x/y/yaw`; operator crouch may map to the five-joint torso column.
- Locomotion intent is recorded as a separate named channel so datasets do not imply a gait.
- Chassis and torso commands use independent confidence, deadband, rate, and workspace gates. Simulation locomotion can never unlock a hardware command path.

## Safety supervisor

### State machine

```text
DISCONNECTED -> QUERY_ONLY -> CALIBRATING -> PREVIEW
PREVIEW -> ARMED -> LIVE
LIVE -> HOLD      (authority stops new submissions; stationary hold is not assumed)
any state -> FAULT (SDK fault, invalid telemetry, limit/clearance breach, protocol violation)
HOLD -> ARMED      only by explicit operator action after a clean preflight
FAULT -> QUERY_ONLY only after human inspection and explicit reset
HOLD/FAULT -> RECOVERY only by explicit observer-approved action
RECOVERY -> QUERY_ONLY after fresh readback and waypoint-by-waypoint verification
```

`LIVE` is impossible unless the physically installed model/tool identity and geometry are attested, telemetry is fresh, the measured robot is stationary in the approved start envelope, the HPU-local dead-man and controller lease are continuously valid, the robot exposes machine-readable e-stop/driver health, and every preflight rule passes. `ARMED` means only that the observable gates passed; it is never inferred from `init()` or healthy-looking getters. A motion probe cannot substitute for a health-status channel.

### Safety layers

1. **Physical:** clear area, trained observer, physical e-stop continuously reachable.
2. **Freshness/identity:** reject missing, stale, reordered, low-confidence, occluded, or person-switched observations.
3. **Intent sanity:** reject NaN/Inf, discontinuities, impossible anthropometry, and coordinate-frame changes.
4. **Kinematics:** soft joint margins of at least 0.09 rad initially, workspace shell, IK residual threshold, named-joint validation.
5. **Dynamics:** per-step, velocity, and acceleration limits; reject teleports instead of clipping them into motion.
6. **Geometry:** collision rejection and conservative minimum-clearance threshold across the interpolated path, not just the endpoint.
7. **Operator enable:** a continuously held dead-man independent of visual gestures. A keyboard is acceptable for simulation. Physical enable must terminate at the HPU/controller safety authority through a spring-return device whose disconnect means released; the Mac UI cannot be the safety authority.
8. **Runtime verification:** every preflight path begins at a fresh, named-joint readback—not the previous command. Require stationary velocity before arming/recovery, verify following error during motion, and force explicit re-sync when target/readback or operator/robot divergence exceeds threshold. `SUCCESS` is not proof of arrival; stale/missing telemetry, `TIMEOUT`, or unexplained motion latches `FAULT` and permits no next target.
9. **Watchdog/lease:** `HOLD` immediately revokes the current arm generation and prevents new submissions, but this alone may not stop an in-flight blocking command. Physical live mode requires a native controller TTL/watchdog or independent enable with a measured maximum time and travel to stationary hold. Command horizon is shorter than the lease and queue depth is one/latest-only.
10. **Fault latch:** any `FAULT`, e-stop state, invalid readback, or unexplained timeout disables motion and requires human inspection.
11. **Process isolation:** the receiver, state authority, dead-man, and lease watchdog never make blocking SDK calls. SDK calls run in a disposable worker with a one-element generation-tagged mailbox; worker death revokes the lease. Process termination is containment, not a substitute for the native stop contract.
12. **Transport generation:** the HPU mints a new unpredictable arm generation for every arm transition, stamps ingress time locally, drains old traffic on state changes, authenticates messages, and rejects delayed, duplicate, stale, wrong-hash, or prior-generation packets.

### Two corrections to the initial draft

- **No automatic return home on heartbeat loss.** A disconnected operator must not cause new autonomous motion. Enter `HOLD`; after the robot/controller’s native hold behavior is understood, define a separately validated recovery initiated by a human.
- **No motion in generic `finally` cleanup.** Cleanup releases resources only. A return-to-home motion is allowed solely during an explicit, dead-man-enabled `DISARM` transition after a fresh path preflight. After a fault or shutdown signal, restoration may be rejected or unsafe.
- **No implied instantaneous `HOLD`.** Until the controller/SDK is measured, time and travel from dead-man release to stationary readback are unknown. If Galbot cannot provide and demonstrate a bounded native stop contract, live streaming remains disabled and only isolated, supervised one-shot characterization targets are considered.

Ctrl-C is not an emergency stop. Day 2 proved that the SDK can capture SIGINT, continue issuing blocking calls, and reject later recovery commands.

### Clearance implementation requirements

The existing checker detects contact only. The replacement must:

- perform adaptive swept-path checking with a bound on maximum geometry displacement between samples, not a fixed sparse waypoint count;
- report the closest geom/link pair and sample time;
- use a reviewed allowed-collision matrix; adjacency alone is not a blanket exclusion;
- never ignore a pair merely because it overlaps in the baseline—baseline overlaps need pair-specific allowable depth and worsening checks;
- handle the MJCF’s disabled collision configuration at model-load time;
- fail closed when a pair cannot be evaluated;
- benchmark approximation error for mesh pairs and use inflated geometry or a larger safety margin when distance is uncertain;
- define clearance reserve as tool/TCP uncertainty + FK/readback/following error + worst-case stopping travel + mesh-distance error + a fixed physical reserve;
- fail closed for any physical arm pose involving an unknown or mismatched tool link. Tool geometry is never an advisory warning in hardware mode;
- require an external keep-out zone for people, furniture, cables, and other geometry absent from the robot model.

## Recording and export

Recording stores both desired and observed motion. A hardware dataset containing only requested targets would be misleading because `SUCCESS` can still miss the requested position.

Capture stores raw camera cadence plus monotonic capture/inference timestamps. A configured `nominal_fps` is metadata, not a claim that camera frames arrive evenly. Dataset export alone resamples onto exact integer `k/fps` timestamps. For a short recorded `HOLD`, export repeats the last approved target/state while marking each generated frame as held and recording held-frame count/duration; it never interpolates across rejected or missing commands. A long or unknown-state gap ends an episode. Simulation exports must declare `source=sim` and `state_equals_action=true` so they cannot be mistaken for physical demonstrations.

The canonical `MotionClip` is the source of truth. Exporters:

- LeRobotDataset v3 for current general tooling.
- Galbot-compatible LeRobotDataset v2.1, checked with `lerobot-dataset-check`.
- Human-readable JSON/CSV summary for debugging.
- Optional future MCAP export only if matching Galbot’s schema is useful.

Replay always re-runs start-pose, dynamics, collision, clearance, and dead-man gates. A previously accepted clip is not intrinsically safe from a different starting state or with a different tool/model hash.

## Implementation phases and gates

### P0 — Discovery and scaffold (Day 1)

Deliver:

- Package skeleton, typed contracts, configuration, model loader, tests, and a null hardware adapter.
- Recorded-video input so development/tests do not require a live webcam.
- Query-only hardware questionnaire and benchmark script specification.

Exit gate:

- Model loads by repository-relative discovery, not an absolute path.
- The pinned model matches the approved hashes and its TCP-frame validation passes.
- Tests cannot import or instantiate the Galbot hardware adapter by default.
- All 23 controlled leg/head/arm/gripper joints are mapped by name and checked against the reference limits.

### P1 — Vision and calibration (Days 2–3)

Deliver:

- Webcam/video landmarks, overlays, neutral calibration, confidence/latency telemetry, and recorded observation fixtures.

Exit gate:

- Sustained >=20 processed FPS and p95 observation age <150 ms during a 10-minute run.
- Dropped, reordered, stale, low-confidence, and identity-switched frames are rejected in tests.
- No raw webcam image is persisted unless recording is explicitly enabled.

### P2 — Digital puppet (Days 4–6)

Deliver:

- Head and left-arm retargeting, MuJoCo IK, digital twin, smoothing, workspace shell, and diagnostics.

Exit gate:

- Head tracks correctly in all directions without reaching soft limits.
- Left-wrist 2D task-space tracking error <=3 cm RMS in a fixed suite of motions.
- Zero joint-limit, NaN, or solver-fallthrough outputs across at least 100,000 fuzzed observations.

### P3 — Safety and sequence studio (Days 7–10)

Deliver:

- Full state machine, interpolated clearance gate, fault injection, record/edit/replay, and canonical clip format.

Exit gate:

- Every injected dropout, teleport, NaN, stale packet, person switch, limit breach, clearance breach, and model-hash mismatch enters `HOLD` or `FAULT` before a command adapter call.
- A 30-minute simulation soak produces zero forbidden commands.
- Playback is deterministic from the same clip/config/model hashes.
- Minimum clearance is checked across the full interpolated path.
- Fault injection also covers a blocked adapter call, release after adapter entry, telemetry freeze, worker death, clamp-plus-`SUCCESS`, partial `TIMEOUT`, prior-generation packets, and late responses after disarm; no later queued command may execute.

### P4 — Dataset export and offline release (Days 11–12)

Deliver:

- v3 and v2.1 adapters, sample dataset, validator report, README, and demo video.

Exit gate:

- v2.1 output passes `lerobot-dataset-check` with no failures.
- Export preserves timestamps, targets, actuals (where present), statuses, and episode boundaries.
- A new user can run webcam -> model -> record -> replay from the README.

This is the first complete release.

### P5 — Query-only robot characterization (third visit)

Before motion, determine:

- exact SDK version/model/tool configuration;
- available nonblocking/streaming command interface and supported rate;
- controller behavior on command expiry, process death, network loss, and e-stop;
- current/fault/e-stop state query availability;
- maximum supported command rate and timeout semantics;
- installed end-effector models, mount/TCP/payload/collision envelopes, and actuation ranges;
- a machine-readable controller, driver-fault, and e-stop health channel that can be checked without commanding motion.

Only after Galbot confirms machine-readable health and the controller stop contract, benchmark latency with the smallest approved head displacement, an engineer present, and explicit recovery criteria. A deliberate motion probe is a supervised characterization step, never a substitute for health preflight. Do not run 100 movements if fewer samples establish the rate.

Exit gate:

- A written, measured command/watchdog contract exists.
- Under maximum permitted test speed, dead-man release, heartbeat loss, sender/receiver death, Ethernet pull, and SDK-worker death each produce a stationary readback within a vendor-specified and measured maximum time and joint travel that fits inside the reserved stopping distance.
- If no such native bounded-stop contract exists, `LIVE` remains disabled; the deliverable remains offline plus isolated supervised one-shot characterization.
- Any return to the measured start pose occurs only through `RECOVERY`, with fresh stationary readback, new generation/lease, dead-man, observer approval, revalidated path, and waypoint-by-waypoint actual-state confirmation.

### P6 — Staged supervised hardware pilot

Order:

1. Query-only bridge.
2. Single preflighted head target.
3. Dead-man-controlled head tracking at conservative speed.
4. Attest the installed tool/TCP/mount/payload/collision envelope against physical measurements.
5. Single preflighted left-arm task-space delta.
6. Recorded left-arm clip at reduced speed.
7. Live left-arm control only after all previous evidence is clean.

No step unlocks automatically. Each needs a human review of the preceding log.

## Test matrix

- Unit: coordinate transforms, asymmetric head map, filters, named-joint maps, limits, rate/acceleration, schema validation.
- Property/fuzz: arbitrary finite/nonfinite landmarks can never create an unchecked command.
- Golden replay: recorded motions produce stable targets within numeric tolerance.
- Geometry: known collision, known near-miss, worsening baseline overlap, adjacent-link exclusion, wrong-tool hash.
- Timing: latency spikes, clock resets, packet loss/reorder/duplication, burst traffic, frozen camera.
- Hardware protocol emulator: old arm generation, delayed high sequence, sender restart, receiver reboot, SDK worker hang/crash, frozen readback, controller clamp, partial timeout, and response after disarm.
- Perception: hands cross, operator exits, second person enters, partial body, back-facing pose, bright/dark scenes.
- State machine: every forbidden transition and reset path.
- Adapter contract: null/simulator adapters first; hardware adapter mocked until on-site.
- Soak: 30 minutes preview, 30 minutes record/replay, bounded memory and no unsafe decisions.

## Questions to send Galbot before physical control

1. Which SDK interface is intended for continuous joint/end-effector streaming, and at what rate?
2. What does the controller do when commands stop, the client dies, or Ethernet disconnects?
3. Is there a controller-side command TTL/watchdog/dead-man setting with a vendor-specified maximum stop time and travel?
4. Is there an explicit cancel/hold API independent of `request_shutdown()`?
5. Can the SDK report e-stop and latched hardware-fault state separately?
6. Can Galbot provide the URDF/MJCF for the installed gripper and suction tools?
7. May the receiver code remain on the HPU between visits?
8. Will the next robot have the same tool configuration and wired addressing?

## Definition of success

Offline success:

- The user’s head and left arm drive the model smoothly from a webcam.
- Motion can be recorded, edited, replayed deterministically, and exported.
- Adversarial inputs visibly fail closed.
- The product runs without the Galbot SDK or physical robot.

Hardware success:

- Only the intended chain moves, at measured bounded rates.
- Releasing the dead-man or losing any authority path makes the robot stationary within a stated, measured maximum time and travel under every enumerated failure mode; merely preventing new targets is insufficient.
- Every target has a matching decision, status, and readback record.
- No joint enters the soft-limit margin and no predicted path violates clearance.
- Physical e-stop behavior and recovery are demonstrated with Galbot staff before arm teleoperation.

## Immediate next step

The architecture is a reference for the simulator-only implementation. No
webcam permission, networking, or robot interaction is needed to read it.
