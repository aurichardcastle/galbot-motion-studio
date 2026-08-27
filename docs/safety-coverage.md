# Failure-matrix coverage — offline release

This table is exhaustive against `docs/safety-model.md`. `IMPLEMENTED` means the offline
path detects the condition before another command can reach the sink and has automated
coverage. `PARTIAL` and `DEFERRED` are explicitly **not complete**; the reason and release
boundary are stated so they cannot be mistaken for safety coverage. Physical robot,
network transport, controller, tool, and dead-man claims remain outside this release.

| ID | Disposition | Evidence or explicit reason |
|---|---|---|
| P1 | IMPLEMENTED | LOST identity is rejected by `ObservationGate`; `test_unstable_observation_holds_before_the_sink`. |
| P2 | IMPLEMENTED | Loss while previewing takes the same pre-sink HOLD path as P1. |
| P3 | DEFERRED | MediaPipe Holistic is configured for one subject and cannot reliably count a second body; multi-person mode is not claimed. |
| P4 | PARTIAL | Reappearance is AMBIGUOUS, but spatial/temporal identity continuity is not implemented; live hardware is absent. |
| P5 | IMPLEMENTED | Driving landmarks and confidence are required before retargeting; observation-gate and pipeline-order tests. |
| P6 | DEFERRED | Crossed-hand handedness continuity is not implemented; v1 drives the pose-model left wrist, not hand gesture control. |
| P7 | IMPLEMENTED | Face nose/eye landmarks are mandatory; absence holds before retargeting. |
| P8 | IMPLEMENTED | Aggregate driving-landmark confidence is gated; `test_gate_order_rejects_confidence_before_retargeting`. |
| P9 | DEFERRED | Pixel-level frozen-camera detection is not implemented; advancing capture timestamps alone cannot prove changing imagery. |
| P10 | PARTIAL | Capture failure closes/releases the source and no target is emitted, but a disconnected camera cannot append a final MotionClip HOLD frame. |
| P11 | IMPLEMENTED | Strict landmark contracts reject NaN/Inf at ingress; unknown boundary exceptions become FAULT; contract and fail-closed tests. |
| P12 | IMPLEMENTED | Shoulder/eye denominators have plausibility floors; calibration and arm-retarget tests. |
| T1 | IMPLEMENTED | Inference age is checked against a same-clock deadline; observation-gate tests. |
| T2 | IMPLEMENTED | Lower sequence is rejected before retargeting; observation-gate tests. |
| T3 | IMPLEMENTED | Equal sequence is rejected by the same monotonic check as T2. |
| T4 | IMPLEMENTED | Sequence gaps are permitted when the observation is fresh; no interpolation occurs before export. |
| T5 | IMPLEMENTED | Non-monotonic target timestamps latch supervisor FAULT; safety-supervisor ordering tests. |
| T6 | DEFERRED | No network receiver or burst queue exists in the offline process. |
| T7 | DEFERRED | No heartbeat transport exists; a camera-source exception stops the offline loop without hardware motion. |
| T8 | DEFERRED | No network transport exists in this release. |
| T9 | DEFERRED | No separate vision-process heartbeat exists in this release. |
| T10 | DEFERRED | No datagram receiver exists in this release. |
| I1 | IMPLEMENTED | Head, arm, and supervisor rate gates reject teleports rather than clamp them; retarget/supervisor tests. |
| I2 | IMPLEMENTED | Calibration/source clock mismatch rejects before target creation; face/arm tests. |
| I3 | IMPLEMENTED | DLS non-convergence returns `IK_DID_NOT_CONVERGE` and pipeline HOLD. |
| I4 | PARTIAL | The mapper gates its measured IK residual and the supervisor checks the declared `RobotTarget.ik_residual_m`, but the target contract does not carry the raw IK joint solution plus TCP goal needed for the supervisor to recompute it. The declared value is therefore not independently verified. |
| I5 | IMPLEMENTED | Branch flips are detected in JOINT space against the previous accepted SOLUTION, not the commanded pose: the governor's 0.05 rad step cap hides the jump entirely (measured max |dq| in the commanded stream is 0.0558 rad, against 4.58 rad in the solution stream). A joint moving more than 0.5 rad while the wrist target moves less than 10 mm holds that limb; measured on the 2026-08-22 trial this fires on 3.95% of right-arm frames and none on the left. A rejected flip does not rebaseline the comparison. `test_left_arm_retargeting.py::test_a_branch_flip_is_rejected`. |
| I6 | IMPLEMENTED | Operator signals are clamped to a configured 2D task-space shell; camera depth remains frozen. |
| I7 | DEFERRED | Operator/physical-robot desynchronization requires hardware readback; no hardware path exists. |
| I8 | IMPLEMENTED | Torso yaw is mapped from the shoulder-line angle and governed like any other tracked joint, so the swept clearance check and the supervisor both see it move; a torso that cannot be read holds the torso group alone and leaves the arms and head tracking. `tests/unit/test_torso.py`, `test_pipeline.py`. |
| I9 | IMPLEMENTED | Arm targets are solved against the torso yaw the robot has actually realized, not the yaw just commanded, and the shoulder-relative offset is rotated by the torso frame read from forward kinematics. Measured: the TCP holds position in the torso frame to 0.000 mm across -60..+60 deg of yaw, and the transform is exactly the identity at home. |
| K1 | IMPLEMENTED | Named revolute limits are checked before approval; supervisor tests. |
| K2 | IMPLEMENTED | A 0.09 rad soft margin is enforced before approval; supervisor limit test. |
| K3 | IMPLEMENTED | Per-target rate and acceleration limits are checked; supervisor and dogfood trajectory tests. The governor's own rate/accel/step triple is additionally required to be internally consistent at construction (`rate <= 2*sqrt(accel*step)`), so a step-capped governor can no longer emit a deceleration the supervisor must reject on a sparse frame; `test_no_constructible_governor_exceeds_its_acceleration_limit`. |
| K4 | IMPLEMENTED | `ClearanceChecker` evaluates interpolated path samples; 7 Aug regression suite. |
| K5 | IMPLEMENTED | Signed distance, not contact count, gates approval; clearance suite. |
| K6 | PARTIAL | Generic-tool hash is bound for simulation, but installed-tool geometry is unknown; this is why physical arm output does not exist. |
| K7 | IMPLEMENTED | Baseline overlaps are tracked and worsening is rejected; clearance regression tests. |
| K8 | IMPLEMENTED | Exceptions, non-finite reports, and unevaluable pairs fault closed; adversarial supervisor tests. |
| K9 | IMPLEMENTED | Model/tool hash mismatch latches FAULT and rotates authority on reset; supervisor test. |
| R1 | DEFERRED | No Galbot SDK or hardware adapter exists, so `init()` cannot be called. |
| R2 | DEFERRED | No Galbot SDK getter exists in the release. |
| R3 | DEFERRED | Simulation records readback, but physical SUCCESS/following-error handling requires a future isolated hardware adapter. |
| R4 | DEFERRED | SDK TIMEOUT handling is modeled only by the fault injector; no hardware sink exists. |
| R5 | DEFERRED | SDK FAULT handling is modeled only by the fault injector; no hardware sink exists. |
| R6 | DEFERRED | Controller-persistent fault probing requires supervised hardware characterization. |
| R7 | DEFERRED | No blocking SDK call exists; process-isolated hardware worker is a future gated phase. |
| R8 | DEFERRED | Physical readback resynchronization cannot be implemented without a hardware adapter. |
| R9 | DEFERRED | Physical telemetry freeze detection cannot be implemented without telemetry. |
| R10 | DEFERRED | Physical start-envelope preflight is outside the simulator-only authority. |
| R11 | DEFERRED | E-stop/driver distinction is unavailable without a machine-readable vendor health channel. |
| O1 | DEFERRED | Physical dead-man must terminate at the HPU/controller; a Mac key cannot claim that authority. |
| O2 | DEFERRED | Physical two-signal operator enable is deferred with O1; perception loss still holds simulation. |
| O3 | DEFERRED | Wrong physical mental-model detection requires readback and explicit resync UI. |
| O4 | IMPLEMENTED | The release imports no SDK, makes no abort claim, and cleanup performs no motion; import guard plus source review. |

The deferred rows are release blockers for physical `LIVE`, not backlog that can be
silently waived. They do not block Mirror/Studio/Dataset because those modes have no
physical command path.
