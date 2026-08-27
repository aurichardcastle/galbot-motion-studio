# Computer-vision / teleoperation engineering handoff — 2026-08-21

## Decision

Do not promote the current system to robot deployment or present it as
production-ready. It is a simulator-only, safety-contained reference with a
useful negative result: on the retained operator trial, reducing arm holds by
accepting less accurate IK solutions systematically consumes the clearance
margin that does not exist.

The default remains a 10 mm mapper quality bar with a damping-0.10 third retry.
That setting is a safe simulator candidate, not a release recommendation. It
completed the retained replay without a whole-frame safety failure, but it
retains local arm holds and has no positive clearance margin.

## What was implemented and verified

- Local IK non-convergence holds only the affected arm; the head and the other
  arm continue only through the existing guarded controller, governor, and
  supervisor checks.
- A third position-priority DLS retry uses damping 0.10 after the established
  primary and position-priority 0.04 attempts fail. On the same 1,455 source
  frames this reduced local IK holds from 1,237 to 295 (76.2%).
- Rejected arm maps retain their finite residual without becoming targets.
- Replays now persist each rejected solve rung—warm, neutral,
  position-priority, and conditioned-position-priority—with its residual. This
  proves that all left and nearly all right residual holds exhausted the full
  ladder; a stale warm seed is not the cause.
- The failure-matrix coverage now accurately marks two controls `PARTIAL`:
  I4 residual verification and I5 branch-flip detection.

## Retained-video evidence

All rows replay the same 1,455 source frames through the simulator safety path.
Every row produced 1,455 `ALLOW` decisions and zero clearance breaches. That is
not a physical-safety claim.

| Mapper quality bar | IK holds (L/R) | Longest hold (L/R) | Accepted residual p95 | Clearance p05 | Frames at 5 mm floor |
|---|---:|---:|---:|---:|---:|
| 10 mm (default) | 175 / 120 | 40 / 31 | 9.390 mm | 5.061 mm | 50 |
| 12 mm | 151 / 95 | 37 / 15 | 11.796 mm | 5.056 mm | 47 |
| 15 mm | 102 / 40 | 30 / 9 | 14.117 mm | 5.002 mm | 72 |
| 20 mm | 34 / 13 | 10 / 9 | 19.503 mm | 5.000 mm | 105 |

The 20 mm setting is the only measured bar that meets the provisional 10-frame
hold-run target, but it more than doubles floor-riding (50 to 105 frames) and
removes all p05 clearance headroom. It is rejected. The 12 mm setting is
clearance-neutral but ineffective on the left arm. The 198.477 mm right-arm
outlier remained rejected in every A/B.

## Release blockers, in implementation order

1. **Clearance standoff.** Establish and enforce a positive clearance margin,
   then validate it on an adversarial multi-operator corpus. The current
   guarded controller can park exactly at the 5 mm floor. Until this is fixed,
   the quality-bar decision is confounded: newly admitted IK solutions are
   enriched for torso-adjacent, floor-riding poses.
2. **I4: independently verify IK quality.** `RobotTarget` currently contains
   a mapper-declared residual scalar. The supervisor compares that scalar to
   30 mm but cannot recompute it. Extend the command contract to carry the raw
   IK solution and each arm's TCP goal in a named coordinate frame; have the
   independently loaded supervisor model recompute the residual before command
   approval. Do not claim two-layer residual enforcement until this passes
   adversarial tampering tests.
3. **I5: live branch-flip detection.** `JOINT_CONTINUITY` is tested at the
   mapper level but unreachable in the live pipeline because no elapsed time is
   supplied. Restore detection with a measured criterion that distinguishes a
   normal requested move from a redundant-IK branch flip; do not simply apply a
   raw per-frame rate cap and create normal-motion holds. Replay the retained
   trial and a dedicated branch-flip corpus afterward.
4. **Measure the two quality criteria.** The default 10 mm bar and the
   provisional 10-frame hold-run target currently lack operator-study
   derivations. Measure task error and tolerated freeze duration by task,
   robot family, operator, lighting, and camera placement. The future quality
   bar must retain a documented margin below the independently verified safety
   authority.
5. **Physical integration work.** The reference has no robot SDK transport,
   installed-tool geometry, occupancy detector, hardware readback, or
   hardware-terminating enable/dead-man. Those are separate deployment gates,
   not features that the current computer-vision code can claim.

## Recommended engineering work split

| Work package | Owner discipline | Exit evidence |
|---|---|---|
| Clearance margin/standoff | Controls + safety | Positive-margin corpus, no floor-riding regression, swept-path proof. |
| IK contract and residual recomputation | Controls + platform | Tampering rejection test; supervisor FK agrees with mapper evidence. |
| Branch-flip detection | Retargeting + controls | Live-wired elapsed/context test plus branch-flip replay corpus. |
| CV validation corpus | Perception + QA | Stratified results for occlusion, lighting, blur, camera pose, handedness, and subject variation. |
| Hardware readiness | Robotics + safety | Tool geometry, readback, stop authority, occupancy, and approved failure-matrix closure. |

## Artifacts

- [Simulator replay validation](replay-validation.md)
- [CV production-readiness boundary](vision-readiness.md)
- [Failure-matrix coverage](safety-coverage.md)

The retained video and replay JSON artifacts are private operator evidence and
must not be included in a team deck without consent.
