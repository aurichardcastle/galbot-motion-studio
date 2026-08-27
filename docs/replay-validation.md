# Simulator replay validation — 2026-08-21

## Scope and decision

This is reproducible digital-twin evidence for the single-camera Motion Studio
reference. It is not physical-robot approval: this repository has no hardware
transport, installed-tool geometry, independent occupancy provider, hardware
readback, or hardware-terminating enable path.

The retained camera artifacts under `artifacts/user-trial-20260821-172220/` are
private test evidence. Do not attach them to an issue, deck, or external service
without the operator's consent.

**Decision:** the current wrist-primary configuration is a safe simulator
candidate, not release-ready. It has no whole-frame safety failure in this trial,
but it has no clearance margin, its IK accuracy distribution rides the mapper
bar, the branch-flip detector is not live-wired, and the supervisor cannot
independently recompute the declared IK residual.

## Method

- A 98.30-second webcam trial was replayed in `--analysis-sync` mode against
  its source-frame map and landmark sidecar. All figures below cover the same
  1,455 source frames, in order.
- The mapper keeps a 10 mm command-quality residual bar, inside the
  supervisor's declared 30 mm ceiling. A rejected solve is a per-limb hold;
  it never becomes a target.
- The selected candidate leaves the existing primary and position-priority DLS
  solves at damping 0.04. Only after both miss the 10 mm bar does it run a
  position-priority retry at damping 0.10. The safety supervisor, guarded path
  controller, and trajectory governor are unchanged.

## Measured replay evidence

| Configuration | Left IK holds | Right IK holds | Whole-frame result | Key result |
|---|---:|---:|---|---|
| Baseline 0.04 | 542 | 695 | 1,455 ALLOW | 1,237 local IK holds; complete-frame safety path remained active. |
| Third-retry damping 0.10 | 175 | 120 | 1,455 ALLOW | 76.2% fewer local IK holds; selected safe simulator candidate. |
| Global damping 0.10 A/B | 174 | 122 | 1,455 ALLOW | Statistically no practical improvement over the third-retry policy; primary damping is not the remaining lever. |

For the selected third-retry replay:

- Accepted target residuals: p50 5.917 mm, p95 9.390 mm, p99 9.782 mm,
  maximum 9.944 mm. The p95 is 94% of the 10 mm mapper bar; this is not robust
  accuracy headroom.
- The supervised target clearance has minimum 5.000 mm, p01 5.000 mm, and p05
  5.061 mm. The trial stayed allowed, but it rides the configured 5 mm floor
  rather than clearing it with a positive margin.
- Of the remaining 295 IK holds, 276 are 10–30 mm misses. The left arm has no
  >100 mm rejected residual; the right arm has one (198.477 mm), which remains
  a distinct outlier class and must not be hidden by a tolerance change.

### Quality-bar A/B (non-production)

The retained replay then swept explicit mapper quality bars while leaving the
supervisor's 30 mm authority, trajectory governor, and clearance floor
unchanged. Every row remained 1,455/1,455 `ALLOW` with zero clearance breach,
and the 198.477 mm right-arm outlier remained rejected.

| Mapper bar | Left/right IK holds | Max hold run (L/R) | Accepted residual p95 | Clearance p05 | Frames exactly at 5 mm |
|---|---:|---:|---:|---:|---:|
| 10 mm | 175 / 120 | 40 / 31 | 9.390 mm | 5.061 mm | 50 |
| 12 mm | 151 / 95 | 37 / 15 | 11.796 mm | 5.056 mm | 47 |
| 15 mm | 102 / 40 | 30 / 9 | 14.117 mm | 5.002 mm | 72 |
| 20 mm | 34 / 13 | 10 / 9 | 19.503 mm | 5.000 mm | 105 |

Only 20 mm meets the 10-frame (about one second) retained-run hold criterion.
It fails the paired clearance criterion: p05 clearance falls by 0.061 mm from
the 10 mm baseline and floor-riding more than doubles (50 to 105 frames). No
tested quality bar satisfies both criteria; the implementation therefore stays
at 10 mm. The 12 mm result is clearance-neutral on this clip but ineffective
(left max hold run only falls from 40 to 37 frames); it is not being rejected
as a clearance hazard. The 10-frame run criterion and the 10 mm default both
still need an operator-derived justification.

## Decisive DLS-rung telemetry

The mapper now records, on every local IK rejection, residuals from the warm,
neutral, position-priority, and conditioned-position-priority rungs. The
provenance-correct rung replay shows:

- Every one of 175 left-arm rejections exhausted all four rungs.
- 108 of 120 right-arm rejections exhausted all four; the first 12 had no warm
  seed and exhausted neutral, position-priority, and conditioned rungs.
- Final conditioned residuals were 16.553 mm median / 33.562 mm p95 on the
  left and 13.754 mm median / 29.019 mm p95 on the right. Therefore the retry
  runs and improves conditioning, but the remaining population is genuinely
  above the 10 mm mapper bar for this solver.
- The longest left continuation run (source sequences 216–297, 40 frames)
  exhausted the full ladder on every frame; its conditioned residual changed
  only from 10.183 mm at entry to 10.061 mm at exit. This rules out a stale
  warm-seed trap as the explanation for the run.

## Release blockers and next evidence

1. **No clearance margin.** The guarded controller can park exactly at the
   5 mm floor. Establish a positive-margin policy and demonstrate it on an
   adversarial clearance corpus before calling the configuration release-ready.
2. **No tested accuracy-quality policy is promotable.** The bar cuts through a
   continuous solver population, but the explicit 12/15/20 mm sweep shows that
   the only bar meeting the retained-run hold criterion also worsens already
   zero-margin clearance. Keep the 10 mm bar; do not spend the supervisor's
   30 mm ceiling as a mapper workaround.
3. **I5 branch-flip detection is partial.** `JOINT_CONTINUITY` is implemented
   in the mapper but the live pipeline passes no elapsed time, so its check is
   not reached. The governor and swept supervisor checks still mitigate emitted
   motion, but the claimed detector is not active. `docs/safety-coverage.md`
   records this honestly as `PARTIAL`.
4. **I4 residual verification is partial.** The supervisor checks the residual
   field declared by the mapper but cannot recompute the TCP error because the
   command contract omits the raw IK solution and TCP goal. Extend that contract
   and independently recompute the residual before claiming two-layer quality
   enforcement.
5. **Physical deployment remains separately blocked.** Occupancy, association,
   liveness, tool geometry, readback, hardware stop, and all physical failure
   matrix rows remain required work; see `docs/vision-readiness.md`.

## Reproduction

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q

PYTHONPATH=src .venv/bin/python -m galbot_motion_studio.cli preview \
  --video artifacts/user-trial-20260821-172220/raw.mp4 \
  --source-frame-map artifacts/user-trial-20260821-172220/raw.frame-map.json \
  --source-landmark-sidecar artifacts/user-trial-20260821-172220/raw.landmarks.json \
  --allow-legacy-source --analysis-sync --arm-mapping wrist-primary \
  --output /tmp/galbot-replay.json
```

This retained source predates v3 capture outcomes. The explicit override makes
the replay diagnostic regression work only; it is not G5/G6 or export evidence.
