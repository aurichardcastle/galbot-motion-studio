# Computer-vision production readiness — engineering handoff

**Decision status:** proposed v1 contract and implementation gates, peer-reviewed against the current Motion Studio reference. It is **not** approval to command a physical robot.

**Reference boundary:** the current prototype is a simulator-only, single-webcam Motion Studio pipeline. It has no robot SDK import, transport, or hardware command adapter. It is evidence for offline perception and digital-twin testing only; it is not a production robot-control path.

## This implementation update

The simulator reference now has a small, test-covered CV safety kernel. This is
useful implementation evidence for the packages below; it is **not** a physical
release approval or a substitute for the required measurement corpus.

| Implemented in the reference | Intentionally not claimed complete |
|---|---|
| Capture-clock regression is rejected before MediaPipe timestamp quantisation. The CLI latches a perception FAULT rather than crashing, and a regressed terminal fault is preserved verbatim in clip/export provenance. | Camera disconnect/reopen control flow is not yet wired to a concrete source lifecycle; hardware stop response remains robot-side. |
| V1 arm mapping always consumes shoulder-relative normalized offsets. Unvalidated pose-world coordinates no longer silently become metres. | Metric mapping remains blocked on the pre-registered accuracy study. |
| Live webcam preview requires five explicit, deployment-owned neutral-window values. It accumulates only consecutive eligible frames in a sliding time window and commits only after source/camera/order/duration/variance checks and per-sample arm checks pass. The clip/export retain both the complete policy and the accepted window's sample count, span, and first/last sequence. | The values are deliberately not defaults: they cannot be approved until the pre-registered camera-placement and operator-motion corpus exists. Offline/synthetic analysis retains an explicitly documented single-frame legacy API. |
| A pure acquisition/continuity kernel now gates both processing and calibration when configured. It requires exact same-frame occupancy evidence, enforces configured normalized operator-zone bounds, keeps missing evidence in a bounded per-gap **and cumulative-generation** continuity budget, and ties calibration to a non-reusable selection generation. Position is checked against the last trusted frame, while shoulder width and eye span remain anchored to acquisition so gradual substitution cannot ratchet through. A shared capture-frame fingerprint helper prevents the landmark and occupancy providers from quietly using different association digests. Every clip/export also attests explicitly that independent occupancy evidence is disabled. | There is no independent occupancy detector, provider lifecycle, selection telemetry export, or live CLI wiring yet; no multi-person claim is made. An enabled attestation requires provider, distinct occupancy/landmark model hashes, zone, association-contract, and maximum-age evidence. |
| Optional frame-liveness is fail-closed, verifies content transition rather than timestamp cadence, preserves ordering across supervisor recovery, requires an explicit budget for webcam preview, and writes its enabled/disabled policy into clips and exports. | The fingerprint is currently SHA-256 over raw BGR pixels. Its false-positive/false-negative behaviour still needs the real-camera still-operator and driver-buffer corpus before a budget can be approved. |

## Executive decision

Build a platform-neutral **perception-to-intent service**. It emits a versioned, quality-gated human-intent envelope; it never emits robot joint names or authorizes motion. Each robot consumes that envelope only through its own independently reviewed capability profile, safety supervisor, and hardware adapter.

V1 is deliberately narrow:

1. One explicitly acquired operator in one configured camera zone. This is **not** multi-person control and does not claim biometric identity or tracking identity.
2. An independent occupancy detector reports the number of people in the operator zone for the same capture instant as landmarks. Zero, more than one, detector/landmarker disagreement, stale occupancy, or a failed association is a HOLD. The result is a session-scoped selection ID, renewed on every acquire; it is not a person track ID.
3. The v1 mapping contract is **shoulder-normalized absolute-offset mapping with profile-owned gains**. Perception emits dimensionless, shoulder-relative offsets and calibration scale basis; robot-specific profile gains create metres. It is neither metric operator-position control nor velocity/rate control.
4. Arms, head, hands, and tool/gripper degrees of freedom are separate intent groups. Any tool DOF that can affect clearance participates in the same feasibility/backoff decision as limb DOF.
5. One camera is in v1. Stereo, depth sensing, cross-camera fusion, and persistent multi-person tracking are future work, not implied requirements.

No v1 ticket may quietly widen these decisions. A task requiring reliable metric depth, wrist orientation, more than one camera, or simultaneous people is a separate scope decision.

## Current evidence and gaps

| Area | Reference evidence | Production decision / gap |
|---|---|---|
| Subject continuity | MediaPipe Holistic still produces one pose landmark list. The reference has an optional independent occupancy-provider boundary plus an explicit acquisition/continuity manager, but ships no provider implementation; the MediaPipe adapter cannot itself emit a validated selection. | Integrate and measure a separately evaluated occupancy detector and live provider lifecycle. Do not build a tracker-shaped API until a detector + tracker + body/hand association stack is selected and measured. |
| Calibration | Live webcam preview refuses to start without all five measured neutral-window limits; it clears the accumulator after an ineligible observation, slides over aged eligible samples, rejects a moving/non-contiguous window without mutating calibration, and records both policy and achieved evidence in the clip/export. When selection is configured, archived window samples are validated without mutating live gate state: they must cite the acquisition camera/clock, exact occupancy frame, zone, identity, and selection continuity, but are not retrospectively rejected by the live frame-age budget merely for belonging to a longer approved window. The calibration-window policy alone owns accumulated **landmark shoulder-centre** motion; selection retains its acquisition-anchored anthropometry check. Live selection still enforces candidate-centre continuity per frame; occupancy-centre jitter during archived calibration is a detector-quality/corpus case, not a hidden extra calibration threshold. Calibration neither resets an established liveness session nor reuses a selection generation. | The live service must prove liveness while each calibration sample is collected; the current reference has no enabled occupancy provider/CLI path that supplies that per-sample ledger. Bind calibration to mirror/model transform, expiry, and the remaining explicit invalidators. Approve actual limits only from the required corpus. |
| Coordinates | The adapter may carry body-centred pose-world landmarks as diagnostic evidence, but the v1 arm retargeter now always selects normalized, shoulder-relative offsets. Palm orientation and neutral calibration remain normalized. | Keep metric operator-position control blocked until a pre-registered per-axis study passes. Never combine body-centred and hand-centred coordinates. |
| Runtime | Capture is latest-wins; Holistic is currently synchronous VIDEO mode. A difficult prototype run was about 8–10 fps, including control/safety work. | Benchmark perception separately from mapping and safety. Live inference needs bounded asynchronous handoff, a submitted/returned timestamp ledger, and per-compute-profile SLOs. |
| Image liveness | An optional content-fingerprint liveness monitor is after freshness/order/future checks and holds the whole frame until a content transition is observed; live webcam preview requires an explicit detection-time budget. Its policy (including an explicit disabled state) is retained in clip/export provenance. | Validate the current raw-pixel SHA-256 fingerprint and its configured detection time against still-operator, repeated-driver-buffer, disconnect/reopen, and real sensor-noise corpora. |
| Safety groups | Simulator analysis found a generic-gripper-only clearance breach while arms were stationary: the gripper was rate-governed but excluded from feasibility easing. | A profile must declare every tool DOF and its geometry. Feasibility/backoff covers all enabled collision-affecting groups, not arms only. This is a simulator finding, not a physical-robot clearance claim. |
| Portability | The reference tree contains robot/model-specific retargeting and simulator safety code, while the detector/contract boundary can be separated. | Vision, generic intent contracts, and detector adapters import no robot-specific model, joint, or SDK. Robot integration occurs only behind a capability profile and adapter. |

## The v1 contract

    Frame
      ├─> Occupancy detector ──> Occupancy observation
      └─> Landmark estimator ─> Landmark estimates
                     \             /
                      Acquisition / association gate
                                │
                      Calibrated OperatorIntent
                                │
                      Robot-specific capability mapper
                                │
                     Independent safety / adapter / readback

Every edge retains the original capture time. A detector result is never used to gate a landmark frame from a different instant beyond the configured staleness limit.

| Envelope | Minimum fields | Fail-closed rule |
|---|---|---|
| Frame | camera ID, source-clock ID, capture monotonic timestamp, sequence, image/source fingerprint, capture status | Missing or non-monotonic provenance is invalid at CV ingress: report CLOCK_REGRESSION and HOLD before any detector timestamp clamp. Persist image content only with explicit consent. |
| OccupancyObservation | frame reference, candidate count **inside the configured operator zone**, candidate geometry/quality, detector model hash, age | 0 or >1 candidates, stale evidence, detector error, or disagreement with landmarks is HOLD and a counted health event. |
| Selection | session-scoped selection ID, state (ACQUIRING, LOCKED, OCCLUDED, LOST), state-start time, acquisition generation, rejection reason | Occupancy ambiguity is an explicit `OCCUPANCY_AMBIGUOUS` rejection and transitions the current reference kernel to LOST. Selection ID changes on reacquire; it resets filters and invalidates calibration. `SWITCHED` is not an operator-selection state. |
| LandmarkEstimate | name, frame reference, coordinate-space tag, visibility/presence, age, provenance (pose, hand, fused), corpus-derived error bound | An estimate without valid provenance, age, or association cannot drive its intent group. Do not invent a covariance the detector does not produce. |
| CalibrationArtifact | selection generation, camera/mirror transform, neutral-window statistics, normalized scale basis, detector/model hashes, creation/expiry, invalidators, content hash | Any selection, camera, model, transform, quality, or expiry mismatch invalidates it. |
| OperatorIntent | envelope/model/calibration hashes; per-group target, validity, age, error bound, HOLD reason; head, left/right arm, left/right hand, and tool intent as first-class groups | No invalid group gets a new command. A whole-frame selection/timing failure holds all groups. |
| CapabilityProfile | profile/model/tool/geometry hashes; enabled groups; mapping identifier and gains; workspace/limit/rate/acceleration/step policy; feasibility groups; hardware-readback attestation | Missing, mismatched, or unattested evidence denies adaptation. Only the robot side has command authority. |

### Operator acquisition and association

The occupancy detector is a safety input, so its own failures need a dedicated failure-matrix row. The operator zone must be a deployment artifact—camera region, minimum image area, and, if available, depth band—not an unbounded view of the room. This prevents routine background bystanders from producing a gate that is later disabled in the field.

- Occupancy and landmark output must cite the same Frame, or an explicit measured maximum timestamp skew.
- Disagreement (occupancy=1 but no usable pose, occupancy=0 while landmarks coast, or incongruent geometry) is OCCUPANCY_LANDMARK_DISAGREEMENT and whole-frame HOLD.
- Any selection loss/occlusion is a whole-frame HOLD; only a limb-specific landmark loss may hold its affected group while selection remains sound. A 1→1 person substitution is not evidence of continuity: re-lock after any gap requires a positional plus anthropometric continuity gate, using timestamped zone geometry, shoulder width, eye span, and orientation against the locked artifact. Failure is LOST, not continued LOCKED.
- Selection grace is distinct from the five-frame landmark-flicker grace. Its maximum is bounded both by measured loss-event ergonomics **and** by a Controls/Safety-owned physically feasible substitution bound. Its positional/anthropometric tolerances, maximum duration, and false-reject rate are configuration items. During grace, no absent or ambiguous body part is synthesized.
- Reacquisition creates a new selection generation and requires a new calibration. Position is compared to the immediately prior trusted frame so a moving operator remains usable; shoulder width and eye span remain compared to acquisition for the full generation, so many small legal deltas cannot ratchet a different person through. Both a per-gap and cumulative-per-generation occlusion budget are configuration items. An untrusted base frame (including frozen/liveness failure) ends the selection rather than preserving authorization through recovery. An external, hardware-terminating operator-enable/dead-man remains the robot safety owner's responsibility; CV emits fresh provenance and state.

### Coordinate and hand-association policy

For v1, perception outputs signed, shoulder-relative normalized offsets and the neutral scale basis (for example shoulder width and eye span). Profile-owned gains and envelopes turn those values into robot-space targets. Units are tagged; no consumer may interpret normalized offsets as metres.

Pose-world landmarks can be measured in Package 0, but may not become a metric control source unless a pre-registered per-axis accuracy/error threshold is met on the intended camera, compute, range, clothing, and lighting conditions. The study states pass/fail before data collection; an inconclusive or failed depth axis is an acceptable outcome.

Hand-to-body association is safety critical. Crossed hands, a left/right swap, mirror disagreement, hand-to-wrist association outside tolerance, or ambiguous handedness produces HAND_ASSOCIATION_AMBIGUOUS; the associated hand/tool group is invalid and cannot reach the mapper. Smoothness or rate limiting is not a substitute for this check.

## Evaluation, replay, and observability

There are two different reproducibility gates:

| Gate | Required assertion |
|---|---|
| Offline replay | Same recorded inputs and pinned model hash, MediaPipe version, delegate, thread count, running mode, and calibration produce byte-identical canonical **intent** records. Pin the capability profile additionally only for mapped-command replay. Replay uses VIDEO or IMAGE mode—never LIVE_STREAM, which may discard input by design. |
| Live operation | Decision class and stable reason enum match the reference policy; landmark/intent drift, capture-to-intent latency, and recovery stay within approved per-profile budgets. Formatted diagnostic strings are not equivalence keys. |

Instrument the following timestamps and ledgers for every run:

- capture → capture/queue handoff: submitted, accepted, and dropped-before-processing;
- detector submission → detector return: a bounded ledger reconciles submitted frame timestamps with returned timestamps, because live MediaPipe has no dropped-frame callback;
- intent → adapter: offered, rejected, accepted, expired, and supervisor verdict;
- capture → intent latency (operator-facing) and inference-only latency (CV-owned), separately from mapping, safety, rendering, and recording time; and
- selection candidate count, detector/landmarker disagreement, per-group validity and age, calibration/model/profile hashes, rejection reason enum, and affected limb/tool group.

Traces are privacy scoped. The existing explicit recording enable for persisted images is a useful starting point; retention period, access, deletion, and consent must be policy artifacts, not debug defaults.

## Required corpus and acceptance tests

The corpus pairs raw input with expected state, reason enum, group validity, and timing outcome. It includes at least:

- second person enters/leaves the operator zone; two people exchange positions; a 1→1 substitution during selection grace; occupancy misses a partly occluded entrant; occupancy/landmarker disagreement;
- hands crossing, left/right swap, hand near face/torso, hand-only visibility, body occlusion, back/side pose, mirror-policy error, low light, and motion blur;
- still operator, frozen image, repeated driver-buffer frame while capture timestamps advance, camera unplug/replug, sequence regression, and clock reset;
- sustained overload, detector drops and recovery, stale callback, selection loss/reacquisition, poor or near-zero shoulder-width neutral pose, and moved camera;
- calibration expiry/model/profile/hash mismatch and every invalidating event; and
- a per-profile tool-DOF feasibility case proving an arm-stationary clear path cannot be approved if a tool/gripper-only step breaches clearance policy.

The existing failure matrix remains the source of truth. Every corpus case records the applicable detection time and the maximum time from fault onset to no new intent (t_cmd); false-positive rates alone never pass a fault gate. Before implementation, add explicit coverage and owner fields without creating duplicate authorities:

| Matrix concern | CV responsibility | Downstream authority |
|---|---|---|
| P3/P4: additional person and re-entry | Occupancy, same-frame association, selection generation, reacquire evidence | Safety supervisor holds; robot adapter cannot override. |
| **New occupancy-model row** | Miss/false detection, stale count, zone error, and detector/landmarker disagreement are detectable, counted, and HOLD. | Controls/Safety owns acceptance of the measured false-HOLD budget. |
| P6: crossed hands / left-right swap | Detect association ambiguity and invalidate the affected hand/tool group before mapping. | Mapper/supervisor rejects invalid intent. |
| P9/P10: frozen/disconnected camera | Liveness, source health, provenance, and configured detection time. | Supervisor/adapter owns physical stop behavior and the physical t_cmd budget. |
| P12: anthropometry denominator | CV owns unit-tagged shoulder/eye denominator validation before normalized offset is emitted. | Mapper/supervisor may also reject the resulting invalid intent; it is not the only guard. |
| T5: clock reset | CV detects capture-clock regression before detector timestamp normalization, emits reason + HOLD. | Supervisor remains authoritative for its downstream FAULT response. |
| I4: IK residual quality | CV/mapper must attach the raw IK joint solution and per-arm TCP goal as auditable evidence. | The current supervisor only checks the declared residual scalar; it cannot recompute it. Contract extension and independent recomputation are required before a two-layer quality claim. |
| I5: IK branch flip | No duplicated CV IK check. | The mapper continuity guard exists but is currently unreachable from the live pipeline because it receives no elapsed time; the governor and supervisor remain active mitigations, not branch-flip detection. Restore and replay-validate detection before any production claim. |
| I7, K6, O1/O2 | Intent provenance, freshness, and group validity only. | Hardware readback, installed-tool geometry, and hardware-terminating enable/dead-man are robot-side obligations. |

## Configuration that must be owned before implementation

These are deployment-profile values, not magic constants to embed in source. A named individual must be assigned for each before its ticket is accepted. Items marked *measured/policy* cannot be selected at a whiteboard.

| Parameter | Accountable discipline |
|---|---|
| Operator-zone geometry: normalized bounds/region, minimum bbox fraction, optional depth band | Perception + Operations |
| Occupancy/landmark maximum timestamp skew | Perception |
| Bystander false-HOLD budget (holds/min on a real corpus) *measured* | Controls/Safety |
| Occupancy/landmark disagreement budget (events/min) | Perception |
| Selection per-gap grace and cumulative-per-generation occlusion budget, distinct from landmark-flicker grace *measured* | Controls/Safety |
| Selection re-lock positional/anthropometric tolerances and false-reject rate *measured* | Perception + Controls/Safety |
| Calibration window, maximum variance, and maximum motion | Perception |
| Calibration expiry and invalidating events | Controls/Safety |
| Inference-only and capture-to-intent p95/p99 budgets per compute profile | Perception for inference; Controls for total envelope |
| Maximum consecutive detector drops and recovery time | Perception |
| Maximum intent age accepted at adapter | Controls/Safety |
| Liveness metric, threshold, and still-operator false-positive rate *measured* | Perception |
| Maximum CV detection time and t_cmd budget for freeze, disconnect, occupancy loss, detector stall, clock regression, and selection discontinuity | Controls/Safety (CV implements detection) |
| Replay pin set and live numeric drift bound | Perception |
| Per-axis world-landmark error threshold that would permit metric mapping *measured* | Perception + Controls |
| Enabled feasibility/backoff DOF, including tools, per profile | Robot integrator + Controls |
| Trace/video retention and consent path *policy* | Product/Legal |

## Work packages and non-negotiable exits

| Package | Deliverable | Exit evidence |
|---|---|---|
| 0. Baseline | Target-camera/compute benchmark; world-landmark measurement; named configuration owners | Pre-registered pass/fail criteria, corpus, separate CV and mapping/safety latency distributions, and a decision whether metric control is out of scope. Benchmark the intended async configuration; if a synchronous baseline is used, its SLOs are provisional and re-measured in Package 3. |
| 1. Contracts | Versioned envelopes, stable reason enums, unit-tagged denominator limits, and ingress clock-regression HOLD | Compatibility/replay tests; generic CV layers contain no robot/model/SDK import; no silent detector timestamp rewrite. |
| 2. Acquisition and calibration | Independent occupancy + selection manager; windowed calibration and re-lock continuity gate | All multi-person, 1→1 substitution, disagreement, zone, loss/reacquire, and calibration-invalidating corpus cases hold correctly. |
| 3. Robust perception | Async latest-wins pipeline, liveness, hand association, telemetry ledgers | Drop/recovery, false-positive, and detection/t_cmd budgets met; every decision reconstructible by frame/selection/calibration/profile hash. |
| 4. Robot profiles | Profile schema and conformance fixtures | A new mock robot integrates without a perception-core edit; profile mismatch fails closed; tools participate in feasibility. |
| 5. Controlled characterization | Instrumented human study and robot-specific digital-twin review | Published trace/error/availability/latency report and signed Safety/Controls decision before isolated hardware characterization. |

## Presentation close: what is ready and what is not

The team can begin Package 0 and contract work immediately with a precise v1 boundary, test corpus, telemetry shape, and ownership model. The simulator is valuable regression evidence—especially the tool-only clearance mechanism—but it does not establish physical clearance, hardware stopping, installed tool geometry, or metric depth accuracy. Those remain deliberately gated.

## References

- [MediaPipe Holistic Landmarker options](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HolisticLandmarkerOptions)
- [MediaPipe Holistic Landmarker live-stream API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HolisticLandmarker)
- [Current failure matrix](safety-model.md)
- [Project plan and physical-safety boundary](architecture.md)
