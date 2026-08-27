# Export spec — MotionClip → LeRobot v2.1

**Status:** IMPLEMENTED as `src/galbot_motion_studio/export_v21.py`. This file is the *contract*, kept because it derives every rule from `lerobot-dataset-check`; it is not a to-do list. Verified 2026-08-10: a demo export passes `10 PASS, 0 WARN, 0 FAIL`.

The contract is derived line-by-line from an independently maintained dataset
validator. Its six checks are documented below so this repository can explain
and test its writer without depending on a private companion checkout.

---

## 1 · The six checks, as hard requirements

| # | Check | Requirement on our writer | Source |
|---|---|---|---|
| C1 | `check_stats_bijection` | `episodes_stats.jsonl` has **exactly one** entry per episode — no duplicate `episode_index`, none missing, none extra | `checks.py:14` |
| C2 | `check_task_integrity` | `task_index` unique in `tasks.jsonl`; **every** `task_index` in parquet resolves | `checks.py:49` |
| C3 | `check_parquet_layout` | Each `episode_NNNNNN.parquet` has a single `episode_index` value matching its filename, and row count == declared `length` | `checks.py:87` |
| C4 | `check_info_totals` | `total_episodes`, `total_frames`, `total_tasks` agree with disk; `splits.train` covers all episodes; video count == cameras × episodes | `checks.py:116` |
| C5 | `check_global_index` | The global `index` column runs **0..N−1 contiguously across all episodes** | `checks.py:174` |
| C6 | `check_timestamps` | Per-episode timestamps monotonic, and every frame gap within **1e-4 s of 1/fps** | `checks.py:204` |

C1, C2 and C5 protect against common merge failures: duplicated or missing
episode statistics, broken task references, and non-contiguous indices.

---

## 2 · Two problems this raises that the plan has not addressed

### 2.1 · Teleop is variable-rate; LeRobot v2.1 is fixed-rate. Resampling is mandatory.

`check_timestamps` allows a frame-gap drift of **0.1 ms** against `1/fps` (`tolerance_s = 1e-4`,
`checks.py:204`). Our capture pipeline cannot meet that from wall-clock timestamps:

- webcam frame arrival jitters by milliseconds;
- MediaPipe live-stream mode **deliberately drops frames** to minimise latency;
- the safety layer drops frames on purpose — every `HOLD` in `docs/safety-model.md` is a gap in
  accepted targets, and dropping teleports (I1) is a design requirement, not a defect.

So a naive "one row per accepted frame" export produces non-uniform timestamps and trips C6 on
essentially every clip. **The writer must resample onto an exact `k/fps` grid** and emit timestamps
computed from the integer index, never from measured wall-clock.

That forces a semantic decision the plan has not made — **what is emitted for a grid slot during a
`HOLD`?** Three options, and they are not equivalent:

| option | meaning | verdict |
|---|---|---|
| Repeat the last approved target | "the robot was commanded to stay put" | **Correct.** It is what actually happened — a held target is a real command. |
| Interpolate across the gap | invents commands nobody issued | **Reject.** It fabricates training data rather than representing a real hold. |
| End the episode at the gap | every dropout splits the clip | Reject as default; keep as an option for long gaps. |

**Decision: repeat-last for gaps under a threshold; split the episode beyond it.** The threshold and
the count of held frames go in the clip metadata, so a downstream consumer can see how much of an
episode was hold rather than motion. A clip that is 60% held is technically valid and practically
worthless, and the metadata must make that visible rather than burying it.

### 2.2 · In simulation, `action` ≡ `observation.state`, and that must be declared

LeRobot's convention: `action` = commanded, `observation.state` = observed. That maps cleanly onto
our `MotionClip`, which records both — and recording both is required, because `SUCCESS` does not
mean the target was reached (`docs/safety-model.md` R3).

But `galbot-sim` is **kinematic**: positions are interpolated and written directly, with no contact
dynamics. So in simulation the readback is the command, and `observation.state` equals `action`
exactly, every frame.

Such a dataset is **structurally valid and semantically empty**. It passes all six checks and a
policy trained on it learns the identity function. This is not a defect to fix — it is a true
property of simulated data — but it is a claim that could be made accidentally, and this project has
an explicit standard about that.

**Requirement:** every exported dataset carries a provenance block naming the source as `sim` or
`hardware`, and for `sim` an explicit `state_equals_action: true` flag. The README for any published
dataset says it plainly. **We do not publish a simulated dataset in a way that implies demonstration
data from a robot.**

---

## 3 · Field mapping

| LeRobot field | Source | Notes |
|---|---|---|
| `action` | approved `RobotTarget` joint values, **named** | Never a bare positional array — `docs/architecture.md` contracts require named joints, and the arms are mirrored, not identical (`joint4`/`joint6` bounds swap left↔right) |
| `observation.state` | readback from the adapter | Equals `action` in sim — see §2.2 |
| `timestamp` | `index / fps`, computed | **Never** wall-clock — see §2.1 |
| `index` | global, contiguous across episodes | C5 |
| `episode_index` | per-episode, matches filename | C3 |
| `frame_index` | per-episode, 0-based | |
| `task_index` | resolves in `tasks.jsonl` | C2 |
| `observation.images.*` | operator webcam, only if recording is explicitly enabled | Default **off**. Only the skeleton is needed to reproduce the motion, and a dataset that ships video of a person is a different privacy proposition than one that ships joint angles |

**Joint ordering is fixed by name at write time and recorded in the clip.** Positional arrays whose
meaning depends on an implicit order are how `galbot-sim` documents a real hazard: *"Actuator order
≠ joint order in the S1 MJCF … all mapping resolved by joint name at load time rather than by index."*
Same rule here.

---

## 4 · Version targeting

`docs/architecture.md` targets both v3 (current general tooling) and v2.1 (Galbot-compatible). The
validator checks **v2.1**, so v2.1 is the gated path and v3 is best-effort. Do not let a v3-first
implementation quietly change the v2.1 output — the whole point is that a specific external tool
says PASS.

---

## 5 · Acceptance

> ### ⚠️ CORRECTED 2026-08-10 — this spec was wrong about what "validated" means
>
> Everything below §5 was written deriving the contract from `lerobot-dataset-check`, and I claimed
> passing it was *"the same bar Galbot already accepted."* **That was wrong.** That tool's own first
> line reads *"Deliberately does NOT import lerobot."* It validates **internal consistency** — which
> is exactly what Galbot used it for, to confirm their own fix — and it **cannot detect that a
> dataset fails to load.**
>
> Measured: a dataset scoring `10 PASS, 0 WARN, 0 FAIL` raises
> `AttributeError: 'numpy.ndarray' object has no attribute 'items'` in
> `lerobot/datasets/compute_stats.py` when opened with `LeRobotDataset`. `lerobot 0.3.3` was
> installed in the pinned venv the entire time and was never called.
>
> **A0 — the primary acceptance criterion, ahead of everything else:**
> ```python
> from lerobot.datasets.lerobot_dataset import LeRobotDataset
> LeRobotDataset(repo_id="local/check", root=<export root>)   # must not raise
> ```
> The linter remains a useful *secondary* check — it catches the merge-corruption class it was built
> for, which the loader swallows silently. It is necessary and it is not sufficient. Known blockers
> to A0 as of this correction: `episodes_stats.jsonl` carries `{held_frames, held_fraction}` where
> v2.1 requires per-feature `{min,max,mean,std,count}`; and `meta/info.json` omits `chunks_size`,
> `data_path`, `video_path`, `robot_type`, `total_chunks`.

### Secondary criteria (the linter)


1. A clip recorded in simulation exports and passes `lerobot-dataset-check` with **0 FAIL, 0 WARN**.
   WARN counts as failure here: C4 and C6 emit WARN for stale splits and timestamp drift, and both
   are writer bugs in our case, not intentional choices.
2. A **multi-episode** export passes. Single-episode exports cannot exercise C1, C2 or C5 — the
   merge bug was invisible until two datasets met. Minimum three episodes with at least two
   distinct task strings.
3. Round-trip: export → read back → compare against the source `MotionClip` field by field.
4. An export containing a `HOLD` gap is checked explicitly: held frames present, count recorded in
   metadata, timestamps still exactly on the grid.
5. A deliberately corrupted export (duplicate `episode_index` in stats) is confirmed to **FAIL** the
   checker. If the validator cannot catch our injected version of the original bug, we are not
   actually validating anything.

Test 5 is the one worth insisting on. Everything else proves the writer works; test 5 proves the
check works, and an unverified validator is just a green light of unknown wiring.
