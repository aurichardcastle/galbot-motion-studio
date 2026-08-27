# Operator guide

Simulation only. No robot is connected, no network transport exists, and there is
no physical command path. Say that out loud once at the start; the window says it
too, in the app bar, on every frame.

---

## 1. The one command

```bash
cd galbot-teleop
pkill -f galbot_motion_studio.cli; mkdir -p artifacts/demo-20260827 && set -o pipefail && \
PYTHONPATH=src .venv/bin/python -m galbot_motion_studio.cli preview \
  --camera builtin --fullscreen --preview-fps 15.3 \
  --calibration-id demo-20260827-v1 \
  --liveness-max-static-ms 500 --liveness-max-history 256 \
  --calibration-window-ms 1500 --calibration-min-samples 15 \
  --calibration-max-center-deviation-normalized 0.03 \
  --calibration-max-shoulder-width-deviation-normalized 0.03 \
  --calibration-max-eye-span-deviation-normalized 0.02 \
  --output artifacts/demo-20260827/live.json \
  --source-video artifacts/demo-20260827/raw.mp4 \
  --landmark-sidecar artifacts/demo-20260827/raw.landmarks.json \
  --preview-video artifacts/demo-20260827/composite.mp4 \
  2>&1 | tee artifacts/demo-20260827/terminal.log
```

`q` or `Esc` closes it. A retry needs **both** a new `--calibration-id` and a new
artifact directory. The ARTIFACT PATHS are what is guarded — a run refuses to
overwrite an existing output without `--force`; nothing enforces uniqueness of the
calibration id itself, so do not claim it does.

**Keep the recording flags.** They were expected to cost real latency and they do
not. Measured photon → twin-pose through one harness, fullscreen, against a paced
30 fps replay of a real session:

| configuration | mean | median | runs |
|---|---|---|---|
| exactly the command above (all three recordings) | 127.1 – 137.7 ms | 95.2 – 106.4 ms | 5 |
| identical, recordings off | 127.0 – 133.0 ms | 95.8 – 100.4 ms | 2 |

Run-to-run scatter is ~10 ms and the two configurations sit inside each other's
range, so **quote it as 127–138 ms mean, 95–106 ms median** and treat the
provenance as free. Those are the full observed ranges over every run, not a
best-of: an earlier draft of this table quoted a narrower band that the very next
run fell outside.

Read the harness before quoting these. The camera cannot be opened by an
automated process on this Mac, so the measurement substitutes a fake capture that
replays a recorded session at a true 30 fps into the real `WebcamSource` (the
same injection `tests/test_live_preview_end_to_end.py` uses); everything
downstream is the real command. `--preview-fps 15.3` is the declared frame rate
of the videos this run WRITES — both the composite and the retained raw capture —
and does not change the 30 fps at which frames are captured or the rate at which
they are solved. "Recordings off" means all
three of `--source-video`, `--landmark-sidecar`, `--preview-video` removed, and
the second row's two figures are two runs of that.

A separate, earlier experiment quoted in `cli.py` reports 97.5 / 100.4 / 97.7 ms
for a *different* configuration — windowed rather than fullscreen, equal panels,
no recordings, a different harness. It is not comparable with this table and
should not be quoted against it.

## 2. Getting calibrated in under ten seconds

Calibration needs **15 consecutive frames** in which your shoulder centre,
shoulder width and eye span all hold still. The window is the thing that usually
fails, not the confidence. Confidence is judged per control group and NOT at one
threshold: wrists, shoulders and three face points must clear **0.5**, while
elbows are required to be present but are gated at **0.3** — MediaPipe reports a
correctly tracked elbow below 0.5 whenever a forearm crosses the torso or face.

- Stand so **head, both shoulders, both elbows and both wrists** are in frame.
  The CALIBRATION card names whichever landmark is missing and which way to move
  the camera. (The TRACKING card below it answers a different question — whether a
  limb's SHAPE is drivable — and judges at a lower threshold, so during
  calibration read the CALIBRATION card; a limb that is visible but not yet
  trusted now reads "weak" there rather than green "tracking".)
- Face the camera square and **stop moving** for about a second. Deliberate
  stillness, not a pose.
- If the CALIBRATION card says *BODY PARTS FOUND BUT NOT TRUSTED*, the camera is
  aimed wrong — the detector is placing landmarks outside the image. Tilt the lid
  down, do not step back.

## 3. What to point at on screen

**Left column, top:** you, with the tracked skeleton drawn on you. A limb the
system has stopped trusting goes grey and hollow and is labelled `... HELD` on
your own body. That is the whole point of the overlay: a stopped robot must never
look like a tracking one.

**TRACKING card:** per limb, whether *arm shape* can be driven this frame. "no
elbow" means the swivel mapper has nothing to place the elbow with and the arm
falls back to wrist-only — worth saying out loud if it appears.

**JOINT ACTIVITY:** all nineteen commanded joints, pinned by a test to the clip's
own `joint_order` so none can go missing again — the torso (`leg_joint4`) was
absent from this card until today, on a card whose entire purpose is showing a
joint that has stopped. Green = moving, grey = still,
**red = not held and not moving anyway**, which is the failure mode that once hid
a dead shoulder joint for a whole session.

**SAFETY card:** the supervisor's live self-clearance, drawn against the 5 mm
floor and the 6.5 mm standoff. This is the honest one to dwell on — it is an
*independent* check that re-derives the clearance itself and rejects the frame if
the retargeter's declared value disagrees.

**The panels are not mirrored.** Raise your left hand and it appears on the RIGHT
of your panel — you are looking at yourself as the room does, not as a mirror
does — and the robot's left arm appears on the right of its panel too. The two
agree, which is the thing that matters. `--mirror-camera` gives you the selfie
view instead; it mirrors both panels, which also mirrors the robot's chassis and
makes the GALBOT wordmark read backwards, so it is off by default.

**Right panel:** the digital twin, and it is the wider of the two on purpose —
across the 463 poses of the last recorded session the robot spans **1.34 m at the
median and 1.87 m at its widest**, and a narrow panel would either clip its hands
or force the camera so far back the robot went small.

If you raise both arms straight out to the sides, the very tips of the grippers
can still touch the panel edge — measured across all 463 poses of the last
recorded session, **3.2%** of them put a robot pixel outside the panel. The
previous framing put a pixel off the left or right edge on **61.8%** of poses and
the wheeled base below the bottom of frame on **100%** of them. The robot is
about a quarter smaller on screen than it was; that is the price of it being
whole, and it is why the twin gets the wider panel rather than an equal one.

## 4. What will happen, and what to say when it does

**Read this one before you start.** A sudden shoulder-yaw discontinuity holds
both arms and the torso. The banner says `TORSO_YAW_RECALIBRATION_REQUIRED` and
asks the operator to face the camera and hold still. It clears only after 15
consecutive face-on observations with continuous timestamps and calm yaw; no
command is emitted while that evidence is accumulating.

Four of five realistic perturbations reproduce it on recorded footage: walking out
of shot and back in, an occluded shoulder, turning side-on, and a sudden light
change. A slow, deliberate quarter-turn does not. So: **stay in frame, keep both
shoulders visible, and turn slowly.** If it latches, say "the torso mapper is
holding until it can re-establish a trustworthy reference." Face the camera and
hold still. If it does not recover, finish the take and begin a new session.

**Limbs will go amber and hold.** This is normal and it is the system working.
The gate is per control group, so an occluded left wrist holds the left arm while
the right arm keeps tracking. The `WHY:` banner names the reason.

**The twin lags you by about a tenth of a second** — 127–138 ms mean, 95–106 ms
median (§1). The dominant term is the control worker — retarget + IK + swept
self-collision checking. Measured on the committed replay path (which anyone can
re-run), it reports **mean 88–89 ms per solve** under `--analysis-sync` and
**95–110 ms** in live mode; the session prints the figure itself on the
`realtime:`/`analysis-sync:` line, which is a few lines above the end (the
`saved …` lines come after it). Note the `| tee` in §1 makes stdout block-buffered,
so the log file fills in bursts rather than live.
If someone asks where the rest went: MediaPipe holistic detection is **p50 30–31
ms** and is a fixed cost of the model on CPU, and the swept collision check is
roughly 72% of the control worker. (An earlier draft of this line said 44–65 ms
for the worker, which does not reproduce in any configuration and did not add up
against the total; these do.) The single
biggest remaining lever is written up in `docs/lookahead-option.md`: the guarded
controller's look-ahead horizon is six governor steps long, so half of every
frame's swept checks run sampling-degraded against the 12-sample cap.

**The arms will not match your shape exactly.** Do not oversell this. Measured
with the project's own metric on the last witness capture: upper-arm direction
error **45.6° left / 38.1° right**. The IK reaches the wrist point it is asked
for to **1.5 mm on average** — p50 0.5 mm, but p95 6.1 mm and max 9.9 mm across
the 582 approved frames, so quote it as "about a millimetre typical, under a
centimetre worst case", not as a bound; the wrist point itself is derived in
anisotropic image units, and
the elbow placement (the swivel) does not converge on most frames. Both defects
are measured in the project validation notes. Say "the wrist tracks, the arm
*shape* is the next piece of work" — that is true, specific, and much stronger
than being caught claiming otherwise.

## 5. If it goes wrong

| Symptom | Cause | Do this |
|---|---|---|
| `cannot open webcam device` | camera held by another app, or Continuity Camera took index 0 | close FaceTime/Zoom; `--camera builtin` already resolves by device type |
| Calibration never locks | you are moving, or a limb is out of frame | read the CALIBRATION card; it names the blocker |
| A limb goes amber and holds, then comes back | landmark confidence dipped | nothing; this is the gate working, and the `WHY:` banner names the reason |
| **Both arms and the torso hold together, reading `TORSO_YAW_RECALIBRATION_REQUIRED`** | the shoulder-yaw reference became discontinuous | Face the camera and hold still. The hold clears only after 15 consecutive face-on, continuous, calm observations; otherwise finish the take and start a new session. |
| Window says FAULT | latched safety fault | this is by design and requires a restart; the artifacts are still written, and the process now exits NON-ZERO |
| Whole thing feels slow | something else is using the CPU | the recordings are not the cause (measured); close other apps |

## 6. Provenance, if anyone asks

Every run writes a MotionClip, the raw camera video, a cryptographically bound
frame map, and a landmark sidecar. Replaying that raw video through
`preview --video ... --analysis-sync` reproduces the run deterministically — the
published **MotionClip is byte-identical** across repeated runs of the same tree
(verified twice this session, same sha256), and **motion-identical** across
today's changes — 582 command frames, 77.989 rad of arm travel, 0.8944 rad/s
peak, 0.0585 rad max step, every commanded joint vector equal to the pre-session
tree's, checked field by field. Note that the clip's sha256 does move when the
SOURCE changes even though the motion does not: `implementation_hash` and the
`arm_generation` token derived from it are part of the record, on purpose. Byte
identity is a claim about re-running one tree, not about two. The **`composite.mp4` is not** byte-identical run to run, and
neither is the `video_sha256` its manifest carries — the encoder is not
reproducible. The clip is the evidence; the composite is an operator review aid.
