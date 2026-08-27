#!/bin/bash
# Close G1/G3/G5/G6/G8 from one retained camera rehearsal.
# Usage: ./close_gates.sh artifacts/rehearsal-20260826-live
set -uo pipefail
D="${1:?usage: ./close_gates.sh <artifact-dir>}"
PY=${GALBOT_MOTION_STUDIO_PYTHON:-.venv/bin/python}
if [ ! -x "$PY" ]; then
  echo "missing Python runtime: $PY" >&2
  exit 2
fi
export PYTHONPATH=src
pass=0; fail=0
ok(){ echo "  PASS  $1"; pass=$((pass+1)); }
no(){ echo "  FAIL  $1"; fail=$((fail+1)); }

echo "== G5  retained artifact set =="
for f in raw.mp4 raw.frame-map.json raw.landmarks.json live.json terminal.log; do
  [ -s "$D/$f" ] && ok "$f" || no "$f missing or empty"
done
$PY - "$D" <<'PYEOF'
import json,sys
d=sys.argv[1]
try:
    m=json.load(open(f"{d}/raw.frame-map.json"))
except Exception as e:
    print(f"  FAIL  frame map unreadable: {e}"); sys.exit(0)
v,o=m.get("schema_version"),m.get("capture_outcome")
print(f"  {'PASS' if v==3 else 'FAIL'}  frame map schema_version={v} (need 3)")
print(f"  {'PASS' if o=='succeeded' else 'FAIL'}  capture_outcome={o!r} (need 'succeeded')")
PYEOF

echo
echo "== G1  live throughput, with motion beside it =="
grep -E "^realtime:" "$D/terminal.log" | tail -1 || no "no realtime metrics line in terminal.log"
$PY - "$D" <<'PYEOF'
import json,sys,collections,statistics
d=sys.argv[1]
fr=json.load(open(f"{d}/live.json"))["frames"]
n=len(fr); print(f"  frames recorded: {n}")
g=collections.Counter()
for f in fr:
    for x in (f.get("held_groups") or ()): g[x]+=1
for k in ("left_arm","right_arm","torso","head"):
    print(f"    {k:10s} held {g.get(k,0):4d}/{n}  ({100*g.get(k,0)/max(n,1):.1f}%)")
ser={}
for f in fr:
    t=f.get("target")
    if not t: continue
    for j in t["joints"]: ser.setdefault(j["name"],[]).append(j["position_rad"])
def reach(p): return sum(max(s)-min(s) for k,s in ser.items() if k.startswith(p))
print(f"  arm reach  L {reach('left_arm'):.3f} rad   R {reach('right_arm'):.3f} rad")
print(f"  torso reach  {reach('leg_joint4'):.3f} rad")
print("  NOTE: an fps figure without these reach numbers beside it is not evidence of tracking.")
PYEOF

echo
echo "== G8  failure-path evidence recorded =="
$PY - "$D" <<'PYEOF'
import json,sys,collections
fr=json.load(open(f"{sys.argv[1]}/live.json"))["frames"]
c=collections.Counter()
for f in fr:
    rs=f.get("held_group_reasons") or []
    for _,r in (rs.items() if isinstance(rs,dict) else rs): c[r]+=1
print("  hold reasons observed:", dict(c.most_common()) or "NONE (no failure path exercised)")
for want in ("MISSING_REQUIRED_LANDMARK","LOW_CONFIDENCE"):
    print(f"  {'PASS' if any(want in k for k in c) else 'note'}  {want}")
if any("STALE" in k for k in c): print("  ATTENTION  STALE observed — whole-frame freeze on the live path")
if any("RECALIB" in k for k in c): print("  ATTENTION  torso recalibration lock fired during the run")
PYEOF

echo
echo "== G6  deterministic replay =="
for r in a b; do
  $PY -m galbot_motion_studio.cli preview --video "$D/raw.mp4" \
    --source-frame-map "$D/raw.frame-map.json" --source-landmark-sidecar "$D/raw.landmarks.json" \
    --analysis-sync --output "$D/replay-$r.json" --force >"$D/replay-$r.log" 2>&1 \
    || { no "replay $r failed — see $D/replay-$r.log"; }
done
if cmp -s "$D/replay-a.json" "$D/replay-b.json"; then ok "replays byte-identical"; else no "replay JSON differs"; fi
$PY - "$D" <<'PYEOF'
import sys
from pathlib import Path
from galbot_motion_studio.recording import MotionClip
d=Path(sys.argv[1])
try:
    good=all(MotionClip.load(d/f"replay-{r}.json").source_replay.publishable for r in "ab")
    print(f"  {'PASS' if good else 'FAIL'}  source provenance publishable")
except Exception as e:
    print(f"  FAIL  provenance check: {e}")
PYEOF

echo
echo "== summary =="
$PY tools/live_rehearsal_summary.py "$D/live.json" --terminal-log "$D/terminal.log" \
  --output "$D/live-summary.json" >/dev/null 2>&1 && ok "live-summary.json written" || no "summary tool failed"
echo
echo "checks passed: $pass   failed: $fail"
