#!/bin/bash
# Extract video frames + plot what the robot actually did.
# Usage: ./look.sh artifacts/rehearsal-20260826-live
set -uo pipefail
D="${1:?usage: ./look.sh <artifact-dir>}"
PY=${GALBOT_MOTION_STUDIO_PYTHON:-.venv/bin/python}
if [ ! -x "$PY" ]; then
  echo "missing Python runtime: $PY" >&2
  exit 2
fi
export PYTHONPATH=src
mkdir -p "$D/look"

VID="$D/composite.mp4"; [ -s "$VID" ] || VID="$D/raw.mp4"
echo "extracting frames from $VID"
$PY - "$VID" "$D/look" <<'PYEOF'
import cv2, sys
cap=cv2.VideoCapture(sys.argv[1]); out=sys.argv[2]
n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
picks=[int(n*f) for f in (0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95)] if n else list(range(0,300,30))
saved=0
for i,idx in enumerate(picks):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok,img=cap.read()
    if not ok: continue
    h,w=img.shape[:2]
    if w>1280: img=cv2.resize(img,(1280,int(h*1280/w)))
    cv2.imwrite(f"{out}/frame_{i:02d}.png", img); saved+=1
print(f"  wrote {saved} frames ({n} in source)")
PYEOF

echo "plotting the robot's own motion"
$PY - "$D" <<'PYEOF'
import json,sys,collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d=sys.argv[1]
fr=json.load(open(f"{d}/live.json"))["frames"]
t0=fr[0]["source_mono_ns"]
t=[(f["source_mono_ns"]-t0)/1e9 for f in fr]
def series(key):
    out=collections.defaultdict(list)
    for f in fr:
        raw=f.get("observed_joints_rad") or []
        src=dict(raw) if isinstance(raw,list) else dict(raw)
        if not src and f.get("target"):
            src={j["name"]:j["position_rad"] for j in f["target"]["joints"]}
        for k,v in src.items():
            if k.startswith(key): out[k].append(v)
    return out
panels=[("leg_joint4","tummy yaw (leg_joint4)"),("left_arm","left arm"),
        ("right_arm","right arm"),("head_joint","head")]
fig,ax=plt.subplots(len(panels),1,figsize=(13,11),sharex=True)
for a,(pre,title) in zip(ax,panels):
    s=series(pre)
    for k,v in sorted(s.items()):
        a.plot(t[:len(v)],v,lw=1.2,label=k.replace(pre,"").lstrip("_") or k)
    a.set_title(title,fontsize=10,loc="left"); a.set_ylabel("rad"); a.grid(alpha=.25)
    if len(s)<=8: a.legend(fontsize=6,ncol=4,loc="upper right")
grp={"leg_joint4":"torso","left_arm":"left_arm","right_arm":"right_arm","head_joint":"head"}
for a,(pre,_) in zip(ax,panels):
    g=grp[pre]
    for i,f in enumerate(fr):
        if g in (f.get("held_groups") or ()):
            a.axvspan(t[i],t[min(i+1,len(t)-1)],color="red",alpha=.10,lw=0)
ax[-1].set_xlabel("seconds (red bands = that group HELD)")
fig.suptitle("what the robot actually did",fontsize=12)
fig.tight_layout(); fig.savefig(f"{d}/look/robot.png",dpi=110)
n=len(fr); held=collections.Counter()
for f in fr:
    for g in (f.get("held_groups") or ()): held[g]+=1
print(f"  frames {n}, duration {t[-1]:.1f}s")
for k in ("torso","left_arm","right_arm","head"):
    print(f"    {k:10s} held {held.get(k,0):4d}/{n} ({100*held.get(k,0)/n:.0f}%)")
ser=series("")
def reach(p): return sum(max(v)-min(v) for k,v in ser.items() if k.startswith(p))
print(f"  reach: torso {reach('leg_joint4'):.3f}  L {reach('left_arm'):.3f}  R {reach('right_arm'):.3f} rad")
print(f"  wrote {d}/look/robot.png")
PYEOF
grep -E "^(realtime|analysis-sync):" "$D/terminal.log" 2>/dev/null | tail -1
