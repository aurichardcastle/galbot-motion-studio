#!/bin/sh
# One command for the demo. Run it again and it starts a NEW session -- its own
# artifact directory and its own calibration id -- so nothing you recorded is
# ever overwritten and every take is kept.
#
#   ./demo_session.sh                          live camera, fullscreen
#   ./demo_session.sh --replay artifacts/witness-live-9/raw.mp4
#   ./demo_session.sh --dry-run                print the command, run nothing
#
# Anything after those is passed straight through to `preview`, so
# `./demo_session.sh --mirror-camera` works.
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${GALBOT_MOTION_STUDIO_PYTHON:-}" ]; then
  python_bin=$GALBOT_MOTION_STUDIO_PYTHON
elif [ -x "$project_dir/.venv/bin/python" ]; then
  python_bin="$project_dir/.venv/bin/python"
else
  echo "missing .venv/bin/python; create it with: python3.11 -m venv .venv && .venv/bin/pip install -e '.[vision,dev]'" >&2
  exit 2
fi

replay=
dry_run=
while [ $# -gt 0 ]; do
  case $1 in
    --replay) replay=${2:?--replay needs a video path}; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) break ;;
  esac
done

# The session id is the clock, so a second run can never collide with the first.
session=demo-$(date +%Y%m%d-%H%M%S)
session_dir="$project_dir/artifacts/$session"
if [ -e "$session_dir" ]; then
  echo "refusing to reuse $session_dir -- wait a second and run again" >&2
  exit 1
fi

# Calibration gates. These are the measured, approved values; they are here
# rather than in your shell history so every take is gated identically.
set -- \
  --calibration-id "$session" \
  --calibration-window-ms 1500 \
  --calibration-min-samples 15 \
  --calibration-max-center-deviation-normalized 0.03 \
  --calibration-max-shoulder-width-deviation-normalized 0.03 \
  --calibration-max-eye-span-deviation-normalized 0.02 \
  --preview-fps 15.3 \
  --output "$session_dir/live.json" \
  --preview-video "$session_dir/composite.mp4" \
  "$@"

if [ -n "$replay" ]; then
  # A recorded session is already its own raw evidence, and --source-video is
  # refused with --video, so a replay writes the clip and the composite only.
  set -- --video "$replay" --analysis-sync "$@"
else
  # macOS hands a nearby iPhone out as a Continuity Camera and will prefer it,
  # sometimes between our enumeration and OpenCV's open. `builtin` resolves by
  # device type, but the only bulletproof fix is turning Continuity Camera off on
  # the phone. GALBOT_CAMERA=0 forces an index when you have no time to argue.
  # The default preserves the robot's orientation and wordmark. Set
  # GALBOT_MIRROR=1 for a mirrored selfie view; both panels flip together.
  if [ "${GALBOT_MIRROR:-0}" = "1" ]; then
    set -- --mirror-camera "$@"
  fi
  set -- --camera "${GALBOT_CAMERA:-builtin}" --fullscreen \
    --liveness-max-static-ms 500 --liveness-max-history 256 \
    --source-video "$session_dir/raw.mp4" \
    --landmark-sidecar "$session_dir/raw.landmarks.json" \
    "$@"
fi

if [ -n "$dry_run" ]; then
  printf 'would run:\n  %s -m galbot_motion_studio.cli preview' "$python_bin"
  while [ $# -gt 0 ]; do
    case ${2:-} in
      --*|'') printf ' \\\n    %s' "$1"; shift ;;
      *) printf ' \\\n    %s %s' "$1" "$2"; shift 2 ;;
    esac
  done
  printf '\n'
  exit 0
fi

mkdir -p "$session_dir"
echo "session $session  ->  $session_dir"
echo

# `q` or Esc ends a live take. A latched safety FAULT exits non-zero on purpose,
# so capture the status rather than letting `set -e` kill the summary below --
# the artifacts are still written and you still want to be told where they are.
status=0
PYTHONPATH="$project_dir/src" "$python_bin" -m galbot_motion_studio.cli preview "$@" \
  2>&1 | tee "$session_dir/terminal.log" || status=$?

echo
echo "session $session"
for f in live.json raw.mp4 raw.frame-map.json raw.landmarks.json composite.mp4 \
         composite.frame-map.json terminal.log; do
  if [ -f "$session_dir/$f" ]; then
    printf '  %-26s %s\n' "$f" "$(du -h "$session_dir/$f" | cut -f1)"
  fi
done
if [ ! -f "$session_dir/live.json" ]; then
  echo "  NO CLIP PUBLISHED -- read terminal.log; the raw capture is retained for diagnosis"
fi
if [ -f "$session_dir/raw.mp4" ]; then
  # Only a LIVE take records its own camera video; a replay has none to offer.
  echo
  echo "  replay this take:  $0 --replay $session_dir/raw.mp4"
fi
exit "$status"
