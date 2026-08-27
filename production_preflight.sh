#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${GALBOT_MOTION_STUDIO_PYTHON:-}" ]; then
  python_bin=$GALBOT_MOTION_STUDIO_PYTHON
elif [ -x "$project_dir/.venv/bin/python" ]; then
  python_bin="$project_dir/.venv/bin/python"
else
  echo "missing .venv/bin/python; create it with: python3.11 -m venv .venv && .venv/bin/pip install -e '.[vision,export-validation,dev]'" >&2
  exit 2
fi

offline_only=0
if [ "${1:-}" = "--offline-only" ]; then
  offline_only=1
elif [ "$#" -ne 0 ]; then
  echo "usage: $0 [--offline-only]" >&2
  exit 2
fi

preflight_tmp=$(mktemp -d "${TMPDIR:-/tmp}/galbot-motion-studio-preflight.XXXXXX")
cleanup() {
  rm -rf -- "$preflight_tmp"
}
trap cleanup EXIT HUP INT TERM

cd "$project_dir"
git diff --check
PYTHONPATH=src "$python_bin" -m compileall -q src
PYTHONPATH=src "$python_bin" -m pytest -q

mkdir -p "$preflight_tmp/wheel"
"$python_bin" -m pip wheel --no-deps --wheel-dir "$preflight_tmp/wheel" .
"$python_bin" -m venv --system-site-packages "$preflight_tmp/venv"
wheel_path=
for candidate in "$preflight_tmp"/wheel/*.whl; do
  wheel_path=$candidate
done
if [ -z "$wheel_path" ] || [ ! -f "$wheel_path" ]; then
  echo "production preflight: wheel build produced no artifact" >&2
  exit 1
fi
# The installed-wheel smoke test must model the customer install path.  Building
# with --no-deps above keeps the wheel artifact isolated; installing it with
# --no-deps here merely manufactures missing runtime imports and never verifies
# the package metadata we are trying to ship.
# The demo validates a LeRobot v2.1 export and the product exposes camera
# preview, so smoke the documented production feature set rather than a bare
# library-only install.
"$preflight_tmp/venv/bin/pip" install "$wheel_path[vision,export-validation]"
"$preflight_tmp/venv/bin/galbot-motion-studio" --help >/dev/null
"$preflight_tmp/venv/bin/galbot-motion-studio" demo --output "$preflight_tmp/demo"

if [ "$offline_only" -eq 0 ]; then
  if [ -z "${GALBOT_AB_REPORT:-}" ] || [ ! -f "$GALBOT_AB_REPORT" ]; then
    echo "production preflight: set GALBOT_AB_REPORT to a fresh real-camera analyze-ab JSON" >&2
    exit 2
  fi
  "$python_bin" - "$GALBOT_AB_REPORT" <<'PY'
from json import loads
from pathlib import Path
import sys

report = loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gate = report.get("release_gate", {})
if gate.get("verdict") != "PASS" or gate.get("promotion_allowed") is not True:
    raise SystemExit("production preflight: real-camera A/B release gate is not PASS")
PY
fi

echo "production preflight: PASS"
if [ "$offline_only" -eq 1 ]; then
  echo "production preflight: real-camera A/B gate intentionally skipped (--offline-only)"
fi
