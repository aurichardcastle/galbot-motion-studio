#!/bin/sh
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

exec "$python_bin" -c '
import sys
sys.path.insert(0, sys.argv.pop(1))
from galbot_motion_studio.cli import main
raise SystemExit(main(sys.argv[1:]))
' "$project_dir/src" demo "$@"
