#!/usr/bin/env bash
# Verify the environment can train/play booster_train tasks.
#
# Picks the right interpreter automatically:
#   1. If `isaaclab` is importable in the active python -> use that python.
#   2. Else use Isaac Lab's launcher: $ISAACLAB_PATH/isaaclab.sh -p
#      ($ISAACLAB_PATH, else a few common locations).
#
# Usage:
#   ./check_env.sh                 # auto-detect
#   ISAACLAB_PATH=/path/to/IsaacLab ./check_env.sh
# Any extra args are forwarded to scripts/check_env.py.
#
# Installing missing dependencies (run with the Isaac Lab python, e.g. after
# `conda activate <isaaclab-env>`, or via `<IsaacLab>/isaaclab.sh -p -m pip ...`):
#   rsl_rl  : python -m pip install rsl-rl-lib==5.0.1 onnxscript>=0.5
#             (PyPI name is "rsl-rl-lib"; import name is "rsl_rl"; Isaac Lab 2.2 pins 5.0.1)
#   assets  : cd <booster_assets> && python -m pip install -e .
#   package : python -m pip install -e source/booster_train
#   IsaacLab: see https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$REPO_ROOT/scripts/check_env.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "error: $PY_SCRIPT not found" >&2
  exit 2
fi

# 1) Active env already has Isaac Lab? (find_spec = no heavy import / no app launch)
if command -v python >/dev/null 2>&1 \
   && python -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("isaaclab") else 1)' 2>/dev/null; then
  exec python "$PY_SCRIPT" "$@"
fi

# 2) Fall back to the Isaac Lab launcher.
CANDIDATES=(
  "${ISAACLAB_PATH:-}"
  "$HOME/programming/IsaacLab"
  "$REPO_ROOT/../IsaacLab"
  "$HOME/IsaacLab"
)
for dir in "${CANDIDATES[@]}"; do
  [[ -n "$dir" && -x "$dir/isaaclab.sh" ]] || continue
  echo "[check_env] using Isaac Lab launcher: $dir/isaaclab.sh"
  exec "$dir/isaaclab.sh" -p "$PY_SCRIPT" "$@"
done

echo "error: no Isaac Lab python found." >&2
echo "  - activate your Isaac Lab conda/venv, or" >&2
echo "  - set ISAACLAB_PATH=/path/to/IsaacLab and re-run." >&2
exit 2
