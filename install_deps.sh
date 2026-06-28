#!/usr/bin/env bash
# Install all booster_train dependencies (companion to check_env.sh).
#
# Installs, into the Isaac Lab python:
#   1. rsl-rl-lib (pinned to Isaac Lab 2.2's version) + onnxscript (ONNX export)
#   2. booster_assets   (editable — robot URDFs + motion data, BOOSTER_ASSETS_DIR)
#   3. booster_train     (editable — this repo's task package)
# then runs check_env.sh to verify.
#
# Interpreter is auto-detected (active Isaac Lab python, else isaaclab.sh -p),
# same as check_env.sh. Installs auto-retry with --user when the system
# site-packages is read-only (common on Isaac Sim cloud images).
#
# Env vars:
#   ISAACLAB_PATH        Isaac Lab dir (for the launcher fallback)
#   BOOSTER_ASSETS_PATH  booster_assets repo dir (else auto-searched)
#   PIP_USER=1           force --user installs from the start
#
# NOTE: this installs project deps; it does NOT repair a broken Isaac Sim pip
# (the setuptools-81 / find_distributions issue). See check_env.sh header for that.
#
# Usage:
#   ./install_deps.sh
#   BOOSTER_ASSETS_PATH=/workspace/booster_assets PIP_USER=1 ./install_deps.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSL_RL_VERSION="5.0.1"

# --- resolve the Isaac Lab python (mirrors check_env.sh) ---
PY=()
if command -v python >/dev/null 2>&1 \
   && python -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("isaaclab") else 1)' 2>/dev/null; then
  PY=(python)
else
  for dir in "${ISAACLAB_PATH:-}" "$HOME/programming/IsaacLab" "$REPO_ROOT/../IsaacLab" "$HOME/IsaacLab" "/workspace/isaaclab"; do
    [[ -n "$dir" && -x "$dir/isaaclab.sh" ]] || continue
    echo "[install_deps] using Isaac Lab launcher: $dir/isaaclab.sh"
    PY=("$dir/isaaclab.sh" -p)
    break
  done
fi
if [[ ${#PY[@]} -eq 0 ]]; then
  echo "error: no Isaac Lab python found." >&2
  echo "  activate your Isaac Lab env, or set ISAACLAB_PATH=/path/to/IsaacLab." >&2
  exit 2
fi
echo "[install_deps] python: ${PY[*]}"

USE_USER="${PIP_USER:-0}"

# pip install with auto --user fallback on a read-only/permission-denied site.
pip_install() {
  local args=()
  [[ "$USE_USER" == "1" ]] && args+=(--user)
  if "${PY[@]}" -m pip install "${args[@]}" "$@"; then
    return 0
  fi
  if [[ "$USE_USER" != "1" ]]; then
    echo "[install_deps] install failed — retrying with --user (system site likely read-only)..." >&2
    USE_USER=1   # stick with --user for the rest of the run
    "${PY[@]}" -m pip install --user "$@"
  else
    return 1
  fi
}

# --- 1. rsl_rl (pinned) + onnxscript ---
echo "[install_deps] (1/3) rsl-rl-lib==$RSL_RL_VERSION + onnxscript"
pip_install "rsl-rl-lib==$RSL_RL_VERSION" "onnxscript>=0.5"

# --- 2. booster_assets (editable) ---
echo "[install_deps] (2/3) booster_assets"
ASSETS="${BOOSTER_ASSETS_PATH:-}"
if [[ -z "$ASSETS" ]]; then
  for c in "$REPO_ROOT/../booster_assets" "$HOME/programming/booster_assets" "/workspace/booster_assets"; do
    if [[ -f "$c/pyproject.toml" || -f "$c/setup.py" ]]; then ASSETS="$c"; break; fi
  done
fi
if [[ -n "$ASSETS" && ( -f "$ASSETS/pyproject.toml" || -f "$ASSETS/setup.py" ) ]]; then
  echo "[install_deps]   editable: $ASSETS"
  pip_install -e "$ASSETS"
else
  echo "[install_deps]   WARN: booster_assets repo not found — set BOOSTER_ASSETS_PATH. Skipping." >&2
fi

# --- 3. booster_train (this repo, editable) ---
echo "[install_deps] (3/3) booster_train (editable): $REPO_ROOT/source/booster_train"
pip_install -e "$REPO_ROOT/source/booster_train"

# --- verify ---
echo "[install_deps] done — verifying with check_env.sh"
"$REPO_ROOT/check_env.sh" || true
