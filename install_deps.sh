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
# same as check_env.sh. Installs auto-retry with sudo when the system
# site-packages is read-only (common on Isaac Sim cloud images) — a --user
# retry does NOT work here, because Isaac Sim puts its bundled site-packages
# ahead of the user site on sys.path, so a --user install is shadowed by the
# pre-bundled copy and never imported.
#
# Env vars:
#   ISAACLAB_PATH        Isaac Lab dir (for the launcher fallback)
#   BOOSTER_ASSETS_PATH  booster_assets repo dir (else auto-searched)
#   PIP_SUDO=1           use sudo for installs from the start
#
# NOTE: this installs project deps; it does NOT repair a broken Isaac Sim pip
# (the setuptools-81 / find_distributions issue). See check_env.sh header for that.
#
# Usage:
#   ./install_deps.sh
#   BOOSTER_ASSETS_PATH=/workspace/booster_assets PIP_SUDO=1 ./install_deps.sh
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

# Resolve the real interpreter binary — sudo needs the executable directly, not
# the isaaclab.sh / python.sh wrapper (which sets up env we don't need for pip).
PYEXE="$("${PY[@]}" -c 'import sys; print(sys.executable)' 2>/dev/null | tail -n1)"
if [[ -z "$PYEXE" || ! -x "$PYEXE" ]]; then
  echo "[install_deps] error: could not resolve python executable (got '$PYEXE')." >&2
  exit 2
fi

USE_SUDO="${PIP_SUDO:-0}"

# pip install with auto sudo fallback on a read-only/permission-denied site.
# --user does NOT help on Isaac Sim images (bundled site shadows the user site),
# so we escalate to sudo against the real interpreter binary instead.
pip_install() {
  if [[ "$USE_SUDO" == "1" ]]; then
    sudo "$PYEXE" -m pip install "$@"
    return $?
  fi
  if "${PY[@]}" -m pip install "$@"; then
    return 0
  fi
  echo "[install_deps] install failed — retrying with sudo (system site likely read-only)..." >&2
  if ! command -v sudo >/dev/null 2>&1; then
    echo "[install_deps] error: sudo not found. Either run as a user who can write to" >&2
    echo "  the Isaac Sim site-packages, or chown it: " >&2
    echo "  sudo chown -R \"\$(whoami)\" \"\$(dirname \"\$($PYEXE -c 'import site;print(site.getsitepackages()[0])')\")\"" >&2
    return 1
  fi
  USE_SUDO=1   # stick with sudo for the rest of the run
  sudo "$PYEXE" -m pip install "$@"
}

# --- 1. rsl_rl (pinned) + onnxscript ---
# --force-reinstall --no-deps cleanly overwrites any pre-bundled rsl-rl-lib in
# the read-only system site (its deps — torch/numpy/etc — are already satisfied
# by Isaac Sim, so we must not let pip try to touch them).
echo "[install_deps] (1/3) rsl-rl-lib==$RSL_RL_VERSION + onnxscript"
pip_install --force-reinstall --no-deps "rsl-rl-lib==$RSL_RL_VERSION" "onnxscript>=0.5"

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
