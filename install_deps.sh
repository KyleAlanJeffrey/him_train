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
# same as check_env.sh.
#
# Isaac Sim cloud images (e.g. Brev) run as a non-root user with a READ-ONLY
# bundled site-packages and no sudo. So everything installs with `pip --user`
# into ~/.local. But Isaac's bundled site-packages sits AHEAD of the user site
# on sys.path, so a plain --user install of a package that already ships in the
# bundle (rsl-rl-lib, pinned there at an older version) would be shadowed and
# never imported. We fix that by prepending the user-site dir to PYTHONPATH and
# persisting that export to ~/.bashrc, so the --user copy wins. The launcher
# (python.sh) preserves PYTHONPATH, so both `python` and `isaaclab.sh -p` see it.
#
# Env vars:
#   ISAACLAB_PATH        Isaac Lab dir (for the launcher fallback)
#   BOOSTER_ASSETS_PATH  booster_assets repo dir (else auto-searched)
#   NO_BASHRC=1          don't append the PYTHONPATH export to ~/.bashrc
#
# NOTE: this installs project deps; it does NOT repair a broken Isaac Sim pip
# (the setuptools-81 / find_distributions issue). See check_env.sh header for that.
# NOTE: --user installs live on the container's writable layer / ~/.local and may
# not survive a container rebuild on ephemeral cloud images — re-run if recreated.
#
# Usage:
#   ./install_deps.sh
#   BOOSTER_ASSETS_PATH=/workspace/booster_assets ./install_deps.sh
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

# User site-packages dir (where --user installs land). We prepend this to
# PYTHONPATH so a --user install overrides anything Isaac ships in its bundled,
# read-only site (the only way a non-root user can win the sys.path race).
USER_SITE="$("${PY[@]}" -c 'import site; print(site.getusersitepackages())' 2>/dev/null | tail -n1)"
if [[ -z "$USER_SITE" ]]; then
  echo "[install_deps] error: could not resolve user site-packages dir." >&2
  exit 2
fi
# Make the override active for THIS run's verification step too.
export PYTHONPATH="$USER_SITE:${PYTHONPATH:-}"

# pip install into the user site (no root / no sudo on Isaac Sim cloud images).
pip_install() {
  "${PY[@]}" -m pip install --user "$@"
}

# --- 1. rsl_rl (pinned) + onnxscript ---
# --force-reinstall --no-deps puts a clean rsl-rl-lib in the user site even when
# an older copy ships in the bundle; --no-deps because its deps (torch/numpy/etc)
# are already satisfied by Isaac Sim and must not be reinstalled. PYTHONPATH
# (set above) then makes this user-site copy win over the bundled one.
echo "[install_deps] (1/3) rsl-rl-lib==$RSL_RL_VERSION + onnxscript"
pip_install --force-reinstall --no-deps "rsl-rl-lib==$RSL_RL_VERSION"
pip_install "onnxscript>=0.5"

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

# --- persist PYTHONPATH so future shells + isaaclab.sh -p see the user-site copies ---
# Without this, a new shell would import the bundled (older) rsl-rl-lib again.
if [[ "${NO_BASHRC:-0}" != "1" ]]; then
  BASHRC="$HOME/.bashrc"
  MARKER="# booster_train: user-site ahead of Isaac Sim bundled site"
  if ! grep -qsF "$MARKER" "$BASHRC" 2>/dev/null; then
    {
      echo ""
      echo "$MARKER"
      echo "export PYTHONPATH=\"$USER_SITE:\$PYTHONPATH\""
    } >> "$BASHRC"
    echo "[install_deps] appended PYTHONPATH export to $BASHRC"
  else
    echo "[install_deps] PYTHONPATH export already present in $BASHRC"
  fi
  echo "[install_deps] NOTE: run 'source $BASHRC' (or open a new shell) for it to take effect."
fi

# --- verify ---
# PYTHONPATH is exported above, so this run already sees the user-site packages.
echo "[install_deps] done — verifying with check_env.sh"
"$REPO_ROOT/check_env.sh" || true
