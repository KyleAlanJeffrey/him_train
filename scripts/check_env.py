#!/usr/bin/env python3
"""Verify the environment has everything needed to train/play booster_train tasks.

Runs WITHOUT launching Isaac Sim — heavy modules (isaacsim/isaaclab) are probed
with importlib.find_spec so this finishes in seconds. Checks Python version, the
RL/sim stack, the local package, and that the external booster_assets data
(URDFs the tasks load) is actually on disk.

Usage:
    python scripts/check_env.py
Exit code 0 if all required checks pass, 1 otherwise (handy for CI / Makefiles).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys

OK, WARN, FAIL = "OK", "WARN", "FAIL"
SYMBOL = {OK: "\033[32m✓\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✗\033[0m"}

results: list[tuple[str, str, str, str]] = []

# Recommended fix commands (printed in the summary for whatever failed).
FIX_INSTALL_DEPS = "./install_deps.sh   (installs rsl_rl, booster_assets, booster_train)"
FIX_ISAACLAB = "install Isaac Lab: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
FIX_TORCH_BROKEN = "Isaac Sim torch prebundle is corrupted — see README 'Troubleshooting' (or use a fresh image)"


def record(status: str, name: str, detail: str = "", fix: str = "") -> None:
    results.append((status, name, detail, fix))
    print(f"  {SYMBOL[status]} {name:28s} {detail}")


def _version(*dist_names: str) -> str | None:
    for d in dist_names:
        try:
            return importlib.metadata.version(d)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def check_python() -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    record(OK if (v.major, v.minor) >= (3, 10) else FAIL, "Python >= 3.10", detail)


def check_importable(
    module: str, name: str, required: bool, version_dists: tuple[str, ...] = (), fix: str = ""
) -> bool:
    """Probe a module via find_spec (no import / no side effects)."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        record(FAIL if required else WARN, name, "not found" + ("" if required else " (optional)"), fix=fix)
        return False
    ver = _version(*version_dists) if version_dists else None
    record(OK, name, f"v{ver}" if ver else "found")
    return True


def check_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        record(FAIL, "torch", "not found")
        return
    try:
        import torch
    except Exception as e:  # noqa: BLE001  (broken install, not just absent)
        record(FAIL, "torch", f"installed but FAILS to import: {type(e).__name__}: {e}", fix=FIX_TORCH_BROKEN)
        return
    try:
        cuda = torch.cuda.is_available()
    except Exception as e:  # noqa: BLE001
        record(WARN, "torch", f"v{getattr(torch, '__version__', '?')} (CUDA query failed: {e})")
        return
    detail = f"v{torch.__version__}, CUDA {'available' if cuda else 'NOT available'}"
    if cuda:
        detail += f" ({torch.cuda.get_device_name(0)})"
    record(OK if cuda else WARN, "torch", detail)
    if not cuda:
        record(WARN, "  └─ GPU", "training/sim needs CUDA; CPU-only will not work for Isaac Sim")


RSL_RL_INSTALL = "pip install rsl-rl-lib==5.0.1 onnxscript>=0.5"


def check_rsl_rl() -> None:
    # PyPI name is "rsl-rl-lib" (import name "rsl_rl"); Isaac Lab 2.2 pins 5.0.1.
    if importlib.util.find_spec("rsl_rl") is None:
        record(FAIL, "rsl_rl >= 5.0", "not found", fix=FIX_INSTALL_DEPS)
        return
    ver = _version("rsl-rl-lib", "rsl_rl", "rsl-rl")
    if ver is None:
        record(WARN, "rsl_rl >= 5.0", "found, version unknown")
        return
    try:
        major = int(ver.split(".")[0])
        status = OK if major >= 5 else FAIL
    except ValueError:
        status = WARN
    detail = f"v{ver}" + (" → need >=5.0" if status == FAIL else "")
    record(status, "rsl_rl >= 5.0", detail, fix=FIX_INSTALL_DEPS if status == FAIL else "")


def check_booster_assets() -> None:
    if importlib.util.find_spec("booster_assets") is None:
        record(FAIL, "booster_assets", "not found (external data package)", fix=FIX_INSTALL_DEPS)
        return
    try:
        from booster_assets import BOOSTER_ASSETS_DIR
    except Exception as e:  # noqa: BLE001
        record(FAIL, "booster_assets", f"import failed: {e}", fix=FIX_INSTALL_DEPS)
        return
    if not os.path.isdir(BOOSTER_ASSETS_DIR):
        record(FAIL, "BOOSTER_ASSETS_DIR", f"missing dir: {BOOSTER_ASSETS_DIR}")
        return
    record(OK, "BOOSTER_ASSETS_DIR", BOOSTER_ASSETS_DIR)
    # Files the registered tasks actually load.
    needed = [
        "robots/T1/T1_23dof.urdf",   # T1-crawl-v0
        "robots/K1/K1_22dof.urdf",   # beyond_mimic K1 tasks
    ]
    for rel in needed:
        path = os.path.join(BOOSTER_ASSETS_DIR, rel)
        record(OK if os.path.isfile(path) else WARN, f"  asset: {rel}", "" if os.path.isfile(path) else "MISSING")


def check_booster_train_editable() -> None:
    spec = importlib.util.find_spec("booster_train")
    if spec is None:
        record(FAIL, "booster_train", "not installed", fix=FIX_INSTALL_DEPS)
        return
    loc = spec.origin or (spec.submodule_search_locations[0] if spec.submodule_search_locations else "?")
    record(OK, "booster_train", loc)


def safe(label: str, fn) -> None:
    """Run a check; if it raises, record a FAIL instead of crashing the whole run."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        record(FAIL, label, f"checker errored: {type(e).__name__}: {e}")


def main() -> int:
    print("\n=== booster_train environment check ===\n")
    print("Python / core:")
    safe("Python", check_python)
    safe("numpy", lambda: check_importable("numpy", "numpy", required=True, version_dists=("numpy",), fix=FIX_ISAACLAB))
    safe("torch", check_torch)
    safe("gymnasium", lambda: check_importable(
        "gymnasium", "gymnasium", required=True, version_dists=("gymnasium",), fix=FIX_ISAACLAB))
    safe("rsl_rl >= 5.0", check_rsl_rl)

    print("\nIsaac stack (probed, not launched):")
    safe("isaacsim", lambda: check_importable("isaacsim", "isaacsim", required=True, fix=FIX_ISAACLAB))
    safe("isaaclab", lambda: check_importable("isaaclab", "isaaclab", required=True, fix=FIX_ISAACLAB))
    safe("isaaclab_tasks", lambda: check_importable("isaaclab_tasks", "isaaclab_tasks", required=True, fix=FIX_ISAACLAB))

    print("\nProject package + data:")
    safe("booster_train", check_booster_train_editable)
    safe("booster_assets", check_booster_assets)

    print("\nBuild/runtime extras:")
    safe("psutil", lambda: check_importable("psutil", "psutil", required=False, version_dists=("psutil",)))
    safe("toml", lambda: check_importable("toml", "toml", required=False, version_dists=("toml",)))

    n_fail = sum(1 for s, *_ in results if s == FAIL)
    n_warn = sum(1 for s, *_ in results if s == WARN)
    print("\n" + "=" * 48)
    if n_fail:
        print(f"\033[31mFAILED\033[0m: {n_fail} required check(s) failed, {n_warn} warning(s).")
        # Recommend fixes — dedup, install_deps.sh first since it covers the common ones.
        fixes = []
        for status, _, _, fix in results:
            if status == FAIL and fix and fix not in fixes:
                fixes.append(fix)
        fixes.sort(key=lambda f: f != FIX_INSTALL_DEPS)  # install_deps.sh first
        if fixes:
            print("\nRecommended:")
            for f in fixes:
                print(f"  • {f}")
    elif n_warn:
        print(f"\033[33mOK with warnings\033[0m: {n_warn} warning(s) — review the ! items.")
    else:
        print("\033[32mAll checks passed.\033[0m Environment is ready to train.")
    print("=" * 48 + "\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
