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

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
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


def check_importable(module: str, name: str, required: bool, version_dists: tuple[str, ...] = ()) -> bool:
    """Probe a module via find_spec (no import / no side effects)."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        record(FAIL if required else WARN, name, "not found" + ("" if required else " (optional)"))
        return False
    ver = _version(*version_dists) if version_dists else None
    record(OK, name, f"v{ver}" if ver else "found")
    return True


def check_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        record(FAIL, "torch", "not found")
        return
    import torch  # safe, light

    cuda = torch.cuda.is_available()
    detail = f"v{torch.__version__}, CUDA {'available' if cuda else 'NOT available'}"
    if cuda:
        detail += f" ({torch.cuda.get_device_name(0)})"
    record(OK if cuda else WARN, "torch", detail)
    if not cuda:
        record(WARN, "  └─ GPU", "training/sim needs CUDA; CPU-only will not work for Isaac Sim")


def check_rsl_rl() -> None:
    if importlib.util.find_spec("rsl_rl") is None:
        record(FAIL, "rsl_rl >= 5.0", "not found")
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
    record(status, "rsl_rl >= 5.0", f"v{ver}")


def check_booster_assets() -> None:
    if importlib.util.find_spec("booster_assets") is None:
        record(FAIL, "booster_assets", "not found (external data package)")
        return
    try:
        from booster_assets import BOOSTER_ASSETS_DIR
    except Exception as e:  # noqa: BLE001
        record(FAIL, "booster_assets", f"import failed: {e}")
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
        record(FAIL, "booster_train", "not installed (run: pip install -e source/booster_train)")
        return
    loc = spec.origin or (spec.submodule_search_locations[0] if spec.submodule_search_locations else "?")
    record(OK, "booster_train", loc)


def main() -> int:
    print("\n=== booster_train environment check ===\n")
    print("Python / core:")
    check_python()
    check_importable("numpy", "numpy", required=True, version_dists=("numpy",))
    check_torch()
    check_importable("gymnasium", "gymnasium", required=True, version_dists=("gymnasium",))
    check_rsl_rl()

    print("\nIsaac stack (probed, not launched):")
    check_importable("isaacsim", "isaacsim", required=True)
    check_importable("isaaclab", "isaaclab", required=True)
    check_importable("isaaclab_tasks", "isaaclab_tasks", required=True)

    print("\nProject package + data:")
    check_booster_train_editable()
    check_booster_assets()

    print("\nBuild/runtime extras:")
    check_importable("psutil", "psutil", required=False, version_dists=("psutil",))
    check_importable("toml", "toml", required=False, version_dists=("toml",))

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print("\n" + "=" * 48)
    if n_fail:
        print(f"\033[31mFAILED\033[0m: {n_fail} required check(s) failed, {n_warn} warning(s).")
        print("Fix the ✗ items above before training.")
    elif n_warn:
        print(f"\033[33mOK with warnings\033[0m: {n_warn} warning(s) — review the ! items.")
    else:
        print("\033[32mAll checks passed.\033[0m Environment is ready to train.")
    print("=" * 48 + "\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
