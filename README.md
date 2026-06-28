# Booster RL Tasks

## Overview

This repository provides a set of reinforcement learning tasks for Booster robots (K1, 22 DOF; T1, 23 DOF) using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html).
It includes:

- **BeyondMimic motion tracking** (K1) — the [BeyondMimic motion tracking](https://github.com/HybridRobotics/whole_body_tracking) framework adapted to Booster K1, tracking reference motions loaded from NPZ.
- **T1 crawl** — quadruped-style crawl locomotion for the Booster T1 (body face-down, all four limbs on the ground).

Policies are trained with [rsl_rl](https://github.com/leggedrobotics/rsl_rl) PPO and exported to TorchScript/ONNX for deployment.
This repository follows the standard Isaac Lab project structure, and is tested with IsaacLab 2.2 and Isaac Sim 5.0.

### Available tasks

| Task ID | Robot | Description |
|---|---|---|
| `T1-crawl-v0` | T1 | Crawl locomotion (velocity-tracking on all fours) |
| `Booster-K1-Fight_001-v0` (+ `-Play`) | K1 | Fight motion tracking |
| `Booster-K1-MJ_Dance_002-v0` (+ `-Play`) | K1 | Dance motion tracking |
| `Booster-K1-MJ_Dance_004-v0` | K1 | Dance motion tracking |

Run `python scripts/list_envs.py` to list the registered tasks.

## Requirements

| Component | Version / notes |
|---|---|
| OS | Linux (Ubuntu 22.04 recommended) |
| GPU | NVIDIA GPU with CUDA — **required** by Isaac Sim (no CPU-only mode) |
| Python | >= 3.10 |
| Isaac Sim | 5.0 |
| Isaac Lab | 2.2 |
| rsl_rl | >= 5.0 |
| PyTorch | provided by the Isaac Lab environment (CUDA build) |

### Required repositories

Clone these **outside** the `IsaacLab` directory (except Isaac Lab itself):

| Repository | Provides | Required? |
|---|---|---|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | Simulation + RL framework (`isaaclab`, `isaaclab_tasks`, Isaac Sim) | Yes |
| [booster_train](https://github.com/BoosterRobotics/booster_train) (this repo) | Tasks, configs, and training/play scripts | Yes |
| [booster_assets](https://github.com/BoosterRobotics/booster_assets) | Robot models (URDFs) + motion data; provides `BOOSTER_ASSETS_DIR` | Yes |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | PPO trainer (installed into the Isaac Lab env) | Yes |
| [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) | Run exported policies in MuJoCo / on real robots | Optional (deploy only) |

Two helper scripts (both auto-detect the Isaac Lab python — active env, else `isaaclab.sh -p`):

```bash
./install_deps.sh    # install rsl_rl (pinned), onnxscript, booster_assets + booster_train (editable)
./check_env.sh       # verify everything is present
```

- `install_deps.sh` installs the project dependencies and auto-retries with `--user` on read-only sites (common on Isaac Sim cloud images). Set `BOOSTER_ASSETS_PATH=/path/to/booster_assets` if it isn't auto-found.
- `check_env.sh` checks the Python version, the Isaac/rsl_rl/torch stack (incl. CUDA), the `booster_train` install, and that the `booster_assets` URDFs the tasks load are present. Run it from the Isaac Lab python directly via `python scripts/check_env.py`.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    ```bash
    git clone https://github.com/BoosterRobotics/booster_train.git
    ```

- Download and install booster_assets:
   - Clone the [booster_assets](https://github.com/BoosterRobotics/booster_assets) which contains Booster robot models and motion data.
   - Install booster_assets python helper following the instructions in the repository.

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/booster_train
    ```

- Prepare BeyondMimic motion data:
    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/csv_to_npz.py --headless --input_file=<PATH_TO_BOOSTER_ASSETS>/motions/K1/<MOTION>.csv --input_fps=<FPS> --output_name=<PATH_TO_BOOSTER_ASSETS>/motions/K1/<MOTION>.npz
    ```

## Usage

### Quick commands

Copy-paste ready (replace `cuda:0` and checkpoint paths as needed). If Isaac Lab
is **not** on your active conda/venv, swap `python` for `<IsaacLab>/isaaclab.sh -p`.

**Train**

```bash
# T1 crawl
python scripts/rsl_rl/train.py --task=T1-crawl-v0 --headless --device cuda:0

# K1 motion tracking
python scripts/rsl_rl/train.py --task=Booster-K1-Fight_001-v0 --headless --device cuda:0
```

**Play + export policy** (writes TorchScript/ONNX to `logs/rsl_rl/<experiment>/<run>/exported/`)

```bash
# Latest run is auto-resolved if you omit --checkpoint
python scripts/rsl_rl/play.py --task=T1-crawl-v0 --num_envs=1

# Or point at a specific checkpoint
python scripts/rsl_rl/play.py --task=T1-crawl-v0 --checkpoint=logs/rsl_rl/t1_crawl/<RUN>/model_<N>.pt
```

**Pose editor** (interactive T1 crawl-pose tool — tweak joints, then `P` prints JSON to paste into the env cfg)

```bash
python scripts/experiments/pose_viewer_t1.py
python scripts/experiments/pose_viewer_t1.py --pose assets/t1-crawl-pose.json
```

**Replay a motion NPZ in sim** (direct file path, or pull from a wandb registry)

```bash
python scripts/replay_npz.py --motion <PATH_TO_BOOSTER_ASSETS>/motions/K1/<MOTION>.npz
python scripts/replay_npz.py --registry_name <WANDB_REGISTRY_NAME>
```

---

- Listing the available tasks:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/list_envs.py
    ```

- Running a task:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/train.py --task=<TASK_NAME> --headless --device cuda:N
    ```

- Play a trained policy and export it for deployment:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/play.py --task=<TASK_NAME> --checkpoint=<CHECKPOINT_PATH>
    ```

    This script also exports the trained policy to a TorchScript/ONNX file for deployment on real robots in `logs/rsl_rl/<EXPERIMENT>/<RUN>/exported/`.

## Deploy

After a model has been trained and exported, you can deploy the trained policy in MuJoCo or on real Booster robots using the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository. For more details, please refer to the instructions in the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository.


## Acknowledgements

- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): the motion tracking training in BeyondMimic, which is a versatile humanoid control framework that provides highly dynamic motion tracking.