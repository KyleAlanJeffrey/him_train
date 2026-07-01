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

### Training

Run `scripts/rsl_rl/train.py`. If Isaac Lab is **not** on your active conda/venv,
swap `python` for `<IsaacLab>/isaaclab.sh -p`.

Parameters (mix and match as needed):

| Flag | Description |
|---|---|
| `--task` | Task id — `T1-crawl-v0`, `Booster-K1-Fight_001-v0`, etc. (required). |
| `--headless` | Run without a local window (servers / Brev). |
| `--device` | Compute device, e.g. `cuda:0`. |
| `--num_envs` | Override the number of parallel envs. |
| `--max_iterations` | Training iterations — when resuming, this many *additional*. |
| `--resume` | Resume from a checkpoint (defaults to the latest run + latest checkpoint). |
| `--load_run` | Run folder to resume from, under `logs/rsl_rl/<experiment>/` (e.g. `2026-06-30_17-33-24`). |
| `--checkpoint` | Checkpoint file within that run (e.g. `model_999.pt`). |
| `--video` | Record periodic clips of env 0 → `logs/rsl_rl/<experiment>/<run>/videos/train/`. |
| `--video_length` | Clip length in **env steps** (~0.02 s each, so 300 ≈ 6 s). |
| `--video_interval` | Steps between clips, in **env steps** = `iterations × num_steps_per_env(24)` (100 iters → `2400`). |
| `--livestream 2` | Stream over WebRTC instead of a window (see below). |

Example — resume a run for 10k more iterations and record a ~6 s clip every ~100 iterations:

```bash
python scripts/rsl_rl/train.py \
    --task=T1-crawl-v0 \
    --headless \
    --resume \
    --load_run=2026-06-30_17-33-24 \
    --checkpoint=model_999.pt \
    --max_iterations 10000 \
    --video \
    --video_length 300 \
    --video_interval 2400
```

Other scripts: `scripts/rsl_rl/play.py` (play a checkpoint and export TorchScript/ONNX
to `.../exported/`), `scripts/experiments/pose_viewer_t1.py` (interactive crawl-pose
editor), and `scripts/replay_npz.py` (replay a motion NPZ) — see `--help` on each.

### Drive the robot with the keyboard (T1 crawl)

Play a trained crawl policy and **steer it yourself** — keys are hold-to-drive
(hold to move at an absolute speed; release returns that axis to 0):

```bash
python scripts/rsl_rl/play_teleop.py \
    --task=T1-crawl-v0 \
    --num_envs=1 \
    --checkpoint=logs/rsl_rl/t1_crawl/<RUN>/model_<N>.pt
```

| Key | Action |
|---|---|
| `W` / `S` | forward / backward |
| `A` / `D` | turn left / right |
| `Q` / `E` | strafe left / right (`lin_vel_y` — disabled during training, so minimal effect) |
| `SPACE` | stop |
| `ESC` | quit |

Tune command magnitudes: `--turn` (yaw-rate for A/D; default 1.5 — higher = steeper
turn, but >1.0 is beyond the trained range) and `--strafe` (Q/E lateral speed).

Omit `--checkpoint` to auto-load the latest run. Keyboard focus needs the Kit
window — it works on a local display but is unreliable over WebRTC/livestream, so
teleop is best on a machine with a real screen. Crawl-only (the command layout is
crawl-specific).

### Viewing the simulation (WebRTC stream)

On the Brev / Isaac Launchable host (VSCode + Kit App Streaming `web-viewer`)
there is no local window. Append `--livestream 2` to stream the running app over
WebRTC, then watch it in the streaming viewer tab.

```bash
# stream a single robot playing a policy (recommended for viewing)
python scripts/rsl_rl/play.py --task=T1-crawl-v0 --num_envs=1 --livestream 2
```

- Use `--livestream 2` (WebRTC) — **not** `--headless`; livestream runs without a
  local window and starts the stream the viewer connects to. (Equivalent: `LIVESTREAM=2` in the env.)
- Open the viewer in a **new browser tab** at your VSCode/shareable URL with
  `/viewer` appended (e.g. `https://<host>/viewer`). Keep **only one** viewer tab open.
- Start the app **first** and wait for `Simulation App Startup Complete` in the
  console, *then* open (or refresh) the `/viewer` tab. First launch is slow while
  shaders cache; subsequent runs just need a refresh. Stop with `Ctrl+C`.
- You can add `--livestream 2` to `train.py` to watch training, but it is heavy
  (thousands of envs) and slows training significantly — prefer `play.py` for viewing.

### TensorBoard

Training writes TensorBoard logs to `logs/rsl_rl/<experiment>/<run>/` (the default
rsl_rl logger). From a VSCode terminal, point TensorBoard at the experiment (or
`logs/rsl_rl` to see all runs):

```bash
python -m tensorboard.main --logdir logs/rsl_rl --port 6006
# T1 crawl specifically:
python -m tensorboard.main --logdir logs/rsl_rl/t1_crawl --port 6006
```

If TensorBoard isn't installed in the Isaac Lab python: `python -m pip install --user tensorboard`.

Unlike the `/viewer` stream, port 6006 is **not** one of the externally-opened
Brev ports, so you can't just hit `<host>:6006`. Two ways to view it:

- **VSCode's built-in TensorBoard / port forwarding** — the easiest: open the
  command palette → *Python: Launch TensorBoard*, or use the **Ports** panel to
  forward 6006, then click the forwarded address. VSCode tunnels it for you.
- **From your local machine via the Brev CLI** — `brev port-forward <instance> -p 6006:6006`, then open `http://localhost:6006` locally.

(`http://localhost:6006` typed directly into the in-browser VSCode Simple Browser
won't work — that "localhost" is *your* machine, not the instance.)

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