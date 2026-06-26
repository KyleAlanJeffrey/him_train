# CLAUDE.md

Guidance for working in this repository. Read this before making changes.

## What this is

`booster_train` is an [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) reinforcement-learning
project for **Booster humanoid robots** (K1, 22 DOF; T1, 23 DOF). It trains
policies with [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (>= 5.0) PPO and
exports them to TorchScript/ONNX for deployment via
[booster_deploy](https://github.com/BoosterRobotics/booster_deploy).

Tested with **Isaac Lab 2.2** and **Isaac Sim 5.0**. Python >= 3.10.

Two families of tasks live here:
- **BeyondMimic motion tracking** (K1) — adapted from
  [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking);
  tracks reference motions loaded from NPZ.
- **T1 crawl** — quadruped-style crawl locomotion for the T1 (the newest task,
  ported from a Unitree `g1_crawl` task).

## Layout

```
source/booster_train/booster_train/      # the installable package (pip install -e)
├── assets/robots/
│   ├── booster.py        # BOOSTER_K1_CFG, BOOSTER_T1_CFG (URDF spawn + actuators + init pose)
│   └── actuator.py       # DelayedImplicitActuator, BoosterDelayedPDActuatorCfg, per-motor joint models
└── tasks/
    ├── __init__.py       # auto-registers all sub-package gym envs via import_packages
    └── manager_based/
        ├── t1_crawl/                 # T1-crawl-v0
        │   ├── t1_crawl_env_cfg.py   # T1CrawlEnvCfg + scene/commands/actions/obs/rewards/events
        │   ├── agents/rsl_rl_ppo_cfg.py   # PPORunnerCfg (experiment "t1_crawl")
        │   ├── agents/symmetry_func.py    # left/right mirror augmentation (23-DOF pairing)
        │   └── mdp/                  # crawl-specific command/rewards/observations/curriculums
        └── beyond_mimic/             # K1 motion-tracking tasks
            ├── mdp/                  # MotionCommand + motion-error rewards/terminations
            └── robots/k1/            # one folder per motion: fight_001, mj_dance_002, mj_dance_004

scripts/
├── rsl_rl/train.py       # main training entry (Hydra + rsl_rl)
├── rsl_rl/play.py        # playback a checkpoint + export policy to TorchScript/ONNX
├── rsl_rl/cli_args.py    # shared CLI args (add_rsl_rl_args)
├── csv_to_npz.py         # convert motion CSV (LAFAN, ~30fps) -> NPZ (50fps) for BeyondMimic
├── replay_npz.py         # replay a motion NPZ in sim
├── list_envs.py          # print all registered "Booster-"/task envs
└── experiments/          # ad-hoc viz/debug utilities (e.g. pose_viewer_t1.py)

assets/                   # repo-root non-package assets (e.g. t1-crawl-pose.json)
logs/, outputs/           # rsl_rl run outputs (gitignored)
```

Robot models and motion data come from the external
[booster_assets](https://github.com/BoosterRobotics/booster_assets) package
(`BOOSTER_ASSETS_DIR`), installed separately — they are **not** in this repo.

## Registered tasks

- `T1-crawl-v0` — T1 crawl locomotion
- `Booster-K1-Fight_001-v0` (+ `-Play`) — K1 fight motion tracking
- `Booster-K1-MJ_Dance_002-v0` (+ `-Play`) — K1 dance motion tracking
- `Booster-K1-MJ_Dance_004-v0` — K1 dance motion tracking

Each task's `__init__.py` calls `gym.register(...)` with an `env_cfg_entry_point`
and an `rsl_rl_cfg_entry_point`. `tasks/__init__.py` uses Isaac Lab's
`import_packages` to walk and import every sub-package, so a new task is picked up
just by adding its folder with a registering `__init__.py`.

## Common commands

Use `python` only if Isaac Lab is on the active conda/venv; otherwise call
`FULL_PATH_TO_isaaclab.sh -p` in its place.

```bash
python -m pip install -e source/booster_train        # install package (editable)
python scripts/list_envs.py                            # list tasks
python scripts/rsl_rl/train.py --task=T1-crawl-v0 --headless --device cuda:0
python scripts/rsl_rl/play.py  --task=T1-crawl-v0 --checkpoint=<PATH>   # + exports policy
```

Exported policies land in `logs/rsl_rl/<experiment>/<run>/exported/`.

## Adding a task

1. Create a folder under `tasks/manager_based/` (locomotion) or
   `tasks/manager_based/beyond_mimic/robots/<robot>/<motion>/` (motion tracking).
2. Add `<task>_env_cfg.py` (a `ManagerBasedRLEnvCfg` subclass) and
   `agents/rsl_rl_ppo_cfg.py` (an `RslRlOnPolicyRunnerCfg` subclass).
3. In the folder's `__init__.py`, `gym.register(...)` pointing at both configs.
   No central registry edit is needed — `import_packages` finds it.

## Conventions & gotchas

- **Robot configs** (`assets/robots/booster.py`) use `BoosterDelayedPDActuatorCfg`
  to model command delay (`min_delay`/`max_delay` in sim steps) and per-motor
  dynamics (`BoosterJointE*` natural-freq/damping models). Keep init poses within
  URDF joint limits — `soft_joint_pos_limit_factor=0.9` clamps to 90% of range.
- **T1 crawl orientation**: the body is face-down — init `rot=(0.7071,0,0.7071,0)`
  (90° about world Y), so body **+X points world-down** and body **+Z points
  world-forward**. Consequently `projected_gravity_b ≈ [1,0,0]`, and "forward"
  velocity commands act on body Z while turning is roll about body X (which is
  world yaw). This is why crawl rewards use `track_lin_vel_yz_base_exp`,
  `track_ang_vel_z_world_exp`, and `align_projected_gravity_plus_x_l2`. Don't
  "fix" these to the usual upright-locomotion terms.
- **Obs dims (T1 crawl)**: policy = 75 (gravity 3 + cmd 3 + qpos 23 + qvel 23 +
  action 23), critic = 81 (adds base lin_vel 3 + ang_vel 3). The symmetry
  augmentation in `symmetry_func.py` hardcodes these layouts and the 23-joint
  left/right pairing — if you change observations, actions, or joint order, update
  it too or mirror loss/augmentation will be wrong.
- **`init_state` z heights** are estimates flagged in comments (e.g. T1 crawl
  `z=0.28`); expect to tune after a first sim run.
- Lint/format via pre-commit (`.pre-commit-config.yaml`, flake8). Code generally
  follows Isaac Lab conventions and the BSD-3 headers on Isaac-Lab-derived files.

## Where to look first

- Task behavior → that task's `*_env_cfg.py` (rewards, commands, events, scene).
- Robot/actuator behavior → `assets/robots/booster.py` and `actuator.py`.
- Training hyperparameters → the task's `agents/rsl_rl_ppo_cfg.py`.
