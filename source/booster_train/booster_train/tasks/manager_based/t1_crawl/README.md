# T1 Crawl Task

Teaches the Booster T1 to crawl on all four limbs (face-down quadruped stance),
ported from the original `g1_crawl` task (Unitree G1).

Train it with:

```bash
python scripts/rsl_rl/train.py --task T1-crawl-v0 --headless
```

---

## Project structure

This task lives inside the `him_train` repo (a fork of `booster_train`). The
pieces relevant to crawling:

```
him_train/
├── assets/
│   └── t1-crawl-pose.json          # the crawl pose you edit + tune (see below)
├── scripts/
│   ├── rsl_rl/train.py             # training entry point (--task T1-crawl-v0)
│   ├── rsl_rl/play.py              # replay a trained checkpoint
│   └── experiments/
│       └── pose_viewer_t1.py       # interactive pose editor (see below)
└── source/booster_train/booster_train/
    ├── assets/robots/booster.py    # BOOSTER_T1_CFG — the T1 articulation (joints, actuators, URDF)
    └── tasks/manager_based/
        ├── __init__.py             # imports t1_crawl so the gym env registers
        └── t1_crawl/               # <-- this task
            ├── __init__.py         # registers gym id "T1-crawl-v0" -> env cfg + PPO cfg
            ├── t1_crawl_env_cfg.py # THE env: scene, the crawl init pose, rewards, events, terminations
            ├── README.md           # this file
            ├── agents/
            │   ├── rsl_rl_ppo_cfg.py   # PPO hyperparameters (rsl_rl 5.x API)
            │   └── symmetry_func.py    # left-right mirror augmentation (joint-order dependent!)
            └── mdp/
                ├── command.py          # CrawlVelocityCommand (tracks vz, vy, ang_vel_x)
                ├── rewards.py          # crawl-specific reward terms
                ├── events.py           # reset randomization
                ├── observations.py     # observation terms
                └── curriculums.py      # curriculum terms
```

**How a task gets built**, top to bottom: `train.py` looks up the gym id
`T1-crawl-v0`, which `t1_crawl/__init__.py` maps to `T1CrawlEnvCfg` (the world +
rewards) and `PPORunnerCfg` (the learning algorithm). `T1CrawlEnvCfg` spawns the
robot from `T1_CRAWL_CFG`, which is just `BOOSTER_T1_CFG` with the crawl init
pose swapped in.

---

## The crawl pose, and how to create/tune it

The single most important thing to get right is the **initial pose**: the robot
must start in a stable face-down quadruped stance with all four limbs resting on
the ground. If the pose floats above the floor or self-intersects, training will
struggle. The angles currently in the code are a hand estimate and **need to be
tuned in sim**.

The pose is defined in two equivalent places:

1. `assets/t1-crawl-pose.json` — the editable source you tweak interactively.
2. The `T1_CRAWL_CFG` `init_state` block in
   [`t1_crawl_env_cfg.py`](./t1_crawl_env_cfg.py) — what training actually uses.

The workflow is: **edit the JSON in the viewer → export → paste back into the env cfg.**

### Step 1 — open the interactive editor

From the repo root with `env_isaaclab` activated:

```bash
python scripts/experiments/pose_viewer_t1.py --pose assets/t1-crawl-pose.json
```

This spawns one T1 with **gravity disabled** and a **free root**, applies the
pose, and prints the joint order and body heights. Gravity is off so the robot
holds the pose instead of collapsing while you edit.

### Step 2 — tweak joints until the stance looks right

| Key            | Action                                              |
|----------------|-----------------------------------------------------|
| `UP` / `DOWN`  | select which joint to edit (cycles through all 23)  |
| `LEFT`/`RIGHT` | decrease / increase the selected joint by 0.05 rad  |
| `[` / `]`      | lower / raise the base spawn height (z) by 0.01 m   |
| `H`            | print Trunk / hand / foot heights                   |
| `R`            | reset to the JSON values (discard edits)            |
| `P`            | print the current pose as JSON + an env-cfg snippet |
| `ESC`          | exit                                                |

Aim for: hands (`left/right_hand_link`) and feet (`left/right_foot_link`) just
touching the ground (z ≈ 0), Trunk a bit above, no limbs passing through the
floor or each other. Use `H` to check the heights as you go, and `[`/`]` to set
the base height so the contact links sit at z ≈ 0.

### Step 3 — export and copy the pose back into the code

Press `P`. It prints two things:

1. A full **JSON block** — paste it over the contents of
   `assets/t1-crawl-pose.json` so your edits are saved for next time.
2. A **`joint_pos` snippet** formatted for Python — paste the joint angles into
   the `T1_CRAWL_CFG` `init_state` in
   [`t1_crawl_env_cfg.py`](./t1_crawl_env_cfg.py), and update `pos=(0.0, 0.0, z)`
   with the base height you settled on (the `base_z` the viewer reports).

> The env cfg uses regex joint names (e.g. `".*_Hip_Pitch"` sets both legs at
> once). If your left and right angles end up different, replace the regex with
> explicit `"Left_..."` / `"Right_..."` entries.

### Step 4 — sanity-check the joint order (one-time)

The viewer prints the full joint order on startup. **Compare it against the
order assumed in [`agents/symmetry_func.py`](./agents/symmetry_func.py).** The
left-right mirror augmentation hard-codes joint indices; if the real order
differs, the swap/invert index lists there must be fixed or symmetry training
will corrupt the data. This only needs checking once.

---

## After tuning: train

```bash
python scripts/rsl_rl/train.py --task T1-crawl-v0 --headless
```

Then watch an early checkpoint to confirm the robot starts in the pose you set:

```bash
python scripts/rsl_rl/play.py --task T1-crawl-v0 --num_envs 16
```
