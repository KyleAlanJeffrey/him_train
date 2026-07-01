# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained T1-crawl policy and DRIVE it with the keyboard.

Loads a checkpoint like play.py, but instead of auto-generated random velocity
commands you steer with the keyboard. It works by overwriting the command slice
of the policy observation each step (policy obs layout is gravity(3) + cmd(3) +
qpos(23) + qvel(23) + action(23) = 75, so the command is indices 3:6 =
[lin_vel_z (forward), lin_vel_y (lateral), ang_vel_x (turn)] in the crawl frame).

Controls (HOLD to drive — releasing a key returns that axis to 0):
    W / S   forward / backward   (lin_vel_z, absolute)
    A / D   turn left / right    (ang_vel_x, absolute)
    Q / E   strafe left / right  (lin_vel_y; disabled during training)
    SPACE   stop (release all)
    ESC     quit

Usage (crawl only):
    python scripts/rsl_rl/play_teleop.py --task=T1-crawl-v0 --num_envs=1
    python scripts/rsl_rl/play_teleop.py --task=T1-crawl-v0 --checkpoint=<PATH>
Add --livestream 2 on a headless/Brev host (keyboard focus may be limited there).
"""

import argparse
from importlib.metadata import version
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Keyboard-teleop play for T1 crawl.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 recommended for teleop).")
parser.add_argument("--task", type=str, default="T1-crawl-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--strafe", type=float, default=0.5, help="Absolute lateral speed for Q/E (untrained axis).")
parser.add_argument("--turn", type=float, default=1.5, help="Absolute turn-rate magnitude (A/D). Higher = steeper turn; >1.0 is beyond the trained range.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import omni
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_train.tasks  # noqa: F401
from booster_train.tasks.manager_based.t1_crawl import constants as c

# Policy obs layout: gravity(3) | velocity_commands(3) | qpos | qvel | actions.
CMD_SLICE = slice(3, 6)  # [lin_vel_z, lin_vel_y, ang_vel_x]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    device = env.unwrapped.device

    # Absolute hold-to-drive: while a key is held the command is a fixed value;
    # releasing it returns that axis to 0. Magnitudes are the training-range
    # extremes so commands stay in-distribution.
    FWD, REV = c.CMD_LIN_VEL_Z[1], c.CMD_LIN_VEL_Z[0]   # forward / reverse speed
    TURN = args_cli.turn                                 # turn-rate magnitude (steeper = higher); symmetric
    STRAFE = args_cli.strafe                             # lateral speed (untrained axis)
    command = torch.zeros(3, device=device)  # [vz, vy, wx]
    pressed: set[str] = set()

    def recompute():
        vz = FWD if pressed & {"W", "UP"} else (REV if pressed & {"S", "DOWN"} else 0.0)
        wx = TURN if pressed & {"A", "LEFT"} else (-TURN if pressed & {"D", "RIGHT"} else 0.0)
        vy = STRAFE if "Q" in pressed else (-STRAFE if "E" in pressed else 0.0)
        command[0], command[1], command[2] = vz, vy, wx
        print(f"[cmd] vz(fwd)={command[0]:+.2f}  vy(lat)={command[1]:+.2f}  wx(turn)={command[2]:+.2f}")

    def on_key(event):
        # Read event.input.name ONLY for key press/release — carb also emits CHAR
        # events where event.input is a plain string (no .name) and would crash.
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            k = event.input.name
            if k == "ESCAPE":
                should_exit["v"] = True
                return
            if k == "SPACE":
                pressed.clear()
            else:
                pressed.add(k)
            recompute()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed.discard(event.input.name)
            recompute()

    input_iface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    should_exit = {"v": False}
    sub = input_iface.subscribe_to_keyboard_events(keyboard, on_key)

    # best-effort: point the debug-vis arrow at the manual command too
    try:
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    except Exception:  # noqa: BLE001
        cmd_term = None

    print("\n[teleop] W/S forward · A/D turn · Q/E strafe · SPACE stop · ESC quit\n")

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs
    dt = env.unwrapped.step_dt
    try:
        while simulation_app.is_running() and not should_exit["v"]:
            start = time.time()
            with torch.inference_mode():
                obs[:, CMD_SLICE] = command            # drive the policy
                if cmd_term is not None:
                    cmd_term.vel_command_b[:] = command  # keep the viz arrow in sync
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            sleep = dt - (time.time() - start)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        input_iface.unsubscribe_to_keyboard_events(keyboard, sub)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
