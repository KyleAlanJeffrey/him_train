"""Check whether the T1 crawl robot spawns penetrating the ground.

Builds the real T1-crawl-v0 env (using whatever CRAWL_FACING / pose is configured
in constants.py), resets, and reports the lowest collision-surface point of each
contact link and the lowest link origin overall. A near-instant `flipped`
termination with a violent spin almost always means a contact body spawns below
z=0 and the solver ejects the robot.

Usage (him_train repo root, Isaac Lab env active):
    python scripts/experiments/check_spawn_t1.py
    python scripts/experiments/check_spawn_t1.py --num_envs 256
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="T1 crawl spawn-penetration check.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of envs to sample.")
parser.add_argument("--task", type=str, default="T1-crawl-v0", help="Registered task name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply

import booster_train.tasks  # noqa: F401  (registers T1-crawl-v0)
from booster_train.tasks.manager_based.t1_crawl.t1_crawl_env_cfg import T1CrawlEnvCfg

# Contact-link collision boxes (center, half-extents) in the link frame, from the
# T1 URDF <collision> tags. Same approximation used by pose_viewer_t1.py.
CONTACT_BOXES = {
    # Hands: the ball end-effector sphere (r=0.035 at y=±0.2115), not the forearm cylinder.
    "left_hand_link": ((0.0, 0.2115, 0.0), (0.035, 0.035, 0.035)),
    "right_hand_link": ((0.0, -0.2115, 0.0), (0.035, 0.035, 0.035)),
    "left_foot_link": ((0.0101079, 0.0, -0.0214208), (0.112434, 0.05, 0.021830)),
    "right_foot_link": ((0.0101079, 0.0, -0.0214208), (0.112434, 0.05, 0.021830)),
}


def box_corners(center, half, device):
    c = torch.tensor(center, dtype=torch.float32, device=device)
    h = torch.tensor(half, dtype=torch.float32, device=device)
    signs = torch.tensor(
        [[sx, sy, sz] for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)],
        dtype=torch.float32,
        device=device,
    )
    return c + signs * h  # (8, 3)


def main():
    env_cfg = T1CrawlEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    device = robot.device
    body_names = [str(n) for n in robot.data.body_names]
    origin_z = env.unwrapped.scene.env_origins[:, 2]  # (N,)

    body_pos = robot.data.body_pos_w   # (N, B, 3)
    body_quat = robot.data.body_quat_w  # (N, B, 4)

    print(f"\n=== Spawn check: {args_cli.task}, {robot.num_instances} envs ===")
    print("(clearance = lowest collision surface above ground; negative = PENETRATING)\n")

    worst = {}
    for name, (center, half) in CONTACT_BOXES.items():
        if name not in body_names:
            continue
        bi = body_names.index(name)
        corners = box_corners(center, half, device)                 # (8,3)
        q = body_quat[:, bi, :].unsqueeze(1).expand(-1, 8, 4)        # (N,8,4)
        world = quat_apply(q, corners.unsqueeze(0).expand(robot.num_instances, 8, 3))
        world = world + body_pos[:, bi, :].unsqueeze(1)              # (N,8,3)
        clearance = world[:, :, 2].min(dim=1).values - origin_z      # (N,)
        worst[name] = (clearance.min().item(), clearance.mean().item())

    for name, (mn, mean) in worst.items():
        flag = "  << PENETRATING" if mn < 0 else ""
        print(f"  {name:18s}: min clearance={mn:+.4f}  mean={mean:+.4f}{flag}")

    # Coarse all-body check on link origins (catches a non-contact link spawning low).
    all_clear = (body_pos[:, :, 2] - origin_z.unsqueeze(1))          # (N,B)
    lowest_val, lowest_idx = all_clear.min(dim=1)
    gi = lowest_val.argmin().item()
    print(
        f"\n  lowest link ORIGIN over all bodies: {body_names[lowest_idx[gi].item()]} "
        f"at clearance {lowest_val[gi].item():+.4f} (origin, not surface)"
    )
    overall = min(mn for mn, _ in worst.values())
    verdict = "PENETRATING — raise the facing height" if overall < 0 else "clear of the ground"
    print(f"\n  VERDICT: lowest contact surface = {overall:+.4f}  ->  {verdict}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
