"""Interactive crawl-pose editor for the Booster T1.

Spawns a single T1 with gravity disabled and a free root, applies a pose from a
JSON file, and lets you tweak joints with the keyboard until the robot sits in a
stable face-down quadruped stance. Then it prints the Trunk height and exports
the final joint angles as JSON to paste into t1_crawl_env_cfg.py.

Usage (from the him_train repo root, env_isaaclab activated):
    python scripts/experiments/pose_viewer_t1.py
    python scripts/experiments/pose_viewer_t1.py --pose assets/t1-crawl-pose.json

Controls:
    UP / DOWN        cycle which joint is selected
    LEFT / RIGHT     decrease / increase the selected joint by the step (0.05 rad)
    [ / ]            decrease / increase the base spawn height (z) by 0.01 m
    R                reapply the pose from JSON (discard edits)
    P                print the current pose as JSON (copy this into the env cfg)
    H                print Trunk / hand / foot heights
    ESC              exit
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive T1 crawl-pose editor.")
parser.add_argument("--pose", type=str, default="assets/t1-crawl-pose.json", help="Path to the pose JSON file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json

import carb
import omni
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from booster_train.assets.robots.booster import BOOSTER_T1_CFG


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Convert intrinsic XYZ roll-pitch-yaw (radians) to a (w, x, y, z) quaternion."""
    hr, hp, hy = 0.5 * roll, 0.5 * pitch, 0.5 * yaw
    cr, sr = torch.cos(torch.tensor(hr)), torch.sin(torch.tensor(hr))
    cp, sp = torch.cos(torch.tensor(hp)), torch.sin(torch.tensor(hp))
    cy, sy = torch.cos(torch.tensor(hy)), torch.sin(torch.tensor(hy))
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = torch.stack([w, x, y, z]).to(dtype=torch.float32)
    return quat / torch.linalg.norm(quat)


def create_scene_cfg():
    class SimpleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )
        Robot = BOOSTER_T1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=BOOSTER_T1_CFG.spawn.replace(
                rigid_props=BOOSTER_T1_CFG.spawn.rigid_props.replace(disable_gravity=True),
                articulation_props=BOOSTER_T1_CFG.spawn.articulation_props.replace(fix_root_link=False),
            ),
        )

    return SimpleSceneCfg


def load_pose(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pose = data["poses"][0]
    for key in ("base_pos", "base_rpy", "joints"):
        if key not in pose:
            raise KeyError(f"Pose is missing required key '{key}'")
    return pose


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device or "cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 1.5], [0.0, 0.0, 0.3])

    scene_cfg = create_scene_cfg()(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["Robot"]
    device = robot.device
    joint_names = [str(n) for n in robot.data.joint_names]
    body_names = [str(n) for n in robot.data.body_names]
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    print("\n[INFO] T1 joint order (use this to verify symmetry_func.py):")
    for i, n in enumerate(joint_names):
        print(f"  {i:2d}: {n}")
    print(f"\n[INFO] body names: {body_names}\n")

    pose = load_pose(args_cli.pose)
    # Working state, edited interactively.
    base_z = float(pose["base_pos"][2])
    joint_vals = {n: float(v) for n, v in pose["joints"].items()}
    selected = 0
    editable = [n for n in joint_names if n in joint_vals]
    STEP = 0.05
    Z_STEP = 0.01

    def apply():
        roll, pitch, yaw = pose["base_rpy"]
        quat = rpy_to_quat_wxyz(float(roll), float(pitch), float(yaw)).to(device)
        origin = scene.env_origins.to(device)[0]
        root = torch.zeros(1, 7, device=device)
        root[0, :3] = torch.tensor([0.0, 0.0, base_z], device=device) + origin
        root[0, 3:7] = quat
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

        jp = robot.data.default_joint_pos.clone()
        for n, v in joint_vals.items():
            jp[0, name_to_idx[n]] = v
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        scene.write_data_to_sim()

    def print_heights():
        bp = robot.data.body_pos_w[0]
        print("\n=== Body heights (z, metres) ===")
        for n in ["Trunk", "left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link"]:
            if n in body_names:
                print(f"  {n:18s}: {bp[body_names.index(n), 2].item():.4f}")
        print("================================\n")

    def export():
        out = {
            "poses": [
                {
                    "base_pos": [0.0, 0.0, round(base_z, 4)],
                    "base_rpy": [round(float(x), 7) for x in pose["base_rpy"]],
                    "joints": {n: round(joint_vals[n], 4) for n in joint_names if n in joint_vals},
                }
            ]
        }
        print("\n=== Current pose JSON (paste into assets/t1-crawl-pose.json) ===")
        print(json.dumps(out, indent=2))
        print("\n=== joint_pos block for t1_crawl_env_cfg.py ===")
        for n in joint_names:
            if n in joint_vals:
                print(f'        "{n}": {joint_vals[n]:.3f},')
        print("=== end ===\n")

    input_iface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    should_exit = {"v": False}

    def on_key(event):
        nonlocal selected, base_z
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        key = event.input.name
        if key == "UP":
            selected = (selected - 1) % len(editable)
            print(f"[selected] {editable[selected]} = {joint_vals[editable[selected]]:.3f}")
        elif key == "DOWN":
            selected = (selected + 1) % len(editable)
            print(f"[selected] {editable[selected]} = {joint_vals[editable[selected]]:.3f}")
        elif key == "RIGHT":
            n = editable[selected]
            joint_vals[n] += STEP
            apply()
            print(f"{n} = {joint_vals[n]:.3f}")
        elif key == "LEFT":
            n = editable[selected]
            joint_vals[n] -= STEP
            apply()
            print(f"{n} = {joint_vals[n]:.3f}")
        elif key == "RIGHT_BRACKET":
            base_z += Z_STEP
            apply()
            print(f"base_z = {base_z:.3f}")
        elif key == "LEFT_BRACKET":
            base_z -= Z_STEP
            apply()
            print(f"base_z = {base_z:.3f}")
        elif key == "R":
            for n, v in pose["joints"].items():
                joint_vals[n] = float(v)
            apply()
            print("[reset to JSON]")
        elif key == "P":
            export()
        elif key == "H":
            print_heights()
        elif key == "ESCAPE":
            should_exit["v"] = True

    sub = input_iface.subscribe_to_keyboard_events(keyboard, on_key)

    apply()
    sim.step()
    scene.update(sim.get_physics_dt())
    print_heights()

    print("Controls: UP/DOWN select joint | LEFT/RIGHT adjust | [ / ] base height")
    print("          R reset | P export JSON | H heights | ESC exit")
    print(f"[selected] {editable[selected]} = {joint_vals[editable[selected]]:.3f}\n")

    dt = sim.get_physics_dt()
    try:
        while simulation_app.is_running() and not should_exit["v"]:
            apply()
            sim.step()
            scene.update(dt)
    finally:
        input_iface.unsubscribe_to_keyboard_events(keyboard, sub)


if __name__ == "__main__":
    main()
    simulation_app.close()
