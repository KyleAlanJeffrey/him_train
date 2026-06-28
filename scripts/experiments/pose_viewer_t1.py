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
    U / O            decrease / increase base ROLL  by 0.05 rad
    I / K            decrease / increase base PITCH by 0.05 rad
    J / L            decrease / increase base YAW   by 0.05 rad
    F                flip chest up <-> down (negate base pitch)
    G                toggle auto-ground (drive base_z so the lowest contact link
                     rests on the ground; on by default)
    R                reapply the pose from JSON (discard edits)
    P                print the pose as JSON + a constants.py block (paste into constants.py)
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

from isaaclab.utils.math import quat_apply

from booster_train.assets.robots.booster import BOOSTER_T1_CFG
from booster_train.tasks.manager_based.t1_crawl import constants as crawl_constants


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
    # Mutable list so closures can edit roll/pitch/yaw in place without `nonlocal`.
    base_rpy = [float(x) for x in pose["base_rpy"]]
    joint_vals = {n: float(v) for n, v in pose["joints"].items()}
    selected = 0
    editable = [n for n in joint_names if n in joint_vals]
    STEP = 0.05
    Z_STEP = 0.01
    RPY_STEP = 0.05

    # Contact links that should sit on the ground in a crawl stance. Auto-ground
    # keeps the lowest of these at GROUND_CLEARANCE so nothing clips through z=0.
    #
    # body_pos_w is the link *origin*, not the collision surface — the hand/foot
    # geometry is offset from it (per the URDF), so measuring the origin reports
    # the wrong contact height. We approximate each contact link's collision shape
    # by its bounding box (center + half-extents in the link frame, from the URDF
    # <collision> tags) and transform the 8 corners into world space to find the
    # true lowest point. Hand cylinders are over-approximated by their bbox, which
    # is conservative (errs toward "touching").
    CONTACT_BOXES = {
        # link: (center_xyz, half_extents_xyz) in the link frame.
        # Hands: the ball end-effector sphere (r=0.035 at y=±0.2115), NOT the forearm
        # cylinder — the ball is the actual crawl contact point.
        "left_hand_link": ((0.0, 0.2115, 0.0), (0.035, 0.035, 0.035)),
        "right_hand_link": ((0.0, -0.2115, 0.0), (0.035, 0.035, 0.035)),
        "left_foot_link": ((0.0101079, 0.0, -0.0214208), (0.112434, 0.05, 0.021830)),
        "right_foot_link": ((0.0101079, 0.0, -0.0214208), (0.112434, 0.05, 0.021830)),
    }
    CONTACT_LINKS = [n for n in CONTACT_BOXES if n in body_names]
    contact_idx = {n: body_names.index(n) for n in CONTACT_LINKS}

    def _corners(center, half):
        c = torch.tensor(center, dtype=torch.float32, device=device)
        h = torch.tensor(half, dtype=torch.float32, device=device)
        signs = torch.tensor(
            [[sx, sy, sz] for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)],
            dtype=torch.float32,
            device=device,
        )
        return c + signs * h  # (8, 3) local corner points

    contact_corners = {n: _corners(*CONTACT_BOXES[n]) for n in CONTACT_LINKS}

    def contact_clearance(name):
        """Lowest world-z of a contact link's collision box, relative to the ground."""
        idx = contact_idx[name]
        pos = robot.data.body_pos_w[0, idx]
        quat = robot.data.body_quat_w[0, idx]
        corners = contact_corners[name]
        world = quat_apply(quat.unsqueeze(0).expand(corners.shape[0], 4), corners) + pos
        return world[:, 2].min().item() - scene.env_origins[0, 2].item()

    GROUND_CLEARANCE = 0.0
    auto_ground = {"v": True}

    def apply():
        roll, pitch, yaw = base_rpy
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
        origin_z = scene.env_origins[0, 2].item()
        print("\n=== Body heights (clearance = collision surface above ground) ===")
        # Trunk: report the link origin (not a contact link).
        if "Trunk" in body_names:
            z = bp[body_names.index("Trunk"), 2].item()
            print(f"  {'Trunk':18s}: origin_z={z:.4f}  clearance={z - origin_z:+.4f}")
        # Contact links: report the true lowest collision-surface clearance.
        for n in CONTACT_LINKS:
            clearance = contact_clearance(n)
            flag = "  << BELOW GROUND" if clearance < 0 else ""
            print(f"  {n:18s}: contact_clearance={clearance:+.4f}{flag}")
        if CONTACT_LINKS:
            min_clear = min(contact_clearance(n) for n in CONTACT_LINKS)
            print(f"  lowest contact clearance: {min_clear:+.4f} (auto-ground {'ON' if auto_ground['v'] else 'OFF'})")
        print("================================\n")

    def export():
        out = {
            "poses": [
                {
                    "base_pos": [0.0, 0.0, round(base_z, 4)],
                    "base_rpy": [round(float(x), 7) for x in base_rpy],
                    "joints": {n: round(joint_vals[n], 4) for n in joint_names if n in joint_vals},
                }
            ]
        }
        print("\n=== Current pose JSON (paste into assets/t1-crawl-pose.json) ===")
        print(json.dumps(out, indent=2))
        # constants.py block: facing comes from CRAWL_FACING in constants.py, so the
        # exported names target whichever pose (face-down / chest-up) is configured.
        # Prints every value the FACING preset needs for positioning: height, rot, joints.
        quat = rpy_to_quat_wxyz(*[float(x) for x in base_rpy])
        w, x, y, z = (round(float(v), 4) for v in quat)
        facing = crawl_constants.CRAWL_FACING
        prefix = "_FACE_DOWN" if facing == "down" else "_CHEST_UP"
        rpy_round = [round(v, 4) for v in base_rpy]
        print(f"\n=== constants.py block (CRAWL_FACING = \"{facing}\") ===")
        print(f"{prefix}_HEIGHT = {base_z:.4f}")
        print(f"{prefix}_ROT = ({w}, {x}, {y}, {z})    # rpy ≈ {rpy_round}")
        print(f"{prefix}_JOINTS = {{")
        for n in joint_names:
            if n in joint_vals:
                print(f'    "{n}": {joint_vals[n]:.3f},')
        print("}")
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
        elif key in ("U", "O"):
            base_rpy[0] += RPY_STEP if key == "O" else -RPY_STEP
            apply()
            print(f"base_rpy = [{base_rpy[0]:.3f}, {base_rpy[1]:.3f}, {base_rpy[2]:.3f}]")
        elif key in ("I", "K"):
            base_rpy[1] += RPY_STEP if key == "I" else -RPY_STEP
            apply()
            print(f"base_rpy = [{base_rpy[0]:.3f}, {base_rpy[1]:.3f}, {base_rpy[2]:.3f}]")
        elif key in ("J", "L"):
            base_rpy[2] += RPY_STEP if key == "L" else -RPY_STEP
            apply()
            print(f"base_rpy = [{base_rpy[0]:.3f}, {base_rpy[1]:.3f}, {base_rpy[2]:.3f}]")
        elif key == "F":
            base_rpy[1] = -base_rpy[1]
            apply()
            print(f"[flip chest] base_rpy = [{base_rpy[0]:.3f}, {base_rpy[1]:.3f}, {base_rpy[2]:.3f}]")
        elif key == "G":
            auto_ground["v"] = not auto_ground["v"]
            print(f"[auto-ground] {'ON' if auto_ground['v'] else 'OFF'}")
        elif key == "R":
            for n, v in pose["joints"].items():
                joint_vals[n] = float(v)
            base_rpy[:] = [float(x) for x in pose["base_rpy"]]
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
    print("          U/O roll | I/K pitch | J/L yaw | F flip chest up<->down")
    print("          G auto-ground (ON) | R reset | P export JSON | H heights | ESC exit")
    print(f"[selected] {editable[selected]} = {joint_vals[editable[selected]]:.3f}\n")

    dt = sim.get_physics_dt()
    try:
        while simulation_app.is_running() and not should_exit["v"]:
            apply()
            sim.step()
            scene.update(dt)
            # Drive base_z so the lowest contact collision surface rests on the ground.
            # body poses are valid after the step, so the correction lands on next apply().
            if auto_ground["v"] and CONTACT_LINKS:
                min_clear = min(contact_clearance(n) for n in CONTACT_LINKS)
                base_z -= min_clear - GROUND_CLEARANCE
    finally:
        input_iface.unsubscribe_to_keyboard_events(keyboard, sub)


if __name__ == "__main__":
    main()
    simulation_app.close()
