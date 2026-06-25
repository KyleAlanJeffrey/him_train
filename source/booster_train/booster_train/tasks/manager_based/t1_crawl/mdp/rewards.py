"""Reward functions for the T1 crawl task.

Ported from g1_crawl. Animation-dependent functions removed; custom crawl
velocity tracking and contact rewards retained.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_yz_base_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for tracking linear velocity commands in body YZ plane.

    Command format: [lin_vel_z, lin_vel_y, ang_vel_x].
    Measures body-frame velocities: vel_b[:, 2] = vz, vel_b[:, 1] = vy.
    In crawl orientation (body +Z = world forward), vz is the forward velocity.
    """
    asset = env.scene[asset_cfg.name]
    vel_b = asset.data.root_lin_vel_b[:, :3]
    cmd_yz = env.command_manager.get_command(command_name)[:, :2]   # [vz, vy]
    meas_yz = vel_b[:, [2, 1]]                                       # [vz, vy]
    lin_vel_error = torch.sum(torch.square(cmd_yz - meas_yz), dim=1)
    return torch.exp(-lin_vel_error / (std ** 2))


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for tracking the roll command as world yaw velocity.

    In crawl orientation (body +X = world down), body roll = world yaw.
    command[:, 2] = ang_vel_x (roll) is tracked against root_ang_vel_w[:, 2] (yaw).
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / (std ** 2))


def align_projected_gravity_plus_x_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment of projected gravity with body +X (crawl orientation).

    In crawl pose, gravity points along body +X (face-down horizontal).
    reward = 1 - 0.5 * ||g_b - [1, 0, 0]||^2, in [-1, 1].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b
    target = torch.tensor([1.0, 0.0, 0.0], dtype=g_b.dtype, device=g_b.device)
    dist_sq = torch.sum(torch.square(g_b - target), dim=1)
    return 1.0 - 0.5 * dist_sq


def joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 penalty for joint positions deviating from their defaults."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(angle), dim=1)


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral (xy) velocity of contact bodies when in contact with ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def both_feet_air(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize when all selected contact bodies are simultaneously in the air.

    Returns 1.0 when every body in sensor_cfg.body_ids is airborne, 0 otherwise.
    Used for feet (2 bodies), hands (2 bodies), and same-side pairs (2 bodies).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    all_in_air = torch.all(air_time > 0.0, dim=1)
    return all_in_air.float()
