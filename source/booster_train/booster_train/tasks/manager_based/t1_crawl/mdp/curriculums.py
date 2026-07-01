"""Curriculum functions for the T1 crawl task.

Ported from g1_crawl. Animation-specific frame-range curricula omitted.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs.mdp.curriculums import modify_term_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fraction_flipped(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    target: tuple[float, float, float] = (-1.0, 0.0, 0.0),
) -> torch.Tensor:
    """Logged metric (NOT a real curriculum): fraction of envs currently flipped.

    "Flipped" = projected gravity misaligned with the facing target by >90°
    (dot(projected_gravity_b, target) < 0). Computed over ALL envs (ignores
    env_ids) and returned unchanged, so it modifies nothing — it just surfaces as
    ``Curriculum/flipped`` in the logs so you can watch the flip rate even though
    there is no flip *termination*. Pass the facing's gravity_target.
    """
    asset: Articulation = env.scene["robot"]
    g_b = asset.data.projected_gravity_b
    target_t = torch.tensor(target, dtype=g_b.dtype, device=g_b.device)
    alignment = torch.sum(g_b * target_t, dim=1)
    return (alignment < 0.0).float().mean()


def terrain_levels_vel_crawl(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain difficulty curriculum based on distance walked.

    Increases terrain difficulty when the robot walks far enough, decreases it
    when the robot walks less than half the required distance.
    Only applicable when terrain_type="generator" is used.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def ramp_weight_linear(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data,
    initial_weight: float = 0.0,
    target_weight: float = -10.0,
    start_step: int = 0,
    end_step: int = 50000,
):
    """Linearly ramp a reward term weight from initial_weight to target_weight.

    Useful for gradually introducing penalty terms as training progresses.
    """
    step = env.common_step_counter
    if step < start_step:
        weight = initial_weight
    elif step >= end_step:
        weight = target_weight
    else:
        progress = (step - start_step) / (end_step - start_step)
        weight = initial_weight + (target_weight - initial_weight) * progress

    if env.common_step_counter > 0:
        return weight
    return modify_term_cfg.NO_CHANGE
