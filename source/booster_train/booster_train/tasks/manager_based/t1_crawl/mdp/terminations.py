"""Termination functions for the T1 crawl task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def crawl_flipped(
    env: ManagerBasedRLEnv,
    target: tuple[float, float, float] = (1.0, 0.0, 0.0),
    min_alignment: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot has flipped away from its crawl facing.

    The crawl facing fixes which body axis gravity should point along (``target``;
    [+1,0,0] face-down, [-1,0,0] chest-up). We measure how well measured gravity
    still aligns with that target via the dot product ``projected_gravity_b · target``
    (both unit-length, so this is the cosine of the angle between them). It is +1
    when perfectly in the intended orientation and -1 when fully upside down.

    Returns True where alignment falls below ``min_alignment``. The default 0.0
    fires once the body has tipped past 90° from its facing (i.e. "upside down").
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b
    target_t = torch.tensor(target, dtype=g_b.dtype, device=g_b.device)
    alignment = torch.sum(g_b * target_t, dim=1)
    return alignment < min_alignment
