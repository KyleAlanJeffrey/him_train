"""Custom observation functions for the T1 crawl task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def last_action_with_log(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """Return the last action, printing a warning if any value exceeds ±40."""
    action = env.action_manager.action
    action_max = action.max().item()
    action_min = action.min().item()
    if action_max > 40 or action_min < -40:
        num_joints = action.shape[1]
        max_idx = action.argmax().item()
        min_idx = action.argmin().item()
        robot = env.scene["robot"]
        joint_names = robot.data.joint_names
        max_name = joint_names[max_idx % num_joints] if max_idx % num_joints < len(joint_names) else f"j{max_idx % num_joints}"
        min_name = joint_names[min_idx % num_joints] if min_idx % num_joints < len(joint_names) else f"j{min_idx % num_joints}"
        print(
            f"action_max: {action_max:.2f} (env={max_idx // num_joints}, joint={max_name}), "
            f"action_min: {action_min:.2f} (env={min_idx // num_joints}, joint={min_name})"
        )
    if action_name is None:
        return env.action_manager.action
    return env.action_manager.get_term(action_name).raw_actions
