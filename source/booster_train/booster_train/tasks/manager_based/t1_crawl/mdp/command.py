"""Velocity commands for the T1 crawl task.

Copied verbatim from g1_crawl: CrawlVelocityCommand operates in body YZ plane (vz, vy, roll_x)
and BooleanCommand is a simple on/off command. Neither class has robot-specific dependencies.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass


class CrawlVelocityCommand(CommandTerm):
    r"""Velocity command in the robot's base YZ plane (vz, vy) plus roll rate (ang_vel_x).

    For a crawling robot with body horizontal, body Z is the forward direction and
    body Y is lateral. The angular velocity command is around body X (= world yaw
    in crawl orientation).
    """

    cfg: CrawlVelocityCommandCfg

    def __init__(self, cfg: CrawlVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "heading_command=True but ranges.heading is None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"ranges.heading={self.cfg.ranges.heading} set but heading_command is False."
            )

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        self.metrics["error_vel_yz"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_roll"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "CrawlVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Desired base velocity [lin_vel_z, lin_vel_y, ang_vel_x]. Shape (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        measured_lin_yz = self.robot.data.root_lin_vel_b[:, [2, 1]]
        lin_err = torch.norm(self.vel_command_b[:, :2] - measured_lin_yz, dim=-1) / max_command_step
        self.metrics["error_vel_yz"] += lin_err
        ang_err = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 0]) / max_command_step
        self.metrics["error_vel_roll"] += ang_err

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        ranges = self.cfg.ranges
        self.vel_command_b[env_ids, 0] = r.uniform_(*ranges.lin_vel_z)
        self.vel_command_b[env_ids, 1] = r.uniform_(*ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*ranges.ang_vel_x)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            ranges = self.cfg.ranges
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=ranges.ang_vel_x[0],
                max=ranges.ang_vel_x[1],
            )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                if hasattr(self, "current_vel_visualizer"):
                    self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        cmd_xyz_b = torch.zeros((self.num_envs, 3), device=self.device)
        cmd_xyz_b[:, 1] = self.command[:, 1]
        cmd_xyz_b[:, 2] = self.command[:, 0]
        vel_des_scale, vel_des_quat = self._resolve_base_velocity_to_arrow(cmd_xyz_b)
        measured_b = self.robot.data.root_lin_vel_b
        vel_cur_scale, vel_cur_quat = self._resolve_base_velocity_to_arrow(measured_b)
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_quat, vel_des_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_cur_quat, vel_cur_scale)

    def _resolve_base_velocity_to_arrow(self, xyz_velocity_b: torch.Tensor):
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xyz_velocity_b.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xyz_velocity_b, dim=1) * 3.0
        base_quat_w = self.robot.data.root_quat_w
        vel_w = math_utils.quat_apply(base_quat_w, xyz_velocity_b)
        vx, vy, vz = vel_w[:, 0], vel_w[:, 1], vel_w[:, 2]
        yaw = torch.atan2(vy, vx)
        horiz = torch.sqrt(vx * vx + vy * vy)
        pitch = torch.atan2(-vz, horiz)
        zeros = torch.zeros_like(yaw)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch, yaw)
        return arrow_scale, arrow_quat


@configclass
class CrawlVelocityCommandCfg(CommandTermCfg):
    """Configuration for CrawlVelocityCommand."""

    class_type: type = CrawlVelocityCommand

    asset_name: str = MISSING

    heading_command: bool = False
    heading_control_stiffness: float = 1.0
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0

    @configclass
    class Ranges:
        lin_vel_z: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_x: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = None

    ranges: Ranges = MISSING

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].color = (1.0, 0.6, 0.1)


class BooleanCommand(CommandTerm):
    """Simple 0/1 command with configurable probability of being True."""

    cfg: BooleanCommandCfg

    def __init__(self, cfg: BooleanCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.boolean_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["command_value"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.boolean_command

    def _update_metrics(self):
        self.metrics["command_value"] = self.boolean_command[:, 0]

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.boolean_command[env_ids, 0] = (r.uniform_(0.0, 1.0) <= self.cfg.true_probability).float()

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class BooleanCommandCfg(CommandTermCfg):
    class_type: type = BooleanCommand
    asset_name: str = MISSING
    true_probability: float = 0.5
