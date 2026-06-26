"""T1 crawl environment configuration.

Ported from g1_crawl (Unitree G1) to Booster T1. Key changes:
- Uses BOOSTER_T1_CFG with a crawl init pose (body horizontal, all 4 limbs on ground)
- T1 body names: Trunk, left/right_foot_link, left/right_hand_link
- No animation-based resets; uses uniform pose reset from crawl default
- Crawl orientation: body +X pointing down (projected_gravity_b ≈ [1, 0, 0])
"""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from booster_train.assets.robots.booster import BOOSTER_T1_CFG
from . import mdp

# T1 crawl initial state: body horizontal, all 4 limbs on ground.
# Rotation (w,x,y,z) = 90° about world Y → body +X points world -Z (face-down),
# body +Z points world +X (forward). This gives projected_gravity_b ≈ [1, 0, 0].
# Joint angles are within T1 limits and position the robot in quadruped stance.
# NOTE: z=0.28 is an estimate; tune after first simulation run.
T1_CRAWL_CFG = BOOSTER_T1_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.28),
        rot=(0.7071, 0.0, 0.7071, 0.0),
        joint_pos={
            "AAHead_yaw": 0.000,
            "Left_Shoulder_Pitch": -1.000,
            "Right_Shoulder_Pitch": -1.000,
            "Waist": 0.000,
            "Head_pitch": -0.300,
            "Left_Shoulder_Roll": -0.300,
            "Right_Shoulder_Roll": 0.300,
            "Left_Hip_Pitch": -1.150,
            "Right_Hip_Pitch": -1.150,
            "Left_Elbow_Pitch": 0.600,
            "Right_Elbow_Pitch": 0.600,
            "Left_Hip_Roll": 1.400,
            "Right_Hip_Roll": -1.400,
            "Left_Elbow_Yaw": -1.300,
            "Right_Elbow_Yaw": 1.300,
            "Left_Hip_Yaw": 0.950,
            "Right_Hip_Yaw": -0.900,
            "Left_Knee_Pitch": 1.700,
            "Right_Knee_Pitch": 1.700,
            "Left_Ankle_Pitch": -0.700,
            "Right_Ankle_Pitch": -0.700,
            "Left_Ankle_Roll": -0.000,
            "Right_Ankle_Roll": 0.000,
        },
        joint_vel={".*": 0.0},
    )
)


@configclass
class T1CrawlSceneCfg(InteractiveSceneCfg):
    """Scene for the T1 crawl task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = T1_CRAWL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class CommandsCfg:
    """Velocity commands in the crawl base frame (body YZ plane + roll about X)."""

    base_velocity = mdp.CrawlVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.05,
        debug_vis=True,
        ranges=mdp.CrawlVelocityCommandCfg.Ranges(
            lin_vel_z=(-1.0, 2.5),  # forward (body Z = world +X when crawling)
            lin_vel_y=(0.0, 0.0),  # lateral (body Y)
            ang_vel_x=(-1.0, 1.0),  # roll about body X (= world yaw when crawling)
        ),
    )


@configclass
class ActionsCfg:
    """Joint position actions for all 23 DOF."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observations: 75-dim policy, 81-dim critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations: gravity(3) + cmds(3) + qpos(23) + qvel(23) + action(23) = 75."""

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action_with_log)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations: lin_vel(3) + ang_vel(3) + policy_obs(75) = 81."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action_with_log)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = CriticCfg()


@configclass
class EventCfg:
    """Domain randomization and reset events."""

    # --- startup ---
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    add_trunk_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )

    # --- reset ---
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # --- interval (disabled initially) ---
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1000.0, 1000.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for crawl locomotion."""

    # --- velocity tracking ---
    # track_ang_vel_x_exp uses track_ang_vel_z_world_exp because in crawl orientation
    # (body +X = world down), rolling about body X = yawing in world frame.
    track_lin_vel_yz_exp = RewTerm(
        func=mdp.track_lin_vel_yz_base_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_x_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # --- orientation (crawl: gravity should point along body +X) ---
    crawl_orientation = RewTerm(
        func=mdp.align_projected_gravity_plus_x_l2,
        weight=0.2,
    )

    # --- base height (Trunk ~0.28m when crawling) ---
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.1,
        params={
            "target_height": 0.28,
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
        },
    )

    # --- joint regularization ---
    joint_deviation_all = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- safety limits ---
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- effort regularization ---
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)

    # --- contact penalties (penalize non-foot/hand contacts) ---
    undesired_body_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="^(?!.*foot_link|.*hand_link).*",
            ),
            "threshold": 1.0,
        },
    )

    # --- slippage penalty (lateral velocity of contact bodies when in contact) ---
    slippage = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_foot_link",
                    "right_foot_link",
                    "left_hand_link",
                    "right_hand_link",
                ],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_foot_link",
                    "right_foot_link",
                    "left_hand_link",
                    "right_hand_link",
                ],
            ),
        },
    )

    # --- ground contact enforcement ---
    both_feet_air = RewTerm(
        func=mdp.both_feet_air,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
        },
    )

    both_hand_air = RewTerm(
        func=mdp.both_feet_air,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_hand_link", "right_hand_link"],
            ),
        },
    )

    both_left_air = RewTerm(
        func=mdp.both_feet_air,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "left_hand_link"],
            ),
        },
    )

    both_right_air = RewTerm(
        func=mdp.both_feet_air,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["right_foot_link", "right_hand_link"],
            ),
        },
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class T1CrawlEnvCfg(ManagerBasedRLEnvCfg):
    """T1 crawl environment configuration."""

    scene: T1CrawlSceneCfg = T1CrawlSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
