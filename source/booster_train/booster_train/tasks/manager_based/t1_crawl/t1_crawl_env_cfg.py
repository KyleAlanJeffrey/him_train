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
from . import constants as c
from . import mdp

# Init state and reward tunables live in constants.py — edit them there.
# CRAWL_FACING selects face-down vs chest-up (rot + reward target + joint pose).
T1_CRAWL_CFG = BOOSTER_T1_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, c.FACING["height"]),
        rot=c.FACING["rot"],
        joint_pos=dict(c.FACING["joints"]),
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
        resampling_time_range=c.CMD_RESAMPLING_TIME,
        rel_standing_envs=c.CMD_REL_STANDING_ENVS,
        debug_vis=True,
        ranges=mdp.CrawlVelocityCommandCfg.Ranges(
            lin_vel_z=c.CMD_LIN_VEL_Z,  # forward (body Z = world +X when crawling)
            lin_vel_y=c.CMD_LIN_VEL_Y,  # lateral (body Y)
            ang_vel_x=c.CMD_ANG_VEL_X,  # roll about body X (= world yaw when crawling)
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
        weight=c.W_TRACK_LIN_VEL,
        params={"command_name": "base_velocity", "std": c.STD_TRACK_LIN_VEL},
    )
    track_ang_vel_x_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=c.W_TRACK_ANG_VEL,
        params={"command_name": "base_velocity", "std": c.STD_TRACK_ANG_VEL},
    )

    # --- orientation (crawl: gravity should point along the facing's body axis) ---
    crawl_orientation = RewTerm(
        func=mdp.align_projected_gravity_l2,
        weight=c.W_CRAWL_ORIENT,
        params={"target": c.FACING["gravity_target"]},
    )

    # --- survival: small constant alive bonus (keeps per-step reward net-positive).
    # No flip penalty term — the flip termination it referenced was removed; flipping
    # is now discouraged continuously by crawl_orientation instead of a one-time hit. ---
    alive = RewTerm(func=mdp.is_alive, weight=c.W_ALIVE)

    # --- base height (Trunk ~0.28m when crawling) ---
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=c.W_BASE_HEIGHT,
        params={
            "target_height": c.TARGET_BASE_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
        },
    )

    # --- joint regularization ---
    joint_deviation_all = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=c.W_JOINT_DEVIATION,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- keep the head static (strong deviation penalty on just the head joints) ---
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=c.W_HEAD_STATIC,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )

    # --- safety limits ---
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=c.W_DOF_POS_LIMITS,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=c.W_TORQUE_LIMITS,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- effort regularization ---
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=c.W_ACTION_RATE)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=c.W_DOF_TORQUES)

    # --- contact penalties (penalize non-foot/hand contacts) ---
    undesired_body_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=c.W_UNDESIRED_CONTACT,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="^(?!.*foot_link|.*hand_link).*",
            ),
            "threshold": c.UNDESIRED_CONTACT_THRESHOLD,
        },
    )

    # --- slippage penalty (lateral velocity of contact bodies when in contact) ---
    slippage = RewTerm(
        func=mdp.feet_slide,
        weight=c.W_SLIPPAGE,
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
        weight=c.W_BOTH_FEET_AIR,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
        },
    )

    both_hand_air = RewTerm(
        func=mdp.both_feet_air,
        weight=c.W_BOTH_HAND_AIR,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_hand_link", "right_hand_link"],
            ),
        },
    )

    both_left_air = RewTerm(
        func=mdp.both_feet_air,
        weight=c.W_BOTH_LEFT_AIR,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "left_hand_link"],
            ),
        },
    )

    both_right_air = RewTerm(
        func=mdp.both_feet_air,
        weight=c.W_BOTH_RIGHT_AIR,
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

    # NOTE: no `flipped` termination. A reset-on-flip is an *escape hatch* — the
    # policy learns to flip on purpose to end an episode that's accruing net
    # reward less than the flip alternative, and no one-time penalty can out-scale
    # escaping a persistent per-step cost. Instead, flipping is discouraged
    # *continuously* by the crawl_orientation reward (and the joint-limit penalty
    # the flipped pose incurs), with no way to bail out. See mdp.crawl_flipped
    # (kept for reference / optional re-enable).


@configclass
class CurriculumCfg:
    """Logged metrics (no actual curriculum). Surfaced under Curriculum/ in logs."""

    # Fraction of envs currently flipped — watch the flip rate without a flip
    # termination. Facing-aware via the same gravity_target as the orientation reward.
    flipped = CurrTerm(
        func=mdp.fraction_flipped,
        params={"target": c.FACING["gravity_target"]},
    )


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
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
