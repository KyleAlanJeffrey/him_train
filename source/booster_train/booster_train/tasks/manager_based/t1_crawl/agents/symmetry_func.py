"""Left-right symmetry augmentation for the T1 crawl policy.

IMPORTANT: Joint ordering must be verified by running:
    print(env.scene["robot"].data.joint_names)
The order below assumes depth-first URDF parsing of T1_23dof.urdf.

Expected T1 joint order (23 joints):
    0:  AAHead_yaw
    1:  Head_pitch
    2:  Left_Shoulder_Pitch
    3:  Left_Shoulder_Roll
    4:  Left_Elbow_Pitch
    5:  Left_Elbow_Yaw
    6:  Right_Shoulder_Pitch
    7:  Right_Shoulder_Roll
    8:  Right_Elbow_Pitch
    9:  Right_Elbow_Yaw
    10: Waist
    11: Left_Hip_Pitch
    12: Left_Hip_Roll
    13: Left_Hip_Yaw
    14: Left_Knee_Pitch
    15: Left_Ankle_Pitch
    16: Left_Ankle_Roll
    17: Right_Hip_Pitch
    18: Right_Hip_Roll
    19: Right_Hip_Yaw
    20: Right_Knee_Pitch
    21: Right_Ankle_Pitch
    22: Right_Ankle_Roll

Observation dimensions:
    Policy (75): gravity(3) + cmds(3) + qpos(23) + qvel(23) + action(23)
    Critic (81): lin_vel(3) + ang_vel(3) + gravity(3) + cmds(3) + qpos(23) + qvel(23) + action(23)

rsl_rl >= 5.x symmetry API:
    data_augmentation_func(env, obs: TensorDict | None, actions: Tensor)
        -> (TensorDict | None, Tensor)
    The returned batch is [original; mirrored] concatenated (doubles the batch size).
"""

import torch
from tensordict import TensorDict


# Pairs that swap on left-right body mirror
_SWAP_PAIRS = [
    (2, 6),    # Left/Right_Shoulder_Pitch
    (3, 7),    # Left/Right_Shoulder_Roll
    (4, 8),    # Left/Right_Elbow_Pitch
    (5, 9),    # Left/Right_Elbow_Yaw
    (11, 17),  # Left/Right_Hip_Pitch
    (12, 18),  # Left/Right_Hip_Roll
    (13, 19),  # Left/Right_Hip_Yaw
    (14, 20),  # Left/Right_Knee_Pitch
    (15, 21),  # Left/Right_Ankle_Pitch
    (16, 22),  # Left/Right_Ankle_Roll
]

# Joints whose sign flips on left-right mirror (yaw, roll axes)
_INVERT_INDICES = [
    0,   # AAHead_yaw  (yaw flips)
    3,   # Left_Shoulder_Roll
    5,   # Left_Elbow_Yaw
    7,   # Right_Shoulder_Roll
    9,   # Right_Elbow_Yaw
    10,  # Waist       (yaw flips)
    12,  # Left_Hip_Roll
    13,  # Left_Hip_Yaw
    16,  # Left_Ankle_Roll
    18,  # Right_Hip_Roll
    19,  # Right_Hip_Yaw
    22,  # Right_Ankle_Roll
]


def mirror_joint_tensor(x: torch.Tensor, offset: int) -> torch.Tensor:
    """Apply left-right mirror to a block of 23 joint values starting at `offset`."""
    x = x.clone()
    for i, j in _SWAP_PAIRS:
        x[:, offset + i], x[:, offset + j] = x[:, offset + j].clone(), x[:, offset + i].clone()
    for k in _INVERT_INDICES:
        x[:, offset + k] = -x[:, offset + k]
    return x


def mirror_observation_policy(obs: torch.Tensor) -> torch.Tensor:
    """Mirror a 75-dim policy observation across the body sagittal plane.

    Layout: gravity(3) | cmds(3) | qpos(23) | qvel(23) | action(23)
    On left-right mirror:
      - gravity[1] (y-component) flips
      - cmd[1] (lin_vel_y) flips, cmd[2] (ang_vel_x) flips
      - qpos, qvel, action blocks: swap L/R pairs and invert roll/yaw joints
    """
    obs = obs.clone()
    # gravity y
    obs[:, 1] = -obs[:, 1]
    # cmd vy and ang_vel_x
    obs[:, 4] = -obs[:, 4]
    obs[:, 5] = -obs[:, 5]
    # joint blocks (offsets: qpos=6, qvel=29, action=52)
    obs = mirror_joint_tensor(obs, offset=6)
    obs = mirror_joint_tensor(obs, offset=29)
    obs = mirror_joint_tensor(obs, offset=52)
    return obs


def mirror_observation_critic(obs: torch.Tensor) -> torch.Tensor:
    """Mirror a 81-dim critic observation across the body sagittal plane.

    Layout: lin_vel(3) | ang_vel(3) | gravity(3) | cmds(3) | qpos(23) | qvel(23) | action(23)
    On left-right mirror:
      - lin_vel[1] (vy) flips
      - ang_vel[0] (roll) flips, ang_vel[2] (yaw) flips
      - gravity[1] (y) flips  (at index 7)
      - cmd[1] (vy) flips, cmd[2] (ang_vel_x) flips  (at indices 10, 11)
      - qpos, qvel, action blocks: swap L/R pairs and invert roll/yaw joints
    """
    obs = obs.clone()
    obs[:, 1] = -obs[:, 1]    # lin_vel y
    obs[:, 3] = -obs[:, 3]    # ang_vel x (roll)
    obs[:, 5] = -obs[:, 5]    # ang_vel z (yaw)
    obs[:, 7] = -obs[:, 7]    # gravity y
    obs[:, 10] = -obs[:, 10]  # cmd vy
    obs[:, 11] = -obs[:, 11]  # cmd ang_vel_x
    # joint blocks (offsets: qpos=12, qvel=35, action=58)
    obs = mirror_joint_tensor(obs, offset=12)
    obs = mirror_joint_tensor(obs, offset=35)
    obs = mirror_joint_tensor(obs, offset=58)
    return obs


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror a 23-dim action tensor across the body sagittal plane."""
    return mirror_joint_tensor(actions, offset=0)


def data_augmentation_func_t1(
    env,
    obs: TensorDict | None,
    actions: torch.Tensor,
) -> tuple[TensorDict | None, torch.Tensor]:
    """Symmetry data augmentation for rsl_rl >= 5.x.

    Returns [original; mirrored] concatenated, doubling the batch size.

    Args:
        env: Environment instance (unused — mirrors are computed from tensors alone).
        obs: TensorDict with "policy" (75-dim) and "critic" (81-dim) keys, or None
             when called for mirror-loss calculation only.
        actions: Action tensor [N, 23].

    Returns:
        (augmented_obs, augmented_actions): batch size doubled to [2N, ...].
    """
    mirrored_actions = mirror_actions(actions)
    combined_actions = torch.cat([actions, mirrored_actions], dim=0)

    if obs is None:
        return None, combined_actions

    n = obs.batch_size[0]
    augmented = {}
    for key in obs.keys():
        orig = obs[key]
        if key == "policy":
            aug = mirror_observation_policy(orig)
        elif key == "critic":
            aug = mirror_observation_critic(orig)
        else:
            aug = orig.clone()
        augmented[key] = torch.cat([orig, aug], dim=0)

    return TensorDict(augmented, batch_size=[n * 2]), combined_actions
