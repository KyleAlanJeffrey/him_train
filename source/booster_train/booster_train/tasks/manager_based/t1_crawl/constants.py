"""Tunable constants for the T1 crawl task.

Single place to adjust the crawl orientation, init pose, and reward weights.
Imported by ``t1_crawl_env_cfg.py`` — edit values here rather than in the cfg.
"""

# =============================================================================
# Orientation
# =============================================================================
# "down" = chest toward ground (prone). Rot = +90° about world Y → body +X points
#          world-down, gravity ≈ [+1, 0, 0]. Original face-down crawl.
# "up"   = chest toward sky (supine). Rot = -90° about world Y → body +X points
#          world-up, gravity ≈ [-1, 0, 0].
# Switching this flips init rot AND the orientation-reward target together (both
# read from FACING below). The joint pose differs per orientation (limbs reach the
# opposite way), so each facing carries its own joint dict — tune them with
# scripts/experiments/pose_viewer_t1.py (press F to flip, P to export).
CRAWL_FACING = "up"

# --- Face-down crawl pose (limbs tucked under, chest to ground) ---
# height/rot/joints are paste targets for the pose_viewer_t1.py "P" export.
_FACE_DOWN_HEIGHT = 0.28                       # base spawn z (m); estimate, tune in sim
_FACE_DOWN_ROT = (0.7071, 0.0, 0.7071, 0.0)    # +90° about world Y (rpy ≈ [0, 1.5708, 0])
_FACE_DOWN_JOINTS = {
    "AAHead_yaw": 0.000,
    "Left_Shoulder_Pitch": -1.000,
    "Right_Shoulder_Pitch": -1.000,
    "Waist": 0.000,
    "Head_pitch": -0.300,
    "Left_Shoulder_Roll": 0.800,
    "Right_Shoulder_Roll": -0.800,
    "Left_Hip_Pitch": -1.150,
    "Right_Hip_Pitch": -1.150,
    "Left_Elbow_Pitch": 0.600,
    "Right_Elbow_Pitch": 0.600,
    "Left_Hip_Roll": 1.400,
    "Right_Hip_Roll": -1.400,
    "Left_Elbow_Yaw": -1.300,
    "Right_Elbow_Yaw": 1.300,
    "Left_Hip_Yaw": 0.950,
    "Right_Hip_Yaw": -0.950,
    "Left_Knee_Pitch": 1.700,
    "Right_Knee_Pitch": 1.700,
    "Left_Ankle_Pitch": -0.700,
    "Right_Ankle_Pitch": -0.700,
    "Left_Ankle_Roll": -0.000,
    "Right_Ankle_Roll": 0.000,
}

# --- Chest-up crawl pose (belly-up, limbs reaching back to ground) ---
_CHEST_UP_HEIGHT = 0.1999
_CHEST_UP_ROT = (0.7071, 0.0, -0.7071, 0.0)    # rpy ≈ [0.0, -1.5708, 0.0]
_CHEST_UP_JOINTS = {
    "AAHead_yaw": 0.000,
    "Left_Shoulder_Pitch": -0.860,
    "Right_Shoulder_Pitch": -0.860,
    "Waist": 0.000,
    "Head_pitch": -0.300,
    "Left_Shoulder_Roll": 0.050,
    "Right_Shoulder_Roll": -0.050,
    "Left_Hip_Pitch": -0.620,
    "Right_Hip_Pitch": -0.620,
    "Left_Elbow_Pitch": 0.860,
    "Right_Elbow_Pitch": 0.860,
    "Left_Hip_Roll": 0.850,
    "Right_Hip_Roll": -0.850,
    "Left_Elbow_Yaw": 1.590,
    "Right_Elbow_Yaw": -1.590,
    "Left_Hip_Yaw": -0.540,
    "Right_Hip_Yaw": 0.540,
    "Left_Knee_Pitch": 1.920,
    "Right_Knee_Pitch": 1.920,
    "Left_Ankle_Pitch": 0.170,
    "Right_Ankle_Pitch": 0.170,
    "Left_Ankle_Roll": -0.110,
    "Right_Ankle_Roll": 0.110,
}

# Per-facing presets: init height (m) + rotation (w,x,y,z) + body-frame gravity
# target for the crawl-orientation reward + the joint pose for that facing.
FACING_PRESETS = {
    "down": {
        "height": _FACE_DOWN_HEIGHT,
        "rot": _FACE_DOWN_ROT,
        "gravity_target": (1.0, 0.0, 0.0),
        "joints": _FACE_DOWN_JOINTS,
    },
    "up": {
        "height": _CHEST_UP_HEIGHT,
        "rot": _CHEST_UP_ROT,
        "gravity_target": (-1.0, 0.0, 0.0),
        "joints": _CHEST_UP_JOINTS,
    },
}
FACING = FACING_PRESETS[CRAWL_FACING]

# =============================================================================
# Reward weights
# =============================================================================
W_TRACK_LIN_VEL = 4.0          # track commanded linear velocity (body YZ) — raised so moving beats standing
W_TRACK_ANG_VEL = 3.0          # track commanded angular velocity (world yaw)
W_CRAWL_ORIENT = 3.0           # keep gravity aligned with facing's body axis — high: the ONLY thing
                               # discouraging flipping now (no flip termination); must beat the crawl basin
W_ALIVE = 1.0                  # small constant alive bonus (keeps per-step reward net-positive)
W_FLIPPED_PENALTY = -2000.0    # UNUSED (flip termination removed) — kept for optional re-enable
W_BASE_HEIGHT = -0.1           # penalize Trunk height deviation from target
W_JOINT_DEVIATION = -0.01      # penalize joint drift from default pose
W_HEAD_STATIC = -1.0           # keep the head joints (AAHead_yaw, Head_pitch) at their default pose
W_DOF_POS_LIMITS = -5.0        # penalize hitting joint position limits
W_TORQUE_LIMITS = -0.01        # gentle torque-saturation regularizer (was -5.0: dominated reward)
W_ACTION_RATE = -0.01          # penalize fast action changes
W_DOF_TORQUES = -1e-4          # penalize large torques
W_UNDESIRED_CONTACT = -5.0     # penalize non-foot/hand ground contact
W_SLIPPAGE = -0.2              # penalize lateral slip of contact bodies
W_BOTH_FEET_AIR = -0.5         # penalize both feet off the ground
W_BOTH_HAND_AIR = -0.5         # penalize both hands off the ground
W_BOTH_LEFT_AIR = -0.1         # penalize left foot+hand both off the ground
W_BOTH_RIGHT_AIR = -0.1        # penalize right foot+hand both off the ground

# =============================================================================
# Reward params
# =============================================================================
STD_TRACK_LIN_VEL = 0.8        # exp-kernel width — wider so partial progress toward the target is rewarded
STD_TRACK_ANG_VEL = 0.8        # exp-kernel width for angular-velocity tracking
TARGET_BASE_HEIGHT = 0.28      # desired Trunk height (m) when crawling
UNDESIRED_CONTACT_THRESHOLD = 1.0  # contact force (N) above which contact is "undesired"

# =============================================================================
# Command velocity ranges (the desired-direction targets the policy tracks)
# =============================================================================
# Keep targets reachable: widening these before the robot reliably crawls
# reintroduces unhittable targets and flattens the tracking-reward gradient.
CMD_LIN_VEL_Z = (-1.0, 1.5)    # forward speed (body Z = world +X when crawling)
CMD_LIN_VEL_Y = (-0.5, 0.5)    # lateral / strafe (body Y) — modest range; strafing is harder than forward
CMD_ANG_VEL_X = (-1.0, 1.0)    # turn rate (roll about body X = world yaw)
CMD_RESAMPLING_TIME = (4.0, 8.0)   # seconds between drawing a new random command
CMD_REL_STANDING_ENVS = 0.05   # fraction of envs given a zero (stand-still) command
