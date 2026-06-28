# T1 Crawl — Reward Weights

Working notes on the reward weights for `T1-crawl-v0`. The authoritative values
live in [`constants.py`](constants.py) (`W_*` / `STD_*` / `TARGET_*`); this file
is for **tracking and commenting** on them — why a value is what it is, what we
tried, and what we observed. Edit `constants.py` to change a weight, then record
the rationale here.

> Orientation context: `CRAWL_FACING` is currently **`"up"`** (chest-up / belly-up,
> gravity target `(-1,0,0)`). The crawl-specific reward terms assume the face-down/up
> body frame (body +X = world up/down, "forward" = body Z, "turn" = roll about body X).
> See the CLAUDE.md "T1 crawl orientation" note before retuning these.

## Rewards (positive)

| Term | Func | Weight | Params | Purpose / notes |
|---|---|---:|---|---|
| `track_lin_vel_yz_exp` | `track_lin_vel_yz_base_exp` | **4.0** | `std=0.5` | Track commanded linear velocity (body YZ). Raised so moving beats standing still. Wide `std` rewards partial progress toward the target. |
| `track_ang_vel_x_exp` | `track_ang_vel_z_world_exp` | **3.0** | `std=0.5` | Track commanded angular velocity (world yaw = roll about body X in crawl frame). |
| `crawl_orientation` | `align_projected_gravity_l2` | **0.2** | `target=FACING.gravity_target` | Keep gravity aligned with the facing's body axis (stay prone/supine, don't roll over). |
| `alive` | `is_alive` | **1.0** | — | Per-step survival bonus. Counters the "give-up" strategy where the policy flips to escape ongoing penalties. Paired with `flipped_penalty`. |

## Penalties (negative)

| Term | Func | Weight | Params | Purpose / notes |
|---|---|---:|---|---|
| `flipped_penalty` | `is_terminated_term` | **-2000.0** | `term_keys="flipped"` | One-time hit when the `flipped` termination fires. Deliberately huge so flipping is never worth it vs. accumulated `alive`. See observation note below. |
| `dof_pos_limits` | `joint_pos_limits` | **-5.0** | all joints | Penalize riding joint position limits. |
| `undesired_body_contact_penalty` | `undesired_contacts` | **-5.0** | non foot/hand bodies, `threshold=1.0 N` | Penalize ground contact from anything that isn't a foot or hand. |
| `slippage` | `feet_slide` | **-0.2** | feet + hands | Penalize lateral velocity of a contact body while it's in contact. |
| `base_height_l2` | `base_height_l2` | **-0.1** | `target=0.28 m`, Trunk | Keep Trunk near crawl height. |
| `both_feet_air` | `both_feet_air` | **-0.5** | both feet | Penalize both feet off the ground. |
| `both_hand_air` | `both_feet_air` | **-0.5** | both hands | Penalize both hands off the ground. |
| `both_left_air` | `both_feet_air` | **-0.1** | left foot+hand | Penalize the whole left side off the ground. |
| `both_right_air` | `both_feet_air` | **-0.1** | right foot+hand | Penalize the whole right side off the ground. |
| `joint_deviation_all` | `joint_deviation_l1` | **-0.01** | all joints | Keep joints near the default crawl pose. |
| `action_rate_l2` | `action_rate_l2` | **-0.01** | — | Penalize fast action changes (smoothness). |
| `torque_limits` | `applied_torque_limits` | **-0.001** | all joints | Gentle torque-saturation regularizer. **Was -5.0** — dominated the reward, lowered to -0.001. |
| `dof_torques_l2` | `joint_torques_l2` | **-1e-4** | — | Penalize large torques (effort). |

## Shared params

| Constant | Value | Used by |
|---|---:|---|
| `STD_TRACK_LIN_VEL` | 0.5 | linear-velocity exp kernel width |
| `STD_TRACK_ANG_VEL` | 0.5 | angular-velocity exp kernel width |
| `TARGET_BASE_HEIGHT` | 0.28 m | `base_height_l2` |
| `UNDESIRED_CONTACT_THRESHOLD` | 1.0 N | `undesired_body_contact_penalty` |

## Tuning log

Record dated, reversible notes here so we can correlate weight changes with runs.

- _(template)_ `YYYY-MM-DD` — changed `W_X` from `A` → `B` because ___. Run: `<run folder>`. Observed: ___.
- `torque_limits` lowered from **-5.0 → -0.001**: at -5.0 it dominated total reward and the
  policy minimized torque (went limp) instead of crawling.
- `track_lin_vel` raised to **4.0** and `STD_TRACK_LIN_VEL` widened to **0.5**: earlier values
  let "stand still" out-score "move," so the robot wouldn't commit to crawling.

## Observations

- **`Episode_Termination/flipped` after a resume**: expect a sharp transient at the resume
  step (all envs reset synchronously + startup domain-randomization is re-drawn + logging
  buffers start empty), then re-convergence to the prior trend within ~one episode length.
  This is a statistics/env-reset artifact, not a policy regression (`empirical_normalization`
  is off, and model + optimizer state are restored). A *sustained* post-resume regression
  would be the thing to worry about.
- The large `flipped_penalty` (-2000) vs. small per-step `alive` (+1) is intentional: it makes
  the rare flip catastrophic without drowning the dense shaping signals each step.
