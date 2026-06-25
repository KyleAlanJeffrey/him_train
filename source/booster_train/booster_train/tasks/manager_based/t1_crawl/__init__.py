"""T1 crawl locomotion task — Booster T1 on all fours."""

import gymnasium as gym

from . import agents
from .t1_crawl_env_cfg import T1CrawlEnvCfg

gym.register(
    id="T1-crawl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.t1_crawl_env_cfg:T1CrawlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
