"""PPO agent configuration for the T1 crawl task (rsl_rl >= 5.0)."""

from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from .symmetry_func import data_augmentation_func_t1


@configclass
class _GaussianDistCfg:
    """Minimal Gaussian distribution config for rsl_rl MLPModel."""

    class_name: str = "GaussianDistribution"
    init_std: float = 1.0
    std_type: str = "scalar"


@configclass
class _MLPModelCfg:
    """Minimal MLP model config — only the fields MLPModel.__init__ accepts.

    Avoids the deprecated 'stochastic'/'init_noise_std' fields on RslRlMLPModelCfg
    that would be serialized as MISSING sentinel objects and break MLPModel construction.
    """

    class_name: str = "MLPModel"
    hidden_dims: list = MISSING
    activation: str = MISSING
    obs_normalization: bool = False
    distribution_cfg: Any = None


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 250
    experiment_name = "t1_crawl"
    empirical_normalization = False
    clip_actions = 5.0

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    actor = _MLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg=_GaussianDistCfg(),
    )

    critic = _MLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg=None,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.5,
            data_augmentation_func=data_augmentation_func_t1,
        ),
    )
