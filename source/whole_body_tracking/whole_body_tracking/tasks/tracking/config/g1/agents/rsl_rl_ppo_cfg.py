from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)


@configclass
class G1BoxCatchV1PPORunnerCfg(G1FlatPPORunnerCfg):
    """PPO runner cfg for box catching.

    The box reuses the balloon observation terms, so the 166-D chest-hold
    observation layout is preserved and chest-hold checkpoints can warm-start
    it. Exploration stays at the defaults (init_noise_std 1.0, entropy_coef
    0.005); higher settings (1.5 / 0.01) led to periodic PPO collapse.
    """
    experiment_name = "catch_box_v1"
    run_name = "box_v1_thrown"
    logger = "wandb"
    wandb_project = "catch_box"
    max_iterations = 3000
    save_interval = 250


@configclass
class G1BoxCatchV31PPORunnerCfg(G1BoxCatchV1PPORunnerCfg):
    """Dense-checkpoint variant: saves every 50 iterations, so a late-training
    collapse can be bracketed and the best checkpoint picked."""
    save_interval = 50


@configclass
class G1BoxCatchV36PPORunnerCfg(G1BoxCatchV31PPORunnerCfg):
    """Runner used by the box-catch tasks: dense checkpoints (every 50
    iterations) with its own run name."""
    run_name = "box_v36_saferobust"
