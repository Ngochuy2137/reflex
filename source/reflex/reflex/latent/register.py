"""Stage 2: gym registration of the latent task (Reflex-Latent-G1-v0).

Import this module before the hydra_task_config decorator runs (reflex/latent/train.py does;
evaluation scripts must too).
"""
import gymnasium as gym

TASK_ID = "Reflex-Latent-G1-v0"
RSL_RL_CFG = "whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg:G1BoxCatchV36PPORunnerCfg"


def _entry():
    import sys
    from reflex.latent.latent_env_cfg import G1BoxCatchLatentEnvCfg
    return G1BoxCatchLatentEnvCfg


if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _entry, "rsl_rl_cfg_entry_point": RSL_RL_CFG},
    )
    print(f"[latent] registered {TASK_ID}", flush=True)
