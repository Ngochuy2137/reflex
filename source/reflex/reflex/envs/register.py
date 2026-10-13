"""Gym registration of the Stage 1 (state-based teacher) task.

Registers ``Reflex-Teacher-G1-v0`` with its env config (teacher_env_cfg) and the
PPO runner config.  Import this module before ``gym.make`` / the Isaac Lab hydra
decorator runs.
"""
import gymnasium as gym

TASK_ID = "Reflex-Teacher-G1-v0"
RSL_RL_CFG = "whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg:G1BoxCatchV36PPORunnerCfg"


def _entry():
    from reflex.envs.teacher_env_cfg import G1BoxCatchStepCatchEnvCfg

    return G1BoxCatchStepCatchEnvCfg


if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _entry, "rsl_rl_cfg_entry_point": RSL_RL_CFG},
    )
