# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 1: train the state-based teacher policy (task Reflex-Teacher-G1-v0) with PPO + AMP."""

"""Launch Isaac Sim Simulator first."""

import argparse
from reflex import paths as _paths
import os
import sys
import time

# Pre-import numpy from PYTHONPATH-priority site-packages BEFORE AppLauncher
# mutates sys.path with omni.kit.pip_archive (which contains a broken numpy
# install missing libopenblas64). Subsequent `import numpy` calls hit sys.modules.
import numpy  # noqa: F401

from isaaclab.app import AppLauncher

# local imports
from reflex.utils import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Local path to motion .npz (skips wandb).")
parser.add_argument("--warm_start_ckpt", type=str, default=None,
                    help="Path to a pre-trained ckpt (.pt). Loads weights with first-layer "
                         "padding if the observation dimension grew.")
parser.add_argument("--auto_render_every", type=int, default=0,
                    help="If >0, render a short clip every N iterations (0 = disabled).")
# Multi-GPU: this Isaac Lab version expects the script to define
# --distributed (AppLauncher only reads it from the parsed namespace).
parser.add_argument("--distributed", action="store_true", default=False,
                    help="Run training with multiple GPUs (torchrun + AppLauncher distributed).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Multi-GPU: avoid cold-booting several Isaac processes at the same moment
# (URDF converter cache race). torchrun spawns all ranks at once, so
# optionally stagger AppLauncher init per local rank (AMP_DDP_STAGGER=seconds).
_ddp_stagger = float(os.environ.get("AMP_DDP_STAGGER", "0"))
_ddp_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if getattr(args_cli, "distributed", False) and _ddp_stagger > 0 and _ddp_local_rank > 0:
    print(f"[amp_ddp] local_rank {_ddp_local_rank}: delaying AppLauncher init "
          f"by {_ddp_local_rank * _ddp_stagger:.0f}s (AMP_DDP_STAGGER={_ddp_stagger:.0f})",
          flush=True)
    time.sleep(_ddp_local_rank * _ddp_stagger)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
import pickle as _pickle

from isaaclab.utils.io import dump_yaml


def dump_pickle(path, obj):
    """Replacement for isaaclab.utils.io.dump_pickle removed in IsaacLab 2.3."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        _pickle.dump(obj, f)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
import reflex.envs.register  # noqa: F401  (registers Reflex-Teacher-G1-v0)
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

# ---- AMP: inject AmpPPO into the namespace where the base runner
# eval()s the algorithm class_name (no upstream rsl_rl file is modified).
import rsl_rl.runners.on_policy_runner as _opr  # noqa: E402

from reflex.amp.amp_ppo import AmpPPO  # noqa: E402

_opr.AmpPPO = AmpPPO

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Multi-GPU launch (torchrun + --distributed): one sim per rank
    # on cuda:<local_rank>. rsl_rl 3.0.1 OnPolicyRunner reads WORLD_SIZE/RANK/
    # LOCAL_RANK itself, inits NCCL, and all-reduces the actor-critic; AmpPPO
    # handles the discriminator side when AMP_DDP=1 (see reflex/amp/amp_ppo.py).
    if getattr(args_cli, "distributed", False):
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        # per-rank seed so rollouts explore differently (standard Isaac Lab recipe)
        agent_cfg.seed += app_launcher.local_rank
        env_cfg.seed = agent_cfg.seed
        print(f"[amp_ddp] rank {app_launcher.global_rank}/"
              f"{os.environ.get('WORLD_SIZE', '?')} (local {app_launcher.local_rank}): "
              f"sim+agent on cuda:{app_launcher.local_rank}, seed {agent_cfg.seed}",
              flush=True)

    # load the motion file: prefer --motion_file (local), else fall back to wandb registry
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        registry_name = None
    else:
        if args_cli.registry_name is None:
            raise ValueError("Provide --motion_file or --registry_name.")
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # AMP: swap the algorithm class before runner construction
    agent_cfg.algorithm.class_name = "AmpPPO"
    print("[AMP] algorithm class -> AmpPPO", flush=True)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # AMP: attach the discriminator context (dataset + weights via env vars)
    runner.alg.set_amp_context(
        env.unwrapped,
        dataset_path=os.environ.get("AMP_DATASET", _paths.AMP_DATASET),
        w_style=float(os.environ.get("AMP_W_STYLE", "1.0")),
        w_task=float(os.environ.get("AMP_W_TASK", "1.0")),
        disc_lr=float(os.environ.get("AMP_DISC_LR", "1e-4")),
        disc_steps=int(os.environ.get("AMP_DISC_STEPS", "4")),
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # Warm start from a checkpoint; the first layer is zero-padded when the
    # observation dimension grew.
    if args_cli.warm_start_ckpt is not None:
        ws_path = args_cli.warm_start_ckpt
        print(f"[INFO][warm_start] loading {ws_path}", flush=True)
        ckpt = torch.load(ws_path, map_location=agent_cfg.device, weights_only=False)
        old_state = ckpt.get("model_state_dict", ckpt)
        # rsl_rl OnPolicyRunner.alg.policy is the actor-critic.
        ac = runner.alg.policy
        new_state = ac.state_dict()
        n_loaded = 0
        n_padded = 0
        n_skipped = 0
        for k, old_v in old_state.items():
            if k not in new_state:
                n_skipped += 1
                continue
            new_v = new_state[k]
            if old_v.shape == new_v.shape:
                new_state[k] = old_v.clone().to(new_v.device).type_as(new_v)
                n_loaded += 1
            elif (k.endswith(".0.weight") and old_v.dim() == 2
                  and old_v.shape[0] == new_v.shape[0]
                  and old_v.shape[1] < new_v.shape[1]):
                # First-layer Linear weight: pad input dim with zeros
                padded = torch.zeros_like(new_v)
                padded[:, :old_v.shape[1]] = old_v.clone().to(new_v.device).type_as(new_v)
                new_state[k] = padded
                n_padded += 1
                print(f"[warm_start] pad {k}: {tuple(old_v.shape)} → {tuple(new_v.shape)}", flush=True)
            else:
                n_skipped += 1
                print(f"[warm_start] skip {k}: shape {tuple(old_v.shape)} vs {tuple(new_v.shape)}", flush=True)
        ac.load_state_dict(new_state)
        print(f"[warm_start] loaded={n_loaded} padded={n_padded} skipped={n_skipped}", flush=True)
        # Observation-normalizer buffers are part of the policy state dict and are copied
        # like any other tensor when their shapes match. If the input dimension changed,
        # the mean/var buffers are not copied (only the scalar sample count is) and re-fit
        # on the new observation distribution.

    # dump the configuration into log-directory
    # Multi-GPU: rank 0 only. Non-zero ranks never write logs/ckpts (runner's
    # disable_logs), so don't let them create stray per-rank log dirs either.
    if not getattr(args_cli, "distributed", False) or int(os.environ.get("RANK", "0")) == 0:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
