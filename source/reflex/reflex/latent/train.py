# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2: train the latent controller (privileged encoder E_phi + action decoder) with joint PPO + AMP.

Task Reflex-Latent-G1-v0.  --warm_start_ckpt takes either the Stage 1 teacher (its first actor layer is
mapped onto the latent architecture, see LatentActorCritic.warm_start_latent) or a latent checkpoint such
as the behavior-cloning initialisation from bc_init.py (loaded 1:1).  With --detector, the Stage 3 visual
encoder runs in the loop on rendered torso-camera images and writes its z into the observation, so the
decoder can be fine-tuned with PPO on predicted z.
"""

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
parser.add_argument("--detector", type=str, default=None, help="[latent 4B] det_*.pt: cameras + detector in the loop, z written into obs (decoder RL fine-tune)")
parser.add_argument("--det_width", type=int, default=192)
parser.add_argument("--det_height", type=int, default=144)
parser.add_argument("--det_fov_h", type=float, default=53.0)
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
if args_cli.detector:
    args_cli.enable_cameras = True

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
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

# ---- AMP: inject AmpPPO into the namespace where the base runner
# eval()s the algorithm class_name (no upstream rsl_rl file is modified).
import rsl_rl.runners.on_policy_runner as _opr  # noqa: E402

from reflex.amp.amp_ppo import AmpPPO  # noqa: E402

_opr.AmpPPO = AmpPPO

# ---- Stage 2: register the task id and inject the latent actor-critic class
import reflex.latent.register  # noqa: E402,F401
from reflex.latent.latent_actor_critic import LatentActorCritic, warm_start_equivalence  # noqa: E402

_opr.LatentActorCritic = LatentActorCritic

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
    log_root_path = os.path.join("logs", agent_cfg.run_name or "default")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Group stage outputs under the shared experiment run name.
    log_dir = "stage2_ppo_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    if args_cli.detector:
        # Visual encoder in the loop: one torso camera per env (same mount/FOV as the rendering and evaluation paths),
        # debug markers off, wide env spacing
        import math as _m
        import isaaclab.sim as _sim_utils
        from isaaclab.sensors import TiledCameraCfg as _TC
        _ap = 20.955; _f = _ap / 2.0 / _m.tan(_m.radians(args_cli.det_fov_h))
        env_cfg.scene.latent_cam = _TC(prim_path="{ENV_REGEX_NS}/latent_cam", height=args_cli.det_height, width=args_cli.det_width,
                                       data_types=["rgb"], spawn=_sim_utils.PinholeCameraCfg(focal_length=_f, focus_distance=400.0,
                                       horizontal_aperture=_ap, clipping_range=(0.05, 50.0)),
                                       offset=_TC.OffsetCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"))
        env_cfg.scene.env_spacing = 40.0
        def _kdv(obj, depth=0):
            if depth > 3 or obj is None: return
            for k in list(getattr(obj, "__dict__", {}).keys()):
                v = getattr(obj, k)
                if k == "debug_vis" and isinstance(v, bool): setattr(obj, k, False)
                elif hasattr(v, "__dict__") and not isinstance(v, (str, int, float)): _kdv(v, depth + 1)
        _kdv(env_cfg)
        print(f"[latent] VISUAL-ENCODER-IN-LOOP TRAINING: camera {args_cli.det_width}x{args_cli.det_height} fov_h {args_cli.det_fov_h}, encoder {args_cli.detector}", flush=True)
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
    if args_cli.detector:
        from reflex.vision.detector_model import load_detector
        from reflex.vision.detector_obs_wrapper import DetectorObsWrapper
        _det, _dck = load_detector(args_cli.detector, agent_cfg.device)
        _zlo = int(os.environ.get("BOX_LATENT_Z_FROM_OBS_LO", "-1")); assert _zlo >= 0, "set BOX_LATENT_Z_FROM_OBS_LO (e.g. 160) with --detector"
        env = DetectorObsWrapper(env, _det, int(_dck["win"]), _zlo, args_cli.det_height, args_cli.det_width)

    # AMP + latent: swap the algorithm and policy classes before runner construction
    agent_cfg.algorithm.class_name = "AmpPPO"
    agent_cfg.policy.class_name = "LatentActorCritic"
    print("[latent] policy class -> LatentActorCritic", flush=True)
    print("[AMP] algorithm class -> AmpPPO", flush=True)

    # create runner from rsl-rl
    from reflex.utils.training_time import make_timed_runner
    runner = make_timed_runner(OnPolicyRunner)(
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
        resume_path = get_checkpoint_path(
            log_root_path, f"(?=stage2_ppo_)(?:{agent_cfg.load_run})", agent_cfg.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # Warm start from a checkpoint. LatentActorCritic maps a Stage 1 teacher checkpoint
    # onto the latent architecture (or loads a latent checkpoint 1:1); the generic path
    # below zero-pads the first layer when the observation dimension grew.
    if args_cli.warm_start_ckpt is not None:
        ws_path = args_cli.warm_start_ckpt
        print(f"[INFO][warm_start] loading {ws_path}", flush=True)
        ckpt = torch.load(ws_path, map_location=agent_cfg.device, weights_only=False)
        old_state = ckpt.get("model_state_dict", ckpt)
        # rsl_rl OnPolicyRunner.alg.policy is the actor-critic.
        ac = runner.alg.policy
        if isinstance(ac, LatentActorCritic):
            cnt = ac.warm_start_latent(old_state)
            if "e_priv_lin.weight" in old_state:
                # latent -> latent continuation: every key loads with equal shape; the
                # legacy-170 equivalence proof does not apply.
                print(f"[latent][warm_start] latent->latent continuation {cnt} (no mapping)", flush=True)
                assert cnt["skipped"] == 0 and cnt["mapped"] == 0, cnt
            else:
                exact, lin = warm_start_equivalence(ac, old_state, device=agent_cfg.device)
                print(f"[latent][warm_start] {cnt}  mapping exact max|dA|={exact:.2e} (must be ~1e-5)  "
                      f"tanh-linearisation mean|dA|={lin:.4f} (info)", flush=True)
                assert exact < 1e-3, f"latent warm start mapping broken: {exact}"
            old_state = {}
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
        # BOX_LATENT_RESUME_OPT=1: true resume for latent -> latent continuations.
        # Without it the warm start builds a fresh Adam at the config learning_rate
        # (1e-3) while the checkpoint's optimizer had settled at ~1e-4: the adaptive
        # schedule then crashes to 1e-5 within a few iterations and the policy
        # collapses (most episodes end in anchor terminations) before it re-learns
        # over ~150 iterations. Loading optimizer_state_dict restores both the Adam
        # moments and the settled lr; alg.learning_rate is what the adaptive
        # schedule writes back into param_groups every update.
        if os.environ.get("BOX_LATENT_RESUME_OPT", "0") == "1":
            opt_sd = ckpt.get("optimizer_state_dict")
            opt = runner.alg.optimizer
            n_param = sum(len(g["params"]) for g in opt.param_groups)
            if opt_sd is None or len(opt_sd["state"]) != n_param:
                raise RuntimeError(f"[latent][resume_opt] cannot resume optimizer: ckpt state "
                                   f"{None if opt_sd is None else len(opt_sd['state'])} vs {n_param} params")
            lr_before = opt.param_groups[0]["lr"]
            opt.load_state_dict(opt_sd)
            lr_ck = opt.param_groups[0]["lr"]
            runner.alg.learning_rate = lr_ck
            print(f"[latent][resume_opt] OPTIMIZER RESUMED: {n_param} params, lr {lr_before:.2e} -> {lr_ck:.2e}, "
                  f"alg.learning_rate={runner.alg.learning_rate:.2e}, ckpt iter={ckpt.get('iter')}", flush=True)
        else:
            print(f"[latent][resume_opt] off: fresh Adam, lr={runner.alg.learning_rate:.2e}", flush=True)
        # BOX_LATENT_INIT_LR: start the fresh Adam at a gentle lr (the adaptive-KL schedule takes over from there).
        # A behavior-cloned z pathway must not be wrecked by the first lr=1e-3 updates.
        _ilr = float(os.environ.get("BOX_LATENT_INIT_LR", "0"))
        if _ilr > 0.0:
            runner.alg.learning_rate = _ilr
            for _g in runner.alg.optimizer.param_groups: _g["lr"] = _ilr
            print(f"[latent][init_lr] fresh Adam starts at lr={_ilr:.2e} (alg.learning_rate={runner.alg.learning_rate:.2e})", flush=True)
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
    training_completed = False
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        training_completed = True
    finally:
        if agent_cfg.logger == "wandb" and not runner.disable_logs:
            from reflex.utils.experiment_logging import finish_ppo_logging
            finish_ppo_logging(log_dir, exit_code=0 if training_completed else 1)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
