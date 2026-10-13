from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def reset_from_catch_end_states(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    end_states_path: str,
    joint_pos_noise_std: float = 0.02,
    joint_vel_noise_std: float = 0.05,
    motion_command_name: str = "motion",
):
    """Reset robot to a randomly-sampled BM catch end-state.

    Loads bm_catch_end_states.pt (list of dicts with joint_pos, joint_vel,
    root_pos_local, root_quat_w, root_lin_vel_w, root_ang_vel_w) on first
    call, caches in env._release_init_states. Each reset, samples one state
    per env_id with small Gaussian noise, writes to robot.

    Also resets the motion command's time_step to 0 so the release tracker
    sees motion[0] (the standing rest pose) as ref at the start of episode.
    """
    if not hasattr(env, "_release_init_states"):
        states = torch.load(end_states_path, weights_only=False)
        if len(states) == 0:
            print(f"[release_init] WARN: {end_states_path} is empty", flush=True)
            return
        env._release_init_states = {
            "joint_pos": torch.stack([s["joint_pos"] for s in states]).to(env.device),
            "joint_vel": torch.stack([s["joint_vel"] for s in states]).to(env.device),
            "root_pos_local": torch.stack([s["root_pos_local"] for s in states]).to(env.device),
            "root_quat_w": torch.stack([s["root_quat_w"] for s in states]).to(env.device),
            "root_lin_vel_w": torch.stack([s["root_lin_vel_w"] for s in states]).to(env.device),
            "root_ang_vel_w": torch.stack([s["root_ang_vel_w"] for s in states]).to(env.device),
        }
        print(f"[release_init] loaded {len(states)} catch end-states from {end_states_path}", flush=True)

    if env_ids is None or len(env_ids) == 0:
        return

    cache = env._release_init_states
    n_states = cache["joint_pos"].shape[0]
    idx = torch.randint(0, n_states, (len(env_ids),), device=env.device)

    joint_pos = cache["joint_pos"][idx]
    joint_vel = cache["joint_vel"][idx]
    root_pos_local = cache["root_pos_local"][idx]
    root_quat = cache["root_quat_w"][idx]
    root_lin_vel = cache["root_lin_vel_w"][idx]
    root_ang_vel = cache["root_ang_vel_w"][idx]

    # Add small noise (helps generalization beyond the 100-sample point cloud).
    if joint_pos_noise_std > 0:
        joint_pos = joint_pos + torch.randn_like(joint_pos) * joint_pos_noise_std
    if joint_vel_noise_std > 0:
        joint_vel = joint_vel + torch.randn_like(joint_vel) * joint_vel_noise_std

    root_pos_w = root_pos_local + env.scene.env_origins[env_ids]

    asset: Articulation = env.scene[asset_cfg.name]
    soft_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.write_root_state_to_sim(
        torch.cat([root_pos_w, root_quat, root_lin_vel, root_ang_vel], dim=-1),
        env_ids=env_ids,
    )

    # Reset motion command time_steps so release ref starts at frame 0
    try:
        cmd = env.command_manager.get_term(motion_command_name)
        if hasattr(cmd, "time_steps"):
            cmd.time_steps[env_ids] = 0
    except Exception:
        pass
