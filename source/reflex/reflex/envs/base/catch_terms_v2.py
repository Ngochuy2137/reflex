"""Catch terms used by the base env layers.

A copy of the fork's `whole_body_tracking/.../mdp/catch_extras.py` (balloon physics,
spawn events, catch rewards and observations, radius randomization, leg-stance
penalty) plus phase-gated chest-cradle rewards, start-state augmentation, reactive
hold rewards and a motion-phase mixture for resets. The base layers use
`randomize_init_state`, `randomize_cmd_time_steps_mixture` and `lean_during_intake_reward`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ============================================================================
# Balloon physics constants
# ============================================================================
BALLOON_MASS = 0.006        # kg
BALLOON_RADIUS = 0.1016     # m (4"), default radius
RHO_AIR = 1.2
CD = 0.5
A = math.pi * BALLOON_RADIUS ** 2
V = (4 / 3) * math.pi * BALLOON_RADIUS ** 3
BUOY_UP = RHO_AIR * V * 9.81
DRAG_C = 0.5 * RHO_AIR * CD * A
WIND_TAU = 0.97
WIND_DRIVE_SCALE = (0.04, 0.04, 0.001)

# Radius-randomization range (radius-finetune env)
BALLOON_RADIUS_MIN = 0.10
BALLOON_RADIUS_MAX = 0.25
BALLOON_RADIUS_SPAWN = 0.18  # mid-range spawn radius when randomizing

# Wrist bodies; their midpoint is the catch end-effector
EEF_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


# ============================================================================
# Helpers
# ============================================================================
def _eef_midpoint_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Mid-point of left and right wrist in world frame. (N, 3)."""
    robot: Articulation = env.scene["robot"]
    body_ids = [robot.body_names.index(n) for n in EEF_BODY_NAMES]
    eef_positions = robot.data.body_pos_w[:, body_ids, :]
    return eef_positions.mean(dim=1)


# ============================================================================
# Events
# ============================================================================
def apply_balloon_force(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    buoy_scale: float = 0.7,
) -> None:
    """Apply buoy + quadratic drag + OU wind every physics step.

    buoy_scale < 1 only partly cancels gravity, so the balloon reaches chest
    height within the 13 s reference motion. Also respawns balloons that reach a
    bad state (on the ground, out of bounds, too fast or NaN).
    """
    del env_ids
    balloon: RigidObject = env.scene[asset_cfg.name]
    device = balloon.device
    n = balloon.num_instances

    if not hasattr(env, "_balloon_wind"):
        env._balloon_wind = torch.zeros((n, 3), device=device)
        env._balloon_drive = torch.tensor(WIND_DRIVE_SCALE, device=device)

    # Buoyancy, drag and wind use the default-radius (0.1016 m) constants
    # regardless of the per-env radius, and the mass stays BALLOON_MASS, so
    # balloons of all sizes fall alike. Make sure the per-env radius buffer
    # exists (read by the radius observation and radius-aware rewards).
    if not hasattr(env, "_balloon_radius_per_env"):
        env._balloon_radius_per_env = torch.full((n,), BALLOON_RADIUS, device=device)

    wind_scale = float(getattr(env, "_wind_scale_override", 1.0))
    drive = (1.0 - WIND_TAU) * torch.randn((n, 3), device=device) * env._balloon_drive * wind_scale
    env._balloon_wind = WIND_TAU * env._balloon_wind + drive

    v = balloon.data.root_lin_vel_w
    speed = torch.linalg.norm(v, dim=-1, keepdim=True)

    f_buoy = torch.zeros_like(v)
    f_buoy[:, 2] = BUOY_UP * buoy_scale

    f_drag = -DRAG_C * speed * v
    f_extra = f_buoy + f_drag + env._balloon_wind

    # NaN guard before applying force (prevents PhysX explosion that NaNs obs).
    f_extra = torch.nan_to_num(f_extra, nan=0.0, posinf=0.0, neginf=0.0)

    forces = f_extra.unsqueeze(1)
    torques = torch.zeros_like(forces)
    balloon.set_external_force_and_torque(forces, torques)

    # Auto-respawn balloons that hit the ground, fly off or NaN. Without this, a
    # single bad-physics env corrupts the policy gradient and PPO crashes with
    # "normal expects std >= 0".
    # Optional per-env grace period (env._balloon_respawn_grace_steps) delays the
    # respawn after landing, e.g. for rendering. 0 = immediate respawn (training).
    pos = balloon.data.root_pos_w
    z_world = pos[:, 2] - env.scene.env_origins[:, 2]
    # Bad state: z < 0.2 m, z > 30 m (high enough for tall spawns),
    # speed > 30 m/s, or NaN.
    bad = (
        (z_world < 0.2)
        | (z_world > 30.0)
        | (speed.squeeze(-1) > 30.0)
        | torch.isnan(z_world)
        | torch.isnan(speed.squeeze(-1))
    )

    grace_steps = int(getattr(env, "_balloon_respawn_grace_steps", 0))
    if grace_steps > 0:
        # Track bad-state transitions to start grace once per "bad event" only.
        if not hasattr(env, "_balloon_grace"):
            env._balloon_grace = torch.zeros(n, dtype=torch.long, device=device)
            env._balloon_was_bad = torch.zeros(n, dtype=torch.bool, device=device)
        # decrement existing grace
        env._balloon_grace = torch.clamp_min(env._balloon_grace - 1, 0)
        # newly bad: now bad but wasn't bad last step → start grace
        newly_bad = bad & ~env._balloon_was_bad
        env._balloon_grace[newly_bad] = grace_steps
        env._balloon_was_bad = bad.clone()
        # respawn: still bad AND grace exhausted
        ready = bad & (env._balloon_grace == 0)
        bad_ids = ready.nonzero(as_tuple=False).flatten()
    else:
        bad_ids = bad.nonzero(as_tuple=False).flatten()

    if len(bad_ids) > 0 and not bool(getattr(env, "_balloon_no_auto_respawn", False)):
        reset_balloon_in_motion_direction(env, bad_ids, asset_cfg)


def randomize_motion_yaw(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    yaw_range: tuple[float, float] = (-0.7854, 0.7854),  # ±45° default
) -> None:
    """Sample per-env yaw offset, push to MotionCommand AND store on env for ball spawn.

    Must run before reset_balloon_in_motion_direction. Order relative to the
    MotionCommand reset does not matter: the yaw offset only rotates the
    command's coordinate frame.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    n = len(env_ids)
    yaws = torch.rand(n, device=env.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]

    # Stash on env for downstream events (e.g. balloon spawn) to read.
    if not hasattr(env, "_motion_yaw_offset"):
        env._motion_yaw_offset = torch.zeros(env.num_envs, device=env.device)
    env._motion_yaw_offset[env_ids] = yaws

    # Push to MotionCommand so its body_pos/quat properties return rotated targets.
    motion_cmd = env.command_manager.get_term("motion")
    if hasattr(motion_cmd, "yaw_offsets"):
        motion_cmd.yaw_offsets[env_ids] = yaws


def reset_balloon_in_motion_direction(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    spawn_height: float = 2.5,
    forward_distance_range: tuple[float, float] = (1.2, 1.8),
    lateral_range: tuple[float, float] = (-0.1, 0.1),
    motion_direction: tuple[float, float] = (0.0, -1.0),
) -> None:
    """Reset balloon in the direction the motion ref walks (world frame).

    motion_direction: unit vector in world XY of where motion goes. Default
    (0, -1) matches walk_then_catch__U001 (walks -Y for ~1.7m).

    Ball spawned at env_origin + forward_dist * motion_direction + lateral
    offset perpendicular, at spawn_height.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    balloon: RigidObject = env.scene[asset_cfg.name]
    device = balloon.device
    n = len(env_ids)

    env_origins = env.scene.env_origins[env_ids]  # (n, 3)

    # Base motion direction in world frame (unrotated motion's walking direction).
    base_fwd = torch.tensor(motion_direction, device=device).float()
    base_fwd = base_fwd / (base_fwd.norm() + 1e-6)

    # Apply per-env yaw rotation to keep ball aligned with rotated motion direction.
    if hasattr(env, "_motion_yaw_offset"):
        yaws = env._motion_yaw_offset[env_ids]  # (n,)
        cos = torch.cos(yaws)
        sin = torch.sin(yaws)
        fwd_x = base_fwd[0] * cos - base_fwd[1] * sin
        fwd_y = base_fwd[0] * sin + base_fwd[1] * cos
        fwd = torch.stack([fwd_x, fwd_y], dim=-1)  # (n, 2)
    else:
        fwd = base_fwd[None, :].expand(n, -1)

    # Perpendicular in XY plane (per env)
    lat = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=-1)  # (n, 2)

    fwd_dist = forward_distance_range[0] + (
        forward_distance_range[1] - forward_distance_range[0]
    ) * torch.rand(n, device=device)
    lat_dist = lateral_range[0] + (
        lateral_range[1] - lateral_range[0]
    ) * torch.rand(n, device=device)

    spawn_xy = env_origins[:, :2] + fwd * fwd_dist[:, None] + lat * lat_dist[:, None]

    new_pose = torch.zeros((n, 7), device=device)
    new_pose[:, 0] = spawn_xy[:, 0]
    new_pose[:, 1] = spawn_xy[:, 1]
    new_pose[:, 2] = env_origins[:, 2] + spawn_height
    new_pose[:, 3] = 1.0  # quat w=1

    balloon.write_root_pose_to_sim(new_pose, env_ids=env_ids)
    balloon.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)

    if hasattr(env, "_balloon_wind"):
        env._balloon_wind[env_ids] = 0.0


# ============================================================================
# Rewards
# ============================================================================
def catch_eef_to_balloon(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Exp(-d²/σ²) where d = eef midpoint -> balloon center. Returns (N,)."""
    balloon: RigidObject = env.scene[asset_cfg.name]
    eef = _eef_midpoint_w(env)
    bp = balloon.data.root_pos_w
    d2 = torch.sum((eef - bp) ** 2, dim=-1)
    return torch.exp(-d2 / (sigma ** 2))


def catch_success_bonus(
    env: "ManagerBasedRLEnv",
    trap_radius: float = 0.25,
    speed_thresh: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Sparse 1.0 if balloon is near eef AND moving slowly. (N,)."""
    balloon: RigidObject = env.scene[asset_cfg.name]
    eef = _eef_midpoint_w(env)
    bp = balloon.data.root_pos_w
    bv = balloon.data.root_lin_vel_w
    near = torch.linalg.norm(eef - bp, dim=-1) < trap_radius
    slow = torch.linalg.norm(bv, dim=-1) < speed_thresh
    return (near & slow).float()


# ============================================================================
# Observations
# ============================================================================
def balloon_pos_pelvis_frame(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Balloon position in robot pelvis (root) local frame. (N, 3).

    Gives the policy a vector pointing from pelvis to balloon in body coords.
    Critical observation for catch behavior.
    """
    balloon: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]
    bp_w = balloon.data.root_pos_w
    rp_w = robot.data.root_pos_w
    rq_w = robot.data.root_quat_w  # wxyz
    delta_w = bp_w - rp_w
    # NaN guard: if balloon physics blew up the obs becomes garbage; clamp it
    # so the policy network doesn't see NaN inputs.
    delta_w = torch.nan_to_num(delta_w, nan=0.0, posinf=10.0, neginf=-10.0)
    # Rotate delta into body frame: q^{-1} * delta
    w, x, y, z = rq_w.unbind(-1)
    # Inverse rotation matrix entries (column-major, applied to delta_w)
    # body = R^T * world ; R = quat_to_mat(q)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y)
    r21 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)
    dx, dy, dz = delta_w.unbind(-1)
    bx = r00 * dx + r01 * dy + r02 * dz
    by = r10 * dx + r11 * dy + r12 * dz
    bz = r20 * dx + r21 * dy + r22 * dz
    return torch.stack([bx, by, bz], dim=-1)


def balloon_lin_vel_pelvis_frame(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Balloon linear velocity in robot pelvis frame. (N, 3)."""
    balloon: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]
    bv_w = balloon.data.root_lin_vel_w
    bv_w = torch.nan_to_num(bv_w, nan=0.0, posinf=10.0, neginf=-10.0)
    rq_w = robot.data.root_quat_w  # wxyz
    w, x, y, z = rq_w.unbind(-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y)
    r21 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)
    vx, vy, vz = bv_w.unbind(-1)
    bx = r00 * vx + r01 * vy + r02 * vz
    by = r10 * vx + r11 * vy + r12 * vz
    bz = r20 * vx + r21 * vy + r22 * vz
    return torch.stack([bx, by, bz], dim=-1)


def balloon_radius_obs(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Per-env balloon radius normalized to [-1, +1]. (N, 1).

    Maps r ∈ [BALLOON_RADIUS_MIN, BALLOON_RADIUS_MAX] linearly to [-1, +1].
    If randomization not enabled, defaults to BALLOON_RADIUS (mapped to its
    normalized value).
    """
    balloon: RigidObject = env.scene[asset_cfg.name]
    n = balloon.num_instances
    device = balloon.device
    if not hasattr(env, "_balloon_radius_per_env"):
        env._balloon_radius_per_env = torch.full((n,), BALLOON_RADIUS, device=device)
    r = env._balloon_radius_per_env
    center = 0.5 * (BALLOON_RADIUS_MIN + BALLOON_RADIUS_MAX)
    half = 0.5 * (BALLOON_RADIUS_MAX - BALLOON_RADIUS_MIN)
    norm = (r - center) / max(half, 1e-6)
    return norm.unsqueeze(-1)


# ============================================================================
# Reward: leg-stance penalty (anti-lunge, reasonable lateral spacing)
# ============================================================================
_LEG_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _leg_body_indices(robot: "Articulation") -> tuple[int, int]:
    """Return (L, R) ankle-roll body indices, cached on the robot."""
    if not hasattr(robot, "_leg_idx_cache"):
        body_names = robot.body_names
        L = body_names.index(_LEG_BODY_NAMES[0])
        R = body_names.index(_LEG_BODY_NAMES[1])
        robot._leg_idx_cache = (L, R)
    return robot._leg_idx_cache


def leg_stance_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lateral_max: float = 0.30,
) -> torch.Tensor:
    """Penalty term: −(fore_aft_diff² + max(0, lateral_diff − lateral_max)²).

    Computed in robot pelvis (root) body frame. Returns negative, weight is
    applied in env_cfg. This pushes the policy to avoid lunge stance and to
    keep lateral spacing within `lateral_max` (default 30 cm).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    L_idx, R_idx = _leg_body_indices(robot)
    pelvis_pos_w = robot.data.root_pos_w
    pelvis_quat_w = robot.data.root_quat_w  # wxyz
    L_w = robot.data.body_pos_w[:, L_idx, :]
    R_w = robot.data.body_pos_w[:, R_idx, :]

    delta_L_w = L_w - pelvis_pos_w
    delta_R_w = R_w - pelvis_pos_w
    delta_L_w = torch.nan_to_num(delta_L_w, nan=0.0, posinf=10.0, neginf=-10.0)
    delta_R_w = torch.nan_to_num(delta_R_w, nan=0.0, posinf=10.0, neginf=-10.0)

    w, x, y, z = pelvis_quat_w.unbind(-1)
    # Body frame = R^T * world (R from quat, see balloon_pos_pelvis_frame)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)

    dxL, dyL, dzL = delta_L_w.unbind(-1)
    dxR, dyR, dzR = delta_R_w.unbind(-1)
    L_bx = r00 * dxL + r01 * dyL + r02 * dzL
    L_by = r10 * dxL + r11 * dyL + r12 * dzL
    R_bx = r00 * dxR + r01 * dyR + r02 * dzR
    R_by = r10 * dxR + r11 * dyR + r12 * dzR

    fore_aft_diff = L_bx - R_bx  # body-frame fore-aft offset between L and R
    lateral_diff = torch.abs(L_by - R_by)

    fore_aft_pen = fore_aft_diff ** 2
    lateral_excess = torch.clamp(lateral_diff - lateral_max, min=0.0)
    lateral_pen = lateral_excess ** 2
    return -(fore_aft_pen + lateral_pen)


# ============================================================================
# Event: per-env balloon radius randomization (called at reset)
# ============================================================================
def randomize_balloon_radius(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    radius_range: tuple[float, float] = (BALLOON_RADIUS_MIN, BALLOON_RADIUS_MAX),
    spawn_radius: float = BALLOON_RADIUS_SPAWN,
) -> None:
    """Sample a new balloon radius per env at reset.

    Stores per-env radius in `env._balloon_radius_per_env` (N,), read by
    `balloon_radius_obs` and the radius-aware reward thresholds.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    balloon: RigidObject = env.scene[asset_cfg.name]
    device = balloon.device
    n_total = balloon.num_instances

    if not hasattr(env, "_balloon_radius_per_env"):
        env._balloon_radius_per_env = torch.full((n_total,), spawn_radius, device=device)

    n = len(env_ids)
    # Optional pin: if env has _radius_pin_per_env, use those values verbatim
    # (for evaluation rendering — fixed radii instead of random).
    if hasattr(env, "_radius_pin_per_env"):
        pinned = env._radius_pin_per_env
        # Broadcast / index per env
        if pinned.numel() == n_total:
            new_r = pinned[env_ids].clone()
        else:
            new_r = pinned[: n].clone() if pinned.numel() >= n else pinned[:1].repeat(n)
    else:
        new_r = radius_range[0] + (radius_range[1] - radius_range[0]) * torch.rand(n, device=device)
    env._balloon_radius_per_env[env_ids] = new_r

    # Write the USD Sphere `radius` attribute of each env's prim. Caveat: after
    # the scene is created this resizes the rendered sphere, but the PhysX
    # collision shape keeps its spawn-time radius.
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        env_ids_cpu = env_ids.detach().cpu().tolist()
        new_r_cpu = new_r.detach().cpu().tolist()
        n_set = 0
        for eid, r_i in zip(env_ids_cpu, new_r_cpu):
            sphere_prim = stage.GetPrimAtPath(
                f"/World/envs/env_{eid}/Balloon/geometry/mesh"
            )
            if not sphere_prim or not sphere_prim.IsValid():
                continue
            attr = sphere_prim.GetAttribute("radius")
            if attr and attr.IsValid():
                attr.Set(float(r_i))
                n_set += 1
        if not getattr(env, "_radius_set_logged", False):
            print(f"[randomize_balloon_radius] set radius on {n_set}/{len(env_ids_cpu)} prims", flush=True)
            env._radius_set_logged = True
    except Exception as _e:
        if not getattr(env, "_radius_scale_warned", False):
            print(f"[randomize_balloon_radius] WARN radius set failed: {_e!r}", flush=True)
            env._radius_scale_warned = True


# ============================================================================
# Phase-gated chest-cradle rewards.
#
# Phases (active per env per step):
#   INTAKE      — ball within 0.30 m of either wrist but not yet within 0.20 m
#                 of the chest. Policy should FUNNEL ball toward chest + lean back.
#   POST_CATCH  — ball within 0.20 m of the chest (proxy for caught). Policy
#                 should recover the torso upright; legs are free (no joint penalty).
# ============================================================================

# G1 chest body.
CHEST_BODY_NAME = "torso_link"

# Discrete radii used at training reset.
BALLOON_RADII_DISCRETE = (0.10, 0.15, 0.20, 0.25)

# Phase thresholds.
_ARM_ZONE_R = 0.30       # m, ball ≤ this from either eef → in arm zone
_AT_CHEST_R = 0.20       # m, ball ≤ this from chest → at chest
_LEAN_TARGET_PITCH = -0.35  # rad (~-20°), torso lean-back target during INTAKE
_SIGMA_FUNNEL = 0.25
_SIGMA_LEAN = 0.15
_SIGMA_UPRIGHT = 0.20


def _chest_idx(robot: Articulation) -> int:
    return robot.body_names.index(CHEST_BODY_NAME)


def _torso_pitch(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Pelvis (root) pitch in radians, (N,). + = lean forward, − = lean back."""
    robot: Articulation = env.scene["robot"]
    q = robot.data.root_quat_w  # wxyz, (N, 4)
    w, x, y, z = q.unbind(-1)
    # Pitch from quaternion (ZYX intrinsic). Standard formula.
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    return torch.asin(sinp)


def _ball_arm_dist(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """min(||ball − eef_L||, ||ball − eef_R||) per env, (N,)."""
    robot: Articulation = env.scene["robot"]
    balloon: RigidObject = env.scene["balloon"]
    body_ids = [robot.body_names.index(n) for n in EEF_BODY_NAMES]
    eefs = robot.data.body_pos_w[:, body_ids, :]      # (N, 2, 3)
    bp = balloon.data.root_pos_w.unsqueeze(1)          # (N, 1, 3)
    d = torch.linalg.norm(eefs - bp, dim=-1)           # (N, 2)
    return d.min(dim=-1).values


def _ball_chest_dist(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """||ball − chest|| per env, (N,)."""
    robot: Articulation = env.scene["robot"]
    balloon: RigidObject = env.scene["balloon"]
    chest = robot.data.body_pos_w[:, _chest_idx(robot), :]
    bp = balloon.data.root_pos_w
    return torch.linalg.norm(chest - bp, dim=-1)


def _phase_intake(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Mask (N,) bool: ball in arm zone AND not yet at chest."""
    in_arm = _ball_arm_dist(env) < _ARM_ZONE_R
    at_chest = _ball_chest_dist(env) < _AT_CHEST_R
    return in_arm & (~at_chest)


def _phase_post_catch(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Mask (N,) bool: ball at chest (proxy for caught)."""
    return _ball_chest_dist(env) < _AT_CHEST_R


# ----------------------------------------------------------------------------
# Reward 1: FUNNEL — ball → chest during INTAKE.
# ----------------------------------------------------------------------------
def chest_funnel_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Maximize ball-to-chest proximity during INTAKE phase. (N,) in [0, 1]."""
    d = _ball_chest_dist(env)
    r = torch.exp(-(d ** 2) / (_SIGMA_FUNNEL ** 2))
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r * _phase_intake(env).float()


# ----------------------------------------------------------------------------
# Reward 2: LEAN — torso back-lean during INTAKE only.
# ----------------------------------------------------------------------------
def lean_during_intake_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Encourage torso pitch ≈ -20° while INTAKE active. (N,) in [0, 1]."""
    pitch = _torso_pitch(env)
    r = torch.exp(-((pitch - _LEAN_TARGET_PITCH) ** 2) / (_SIGMA_LEAN ** 2))
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r * _phase_intake(env).float()


# ----------------------------------------------------------------------------
# Reward 3: UPRIGHT RECOVER — torso pitch → 0 during POST_CATCH.
# ----------------------------------------------------------------------------
def upright_recover_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Encourage torso pitch ≈ 0 once caught. Soft (low weight) so legs
    retain freedom to compensate. (N,) in [0, 1].
    """
    pitch = _torso_pitch(env)
    r = torch.exp(-(pitch ** 2) / (_SIGMA_UPRIGHT ** 2))
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r * _phase_post_catch(env).float()


# ============================================================================
# Discrete-radius randomization at reset.
# ============================================================================
def randomize_balloon_radius_discrete(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    radii: tuple[float, ...] = BALLOON_RADII_DISCRETE,
    spawn_radius: float = BALLOON_RADIUS_SPAWN,
) -> None:
    """Sample new balloon radius from a DISCRETE set per env at reset.

    Mirrors `randomize_balloon_radius` but draws from `radii` (default
    {0.10, 0.15, 0.20, 0.25}). Same caveat: the USD Sphere `radius` attribute
    written here resizes the rendered sphere, not the PhysX collision shape.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    balloon: RigidObject = env.scene[asset_cfg.name]
    device = balloon.device
    n_total = balloon.num_instances
    if not hasattr(env, "_balloon_radius_per_env"):
        env._balloon_radius_per_env = torch.full((n_total,), spawn_radius, device=device)

    n = len(env_ids)
    if hasattr(env, "_radius_pin_per_env"):
        pinned = env._radius_pin_per_env
        if pinned.numel() == n_total:
            new_r = pinned[env_ids].clone()
        else:
            new_r = pinned[:n].clone() if pinned.numel() >= n else pinned[:1].repeat(n)
    else:
        idx = torch.randint(0, len(radii), (n,), device=device)
        radii_t = torch.tensor(radii, device=device, dtype=torch.float32)
        new_r = radii_t[idx]
    env._balloon_radius_per_env[env_ids] = new_r

    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        env_ids_cpu = env_ids.detach().cpu().tolist()
        new_r_cpu = new_r.detach().cpu().tolist()
        n_set = 0
        for eid, r_i in zip(env_ids_cpu, new_r_cpu):
            sphere_prim = stage.GetPrimAtPath(
                f"/World/envs/env_{eid}/Balloon/geometry/mesh"
            )
            if not sphere_prim or not sphere_prim.IsValid():
                continue
            attr = sphere_prim.GetAttribute("radius")
            if attr and attr.IsValid():
                attr.Set(float(r_i))
                n_set += 1
        if not getattr(env, "_radius_set_logged_v3", False):
            print(f"[randomize_balloon_radius_discrete] set radius on "
                  f"{n_set}/{len(env_ids_cpu)} prims (radii={radii})", flush=True)
            env._radius_set_logged_v3 = True
    except Exception as _e:
        if not getattr(env, "_radius_scale_warned_v3", False):
            print(f"[randomize_balloon_radius_discrete] WARN: {_e!r}", flush=True)
            env._radius_scale_warned_v3 = True


# ============================================================================
# Start-state augmentation at reset.
#
# Perturb the robot's joint pose and root xy velocity at each episode reset so
# the policy also learns from start states off the reference motion and does not
# over-rely on a clean initial pose.
# ============================================================================
def randomize_init_state(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    joint_pose_std: float = 0.3,
    root_vel_xy_max: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset-time noise on joint pose + root velocity.

    Joint pose: current pose + N(0, joint_pose_std²) per joint.
    Root velocity (xy): uniform in [-root_vel_xy_max, +root_vel_xy_max].
    Root velocity z and ang vel: kept at 0.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    robot: Articulation = env.scene[asset_cfg.name]
    device = env.device
    n = len(env_ids)
    # Perturb joint positions (writing AFTER other reset events set defaults).
    cur_jp = robot.data.joint_pos[env_ids].clone()
    noise = torch.randn_like(cur_jp) * joint_pose_std
    new_jp = cur_jp + noise
    robot.write_joint_state_to_sim(
        new_jp, torch.zeros_like(new_jp), env_ids=env_ids,
    )
    # Perturb root xy velocity. Uniform in [-vmax, +vmax] per axis.
    new_root_vel = torch.zeros((n, 6), device=device)
    new_root_vel[:, 0] = (torch.rand(n, device=device) - 0.5) * 2 * root_vel_xy_max
    new_root_vel[:, 1] = (torch.rand(n, device=device) - 0.5) * 2 * root_vel_xy_max
    robot.write_root_velocity_to_sim(new_root_vel, env_ids=env_ids)
    if not getattr(env, "_init_aug_logged", False):
        print(f"[randomize_init_state] joint_pose_std={joint_pose_std} "
              f"root_vel_xy_max={root_vel_xy_max} applied to "
              f"{n}/{robot.num_instances} envs", flush=True)
        env._init_aug_logged = True


# ============================================================================
# Reactive hold rewards.
#
# Reward "ball is near the robot's arms/chest AND moving slowly", i.e. the robot
# is actually holding the ball rather than just being in position, plus a
# persistence bonus for a sustained hold.
#
# All gates are state-based (ball/robot positions), not motion-clock-based.
# Together with the motion-phase and yaw augmentation, this decouples the reward
# from the reference-motion phase, so the policy learns a reactive grip.
# ============================================================================
def reactive_hold_reward(
    env: "ManagerBasedRLEnv",
    near_eef_r: float = 0.30,
    near_chest_r: float = 0.35,
    v_sigma: float = 0.6,
) -> torch.Tensor:
    """Reward = (slow ball) when ball is near arms or chest. Pure state-based.

    Returns (N,) in [0, 1]. Multiply by reward weight in env cfg.
    """
    robot: Articulation = env.scene["robot"]
    balloon: RigidObject = env.scene["balloon"]
    eef_ids = [robot.body_names.index(n) for n in EEF_BODY_NAMES]
    chest_idx = robot.body_names.index(CHEST_BODY_NAME)

    bp = balloon.data.root_pos_w
    bv = balloon.data.root_lin_vel_w
    eef_mid = robot.data.body_pos_w[:, eef_ids, :].mean(dim=1)
    chest = robot.data.body_pos_w[:, chest_idx, :]

    d_eef = torch.linalg.norm(bp - eef_mid, dim=-1)
    d_chest = torch.linalg.norm(bp - chest, dim=-1)
    near = (d_eef < near_eef_r) | (d_chest < near_chest_r)

    speed = torch.linalg.norm(bv, dim=-1)
    slow = torch.exp(-(speed ** 2) / (v_sigma ** 2))
    r = slow * near.float()
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def reactive_hold_persistence(
    env: "ManagerBasedRLEnv",
    near_eef_r: float = 0.30,
    near_chest_r: float = 0.35,
    v_thresh: float = 0.8,
) -> torch.Tensor:
    """Counts consecutive frames where (near & slow). Caps at 100 (2s @ 50Hz).

    Persistence is encoded in env._reactive_hold_counter (B,), incremented
    each call if conditions met, reset on miss. Reward returned = counter / 100.
    """
    robot: Articulation = env.scene["robot"]
    balloon: RigidObject = env.scene["balloon"]
    eef_ids = [robot.body_names.index(n) for n in EEF_BODY_NAMES]
    chest_idx = robot.body_names.index(CHEST_BODY_NAME)
    bp = balloon.data.root_pos_w
    bv = balloon.data.root_lin_vel_w
    eef_mid = robot.data.body_pos_w[:, eef_ids, :].mean(dim=1)
    chest = robot.data.body_pos_w[:, chest_idx, :]

    d_eef = torch.linalg.norm(bp - eef_mid, dim=-1)
    d_chest = torch.linalg.norm(bp - chest, dim=-1)
    near = (d_eef < near_eef_r) | (d_chest < near_chest_r)
    slow = torch.linalg.norm(bv, dim=-1) < v_thresh
    cond = near & slow

    B = bp.shape[0]
    device = bp.device
    if not hasattr(env, "_reactive_hold_counter") or env._reactive_hold_counter.shape[0] != B:
        env._reactive_hold_counter = torch.zeros((B,), dtype=torch.long, device=device)

    env._reactive_hold_counter = torch.where(
        cond,
        torch.clamp(env._reactive_hold_counter + 1, max=100),
        torch.zeros((B,), dtype=torch.long, device=device),
    )
    return env._reactive_hold_counter.float() / 100.0


# ============================================================================
# Motion-phase mixture randomization at reset.
# ============================================================================
def randomize_cmd_time_steps_mixture(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    motion_total_frames: int = 1500,
    p_normal: float = 0.5,
    p_catch_window: float = 0.3,
    p_global: float = 0.2,
    catch_window: tuple[int, int] = (250, 800),
) -> None:
    """At each reset, randomize cmd.time_steps via a mixture:
       p_normal: keep current value (let IsaacLab default sampling decide)
       p_catch_window: sample uniform in [catch_window]
       p_global: sample uniform in [0, motion_total_frames-1]
    This breaks motion-phase ↔ ball-state correlation during training.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    try:
        cmd = env.command_manager.get_term("motion")
        if not hasattr(cmd, "time_steps"):
            return
    except Exception:
        return

    n = len(env_ids)
    device = env.device
    rand = torch.rand(n, device=device)
    new_ts = cmd.time_steps[env_ids].clone()
    catch_mask = (rand >= p_normal) & (rand < p_normal + p_catch_window)
    global_mask = rand >= p_normal + p_catch_window
    if catch_mask.any():
        sampled = torch.randint(catch_window[0], catch_window[1], (int(catch_mask.sum()),), device=device)
        new_ts[catch_mask] = sampled
    if global_mask.any():
        sampled = torch.randint(0, motion_total_frames, (int(global_mask.sum()),), device=device)
        new_ts[global_mask] = sampled
    cmd.time_steps[env_ids] = new_ts
    if not getattr(env, "_phase_mix_logged", False):
        print(f"[randomize_cmd_time_steps_mixture] mix={p_normal}/{p_catch_window}/{p_global} "
              f"catch_window={catch_window} applied to {n} envs", flush=True)
        env._phase_mix_logged = True
