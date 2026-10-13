"""Stage 2: stateless privileged box observation, the input of the privileged encoder E_phi.

Why stateless: the Stage 3 visual encoder sees only the last WIN camera frames and is trained
to regress z* = E_phi(this vector).  The vector may therefore contain only quantities that are
computable from those WIN frames plus fixed constants -- no Kalman-filter state, no time since
launch, no episode history.  Every value below is a function of the ground-truth box state in
the frame window [t-k-WIN+1, t-k] and of the per-frame visibility of the box, nothing else.

Layout (PRIV_DIM = 11, or 26 with BOX_LATENT_PRIV_RICH=1), in the pelvis frame like the legacy
channels obs[160:166], or in the camera frame with BOX_LATENT_PRIV_FRAME=camera:
  [0:3]   box position at the freshest visible frame in the window
  [3:6]   box velocity at that frame
  [6]     age of that frame relative to t-k, in frames / WIN   (0 .. 1)
  [7]     visibility flag: 1 if the box was visible in any frame of the window
  [8:11]  box half extents (constant per episode; the camera sees the size)
  [11:26] rich variant only: symmetry-invariant orientation and angular velocity (see below)
  An invisible window gives pos/vel 0, age 1, vis 0.

k (pipeline latency, in frames) is sampled per episode in [DELAY_LO, DELAY_HI] and is NOT
exposed: the policy must be robust to it.  At label time k is the known render-pipeline latency.

Per-frame visibility = (box inside the camera field of view: a cone of half-angle FOV_DEG, or a
FOV_H_DEG x FOV_V_DEG rectangle) & (range > MIN_RANGE) & (rand > DROP_P).  The camera is the same
torso mount the state estimator uses (CAM_OFF / CAM_PITCH_UP from reflex.envs.robust_estimator),
so a render camera placed there reproduces the geometry.

All knobs are BOX_LATENT_* environment variables; the active configuration is printed once on
the first tick.
"""
from __future__ import annotations

import math
import os

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

from reflex.envs import throw as _bt
from reflex.envs.robust_estimator import CAM_OFF, CAM_PITCH_UP

# BOX_LATENT_PRIV_RICH=1: the E_phi input also carries the box orientation and angular velocity of the same
# delayed freshest-visible frame, in the same frame as pos/vel.  Orientation is encoded in a form that is invariant
# to the cuboid's 180-degree symmetries (an image cannot tell a box from its flipped copy): for the long axis
# a = R e_x and the short axis c = R e_z, the 6 unique entries of a a^T and of c c^T.
# Layout: [0:11) as above | [11:17) long-axis outer product | [17:23) short-axis outer product | [23:26) angular velocity / 10.
LAT_PRIV_RICH = os.environ.get("BOX_LATENT_PRIV_RICH", "0") == "1"
PRIV_DIM = 26 if LAT_PRIV_RICH else 11
LAT_WIN = int(os.environ.get("BOX_LATENT_WIN", "5"))
LAT_DELAY_LO = int(os.environ.get("BOX_LATENT_DELAY_LO", "1"))
LAT_DELAY_HI = int(os.environ.get("BOX_LATENT_DELAY_HI", "3"))
LAT_FOV_DEG = float(os.environ.get("BOX_LATENT_FOV_DEG", "40"))
# Optional rectangular field of view (half-angles, deg) matching a real camera (OAK-D Pro W, 105.9 x 79.4 deg
# full angles -> 53 x 40).  Unset = a single cone of half-angle LAT_FOV_DEG.
LAT_FOV_H_DEG = float(os.environ.get("BOX_LATENT_FOV_H_DEG", "0"))
LAT_FOV_V_DEG = float(os.environ.get("BOX_LATENT_FOV_V_DEG", "0"))
LAT_DROP_P = float(os.environ.get("BOX_LATENT_DROP_P", "0.10"))
LAT_MIN_RANGE = float(os.environ.get("BOX_LATENT_MIN_RANGE", "0.25"))
# Frame of the stored pos/vel.  "pelvis" = box pos/vel in the pelvis frame (legacy layout);
# "camera" = box position in the camera frame (torso_link + CAM_OFF, pitched up) and box velocity relative to the
# camera, in the camera frame.  Only the camera variant is a pure function of the image history: during flight the
# torso rotates 5-8 deg on average (17 deg p90) relative to the pelvis, i.e. ~0.2 m of label ambiguity for an
# image-only encoder.
LAT_PRIV_FRAME = os.environ.get("BOX_LATENT_PRIV_FRAME", "pelvis")
assert LAT_PRIV_FRAME in ("pelvis", "camera"), LAT_PRIV_FRAME
_HIST = LAT_DELAY_HI + LAT_WIN          # ring length, newest at index 0

_PROOF = False


def _state(env) -> dict:
    if not hasattr(env, "_latent_state"):
        n, dev = env.num_envs, env.device
        env._latent_state = {
            "pos": torch.zeros(n, _HIST, 3, device=dev),
            "vel": torch.zeros(n, _HIST, 3, device=dev),
            "vis": torch.zeros(n, _HIST, dtype=torch.bool, device=dev),
            "ori": torch.zeros(n, _HIST, 15, device=dev),
            "k": torch.full((n,), LAT_DELAY_LO, dtype=torch.long, device=dev),
            "fill": torch.ones(n, dtype=torch.bool, device=dev),
            "stepped": torch.full((1,), -1, dtype=torch.long, device=dev),
            "torso_i": list(env.scene["robot"].data.body_names).index("torso_link"),
        }
    return env._latent_state


def latent_reset(env, env_ids: torch.Tensor,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")) -> None:
    """Reset event: resample the latency k and mark the ring for a refill."""
    s = _state(env)
    n = env_ids.numel()
    s["k"][env_ids] = torch.randint(LAT_DELAY_LO, LAT_DELAY_HI + 1, (n,), device=env.device)
    s["fill"][env_ids] = True


def _tick(env, asset_cfg: SceneEntityCfg) -> dict:
    """Push one ground-truth sample into the ring buffer.  Idempotent within an env step."""
    global _PROOF
    s = _state(env)
    step_now = int(env.common_step_counter)
    if int(s["stepped"][0].item()) == step_now:
        return s
    s["stepped"][:] = step_now
    if not _PROOF:
        _PROOF = True
        print(f"[latent_obs] STATELESS PRIV FEATURES ACTIVE: dim={PRIV_DIM} win={LAT_WIN} "
              f"delay k in [{LAT_DELAY_LO},{LAT_DELAY_HI}] fov={LAT_FOV_DEG} deg "
              f"{'RECT fov_h=%g fov_v=%g ' % (LAT_FOV_H_DEG, LAT_FOV_V_DEG) if LAT_FOV_H_DEG > 0 and LAT_FOV_V_DEG > 0 else ''}"
              f"{'RICH(+sym-invariant orientation 12 + angvel 3) ' if LAT_PRIV_RICH else ''}"
              f"drop_p={LAT_DROP_P} min_range={LAT_MIN_RANGE} m (FRAME={LAT_PRIV_FRAME}: " + ("pos in camera frame, vel relative to camera" if LAT_PRIV_FRAME == "camera" else "pelvis frame") + ")", flush=True)

    robot: Articulation = env.scene["robot"]
    box: RigidObject = env.scene[asset_cfg.name]
    dev, n = env.device, env.num_envs
    rq = robot.data.root_quat_w
    rp = robot.data.root_pos_w
    bp = torch.nan_to_num(box.data.root_pos_w, nan=0.0, posinf=10.0, neginf=-10.0)
    bv = torch.nan_to_num(box.data.root_lin_vel_w, nan=0.0, posinf=10.0, neginf=-10.0)
    # visibility from the estimator's camera mount (torso_link + CAM_OFF, pitched up)
    ti = s["torso_i"]
    tq = robot.data.body_quat_w[:, ti]
    cam = robot.data.body_pos_w[:, ti] + quat_apply(tq, torch.tensor(CAM_OFF, device=dev).expand(n, 3))
    cq_ = quat_mul(tq, torch.tensor([math.cos(-CAM_PITCH_UP / 2), 0.0, math.sin(-CAM_PITCH_UP / 2), 0.0], device=dev).expand(n, 4))
    if LAT_PRIV_FRAME == "camera":
        cam_v = robot.data.body_lin_vel_w[:, ti]                      # camera ego-motion (lever arm of CAM_OFF ignored)
        pos_b = quat_apply_inverse(cq_, bp - cam)
        vel_b = quat_apply_inverse(cq_, bv - cam_v)
    else:
        pos_b = quat_apply_inverse(rq, bp - rp)
        vel_b = quat_apply_inverse(rq, bv)
    axis = quat_apply(tq, torch.tensor([math.cos(CAM_PITCH_UP), 0.0, math.sin(CAM_PITCH_UP)],
                                       device=dev).expand(n, 3))
    d = bp - cam
    rng = d.norm(dim=-1).clamp(min=1e-4)
    if LAT_FOV_H_DEG > 0.0 and LAT_FOV_V_DEG > 0.0:
        # rectangular FOV in the camera frame: x = optical axis, y = left, z = up
        # (camera = torso frame pitched up by CAM_PITCH_UP about torso y)
        dc = quat_apply_inverse(cq_, d)
        fwd = dc[:, 0].clamp(min=1e-4)
        in_h = torch.rad2deg(torch.atan2(dc[:, 1].abs(), fwd)) < LAT_FOV_H_DEG
        in_v = torch.rad2deg(torch.atan2(dc[:, 2].abs(), fwd)) < LAT_FOV_V_DEG
        in_fov = in_h & in_v & (dc[:, 0] > 0.0)
    else:
        cos = ((d / rng[:, None]) * axis).sum(-1).clamp(-1.0, 1.0)
        in_fov = torch.rad2deg(torch.acos(cos)) < LAT_FOV_DEG
    vis = in_fov & (rng > LAT_MIN_RANGE) & (torch.rand(n, device=dev) >= LAT_DROP_P)

    if LAT_PRIV_RICH:
        bq = box.data.root_quat_w
        bw = torch.nan_to_num(box.data.root_ang_vel_w, nan=0.0, posinf=50.0, neginf=-50.0)
        def _outer6(a):
            return torch.stack([a[:, 0] * a[:, 0], a[:, 1] * a[:, 1], a[:, 2] * a[:, 2],
                                a[:, 0] * a[:, 1], a[:, 0] * a[:, 2], a[:, 1] * a[:, 2]], dim=-1)
        ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        fq = cq_ if LAT_PRIV_FRAME == "camera" else rq
        a_long = quat_apply_inverse(fq, quat_apply(bq, ex))
        a_short = quat_apply_inverse(fq, quat_apply(bq, ez))
        ori_b = torch.nan_to_num(torch.cat([_outer6(a_long), _outer6(a_short), quat_apply_inverse(fq, bw) / 10.0], dim=-1))
        s["ori"] = torch.roll(s["ori"], 1, dims=1)
        s["ori"][:, 0] = ori_b
    s["pos"] = torch.roll(s["pos"], 1, dims=1)
    s["vel"] = torch.roll(s["vel"], 1, dims=1)
    s["vis"] = torch.roll(s["vis"], 1, dims=1)
    s["pos"][:, 0] = pos_b
    s["vel"][:, 0] = vel_b
    s["vis"][:, 0] = vis
    f = s["fill"]
    if bool(f.any()):
        s["pos"][f] = pos_b[f][:, None, :].expand(-1, _HIST, -1)
        s["vel"][f] = vel_b[f][:, None, :].expand(-1, _HIST, -1)
        s["vis"][f] = vis[f][:, None].expand(-1, _HIST)
        if LAT_PRIV_RICH:
            s["ori"][f] = ori_b[f][:, None, :].expand(-1, _HIST, -1)
        s["fill"][:] = False
    return s


def box_priv_stateless(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")) -> torch.Tensor:
    """Policy observation term of shape (num_envs, PRIV_DIM): the input of E_phi.  See the module docstring."""
    s = _tick(env, asset_cfg)
    n, dev = env.num_envs, env.device
    idx = s["k"][:, None] + torch.arange(LAT_WIN, device=dev)[None, :]        # (n, WIN) ring indices
    vis_w = torch.gather(s["vis"], 1, idx)                                     # (n, WIN)
    any_vis = vis_w.any(dim=1)
    first = torch.argmax(vis_w.long(), dim=1)                                  # 0 when none (masked below)
    sel = torch.gather(idx, 1, first[:, None])                                 # (n, 1)
    pos = torch.gather(s["pos"], 1, sel[:, :, None].expand(-1, 1, 3))[:, 0]
    vel = torch.gather(s["vel"], 1, sel[:, :, None].expand(-1, 1, 3))[:, 0]
    age = first.float() / float(LAT_WIN)
    m = any_vis[:, None].float()
    pos = pos * m
    vel = vel * m
    age = torch.where(any_vis, age, torch.ones_like(age))
    he = env._box_half_extents if hasattr(env, "_box_half_extents") else \
        torch.tensor([x * 0.5 for x in _bt.BOX_SIZE], device=dev).expand(n, 3)
    if LAT_PRIV_RICH:
        ori = torch.gather(s["ori"], 1, sel[:, :, None].expand(-1, 1, 15))[:, 0] * m
        return torch.cat([pos, vel, age[:, None], m, he, ori], dim=-1)
    return torch.cat([pos, vel, age[:, None], m, he], dim=-1)
