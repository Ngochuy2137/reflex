"""Safety/robustness reward terms and the walking thrower (Stage 1 teacher).

Trust-gated catch rewards, low-trust stillness, soft-fall shaping, a pre-clamp
command-limit penalty, a one-shot fall penalty, and a hover/launch event whose
release point random-walks before the throw.  Wired up in robust_env_cfg.
"""
# Terms and scale accounting (living reward ~0.17/step; the pre-existing
# penalties already sum to ~-0.18/step, so each new term stays below that):
#   1. trust-gated catch rewards: catching a box the policy cannot see pays
#      ~nothing, so blind lunges are not rewarded.  Multiplier in [floor, 1],
#      never negative.
#   2. low-trust stillness (w=+1.5): reward for staying calm while the box
#      estimate is untrustworthy; <= +0.03/step and income-only, so it never
#      makes early termination attractive.
#   3. soft-fall shaping (w=-2.0): while toppling, joint speed is priced
#      (learn to damp, don't thrash into the ground).  Zero unless the tilt
#      ramp is > 0 (up_z < 0.87, ~30 deg) AND |dq| > 8 rad/s; a 20 rad/s thrash
#      while falling costs ~-0.048/step.
#   4. command-limit excess (w=-2.0): prices the PRE-clamp commanded target
#      beyond the joint limit, the gradient the clamp's flat region lacks.
#      A 0.61 rad excess costs ~-0.024/step, 2.97 rad ~-0.119/step.
#   5. fall termination (w=-50): one-shot -1.0 at a non-timeout termination
#      (~6 s of living reward); the forfeited episode remains the main price.
#   6. walking thrower: hover 1-6 s while the release point random-walks;
#      the launch re-aims from the live position, so a moving thrower is
#      in-distribution.
from __future__ import annotations

import os

import torch

from isaaclab.assets import RigidObject  # noqa: F401
from isaaclab.managers import SceneEntityCfg

# gate floor: a fully-distrusted catch still pays this fraction (0 = hard)
V36_GATE_FLOOR = float(os.environ.get("BOX_V36_GATE_FLOOR", "0.0"))
# stillness shaping: dq scale (rad/s) for the calm bell
V36_STILL_SIGMA = float(os.environ.get("BOX_V36_STILL_SIGMA", "3.0"))
# soft-fall: tilt ramp (up_z of torso z-axis) and dq threshold
V36_FALL_UPZ_START = float(os.environ.get("BOX_V36_FALL_UPZ_START", "0.87"))
V36_FALL_UPZ_END = float(os.environ.get("BOX_V36_FALL_UPZ_END", "0.50"))
V36_FALL_DQ_THR = float(os.environ.get("BOX_V36_FALL_DQ_THR", "8.0"))
# walking thrower: speed band + max wander radius from the spawn point
V36_WALK_V_LO = float(os.environ.get("BOX_V36_WALK_V_LO", "0.3"))
V36_WALK_V_HI = float(os.environ.get("BOX_V36_WALK_V_HI", "0.9"))
V36_WALK_R = float(os.environ.get("BOX_V36_WALK_R", "1.5"))
V36_WALK_TURN_P = float(os.environ.get("BOX_V36_WALK_TURN_P", "0.02"))

# ---- trust access ----------------------------------------------------------


def _bad_trust(env) -> torch.Tensor:
    """Per-env 'how much NOT to trust the box estimate', in [0,1].

    max(jitter, freshness, frozen) from the estimator state; clean episodes
    naturally report ~0.  Returns zeros when the estimator is off, so every
    trust-dependent term is inert in a ground-truth run.
    """
    s = getattr(env, "_v31_est", None)
    if s is None or "trust" not in s:
        return torch.zeros(env.num_envs, device=env.device)
    bad = s["trust"].max(dim=-1).values
    # blind_age folds in through freshness already; clamp for safety
    return bad.clamp(0.0, 1.0)


# ---- 1. trust-gated catch rewards -----------------------------------------
# Registered by name to keep RewTerm params yaml-serializable: the env cfg
# captures each term's ORIGINAL func (whatever the parent config routed it
# to) under a string key, and the manager calls this dispatcher.
_INNER: dict = {}


def register_inner(key: str, fn) -> None:
    _INNER[key] = fn


def trust_gated_reward(env, inner: str,
                       asset_cfg: SceneEntityCfg | None = None,
                       hold_sec: float | None = None,
                       sigma: float | None = None,
                       z_low: float | None = None,
                       z_high: float | None = None,
                       near_max: float | None = None) -> torch.Tensor:
    """Evaluate the registered reward `inner`, scaled by the trust gate.

    Explicit signature: Isaac Lab's manager introspects parameters, so a
    **kwargs catch-all is rejected.  The optional params are the union of
    every gated term's params."""
    kw = {}
    for k, v in (("asset_cfg", asset_cfg), ("hold_sec", hold_sec),
                 ("sigma", sigma), ("z_low", z_low), ("z_high", z_high),
                 ("near_max", near_max)):
        if v is not None:
            kw[k] = v
    r = _INNER[inner](env, **kw)
    g = 1.0 - (1.0 - V36_GATE_FLOOR) * _bad_trust(env)
    return r * g


# ---- 2. low-trust stillness ------------------------------------------------
def low_trust_stillness(env,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
    """Income for being CALM exactly when the box estimate is untrustworthy.

    bell(|dq|) * bad_trust: a still robot under garbage vision earns; a
    lunging robot earns nothing.  Income-only (>= 0)."""
    robot = env.scene[asset_cfg.name]
    dq2 = (robot.data.joint_vel ** 2).mean(dim=-1)
    calm = torch.exp(-dq2 / (V36_STILL_SIGMA ** 2))
    return calm * _bad_trust(env)


# ---- 2b. distrust action-amplitude penalty --------------------------------
def distrust_action_amp(env) -> torch.Tensor:
    """bad_trust x mean(relu(|a_raw| - 8)): price action amplitude while the
    box estimate is distrusted.

    Falls under bad perception showed |a| saturated at the +-30 sanity clip
    while trust was bad (e.g. a frozen track pinned at the chest): detection
    worked, but ungated dense shaping paid the policy to reach for the phantom
    at full amplitude.  Healthy catches peak ~22 on single joints briefly;
    sustained multi-joint > 8 under bad trust is the thrash signature.  Zero
    in clean episodes."""
    a = env.action_manager.action
    return _bad_trust(env) * torch.relu(a.abs() - 8.0).mean(dim=-1)


# ---- 3. soft-fall shaping --------------------------------------------------
def soft_fall_shaping(env,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                      ) -> torch.Tensor:
    """While toppling (torso z-axis leaving vertical), price joint speed.

    ramp(up_z) * mean(relu(|dq| - THR)): zero in nominal catching (tilt ramp
    starts ~30 deg, healthy catches stay under), grows as the fall develops.
    Teaches a damping reflex: crumple, don't thrash.
    """
    robot = env.scene[asset_cfg.name]
    q = robot.data.root_quat_w
    w, x, y, z = q.unbind(-1)
    up_z = 1.0 - 2.0 * (x * x + y * y)          # world-z of body z-axis
    ramp = ((V36_FALL_UPZ_START - up_z)
            / (V36_FALL_UPZ_START - V36_FALL_UPZ_END)).clamp(0.0, 1.0)
    dq = robot.data.joint_vel.abs()
    thrash = torch.relu(dq - V36_FALL_DQ_THR).mean(dim=-1)
    return ramp * thrash


# ---- 4. commanded-target limit excess (PRE-clamp gradient) -----------------
def cmd_limit_excess(env) -> torch.Tensor:
    """Sum over joints of relu(commanded PD target beyond the joint limit),
    reconstructed PRE-clamp from the raw action (offset + scale * action):
    the clamp edits processed_actions in place, so reading it after apply
    would see only in-range values and carry no gradient signal.  Prices a
    persistent out-of-range command habit (e.g. an ankle-roll target held
    ~0.6 rad past its limit), which the clamp alone does not discourage
    because clamped ticks are reward-identical."""
    if not hasattr(env, "_v36_lim"):
        term = env.action_manager.get_term("joint_pos")
        d = env.scene["robot"].data
        lim = getattr(d, "joint_pos_limits", None)
        if lim is None:
            lim = getattr(d, "joint_limits", None)
        if lim is None:
            lim = d.soft_joint_pos_limits
        ids = term._joint_ids
        if isinstance(ids, slice):
            lo, hi = lim[..., 0], lim[..., 1]
        else:
            lo, hi = lim[:, ids, 0], lim[:, ids, 1]
        env._v36_lim = (term, lo.clone(), hi.clone())
    term, lo, hi = env._v36_lim
    tgt = term._offset + term._scale * term.raw_actions
    exc = torch.relu(tgt - hi) + torch.relu(lo - tgt)
    return exc.sum(dim=-1)


# ---- 5. fall termination one-shot ------------------------------------------
def fall_terminated(env) -> torch.Tensor:
    """1.0 on the tick a NON-timeout termination fires (anchor_pos/ori,
    ee_body_pos = the fall detectors), else 0.  Weight x dt makes the price
    a one-shot: w=-50 -> -1.0 per fall."""
    tm = env.termination_manager
    return tm.terminated.float()


# ---- 6. walking-thrower hover ----------------------------------------------
# Same as throw.box_hover_and_launch with one addition: while hovering,
# env._box_p0 random-walks in xy inside a V36_WALK_R disc around its spawn
# point.  The launch solve already re-aims from the live p0 at the release
# tick, so the moving origin needs no other change.
from reflex.envs.throw import (  # noqa: E402
    BOX_LIN_DAMPING, GRAVITY, LAUNCH_SPEED_RANGE, LOFTY_SPEED_BONUS,
    RETHROW_GRACE_STEPS, _rot2d, reset_box_throw,
)


def v36_walking_hover_and_launch(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> None:
    """Per-step hover/launch event with a random-walking hover point."""
    del env_ids  # operate on all envs every step
    if not hasattr(env, "_box_p0"):
        return
    box = env.scene[asset_cfg.name]
    device = box.device
    step = env.episode_length_buf  # (N,)

    hovering = (~env._box_launched) & (step < env._box_launch_step)
    launching = (~env._box_launched) & (step >= env._box_launch_step)

    # ---- walking thrower: random-walk the hover point ----------------------
    n = env.num_envs
    if not hasattr(env, "_v36_walk_v"):
        env._v36_walk_v = torch.zeros(n, 2, device=device)
        env._v36_walk_anchor = env._box_p0[:, :2].clone()
    fresh = hovering & (env._v36_walk_v.norm(dim=-1) < 1e-6)
    turn = hovering & (torch.rand(n, device=device) < V36_WALK_TURN_P)
    redo = fresh | turn
    if bool(redo.any()):
        k = int(redo.sum())
        ang = torch.rand(k, device=device) * 6.283185
        spd = V36_WALK_V_LO + (V36_WALK_V_HI - V36_WALK_V_LO) \
            * torch.rand(k, device=device)
        env._v36_walk_v[redo] = torch.stack(
            [torch.cos(ang), torch.sin(ang)], dim=-1) * spd[:, None]
        # new episodes re-anchor the wander disc at the fresh spawn point
        env._v36_walk_anchor[fresh] = env._box_p0[fresh, :2]
    stepv = env._v36_walk_v * env.step_dt
    newxy = env._box_p0[:, :2] + stepv
    off = newxy - env._v36_walk_anchor
    r = off.norm(dim=-1)
    out = r > V36_WALK_R
    if bool(out.any()):
        # bounce inward: reflect the velocity toward the anchor
        env._v36_walk_v[out] = -env._v36_walk_v[out]
        newxy[out] = env._box_p0[out, :2]
    env._box_p0[:, :2] = torch.where(hovering[:, None], newxy,
                                     env._box_p0[:, :2])
    env._v36_walk_v[~hovering] = 0.0

    # ---- below here: identical to throw.box_hover_and_launch ---------------
    robot = env.scene["robot"]
    origins_all = env.scene.env_origins
    p0 = env._box_p0
    to_robot = robot.data.root_pos_w[:, :2] - p0[:, :2]
    d_live = to_robot.norm(dim=-1).clamp_min(0.5)
    to_robot = _rot2d(to_robot, env._box_aim_err)
    target = torch.cat(
        [p0[:, :2] + to_robot,
         (origins_all[:, 2] + env._box_catch_h)[:, None]], dim=-1)
    k = BOX_LIN_DAMPING
    g = torch.tensor([0.0, 0.0, -GRAVITY], device=device)
    v_term = (g / k)[None, :]
    T = torch.sqrt(d_live / 4.0) * env._box_tmult
    expk = torch.exp(-k * T)[:, None]
    v0 = v_term + k * (target - p0 - v_term * T[:, None]) / (1.0 - expk)
    v0 = v0 * env._box_jitter[:, None]
    speed = v0.norm(dim=-1).clamp_min(1e-6)
    cap = LAUNCH_SPEED_RANGE[1] + LOFTY_SPEED_BONUS \
        * (env._box_tmult - 1.0).clamp(min=0.0)
    clamped = torch.minimum(speed.clamp_min(LAUNCH_SPEED_RANGE[0]), cap)
    v0 = v0 * (clamped / speed)[:, None]

    stored_pose = torch.cat([p0, env._box_q0], dim=-1)
    cur_pose = torch.cat([box.data.root_pos_w, box.data.root_quat_w], dim=-1)
    hold_or_launch = (hovering | launching).unsqueeze(-1)
    pose = torch.where(hold_or_launch, stored_pose, cur_pose)

    launch_vel = torch.cat([v0, env._box_omega], dim=-1)
    cur_vel = torch.cat([box.data.root_lin_vel_w, box.data.root_ang_vel_w],
                        dim=-1)
    vel = torch.where(launching.unsqueeze(-1), launch_vel,
                      torch.where(hovering.unsqueeze(-1),
                                  torch.zeros_like(cur_vel), cur_vel))

    box.write_root_pose_to_sim(pose)
    box.write_root_velocity_to_sim(vel)
    env._box_launched = env._box_launched | launching

    if not hasattr(env, "_box_regrace"):
        env._box_regrace = torch.zeros(env.num_envs, dtype=torch.long,
                                       device=device)
    bp = box.data.root_pos_w
    bv = box.data.root_lin_vel_w
    z = bp[:, 2] - env.scene.env_origins[:, 2]
    bspeed = bv.norm(dim=-1)
    bad = env._box_launched & (
        (z < 0.2) | (z > 30.0) | (bspeed > 30.0)
        | torch.isnan(z) | torch.isnan(bspeed))
    env._box_regrace = torch.where(bad, env._box_regrace + 1,
                                   torch.zeros_like(env._box_regrace))
    if int(env.common_step_counter) % 5 == 0:
        rethrow = env._box_regrace >= RETHROW_GRACE_STEPS
        if rethrow.any():
            ids = rethrow.nonzero(as_tuple=False).squeeze(-1)
            reset_box_throw(env, ids)
            env._box_launch_step[ids] += env.episode_length_buf[ids]
            env._box_regrace[ids] = 0
