"""Reward, event and observation terms of the Stage 1 teacher (stepping catch).

Adds to the safety/robustness terms: a per-episode aim-offset curriculum
(including far "abandon" throws where the right play is to stand still),
cradle perturbations with an anchor reward, facing/landing-point terms, and the
reactive two-stage arm reference.  Most features are opt-in BOX_V38_* knobs.
"""
# Aim-offset curriculum (per episode, radial offset from the robot's live
# position; sample_v38_aim_offsets):
#   52%  no offset
#   40%  catch curriculum   55%: U(0.35,0.55)  35%: U(0.55,0.70)
#                           10%: U(0.70,1.00) thin tail
#    8%  abandon throws     U(1.00,2.00): optimal play is to stand still, which
#                           keeps the trained restraint (no chasing of
#                           unreachable boxes) alive
# The curriculum itself pays for catches only: approach rewards breed lunging.
#
# Cradle perturbation + anchor: a box landing off-centre in the arms tends to
# roll off, and the recovery has to be trained.
#   * v38_cradle_perturb: interval event; a held box occasionally gets a
#     velocity/spin kick (emulating off-centre landings and slips).
#   * v38_cradle_anchor: reward for keeping the box AT the cradle anchor,
#     default 0.3 * exp(-(d/0.18)^2), only while a launched box is cradled
#     (< 0.5 m, slow).  Comparable to the other hold rewards; a dropped box
#     earns 0 (ground-truth based, sim only).
#
# Observation contract: 170-D, unchanged (173-D with BOX_V38_LAND_OBS=1).
from __future__ import annotations

import math
import os

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

# walking-thrower constants and launch helpers reused from throw / robust_terms
from reflex.envs.throw import lib_align_track as _lib_align_track  # noqa: E402
from reflex.envs.throw import (  # noqa: E402
    BOX_LIN_DAMPING, GRAVITY, LAUNCH_SPEED_RANGE, LOFTY_SPEED_BONUS,
    RETHROW_GRACE_STEPS, RETHROW_ENABLED, _rot2d, reset_box_throw,
)
from reflex.envs.robust_terms import (  # noqa: E402
    V36_WALK_R, V36_WALK_TURN_P, V36_WALK_V_HI, V36_WALK_V_LO,
)

# ---------------------------------------------------------------- knobs ----
V38_P_OFFSET = float(os.environ.get("BOX_V38_P_OFFSET", "0.40"))
V38_P_ABANDON = float(os.environ.get("BOX_V38_P_ABANDON", "0.08"))
V38_ABANDON_LO = float(os.environ.get("BOX_V38_ABANDON_LO", "1.00"))
V38_ABANDON_HI = float(os.environ.get("BOX_V38_ABANDON_HI", "2.00"))
V38_PERTURB_P = float(os.environ.get("BOX_V38_PERTURB_P", "0.15"))
V38_PERTURB_V = float(os.environ.get("BOX_V38_PERTURB_V", "0.5"))
V38_PERTURB_W = float(os.environ.get("BOX_V38_PERTURB_W", "3.0"))
V38_ANCHOR_SIG = float(os.environ.get("BOX_V38_ANCHOR_SIG", "0.18"))
# facing reward (dead-band + width, degrees)
V38_FACE_DEAD = float(os.environ.get("BOX_V38_FACE_DEAD", "15"))
V38_FACE_SIG = float(os.environ.get("BOX_V38_FACE_SIG", "35"))
# slew the reference-motion yaw toward the box (deg/s; 0 = off)
V38_ANCHOR_FACE = float(os.environ.get("BOX_V38_ANCHOR_FACE", "0"))
# cradle anchor point in the robot (pelvis) frame
V38_CHEST_LOCAL = (0.25, 0.0, 0.35)

# Wide box-mass randomisation (payload-dominated band, decoupled from box
# size; a heavier box gives a clearer contact-force signal).  Off unless
# BOX_V38_MASS_DR=1.
V38_MASS_LO = float(os.environ.get("BOX_V38_MASS_LO", "0.10"))
V38_MASS_HI = float(os.environ.get("BOX_V38_MASS_HI", "0.70"))
# mass curriculum: ramp the mass ceiling from MASS_HI to MASS_HI_FINAL over
# MASS_RAMP env steps (0 = off, flat band)
V38_MASS_RAMP = float(os.environ.get("BOX_V38_MASS_RAMP", "0"))
V38_MASS_HI_FINAL = float(os.environ.get("BOX_V38_MASS_HI_FINAL", "0.55"))

# Throw-bearing fan half-width (deg).  Applied as a module-global patch of
# throw.BEARING_RANGE_DEG in teacher_env_cfg (see the comment there); this
# constant is the knob and the audit reference.  0 = off.
V38_BEARING_DEG = float(os.environ.get("BOX_V38_BEARING_DEG", "0"))
V38_DBG_PRINTS = int(os.environ.get("BOX_V38_DBG_PRINTS", "8"))


def v38_wide_mass_dr(env, env_ids,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                     ) -> None:
    """Startup event: widen per-env box mass to U(V38_MASS_LO, V38_MASS_HI).

    Model identity (radius/half-extents) MUST be cached from the ORIGINAL
    build masses before overwriting them: throw.py infers the box model from a
    0.132 kg mass threshold, which the wide band destroys.  Flight is
    unaffected (launch sets velocities; PhysX lin-damping is mass-
    independent): only CONTACT momentum changes, which is the point.
    Inertia is scaled proportionally with mass (payload approximation).
    """
    from reflex.envs.throw import BOX_MODELS
    box = env.scene[asset_cfg.name]
    device = box.device
    view = box.root_physx_view
    m0 = view.get_masses()
    flat0 = m0.reshape(-1)
    is_big = (flat0.to(device) > 0.132)
    big = torch.tensor(BOX_MODELS[0]["size"], device=device) * 0.5
    small = torch.tensor(BOX_MODELS[1]["size"], device=device) * 0.5
    env._balloon_radius_per_env = torch.where(
        is_big,
        torch.full((env.num_envs,), BOX_MODELS[0]["eff_radius"], device=device),
        torch.full((env.num_envs,), BOX_MODELS[1]["eff_radius"], device=device))
    env._box_half_extents = torch.where(
        is_big.unsqueeze(-1), big[None, :], small[None, :])
    import os as _os
    from reflex.envs.throw import mix_models as _mix_models, infer_models_from_geometry as _infer_geo
    _mix = _mix_models(_os.environ.get("BOX_MODEL", "random"))
    if _mix is not None:   # mixed box models: identity by nearest build mass (overrides the 2-model threshold above)
        env._balloon_radius_per_env, env._box_half_extents, env._box_model_idx = _infer_geo(box, _mix, device)
    # Mass curriculum: sampling the whole wide band from step 0 hurt catching
    # (heavy-box contact may want a different arm stiffness than light-box),
    # so the ceiling can RAMP: start narrow and widen toward the real ballast
    # weight late in training, when the catch skill is already solid.
    hi = V38_MASS_HI
    if V38_MASS_RAMP > 0:
        _it = float(getattr(env, "common_step_counter", 0)) / max(
            float(V38_MASS_RAMP), 1.0)
        frac = min(max(_it, 0.0), 1.0)
        hi = V38_MASS_HI + (V38_MASS_HI_FINAL - V38_MASS_HI) * frac
    new_flat = V38_MASS_LO + (hi - V38_MASS_LO) * torch.rand_like(flat0)
    ratio = (new_flat / flat0.clamp_min(1e-6))
    inert = view.get_inertias()
    new_inert = inert * ratio.reshape(-1, *([1] * (inert.dim() - 1)))
    idx = torch.arange(m0.shape[0])
    view.set_masses(new_flat.reshape(m0.shape), idx)
    view.set_inertias(new_inert, idx)
    print(f"[teacher] wide mass DR: U({V38_MASS_LO},{V38_MASS_HI}) kg on "
          f"{m0.shape[0]} boxes (model identity cached first)", flush=True)


def _v38_bearing_now(env, ids: torch.Tensor) -> torch.Tensor:
    """Signed bearing (rad) of the hover point p0 relative to the robot's
    LIVE facing, wrapped to (-pi, pi]."""
    robot = env.scene["robot"]
    q = robot.data.root_quat_w[ids]
    qw, qx, qy, qz = q.unbind(-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    rel = env._box_p0[ids, :2] - robot.data.root_pos_w[ids, :2]
    b = torch.atan2(rel[:, 1], rel[:, 0]) - yaw
    return torch.atan2(torch.sin(b), torch.cos(b))


def _v38_audit_tick(env, hovering: torch.Tensor, step: torch.Tensor) -> None:
    """Bearing-randomisation audit (print-capped at V38_DBG_PRINTS).

    Records the FINAL placement bearing at episode step 2: first_throw_realign
    (interval event, BOX_THROW_REALIGN=1) re-samples the whole throw at step
    <= 1, so anything measured earlier would be overwritten.  Re-measures at
    step 10 (0.16 s later: past the realign window, before the walking thrower
    wanders far) and prints the running corr(applied, hover) plus the
    |bearing| > 15 deg fraction (0.625 for U(+-40)).
    `env._v38_bearing_audit` keeps the applied bearing for offline probes."""
    n = env.num_envs
    device = env.device
    if not hasattr(env, "_v38_bearing_audit"):
        env._v38_bearing_audit = torch.zeros(n, device=device)
        env._v38_audit_armed = torch.zeros(n, dtype=torch.bool, device=device)
        env._v38_audit_pairs: list[tuple[float, float]] = []
        env._v38_audit_printed = 0
    rec = hovering & (step == 2)
    if bool(rec.any()):
        ids = rec.nonzero(as_tuple=False).squeeze(-1)
        env._v38_bearing_audit[ids] = _v38_bearing_now(env, ids)
        env._v38_audit_armed[ids] = True
    chk = hovering & (step == 10) & env._v38_audit_armed
    if not bool(chk.any()):
        return
    ids = chk.nonzero(as_tuple=False).squeeze(-1)
    env._v38_audit_armed[ids] = False
    if len(env._v38_audit_pairs) >= 4096:      # memory cap, audit-only buffer
        return
    app = env._v38_bearing_audit[ids]
    hov = _v38_bearing_now(env, ids)
    env._v38_audit_pairs += [(float(a), float(h)) for a, h in zip(app, hov)]
    n_pairs = len(env._v38_audit_pairs)
    if (env._v38_audit_printed >= V38_DBG_PRINTS
            or n_pairs < 32 * (env._v38_audit_printed + 1)):
        return
    env._v38_audit_printed += 1
    t = torch.tensor(env._v38_audit_pairs)
    a_, h_ = t[:, 0], t[:, 1]
    va, vh = a_ - a_.mean(), h_ - h_.mean()
    corr = float((va * vh).sum() / (va.norm() * vh.norm()).clamp_min(1e-9))
    frac = float((a_.abs() > math.radians(15.0)).float().mean())
    d = torch.rad2deg(a_)
    print(f"[teacher] bearing audit #{env._v38_audit_printed}: n={n_pairs} "
          f"corr(applied@2, hover@10)={corr:.3f} frac|b|>15deg={frac:.3f} "
          f"applied deg min/p50/max={d.min():.0f}/{d.median():.0f}/"
          f"{d.max():.0f} (fan ±{V38_BEARING_DEG:g})", flush=True)


def v38_mass_dr_reset(env, env_ids,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                      ) -> None:
    """Per-episode mass re-roll, so the curriculum ramp can actually move (a
    startup-only event would make BOX_V38_MASS_RAMP a no-op).  Model identity
    was cached at startup by v38_wide_mass_dr; only masses are rewritten here."""
    if V38_MASS_RAMP <= 0 or env_ids is None or len(env_ids) == 0:
        return
    box = env.scene[asset_cfg.name]
    view = box.root_physx_view
    m0 = view.get_masses()
    flat = m0.reshape(-1).clone()
    frac = min(max(float(getattr(env, "common_step_counter", 0))
                   / max(V38_MASS_RAMP, 1.0), 0.0), 1.0)
    hi = V38_MASS_HI + (V38_MASS_HI_FINAL - V38_MASS_HI) * frac
    ids = env_ids.detach().cpu() if hasattr(env_ids, "detach") else env_ids
    draw = V38_MASS_LO + (hi - V38_MASS_LO) * torch.rand(len(ids))
    inert = view.get_inertias()
    ratio = draw / flat[ids].clamp_min(1e-6)
    flat[ids] = draw
    new_inert = inert.clone()
    new_inert[ids] = inert[ids] * ratio.reshape(-1, *([1] * (inert.dim() - 1)))
    idx = torch.arange(m0.shape[0])
    view.set_masses(flat.reshape(m0.shape), idx)
    view.set_inertias(new_inert, idx)
    n = getattr(env, "_v38_massramp_log", 0)
    if n < 3:
        env._v38_massramp_log = n + 1
        print(f"[teacher] mass ramp: step-frac {frac:.2f} -> ceiling "
              f"{hi:.2f} kg (n={len(ids)})", flush=True)
    # periodic log that the ramp is actually moving.  BOX_V38_MASS_RAMP is in
    # env steps (common_step_counter), not env steps x num_envs.
    _sc = int(getattr(env, "common_step_counter", 0))
    if _sc % 4000 < 30 and _sc > 0:
        print(f"[teacher] mass ramp @step {_sc}: ceiling {hi:.2f} kg",
              flush=True)


def sample_v38_aim_offsets(env, env_ids: torch.Tensor,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                           ) -> None:
    """Reset event: per-episode radial aim offset (world-frame XY)."""
    del asset_cfg
    if env_ids is None or len(env_ids) == 0:
        return
    device = env.device
    n = len(env_ids)
    if not hasattr(env, "_v38_aim_off"):
        env._v38_aim_off = torch.zeros(env.num_envs, 2, device=device)
    u = torch.rand(n, device=device)
    r = torch.zeros(n, device=device)
    # Band override (evaluation): a single radius band, e.g. 0.6-0.7 m.
    # Unset = the 55/35/10 curriculum below.
    _blo = float(os.environ.get("BOX_V38_OFF_LO", "0"))
    _bhi = float(os.environ.get("BOX_V38_OFF_HI", "0"))
    if _bhi > 0:
        r = _blo + (_bhi - _blo) * torch.rand(n, device=device)
        # Angle lock: restrict the offset direction to [OFF_ANG_LO,
        # OFF_ANG_HI] deg (default: uniform on the circle, which averages
        # front/back/left/right together), to test whether the catch envelope
        # is anisotropic (forward reach vs lateral reach).  Unset = uniform.
        _alo = float(os.environ.get("BOX_V38_OFF_ANG_LO", "0"))
        _ahi = float(os.environ.get("BOX_V38_OFF_ANG_HI", "0"))
        if os.environ.get("BOX_V38_OFF_LATERAL", "0") == "1":
            # Lateral lock: offset perpendicular to the robot's CURRENT yaw (the thrower stands ahead along it),
            # random left / right per episode, so "offset" is a single quantity: lateral metres.
            robot = env.scene["robot"]; q = robot.data.root_quat_w[env_ids]
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]))
            side = torch.where(torch.rand(n, device=device) < 0.5, 1.0, -1.0)
            ang = yaw + side * 1.5707963
            if not getattr(env, "_v38_lat_proof", False):
                env._v38_lat_proof = True
                print(f"[teacher] lateral offset lock active: radius {_blo:.2f}-{_bhi:.2f} m, perpendicular to robot yaw, random side (non-default)", flush=True)
        elif _ahi != _alo:
            ang = (math.radians(_alo)
                   + math.radians(_ahi - _alo) * torch.rand(n, device=device))
        else:
            ang = torch.rand(n, device=device) * 6.283185307
        env._v38_aim_off[env_ids] = torch.stack(
            [r * torch.cos(ang), r * torch.sin(ang)], dim=-1)
        return
    off_band = u < V38_P_OFFSET
    # Optional re-weighting: BOX_V38_HARDBAND=1 moves curriculum mass to the
    # harder 0.55-0.70 m band (the weak edge), and BOX_V38_FOREAFT compresses
    # offset directions toward the fore-aft axis, where real throws tend to
    # miss more often than laterally.  Unset = 55/35/10 + uniform direction.
    _hard = os.environ.get("BOX_V38_HARDBAND", "0") == "1"
    aband = (u >= V38_P_OFFSET) & (u < V38_P_OFFSET + V38_P_ABANDON)
    # curriculum bands within the offset population: 55/35/10
    b = torch.rand(n, device=device)
    _p_easy, _p_hard = (0.35, 0.90) if _hard else (0.55, 0.90)
    # hard mode: 35 % easy (0.35-0.55), 55 % hard (0.55-0.70), 10 % tail
    r = torch.where(off_band & (b < _p_easy),
                    0.35 + 0.20 * torch.rand(n, device=device), r)
    r = torch.where(off_band & (b >= _p_easy) & (b < _p_hard),
                    0.55 + 0.15 * torch.rand(n, device=device), r)
    r = torch.where(off_band & (b >= 0.90),
                    0.70 + 0.30 * torch.rand(n, device=device), r)
    r = torch.where(aband,
                    V38_ABANDON_LO + (V38_ABANDON_HI - V38_ABANDON_LO)
                    * torch.rand(n, device=device), r)
    ang = torch.rand(n, device=device) * 6.283185307
    _fa = float(os.environ.get("BOX_V38_FOREAFT", "0"))
    if _fa > 0:
        # squeeze directions toward the fore-aft axis: keep sign, compress
        # the angle toward 0/pi by factor (1-_fa) around the nearest axis
        near_pi = (ang > 1.5708) & (ang < 4.7124)
        axis = torch.where(near_pi, torch.full_like(ang, 3.14159),
                           torch.zeros_like(ang))
        ang = axis + (ang - axis) * (1.0 - _fa)
    env._v38_aim_off[env_ids] = torch.stack(
        [r * torch.cos(ang), r * torch.sin(ang)], dim=-1)


def v38_walking_hover_and_launch(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> None:
    """Same as robust_terms.v36_walking_hover_and_launch with ONE change: the
    launch aims at (live robot XY + per-episode offset) instead of the robot."""
    _apply_flight_rgate(env)   # flight reward gate (idempotent); this function is the teacher env's per-step event
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
    # library-replay throws (fitted real throws): the thrower stood still and
    # the fitted v0 is injected verbatim, so the walking thrower must not move p0
    _libm = getattr(env, "_box_lib_mask", None)
    hov_walk = hovering if _libm is None else (hovering & ~_libm)
    if not hasattr(env, "_v36_walk_v"):
        env._v36_walk_v = torch.zeros(n, 2, device=device)
        env._v36_walk_anchor = env._box_p0[:, :2].clone()
    fresh = hov_walk & (env._v36_walk_v.norm(dim=-1) < 1e-6)
    turn = hov_walk & (torch.rand(n, device=device) < V36_WALK_TURN_P)
    redo = fresh | turn
    if bool(redo.any()):
        k = int(redo.sum())
        ang = torch.rand(k, device=device) * 6.283185
        spd = V36_WALK_V_LO + (V36_WALK_V_HI - V36_WALK_V_LO) \
            * torch.rand(k, device=device)
        env._v36_walk_v[redo] = torch.stack(
            [torch.cos(ang), torch.sin(ang)], dim=-1) * spd[:, None]
        env._v36_walk_anchor[fresh] = env._box_p0[fresh, :2]
    stepv = env._v36_walk_v * env.step_dt
    newxy = env._box_p0[:, :2] + stepv
    off = newxy - env._v36_walk_anchor
    r = off.norm(dim=-1)
    out = r > V36_WALK_R
    if bool(out.any()):
        env._v36_walk_v[out] = -env._v36_walk_v[out]
        newxy[out] = env._box_p0[out, :2]
    env._box_p0[:, :2] = torch.where(hov_walk[:, None], newxy,
                                     env._box_p0[:, :2])
    env._v36_walk_v[~hov_walk] = 0.0

    # bearing audit (print-capped): the widened fan lives in
    # throw.BEARING_RANGE_DEG (patched by teacher_env_cfg); this only verifies
    # that the placement survives through the hover.
    if V38_BEARING_DEG > 0:
        _v38_audit_tick(env, hovering, step)

    # ---- below: identical to the walking thrower EXCEPT the aim target -------
    robot = env.scene["robot"]
    # Bearing sweep (demo/probe): during hover, drive the hover point along an
    # arc around the robot, bearing = A*sin(2*pi*t/T) relative to the robot's
    # yaw, to check whether the robot turns to face it.  Unset (0) = off.
    _sw = float(os.environ.get("BOX_V38_SWEEP_DEG", "0"))
    if _sw > 0 and bool(hovering.any()):
        _T = float(os.environ.get("BOX_V38_SWEEP_T", "6.0"))
        _t = step.float() * env.step_dt
        # base yaw is LATCHED at the first hover ticks: sweeping around the
        # LIVE yaw makes the box chase the robot that is chasing the box
        # (runaway spin).
        _rq = robot.data.root_quat_w
        _w, _x, _y, _z = _rq[:, 0], _rq[:, 1], _rq[:, 2], _rq[:, 3]
        _live_yaw = torch.atan2(2 * (_w * _z + _x * _y),
                                1 - 2 * (_y * _y + _z * _z))
        if not hasattr(env, "_v38_sweep_base"):
            env._v38_sweep_base = _live_yaw.clone()
        _fresh = hovering & (step < 5)
        if bool(_fresh.any()):
            env._v38_sweep_base = torch.where(_fresh, _live_yaw,
                                              env._v38_sweep_base)
        _yaw = env._v38_sweep_base
        _beta = math.radians(_sw) * torch.sin(2 * math.pi * _t / _T)
        _rp = robot.data.root_pos_w[:, :2]
        _rad = (env._box_p0[:, :2] - _rp).norm(dim=-1).clamp(min=1.0)
        _ang = _yaw + _beta
        _new = _rp + torch.stack([_rad * torch.cos(_ang),
                                  _rad * torch.sin(_ang)], dim=-1)
        env._box_p0[:, :2] = torch.where(hovering[:, None], _new,
                                         env._box_p0[:, :2])

    _lib_align_track(env, hovering, launching)  # no-op unless BOX_THROW_LIB_ALIGN is set
    origins_all = env.scene.env_origins
    p0 = env._box_p0
    if not hasattr(env, "_v38_aim_off"):
        env._v38_aim_off = torch.zeros(n, 2, device=device)
    # aim at robot + per-episode radial offset
    to_robot = (robot.data.root_pos_w[:, :2] + env._v38_aim_off
                - p0[:, :2])
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
    if _libm is not None:
        # library throws: fitted real v0 verbatim (no re-aim / offset / clamp)
        v0 = torch.where(_libm[:, None], env._box_v0, v0)
    # publish the ACTUAL launch state for evaluation (read-only, no effect on
    # the sim)
    if not hasattr(env, "_box_v0_pub"):
        env._box_v0_pub = torch.zeros_like(v0)
        env._box_p0_pub = torch.zeros_like(p0)
    env._box_v0_pub = torch.where(launching[:, None], v0, env._box_v0_pub)
    env._box_p0_pub = torch.where(launching[:, None], p0, env._box_p0_pub)

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
    if RETHROW_ENABLED and int(env.common_step_counter) % 5 == 0:
        rethrow = env._box_regrace >= RETHROW_GRACE_STEPS
        if rethrow.any():
            ids = rethrow.nonzero(as_tuple=False).squeeze(-1)
            reset_box_throw(env, ids)
            env._box_launch_step[ids] += env.episode_length_buf[ids]
            env._box_regrace[ids] = 0


def _chest_w(env):
    robot = env.scene["robot"]
    local = torch.tensor(V38_CHEST_LOCAL, device=env.device
                         ).expand(env.num_envs, 3)
    return robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, local)


def _cradled(env, box) -> torch.Tensor:
    """Launched box, near the chest anchor, slow relative to the robot, AND
    held off the ground (z-gate).  Without the z-gate the policy farmed the
    anchor reward by crouching over a box lying on the floor.  A cradled box
    rides at ~0.9-1.1 m, a floor box at ~0.11 m; the 0.45 m gate (above the
    env origin) removes the exploit."""
    robot = env.scene["robot"]
    d = (box.data.root_pos_w - _chest_w(env)).norm(dim=-1)
    relv = (box.data.root_lin_vel_w
            - robot.data.root_lin_vel_w).norm(dim=-1)
    z_rel = box.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return env._box_launched & (d < 0.50) & (relv < 2.0) & (z_rel > 0.45) \
        if hasattr(env, "_box_launched") \
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def v38_face_box(env,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                 ) -> torch.Tensor:
    """Facing reward: turn toward the box.

    With a sweeping hover box, a policy trained without it knows WHICH way to
    turn (corr 0.70) but follows only ~28% of the bearing swing (median facing
    error 42 deg); the bearing curriculum poses the question, this term pays
    for the answer.  On the robot the camera rides the head, so facing the
    box is what keeps it detectable.

    Shape: dead-band |err| <= FACE_DEAD (no gradient inside; the policy does
    not twitch along with a thrower's hand jitter), then a smooth exp() ramp.
    Applies while the box is a target (in range); the reward is 1 otherwise.
    """
    box = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    rel = box.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist = rel.norm(dim=-1)
    q = robot.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    bear = torch.atan2(rel[:, 1], rel[:, 0])
    err = torch.atan2(torch.sin(bear - yaw), torch.cos(bear - yaw)).abs()
    dead = math.radians(V38_FACE_DEAD)
    excess = (err - dead).clamp(min=0.0)
    rew = torch.exp(-(excess / math.radians(V38_FACE_SIG)) ** 2)
    # only while the box is a target: in range, not parked, not cradled
    live = (dist > 0.35) & (dist < 8.0)
    if hasattr(env, "_box_launched"):
        pass  # hovering counts too: facing early is the point
    return torch.where(live, rew, torch.ones_like(rew))


def v38_anchor_face_box(env, env_ids,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                        ) -> None:
    """Interval event: rotate the REFERENCE MOTION to face the box instead of
    paying a reward to fight it.

    The facing reward alone (weight 0.10) barely moves behaviour: the tracking
    rewards pin global heading to the clip via motion_global_anchor_ori
    (weight 0.5, std 0.4 rad), so a 42 deg turn COSTS ~0.48/step while the
    facing reward pays at most 0.10/step.  The command already carries a
    per-env `yaw_offsets` that rotates the whole clip (positions, quats,
    velocities), set at reset by randomize_motion_yaw.  Here it is slewed
    toward the box bearing before the catch, rate-limited, so tracking and
    facing agree.  Note that yaw_offsets rotate the clip about the env origin
    (see v38_recentre_clip_on_anchor).  Knob unset = no effect.
    """
    del env_ids, asset_cfg
    if V38_ANCHOR_FACE <= 0.0:
        return
    cmd = env.command_manager.get_term("motion")
    if not hasattr(cmd, "yaw_offsets"):
        return
    box = env.scene["balloon"]
    robot = env.scene["robot"]
    # Driven by the ESTIMATOR, not ground truth: the real robot only has the
    # filtered estimate, so driving the slew from box.data.root_pos_w would
    # train against a signal it cannot reproduce.  Falls back to ground truth
    # only when the estimator is off.
    box_w = box.data.root_pos_w
    if os.environ.get("BOX_DP_EST", "0") == "1":
        try:
            from reflex.envs import robust_estimator as _v36e
            _s = _v36e._advance(env, asset_cfg)
            box_w = torch.where(_s["clean_ep"][:, None], box_w, _s["est_p_w"])
        except Exception:
            pass
    rel = box_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist = rel.norm(dim=-1)
    live = (dist > 0.5) & (dist < 8.0)          # hovering or inbound
    if not bool(live.any()):
        return
    want = torch.atan2(rel[:, 1], rel[:, 0])
    # measure the error on the ANCHOR's actual world heading, not on the
    # offset variable: the clip carries its own intrinsic yaw, so setting
    # yaw_offsets = bearing leaves a fixed ~47 deg bias.
    aq = cmd.anchor_quat_w
    aw, ax, ay, az = aq[:, 0], aq[:, 1], aq[:, 2], aq[:, 3]
    anchor_yaw = torch.atan2(2 * (aw * az + ax * ay),
                             1 - 2 * (ay * ay + az * az))
    cur = cmd.yaw_offsets
    err = torch.atan2(torch.sin(want - anchor_yaw),
                      torch.cos(want - anchor_yaw))
    step_max = math.radians(V38_ANCHOR_FACE) * env.step_dt   # deg/s cap
    delta = err.clamp(-step_max, step_max)
    cmd.yaw_offsets = torch.where(live, cur + delta, cur)
    n = getattr(env, "_v38_face_log", 0)
    if n < 3:
        env._v38_face_log = n + 1
        print(f"[teacher] anchor-face slew active: median |err| "
              f"{torch.rad2deg(err[live].abs()).median():.0f} deg, "
              f"cap {V38_ANCHOR_FACE:g} deg/s", flush=True)


def v38_clamp_excess(env) -> torch.Tensor:
    """Penalise commanding PAST the joint limit.

    The target clamp protects the hardware; what it cannot fix is the gap
    between what the policy commands and what the robot does.  This term
    makes staying inside the clamp cheaper than being clipped, without
    touching the clamp itself.  Returns the per-env total excess in radians
    (weight should be NEGATIVE).
    """
    ex = getattr(env, "_clamp_excess", None)
    if ex is None:
        return torch.zeros(env.num_envs, device=env.device)
    return ex


def v38_cradle_perturb(env, env_ids: torch.Tensor,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                       ) -> None:
    """Interval event: kick a cradled box (off-centre slips the policy must
    learn to recover from by feel)."""
    del env_ids
    box = env.scene[asset_cfg.name]
    device = box.device
    n = env.num_envs
    # Kick only after >= 1.5 s of CONTINUOUS cradling (75 ticks at 50 Hz):
    # kicks landing while the catch is still settling taught sloppy catches.
    if not hasattr(env, "_v38_cradle_ticks"):
        env._v38_cradle_ticks = torch.zeros(n, dtype=torch.long, device=device)
    cr = _cradled(env, box)
    env._v38_cradle_ticks = torch.where(
        cr, env._v38_cradle_ticks + 1,
        torch.zeros_like(env._v38_cradle_ticks))
    kick = cr & (env._v38_cradle_ticks >= 75) \
        & (torch.rand(n, device=device) < V38_PERTURB_P)
    if not bool(kick.any()):
        return
    dv = torch.zeros(n, 6, device=device)
    ang = torch.rand(n, device=device) * 6.283185307
    mag = torch.rand(n, device=device) * V38_PERTURB_V
    dv[:, 0] = mag * torch.cos(ang)
    dv[:, 1] = mag * torch.sin(ang)
    dv[:, 2] = (torch.rand(n, device=device) - 0.6) * 0.5   # slight down bias
    dv[:, 3:6] = (torch.rand(n, 3, device=device) - 0.5) * 2 * V38_PERTURB_W
    vel = torch.cat([box.data.root_lin_vel_w, box.data.root_ang_vel_w], dim=-1)
    box.write_root_velocity_to_sim(
        torch.where(kick[:, None], vel + dv, vel))


def v38_cradle_anchor(env,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                      ) -> torch.Tensor:
    """Reward: keep the cradled box AT the anchor.  Ground-truth based (sim
    only); zero for dropped boxes and before launch."""
    box = env.scene[asset_cfg.name]
    d = (box.data.root_pos_w - _chest_w(env)).norm(dim=-1)
    r = torch.exp(-(d / V38_ANCHOR_SIG) ** 2)
    return torch.where(_cradled(env, box), r, torch.zeros_like(r))


# ============================================================================
# Go-to-landing-point reward (optional, BOX_V38_LAND_W > 0)
#
# The only term that pulls the robot toward the box is catch_eef_to_balloon =
# exp(-d^2/sigma^2) with d measured to the box's CURRENT position.  Mid-flight
# the box is high in the air, so that gradient points mostly UP, not at the
# place the box is going to arrive, and widening sigma only extends a
# mis-pointed gradient.  On real throws the robot does move, but its
# displacement is uncorrelated with the box's lateral offset: the missing
# signal is directional, not a matter of penalty size.
#
# This term pays exp(-d_xy^2 / sigma^2) on the HORIZONTAL distance from the
# base to where the box will descend through catch height, computed with the
# same linear-drag ballistic model the thrower uses (dv/dt = g - k v).  It is
# active only while the box is airborne (and falling, unless ALLFLIGHT), so it
# says nothing about where to stand when there is no throw.  The landing point
# follows from the box position+velocity, which the robot's filter provides,
# so this is not a sim-only signal.
BOX_LAND_W = float(os.environ.get("BOX_V38_LAND_W", "0"))
BOX_LAND_SIG = float(os.environ.get("BOX_V38_LAND_SIG", "0.6"))
BOX_LAND_H = float(os.environ.get("BOX_V38_LAND_H", "1.0"))
BOX_LAND_ALLFLIGHT = os.environ.get("BOX_V38_LAND_ALLFLIGHT", "0") == "1"


def _predict_landing_xy(p_w, v_w, z_target, k=None, iters=6):
    """Horizontal position where a linear-drag projectile crosses z_target on
    the way down.  Closed form for p(t); Newton on z(t) - z_target."""
    k = BOX_LIN_DAMPING if k is None else k
    vt_z = -GRAVITY / k                       # terminal vertical velocity
    z0 = p_w[:, 2]
    vz0 = v_w[:, 2]
    # start from the drag-free estimate, clamped to a sane horizon
    disc = (vz0 * vz0 + 2.0 * GRAVITY * (z0 - z_target)).clamp(min=0.0)
    t = ((vz0 + disc.sqrt()) / GRAVITY).clamp(0.02, 3.0)
    for _ in range(iters):
        e = torch.exp(-k * t)
        z = z0 + vt_z * t + (vz0 - vt_z) * (1.0 - e) / k
        dz = vt_z + (vz0 - vt_z) * e          # dz/dt
        t = (t - (z - z_target) / dz.clamp(max=-1e-3)).clamp(0.02, 3.0)
    e = torch.exp(-k * t)[:, None]
    vt_xy = torch.zeros_like(p_w[:, :2])      # no horizontal terminal velocity
    return p_w[:, :2] + vt_xy * t[:, None] + (v_w[:, :2] - vt_xy) * (1.0 - e) / k


def v38_go_to_landing(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """exp(-|base_xy - landing_xy|^2 / sigma^2) while the box is falling."""
    box = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    p = box.data.root_pos_w
    v = box.data.root_lin_vel_w
    land = _predict_landing_xy(p, v, BOX_LAND_H)
    d = (robot.data.root_pos_w[:, :2] - land).norm(dim=-1)
    r = torch.exp(-(d / BOX_LAND_SIG) ** 2)
    # The descending-only gate turns this signal on too late: a 3 m throw
    # hangs ~1.13 s and only the last ~0.5 s of that is descent, while walking
    # 0.5 m takes ~0.9 s.  The ballistic solve is just as valid on the way up,
    # so BOX_V38_LAND_ALLFLIGHT=1 pays from the moment of release.
    if BOX_LAND_ALLFLIGHT:
        live = p[:, 2] > BOX_LAND_H + 0.15
    else:
        live = (p[:, 2] > BOX_LAND_H + 0.15) & (v[:, 2] < 0.0)
    launched = getattr(env, "_box_launched", None)
    if launched is not None:
        live = live & launched
    return torch.where(live, r, torch.zeros_like(r))


# ============================================================================
# Anchor-orientation reward with the yaw target swapped to the box bearing
# (optional, BOX_V38_ORI_FACE=1)
#
# Why the other facing mechanisms fall short:
#   * motion_global_anchor_ori (w 0.5, std 0.4 rad) pins global heading to the
#     clip, so turning 42 deg costs ~0.48/step while a facing reward pays at
#     most 0.10/step;
#   * rotating the clip toward the box (v38_anchor_face_box) rotates it about
#     the ENV ORIGIN, and this clip's torso sits 1.679 m from that origin, so
#     a 16 deg slew silently commands a 0.47 m translation and trades the
#     heading penalty for an equal-or-larger position penalty.
#
# This term changes the TARGET instead of the command: roll/pitch still track
# the clip (uprightness preserved), while yaw tracks the bearing to the box
# whenever one is in the air and falls back to the clip's yaw otherwise.  Same
# weight, std and exp shape, so the reward budget is unchanged, and it touches
# no observation (the box position is already in obs[160:163]).
ORI_FACE_H = float(os.environ.get("BOX_V38_ORI_FACE_H", "1.15"))

# Hover-phase yaw std.  exp(-yaw_err^2 / 0.4^2) is FLAT past ~40 deg: from a
# 60 deg bearing error, turning 10 deg pays 0.004/step against a ~0.17/step
# living reward, under the noise floor, even though the optimum is worth
# 0.5/step.  The policy cannot SEE the gradient that leads there (a classic
# narrow-Gaussian exploration trap).  So widen the yaw Gaussian ONLY during
# the hover, which is (a) the only phase with time to turn 60 deg and (b) the
# only phase where the error gets large.  Once the box is launched the
# original std is restored, so catch-time yaw precision is untouched.
#   std 1.0 rad: 60 deg -> 0.33, 50 deg -> 0.47  => 0.067/step per 10 deg,
#   ~39% of the living reward, a gradient the policy can follow.
# 0 = use `std` throughout.
ORI_YAW_STD_HOVER = float(os.environ.get("BOX_V38_ORI_YAW_STD_HOVER", "0"))


def _yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _qz(yaw):
    h = 0.5 * yaw
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(h), z, z, torch.sin(h)], dim=-1)


def _qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qconj(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


# ============================================================================
# Orbiting thrower (optional, BOX_V38_ORBIT_DPS > 0)
#
# The walking thrower (v36_walking_hover_and_launch) random-walks inside a
# 1.5 m disc around its OWN spawn point with the throw 3.0-3.5 m out, so the
# bearing seen by the robot only swings +-atan(1.5/3.5) = +-23 deg.  That
# cannot teach a robot to keep facing someone who walks a real arc around it.
#
# This moves the throw origin ALONG AN ARC CENTRED ON THE ROBOT instead: same
# distance, swinging bearing, i.e. a person circling the robot.  It composes
# with the disc walk rather than replacing it (the walk's anchor is dragged
# along), so the radius jitter and launch re-aim are untouched.
V38_ORBIT_DPS = float(os.environ.get("BOX_V38_ORBIT_DPS", "0"))    # deg/s
V38_ORBIT_SPAN = float(os.environ.get("BOX_V38_ORBIT_SPAN", "120"))  # +- deg


def v38_orbit_thrower(env, env_ids,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")):
    """Interval event: swing the hover point around the robot on an arc.

    `env_ids` must stay MANDATORY (no default).  Isaac Lab's EventManager
    inspects the signature and rejects the term if env_ids carries a default
    ("expects mandatory parameters: [] ... but received: []"), which surfaces
    only at env construction -- i.e. after a run has been launched.
    """
    del env_ids, asset_cfg
    if V38_ORBIT_DPS <= 0.0 or not hasattr(env, "_box_p0"):
        return
    n = env.num_envs
    dev = env._box_p0.device
    if not hasattr(env, "_v38_orbit_w"):
        env._v38_orbit_w = torch.zeros(n, device=dev)
        env._v38_orbit_a0 = torch.zeros(n, device=dev)
        env._v38_orbit_c = torch.zeros(n, 2, device=dev)
        env._v38_orbit_r = torch.zeros(n, device=dev)
    step = env.episode_length_buf
    hovering = (~env._box_launched) & (step < env._box_launch_step)
    rp = env.scene["robot"].data.root_pos_w[:, :2]

    fresh = hovering & (env._v38_orbit_w.abs() < 1e-6)
    if bool(fresh.any()):
        k = int(fresh.sum())
        sign = torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
        env._v38_orbit_w[fresh] = sign * math.radians(V38_ORBIT_DPS)
        # FREEZE the circle at hover start.  Recomputing centre and radius from
        # the LIVE robot position every tick lets a walking robot drag the
        # thrower along, and walking toward the thrower collapses the radius
        # and degenerates the launch solve (T = sqrt(d/4) -> 0), which nearly
        # stops all throws.  A person circling you does not follow you around.
        _d0 = env._box_p0[fresh, :2] - rp[fresh]
        env._v38_orbit_c[fresh] = rp[fresh]
        env._v38_orbit_r[fresh] = _d0.norm(dim=-1).clamp_min(1.0)
        env._v38_orbit_a0[fresh] = torch.atan2(_d0[:, 1], _d0[:, 0])

    d = env._box_p0[:, :2] - env._v38_orbit_c
    ang = torch.atan2(d[:, 1], d[:, 0])
    r = env._v38_orbit_r

    new_ang = ang + env._v38_orbit_w * env.step_dt
    off = torch.atan2(torch.sin(new_ang - env._v38_orbit_a0),
                      torch.cos(new_ang - env._v38_orbit_a0))
    flip = off.abs() > math.radians(V38_ORBIT_SPAN)
    if bool(flip.any()):                                # bounce at the arc ends
        env._v38_orbit_w[flip] = -env._v38_orbit_w[flip]
        new_ang = torch.where(flip, ang, new_ang)
    new_xy = rp + torch.stack([torch.cos(new_ang), torch.sin(new_ang)],
                              dim=-1) * r[:, None]
    env._box_p0[:, :2] = torch.where(hovering[:, None], new_xy,
                                     env._box_p0[:, :2])
    # drag the disc-walk anchor along so the two mechanisms compose instead of
    # fighting (the walk clamps to a disc around this anchor).
    if hasattr(env, "_v36_walk_anchor"):
        env._v36_walk_anchor = torch.where(hovering[:, None], new_xy,
                                           env._v36_walk_anchor)
    env._v38_orbit_w = torch.where(hovering, env._v38_orbit_w,
                                   torch.zeros_like(env._v38_orbit_w))


def v38_anchor_ori_facebox(env, command_name: str = "motion", std: float = 0.4,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                           ) -> torch.Tensor:
    """Drop-in for motion_global_anchor_orientation_error_exp with the yaw
    target swapped to the box bearing while a throw is in the air."""
    from isaaclab.utils.math import quat_error_magnitude
    cmd = env.command_manager.get_term(command_name)
    q_ref, q_rob = cmd.anchor_quat_w, cmd.robot_anchor_quat_w
    yaw_ref, yaw_rob = _yaw_of(q_ref), _yaw_of(q_rob)

    # roll/pitch: compare after removing each one's own yaw
    rp_ref = _qmul(_qconj(_qz(yaw_ref)), q_ref)
    rp_rob = _qmul(_qconj(_qz(yaw_rob)), q_rob)
    rp_err = quat_error_magnitude(rp_ref, rp_rob)

    # yaw target: bearing to the box while it is airborne, else the clip's yaw
    box = env.scene[asset_cfg.name]
    d = box.data.root_pos_w[:, :2] - cmd.robot_anchor_pos_w[:, :2]
    want = torch.atan2(d[:, 1], d[:, 0])
    # Active while the box is airborne (hover OR flight) and not cradled.
    # Gating on _box_launched alone would miss the hover, which is the only
    # window with time to turn ~45 deg (the flight is ~1 s).
    live = box.data.root_pos_w[:, 2] > ORI_FACE_H
    live = live & (~_cradled(env, box))
    want = torch.where(live, want, yaw_ref)
    yaw_err = torch.atan2(torch.sin(want - yaw_rob), torch.cos(want - yaw_rob))

    # phase-dependent yaw sharpness (see ORI_YAW_STD_HOVER above): wide while the
    # thrower still holds the box so the turn has a followable gradient, back to
    # the clip's std the instant it is released so catch-time yaw stays precise.
    yaw_std = torch.full_like(yaw_err, std)
    if ORI_YAW_STD_HOVER > 0.0:
        launched = getattr(env, "_box_launched", None)
        if launched is not None:
            yaw_std = torch.where(launched, yaw_std,
                                  torch.full_like(yaw_err, ORI_YAW_STD_HOVER))
    err = rp_err ** 2 / std ** 2 + yaw_err ** 2 / yaw_std ** 2
    return torch.exp(-err)


# ============================================================================
# Predicted landing point as an OBSERVATION (optional, BOX_V38_LAND_OBS=1)
#
# Why an observation and not another reward: under ground-truth perception the
# legs are not the limit (a GT-trained policy catches 93% at 0.4-0.6 m and 64%
# at 0.6-0.8 m offsets), and the landing point IS recoverable from the noisy
# filter state (0.21 m median landing error with >= 0.6 s of flight left), but
# a network trained on noisy perception evidently does not learn the
# drag-ballistic solve from noisy position+velocity.  This term does the solve
# in closed form and hands over the answer.  It consumes exactly the filtered
# position/velocity the robot also has; nothing here is sim-only.
LAND_OBS_H = float(os.environ.get("BOX_V38_LANDOBS_H", "1.0"))
LAND_OBS_TMAX = float(os.environ.get("BOX_V38_LANDOBS_TMAX", "2.0"))


def balloon_landing_pelvis_frame(env,
                                 asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                                 ) -> torch.Tensor:
    """(N,3): predicted landing point in pelvis frame (x,y) + seconds to land.

    Uses the SAME estimate the policy's box obs is built from, so it inherits
    the estimator's noise instead of smuggling in ground truth.
    """
    from isaaclab.utils.math import quat_apply_inverse
    from reflex.envs import robust_estimator as v36e
    s = v36e._advance(env, asset_cfg)
    robot = env.scene["robot"]
    box = env.scene[asset_cfg.name]
    p_w = torch.where(s["clean_ep"][:, None], box.data.root_pos_w, s["est_p_w"])
    bv = torch.nan_to_num(box.data.root_lin_vel_w, nan=0.0, posinf=10.0, neginf=-10.0)
    v_w = torch.where(s["clean_ep"][:, None], bv, s["est_v_w"])

    k = BOX_LIN_DAMPING
    vt_z = -GRAVITY / k
    dev = p_w.device
    # dense scan for the first DESCENDING crossing of LAND_OBS_H: a Newton
    # solve with a clamped derivative silently diverges while the box is still
    # rising.
    t = torch.arange(0.02, LAND_OBS_TMAX, 0.02, device=dev)          # (T,)
    e = torch.exp(-k * t)[None, :]                                   # (1,T)
    z = (p_w[:, 2:3] + vt_z * t[None, :]
         + (v_w[:, 2:3] - vt_z) * (1.0 - e) / k)                     # (N,T)
    below = z <= LAND_OBS_H
    first = torch.where(below.any(dim=1),
                        below.float().argmax(dim=1),
                        torch.full((p_w.shape[0],), len(t) - 1, device=dev,
                                   dtype=torch.long))
    t_land = t[first]                                                # (N,)
    ee = torch.exp(-k * t_land)[:, None]
    land_w = (p_w + torch.tensor([0.0, 0.0, vt_z], device=dev) * t_land[:, None]
              + (v_w - torch.tensor([0.0, 0.0, vt_z], device=dev)) * (1.0 - ee) / k)
    rel = quat_apply_inverse(robot.data.root_quat_w,
                             land_w - robot.data.root_pos_w)
    return torch.cat([rel[:, :2], t_land[:, None]], dim=-1)


# ============================================================================
# Re-centre the motion clip on its catch-window anchor (optional,
# BOX_V38_RECENTRE=1)
#
# Goal: BEFORE the throw the robot should keep facing the thrower, turning to
# follow if the thrower walks around it.  Mid-flight turning is far too late
# (the flight is ~1 s); the pre-throw hover is where there is time, and the
# walking thrower already moves during it.
#
# What blocks it: yaw_offsets rotates the clip about the ENV ORIGIN, and this
# clip's torso sits 1.679 m from that origin during the catch window, so a
# 16 deg slew silently commands a 0.47 m translation (a 90 deg turn would
# command 2.4 m), trading a heading penalty for a position penalty.
#
# Rather than subclass MotionCommand to add a position offset, shift the
# loaded clip once at startup so the catch-window anchor IS the origin.  Then
# "rotate about the origin" and "rotate in place" are the same operation.
# Unset = no change (the walking phase simply starts 1.68 m further out).
def v38_recentre_clip_on_anchor(env, env_ids=None):
    """Startup event: translate the motion clip so the still-phase anchor is at
    the motion origin.  Runs once; idempotent via a flag on env."""
    if getattr(env, "_v38_clip_recentred", False):
        return
    cmd = env.command_manager.get_term("motion")
    m = cmd.motion
    bi = cmd.motion_anchor_body_index
    p = m.body_pos_w                                  # (T, B, 3)
    # the still phase is the tail of the clip: take the last frame's anchor xy
    ref_xy = p[-1, bi, :2].clone()
    if torch.allclose(ref_xy, torch.zeros_like(ref_xy), atol=1e-4):
        print("[teacher] clip already centred on anchor; nothing to do",
              flush=True)
        env._v38_clip_recentred = True
        return
    m.body_pos_w[:, :, :2] -= ref_xy[None, None, :]
    env._v38_clip_recentred = True
    print(f"[teacher] clip re-centred on catch-window anchor: shifted by "
          f"({-ref_xy[0]:+.3f}, {-ref_xy[1]:+.3f}) m -> yaw_offsets now "
          f"rotates IN PLACE (was an arc of radius "
          f"{float(ref_xy.norm()):.3f} m)", flush=True)


# ============================================================================
# Reactive arms: reward gate (optional, BOX_V38_ARMS_REACTIVE=1)
#
# Without it the robot pins its arms in the receive pose from the FIRST step
# (arm_snap_apply: every episode starts palms-up by construction) and the
# ready-pose rewards (cradle_pose w 3.5, palm_up, arm_idle_still) hold them
# there all episode, which reads as a prop rather than something reacting.  A
# person keeps their arms down and brings them up once the throw is on its way.
#
# Timing is not the obstacle: during the catch the arms barely move, and a
# 1.5 rad shoulder lift needs ~0.3 s against ~1.5 s of flight.  Simply
# switching ARM_SNAP off on a policy TRAINED with it is out of distribution;
# the fix is to retrain with the gate.
#
# Mechanism: leave the reward functions alone and multiply them by a gate that
# is 1 while the box is in flight or already cradled and 0 while waiting, so
# the motion-tracking terms own the idle pose (arms down in the clip).
# Unset => no gating.
V38_ARMS_REACTIVE = os.environ.get("BOX_V38_ARMS_REACTIVE", "0") == "1"


def _reactive_gate(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")):
    """1 while the box is genuinely IN FLIGHT or held, else 0.

    `_box_launched` LATCHES for the rest of the episode once a throw happens,
    so gating on it alone leaves the arms up after a miss, with the box lying
    on the floor.  Require the box to actually be up in the air as well, so a
    dropped box lets the arms come back down.
    """
    box = env.scene[asset_cfg.name]
    launched = getattr(env, "_box_launched", None)
    if launched is None:
        return torch.ones(env.num_envs, device=box.data.root_pos_w.device)
    airborne = box.data.root_pos_w[:, 2] > GATE_AIRBORNE_H
    return ((launched & airborne) | _cradled(env, box)).float()


# height above which the box counts as "in the air" for the reactive gate; a
# box resting on the floor sits at roughly half its own height (~0.11-0.17 m).
GATE_AIRBORNE_H = float(os.environ.get("BOX_V38_GATE_AIR_H", "0.45"))


def wrap_reactive(fn):
    """Multiply a ready-pose reward by the reactive gate, PRESERVING the real
    signature.

    Three traps, all firing only at env construction:
      1. IsaacLab serialises RewTerm.func to "module:name" and re-resolves it,
         so the wrapper must be a module attribute (else AttributeError).
      2. It inspects the RAW code object, which does NOT follow functools.wraps'
         __wrapped__ -- a `def w(env, *a, **kw)` wrapper reports two mandatory
         params named 'a' and 'kw'.  functools.wraps does not help.
      3. Building the def from str(signature) embeds the repr of default values
         and the return annotation, which is not valid source
         (SceneEntityCfg(name='balloon', ...) -> unterminated string literal).
    So emit only parameter NAMES and bind the defaults through the namespace.
    """
    import inspect as _i
    import sys as _s

    sig = _i.signature(fn)
    ps = list(sig.parameters.values())
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in ps):
        return fn                       # exotic signature: leave it alone
    ns = {"_inner": fn, "_gate": _reactive_gate}
    decl = []
    for i, p in enumerate(ps):
        if p.default is _i.Parameter.empty:
            decl.append(p.name)
        else:
            ns["_d%d" % i] = p.default
            decl.append("%s=_d%d" % (p.name, i))
    name = getattr(fn, "__name__", "fn") + "_reactive"
    src = ("def %s(%s):\n    return _inner(%s) * _gate(%s)\n"
           % (name, ", ".join(decl), ", ".join(p.name for p in ps), ps[0].name))
    exec(compile(src, "<reactive>", "exec"), ns)          # noqa: S102
    w = ns[name]
    w.__module__ = __name__
    setattr(_s.modules[__name__], name, w)
    return w


# ============================================================================
# Reactive arms: reference motion (optional, BOX_V38_REF_ARMS=1)
#
# The reference clip itself holds the arms up:
#     0.0-6.0 s   hands +0.01 m above the pelvis      (arms DOWN)
#     6.0-9.0 s   rise
#     9.0-30.0 s  hands +0.29 m, frozen               (arms UP, 70% of the clip)
# and every catch happens inside that frozen stretch.  MotionCommand.joint_pos
# feeds those angles to the tracking reward AND to the policy's observation,
# so the arms are pulled up no matter what the reward gating says.
#
# Fix without new motion data: while no throw is live, serve the ARM joints
# from the clip's own early (arms-down) window instead of the current frame.
# Legs, torso and timing are untouched, both poses come from the same clip, and
# the policy OBSERVES the lowered reference rather than having to infer it.
# Unset => property not patched.
V38_REF_ARMS_REACTIVE = os.environ.get("BOX_V38_REF_ARMS", "0") == "1"
V38_REF_DOWN_LO = float(os.environ.get("BOX_V38_REF_DOWN_LO", "0.0"))   # s
V38_REF_DOWN_HI = float(os.environ.get("BOX_V38_REF_DOWN_HI", "5.0"))   # s


def patch_reference_arms(arm_keywords=("shoulder", "elbow", "wrist")):
    """Monkey-patch MotionCommand.joint_pos so the arms follow the clip's own
    arms-down window while no throw is live.  Called once from the env cfg."""
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
    if getattr(MotionCommand, "_v38_ref_arms_patched", False):
        return
    _orig = MotionCommand.joint_pos.fget

    def _reactive(self):
        q = _orig(self)
        env = getattr(self, "_env", None)
        if env is None or not hasattr(env, "_box_launched"):
            return q
        cache = getattr(self, "_v38_arm_cache", None)
        if cache is None:
            names = list(env.scene["robot"].data.joint_names)
            idx = torch.tensor(
                [i for i, n in enumerate(names)
                 if any(k in n.lower() for k in arm_keywords)], device=q.device)
            fps = float(getattr(self.motion, "fps", 50.0))
            lo = int(V38_REF_DOWN_LO * fps)
            hi = max(lo + 1, int(V38_REF_DOWN_HI * fps))
            down = self.motion.joint_pos[lo:hi][:, idx].mean(dim=0)
            cache = (idx, down)
            self._v38_arm_cache = cache
            print(f"[teacher] reference arms patched: {len(idx)} arm joints served from "
                  f"clip {V38_REF_DOWN_LO:g}-{V38_REF_DOWN_HI:g}s while waiting",
                  flush=True)
        idx, down = cache
        box = env.scene["balloon"]
        live = env._box_launched | _cradled(env, box)
        q = q.clone()
        q[:, idx] = torch.where(live[:, None], q[:, idx], down[None, :])
        return q

    MotionCommand.joint_pos = property(_reactive)
    MotionCommand._v38_ref_arms_patched = True


# ============================================================================
# Reactive arms: early-raise penalty (optional, BOX_V38_ARM_EARLY_W > 0)
#
# Removing the REASONS to raise (reward gate, reference arms) left the hand
# height above the pelvis essentially unchanged between hover and flight
# (~17-19 cm in both): the catch bonus dwarfs the tracking penalty, so the
# policy eats the tracking cost and pre-raises anyway.  This adds a PRICE.
#
# Penalise deviation from the clip's own arms-down pose (NOT hand height above
# the pelvis, which the policy can game by moving the pelvis while the arms
# stay up).  Active only while the box is hovering (not in flight, not held),
# so the catch itself is never penalised.  Scale: ~0.05/step at w=0.2, ~30% of
# the ~0.17/step living reward.  Keep it at 0 together with the two-stage arm
# target: penalising an arm the blend itself is raising makes ending the
# episode attractive.
V38_ARM_EARLY_W = float(os.environ.get("BOX_V38_ARM_EARLY_W", "0"))


def v38_arm_early_raise_penalty(
        env, asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")
) -> torch.Tensor:
    """Squared arm-joint deviation from the clip's down pose, during hover only."""
    if V38_ARM_EARLY_W <= 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    box = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    cache = getattr(env, "_v38_early_cache", None)
    if cache is None:
        names = list(robot.data.joint_names)
        idx = torch.tensor(
            [i for i, n in enumerate(names)
             if any(k in n.lower() for k in ("shoulder", "elbow", "wrist"))],
            device=robot.data.joint_pos.device)
        cmd = env.command_manager.get_term("motion")
        fps = float(getattr(cmd.motion, "fps", 50.0))
        lo, hi = int(V38_REF_DOWN_LO * fps), max(1, int(V38_REF_DOWN_HI * fps))
        down = cmd.motion.joint_pos[lo:hi][:, idx].mean(dim=0)
        cache = (idx, down)
        env._v38_early_cache = cache
        print(f"[teacher] arm early-raise penalty: w={V38_ARM_EARLY_W:g} on "
              f"{len(idx)} arm joints vs clip {V38_REF_DOWN_LO:g}-"
              f"{V38_REF_DOWN_HI:g}s down pose", flush=True)
    idx, down = cache
    launched = getattr(env, "_box_launched", None)
    if launched is None:
        return torch.zeros(env.num_envs, device=robot.data.joint_pos.device)
    hovering = (~launched) & (~_cradled(env, box))
    dev = (robot.data.joint_pos[:, idx] - down[None, :]).pow(2).sum(dim=-1)
    return -(dev * hovering.float())


# ---------------------------------------------------------------------------
# Reactive arms: two-stage arm POSITION TARGET (BOX_V38_ARM_TWOSTAGE=1)
#
# The arms are not learned but pinned by config (catch_env_cfg):
#     BOX_ARM_INIT_CRADLE=1 -> shoulder/elbow default_joint_pos = cradle pose
#     BOX_ARM_SCALE=0.02    -> shoulder action scale 0.44 -> 0.02
# so the PD target IS the raised cradle and the policy owns only +-1.15 deg of
# the shoulder (the elbow has more authority, BOX_ELBOW_SCALE=0.35).  No reward
# can lower the arms, so the scripted pose is made two-stage instead: while the
# throw is not live the arm base pose is the clip's own down pose; once it is
# live it is the cradle (+ the policy's delta), exactly as before.  The cradle
# that does the catching is untouched; only the idle pose changes.
# A linear blend (V38_TS_RAMP steps) avoids stepping a 1.2 rad target change
# into a 25 Nm shoulder in one tick.
V38_ARM_TWO_STAGE = os.environ.get("BOX_V38_ARM_TWOSTAGE", "0") == "1"
V38_TS_RAMP = int(os.environ.get("BOX_V38_TS_RAMP", "8"))          # steps
# ---- perception-triggered variant (BOX_V38_TS_FROM_PERC=1) ----
# env._box_launched comes from the simulator's throw scheduler and does not
# exist on the robot, so the deployable trigger uses the filtered box estimate
# (env._v31_est["est_p_w"] / ["est_v_w"], the same filter as on the robot).
# Rules learned from replaying real filter logs:
#   * speed alone is the wrong signal: the onboard filter zeroes velocity on
#     contact freeze and on runaway resets, and in the last 0.3 s before
#     contact the median filtered speed was 0, so the arms would still be down
#     when the box arrives;
#   * trigger on DISTANCE instead (the box is 3.0-3.5 m away while held, so
#     "closer than D_TRIG" mostly happens in flight, and position survives
#     the freeze), AND require the distance to be closing (see TS_CLOSE);
#   * LATCH the raise for TS_LATCH steps instead of following the signal tick
#     by tick, so a dropout or freeze mid-flight cannot drop the arms (real
#     flights are 0.25-1.5 s; a 1.5 s latch covers one throw and expires
#     before the next);
#   * rise fast, fall slow (asymmetric rates), for the same reason.
# On no-track the estimator returns the far-ahead prior (3.0 m, v=0), so
# nothing fires and the latch alone decides.
V38_TS_D_TRIG = float(os.environ.get("BOX_V38_TS_D_TRIG", "2.5"))   # m
V38_TS_LATCH = int(os.environ.get("BOX_V38_TS_LATCH", "75"))        # steps=1.5s
V38_TS_CLOSE = float(os.environ.get("BOX_V38_TS_CLOSE", "1.0"))    # m/s closing
# The optional SPEED path (BOX_V38_TS_USE_SPEED) was the main source of hover
# false triggers in replays (hover raise 0.27-0.39 vs 0.12-0.13 without it,
# against a 0.15 bar, with unchanged contact raise), hence off by default.
# `miss <= MISS_MAX` blocks the "filter lost the box, the prior is close,
# distance fires" path.  Latch 75 steps: 50 hurt the low tail of the raise.
#
# Down-pose override (one definition shared by sim and robot).  By default the
# down pose is computed from the clip, but the clip's standing pose is
# asymmetric under the humanoid mirror convention (shoulder yaw +0.061 /
# +0.557, both positive, where mirroring requires opposite signs), so one arm
# would look visibly rotated.  Standing-only frames do not help; the asymmetry
# is intrinsic to the clip.  8 comma-separated values in Isaac joint order
# [11,12,15,16,19,20,21,22]; unset = compute from the clip.
V38_TS_DOWN = os.environ.get("BOX_V38_TS_DOWN", "")
# Raised-pose override: a delta added to the cradle (the action term's offset)
# while the arms are up, to test whether a higher "bed" during flight helps
# (a higher hand height while the box is still > 1 m away was the main
# difference found between a stronger and a weaker policy).  8 comma-separated
# values in Isaac order [11,12,15,16,19,20,21,22]; unset = cradle.
V38_TS_UP = os.environ.get("BOX_V38_TS_UP", "")
# Flight reward gate (optional): scale the listed tracking/stillness rewards by
# FLIGHT_RGATE_FAC while a throw is in flight, so they do not brake the catch.
V38_FLIGHT_RGATE = os.environ.get("BOX_V38_FLIGHT_RGATE", "0") == "1"
V38_FLIGHT_RGATE_FAC = float(os.environ.get("BOX_V38_FLIGHT_RGATE_FAC", "0.2"))
V38_FLIGHT_RGATE_TERMS = os.environ.get(
    "BOX_V38_FLIGHT_RGATE_TERMS",
    "motion_body_pos,motion_body_lin_vel,motion_body_ang_vel,"
    "low_trust_stillness,hold_still")


def _apply_flight_rgate(env):
    """Runtime reward-manager patch for the flight reward gate.  Called once
    from the per-step event path: after env construction there is no further
    cfg serialization, so the wrapper cannot be silently re-resolved away (as
    happens when cfg.func is wrapped in the env cfg)."""
    if not V38_FLIGHT_RGATE or getattr(env, "_frg_done", False):
        return
    env._frg_done = True
    rm = env.reward_manager
    names = [n.strip() for n in V38_FLIGHT_RGATE_TERMS.split(",")]
    patched = []
    for idx, tn in enumerate(rm.active_terms):
        if tn not in names:
            continue
        cfg = rm._term_cfgs[idx]
        _orig = cfg.func

        def _mk(_f, _n):
            def _gated(e, *a, **kw):
                r = _f(e, *a, **kw)
                fl = getattr(e, "_box_launched", None)
                if fl is None or not bool(fl.any()):
                    return r
                if not getattr(e, "_frg_fired", False):
                    e._frg_fired = True
                    print("[box_v38] FLIGHT REWARD GATE FIRED (x%.2f on %s; "
                          "runtime, post-serialization)"
                          % (V38_FLIGHT_RGATE_FAC, _n), flush=True)
                return torch.where(fl, r * V38_FLIGHT_RGATE_FAC, r)
            return _gated
        cfg.func = _mk(_orig, tn)
        patched.append(tn)
    print("[box_v38] FLIGHT RGATE v2 patched %d terms: %s" % (len(patched), patched),
          flush=True)

# Delta gate (optional): scale the policy's arm-action delta by the two-stage
# blend b: TS_DELTA_MIN while waiting (b=0), full authority after the trigger
# (b->1).  A hand-height metric cannot tell "hanging at the sides" from
# "reaching forward at hip height", and reward-side fixes invite the same
# gaming, so hanging is made mechanical rather than learned.
V38_TS_DELTA_GATE = os.environ.get("BOX_V38_TS_DELTA_GATE", "0") == "1"
V38_TS_DELTA_MIN = float(os.environ.get("BOX_V38_TS_DELTA_MIN", "0.15"))
# blend the BASE POSE and keep the policy delta at full authority (default),
# instead of blending the whole target and thereby gating the policy's arm
# action by b (0 = whole-target blend).
V38_TS_BASE_BLEND = os.environ.get("BOX_V38_TS_BASE_BLEND", "1") == "1"
V38_TS_USE_SPEED = os.environ.get("BOX_V38_TS_USE_SPEED", "0") == "1"
V38_TS_MISS_MAX = int(os.environ.get("BOX_V38_TS_MISS_MAX", "6"))
V38_TS_FROM_PERC = os.environ.get("BOX_V38_TS_FROM_PERC", "0") == "1"
V38_TS_V_LO = float(os.environ.get("BOX_V38_TS_V_LO", "1.5"))      # m/s
# Speed-path debounce: the fire condition must hold this many CONSECUTIVE
# steps.  Filter jump frames (dirty MIX_JUMP) spike the estimated speed for 1-3
# steps while a real throw sustains > V_LO all flight; without debouncing,
# constant false fires bake a raised-arm hover pose into the policy.
V38_TS_SPEED_SUSTAIN = int(os.environ.get("BOX_V38_TS_SPEED_SUSTAIN", "1"))

# ---- pre-launch arm-hang reward (optional, BOX_V38_HANG_W > 0) ----
# With the arm standard enforced only by mechanical fences (two-stage blend,
# delta gate, trigger) the objective has zero gradient toward hanging arms, so
# any leak in a fence lets the policy drift toward arms-always-up.  This puts
# the standard INTO the objective: a POSITIVE reward for the down pose while
# the box has not been launched.  _box_launched is sim-only, which is fine
# here because rewards never deploy (unlike observations/triggers).  Before
# launch the two-stage blend already holds the arms down, so the reward is
# attainable; this is unlike ARM_EARLY_W, which penalised an arm the blend
# itself was raising.
V38_HANG_W = float(os.environ.get("BOX_V38_HANG_W", "0"))
V38_HANG_SIG = float(os.environ.get("BOX_V38_HANG_SIG", "0.35"))   # rad


def v38_hang_prelaunch_reward(env) -> torch.Tensor:
    """exp(-mean sq. shoulder/elbow deviation from the down pose / HANG_SIG^2),
    paid only before launch and with no caught box in the arms."""
    robot = env.scene["robot"]
    cache = getattr(env, "_v38_hang_cache", None)
    if cache is None:
        names = list(robot.data.joint_names)
        jidx = [i for i, n in enumerate(names)
                if "shoulder" in n.lower() or "elbow" in n.lower()]
        if V38_TS_DOWN.strip():
            vals = [float(x) for x in V38_TS_DOWN.split(",")]
            if len(vals) != len(jidx):
                raise ValueError("BOX_V38_TS_DOWN has %d values, need %d"
                                 % (len(vals), len(jidx)))
            down = torch.tensor(vals, device=robot.device)
        else:
            down = robot.data.default_joint_pos[0, jidx].clone()
        cache = (torch.tensor(jidx, dtype=torch.long, device=robot.device),
                 down)
        env._v38_hang_cache = cache
        print("[box_v38] HANG-PRELAUNCH REWARD FIRED: w=%g sig=%g joints=%d"
              % (V38_HANG_W, V38_HANG_SIG, len(jidx)), flush=True)
    jidx_t, down = cache
    q = robot.data.joint_pos[:, jidx_t]
    err = ((q - down.unsqueeze(0)) ** 2).mean(dim=-1)
    r = torch.exp(-err / (V38_HANG_SIG ** 2))
    fl = getattr(env, "_box_launched", None)
    if fl is None:
        return torch.zeros_like(r)
    # Pay ONLY when (a) box not launched AND (b) no caught box in the arms.
    # Without (b), the flag reset at the next hover would pay the robot to
    # dump a held box early, fighting the hold rewards and the _cradled latch.
    idle = ~fl
    try:
        idle = idle & ~_cradled(env, env.scene["balloon"])
    except Exception:
        pass
    return torch.where(idle, r, torch.zeros_like(r))
V38_TS_V_HI = float(os.environ.get("BOX_V38_TS_V_HI", "3.0"))      # m/s
V38_TS_FALL = int(os.environ.get("BOX_V38_TS_FALL", "40"))         # steps


def patch_arm_two_stage():
    """Override the arm POSITION TARGET (not the reward) while no throw is
    live.  Called once from the env cfg, after the action term exists."""
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
    if getattr(JointPositionAction, "_v38_two_stage", False):
        return
    _orig = JointPositionAction.process_actions

    def _proc(self, actions):
        _orig(self, actions)
        env = getattr(self, "_env", None)
        if env is None or not hasattr(env, "_box_launched"):
            return
        cache = getattr(self, "_v38_ts_cache", None)
        if cache is None:
            names = list(self._asset.data.joint_names)
            jids = self._joint_ids
            jlist = (list(range(len(names))) if isinstance(jids, slice)
                     else list(jids))
            # action column -> joint name
            cols, jidx = [], []
            for _c, _j in enumerate(jlist):
                if any(k in names[_j].lower() for k in ("shoulder", "elbow")):
                    cols.append(_c)
                    jidx.append(_j)
            # down pose: the same clip window as the reference-arms patch
            down = None
            for _t in getattr(env, "command_manager", None)._terms.values():
                _m = getattr(_t, "motion", None)
                if _m is not None and hasattr(_m, "joint_pos"):
                    _fps = float(getattr(_m, "fps", 50.0))
                    _lo = int(V38_REF_DOWN_LO * _fps)
                    _hi = max(_lo + 1, int(V38_REF_DOWN_HI * _fps))
                    down = _m.joint_pos[_lo:_hi][:, jidx].mean(dim=0)
                    break
            if V38_TS_DOWN.strip():
                _ov = [float(x) for x in V38_TS_DOWN.split(",")]
                if len(_ov) != len(cols):
                    raise ValueError(
                        "BOX_V38_TS_DOWN has %d values, need %d (ISAAC order "
                        "%s)" % (len(_ov), len(cols), jidx))
                down = torch.tensor(_ov, device=self.device)
                print("[box_v38] TWO-STAGE ARMS: down pose OVERRIDDEN by "
                      "BOX_V38_TS_DOWN = %s" % _ov, flush=True)
            if down is None:
                print("[teacher] two-stage arms: no motion term found; "
                      "cannot build the down pose, two-stage target disabled", flush=True)
                self._v38_ts_cache = (None, None, None)
                return
            cols_t = torch.tensor(cols, device=self.device, dtype=torch.long)
            # the action term's own offset for these columns = the cradle
            # default; needed to separate "base pose" from "policy delta".
            _off = self._offset
            self._offset_arm = (_off[:, cols_t] if torch.is_tensor(_off)
                                and _off.dim() == 2 else
                                torch.zeros(1, len(cols), device=self.device))
            # raised base pose: cradle by default, shiftable to test a higher
            # bed during flight (see V38_TS_UP above).
            #
            # DELTA, not replacement: the cradle offset is (num_envs, 8) and
            # domain-randomised per env (spread ~0.02 rad).  An absolute
            # override would replace that randomisation with one fixed pose,
            # so a raised-pose comparison would differ in two things (the lift
            # AND the loss of randomisation).  Adding a delta keeps the spread.
            self._up_arm = self._offset_arm
            if V38_TS_UP.strip():
                _uv = [float(x) for x in V38_TS_UP.split(",")]
                if len(_uv) != len(cols):
                    raise ValueError(
                        "BOX_V38_TS_UP has %d values, need %d (ISAAC order "
                        "[11,12,15,16,19,20,21,22])" % (len(_uv), len(cols)))
                self._up_arm = self._offset_arm + torch.tensor(
                    _uv, device=self.device, dtype=torch.float32)[None, :]
            blend = torch.zeros(env.num_envs, device=self.device)
            cache = (cols_t, down.to(self.device), blend)
            self._v38_ts_cache = cache
            _trig = ("PERCEPTION LATCHED (dist<%.1f m AND closing>%.1f m/s AND "
                     "miss<=%d%s -> hold %d steps; rise %d / fall %d) — DEPLOYABLE"
                     % (V38_TS_D_TRIG, V38_TS_CLOSE, V38_TS_MISS_MAX,
                        (" OR speed>%.1f x%d" % (V38_TS_V_LO, V38_TS_SPEED_SUSTAIN)) if V38_TS_USE_SPEED
                        else "", V38_TS_LATCH, V38_TS_RAMP, V38_TS_FALL)
                     if V38_TS_FROM_PERC else
                     "GT _box_launched flag — SIM ONLY, NOT DEPLOYABLE")
            # print the actual 8 values (not just "clip 0-5s mean"), so the
            # down pose can be compared value-for-value with the real-robot
            # controller's log.
            _dv = ", ".join("%.5f" % float(x) for x in down)
            # also print the RAISED pose the blend actually uses (env 0 only;
            # the tensor is (num_envs, 8)), plus the per-env spread so the
            # domain randomisation stays visible.
            _u0 = self._up_arm[0] if self._up_arm.dim() == 2 else self._up_arm
            _uv_s = ", ".join("%.5f" % float(x) for x in _u0.tolist())
            _spread = (float(self._up_arm.std(dim=0).max())
                       if self._up_arm.dim() == 2 and self._up_arm.shape[0] > 1
                       else 0.0)
            print("[box_v38] TS DELTA GATE %s"
                  % ("ACTIVE: waiting delta x%.2f -> full on trigger"
                     % V38_TS_DELTA_MIN if V38_TS_DELTA_GATE else
                     "off (policy delta full authority in hover — v82a gamed "
                     "the height gate this way)"), flush=True)
            print("[box_v38] TWO-STAGE ARMS raised pose (env0) = [%s] "
                  "| per-env std max %.4f rad | %s"
                  % (_uv_s, _spread,
                     ("SHIFTED by BOX_V38_TS_UP = %s" % V38_TS_UP.strip())
                     if V38_TS_UP.strip() else "cradle default"), flush=True)
            # The cradle thresholds are stored separately in sim (_cradled) and
            # on the robot; print them from the execution path so the two can
            # be cross-checked.
            print("[teacher] cradle thresholds: d<0.50 relv<2.0 z_rel>0.45 "
                  "| launched gate = env._box_launched",
                  flush=True)
            print(f"[teacher] two-stage arms: {len(cols)} shoulder/elbow "
                  f"targets = clip {V38_REF_DOWN_LO:g}-{V38_REF_DOWN_HI:g}s down "
                  f"pose while waiting | trigger = {_trig} | down pose = [{_dv}]"
                  f" | base_blend={int(V38_TS_BASE_BLEND)}", flush=True)
        cols_t, down, blend = cache
        if cols_t is None:
            return
        box = env.scene["balloon"]
        up = 1.0 / max(1, V38_TS_RAMP)
        if V38_TS_FROM_PERC:
            # DEPLOYABLE PATH: drive the blend from the filtered box estimate,
            # which the robot also has; no ground-truth launch flag.
            est = getattr(env, "_v31_est", None)
            v = est.get("est_v_w") if isinstance(est, dict) else None
            if v is None:                       # GT eval arm / estimator off
                v = box.data.root_lin_vel_w
            spd = v.norm(dim=-1)
            pw = est.get("est_p_w") if isinstance(est, dict) else None
            if pw is None:
                pw = box.data.root_pos_w
            dist = (pw[:, :2] - env.scene["robot"].data.root_pos_w[:, :2]
                    ).norm(dim=-1)
            # Require the distance to be CLOSING: dist < D_TRIG alone holds for
            # long stretches while the box is still in the thrower's hands or on
            # the ground (hover raise 0.40-0.55 against a 0.15 bar in replays),
            # and a held box does not approach.  Raising V_LO does not help:
            # distance is the dominant term, so the fix has to be on distance.
            dprev = getattr(self, "_v38_ts_dprev", None)
            if dprev is None or dprev.shape != dist.shape:
                dprev = dist.clone()
            closing = (dprev - dist) / max(1e-6, env.step_dt)      # m/s, +ve
            self._v38_ts_dprev = dist.clone()
            fire = (dist < V38_TS_D_TRIG) & (closing > V38_TS_CLOSE)
            _kf = est.get("kf") if isinstance(est, dict) else None
            if V38_TS_MISS_MAX > 0 and _kf is not None and hasattr(_kf, "miss"):
                fire = fire & (_kf.miss <= V38_TS_MISS_MAX)
            if V38_TS_USE_SPEED:
                # Speed path = the "saw the release" trigger (KF speed jumps
                # past V_LO the moment the box leaves the hand, at ANY range).
                # It must carry the same hygiene gates as the distance path:
                # closing>CLOSE kills held/carried boxes (a walking thrower
                # closes at ~1.5 m/s with the box in hand — that is why V_LO
                # must sit above walking speed), and miss<=MISS_MAX kills
                # standby ghost detections with fabricated velocities.
                fire_spd = (spd > V38_TS_V_LO) & (closing > V38_TS_CLOSE)
                if V38_TS_MISS_MAX > 0 and _kf is not None                         and hasattr(_kf, "miss"):
                    fire_spd = fire_spd & (_kf.miss <= V38_TS_MISS_MAX)
                if V38_TS_SPEED_SUSTAIN > 1:
                    sc = getattr(self, "_v38_ts_spd_ctr", None)
                    if sc is None or sc.shape != fire_spd.shape:
                        sc = torch.zeros(fire_spd.shape, dtype=torch.float32,
                                         device=fire_spd.device)
                    sc = torch.where(fire_spd, sc + 1.0, torch.zeros_like(sc))
                    self._v38_ts_spd_ctr = sc
                    fire_spd = sc >= float(V38_TS_SPEED_SUSTAIN)
                fire = fire | fire_spd
            # Diagnostic trace (BOX_V38_TS_TRACE=1): per-tick trigger inputs for
            # envs 0-3 while the box is launched.  Print-only, off by default.
            if os.environ.get("BOX_V38_TS_TRACE", "0") == "1":
                _lau = getattr(env, "_box_launched", None)
                if _lau is not None:
                    _ctr0 = getattr(self, "_v38_ts_ctr", None)
                    for _e in range(min(4, spd.shape[0])):
                        if bool(_lau[_e]):
                            print(f"[ts-trace] e{_e} t{int(env.common_step_counter)} spd {float(spd[_e]):.2f} dist {float(dist[_e]):.2f} "
                                  f"closing {float(closing[_e]):.2f} fire {int(bool(fire[_e]))} "
                                  f"ctr {float(_ctr0[_e]) if _ctr0 is not None else -1:.0f} blend {float(blend[_e]):.2f}"
                                  + (f" miss {int(_kf.miss[_e])}" if _kf is not None and hasattr(_kf, "miss") else ""), flush=True)
            ctr = getattr(self, "_v38_ts_ctr", None)
            if ctr is None or ctr.shape[0] != blend.shape[0]:
                ctr = torch.zeros_like(blend)
                self._v38_ts_ctr = ctr
            ctr.copy_(torch.where(fire, torch.full_like(ctr, V38_TS_LATCH),
                                  (ctr - 1.0).clamp(min=0.0)))
            tgt = (ctr > 0).float()
            # once cradled the arms stay up regardless of what the box is doing
            tgt = torch.maximum(tgt, _cradled(env, box).float())
            dn = 1.0 / max(1, V38_TS_FALL)
            delta = (tgt - blend).clamp(-dn, up)
            blend.copy_((blend + delta).clamp(0.0, 1.0))
        else:
            live = (env._box_launched | _cradled(env, box)).float()
            blend.copy_(torch.clamp(blend + (2.0 * live - 1.0) * up, 0.0, 1.0))
        # publish the blend for evaluation (e.g. whether the arms were up at
        # contact)
        env._v38_ts_blend_pub = blend
        b = blend[:, None]
        # Blend the BASE POSE, not the whole target.  Blending the whole target,
        #     target = b*(cradle + scale*a) + (1-b)*down
        #            = [b*cradle + (1-b)*down] + b*scale*a
        # multiplies the policy's arm action by b: it is dead during hover
        # (b=0), so those outputs drift without gradient pressure, and whatever
        # they drifted to takes full effect the moment b reaches 1 (arm height
        # at contact overshooting the cradle, with losses concentrated at far
        # offsets where arm placement must be precise while authority ramps).
        # Blending the base pose leaves the action at full authority
        # throughout, so there is no hidden state and no ramp-in of control.
        base = b * self._up_arm + (1.0 - b) * down[None, :]
        delta = self._processed_actions[:, cols_t] - self._offset_arm
        if V38_TS_DELTA_GATE:
            # waiting: delta shrunk to TS_DELTA_MIN of itself; trigger restores it
            _g = V38_TS_DELTA_MIN + (1.0 - V38_TS_DELTA_MIN) * b
            delta = delta * _g
        self._processed_actions[:, cols_t] = (
            base + delta if V38_TS_BASE_BLEND
            else b * self._processed_actions[:, cols_t] + (1.0 - b) * down[None, :])

    JointPositionAction.process_actions = _proc
    JointPositionAction._v38_two_stage = True


# ---------------------------------------------------------------------------
# Anchor-position error injection (optional; all knobs 0 by default).
# On hardware the robot's own anchor estimate (odometry) drifts: anchor_z was
# observed sliding ~0.13 m/min, to ~1 m after a long session, and the
# anchor_pos observation (obs[58:61], "how far am I from where I should be")
# read 0.2-0.9 m off at throw start, versus ~0.01 m in simulation.  0.25 m is
# the training termination threshold (bad_anchor_pos_z_only), so the real
# robot spends much of its time in a state that training treats as "episode
# over".  Since the policy has a learned pull BACK to the anchor, starting
# 0.5 m "off" makes a step toward an offset box INCREASE the perceived error,
# so the learned behaviour can retract the foot.
#
# The robot's OWN anchor estimate is what drifts, so this offsets
# robot_anchor_pos_w (not the clip's anchor):
#   BOX_V38_AZ_LO/HI   per-episode z bias ~ U[LO, HI] (m)
#   BOX_V38_AZ_RATE    z drift rate (m/min)
#   BOX_V38_AB_X/Y/Z   constant xyz bias magnitude, sign re-drawn per episode
# A per-episode bias dominates: a ~13 s episode at 0.13 m/min accumulates only
# 0.03 m, two orders below what the robot actually carries.
V38_AB = tuple(float(os.environ.get("BOX_V38_AB_%s" % k, "0"))
               for k in ("X", "Y", "Z"))
V38_AZ_RATE = float(os.environ.get("BOX_V38_AZ_RATE", "0"))     # m/min, signed
V38_AZ_LO = float(os.environ.get("BOX_V38_AZ_LO", "0"))         # m, bias range
V38_AZ_HI = float(os.environ.get("BOX_V38_AZ_HI", "0"))


def patch_anchor_z_drift():
    """Monkey-patch MotionCommand.robot_anchor_pos_w to add the anchor error."""
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
    if getattr(MotionCommand, "_v38_az_patched", False):
        return
    _orig = MotionCommand.robot_anchor_pos_w.fget

    def _drifted(self):
        p = _orig(self)
        env = getattr(self, "_env", None)
        if env is None:
            return p
        bias = getattr(self, "_v38_az_bias", None)
        if bias is None or bias.shape[0] != p.shape[0]:
            bias = torch.empty(p.shape[0], device=p.device).uniform_(
                V38_AZ_LO, V38_AZ_HI)
            self._v38_az_bias = bias
        # fresh draw at every episode start, like walking up to the robot at a
        # random point in a long session
        fresh = env.episode_length_buf == 0
        if bool(fresh.any()):
            bias = torch.where(
                fresh, torch.empty_like(bias).uniform_(V38_AZ_LO, V38_AZ_HI),
                bias)
            self._v38_az_bias = bias
        t = env.episode_length_buf.float() * env.step_dt
        p = p.clone()
        p[:, 2] = p[:, 2] + bias + (V38_AZ_RATE / 60.0) * t
        # three-axis constant bias, sign re-drawn per episode
        if any(a != 0.0 for a in V38_AB):
            sgn = getattr(self, "_v38_ab_sgn", None)
            if sgn is None or sgn.shape[0] != p.shape[0]:
                sgn = torch.where(torch.rand(p.shape[0], 3, device=p.device)
                                  < 0.5, -1.0, 1.0)
                self._v38_ab_sgn = sgn
            if bool(fresh.any()):
                new = torch.where(torch.rand_like(sgn) < 0.5, -1.0, 1.0)
                sgn = torch.where(fresh[:, None], new, sgn)
                self._v38_ab_sgn = sgn
            amp = torch.tensor(V38_AB, device=p.device)
            p = p + sgn * amp[None, :]
        return p

    MotionCommand.robot_anchor_pos_w = property(_drifted)
    MotionCommand._v38_az_patched = True
    print("[box_v38] ANCHOR ERROR injected: z bias U[%.2f, %.2f] + %.3f m/min,"
          " xyz constant +-(%.2f, %.2f, %.2f) m  (real robot carries 0.2-0.9 m;"
          " twin 0.007; training termination threshold 0.25)"
          % (V38_AZ_LO, V38_AZ_HI, V38_AZ_RATE, *V38_AB), flush=True)


# ---------------------------------------------------------------------------
# Joint-limit target clamp (optional, BOX_V38_TGT_CLIP=1).
# The real robot clamps the position target to the URDF limits every tick
# (end-stop protection), whereas the training action term uses `clip: null`.
# On hardware 34-40% of flight-window ticks had at least one joint clamped,
# waist pitch most often (limit +-0.52 rad vs action scale 0.439, so any
# |a| > 1.2 hits the end stop and +-3 sigma reaches 2.5x the limit).  This
# knob applies the same clamp in simulation so training and deployment match.
V38_TGT_CLIP = os.environ.get("BOX_V38_TGT_CLIP", "0") == "1"


def patch_target_clip():
    """Clamp the position target to the robot's joint limits, as the real robot does.
    Must be applied AFTER patch_arm_two_stage so the clamp sees the final
    target -- same order as on the robot."""
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
    if getattr(JointPositionAction, "_v38_tgt_clip", False):
        return
    _prev = JointPositionAction.process_actions

    def _clipped(self, actions):
        _prev(self, actions)
        lim = getattr(self, "_v38_lim", None)
        if lim is None:
            # Depending on the Isaac Lab version, ArticulationData exposes
            # `joint_pos_limits` or only `joint_limits`.  Same fallback chain
            # as gimbal_action.py, which lands on the HARD limits (not the
            # 0.9-scaled soft ones), matching the real robot.
            d = self._asset.data
            jl = getattr(d, "joint_pos_limits", None)
            if jl is None:
                jl = getattr(d, "joint_limits", None)
            if jl is None:
                jl = d.soft_joint_pos_limits
            ids = self._joint_ids
            jl = jl if isinstance(ids, slice) else jl[:, ids]
            lim = (jl[..., 0], jl[..., 1])
            self._v38_lim = lim
            print("[box_v38] TARGET CLIP on: %d joints clamped to URDF limits "
                  "(matches deploy np.clip at onboard:1074)"
                  % lim[0].shape[-1], flush=True)
        lo, hi = lim
        self._processed_actions = self._processed_actions.clamp(min=lo, max=hi)

    JointPositionAction.process_actions = _clipped
    JointPositionAction._v38_tgt_clip = True


# ---------------------------------------------------------------------------
# Leg torque headroom (evaluation probe, BOX_LEG_EFFORT_SCALE > 0).
# Motor friction, backlash and thermal derating all show up as LESS USABLE
# TORQUE than the ideal actuator the simulator assumes.  Scaling the effort
# limit leaves the control law untouched and only caps the torque the
# actuator can deliver; this is cleaner than lowering kp (BOX_LEG_KP_SCALE),
# which makes the robot fall for reasons unrelated to stepping.
V38_LEG_EFFORT = float(os.environ.get("BOX_LEG_EFFORT_SCALE", "0"))


def apply_leg_effort_scale(scene_robot_cfg):
    """Scale effort_limit_sim on legs+feet.  Called from the env cfg."""
    if V38_LEG_EFFORT <= 0.0:
        return
    hit = {}
    for grp in ("legs", "feet"):
        act = scene_robot_cfg.actuators.get(grp)
        if act is None:
            continue
        el = getattr(act, "effort_limit_sim", None)
        if isinstance(el, dict):
            for k in list(el.keys()):
                el[k] = el[k] * V38_LEG_EFFORT
                hit[k] = round(float(el[k]), 1)
        elif el is not None:
            act.effort_limit_sim = el * V38_LEG_EFFORT
            hit[grp] = round(float(act.effort_limit_sim), 1)
    print("[box_v38] LEG EFFORT SCALE x%.2f -> %s  (models the torque the real "
          "motors actually deliver: friction/backlash/derating)"
          % (V38_LEG_EFFORT, hit), flush=True)
