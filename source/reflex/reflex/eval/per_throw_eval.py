r"""Per-throw catch evaluation (the paper's CLEAN2 benchmark protocol).

Runs ``--num_envs`` parallel environments for ``--episode_steps`` control steps (50 Hz) and scores
every throw separately, keyed on the rising edge of ``env._box_launched``.  With the settings used by
``scripts/eval.sh`` (static thrower at 3.0-3.5 m, 640 steps, motion phase pinned to the standing part
of the reference clip, no push / re-throw / curriculum events) each environment observes exactly one
throw.

Criteria scored per throw (each must hold for ``--persist_frames`` consecutive steps unless noted):

  strict (chest hold)  catch success.  The box is in the chest pocket (torso-frame box in front of the
                       chest, at chest height, centred), both wrists flank it, head and shoulders are
                       clear of it (rejects a box resting on the head or a shoulder) and it is slow
                       (|v| < 0.5 m/s).  The geometry comes from
                       ``reflex.envs.base.chest_v5_terms.compute_strict_components``, shared with the
                       training reward so the two cannot drift apart.
  size-aware strict    (v2) geometry-agnostic hold used for the box-size sweep: the same tests, but the
                       pocket height is checked at the box BOTTOM instead of its centre, so tall and
                       flat boxes are judged alike.
  intercept            interception: box within r + 0.45 m of the torso, > 0.30 m above the ground and
                       slower than 1.0 m/s for >= 5 steps.
  twin                 the MuJoCo sim-to-sim twin's criterion: box within 0.65 m of the pelvis,
                       |v| < 1.2 m/s and above pelvis - 0.25 m.
  loose                the training reward's hold test (d_eef < 0.30 m or d_chest < 0.35 m, slow); the
                       loose-vs-strict gap is the false-positive rate of the loose test.
  dropped after        caught (strict, twin or size-aware) and later on the floor (box z < 0.35 m).

Per throw the script also logs base displacement (full cycle / flight window / first 0.6 s), falls
(base z < 0.45 m), minimum base height, maximum torso tilt, the analytic (contact-free) arrival point
of the box and the box size.  Everything is written to the ``--out`` JSON, including ``throw_log``
with one record per throw; summary lines are printed with greppable tags ([PERTHROW], [PERTHROW-V2],
[PERTHROW-DROP], [PERTHROW-FIRST], [PERTHROW-TWIN], [BIN], [COND], [PROTOCOL], ...).

Usage (release launcher; it sets the protocol and condition environment variables)::

    scripts/eval.sh <visual|true_z|teacher> [--condition NAME]

Direct invocation (after sourcing configs/eval/clean2_*.env and a condition file)::

    python -m reflex.eval.per_throw_eval --headless --num_envs 320 --episode_steps 640 \
        --eval_start_step 0 --push_robot 0 --motion_phase 300,860 \
        --env_id Reflex-Latent-G1-v0 --motion_file MOTION.npz --bm_ckpt CKPT.pt \
        [--detector VISUAL_ENCODER.pt] --out result.json
"""
from __future__ import annotations

import argparse
import os
import hashlib
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--bm_ckpt", required=True)
parser.add_argument("--bm_ckpt_center", default=None,
                    help="MoE: second policy used when predicted landing radius "
                         "< --switch_r (latched once per throw at launch). "
                         "Absent = single-policy, byte-identical behaviour.")
parser.add_argument("--switch_r", type=float, default=0.3)
parser.add_argument("--env_id", required=True,
                    help="gym task id: Reflex-Teacher-G1-v0 or Reflex-Latent-G1-v0")
parser.add_argument("--motion_file", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episode_steps", type=int, default=650)
# Benchmark-protocol flags (scripts/eval.sh sets the CLEAN2 values).  Defaults keep the original behaviour.
parser.add_argument("--push_robot", type=int, default=1, help="0 = remove the training push_robot interval event (random base velocity kicks every 1-3 s)")
parser.add_argument("--motion_phase", type=str, default="", help="LO,HI = start every episode at a uniform clip frame in [LO,HI) (policy-independent, no clip-end teleport inside the eval); empty = env default (failure-adaptive sampler over the whole clip)")
parser.add_argument("--eval_start_step", type=int, default=200,
                    help="Start evaluating strict criteria from this step (after motion phase reaches catch window).")
parser.add_argument("--persist_frames", type=int, default=30)
parser.add_argument("--ball_radius", type=float, default=None,
                    help="Pin ball radius; if None, env randomization runs.")
parser.add_argument("--out", required=True)
# Visual encoder in the loop: frames from a torso-mounted TiledCamera -> visual encoder ("detector") -> z -> latent decoder
parser.add_argument("--detector", default=None, help="Stage 3 visual encoder checkpoint (reflex.vision.train_seqdet); replaces the privileged encoder E_phi with the image-based encoder")
parser.add_argument("--det_width", type=int, default=192)
parser.add_argument("--det_height", type=int, default=144)
parser.add_argument("--det_fov_h", type=float, default=53.0)
parser.add_argument("--save_frames_dir", default=None, help="detector mode: save per-step rgb/obs/priv/pos/vel chunks (renderer format) for DAgger aggregation")
parser.add_argument("--det_geo", action="store_true", help="detector mode: ANALYTIC temporal part — per-frame box heatmap (kp/kpm detector) -> (u,v) + local-min depth -> camera-frame 3D point; window logic of box_priv_stateless (freshest visible frame, age, delay 2) and LS velocity (+ w x p ego-rotation term) rebuilt in code; z = frozen E_priv(priv_hat)")
parser.add_argument("--det_geo_k", type=int, default=16, help="frames in the world-frame LS velocity fit (16 = 0.32 s; sigma_v ~ 2.7 x sigma_p)")
parser.add_argument("--det_geo_movfit", type=int, default=0, help="2: model selection by z residual between constant-velocity and free-flight (gravity-compensated) fits, velocity at t=0 without any shift (v4); 0: v2 (speed-gated gravity shift). (1 = moving-frames fit, removed: thrower motion during hover tripped it, z bias -1.1 m/s.) OLD: fit the velocity only on frames where the box is MOVING (>0.7 m/s between consecutive visible frames) when >= 4 such frames exist (flight), so hover frames right after release do not dilute the fit; else plain visible-frame fit without the gravity shift")
parser.add_argument("--det_geo_depth", default="min", choices=["min", "robust"], help="min: min depth in a 15 px window (+det_geo_he). robust: mean depth of the box pixels in a 15x15 px crop around (u,v) (pixels within 0.35 m of the crop minimum) + det_geo_he — steadier than the nearest corner while the box tumbles")
parser.add_argument("--det_geo_thr", type=float, default=0.0, help="heatmap peak-logit threshold for 'box visible in this frame'")
parser.add_argument("--det_geo_he", type=float, default=0.19, help="front-face -> centre depth offset (m); measured bias of the local-min depth readout (box half extent 0.17)")
parser.add_argument("--save_feats", default=None, help="kpseq detector: log per-step 17-D frame features + z* = E_priv(priv) + camera-frame pos/vel labels to this .npz (student data for GRU DAgger; a few MB, no images)")
parser.add_argument("--det_shadow", default=None, help="detector mode: SHADOW — the actor keeps E_priv z, the detector runs alongside; per-step (z_det, z_priv, pos, vis, env, tick) saved to this .npz (closed-loop residual on the eval pipeline)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--spawn_radius", type=float, default=None,
                    help="REAL physical spawn radius — replaces scene.balloon at "
                         "scene-build, so PhysX collision matches visual. For "
                         "honest multi-size physics tests.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = bool(args.detector)   # cameras only for the visual-encoder (detector-in-loop) path
if args.detector:
    import faulthandler, sys as _fh_sys
    faulthandler.dump_traceback_later(240, repeat=True, file=_fh_sys.stderr)   # debug aid: periodic stack dump if the camera path stalls
args.headless = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import sys as _sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import whole_body_tracking.tasks  # noqa: F401, E402


# ---------- Strict criteria: single source of truth ----------
# Reward and eval both take the geometry from `chest_v5_terms.compute_strict_components`,
# so the two cannot drift out of sync.
from reflex.envs.base import chest_v5_terms as mdp_v5

POCKET_X_RANGE         = mdp_v5.POCKET_X_RANGE
POCKET_Z_RANGE         = mdp_v5.POCKET_Z_RANGE
POCKET_Y_ABS_MAX       = mdp_v5.POCKET_Y_ABS_MAX
WRIST_NEAR_MAX         = mdp_v5.WRIST_NEAR_MAX
HEAD_FORBIDDEN_MAX     = mdp_v5.HEAD_FORBIDDEN_MAX
SHOULDER_FORBIDDEN_MAX = mdp_v5.SHOULDER_FORBIDDEN_MAX
V_SLOW_MAX             = mdp_v5.V_SLOW_MAX
BODIES                 = mdp_v5.BODY_NAMES_REQUIRED   # dict short_name → link

# Loose criterion (from the training reward), used to measure false positives.
# Kept local because the chest reward no longer uses these; the eval reports
# them to expose the loose-vs-strict gap.
LOOSE_NEAR_EEF = 0.30
LOOSE_NEAR_CHEST = 0.35
LOOSE_V_SLOW = 0.5  # matches the catch_success_bonus reward term


def _build_actor(ckpt_path: str, num_obs: int, device: str):
    """Load the actor for an env with ``num_obs`` observations.

    Latent checkpoints (they contain the ``e_priv_lin`` encoder) are rebuilt as a
    ``LatentActorCritic``; plain checkpoints are loaded into a 512-256-128 ELU MLP.
    If the checkpoint's first-layer input dimension is smaller than ``num_obs``,
    the missing input columns are zero-padded (same rule as warm-starting).

    ASSERTS that the checkpoint carries no active observation normalizer: this
    eval feeds raw observations, so a normalized actor would output garbage.
    Fail loudly instead.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    if any(k.startswith("e_priv_lin.") for k in state):
        # latent ckpt: rebuild the bottleneck actor-critic and
        # return a callable with the same obs->action contract.  Identity normalizer
        # is asserted by construction (actor_obs_normalization={} in the PPO cfg).
        from reflex.latent.latent_actor_critic import LatentActorCritic
        n_crit = state["critic.0.weight"].shape[1]
        dummy = {"policy": torch.zeros(1, num_obs), "critic": torch.zeros(1, n_crit)}
        ac = LatentActorCritic(dummy, {"policy": ["policy"], "critic": ["critic"]}, 29,
                               actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                               activation="elu", init_noise_std=1.0, noise_std_type="scalar")
        state = {k: v for k, v in state.items() if k != "z_noise_vec"}   # training-only buffer (residual replay)
        ac.load_state_dict(state, strict=True)
        ac = ac.to(device).eval()
        print(f"[_build_actor] LATENT ckpt: e_priv z{ac.zdim}, actor in={ac.actor[0].in_features}, "
              f"num_obs={num_obs}", flush=True)
        class _LatentPolicy:
            def __init__(self, ac): self.ac = ac
            def __call__(self, obs): return self.ac.actor(self.ac.features(obs))
        return _LatentPolicy(ac)
    actor = nn.Sequential(
        nn.Linear(num_obs, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 29),
    )

    # Normalizer assertion: the released PPO configs use an identity observation
    # normalizer (actor_obs_normalization={}).  Detect any active normalizer keys
    # and abort.
    norm_keys = [k for k in state.keys()
                 if "obs_normalizer" in k or "normalizer.mean" in k or "normalizer.std" in k
                 or k.startswith("actor_obs_normalizer.") or k.startswith("critic_obs_normalizer.")]
    # Allow empty / running_mean fields that rsl_rl writes even for Identity (rare).
    active_norm = [k for k in norm_keys if not k.endswith(".num_batches_tracked")
                   and state[k].numel() > 0]
    if active_norm:
        raise RuntimeError(
            f"ckpt {ckpt_path} contains active obs_normalizer keys {active_norm[:5]}. "
            "This eval script assumes Identity normalizer. Either retrain with "
            "actor_obs_normalization={} OR rewrite eval to use runner.get_inference_policy()."
        )

    actor_state = {k.replace("actor.", ""): v for k, v in state.items() if k.startswith("actor.")}

    # First-layer obs-dim mismatch -> zero-pad the new dims (warm-start style).
    w_key = "0.weight"
    if w_key in actor_state:
        old_w = actor_state[w_key]
        new_w_shape = actor[0].weight.shape  # (512, num_obs)
        if old_w.shape != new_w_shape:
            if (old_w.dim() == 2
                and old_w.shape[0] == new_w_shape[0]
                and old_w.shape[1] < new_w_shape[1]):
                padded = torch.zeros(new_w_shape, dtype=old_w.dtype)
                padded[:, :old_w.shape[1]] = old_w
                actor_state[w_key] = padded
                print(f"[_build_actor] zero-padded {w_key}: {tuple(old_w.shape)} → {tuple(new_w_shape)}",
                      flush=True)
            else:
                raise ValueError(
                    f"actor.0.weight shape {tuple(old_w.shape)} cannot map to {tuple(new_w_shape)} via zero-pad")
    actor.load_state_dict(actor_state)
    return actor.to(device).eval()


def _quat_apply_inv(q, v):
    """Rotate v by q^-1, q is wxyz."""
    w, x, y, z = q.unbind(-1)
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y + w * z); r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z); r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y); r21 = 2 * (y * z - w * x); r22 = 1 - 2 * (x * x + y * y)
    dx, dy, dz = v.unbind(-1)
    bx = r00 * dx + r01 * dy + r02 * dz
    by = r10 * dx + r11 * dy + r12 * dz
    bz = r20 * dx + r21 * dy + r22 * dz
    return torch.stack([bx, by, bz], dim=-1)


def make_env(env_id, motion_file, num_envs, ball_radius=None, seed=42, spawn_radius=None):
    if env_id == "Reflex-Teacher-G1-v0":
        import reflex.envs.register  # noqa: F401  (gym id)
        from reflex.envs.teacher_env_cfg import G1BoxCatchStepCatchEnvCfg as Cfg
    elif env_id == "Reflex-Latent-G1-v0":
        import reflex.latent.register  # noqa: F401  (gym id)
        from reflex.latent.latent_env_cfg import G1BoxCatchLatentEnvCfg as Cfg
    else:
        raise ValueError(f"unsupported env_id {env_id} (expected Reflex-Teacher-G1-v0 or Reflex-Latent-G1-v0)")
    env_cfg = Cfg()
    if args.detector and env_id == "Reflex-Latent-G1-v0":
        import math as _m
        import isaaclab.sim as _sim_utils
        _det_in_ch = int(torch.load(args.detector, map_location="cpu", weights_only=False).get("in_ch", 3))
        print(f"[latent] detector input channels: {_det_in_ch} ({'RGB-D' if _det_in_ch == 4 else 'RGB'})", flush=True)
        from isaaclab.sensors import TiledCameraCfg as _TC
        _ap = 20.955; _f = _ap / 2.0 / _m.tan(_m.radians(args.det_fov_h))
        env_cfg.scene.latent_cam = _TC(prim_path="{ENV_REGEX_NS}/latent_cam", height=args.det_height, width=args.det_width,
                                       data_types=["rgb"] + (["distance_to_image_plane"] if _det_in_ch == 4 else []), spawn=_sim_utils.PinholeCameraCfg(focal_length=_f, focus_distance=400.0,
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
        print(f"[latent] DETECTOR-IN-LOOP camera armed: {args.det_width}x{args.det_height} fov_h {args.det_fov_h} deg, spacing 40 m", flush=True)
    env_cfg.scene.num_envs = num_envs
    # G1_URDF_OVERRIDE: swap the robot asset (scripts/eval.sh uses main_hands.urdf, the G1 with
    # box-shaped hand colliders).  Unset = stock asset.  The [asset] line below confirms it.
    _urdf = os.environ.get("G1_URDF_OVERRIDE", "")
    if _urdf:
        env_cfg.scene.robot.spawn.asset_path = _urdf
        print(f"[asset] G1 URDF OVERRIDE active: {_urdf}", flush=True)
    env_cfg.commands.motion.motion_file = motion_file
    env_cfg.sim.device = "cuda:0"
    env_cfg.seed = seed

    # Disable terminations so all envs run full length.
    for tname in ["bad_orientation", "fall", "early_termination"]:
        if hasattr(env_cfg.terminations, tname):
            setattr(env_cfg.terminations, tname, None)
    # Also disable noise sources that bias the eval: keep motion yaw randomization
    # but turn off init pose noise / command phase mixture.
    for ename in ["randomize_init_state", "randomize_cmd_time_steps"]:
        if hasattr(env_cfg.events, ename):
            setattr(env_cfg.events, ename, None)

    # Physical spawn radius (legacy ball option): replaces scene.balloon at scene
    # build so the PhysX collision sphere matches the visual (changing the USD
    # radius at runtime does not update the collider).
    if spawn_radius is not None:
        from whole_body_tracking.tasks.tracking.config.g1.catch_radius_finetune_env_cfg import (
            _make_balloon_cfg_spawn,
        )
        env_cfg.scene.balloon = _make_balloon_cfg_spawn(float(spawn_radius))
        # Also disable any per-env radius randomization that would override.
        for ename in ("randomize_balloon_radius", "randomize_balloon_height"):
            if hasattr(env_cfg.events, ename):
                setattr(env_cfg.events, ename, None)
        print(f"[strict] REPLACED scene.balloon spawn radius = {spawn_radius}m (physics+visual)",
              flush=True)
    if int(args.push_robot) == 0 and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
        print("[protocol] push_robot event REMOVED (no random base kicks during eval)", flush=True)
    return gym.make(env_id, cfg=env_cfg), env_cfg


def main():
    env, env_cfg = make_env(args.env_id, args.motion_file, args.num_envs,
                             args.ball_radius, args.seed, args.spawn_radius)
    if args.motion_phase:
        _lo, _hi = [int(x) for x in args.motion_phase.split(",")]
        _mterm = env.unwrapped.command_manager.get_term("motion")
        _tot = int(_mterm.motion.time_step_total)
        assert 0 <= _lo < _hi <= _tot, (_lo, _hi, _tot)
        def _uniform_phase(env_ids, _t=_mterm, _lo=_lo, _hi=_hi):
            _ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids), device=_t.device)
            _t.time_steps[_ids] = torch.randint(_lo, _hi, (len(_ids),), device=_t.device)
        _mterm._adaptive_sampling = _uniform_phase
        print(f"[protocol] motion start phase pinned U[{_lo},{_hi}) of {_tot} frames (uniform, policy-independent; "
              f"clip end not reached within {args.episode_steps} steps: {_hi + args.episode_steps <= _tot})", flush=True)
    obs_dict, _ = env.reset(seed=args.seed)
    obs = obs_dict["policy"]
    num_obs = obs.shape[1]
    print(f"[strict] env={args.env_id} num_envs={args.num_envs} num_obs={num_obs}", flush=True)

    actor = _build_actor(args.bm_ckpt, num_obs=num_obs, device="cuda:0")
    actor_c = None
    moe_use_c = None
    if args.bm_ckpt_center:
        actor_c = _build_actor(args.bm_ckpt_center, num_obs=num_obs,
                               device="cuda:0")
        print(f"[moe] LATCHED LANDING-GATED MoE armed: r<{args.switch_r} -> "
              f"center policy {args.bm_ckpt_center}", flush=True)
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    if args.detector:
        # Visual encoder in the loop: every policy call renders the torso camera, pushes the frame into a WIN ring
        # per env, runs the visual encoder ("detector"), and feeds z to the decoder via z_external (E_priv bypassed).
        import math as _dp_m
        from isaaclab.utils.math import quat_apply as _dp_qa, quat_mul as _dp_qm
        from reflex.vision.detector_model import load_detector
        from reflex.envs.robust_estimator import CAM_OFF as _dp_CO, CAM_PITCH_UP as _dp_CP
        _dp_det, _dp_ck = load_detector(args.detector, "cuda:0"); _dp_W = int(_dp_ck["win"])
        _dp_cam = raw_env.scene["latent_cam"]; _dp_ti = list(robot.data.body_names).index("torso_link"); _dp_n = args.num_envs
        _dp_qp = torch.tensor([_dp_m.cos(-_dp_CP / 2), 0.0, _dp_m.sin(-_dp_CP / 2), 0.0], device="cuda:0").expand(_dp_n, 4)
        _dp_noring = hasattr(_dp_det, "seq_forward") or args.det_geo      # these modes consume only the newest frame: no image ring (3.5 GB at 320 envs)
        _dp_C = int(getattr(_dp_det, "in_ch", 3)); _dp_ring = torch.zeros(1 if _dp_noring else _dp_n, _dp_W, _dp_C, args.det_height, args.det_width, device="cuda:0"); _dp_filled = [0]
        _dp_fs = dict(feat=[], z=[], pos=[], vel=[], env=[], ep=[], tick=[], t=0, ep_id=torch.zeros(_dp_n, dtype=torch.long, device="cuda:0"))
        _dp_ac = actor.ac
        _dp_save = [None]
        if args.save_frames_dir:
            import numpy as _np
            os.makedirs(args.save_frames_dir, exist_ok=True)
            _dp_save[0] = dict(rgb=[], obs=[], priv=[], pos=[], vel=[], env=[], ep=[], tick=[], t=0, chunk=0,
                               ep_id=torch.zeros(_dp_n, dtype=torch.long, device="cuda:0"))
            from isaaclab.utils.math import quat_apply_inverse as _qai
            def _dp_flush():
                d = _dp_save[0]
                if not d["rgb"]: return
                _xd = {"depth": _np.concatenate(d["depth"])} if d.get("depth") else {}
                if d.get("ego"): _xd["ego"] = _np.concatenate(d["ego"])
                _np.savez_compressed(os.path.join(args.save_frames_dir, f"chunk_{d['chunk']:04d}.npz"), rgb=_np.concatenate(d["rgb"]), obs=_np.concatenate(d["obs"]), **_xd,
                                     priv=_np.concatenate(d["priv"]), pos=_np.concatenate(d["pos"]), vel=_np.concatenate(d["vel"]), env=_np.concatenate(d["env"]),
                                     ep=_np.concatenate(d["ep"]), tick=_np.concatenate(d["tick"]))
                d["chunk"] += 1
                for k in ("rgb", "obs", "priv", "pos", "vel", "env", "ep", "tick"): d[k] = []
                if "depth" in d: d["depth"] = []
                if "ego" in d: d["ego"] = []
            _dp_save[0]["flush"] = _dp_flush
            from reflex.latent import latent_obs as _dp_lo
            print(f"[latent] DAgger data collection -> {args.save_frames_dir} (aux label frame = {_dp_lo.LAT_PRIV_FRAME})", flush=True)
        from isaaclab.utils.math import quat_apply_inverse as _qai
        import numpy as _np
        _dp_geo = dict(pos=torch.zeros(_dp_n, _dp_W + 12, 3, device="cuda:0"), posw=torch.zeros(_dp_n, _dp_W + 12, 3, device="cuda:0"), vel=torch.zeros(_dp_n, _dp_W + 12, 3, device="cuda:0"),
                       vis=torch.zeros(_dp_n, _dp_W + 12, dtype=torch.bool, device="cuda:0"), tan=_dp_m.tan(_dp_m.radians(args.det_fov_h)))
        if args.det_geo: print(f"[latent] DETECTOR GEO MODE: heatmap (u,v) + local-min depth (+{args.det_geo_he} m) -> 3D, analytic window/velocity, z = frozen E_priv(priv_hat); thr {args.det_geo_thr}", flush=True)
        _dp_fring = [None]
        # SENSOR-REALISM / INPUT-ABLATION knobs for the learned (kpseq) visual encoder, inference only, all default OFF (robustness benchmark):
        #   BOX_DET_NOISE_DEPTH_K  depth noise sigma = K * d^2 [m]      BOX_DET_NOISE_DROP_P  per-env probability that a frame is dropped (previous features repeated)
        #   BOX_DET_NOISE_LAT      extra latency in frames (20 ms each)  BOX_DET_NOISE_EGO     ego-motion noise scale (1 = 0.05 rad/s, 0.05 m/s, 0.01 gravity-dir)
        #   BOX_DET_NO_EGO=1       zero the 9 ego-motion inputs (pair with an encoder trained with reflex.vision.train_seqdet --no_ego)
        _dp_nz = dict(k=float(os.environ.get("BOX_DET_NOISE_DEPTH_K", "0")), drop=float(os.environ.get("BOX_DET_NOISE_DROP_P", "0")), lat=int(os.environ.get("BOX_DET_NOISE_LAT", "0")),
                      ego=float(os.environ.get("BOX_DET_NOISE_EGO", "0")), noego=os.environ.get("BOX_DET_NO_EGO", "0") == "1", q=[], prev=[None])
        if _dp_nz["k"] or _dp_nz["drop"] or _dp_nz["lat"] or _dp_nz["ego"] or _dp_nz["noego"]:
            print(f"[latent] DETECTOR INPUT PERTURBATION ACTIVE: depth sigma = {_dp_nz['k']} * d^2, frame drop p = {_dp_nz['drop']}, extra latency = {_dp_nz['lat']} frames, ego noise x{_dp_nz['ego']}, no_ego = {_dp_nz['noego']}", flush=True)
        _dp_shadow = dict(z_det=[], z_priv=[], pos=[], vis=[], age=[], tick=[], ep0=[], t=0)
        if args.det_shadow: print(f"[latent] DETECTOR SHADOW MODE: actor uses E_priv, detector logged -> {args.det_shadow}", flush=True)
        def _det_policy(obs):
            tq = robot.data.body_quat_w[:, _dp_ti]; tp = robot.data.body_pos_w[:, _dp_ti]
            _dp_cam.set_world_poses(tp + _dp_qa(tq, torch.tensor(_dp_CO, device="cuda:0").expand(_dp_n, 3)), _dp_qm(tq, _dp_qp), convention="world")
            _dp_cam.update(raw_env.step_dt, force_recompute=True)
            fr = _dp_cam.data.output["rgb"][..., :3].permute(0, 3, 1, 2).float() / 255.0
            if _dp_C == 4:
                from reflex.vision.detector_model import depth_to_channel as _d2c
                _dp_d = _dp_cam.data.output["distance_to_image_plane"]; _dp_d = _dp_d[..., 0] if _dp_d.dim() == 4 else _dp_d
                if _dp_nz["k"] > 0.0: _dp_d = torch.nan_to_num(_dp_d, nan=20.0, posinf=20.0); _dp_d = (_dp_d + torch.randn_like(_dp_d) * _dp_nz["k"] * _dp_d.clamp(max=8.0) ** 2).clamp(min=0.05)
                fr = torch.cat([fr, _d2c(torch.nan_to_num(_dp_d, nan=20.0, posinf=20.0))[:, None]], 1)
            if not _dp_noring:
                _dp_ring[:] = torch.roll(_dp_ring, -1, dims=1); _dp_ring[:, -1] = fr
                if _dp_filled[0] < _dp_W:
                    _dp_ring[:, : _dp_W - _dp_filled[0] - 1] = fr[:, None]; _dp_filled[0] += 1
            with torch.no_grad():
                if args.det_geo:
                    import torch.nn.functional as _F
                    G = _dp_geo; Hi, Wi = fr.shape[-2:]
                    fm = _dp_det.cnn(fr); hl = _dp_det.heat(fm)[:, 0]; _b, Hh, Wh = hl.shape
                    ha = torch.softmax(hl.flatten(1), -1).view(_b, Hh, Wh)
                    hu = (ha.sum(1) * torch.linspace(-1, 1, Wh, device="cuda:0")).sum(-1); hv = (ha.sum(2) * torch.linspace(-1, 1, Hh, device="cuda:0")).sum(-1)
                    pk = hl.flatten(1).amax(-1); gg = torch.stack([hu, hv], -1)[:, None, None, :]
                    d15 = _F.grid_sample(-_F.max_pool2d(-fr[:, 3:4], 15, 1, 7), gg, mode="nearest", align_corners=True)[:, 0, 0, 0] * 8.0
                    if args.det_geo_depth == "robust":
                        _o = torch.arange(-7, 8, device="cuda:0").float(); _gy, _gx = torch.meshgrid(_o, _o, indexing="ij")
                        cg = torch.stack([hu[:, None, None] + _gx[None] * (2.0 / (Wi - 1)), hv[:, None, None] + _gy[None] * (2.0 / (Hi - 1))], -1)   # (n,15,15,2)
                        crop = _F.grid_sample(fr[:, 3:4], cg, mode="nearest", padding_mode="border", align_corners=True)[:, 0] * 8.0       # metres
                        cmin = crop.flatten(1).amin(-1); msk = (crop < cmin[:, None, None] + 0.35).float()
                        d15 = (crop * msk).sum((1, 2)) / msk.sum((1, 2)).clamp(min=1.0)
                    X = d15 + args.det_geo_he; Y = -hu * G["tan"] * X; Z = -hv * G["tan"] * (Hi / Wi) * X
                    pnow = torch.stack([X, Y, Z], -1); visf = (pk > args.det_geo_thr) & (d15 < 7.5)
                    cq_ = _dp_qm(tq, _dp_qp); cpw = tp + _dp_qa(tq, torch.tensor(_dp_CO, device="cuda:0").expand(_dp_n, 3))
                    pw = cpw + _dp_qa(cq_, pnow)                                          # box point in the WORLD frame (camera pose = robot proprioception)
                    rs = raw_env.episode_length_buf == 0
                    if bool(rs.any()): G["vis"][rs] = False; G["vel"][rs] = 0.0
                    # spurious-detection gate: a visible point that jumps > 0.6 m from the previous visible point is dropped
                    jump = G["vis"][:, 0] & ((pw - G["posw"][:, 0]).norm(dim=-1) > 0.6); visf = visf & ~jump
                    G["pos"] = torch.roll(G["pos"], 1, 1); G["posw"] = torch.roll(G["posw"], 1, 1); G["vis"] = torch.roll(G["vis"], 1, 1)
                    G["pos"][:, 0] = pnow; G["posw"][:, 0] = pw; G["vis"][:, 0] = visf
                    # world-frame constant-velocity LS over the newest K frames (velocity at the weighted mean time tm <= 0),
                    # moved to t = 0 with gravity when the box is in flight (|v_fit| > 1 m/s): v0 = v_fit + a * (0 - tm)
                    K = args.det_geo_k; tt = -torch.arange(K, device="cuda:0").float() * raw_env.step_dt; w_ = G["vis"][:, :K].float()
                    sw = w_.sum(1).clamp(min=1.0); tm = (w_ * tt).sum(1) / sw; den = (w_ * (tt[None, :] - tm[:, None]) ** 2).sum(1).clamp(min=1e-6)
                    def _lin(P):   # weighted LS line per env/axis: slope (= velocity) and residual sum of squares
                        pm = (w_[..., None] * P).sum(1) / sw[:, None]
                        sl = (w_[..., None] * (tt[None, :, None] - tm[:, None, None]) * (P - pm[:, None, :])).sum(1) / den[:, None]
                        res = P - pm[:, None, :] - sl[:, None, :] * (tt[None, :, None] - tm[:, None, None])
                        return sl, (w_[..., None] * res ** 2).sum(1)
                    P0 = G["posw"][:, :K]
                    if args.det_geo_movfit == 2:
                        # model selection: M0 = constant velocity; M1 = free flight, z + 0.5 g t^2 is linear and its slope IS v_z(t=0).
                        P1 = P0.clone(); P1[..., 2] = P1[..., 2] + 4.905 * tt[None, :] ** 2
                        s0, r0 = _lin(P0); s1, r1 = _lin(P1); flight = (r1[:, 2] < r0[:, 2]) & (w_.sum(1) >= 6)
                        v0w = torch.where(flight[:, None], s1, s0)
                    else:
                        vfit, _ = _lin(P0); fly = (vfit.norm(dim=-1) > 1.0).float()[:, None]
                        v0w = vfit + fly * torch.tensor([0.0, 0.0, -9.81], device="cuda:0") * (-tm)[:, None]
                    vnow = _qai(cq_, v0w - robot.data.body_lin_vel_w[:, _dp_ti]); okv = (w_.sum(1) >= 5) & visf & (vnow.norm(dim=-1) < 12.0)
                    vprev = G["vel"][:, 0].clone(); G["vel"] = torch.roll(G["vel"], 1, 1); G["vel"][:, 0] = torch.where(okv[:, None], vnow, vprev)
                    kd = 2; vw = G["vis"][:, kd:kd + _dp_W]; anyv = vw.any(1); first = torch.argmax(vw.long(), 1); sel = (first + kd)[:, None, None].expand(-1, 1, 3)
                    gp = torch.gather(G["pos"], 1, sel)[:, 0]; gv = torch.gather(G["vel"], 1, sel)[:, 0]; mm = anyv.float()[:, None]
                    age_ = torch.where(anyv, first.float() / float(_dp_W), torch.ones_like(hu))
                    priv_hat = torch.cat([gp * mm, gv * mm, age_[:, None], mm, obs[:, 178:181]], -1)
                    z = _dp_ac.encode(priv_hat); G["last_priv_hat"] = priv_hat
                elif hasattr(_dp_det, "seq_forward"):
                    # arch kpseq (LEARNED end-to-end visual encoder): newest frame -> geometric layer (8) + camera ego-motion (9, IMU-like),
                    # feature ring of WIN frames -> GRU -> z.  No hand-written temporal logic here.
                    cq_ = _dp_qm(tq, _dp_qp)
                    ego = torch.cat([_qai(cq_, robot.data.body_ang_vel_w[:, _dp_ti]), _qai(cq_, robot.data.body_lin_vel_w[:, _dp_ti]),
                                     _qai(cq_, torch.tensor([0.0, 0.0, -1.0], device="cuda:0").expand(_dp_n, 3))], -1)
                    if _dp_nz["ego"] > 0.0: ego = ego + torch.randn_like(ego) * _dp_nz["ego"] * torch.tensor([0.05] * 3 + [0.05] * 3 + [0.01] * 3, device="cuda:0")
                    if _dp_nz["noego"]: ego = torch.zeros_like(ego)
                    fnow = torch.cat([_dp_det.frame_feats(fr), ego], -1)
                    if _dp_nz["drop"] > 0.0 and _dp_nz["prev"][0] is not None:
                        _dr = torch.rand(_dp_n, device="cuda:0") < _dp_nz["drop"]; fnow = torch.where(_dr[:, None], _dp_nz["prev"][0], fnow)
                    _dp_nz["prev"][0] = fnow
                    if _dp_nz["lat"] > 0:
                        _dp_nz["q"].append(fnow); fnow = _dp_nz["q"][0]
                        if len(_dp_nz["q"]) > _dp_nz["lat"]: _dp_nz["q"].pop(0)
                    if _dp_fring[0] is None: _dp_fring[0] = fnow[:, None, :].repeat(1, _dp_W, 1)
                    rs = raw_env.episode_length_buf == 0
                    if bool(rs.any()): _dp_fring[0][rs] = fnow[rs][:, None, :]
                    _dp_fring[0] = torch.roll(_dp_fring[0], -1, 1); _dp_fring[0][:, -1] = fnow
                    z = _dp_det.seq_forward(_dp_fring[0])[0]
                    if args.save_feats:
                        q = _dp_fs; box_ = raw_env.scene["balloon"]; q["ep_id"] += (raw_env.episode_length_buf == 0).long()
                        _cp = tp + _dp_qa(tq, torch.tensor(_dp_CO, device="cuda:0").expand(_dp_n, 3))
                        q["feat"].append(fnow.cpu().numpy().astype("float32")); q.setdefault("priv", []).append(obs[:, _dp_ac.plo:_dp_ac.phi].cpu().numpy().astype("float32")); q["z"].append(_dp_ac.encode(obs[:, _dp_ac.plo:_dp_ac.phi]).cpu().numpy().astype("float32"))
                        q["pos"].append(_qai(cq_, box_.data.root_pos_w - _cp).cpu().numpy().astype("float32"))
                        q["vel"].append(_qai(cq_, box_.data.root_lin_vel_w - robot.data.body_lin_vel_w[:, _dp_ti]).cpu().numpy().astype("float32"))
                        q["env"].append(_np.arange(_dp_n, dtype=_np.int32)); q["ep"].append(q["ep_id"].cpu().numpy().astype(_np.int32)); q["tick"].append(_np.full(_dp_n, q["t"], dtype=_np.int32)); q["t"] += 1
                        if q["t"] % 100 == 0:
                            _np.savez(args.save_feats, **{k: _np.concatenate(v) for k, v in q.items() if isinstance(v, list)})
                else:
                    z = _dp_det(_dp_ring)[0]
            if _dp_save[0] is not None:
                d = _dp_save[0]; box_ = raw_env.scene["balloon"]
                d["ep_id"] += (raw_env.episode_length_buf == 0).long()   # new episode -> new sequence id
                d["rgb"].append(_dp_cam.data.output["rgb"][..., :3].cpu().numpy().astype("uint8")); d["obs"].append(obs.cpu().numpy().astype("float16"))
                _cqs = _dp_qm(tq, _dp_qp)   # camera ego-motion (w, v, gravity dir in the camera frame), same 9-D convention as reflex.vision.render_frames
                d.setdefault("ego", []).append(torch.cat([_qai(_cqs, robot.data.body_ang_vel_w[:, _dp_ti]), _qai(_cqs, robot.data.body_lin_vel_w[:, _dp_ti]),
                                                          _qai(_cqs, torch.tensor([0.0, 0.0, -1.0], device="cuda:0").expand(_dp_n, 3))], -1).cpu().numpy().astype("float32"))
                if _dp_C == 4:   # RGB-D student frames: depth as uint16 mm, same convention as reflex.vision.render_frames --depth
                    d.setdefault("depth", []).append((torch.nan_to_num(_dp_d, nan=20.0, posinf=20.0).clamp(0, 20.0) * 1000.0).round().to(torch.int32).cpu().numpy().astype("uint16"))
                d["priv"].append(obs[:, _dp_ac.plo:_dp_ac.phi].cpu().numpy().astype("float32"))   # the policy's own privileged slice (26-D with PRIV_RICH)
                if _dp_lo.LAT_PRIV_FRAME == "camera":   # aux labels in the camera frame (pos in the camera frame, vel relative to the camera), like the renderer
                    _cp = tp + _dp_qa(tq, torch.tensor(_dp_CO, device="cuda:0").expand(_dp_n, 3)); _cq = _dp_qm(tq, _dp_qp)
                    d["pos"].append(_qai(_cq, box_.data.root_pos_w - _cp).cpu().numpy().astype("float32"))
                    d["vel"].append(_qai(_cq, box_.data.root_lin_vel_w - robot.data.body_lin_vel_w[:, _dp_ti]).cpu().numpy().astype("float32"))
                else:
                    d["pos"].append(_qai(robot.data.root_quat_w, box_.data.root_pos_w - robot.data.root_pos_w).cpu().numpy().astype("float32"))
                    d["vel"].append(_qai(robot.data.root_quat_w, box_.data.root_lin_vel_w).cpu().numpy().astype("float32"))
                d["env"].append(_np.arange(_dp_n, dtype=_np.int32)); d["ep"].append(d["ep_id"].cpu().numpy().astype(_np.int32)); d["tick"].append(_np.full(_dp_n, d["t"], dtype=_np.int32))
                d["t"] += 1
                if d["t"] % 100 == 0: d["flush"]()
            if args.det_shadow:
                with torch.no_grad():
                    zp = _dp_ac.encode(obs[:, _dp_ac.plo:_dp_ac.phi])
                d = _dp_shadow; box_ = raw_env.scene["balloon"]
                d["z_det"].append(z.cpu().numpy().astype("float32")); d["z_priv"].append(zp.cpu().numpy().astype("float32"))
                d["pos"].append(_qai(robot.data.root_quat_w, box_.data.root_pos_w - robot.data.root_pos_w).cpu().numpy().astype("float32"))
                d["vis"].append(obs[:, 177].cpu().numpy().astype("float32")); d["age"].append(obs[:, 176].cpu().numpy().astype("float32"))
                if args.det_geo:
                    d.setdefault("priv_hat", []).append(_dp_geo["last_priv_hat"].cpu().numpy().astype("float32")); d.setdefault("priv", []).append(obs[:, 170:181].cpu().numpy().astype("float32"))
                d["tick"].append(_np.full(_dp_n, d["t"], dtype=_np.int32)); d["ep0"].append((raw_env.episode_length_buf == 0).cpu().numpy()); d["t"] += 1
                if d["t"] % 200 == 0:
                    _np.savez_compressed(args.det_shadow, **{k: _np.concatenate(v) for k, v in d.items() if isinstance(v, list)})
                out = _dp_ac.actor(_dp_ac.features(obs))          # E_priv drives; z_external stays None
                return out
            _dp_ac.z_external = z
            out = _dp_ac.actor(_dp_ac.features(obs)); _dp_ac.z_external = None
            return out
        actor = _det_policy
        print(f"[latent] DETECTOR-IN-LOOP ACTIVE: {args.detector} win={_dp_W} (E_priv bypassed; z from images)", flush=True)
        print(f"[COND] perception = visual encoder in the loop {args.detector}", flush=True)
    balloon = raw_env.scene["balloon"]
    # [SIZEAWARE] Under BOX_V38_MASS_DR the startup mass-DR event caches the box identity with the old two-model mass
    # threshold (small / mini), so a single real_* box gets the 34x26x22 "small" radius + half-extents -- which the latent
    # privileged obs (latent_obs.py: he) and the strict/intercept radii read.  Re-pin from the prim geometry (asserts < 1 cm).
    _bm = os.environ.get("BOX_MODEL", "")
    if os.environ.get("BOX_V38_MASS_DR", "0") == "1" and _bm not in ("small", "mini", "random", "") and not _bm.startswith("mix"):
        from reflex.envs.throw import BOX_MODELS as _BMS, infer_models_from_geometry as _infer_geo
        _one = next((m for m in _BMS if m["name"] == _bm), None)
        if _one is not None:
            _old = float(raw_env._balloon_radius_per_env[0]) if hasattr(raw_env, "_balloon_radius_per_env") else float("nan")
            raw_env._balloon_radius_per_env, raw_env._box_half_extents, raw_env._box_model_idx = _infer_geo(balloon, (_one,), "cuda:0")
            print(f"[SIZEAWARE] box identity re-pinned after mass DR: {_one['name']} {_one['size']} eff_r {_one['eff_radius']} (was eff_r {_old:.3f})", flush=True)

    # Tell compute_strict_components the actual physical ball radius so the
    # radius-aware thresholds are correct (legacy ball option).
    if args.spawn_radius is not None:
        import torch as _torch
        raw_env._balloon_radius_per_env = _torch.full(
            (args.num_envs,), float(args.spawn_radius), device="cuda:0")
        print(f"[strict] _balloon_radius_per_env set to {args.spawn_radius}m for radius-aware thresholds", flush=True)
    body_names = list(robot.data.body_names)
    print(f"[strict] body_names ({len(body_names)}): {body_names}", flush=True)
    body_ids = {}
    for k, v in BODIES.items():
        if v in body_names:
            body_ids[k] = body_names.index(v)
        else:
            raise ValueError(f"required body {v} not in body_names: {body_names}")
    print(f"[strict] body_ids resolved: {body_ids}", flush=True)

    B = args.num_envs
    device = "cuda:0"

    # Optional pin of ball radius per env (for clean comparison)
    if args.ball_radius is not None:
        raw_env._balloon_radius_per_env = torch.full(
            (B,), float(args.ball_radius), device=device)
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for i in range(B):
            sphere = stage.GetPrimAtPath(f"/World/envs/env_{i}/Balloon/geometry/mesh")
            if sphere and sphere.IsValid():
                attr = sphere.GetAttribute("radius")
                if attr and attr.IsValid():
                    attr.Set(float(args.ball_radius))

    # Persistence counters (per env)
    strict_persist = torch.zeros(B, dtype=torch.long, device=device)
    loose_persist  = torch.zeros(B, dtype=torch.long, device=device)
    strict_max     = torch.zeros(B, dtype=torch.long, device=device)
    loose_max      = torch.zeros(B, dtype=torch.long, device=device)
    _ARM_GAIN = float(os.environ.get('EVAL_ARM_GAIN', '1.0'))
    _arm_act_idx = None
    if _ARM_GAIN != 1.0:
        _jn = list(raw_env.scene["robot"].data.joint_names)
        _arm_act_idx = torch.tensor(
            [i for i, n in enumerate(_jn)
             if any(k in n.lower() for k in ("shoulder", "elbow", "wrist"))],
            device=device)
        print(f"[armgain] arm action gain {_ARM_GAIN} on {len(_arm_act_idx)} joints",
              flush=True)
    _ACT_DELAY = int(os.environ.get('EVAL_ACT_DELAY', '0'))
    _act_buf = []
    strict_pass    = torch.zeros(B, dtype=torch.bool, device=device)
    loose_pass     = torch.zeros(B, dtype=torch.bool, device=device)

    # Record one snapshot per env at the moment of strict_max for forensics
    snapshots = [None] * B
    # ---- PER-THROW accounting -------------------------------------------
    # strict_pass above is a LATCHING per-episode counter ("did this env ever
    # catch during the episode"): with n throws per episode it measures
    # 1-(1-p)^n, saturates near the top, and moves with the throw COUNT.  Envs
    # that start with the box already cradled (hold-init) would also satisfy
    # `strict` trivially.  Here every throw is scored separately, keyed on the
    # rising edge of env._box_launched, so hold-init episodes contribute NO
    # throw until they are actually thrown to.
    throw_n      = torch.zeros(B, dtype=torch.long, device=device)   # throws seen
    throw_lstep  = torch.full((B,), -1, dtype=torch.long, device=device)  # eval step of the launch edge
    throw_mts    = torch.full((B,), -1, dtype=torch.long, device=device)  # motion clip frame at launch
    throw_hit    = torch.zeros(B, dtype=torch.long, device=device)   # throws caught
    # LOOSE per-throw: a pure DISTANCE test (box parked near the hands /
    # chest), while `strict` additionally demands both wrists flanking the box
    # and head/shoulders clear.  Scoring both per throw shows how much of a
    # catch-rate gap is due to the metric rather than the policy.
    throw_hit_l  = torch.zeros(B, dtype=torch.long, device=device)
    # TWIN criterion (the MuJoCo sim-to-sim twin's success test): box within 0.65 m of the pelvis,
    # |v| < 1.2 m/s, higher than pelvis - 0.25 m, for persist_frames.  Reported alongside the strict criterion.
    throw_hit_t  = torch.zeros(B, dtype=torch.long, device=device)
    throw_done_t = torch.zeros(B, dtype=torch.bool, device=device)
    twin_persist = torch.zeros(B, dtype=torch.long, device=device)
    strict2_persist = torch.zeros(B, dtype=torch.long, device=device); icpt_persist = torch.zeros(B, dtype=torch.long, device=device)
    dropped_after = torch.zeros(B, dtype=torch.bool, device=device); first_latch = torch.full((B,), -1, dtype=torch.long, device=device); drop_step = torch.full((B,), -1, dtype=torch.long, device=device)   # [DROP] box on the floor (z<0.35 m) after any catch latch
    throw_done2 = torch.zeros(B, dtype=torch.bool, device=device); throw_icpt = torch.zeros(B, dtype=torch.bool, device=device)
    fail_cnt = torch.zeros(B, 6, dtype=torch.long, device=device); twin_cnt = torch.zeros(B, dtype=torch.long, device=device)
    print("[SIZEAWARE] armed: size-aware strict (v2) = strict pocket x/y, pocket z tested on the box BOTTOM (bz - h_vert) in (-0.205, 0.195) [= the centre band for a flat 21 cm box], wrist / head-shoulder / slow tests unchanged; intercept = |v|<1.0 m/s for >=5 frames within r+0.45 m of the torso and >0.30 m above ground", flush=True)
    throw_done_l = torch.zeros(B, dtype=torch.bool, device=device)
    throw_active = torch.zeros(B, dtype=torch.bool, device=device)   # mid-throw
    # Stability card: idle-phase stillness = base xy speed RMS + mean leg/waist
    # joint |vel| over ticks with no active throw and a static box (same measure
    # as the twin's idle warning).
    idle_v2_sum = torch.zeros(B, device=device); idle_lv_sum = torch.zeros(B, device=device); idle_n = torch.zeros(B, device=device)
    _leg_ids = torch.tensor([i for i, n in enumerate(robot.joint_names) if ('hip' in n or 'knee' in n or 'ankle' in n or 'waist' in n)], dtype=torch.long, device=device)
    throw_done   = torch.zeros(B, dtype=torch.bool, device=device)   # scored already
    prev_launch  = None
    # ---- STEP DISPLACEMENT per throw ------------------------------------
    # Catch rate is blind to "does it even try to move": a policy that does not
    # pursue and one that steps but still misses score the same.  So log, for
    # every throw, how far the base travels from where it stood when the box
    # was released.
    throw_base0  = torch.zeros(B, 2, device=device)   # base xy at throw start
    throw_fell   = torch.zeros(B, dtype=torch.bool, device=device)  # base_z dropped
    # `fell` (base_z < 0.45 m) misses a robot that is destabilised but not yet
    # down (large tilt, base sinking), so zero falls is not by itself evidence
    # of stability.  Also record, per throw, the minimum base height and the
    # maximum torso tilt -- the quantities the twin's safety guard checks.
    # Running max/min on GPU: no per-step sync.
    throw_zmin   = torch.full((B,), 9.9, device=device)
    throw_tiltmx = torch.zeros(B, device=device)
    # ---- LANDING RADIUS per throw ---------------------------------------
    # Catch rate must be binned by distance: a single aggregate averages a
    # chest-high throw together with one landing 0.7 m away, which are
    # different tasks.
    # Radius = horizontal distance from the base position AT THROW START to
    # where the ball crosses catch height on the way down (base at throw start,
    # not current, so the number does not move when the robot moves).
    throw_rad    = torch.full((B,), float("nan"), device=device)
    throw_armed  = torch.zeros(B, device=device)   # max TS blend this throw
    # Arm-rise latency = ticks after launch when the two-stage arm blend first
    # reaches 0.5 (nan = never rose this throw).
    throw_rise   = torch.full((B,), float("nan"), device=device)
    # CONTACT-FREE arrival: throw_rad above needs the PHYSICAL box to descend
    # through z=1.0 -- a box caught or touched above that never does, so a
    # large share of throws get rad=None and drop out of the bins (biased
    # toward good catches).  These are solved analytically from
    # the launch state the env publishes (same k=0.85 closed form the sampler
    # uses), so EVERY throw gets a value.  hz/hy = height / lateral offset
    # where the flight crosses the HAND PLANE (0.35 m ahead of the pelvis at
    # launch, along the robot's facing); cf_dx/cf_dy = the z=1.0 descending
    # crossing in the same frame, + = short (thrower side).
    throw_hz     = torch.full((B,), float("nan"), device=device)
    throw_hy     = torch.full((B,), float("nan"), device=device)
    throw_cfdx   = torch.full((B,), float("nan"), device=device)
    throw_cfdy   = torch.full((B,), float("nan"), device=device)
    throw_ld     = torch.full((B,), float("nan"), device=device)   # launch dist thrower->pelvis (m)
    throw_lv     = torch.full((B,), float("nan"), device=device)   # launch speed (m/s)
    HAND_PLANE_X = 0.35
    _CF_TS       = torch.linspace(0.0, 3.0, 601, device=device)   # 5 ms grid
    def _contact_free(_ids):
        """Analytic drag ballistics from the published launch state."""
        _u = env.unwrapped
        _v0 = getattr(_u, "_box_v0_pub", None)
        if _v0 is None:
            return
        _p0 = _u._box_p0_pub[_ids]; _v0 = _v0[_ids]
        _k = 0.85; _g = torch.tensor([0.0, 0.0, -9.81], device=device)
        _vt = (_g / _k)[None, None, :]
        _ts = _CF_TS[None, :, None]
        _P = (_p0[:, None, :] + _vt * _ts
              + (_v0[:, None, :] - _vt[:, 0]) * ((1.0 - torch.exp(-_k * _ts)) / _k))
        _q = robot.data.root_quat_w[_ids]
        _fw = torch.stack([1 - 2 * (_q[:, 2] ** 2 + _q[:, 3] ** 2),
                           2 * (_q[:, 1] * _q[:, 2] + _q[:, 0] * _q[:, 3])], dim=-1)
        _fw = _fw / _fw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        _lf = torch.stack([-_fw[:, 1], _fw[:, 0]], dim=-1)
        _rel = _P[:, :, :2] - robot.data.root_pos_w[_ids, None, :2]
        _al = (_rel * _fw[:, None, :]).sum(-1); _la = (_rel * _lf[:, None, :]).sum(-1)
        _z = _P[:, :, 2] - _u.scene.env_origins[_ids, None, 2]
        _vz = torch.gradient(_z, spacing=(_CF_TS,), dim=1)[0]
        _ar = torch.arange(len(_ids), device=device)
        _hit = _al <= HAND_PLANE_X; _i = _hit.float().argmax(dim=1); _ok = _hit.any(dim=1)
        throw_hz[_ids] = torch.where(_ok, _z[_ar, _i], torch.full_like(_i, float("nan"), dtype=torch.float))
        throw_hy[_ids] = torch.where(_ok, _la[_ar, _i], torch.full_like(_i, float("nan"), dtype=torch.float))
        _ds = (_z <= 1.0) & (_vz < 0); _j = _ds.float().argmax(dim=1); _okj = _ds.any(dim=1)
        throw_cfdx[_ids] = torch.where(_okj, _al[_ar, _j], torch.full_like(_j, float("nan"), dtype=torch.float))
        throw_cfdy[_ids] = torch.where(_okj, _la[_ar, _j], torch.full_like(_j, float("nan"), dtype=torch.float))
        throw_ld[_ids] = (_p0[:, :2] - robot.data.root_pos_w[_ids, :2]).norm(dim=-1)
        throw_lv[_ids] = _v0.norm(dim=-1)
    rad_log      = []     # (radius, caught_strict) for completed throws
    CATCH_H_REF  = 1.0    # m, chest-ish height the ball descends through
    def _nz(_t):
        return None if bool(torch.isnan(_t)) else float(_t)
    throw_disp   = torch.zeros(B, device=device)      # max |xy - base0| so far
    disp_log     = []                                 # completed throws
    # --- FLIGHT-WINDOW displacement -------------------------------------
    # throw_disp above runs launch -> next launch, so it also banks whatever
    # the robot wanders during the NEXT throw's hover.  This second
    # accumulator covers the flight only: launch -> catch-or-landing + 0.3 s
    # (the same window used when analysing hardware logs).
    FLIGHT_TAIL  = 15                                 # steps ~= 0.3 s @50Hz
    LAND_H       = 0.30                               # m, box has landed
    # START-PHASE window: displacement in the first 0.6 s (30 steps) after
    # release.  Perception quality mostly affects the LATE approach, so this
    # window isolates the policy's early reaction to the launch.
    START_STEPS = 30                                   # 0.6 s at 50 Hz
    throw_disp_st = torch.zeros(B, device=device)
    throw_age = torch.zeros(B, device=device)
    throw_disp_fl = torch.zeros(B, device=device)
    throw_end_ctr = torch.full((B,), -1, dtype=torch.long, device=device)
    # --- LEG command vs actual, inside the flight window ----------------
    # Amplitude (max - min over the flight window) of the leg joint targets vs
    # the measured joint positions, to tell "commanded but not executed" from
    # "not commanded".  Load-bearing joints track loosely even when stepping,
    # so this is mainly useful as a sim reference for hardware logs.
    _jn_all = list(raw_env.scene["robot"].data.joint_names)
    _LEGG = {}
    for _g, _key in (("hip_pitch", "_hip_pitch_"), ("knee", "_knee_"),
                     ("ankle_pitch", "_ankle_pitch_"), ("leg_all", None)):
        if _key is None:
            _ids = [i for i, n in enumerate(_jn_all)
                    if any(k in n for k in ("_hip_", "_knee_", "_ankle_"))]
        else:
            _ids = [i for i, n in enumerate(_jn_all) if _key in n]
        if _ids:
            _LEGG[_g] = torch.tensor(_ids, device=device)
    _leg_ids = _LEGG.get("leg_all")
    _NL = 0 if _leg_ids is None else len(_leg_ids)
    _BIG = 1.0e9
    cmd_lo = torch.full((B, _NL), _BIG, device=device)
    cmd_hi = torch.full((B, _NL), -_BIG, device=device)
    act_lo = torch.full((B, _NL), _BIG, device=device)
    act_hi = torch.full((B, _NL), -_BIG, device=device)
    # column of each group's joints WITHIN _leg_ids, so amplitudes can be
    # sliced per group without re-reading the articulation
    _LEGCOL = {}
    if _NL:
        _pos = {int(j): c for c, j in enumerate(_leg_ids.tolist())}
        for _g, _ids in _LEGG.items():
            _LEGCOL[_g] = torch.tensor([_pos[int(j)] for j in _ids.tolist()],
                                       device=device)
    # --- FOOT EXCURSION vs NET ------------------------------------------
    # Max foot excursion vs net foot displacement over the flight window.
    # Base displacement alone cannot tell a "small step" from an "aborted
    # step" (the foot swings out, then is pulled back); net/max separates
    # these two failure modes.
    # --- WHAT THE POLICY ACTUALLY SEES ----------------------------------
    # Box speed as reported by the perception estimator (not the true speed)
    # during the first 0.6 s of flight.  The Kalman filter zeroes velocity on
    # re-acquire, so with sparse measurements the policy can be looking at a
    # box that reads as STATIONARY for much of the flight -- and then it does
    # not chase.
    # The diagnostic instruments (arm-at-contact, blend bins, anchor obs,
    # observed box speed, foot / leg amplitudes) add GPU->CPU syncs and slow
    # the eval considerably.  They are opt-in and default OFF; the headline
    # catch-rate path does not use them.  Enable with BOX_EVAL_DIAG=1.
    _DIAG = os.environ.get("BOX_EVAL_DIAG", "0") == "1"
    obs_spd_log = []
    # --- IS THE ARM ACTUALLY LATE? --------------------------------------
    # Flight takes >= 1 s and the arm needs ~0.5 s to rise from hanging, so
    # the timing budget should close.  Measure it directly: hand height above
    # the pelvis at the moment the box is closest, i.e. when the catch has to
    # happen.
    _hn = list(raw_env.scene["robot"].data.body_names)
    _HAND = [i for i, n in enumerate(_hn) if "wrist_yaw_link" in n] or \
            [i for i, n in enumerate(_hn) if "wrist_roll_link" in n]
    _PELV = [i for i, n in enumerate(_hn) if n == "pelvis"] or [0]
    # --- DID AN ANCHOR INJECTION REACH THE OBSERVATION? ----------------
    # A config being set is not proof the value reached the policy input.
    # obs[58:61] is motion_anchor_pos_b, computed by
    # subtract_frame_transforms(robot_anchor_pos_w, ...); log the distribution
    # of its z component during the start window to verify it in the
    # observation itself (native sim value is ~0).
    # --- ARM BLEND AT CONTACT -------------------------------------------
    # The two-stage arm target is b*cradle + (1-b)*down, and `down` has
    # shoulder yaw (+0.061, +0.557) -- same sign both sides, 0.5 rad apart,
    # i.e. NOT mirror symmetric.  So any b < 1 leaves that skew in the bed:
    # the two forearms sit at the wrong relative angle and the box rolls off a
    # lopsided trough.  Hand HEIGHT cannot see this (height is driven by
    # shoulder pitch, the skew is in yaw), so "arm not late" and "bed wrong"
    # can both be true.  Bin catch rate by b at closest approach.
    _ATERM = None
    for _t in getattr(raw_env, "action_manager", None)._terms.values():
        if hasattr(_t, "_v38_ts_cache") or "joint" in type(_t).__name__.lower():
            _ATERM = _t
    blend_log = []
    _blend_at_best = torch.zeros(B, device=device)
    anchor_obs_log = []
    arm_at_contact = []
    _best_d = torch.full((B,), 1e9, device=device)
    _arm_at_best = torch.zeros(B, device=device)
    # Measuring AT contact is the most optimistic slice: the cradle has to be
    # formed BEFORE the box arrives, so also sample ~0.3 s earlier.  0.3 s at
    # ~3.5 m/s is ~1.05 m of remaining distance.
    D_EARLY = 1.05
    arm_early = []
    _arm_at_early = torch.full((B,), float("nan"), device=device)
    _fb = list(raw_env.scene["robot"].data.body_names)
    _FOOT = [i for i, n in enumerate(_fb) if "ankle_roll_link" in n]
    _NF = len(_FOOT)
    foot0 = torch.zeros(B, _NF, 3, device=device)
    foot_max = torch.zeros(B, _NF, device=device)
    foot_net = torch.zeros(B, _NF, device=device)
    print("[FOOT] tracking %d feet: %s"
          % (_NF, [_fb[i] for i in _FOOT]), flush=True)
    print("[LEG] tracking %d leg joints: %s" % (
        _NL, {k: len(v) for k, v in _LEGG.items()}), flush=True)

    def _leg_amps(_b):
        """(cmd, act) mean max-min amplitude per group for env _b, or None."""
        if not _NL or float(cmd_hi[_b, 0]) < -_BIG / 2:
            return None
        out = {}
        for _g, _cols in _LEGCOL.items():
            out[_g] = (float((cmd_hi[_b, _cols] - cmd_lo[_b, _cols]).mean()),
                       float((act_hi[_b, _cols] - act_lo[_b, _cols]).mean()))
        return out
    # One unified per-throw record (rad_log / disp_log use different filters
    # and cannot be joined).  Fields include radius, outcome per criterion,
    # displacement (full cycle / flight window / start window), fall and
    # stability, box size and the contact-free arrival point.
    throw_log    = []


    with torch.inference_mode():
        for step in range(args.episode_steps):
            action = actor(obs)
            if actor_c is not None:
                if moe_use_c is None:
                    moe_use_c = torch.zeros(obs.shape[0], dtype=torch.bool,
                                            device=obs.device)
                action_c = actor_c(obs)
                action = torch.where(moe_use_c[:, None], action_c, action)
            # --- EXECUTION-SIDE DELAY (EVAL_ACT_DELAY) -------------------
            # The sim applies the policy's joint target instantly; on hardware
            # there is additional ACTUATION latency (policy output -> joint
            # actually moves).  Delaying the applied action by N ticks models
            # it: the robot tracks a target that is N ticks stale.
            # Unset = 0 = unchanged.
            # --- ARM STEADY-STATE GAIN (EVAL_ARM_GAIN) -------------------
            # In a step-response test the hardware elbow settles short of the
            # commanded displacement (about 0.64 of it), while the sim arm
            # reaches it.  A systematic under-reach is centimetres at the hand,
            # right where the arms close on the box.  The action is a residual
            # on the default pose, so scaling it reproduces the shortfall.
            # Unset = 1.0 = unchanged.
            if _ARM_GAIN != 1.0 and _arm_act_idx is not None:
                action = action.clone()
                action[:, _arm_act_idx] *= _ARM_GAIN
            if _ACT_DELAY > 0:
                _act_buf.append(action)
                if len(_act_buf) > _ACT_DELAY:
                    action = _act_buf.pop(0)   # apply the N-tick-old target
            obs_dict, _, _, _, _ = env.step(action)
            obs = obs_dict["policy"]

            if step < args.eval_start_step:
                continue

            # Use shared compute_strict_components — same geometry as training.
            comp = mdp_v5.compute_strict_components(raw_env)
            d_head   = comp["d_head"]
            d_lsh    = comp["d_lsh"]
            d_rsh    = comp["d_rsh"]
            d_lwrist = comp["d_lwrist"]
            d_rwrist = comp["d_rwrist"]
            ball_in_torso = comp["ball_in_torso"]
            speed = comp["speed"]
            strict = comp["strict_all"]
            # comp["persist"] is auto-managed by compute_strict_components.

            # Loose criterion (training-reward style, kept local for comparison)
            bp = balloon.data.root_pos_w
            body_pos = robot.data.body_pos_w
            torso_pos = body_pos[:, body_ids["chest"], :]
            lwrist = body_pos[:, body_ids["lwrist"], :]
            rwrist = body_pos[:, body_ids["rwrist"], :]
            d_chest = (bp - torso_pos).norm(dim=-1)
            d_eef = (bp - 0.5 * (lwrist + rwrist)).norm(dim=-1)
            loose = ((d_eef < LOOSE_NEAR_EEF) | (d_chest < LOOSE_NEAR_CHEST)) & (speed < LOOSE_V_SLOW)
            _pel = robot.data.root_pos_w
            twin_ok = ((bp - _pel).norm(dim=-1) < 0.65) & (speed < 1.2) & (bp[:, 2] > _pel[:, 2] - 0.25)
            twin_persist = torch.where(twin_ok, twin_persist + 1, torch.zeros_like(twin_persist))
            # ---- [SIZEAWARE] size-aware strict (v2) and intercept ----
            _tq = robot.data.body_quat_w[:, body_ids["chest"], :]; _bq = balloon.data.root_quat_w
            _he = raw_env._box_half_extents
            _hv = torch.zeros(B, device=device)
            for _i in range(3):
                _ax = torch.zeros(B, 3, device=device); _ax[:, _i] = 1.0
                _axt = mdp_v5._quat_apply_inv_vec(_tq, mdp_v5._quat_apply_vec(_bq, _ax))
                _hv = _hv + _axt[:, 2].abs() * _he[:, _i]
            _rb = getattr(raw_env, "_balloon_radius_per_env", None)
            _r = _rb.to(device).clamp_min(0.05) if isinstance(_rb, torch.Tensor) and _rb.shape[0] == B else torch.full((B,), 0.1016, device=device)
            _bx, _by, _bz = ball_in_torso.unbind(-1)
            _px = (_bx > _r) & (_bx < _r + 0.30); _py = _by.abs() < 0.25; _bot = _bz - _hv; _pz = (_bot > -0.205) & (_bot < 0.195)
            _wr = comp["wrist_pass"]; _hs = ~comp["head_shoulder_bad"]; _sl = comp["slow"]
            strict2 = _px & _py & _pz & _wr & _hs & _sl
            strict2_persist = torch.where(strict2, torch.clamp(strict2_persist + 1, max=100), torch.zeros_like(strict2_persist))
            _near = ((bp - torso_pos).norm(dim=-1) < _r + 0.45) & (bp[:, 2] > 0.30)
            icpt_persist = torch.where(_near & (speed < 1.0), icpt_persist + 1, torch.zeros_like(icpt_persist))

            strict_persist = comp["persist"]
            loose_persist  = torch.where(loose, loose_persist + 1,
                                          torch.zeros_like(loose_persist))
            strict_max = torch.maximum(strict_max, strict_persist)
            loose_max  = torch.maximum(loose_max,  loose_persist)

            # Mark pass if persistence reached threshold this step
            newly_strict = (strict_persist >= args.persist_frames) & (~strict_pass)
            strict_pass |= (strict_persist >= args.persist_frames)
            loose_pass  |= (loose_persist  >= args.persist_frames)

            # ---- PER-THROW scoring ------------------------------------------
            launched = getattr(raw_env, "_box_launched", None)
            if launched is not None:
                # NOTE: `rising` must exist on EVERY tick — the displacement
                # block below reads it, and on the very first tick we take the
                # `prev_launch is None` branch.  Leaving it undefined there
                # raised NameError, which Isaac swallows into a silent hang
                # (no traceback in the log, process alive at 100% CPU).
                rising = torch.zeros(B, dtype=torch.bool, device=device)
                if prev_launch is None:
                    # First observed tick: adopt current state WITHOUT scoring,
                    # so hold-init envs (already launched at reset) are not
                    # counted as a throw they never received.
                    prev_launch = launched.clone()
                else:
                    rising = launched & (~prev_launch)
                    if rising.any():
                        # A new throw starts: close out the previous one as a
                        # miss if it was never scored a hit.
                        _prev_lstep, _prev_mts = throw_lstep.clone(), throw_mts.clone()
                        throw_n      = throw_n + rising.long()
                        throw_lstep  = torch.where(rising, torch.full_like(throw_lstep, int(step)), throw_lstep)
                        throw_mts    = torch.where(rising, raw_env.command_manager.get_term("motion").time_steps.long(), throw_mts)
                        throw_active = torch.where(rising,
                                                   torch.ones_like(throw_active),
                                                   throw_active)
                        if actor_c is not None and rising.any():
                            from reflex.envs.teacher_terms \
                                import balloon_landing_pelvis_frame as _blp
                            _land = _blp(env.unwrapped)
                            _rpred = _land[:, :2].norm(dim=-1)
                            moe_use_c = torch.where(
                                rising, _rpred < float(args.switch_r),
                                moe_use_c if moe_use_c is not None
                                else torch.zeros_like(rising))
                        # radius + outcome MUST be banked before throw_done
                        # is cleared for the new throw — reading it after the
                        # reset scored every throw as a miss (all bins 0%).
                        for _b in rising.nonzero(as_tuple=False).flatten().tolist():
                            if (int(throw_n[_b]) > 1
                                    and not bool(torch.isnan(throw_rad[_b]))):
                                rad_log.append((float(throw_rad[_b]),
                                                bool(throw_done[_b])))
                            # unified record — banked HERE, before throw_done
                            # is cleared, same reason as rad_log above.
                            if int(throw_n[_b]) > 1:
                                throw_log.append(dict(
                                    env=int(_b), idx=int(throw_n[_b]) - 1,   # env + 1-based throw index (idx>1 = automatic re-throw after a landed box)
                                    l_step=int(_prev_lstep[_b]), mot_ts=int(_prev_mts[_b]),
                                    rad=(None if bool(torch.isnan(throw_rad[_b]))
                                         else float(throw_rad[_b])),
                                    caught=bool(throw_done[_b]), caught_twin=bool(throw_done_t[_b]), caught_v2=bool(throw_done2[_b]), intercepted=bool(throw_icpt[_b]), v2_fail=[int(_x) for _x in fail_cnt[_b].tolist()], v2_twin_frames=int(twin_cnt[_b]), dropped_after=bool(dropped_after[_b]), hold_s=(round((int(drop_step[_b]) - int(first_latch[_b])) / 50.0, 2) if bool(dropped_after[_b]) else None), watch_s=(round((int(step) - int(first_latch[_b])) / 50.0, 2) if int(first_latch[_b]) >= 0 else None),
                box_long=(round(float(raw_env._box_half_extents[_b, 0]) * 2, 3) if hasattr(raw_env, "_box_half_extents") else None),
                                    disp=float(throw_disp[_b]),
                                    disp_flight=float(throw_disp_fl[_b]),
                                    disp_start=float(throw_disp_st[_b]),
                                    foot_max=(float(foot_max[_b].max())
                                              if _NF else None),
                                    foot_net=(float(foot_net[_b].max())
                                              if _NF else None),
                                    fell=bool(throw_fell[_b]),
                                    zmin=float(throw_zmin[_b]),
                                    tiltmax=float(throw_tiltmx[_b]),
                                    step=int(step),
                                    armed=float(throw_armed[_b]),
                                    rise=_nz(throw_rise[_b]),
                                    hz=_nz(throw_hz[_b]), hy=_nz(throw_hy[_b]),
                                    cf_dx=_nz(throw_cfdx[_b]), cf_dy=_nz(throw_cfdy[_b]),
                                    l_d=_nz(throw_ld[_b]), l_v=_nz(throw_lv[_b])))
                        throw_done   = torch.where(rising,
                                                   torch.zeros_like(throw_done),
                                                   throw_done)
                        throw_done_l = torch.where(rising,
                                                   torch.zeros_like(throw_done_l),
                                                   throw_done_l)
                        throw_done_t = torch.where(rising, torch.zeros_like(throw_done_t), throw_done_t)
                        throw_done2 = torch.where(rising, torch.zeros_like(throw_done2), throw_done2); throw_icpt = torch.where(rising, torch.zeros_like(throw_icpt), throw_icpt)
                        fail_cnt = torch.where(rising[:, None], torch.zeros_like(fail_cnt), fail_cnt); twin_cnt = torch.where(rising, torch.zeros_like(twin_cnt), twin_cnt)
                        dropped_after = torch.where(rising, torch.zeros_like(dropped_after), dropped_after); first_latch = torch.where(rising, torch.full_like(first_latch, -1), first_latch); drop_step = torch.where(rising, torch.full_like(drop_step, -1), drop_step)
                    prev_launch = launched.clone()

                # --- step displacement bookkeeping ---
                base_xy = robot.data.root_pos_w[:, :2]
                if prev_launch is not None and rising.any():
                    for _b in rising.nonzero(as_tuple=False).flatten().tolist():
                        # STILL-PHASE ONLY: the reference clip walks 0 -> -1.68 m
                        # over its first 6 s (frames 0-300) then stands still for
                        # 24 s.  A throw overlapping the walking phase records
                        # that 1.68 m as "chasing", which it is not.  Only throws
                        # starting after the walk measure real pursuit (the
                        # anchor is a constant there).
                        if (bool(throw_active[_b]) and int(throw_n[_b]) > 1
                                and step >= 350 and not bool(throw_fell[_b])):
                            disp_log.append(float(throw_disp[_b]))
                    throw_fell = torch.where(rising,
                                             torch.zeros_like(throw_fell),
                                             throw_fell)
                    throw_zmin = torch.where(
                        rising, torch.full_like(throw_zmin, 9.9), throw_zmin)
                    throw_tiltmx = torch.where(
                        rising, torch.zeros_like(throw_tiltmx), throw_tiltmx)
                    throw_rad = torch.where(rising,
                                            torch.full_like(throw_rad,
                                                            float("nan")),
                                            throw_rad)
                    throw_armed = torch.where(rising,
                                              torch.zeros_like(throw_armed),
                                              throw_armed)
                    throw_rise = torch.where(rising,
                                             torch.full_like(throw_rise, float("nan")),
                                             throw_rise)
                    throw_base0 = torch.where(rising[:, None], base_xy,
                                              throw_base0)
                    _contact_free(rising.nonzero(as_tuple=False).flatten())
                    throw_disp = torch.where(rising,
                                             torch.zeros_like(throw_disp),
                                             throw_disp)
                    throw_disp_fl = torch.where(rising,
                                                torch.zeros_like(throw_disp_fl),
                                                throw_disp_fl)
                    if _HAND:
                        for _b in rising.nonzero(as_tuple=False).flatten().tolist():
                            if float(_best_d[_b]) < 1e8:
                                arm_at_contact.append(float(_arm_at_best[_b]))
                            blend_log.append((float(_blend_at_best[_b]),
                                              bool(throw_done[_b])))
                            _e = float(_arm_at_early[_b])
                            if _e == _e:
                                arm_early.append(_e)
                        _arm_at_early = torch.where(
                            rising, torch.full_like(_arm_at_early, float("nan")),
                            _arm_at_early)
                        _best_d = torch.where(rising,
                                              torch.full_like(_best_d, 1e9),
                                              _best_d)
                    throw_disp_st = torch.where(rising,
                                                torch.zeros_like(throw_disp_st),
                                                throw_disp_st)
                    throw_age = torch.where(rising, torch.zeros_like(throw_age),
                                            throw_age)
                    if _NF:
                        _r3 = rising[:, None, None]
                        foot0 = torch.where(
                            _r3, robot.data.body_pos_w[:, _FOOT], foot0)
                        _r4 = rising[:, None]
                        foot_max = torch.where(_r4, torch.zeros_like(foot_max),
                                               foot_max)
                        foot_net = torch.where(_r4, torch.zeros_like(foot_net),
                                               foot_net)
                    if _NL:
                        _r2 = rising[:, None]
                        cmd_lo = torch.where(_r2, torch.full_like(cmd_lo, _BIG), cmd_lo)
                        cmd_hi = torch.where(_r2, torch.full_like(cmd_hi, -_BIG), cmd_hi)
                        act_lo = torch.where(_r2, torch.full_like(act_lo, _BIG), act_lo)
                        act_hi = torch.where(_r2, torch.full_like(act_hi, -_BIG), act_hi)
                    throw_end_ctr = torch.where(
                        rising, torch.full_like(throw_end_ctr, -1),
                        throw_end_ctr)
                # A fallen robot slides metres; that is not pursuit.  Flag any
                # throw during which the base drops below 0.45 m and exclude it
                # from the displacement statistic (max 2.1-2.4 m seen otherwise).
                throw_fell = torch.where(
                    robot.data.root_pos_w[:, 2] < 0.45,
                    torch.ones_like(throw_fell), throw_fell)
                # same tick, same tensors: min base height / max tilt (the twin-comparable instability pair)
                _bz_r = robot.data.root_pos_w[:, 2]
                _qr = robot.data.root_quat_w
                _upz = 1.0 - 2.0 * (_qr[:, 1] ** 2 + _qr[:, 2] ** 2)
                _tl_r = torch.rad2deg(torch.acos(_upz.clamp(-1.0, 1.0)))
                throw_zmin = torch.where(throw_active,
                                         torch.minimum(throw_zmin, _bz_r),
                                         throw_zmin)
                throw_tiltmx = torch.where(throw_active,
                                           torch.maximum(throw_tiltmx, _tl_r),
                                           throw_tiltmx)
                # Only count ticks where the reference clip is STANDING
                # (walk_then_catch_v17_30s walks frames 60-298); otherwise the reference
                # walk leaks into 'idle' and hides fidgeting.
                _mot_ts = raw_env.command_manager.get_term("motion").time_steps
                # A hovering box moving with a WALKING thrower (0.3-1 m/s) is not idle either
                # -> require a really static box (<0.1 m/s); idle = thrower standing still.
                _idle = (~throw_active) & (balloon.data.root_lin_vel_w.norm(dim=-1) < 0.1) & ((_mot_ts >= 300) | (_mot_ts < 60))
                _bv2 = robot.data.root_lin_vel_w[:, :2].norm(dim=-1) ** 2
                _lv = robot.data.joint_vel[:, _leg_ids].abs().mean(dim=-1)
                idle_v2_sum = idle_v2_sum + torch.where(_idle, _bv2, torch.zeros_like(_bv2))
                idle_lv_sum = idle_lv_sum + torch.where(_idle, _lv, torch.zeros_like(_lv))
                idle_n = idle_n + _idle.float()
                # radius: first descending crossing of CATCH_H_REF this throw
                _bz = balloon.data.root_pos_w[:, 2]
                _bvz = balloon.data.root_lin_vel_w[:, 2]
                _bl = getattr(env.unwrapped, "_v38_ts_blend_pub", None)
                if _bl is not None:
                    throw_armed = torch.maximum(throw_armed, _bl.detach())
                    throw_rise = torch.where(
                        torch.isnan(throw_rise) & throw_active & (_bl.detach() >= 0.5),
                        throw_age, throw_rise)
                _cross = (throw_active & torch.isnan(throw_rad)
                          & (_bz <= CATCH_H_REF) & (_bvz < 0.0))
                if _cross.any():
                    _r = (balloon.data.root_pos_w[:, :2]
                          - throw_base0).norm(dim=-1)
                    throw_rad = torch.where(_cross, _r, throw_rad)
                _d = (base_xy - throw_base0).norm(dim=-1)
                throw_disp = torch.where(throw_active,
                                         torch.maximum(throw_disp, _d),
                                         throw_disp)
                # flight window: open until caught-or-landed, then 0.3 s more
                _ended_now = (throw_active & (throw_end_ctr < 0)
                              & (throw_done | (_bz < LAND_H)))
                throw_end_ctr = torch.where(
                    _ended_now, torch.full_like(throw_end_ctr, FLIGHT_TAIL),
                    throw_end_ctr)
                _fl_open = throw_active & ((throw_end_ctr < 0)
                                           | (throw_end_ctr > 0))
                throw_disp_fl = torch.where(_fl_open,
                                            torch.maximum(throw_disp_fl, _d),
                                            throw_disp_fl)
                throw_end_ctr = torch.where(throw_end_ctr > 0,
                                            throw_end_ctr - 1, throw_end_ctr)
                # sample every 5th step: float(...) forces a GPU->CPU sync,
                # and doing that twice per step at 256 envs makes the eval
                # several times slower.
                if _DIAG and (step % 5 == 0) and obs.shape[1] > 60:
                    _az = obs[:, 60][_in_start] if _in_start.any() else None
                    if _az is not None and _az.numel():
                        anchor_obs_log.append(float(_az.median()))
                if _DIAG and _HAND:
                    _hz = (robot.data.body_pos_w[:, _HAND, 2].mean(dim=-1)
                           - robot.data.body_pos_w[:, _PELV[0], 2])
                    _bd = (balloon.data.root_pos_w
                           - robot.data.body_pos_w[:, _PELV[0]]).norm(dim=-1)
                    _bl = None
                    if _ATERM is not None:
                        _c = getattr(_ATERM, "_v38_ts_cache", None)
                        if _c is not None and _c[0] is not None:
                            _bl = _c[2]
                    _closer = throw_active & (_bd < _best_d)
                    if _bl is not None:
                        _blend_at_best = torch.where(_closer, _bl, _blend_at_best)
                    _best_d = torch.where(_closer, _bd, _best_d)
                    _arm_at_best = torch.where(_closer, _hz, _arm_at_best)
                    _hit_early = (throw_active & torch.isnan(_arm_at_early)
                                  & (_bd < D_EARLY))
                    _arm_at_early = torch.where(_hit_early, _hz, _arm_at_early)
                throw_age = torch.where(throw_active, throw_age + 1.0,
                                        throw_age)
                _in_start = throw_active & (throw_age <= START_STEPS)
                throw_disp_st = torch.where(
                    _in_start, torch.maximum(throw_disp_st, _d), throw_disp_st)
                _est = (getattr(raw_env, "_v31_est", None)
                        if (_DIAG and step % 5 == 0) else None)
                if isinstance(_est, dict) and _est.get("est_v_w") is not None:
                    # 0.6 s HARD window, not the flight window: the flight
                    # window runs past catch/landing, where the box is legit-
                    # imately still, and that tail drags the median to ~0.3 m/s.
                    _os = _est["est_v_w"].norm(dim=-1)[_in_start]
                    if _os.numel():
                        obs_spd_log.append(float(_os.median()))
                if _DIAG and _NF:
                    _fd = (robot.data.body_pos_w[:, _FOOT] - foot0).norm(dim=-1)
                    _o2 = _fl_open[:, None]
                    foot_max = torch.where(_o2, torch.maximum(foot_max, _fd),
                                           foot_max)
                    foot_net = torch.where(_o2, _fd, foot_net)   # last = net
                if _DIAG and _NL:
                    _open = _fl_open[:, None]
                    _c = robot.data.joint_pos_target[:, _leg_ids]
                    _a = robot.data.joint_pos[:, _leg_ids]
                    cmd_lo = torch.where(_open, torch.minimum(cmd_lo, _c), cmd_lo)
                    cmd_hi = torch.where(_open, torch.maximum(cmd_hi, _c), cmd_hi)
                    act_lo = torch.where(_open, torch.minimum(act_lo, _a), act_lo)
                    act_hi = torch.where(_open, torch.maximum(act_hi, _a), act_hi)

                held = strict_persist >= args.persist_frames
                newly = throw_active & (~throw_done) & held
                if newly.any():
                    throw_hit  = throw_hit + newly.long()
                    throw_done = throw_done | newly
                held_t = twin_persist >= args.persist_frames
                newly_t = throw_active & (~throw_done_t) & held_t
                if newly_t.any():
                    throw_hit_t  = throw_hit_t + newly_t.long()
                    throw_done_t = throw_done_t | newly_t
                held_l = loose_persist >= args.persist_frames
                newly_l = throw_active & (~throw_done_l) & held_l
                if newly_l.any():
                    throw_hit_l  = throw_hit_l + newly_l.long()
                    throw_done_l = throw_done_l | newly_l
                throw_done2 = throw_done2 | (throw_active & (strict2_persist >= args.persist_frames))
                throw_icpt = throw_icpt | (throw_active & (icpt_persist >= 5))
                _anyc = throw_active & (throw_done | throw_done_t | throw_done2)
                first_latch = torch.where(_anyc & (first_latch < 0), torch.full_like(first_latch, int(step)), first_latch)
                _low = _anyc & (balloon.data.root_pos_w[:, 2] < 0.35) & (~dropped_after)
                drop_step = torch.where(_low, torch.full_like(drop_step, int(step)), drop_step); dropped_after = dropped_after | _low
                _twn = throw_active & twin_ok & (~strict2)
                fail_cnt = fail_cnt + torch.stack([_twn & ~_px, _twn & ~_py, _twn & ~_pz, _twn & ~_wr, _twn & ~_hs, _twn & ~_sl], -1).long()
                twin_cnt = twin_cnt + _twn.long()

            # Snapshot on newly_strict -- diagnostic of edge cases.
            # newly_strict latches and fires at most once per env, so iterate
            # over its nonzero entries (one .any() sync per step) instead of a
            # blocking per-env loop.
            for b in (newly_strict.nonzero(as_tuple=False).flatten().tolist()
                      if bool(newly_strict.any()) else []):
                if snapshots[b] is None:
                    snapshots[b] = {
                        "step": step,
                        "d_head":   float(d_head[b]),
                        "d_chest":  float(d_chest[b]),
                        "d_lsh":    float(d_lsh[b]),
                        "d_rsh":    float(d_rsh[b]),
                        "d_lwrist": float(d_lwrist[b]),
                        "d_rwrist": float(d_rwrist[b]),
                        "ball_in_torso": ball_in_torso[b].cpu().tolist(),
                        "speed": float(speed[b]),
                    }

            if step % 50 == 0:
                n_strict = int(strict_pass.sum().item())
                n_loose = int(loose_pass.sum().item())
                tn = int(throw_n.sum().item()); th = int(throw_hit.sum().item())
                thl = int(throw_hit_l.sum().item())
                rate = (100.0 * th / tn) if tn else float("nan")
                rate_l = (100.0 * thl / tn) if tn else float("nan")
                # displacement so far, including throws still in flight, so the
                # number is available even if the run is killed before the
                # final summary block
                _dl = list(disp_log)
                if step >= 350:
                    _dl += [float(x) for x in
                            throw_disp[throw_active & (~throw_fell)].cpu().tolist()]
                if _dl:
                    _s = sorted(_dl); _n = len(_s)
                    _p50 = _s[_n // 2]; _mx = _s[-1]
                    _g15 = 100.0 * sum(1 for x in _s if x > 0.15) / _n
                    _g30 = 100.0 * sum(1 for x in _s if x > 0.30) / _n
                    _sd = (f" | STEP n={_n} p50 {_p50:.3f} max {_mx:.3f} "
                           f">15cm {_g15:.1f}% >30cm {_g30:.1f}%")
                else:
                    _sd = ""
                if rad_log and step % 200 == 0:
                    _edges = [0.0, 0.2, 0.4, 0.6, 0.8, 99.0]
                    _parts = []
                    for _lo, _hi in zip(_edges[:-1], _edges[1:]):
                        _sel = [c for r, c in rad_log if _lo <= r < _hi]
                        if _sel:
                            _k = sum(_sel); _n2 = len(_sel)
                            _parts.append(f"{_lo:.1f}-{_hi if _hi<9 else 9:.1f}:"
                                          f"{_k}/{_n2}={100.0*_k/_n2:.0f}%")
                    if _parts:
                        print("[RADBIN] " + "  ".join(_parts), flush=True)
                print(f"[strict] step={step} strict_pass={n_strict}/{B} loose_pass={n_loose}/{B}"
                      f" | PERTHROW {th}/{tn} = {rate:.1f}%"
                      f" LOOSE {thl}/{tn} = {rate_l:.1f}%{_sd}", flush=True)

    # Final summary
    result = {
        "ckpt": args.bm_ckpt,
        "env_id": args.env_id,
        "num_envs": B,
        "persist_frames": args.persist_frames,
        "ball_radius_pinned": args.ball_radius,
        "throws_total": int(throw_n.sum().item()),
        "throws_caught": int(throw_hit.sum().item()),
        "throws_per_env": throw_n.cpu().tolist(),
        "caught_per_env": throw_hit.cpu().tolist(),
        "strict_pass": int(strict_pass.sum().item()),
        "loose_pass": int(loose_pass.sum().item()),
        "false_positive": int((loose_pass & ~strict_pass).sum().item()),
        "strict_pass_per_env": strict_pass.cpu().tolist(),
        "loose_pass_per_env": loose_pass.cpu().tolist(),
        "strict_max_persist_per_env": strict_max.cpu().tolist(),
        "loose_max_persist_per_env": loose_max.cpu().tolist(),
        "snapshots_on_strict_pass": snapshots,
        "criteria": {
            "source": "reflex.envs.base.chest_v5_terms",
            "POCKET_X_RANGE": list(POCKET_X_RANGE),
            "POCKET_Z_RANGE": list(POCKET_Z_RANGE),
            "POCKET_Y_ABS_MAX": POCKET_Y_ABS_MAX,
            "WRIST_NEAR_MAX": WRIST_NEAR_MAX,
            "HEAD_FORBIDDEN_MAX": HEAD_FORBIDDEN_MAX,
            "SHOULDER_FORBIDDEN_MAX": SHOULDER_FORBIDDEN_MAX,
            "V_SLOW_MAX": V_SLOW_MAX,
            "LOOSE_NEAR_EEF": LOOSE_NEAR_EEF,
            "LOOSE_NEAR_CHEST": LOOSE_NEAR_CHEST,
            "LOOSE_V_SLOW": LOOSE_V_SLOW,
        },
    }

    # ---- per-throw records and derived cuts ----------------------------
    # (a) catch rate BY OFFSET BIN -- a catch rate without its conditions is
    #     not meaningful.  (The running [RADBIN] print only covers throws whose
    #     physical box descended through CATCH_H_REF, so it does not sum to
    #     the headline number.)
    # (b) catch rate by how far the robot moved during the flight.
    # FLUSH the throw still open in each env FIRST.  Records are banked on the
    # NEXT launch's rising edge, so without this the last throw of every env
    # would be lost -- and lost with a BIAS: a throw only gets a successor
    # when the previous one was MISSED (after a catch the box is cradled and
    # the next throw often never launches).  This flush must run before the
    # tables below are computed (the disp_log flush after json.dump is too
    # late for them).
    # [END] episode-end box geometry in the torso frame (size-aware diagnostics): where is a box that was never dropped?
    _ce = mdp_v5.compute_strict_components(raw_env); _eb = _ce["ball_in_torso"]; _edh = _ce["d_head"]; _edw = torch.maximum(_ce["d_lwrist"], _ce["d_rwrist"]); _ez = balloon.data.root_pos_w[:, 2]
    _etq = robot.data.body_quat_w[:, body_ids["chest"], :]; _ebq = balloon.data.root_quat_w; _ehe = raw_env._box_half_extents; _ehv = torch.zeros(B, device=device)
    for _i in range(3):
        _ax = torch.zeros(B, 3, device=device); _ax[:, _i] = 1.0
        _ehv = _ehv + mdp_v5._quat_apply_inv_vec(_etq, mdp_v5._quat_apply_vec(_ebq, _ax))[:, 2].abs() * _ehe[:, _i]
    for _b in range(B):
        if bool(throw_active[_b]):
            throw_log.append(dict(
                env=int(_b), idx=int(throw_n[_b]),   # see the record banked on the rising edge
                l_step=int(throw_lstep[_b]), mot_ts=int(throw_mts[_b]),
                rad=(None if bool(torch.isnan(throw_rad[_b]))
                     else float(throw_rad[_b])),
                caught=bool(throw_done[_b]), caught_twin=bool(throw_done_t[_b]), caught_v2=bool(throw_done2[_b]), intercepted=bool(throw_icpt[_b]), v2_fail=[int(_x) for _x in fail_cnt[_b].tolist()], v2_twin_frames=int(twin_cnt[_b]), end_b=[round(float(_x), 3) for _x in _eb[_b].tolist()], end_bot=round(float(_eb[_b, 2] - _ehv[_b]), 3), end_top=round(float(_eb[_b, 2] + _ehv[_b]), 3), end_dhead=round(float(_edh[_b]), 3), end_dw=round(float(_edw[_b]), 3), end_z=round(float(_ez[_b]), 3), end_v=round(float(_ce['speed'][_b]), 3), dropped_after=bool(dropped_after[_b]), hold_s=(round((int(drop_step[_b]) - int(first_latch[_b])) / 50.0, 2) if bool(dropped_after[_b]) else None), watch_s=(round((int(step) - int(first_latch[_b])) / 50.0, 2) if int(first_latch[_b]) >= 0 else None),
                box_long=(round(float(raw_env._box_half_extents[_b, 0]) * 2, 3) if hasattr(raw_env, "_box_half_extents") else None),
                disp=float(throw_disp[_b]),
                disp_flight=float(throw_disp_fl[_b]),
                disp_start=float(throw_disp_st[_b]),
                foot_max=(float(foot_max[_b].max()) if _NF else None),
                foot_net=(float(foot_net[_b].max()) if _NF else None),
                fell=bool(throw_fell[_b]),
                zmin=float(throw_zmin[_b]),
                tiltmax=float(throw_tiltmx[_b]),
                step=int(args.episode_steps),
                armed=float(throw_armed[_b]),
                                    rise=_nz(throw_rise[_b]),
                hz=_nz(throw_hz[_b]), hy=_nz(throw_hy[_b]),
                cf_dx=_nz(throw_cfdx[_b]), cf_dy=_nz(throw_cfdy[_b]),
                l_d=_nz(throw_ld[_b]), l_v=_nz(throw_lv[_b]),
                # A throw still IN FLIGHT at the episode end (never caught,
                # never landed) has an unknown outcome, but the flush books it
                # as a miss.  Flagged here; the headline is NOT changed (see
                # [PERTHROW-EX]).  "Never ended" alone over-counts -- a box
                # resting on the robot below the hold criterion never trips
                # LAND_H either -- so unknown outcome = still BALLISTIC at the
                # final tick (> 1 m/s).
                truncated=(bool(throw_end_ctr[_b] < 0) and not bool(throw_done[_b])
                           and float(balloon.data.root_lin_vel_w[_b].norm()) > 1.0)))
    print("[BIN] %d per-throw records (headline throw total: see [PERTHROW]); the two should be close, "
          "a large gap means throws are missing from the records" % len(throw_log), flush=True)
    _tl = [t for t in throw_log if not t["fell"] and t["step"] >= 350]
    _in = float(idle_n.sum().item())
    result["idle_base_rms"] = float((idle_v2_sum.sum() / idle_n.sum()).sqrt().item()) if _in > 0 else None
    result["idle_leg_vel"] = float((idle_lv_sum.sum() / idle_n.sum()).item()) if _in > 0 else None
    result["idle_ticks"] = int(_in)
    result["throw_log"] = throw_log
    if _tl:
        _EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 9.0]
        print("\n[BIN] strict catch rate by landing offset (n=%d, falls and walking-phase throws excluded)" % len(_tl), flush=True)
        _bins = {}
        for _i in range(len(_EDGES) - 1):
            _lo, _hi = _EDGES[_i], _EDGES[_i + 1]
            _s = [t for t in _tl if t["rad"] is not None and _lo <= t["rad"] < _hi]
            if _s:
                _c = sum(1 for t in _s if t["caught"])
                _bins["%.1f-%.1f" % (_lo, _hi)] = [_c, len(_s)]
                print("   offset %.1f-%.1f m : %3d/%3d = %4.1f%%"
                      % (_lo, _hi, _c, len(_s), 100.0 * _c / len(_s)), flush=True)
        result["bins_by_offset"] = _bins
        _sizes = sorted({t.get("box_long") for t in _tl if t.get("box_long") is not None})
        if len(_sizes) > 1:   # mixed box sizes: catch rate per box size (both record sites carry box_long)
            print("\n[BIN] strict catch rate by box size (long side, m)", flush=True); _bs = {}
            for _sz in _sizes:
                _s = [t for t in _tl if t.get("box_long") == _sz]; _c = sum(1 for t in _s if t["caught"])
                _bs["%.2f" % _sz] = [_c, len(_s)]
                print("   long side %.2f m : %3d/%3d = %4.1f%%" % (_sz, _c, len(_s), 100.0 * _c / max(1, len(_s))), flush=True)
            result["bins_by_box"] = _bs
        # ---- contact-free tables: no exclusions ----------------------------
        _hv = [t for t in _tl if t.get("hz") is not None]
        print("[HAND] contact-free crossing of the hand plane (0.35 m ahead of the pelvis; %d/%d throws have a value, "
              "the rest had no published launch state)"
              % (len(_hv), len(_tl)), flush=True)
        # The bin labels below are JSON keys of result["bins_hand_plane"] and are kept verbatim.  In order:
        # lands short of the hand plane (needs a forward step) / 0-0.5 m low / 0.5-0.8 m belly /
        # 0.8-1.1 m chest / 1.1-1.4 m face / above 1.4 m (overhead).
        _hb = {}
        for _lo, _hi, _lab in ((-9.0, 0.0, "lands before the hands (needs a step)"), (0.0, 0.5, "0-0.5 low"),
                               (0.5, 0.8, "0.5-0.8 belly"), (0.8, 1.1, "0.8-1.1 chest"),
                               (1.1, 1.4, "1.1-1.4 face"), (1.4, 9.0, ">1.4 overhead")):
            _s = [t for t in _hv if _lo <= t["hz"] < _hi]
            if _s:
                _c = sum(1 for t in _s if t["caught"])
                _hb[_lab] = [_c, len(_s)]
                print("   hand-plane z %-16s: %3d/%3d = %4.1f%%" % (_lab, _c, len(_s), 100.0 * _c / len(_s)), flush=True)
        for _lo, _hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 9.0)):
            _s = [t for t in _hv if t.get("hy") is not None and _lo <= abs(t["hy"]) < _hi]
            if _s:
                _c = sum(1 for t in _s if t["caught"])
                _hb["|y| %.2f-%.2f" % (_lo, _hi)] = [_c, len(_s)]
                print("   hand-plane |y| %.2f-%.2f      : %3d/%3d = %4.1f%%" % (_lo, _hi, _c, len(_s), 100.0 * _c / len(_s)), flush=True)
        result["bins_hand_plane"] = _hb
        _cv = [t for t in _tl if t.get("cf_dx") is not None]
        print("[CFBIN] strict catch rate by contact-free landing offset r = hypot(dx, dy) at the z = 1.0 m descending crossing (n=%d, no exclusions)" % len(_cv), flush=True)
        _cb = {}
        for _i in range(len(_EDGES) - 1):
            _lo, _hi = _EDGES[_i], _EDGES[_i + 1]
            _s = [t for t in _cv if _lo <= (t["cf_dx"] ** 2 + t["cf_dy"] ** 2) ** 0.5 < _hi]
            if _s:
                _c = sum(1 for t in _s if t["caught"])
                _cb["%.1f-%.1f" % (_lo, _hi)] = [_c, len(_s)]
                print("   cf offset %.1f-%.1f m : %3d/%3d = %4.1f%%" % (_lo, _hi, _c, len(_s), 100.0 * _c / len(_s)), flush=True)
        if _cv:
            import statistics as _st
            print("   cf dx (+ = short) p50 %+.2f | |dy| p50 %.2f | r p50 %.2f" % (
                _st.median([t["cf_dx"] for t in _cv]), _st.median([abs(t["cf_dy"]) for t in _cv]),
                _st.median([(t["cf_dx"] ** 2 + t["cf_dy"] ** 2) ** 0.5 for t in _cv])), flush=True)
        result["bins_cf_offset"] = _cb
        print("[MOVE] strict catch rate by base displacement in the flight window (launch -> catch or landing + 0.3 s)",
              flush=True)
        # The first label is a JSON key of result["bins_by_flight_disp"] and is kept verbatim
        # (it reads "< 5 cm, the magnitude seen on hardware").
        _mv = {}
        for _lab, _f in (("<5cm", lambda d: d < 0.05),
                         ("<15cm", lambda d: d < 0.15),
                         ("<30cm", lambda d: d < 0.30),
                         (">=30cm", lambda d: d >= 0.30)):
            _s = [t for t in _tl if _f(t["disp_flight"])]
            if _s:
                _c = sum(1 for t in _s if t["caught"])
                _mv[_lab] = [_c, len(_s)]
                print("   %-16s : %3d/%3d = %4.1f%%"
                      % (_lab, _c, len(_s), 100.0 * _c / len(_s)), flush=True)
        result["bins_by_flight_disp"] = _mv
        _fl = sorted(t["disp_flight"] for t in _tl)
        result["disp_flight_p50"] = _fl[len(_fl) // 2]
        result["disp_flight_p90"] = _fl[min(len(_fl) - 1, int(0.9 * len(_fl)))]
        result["disp_flight_frac_gt_30cm"] = (
            sum(1 for d in _fl if d > 0.30) / len(_fl))
        if anchor_obs_log:
            _ao = sorted(anchor_obs_log)
            result["anchor_obs_z_p50"] = _ao[len(_ao) // 2]
            result["anchor_obs_z_absp90"] = sorted(abs(x) for x in _ao)[
                int(0.9 * len(_ao))]
            print("[ANCHOR-OBS] anchor z in the policy observation: median %+.4f | |z| p90 %.4f"
                  "  (native sim value ~ -0.007)"
                  % (result["anchor_obs_z_p50"],
                     result["anchor_obs_z_absp90"]), flush=True)
        _bl2 = [x for x in blend_log if x[0] > 0.0]
        if _bl2:
            print("[BLEND@CONTACT] strict catch rate by arm blend b at closest approach (b = 1: arms fully in the cradle pose)",
                  flush=True)
            # Bin edges from a forward-kinematics sweep of bed geometry vs blend:
            # blend 0.90 changes hand spacing by only 0.8 cm against a 32 cm
            # box -- negligible.  The bed only deforms meaningfully (>3 cm)
            # below blend 0.6, and by then the arm is simply still on its way
            # up, which is a different failure from "bed is skewed".  So split
            # where the geometry actually differs, and report hand height per
            # bin so the two can be told apart.
            for _lab, _lo, _hi in (("<0.50", -1e9, 0.50), ("0.50-0.80", 0.50, 0.80),
                                   ("0.80-0.95", 0.80, 0.95), (">=0.95", 0.95, 1e9)):
                _sub = [x for x in _bl2 if _lo <= x[0] < _hi]
                if _sub:
                    _c2 = sum(1 for x in _sub if x[1])
                    print("   b %-10s %3d/%3d = %4.1f%%"
                          % (_lab, _c2, len(_sub), 100.0 * _c2 / len(_sub)),
                          flush=True)
            result["blend_at_contact"] = _bl2
        if arm_early:
            _ae = sorted(arm_early)
            result["arm_h_early_p50"] = _ae[len(_ae) // 2]
            result["arm_h_early_p10"] = _ae[int(0.1 * len(_ae))]
            print("[ARM@-0.3s] hand height above pelvis ~0.3 s before contact (box 1.05 m away): p50 %.3f / p10 "
                  "%.3f m (n=%d) -- the cradle should already be formed here" % (result["arm_h_early_p50"],
                                  result["arm_h_early_p10"], len(_ae)),
                  flush=True)
        if arm_at_contact:
            _ac = sorted(arm_at_contact)
            result["arm_h_at_contact_p50"] = _ac[len(_ac) // 2]
            result["arm_h_at_contact_p10"] = _ac[int(0.1 * len(_ac))]
            print("[ARM@CONTACT] hand height above pelvis at the box's closest approach: p50 %.3f m / p10 %.3f m"
                  "  (n=%d; cradle pose ~0.18-0.19 m, arms hanging ~0.01-0.04 m -- a p50 near the"
                  " cradle pose means the arm is not late and far-throw failures have another cause)"
                  % (result["arm_h_at_contact_p50"],
                     result["arm_h_at_contact_p10"], len(_ac)), flush=True)
        if obs_spd_log:
            _os2 = sorted(obs_spd_log)
            result["obs_box_speed_p50"] = _os2[len(_os2) // 2]
            result["obs_box_static_frac"] = (
                sum(1 for x in _os2 if x < 0.5) / len(_os2))
            print("[OBS-V] box speed seen by the policy (estimator), median over the first 0.6 s of flight %.2f m/s"
                  " | reads as static (<0.5 m/s) %.0f%%"
                  % (result["obs_box_speed_p50"],
                     100 * result["obs_box_static_frac"]), flush=True)
        _ft = [t for t in _tl if t.get("foot_max") and t["foot_max"] > 0.02]
        if _ft:
            _mx = sorted(t["foot_max"] for t in _ft)
            _nt = sorted(t["foot_net"] for t in _ft)
            _rt = sorted(t["foot_net"] / t["foot_max"] for t in _ft)
            _q = lambda v: v[len(v) // 2]
            result["foot_max_p50"] = _q(_mx)
            result["foot_net_p50"] = _q(_nt)
            result["foot_net_over_max_p50"] = _q(_rt)
            print("[FOOT] n=%d  max foot excursion p50 %.3f m | net foot displacement p50 %.3f m | "
                  "net/max p50 %.2f   (low = a step was started and then pulled back)"
                  % (len(_ft), _q(_mx), _q(_nt), _q(_rt)), flush=True)
        _st = sorted(t["disp_start"] for t in _tl if t.get("disp_start") is not None)
        if _st:
            result["disp_start_p50"] = _st[len(_st) // 2]
            print("[START-0.6s] base displacement in the first 0.6 s after launch: p50 %.3f m"
                  % result["disp_start_p50"], flush=True)
        print("[STEP-FLIGHT] flight-window base displacement p50 %.3f p90 %.3f m | >30cm %.1f%%"
              % (result["disp_flight_p50"], result["disp_flight_p90"],
                 100 * result["disp_flight_frac_gt_30cm"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[strict] SAVED → {args.out}", flush=True)
    for _b in range(B):
        if bool(throw_active[_b]):
            disp_log.append(float(throw_disp[_b]))
    if disp_log:
        _sd = sorted(disp_log)
        _n = len(_sd)
        _q = lambda f: _sd[min(_n - 1, int(f * _n))]
        _frac = lambda t: sum(1 for x in _sd if x > t) / _n
        result["step_disp_p50"] = _q(0.50)
        result["step_disp_p90"] = _q(0.90)
        result["step_disp_max"] = _sd[-1]
        result["step_frac_gt_15cm"] = _frac(0.15)
        result["step_frac_gt_30cm"] = _frac(0.30)
        print(f"[STEP] base displacement per throw (n={_n}): "
              f"p50 {_q(0.50):.3f} p90 {_q(0.90):.3f} max {_sd[-1]:.3f} m | "
              f">15cm {100*_frac(0.15):.1f}%  >30cm {100*_frac(0.30):.1f}%",
              flush=True)
    _tn = result["throws_total"]; _th = result["throws_caught"]
    if _tn:
        _p = _th / _tn
        _se = (_p * (1.0 - _p) / _tn) ** 0.5
        # ---- CONDITION STAMP -------------------------------------------------
        # A bare percentage is meaningless here: the same policy scores very
        # differently depending on perception / distance / offset / box.  Print
        # the conditions WITH the number so it cannot travel without them.
        # The perception / offset / coverage / arm strings assembled below are
        # code constants and are printed verbatim (partly in Chinese).  They
        # mean: ground-truth box state with the estimator bypassed, or the
        # hardware-like perception chain (coverage +/- jitter, dead fraction,
        # delay ticks, depth noise sigma(z), tangential scatter px, dirty
        # frames; unset sigma = "constant"); offset-throw probability and
        # offset range; distance-dependent detection coverage far/near (OFF =
        # the sim gets an early warning real perception does not have); arm
        # mode = two-stage (hang, then raise) or arms always raised.  An unset
        # dirty-curriculum value prints as "default, see the [box_v31] line".
        _e = os.environ.get
        _perc = (f"LATENT DETECTOR {args.detector}" if getattr(args, "detector", None) else "ground truth (estimator bypassed)" if _e("BOX_DP_EST", "0") != "1" else
                 "detection model (coverage %s±%s / dead %s / delay %s-%s ticks / sigma(z)=%sz^%s cm"
                 " / tangential %spx / dirty %s)" % (
                     _e("BOX_DP_COVERAGE", "0.409"), _e("BOX_DP_COV_JITTER", "0"),
                     _e("BOX_DP_DEAD_FRAC", "0"), _e("BOX_V36_DELAY_LO", "2"),
                     _e("BOX_V36_DELAY_HI", "5"), _e("BOX_DP_RAD_SIG_A", "const"),
                     _e("BOX_DP_RAD_SIG_B", "-"), _e("BOX_DP_TANG_SCATTER_PX", "13.4"),
                     _e("BOX_DP_DIRTY_P", "0")))
        _off = ("P(offset)=%s offset %s-%s m" % (_e("BOX_V38_P_OFFSET", "0.40"),
                                           _e("BOX_V38_OFF_LO", "0"),
                                           _e("BOX_V38_OFF_HI", "0")))
        # Every knob that changes the answer is stamped: distance-dependent
        # detection coverage (FAR_COV / NEAR_COV, default 0 = OFF), the
        # dirty-perception curriculum (noise / jump / freeze / garbage), the arm
        # configuration (it decides which behaviour is being scored), and the
        # md5 of this file (which version of the scoring code produced the log).
        # The curriculum values are read from the estimator module itself, not
        # from os.environ with a guessed default: printing an env var proves
        # nothing, only the live value does.
        try:
            from reflex.envs import perception_estimator as _d31
            _dirty = ("noise %s jump %s freeze %s garbage %s [live]" % (
                getattr(_d31, "V31_MIX_NOISE", "?"), getattr(_d31, "V31_MIX_JUMP", "?"),
                getattr(_d31, "V31_MIX_FREEZE", "?"), getattr(_d31, "V31_MIX_GARBAGE", "?")))
        except Exception:
            _dirty = ("noise %s jump %s freeze %s garbage %s" % tuple(
                (_e(k) if _e(k) is not None else "default")
                for k in ("BOX_V31_MIX_NOISE", "BOX_V31_MIX_JUMP",
                          "BOX_V31_MIX_FREEZE", "BOX_V31_MIX_GARBAGE")))
        _fc, _nc = _e("BOX_DP_FAR_COV", "0"), _e("BOX_DP_NEAR_COV", "0")
        _cov = ("distance-dependent detection far=%s near=%s" % (_fc, _nc)
                if (_fc != "0" or _nc != "0")
                else "distance-dependent detection OFF")
        _arm = ("two-stage arms (D_TRIG=%s LATCH=%s DOWN=%s)" % (
                    _e("BOX_V38_TS_D_TRIG", "-"), _e("BOX_V38_TS_LATCH", "-"),
                    (_e("BOX_V38_TS_DOWN", "clip") or "clip")[:28])
                if _e("BOX_V38_ARM_TWOSTAGE", "0") == "1" else
                "arms always raised (TWOSTAGE=0)")
        try:
            _rv = hashlib.md5(open(__file__, "rb").read()).hexdigest()[:8]
        except Exception:
            _rv = "unknown"
        print("[COND] perception = %s" % _perc, flush=True)
        print("[COND] dirty-perception curriculum = %s | %s" % (_dirty, _cov), flush=True)
        print("[COND] arm = %s" % _arm, flush=True)
        print("[COND] throw distance %s-%s m | %s | box %s | arm gain %s | actuation delay %s tick"
              % (_e("BOX_DIST_LO", "3.0"), _e("BOX_DIST_HI", "4.2"), _off,
                 _e("BOX_MODEL", "random"), _e("EVAL_ARM_GAIN", "1.0"),
                 _e("EVAL_ACT_DELAY", "0")), flush=True)
        print("[COND] scorer = per_throw_eval.py md5 %s | ckpt %s"
              % (_rv, args.bm_ckpt), flush=True)
        print(f"[PROTOCOL] eval_start_step={args.eval_start_step} episode_steps={args.episode_steps} push_robot={args.push_robot} motion_phase={args.motion_phase or 'env-default'} persist_frames={args.persist_frames} | env: hold_init_frac={os.environ.get('BOX_HOLD_INIT_FRAC','0')} rethrow={os.environ.get('BOX_RETHROW','1')} walk_thrower={os.environ.get('BOX_V36_WALK_THROWER','1')} p_abandon={os.environ.get('BOX_V38_P_ABANDON','0.08')} perturb={os.environ.get('BOX_V38_PERTURB','1')}", flush=True)
        print(f"[PERTHROW] catch rate {_th}/{_tn} = {100*_p:.1f}% "
              f"+/- {100*1.96*_se:.1f}pp (95% CI)", flush=True)
        # Twin criterion over the per-throw records (reported alongside the strict headline).
        _ct = sum(1 for t in throw_log if t.get("caught_twin")); _cn = len(throw_log)
        if _cn:
            print(f"[PERTHROW-TWIN] twin criterion (box < 0.65 m from pelvis, |v| < 1.2 m/s, above pelvis - 0.25 m, held 0.6 s): {_ct}/{_cn} = {100*_ct/_cn:.1f}%  (over per-throw records)", flush=True)
        _in = float(idle_n.sum().item())
        if _in > 0:
            _rms = float((idle_v2_sum.sum() / idle_n.sum()).sqrt().item()); _lvm = float((idle_lv_sum.sum() / idle_n.sum()).item())
            print(f"[PERTHROW-IDLE] idle stability (box static < 0.1 m/s, reference clip standing): base speed RMS {_rms:.3f} m/s  mean leg/waist joint |v| {_lvm:.2f} rad/s  (ticks {int(_in)}; twin warning thresholds 0.08 / 0.6)", flush=True)
        # FIRST-THROW headline: the automatic re-throw after a landed box (RETHROW_GRACE_STEPS) is a training-
        # curriculum feature (off under CLEAN2, BOX_RETHROW=0); re-throws start with the robot wherever the miss left
        # it and are not independent trials.  Each env's first throw is the clean, independent trial.
        _f1 = [t for t in throw_log if t.get("idx") == 1]
        if _f1:
            _n1 = len(_f1); _v2 = sum(1 for t in _f1 if t.get("caught_v2")); _ic = sum(1 for t in _f1 if t.get("intercepted")); _dr = sum(1 for t in _f1 if t.get("intercepted") and not t.get("caught_v2"))
            _fs = [sum(t["v2_fail"][k] for t in _f1 if t.get("v2_fail")) for k in range(6)]; _tf = sum(t.get("v2_twin_frames", 0) for t in _f1)
            def _drp(k):
                _cc = [t for t in _f1 if t.get(k)]; _dd = [t for t in _cc if t.get("dropped_after")]
                return f"{len(_dd)}/{len(_cc)}"
            _hs = sorted(t["hold_s"] for t in _f1 if t.get("hold_s") is not None); _ws = sorted(t["watch_s"] for t in _f1 if t.get("watch_s") is not None)
            _kept = lambda k: sum(1 for t in _f1 if t.get(k) and not t.get("dropped_after"))
            print(f"[PERTHROW-DROP] first throw | caught then dropped (box z < 0.35 m): strict {_drp('caught')} | twin {_drp('caught_twin')} | size-aware strict {_drp('caught_v2')} | caught and not dropped by episode end: twin {_kept('caught_twin')}/{_n1} = {100*_kept('caught_twin')/_n1:.1f}% | size-aware strict {_kept('caught_v2')}/{_n1} = {100*_kept('caught_v2')/_n1:.1f}% | median hold time before the drop {(_hs[len(_hs)//2] if _hs else float('nan')):.2f} s | median time observed after the catch {(_ws[len(_ws)//2] if _ws else float('nan')):.2f} s", flush=True)
            print(f"[PERTHROW-V2] first throw | size-aware strict {_v2}/{_n1} = {100*_v2/_n1:.1f}% | intercept {_ic}/{_n1} = {100*_ic/_n1:.1f}% | intercepted but not held (size-aware) {_dr}/{_n1} = {100*_dr/_n1:.1f}% | frames passing twin but not size-aware strict {_tf}: failing component x/y/z/wrist/headsh/slow = {_fs}", flush=True)
            _c1 = sum(1 for t in _f1 if t.get("caught")); _re = [t for t in throw_log if (t.get("idx") or 0) > 1]
            _cr = sum(1 for t in _re if t.get("caught"))
            print(f"[PERTHROW-FIRST] strict, first throw only (one per env): {_c1}/{len(_f1)} = {100*_c1/len(_f1):.1f}%  | re-throws (automatic, after a landed box) "
                  f"{_cr}/{len(_re)} = {(100*_cr/len(_re)) if _re else float('nan'):.1f}%", flush=True)
        # Same strict number with the unresolved end-of-episode throws removed from the
        # denominator (outcome unknown, not a miss).  Reported alongside; the headline is unchanged.
        _tr = [t for t in throw_log if t.get("truncated")]
        _ok = [t for t in throw_log if not t.get("truncated")]
        if _ok:
            _c2 = sum(1 for t in _ok if t.get("caught"))
            print(f"[PERTHROW-EX] strict, excluding {len(_tr)} throws still in flight at episode end (box speed > 1 m/s, outcome unknown): "
                  f"{_c2}/{len(_ok)} = {100*_c2/len(_ok):.1f}%  (over per-throw records)", flush=True)
    else:
        print("[PERTHROW] no throws observed", flush=True)
    print(f"[strict] STRICT pass: {result['strict_pass']}/{B}", flush=True)
    print(f"[strict] LOOSE  pass: {result['loose_pass']}/{B}", flush=True)
    print(f"[strict] False positives (loose pass but strict fail): {result['false_positive']}/{B}", flush=True)
    if args.detector and args.save_frames_dir:
        _dp_save[0]["flush"](); print(f"[latent] DAgger data saved: {_dp_save[0]['chunk']} chunks", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # print BEFORE sim_app.close(): with cameras enabled close() can deadlock in the replicator
        # orchestrator and the traceback would never be seen
        import traceback as _tb
        _tb.print_exc(); _sys.stderr.flush(); _sys.stdout.flush()
        raise
    finally:
        sim_app.close()
