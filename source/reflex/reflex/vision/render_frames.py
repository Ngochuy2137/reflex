"""Stage 3 data: egocentric camera frames + privileged labels, rendered while a latent policy drives the robot.

A TiledCamera per env is placed every tick at the torso camera mount (CAM_OFF, pitched up by CAM_PITCH_UP), with a
pinhole FOV matching the training visibility rectangle (2*FOV_H x 2*FOV_V deg). Each policy tick (50 Hz) records:
the RGB frame (uint8 HxWx3), optionally depth (uint16 mm, --depth), the privileged vector obs[170:181] (11-D layout;
the input of the privileged encoder, i.e. the label source), GT box pos/vel (camera frame if
BOX_LATENT_PRIV_FRAME=camera, else pelvis frame), the 9-D camera ego-motion, the episode id per env, the tick index
and the policy observation. z* = E_phi(priv) is computed offline by make_labels.py, so the frames do not depend on
a particular encoder; the driving policy (--bm_ckpt) only shapes the state distribution.
Output: compressed chunk_*.npz files in --out_dir.

Usage (Isaac Lab Python, from the repo root, with the latent configuration exported as in scripts/eval.sh):
  python -m reflex.vision.render_frames --headless --num_envs 32 --steps 1500 --depth --width 192 --height 144
      --bm_ckpt <latent_policy.pt> --out_dir data/frames/<tag>
"""
import argparse, math, os, sys, time
from reflex import paths as _paths
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=32); p.add_argument("--steps", type=int, default=1500)
p.add_argument("--bm_ckpt", required=True); p.add_argument("--out_dir", required=True)
p.add_argument("--width", type=int, default=128); p.add_argument("--height", type=int, default=96)
p.add_argument("--fov_h", type=float, default=53.0); p.add_argument("--fov_v", type=float, default=40.0)
p.add_argument("--chunk", type=int, default=250); p.add_argument("--seed", type=int, default=7)
p.add_argument("--depth", action="store_true", help="also render distance_to_image_plane; saved as uint16 millimetres (clip 20 m) under key depth (RGB-D visual encoder)")
p.add_argument("--episode_s", type=float, default=0.0, help="override episode length (s); 0 = env default (10 s); shorter episodes give more flight frames per rendered step")
p.add_argument("--motion_file", default=_paths.MOTION_FILE)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app
import gymnasium as gym, numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
import whole_body_tracking.tasks  # noqa
import reflex.latent.register  # noqa
from reflex.latent.latent_env_cfg import G1BoxCatchLatentEnvCfg
from reflex.latent.latent_actor_critic import LatentActorCritic
from reflex.latent import latent_obs as lo
from reflex.envs.robust_estimator import CAM_OFF, CAM_PITCH_UP

cfg = G1BoxCatchLatentEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.commands.motion.motion_file = a.motion_file
if a.episode_s > 0: cfg.episode_length_s = a.episode_s; print(f"[render] episode_length_s -> {a.episode_s}", flush=True)
# rendering hygiene: no debug markers in the image, and neighbours far enough that their boxes/robots
# are out of view (label = this env's box only).  Neither changes the policy's observations.
cfg.scene.env_spacing = float(os.environ.get("BOX_LATENT_RENDER_SPACING", "40.0"))
def _kill_debug_vis(obj, depth=0):
    if depth > 3 or obj is None: return
    for k in list(getattr(obj, "__dict__", {}).keys()):
        v = getattr(obj, k)
        if k == "debug_vis" and isinstance(v, bool): setattr(obj, k, False)
        elif hasattr(v, "__dict__") and not isinstance(v, (str, int, float)): _kill_debug_vis(v, depth + 1)
_kill_debug_vis(cfg)
# pinhole with horizontal FOV = 2*fov_h: focal = aperture/2 / tan(fov_h); vertical FOV follows aspect,
# so pick height/width = tan(fov_v)/tan(fov_h) (96/128 = 0.75 ~ tan40/tan53 = 0.63 -> vertical FOV slightly
# larger than the label rectangle; the label rule stays the rectangle, the detector sees a bit more).
aperture = 20.955; focal = aperture / 2.0 / math.tan(math.radians(a.fov_h))
cfg.scene.latent_cam = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/latent_cam", height=a.height, width=a.width, data_types=["rgb"] + (["distance_to_image_plane"] if a.depth else []),
    spawn=sim_utils.PinholeCameraCfg(focal_length=focal, focus_distance=400.0, horizontal_aperture=aperture,
                                     clipping_range=(0.05, 50.0)),
    offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"))
env = gym.make("Reflex-Latent-G1-v0", cfg=cfg); raw = env.unwrapped
obs, _ = env.reset(seed=a.seed); n_obs = obs["policy"].shape[1]
st = torch.load(a.bm_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
ac = LatentActorCritic({"policy": torch.zeros(1, n_obs), "critic": torch.zeros(1, st["critic.0.weight"].shape[1])},
                       {"policy": ["policy"], "critic": ["critic"]}, 29, actor_hidden_dims=[512, 256, 128],
                       critic_hidden_dims=[512, 256, 128], activation="elu", init_noise_std=1.0, noise_std_type="scalar")
ac.load_state_dict(st, strict=True); ac = ac.to("cuda:0").eval()
cam = raw.scene["latent_cam"]; robot = raw.scene["robot"]; box = raw.scene["balloon"]
print(f"[render] aux label frame = {lo.LAT_PRIV_FRAME}", flush=True)
ti = list(robot.data.body_names).index("torso_link"); dev = raw.device; n = a.num_envs
# camera quaternion: torso frame pitched up by CAM_PITCH_UP (x = optical axis); Isaac camera convention:
# "world" offset + set_world_poses(..., convention="world") expects the ROS/world camera frame with +X forward.
q_pitch = torch.tensor([math.cos(-CAM_PITCH_UP / 2), 0.0, math.sin(-CAM_PITCH_UP / 2), 0.0], device=dev).expand(n, 4)
os.makedirs(a.out_dir, exist_ok=True)
ep_id = torch.zeros(n, dtype=torch.long, device=dev); env_idx = np.arange(n, dtype=np.int32); frames, privs, poss, vels, eps, ticks, envs, obss = [], [], [], [], [], [], [], []
depths = []; egos = []   # egos: per-frame camera ego-motion (w_cam, v_cam, gravity dir) in the camera frame, 9-D (IMU-like), input of the visual encoder's GRU
if a.depth: print("[render] DEPTH channel on: distance_to_image_plane -> uint16 mm (clip 20 m)", flush=True)
chunk_i = 0; t0 = time.time()
def flush():
    global chunk_i, frames, privs, poss, vels, eps, ticks, envs, obss, depths, egos
    if not frames: return
    _extra = {"depth": np.concatenate(depths)} if depths else {}
    _extra["ego"] = np.concatenate(egos)
    np.savez_compressed(os.path.join(a.out_dir, f"chunk_{chunk_i:04d}.npz"), rgb=np.concatenate(frames), priv=np.concatenate(privs), **_extra,
                        pos=np.concatenate(poss), vel=np.concatenate(vels), ep=np.concatenate(eps), tick=np.concatenate(ticks), env=np.concatenate(envs), obs=np.concatenate(obss))
    print(f"[render] chunk {chunk_i} written ({sum(f.shape[0] for f in frames)} frames, {time.time()-t0:.0f}s)", flush=True)
    chunk_i += 1; frames, privs, poss, vels, eps, ticks, envs, obss = [], [], [], [], [], [], [], []; depths = []; egos = []
with torch.no_grad():
    for t in range(a.steps):
        tq = robot.data.body_quat_w[:, ti]; tpos = robot.data.body_pos_w[:, ti]
        cam_pos = tpos + quat_apply(tq, torch.tensor(CAM_OFF, device=dev).expand(n, 3))
        cam_q = quat_mul(tq, q_pitch)
        cam.set_world_poses(cam_pos, cam_q, convention="world")
        o = obs["policy"]; priv = o[:, 170:181]
        if lo.LAT_PRIV_FRAME == "camera":   # aux labels in the same frame as the priv term (camera frame, velocity relative to the camera)
            pos_b = quat_apply_inverse(cam_q, box.data.root_pos_w - cam_pos)
            vel_b = quat_apply_inverse(cam_q, box.data.root_lin_vel_w - robot.data.body_lin_vel_w[:, ti])
        else:
            pos_b = quat_apply_inverse(robot.data.root_quat_w, box.data.root_pos_w - robot.data.root_pos_w)
            vel_b = quat_apply_inverse(robot.data.root_quat_w, box.data.root_lin_vel_w)
        rgb = cam.data.output["rgb"][..., :3]
        egos.append(torch.cat([quat_apply_inverse(cam_q, robot.data.body_ang_vel_w[:, ti]), quat_apply_inverse(cam_q, robot.data.body_lin_vel_w[:, ti]),
                               quat_apply_inverse(cam_q, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n, 3))], -1).cpu().numpy().astype(np.float32))
        if a.depth:
            dpt = cam.data.output["distance_to_image_plane"]; dpt = dpt[..., 0] if dpt.dim() == 4 else dpt
            depths.append((torch.nan_to_num(dpt, nan=20.0, posinf=20.0).clamp(0, 20.0) * 1000.0).round().to(torch.int32).cpu().numpy().astype(np.uint16))
        frames.append(rgb.cpu().numpy().astype(np.uint8)); privs.append(priv.cpu().numpy().astype(np.float32))
        poss.append(pos_b.cpu().numpy().astype(np.float32)); vels.append(vel_b.cpu().numpy().astype(np.float32))
        eps.append(ep_id.cpu().numpy().astype(np.int32)); ticks.append(np.full(n, t, dtype=np.int32)); envs.append(env_idx.copy()); obss.append(o.cpu().numpy().astype(np.float16))
        obs, _, term, trunc, _ = env.step(ac.actor(ac.features(o)))
        done = (term | trunc); ep_id = ep_id + done.long()   # episode counter per env; key = (env, ep)
        if (t + 1) % a.chunk == 0: flush()
flush()
print(f"[render] DONE {a.steps} steps x {n} envs = {a.steps*n} frames -> {a.out_dir} in {time.time()-t0:.0f}s", flush=True)
env.close(); app.close()
