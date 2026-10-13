"""Visual encoder in the loop: PPO fine-tuning of the latent decoder on z_hat from the frozen Stage 3 visual encoder
(used by reflex/latent/train.py --detector).

Wraps the rsl_rl VecEnv: after every reset/step it renders the torso camera of each env, keeps a WIN-frame ring
buffer, runs the frozen visual encoder and writes z_hat into the policy observation at columns [z_lo, z_lo + ZDIM).
With BOX_LATENT_Z_FROM_OBS_LO=z_lo the LatentActorCritic reads z from there, so rollouts, stored transitions and the
PPO update all use the same z_hat (the privileged encoder is unused). Overwritten columns: the estimator channels
[160:170) and the first privileged dims; nothing in the actor reads them in this mode, and the critic has its own
observation group. The env's own observation terms are untouched (only the returned tensor is edited in place).
"""
import math, os
import torch
from isaaclab.utils.math import quat_apply, quat_mul


class DetectorObsWrapper:
    def __init__(self, env, detector, win, z_lo, height, width, cam_name="latent_cam"):
        self.env = env; self.raw = env.unwrapped; self.det = detector; self.win = win; self.z_lo = z_lo
        from reflex.envs.robust_estimator import CAM_OFF, CAM_PITCH_UP
        self.cam = self.raw.scene[cam_name]; self.robot = self.raw.scene["robot"]
        self.ti = list(self.robot.data.body_names).index("torso_link"); n = self.raw.num_envs; dev = self.raw.device
        self.cam_off = torch.tensor(CAM_OFF, device=dev).expand(n, 3)
        self.qp = torch.tensor([math.cos(-CAM_PITCH_UP / 2), 0.0, math.sin(-CAM_PITCH_UP / 2), 0.0], device=dev).expand(n, 4)
        self.in_ch = int(getattr(detector, "in_ch", 3))
        self.ring = torch.zeros(n, win, self.in_ch, height, width, device=dev); self.filled = 0; self.zdim = None
        # Latency sensitivity: BOX_LATENT_Z_DELAY=N injects the encoder output computed N control ticks
        # (N x 20 ms) earlier, i.e. the whole perception result arrives N ticks late (0 = no delay).
        import os
        self.z_delay = int(os.environ.get("BOX_LATENT_Z_DELAY", "0")); self._zq = []
        print(f"[latent] DetectorObsWrapper: {n} envs, ring {win}x{self.in_ch}x{height}x{width}, z -> obs[{z_lo}:...]"
              f" | Z_DELAY={self.z_delay} ticks ({self.z_delay * 20} ms){' (ACTIVE)' if self.z_delay > 0 else ''}", flush=True)

    def _inject(self, obs):
        # rsl_rl returns a tensordict.TensorDict (NOT a dict subclass) or (TensorDict, extras) on reset:
        # write ONLY into the policy group's tensor (a TensorDict slice-assign would hit every group, incl. critic)
        if isinstance(obs, tuple):
            self._inject(obs[0]); return obs
        pol = obs["policy"] if hasattr(obs, "keys") and "policy" in obs.keys() else obs
        tq = self.robot.data.body_quat_w[:, self.ti]; tp = self.robot.data.body_pos_w[:, self.ti]
        self.cam.set_world_poses(tp + quat_apply(tq, self.cam_off), quat_mul(tq, self.qp), convention="world")
        self.cam.update(self.raw.step_dt, force_recompute=True)
        fr = self.cam.data.output["rgb"][..., :3].permute(0, 3, 1, 2).float() / 255.0
        if self.in_ch == 4:
            from reflex.vision.detector_model import depth_to_channel
            dpt = self.cam.data.output["distance_to_image_plane"]; dpt = dpt[..., 0] if dpt.dim() == 4 else dpt
            fr = torch.cat([fr, depth_to_channel(torch.nan_to_num(dpt, nan=20.0, posinf=20.0))[:, None]], 1)
        self.ring = torch.roll(self.ring, -1, dims=1); self.ring[:, -1] = fr
        if self.filled < self.win:
            self.ring[:, : self.win - self.filled - 1] = fr[:, None]; self.filled += 1
        with torch.no_grad():
            z = self.det(self.ring)[0]
        if self.z_delay > 0:
            self._zq.append(z)
            if len(self._zq) > self.z_delay: z = self._zq.pop(0)   # output from z_delay ticks ago
            else: z = self._zq[0]                                  # warm-up: oldest available
        self.zdim = z.shape[1]; pol[:, self.z_lo:self.z_lo + self.zdim] = z.to(pol.dtype)
        return obs

    # ---- VecEnv interface passthrough ----
    def get_observations(self):
        return self._inject(self.env.get_observations())
    def reset(self):
        return self._inject(self.env.reset())
    def step(self, actions):
        obs, rew, dones, extras = self.env.step(actions)
        return self._inject(obs), rew, dones, extras
    def __getattr__(self, name):
        return getattr(self.env, name)
