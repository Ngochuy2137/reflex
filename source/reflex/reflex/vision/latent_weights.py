"""Probe what the Stage 2 latent z encodes and how strongly the decoder uses each dimension; optionally write
task-aware per-dimension z loss weights for train_seqdet.py (--zw).

Rolls the latent policy in its env and collects (z, privileged vector, actor input) on flight ticks with the box
inside the visibility window. Targets come from the same (possibly delayed) privileged observation that z is
computed from: pos, vel, box long edge, time to contact, lateral/vertical crossing point, and orientation / angular
velocity (26-D privileged layout only). Reports the held-out R2 of a linear ridge probe and of a quadratic-feature
ridge probe (z, z_i z_j), and the decoder sensitivity per z dim (mean |a(z + sigma_d e_d) - a(z)|, sigma_d = std of
z_d), also normalised to sum to 1. --zw_out saves these sensitivities, normalised to mean 1, as the weights 'w'.

Usage (Isaac Lab Python, from the repo root, with the latent configuration exported as in scripts/eval.sh):
  python -m reflex.vision.latent_weights --headless --num_envs 64 --steps 1500
      --bm_ckpt data/checkpoints/latent_stage2.pt --out outputs/vision/latent_probe.json --zw_out outputs/vision/zw.npz
"""
import argparse, json, sys
from reflex import paths as _paths
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=64); p.add_argument("--steps", type=int, default=1500)
p.add_argument("--bm_ckpt", required=True); p.add_argument("--out", default=None); p.add_argument("--zw_out", default=None, help="write task-aware z loss weights (npz) for reflex.vision.train_seqdet --zw")
p.add_argument("--motion_file", default=_paths.MOTION_FILE)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import gymnasium as gym, torch
import whole_body_tracking.tasks  # noqa
import reflex.latent.register  # noqa
from isaaclab.utils.math import quat_apply_inverse
from reflex.latent.latent_env_cfg import G1BoxCatchLatentEnvCfg
from reflex.latent.latent_actor_critic import LatentActorCritic
cfg = G1BoxCatchLatentEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.commands.motion.motion_file = a.motion_file
env = gym.make("Reflex-Latent-G1-v0", cfg=cfg); raw = env.unwrapped
obs, _ = env.reset(seed=7); n_obs = obs["policy"].shape[1]
st = torch.load(a.bm_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
ac = LatentActorCritic({"policy": torch.zeros(1, n_obs), "critic": torch.zeros(1, st["critic.0.weight"].shape[1])},
                       {"policy": ["policy"], "critic": ["critic"]}, 29, actor_hidden_dims=[512, 256, 128],
                       critic_hidden_dims=[512, 256, 128], activation="elu", init_noise_std=1.0, noise_std_type="scalar")
ac.load_state_dict(st, strict=True); ac = ac.to("cuda:0").eval()
Z, PRIV, FEAT = [], [], []
with torch.no_grad():
    for t in range(a.steps):
        o = obs["policy"]; priv = o[:, ac.plo:ac.phi]; z = ac.encode(priv)
        box = raw.scene["balloon"]
        flight = (box.data.root_lin_vel_w.norm(dim=-1) > 1.5) & (box.data.root_pos_w[:, 2] > 0.4) & (priv[:, 7] > 0.5)
        if int(flight.sum()):
            Z.append(z[flight].cpu()); PRIV.append(priv[flight].cpu()); FEAT.append(ac.features(o)[flight].cpu())
        obs, _, _, _, _ = env.step(ac.actor(ac.features(o)))
Z, PRIV, FEAT = torch.cat(Z), torch.cat(PRIV), torch.cat(FEAT)
N = Z.shape[0]; g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g); tr, te = perm[: int(0.8 * N)], perm[int(0.8 * N):]
pos, vel = PRIV[:, 0:3], PRIV[:, 3:6]
vx = vel[:, 0].clamp(max=-0.3)                                   # camera frame: x forward, the box approaches with vx < 0
ttc = (-(pos[:, 0] - 0.3) / vx).clamp(0.0, 2.0)
cross = torch.stack([pos[:, 1] + vel[:, 1] * ttc, pos[:, 2] + vel[:, 2] * ttc], 1)
T = {"pos": pos, "vel": vel, "box_long_edge": PRIV[:, 8:9] * 2, "ttc": ttc[:, None], "cross_yz": cross}
if PRIV.shape[1] >= 26: T["ori_outer"] = PRIV[:, 11:23]; T["angvel"] = PRIV[:, 23:26]

def quad(X):
    iu = torch.triu_indices(X.shape[1], X.shape[1]); return torch.cat([X, X[:, iu[0]] * X[:, iu[1]]], 1)

def ridge_r2(X, Y, lam=1e-3):
    X = (X - X[tr].mean(0)) / X[tr].std(0).clamp(min=1e-6); X1 = torch.cat([X, torch.ones(X.shape[0], 1)], 1).double(); Y = Y.double()
    A = X1[tr]; W = torch.linalg.solve(A.T @ A + lam * len(tr) * 1e-3 * torch.eye(A.shape[1], dtype=torch.double), A.T @ Y[tr])
    pred = X1[te] @ W; ss_res = ((Y[te] - pred) ** 2).sum(0); ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0).clamp(min=1e-12)
    return [round(float(v), 3) for v in (1 - ss_res / ss_tot)]
res = {"ckpt": a.bm_ckpt, "n_ticks": int(N), "z_dim": int(Z.shape[1]), "priv_dim": int(PRIV.shape[1]),
       "z_std_per_dim": [round(float(v), 3) for v in Z.std(0)]}
for k, Y in T.items():
    const = float(Y.std(0).max()) < 1e-6
    res[k] = {"linear_R2": None if const else ridge_r2(Z, Y), "quadratic_R2": None if const else ridge_r2(quad(Z), Y), "constant_target": const}
# decoder sensitivity per z dim (finite difference at the data's own scale)
dev = "cuda:0"; F = FEAT[te[:20000]].to(dev); sens = []
with torch.no_grad():
    a0 = ac.actor(F); zs = Z.std(0).to(dev); k0 = ac.keep
    for d in range(Z.shape[1]):
        F2 = F.clone(); F2[:, k0 + d] = (F2[:, k0 + d] + zs[d]).clamp(-1, 1); sens.append(float((ac.actor(F2) - a0).abs().mean()))
tot = sum(sens) or 1.0
res["decoder_sensitivity_per_dim"] = [round(v, 4) for v in sens]; res["decoder_sensitivity_share"] = [round(v / tot, 3) for v in sens]
print("[latent_weights]", json.dumps(res))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
if a.zw_out:
    import numpy as np; w_ = np.array(sens); np.savez(a.zw_out, w=(w_ / w_.mean()).astype(np.float32), raw=w_.astype(np.float32), zstd=Z.std(0).numpy().astype(np.float32))
env.close(); app.close()
