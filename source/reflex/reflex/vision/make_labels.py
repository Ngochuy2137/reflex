"""Offline latent labels for rendered chunks: z* = E_phi(priv) from the privileged encoder of a latent checkpoint.

Reads every chunk_*.npz in --frames_dir (priv = obs[170:181] recorded at render time), runs the privileged encoder of
--bm_ckpt on it (CPU is fine) and writes <frames_dir>/labels_<tag>.npz with z (N, ZDIM), copies of
pos/vel/env/ep/tick/priv and the chunk/row index, so the trainers can build per-episode windows. Also prints the
per-dim std of z (dead-dim check). Assumes the 11-D privileged layout (a 181-D policy observation); run with the
checkpoint's latent configuration exported (BOX_LATENT_ZDIM, ...).

Usage:
  python -m reflex.vision.make_labels --frames_dir data/frames/<tag> --bm_ckpt <latent_policy.pt> --tag <tag>
"""
import argparse, glob, os, sys
import numpy as np, torch
p = argparse.ArgumentParser(); p.add_argument("--frames_dir", required=True); p.add_argument("--bm_ckpt", required=True); p.add_argument("--tag", required=True)
a = p.parse_args()
from reflex.latent.latent_actor_critic import LatentActorCritic
st = torch.load(a.bm_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
n_obs = st["actor.0.weight"].shape[1] - int(os.environ.get("BOX_LATENT_ZDIM", "16")) + 21   # 160 kept + 10 estimator channels + 11 priv
ac = LatentActorCritic({"policy": torch.zeros(1, 181), "critic": torch.zeros(1, st["critic.0.weight"].shape[1])},
                       {"policy": ["policy"], "critic": ["critic"]}, 29, actor_hidden_dims=[512, 256, 128],
                       critic_hidden_dims=[512, 256, 128], activation="elu", init_noise_std=1.0, noise_std_type="scalar")
ac.load_state_dict(st, strict=True); ac.eval()
Z, POS, VEL, ENV, EP, TICK, CHUNK, ROW, PRIV = [], [], [], [], [], [], [], [], []
files = sorted(glob.glob(os.path.join(a.frames_dir, "chunk_*.npz")))
with torch.no_grad():
    for ci, f in enumerate(files):
        d = np.load(f); priv = torch.from_numpy(d["priv"]).float(); z = ac.encode(priv).numpy().astype(np.float32)
        Z.append(z); POS.append(d["pos"]); VEL.append(d["vel"]); ENV.append(d["env"]); EP.append(d["ep"]); TICK.append(d["tick"]); PRIV.append(d["priv"])
        CHUNK.append(np.full(len(z), ci, dtype=np.int32)); ROW.append(np.arange(len(z), dtype=np.int32))
Z = np.concatenate(Z); out = os.path.join(a.frames_dir, f"labels_{a.tag}.npz")
np.savez_compressed(out, z=Z, pos=np.concatenate(POS), vel=np.concatenate(VEL), env=np.concatenate(ENV), ep=np.concatenate(EP),
                    tick=np.concatenate(TICK), chunk=np.concatenate(CHUNK), row=np.concatenate(ROW), priv=np.concatenate(PRIV), ckpt=a.bm_ckpt)
print(f"[labels] {len(files)} chunks, {len(Z)} frames -> {out}; z std per dim: {np.round(Z.std(0), 3).tolist()}", flush=True)
