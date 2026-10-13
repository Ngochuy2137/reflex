"""Recompute the z* labels of feature files written by `per_throw_eval --save_feats` with the privileged encoder of another
latent checkpoint. The files carry the raw privileged vector, so feature collections can be reused across checkpoints
with the same privileged layout. Writes <name><out_suffix>.npz next to each input file.
Run with that checkpoint's latent configuration exported (BOX_LATENT_ZDIM, BOX_LATENT_PRIV_HI, BOX_LATENT_ENC_INIT, ...).

Usage:
  python -m reflex.vision.relabel_feats --ckpt data/checkpoints/latent_stage2.pt --out_suffix _relabel feats_0.npz feats_1.npz
"""
import argparse, sys, numpy as np, torch
p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--out_suffix", required=True); p.add_argument("files", nargs="+"); a = p.parse_args()
from reflex.latent.latent_actor_critic import LatentActorCritic, PRIV_HI
st = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
ac = LatentActorCritic({"policy": torch.zeros(1, PRIV_HI), "critic": torch.zeros(1, st["critic.0.weight"].shape[1])}, {"policy": ["policy"], "critic": ["critic"]}, 29,
                       actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu", init_noise_std=1.0, noise_std_type="scalar")
ac.load_state_dict(st, strict=True); ac.eval()
for f in a.files:
    d = dict(np.load(f)); assert "priv" in d, f"{f} has no priv (feature file written without the privileged observation)"
    with torch.no_grad(): z = ac.encode(torch.from_numpy(d["priv"])).numpy().astype(np.float32)
    print(f"[relabel] {f}: {z.shape[0]} frames, mean|dz| vs stored = {float(np.abs(z - d['z']).mean()) if d['z'].shape == z.shape else float('nan'):.4f}", flush=True)
    d["z"] = z; np.savez(f.replace(".npz", a.out_suffix + ".npz"), **d)
