"""Stage 3, step 2: train the temporal part (GRU) of the visual encoder (arch 'kpseq').

The per-frame RGB-D CNN + box heatmap come from a checkpoint trained with train_detector.py (--kp, arch 'kp' or 'kpm';
a 'kpseq' checkpoint also works: its front end is reused and the GRU is trained from scratch) and stay frozen.
Per-frame geometric features (8-D) are computed once for the whole frame set on the GPU; a GRU over WIN frames of
[geo 8, ego-motion 9] is then trained to regress z* (+ aux pos/vel), with flight frames up-weighted (--w_flight) and
optional per-dim z weights (--zw, e.g. from latent_weights.py --zw_out).
Output: one checkpoint (arch 'kpseq') = front-end CNN/heatmap weights + GRU + feature normalisation, i.e.
image history (+ ego-motion) -> z_hat; load it with detector_model.load_detector.

Data: image sets = chunk_*.npz with rgb, depth and ego (render_frames.py --depth) + labels_<tag>.npz (make_labels.py),
and/or feature sets = 17-D per-frame features + z* labels logged by `per_throw_eval --save_feats` during rollouts
(DAgger; no images needed). --frames_dir none trains on feature sets only.

Usage (recipe of the released visual encoder: feature sets only, 8-D latent):
  python -m reflex.vision.train_seqdet --frames_dir none --labels none --kp outputs/vision/det_kp.pt --zdim 8
      --feat_sets outputs/vision/feats_0.npz,outputs/vision/feats_1.npz --zw outputs/vision/zw.npz
      --out outputs/vision/visual_encoder.pt
"""
import argparse, glob, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
p = argparse.ArgumentParser(); p.add_argument("--frames_dir", required=True); p.add_argument("--labels", required=True); p.add_argument("--kp", required=True); p.add_argument("--out", required=True)
p.add_argument("--win", type=int, default=25); p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=512); p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--w_aux", type=float, default=0.5); p.add_argument("--w_flight", type=float, default=8.0); p.add_argument("--hidden", type=int, default=128)
p.add_argument("--no_ego", action="store_true", help="ABLATION: zero the 9 camera ego-motion inputs (evaluate with BOX_DET_NO_EGO=1)"); p.add_argument("--gt_feats", type=int, default=0, help="DIAGNOSTIC (feature sets only): 1 = replace the 8 image-derived geometric features by their GROUND-TRUTH equivalents (projected u,v, depth, in-FOV flag, camera-frame X/Y/Z of the CURRENT frame); the ego-motion 9 stay. Tells whether the residual z error comes from the vision front-end or from the temporal/latent mapping itself."); p.add_argument("--zdim", type=int, default=0, help="latent dim of the labels; 0 = take it from the --kp ckpt"); p.add_argument("--feat_sets", default="", help="comma-separated feats_*.npz logged by per_throw_eval --save_feats (student data: 17-D features + z* + pos/vel + env/ep/tick); appended to the image sets")
p.add_argument("--zw", default="auto", help="per-dim task-aware z loss weights (npz from reflex.vision.latent_weights --zw_out: action change per 1-sigma z error). auto = <dir of --out>/zw_v10.npz if present; none = uniform")
p.add_argument("--feat_noise", type=float, default=0.02); p.add_argument("--dropout", type=float, default=0.0); p.add_argument("--wd", type=float, default=1e-4); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda:0")
a = p.parse_args(); dev = a.device; torch.manual_seed(a.seed); np.random.seed(a.seed)
from reflex.vision.detector_model import DetectorKPSeq, depth_to_channel
SETS = [] if a.frames_dir == "none" else list(zip(a.frames_dir.split(","), a.labels.split(",")))   # "none" = feature sets only (front-end CNN reused, no image rendering); several image sets (DAgger / multiple renders): env ids offset by 10000*k
kck = torch.load(a.kp, map_location="cpu", weights_only=False); ZD = int(a.zdim) if a.zdim > 0 else int(kck["zdim"])
model = DetectorKPSeq(a.win, ZD, 4, a.hidden, a.dropout).to(dev)
_kpsd = {(k[3:] if k.startswith("kp.") else k): v for k, v in kck["model"].items()}   # --kp may also be a kpseq detector ckpt (its CNN lives under "kp."): the front-end is reused, the GRU is trained fresh
missing = model.kp.load_state_dict({k: v for k, v in _kpsd.items() if k.startswith(("cnn.", "heat.", "ssm."))}, strict=False)
print(f"[seq] kp weights from {a.kp} (arch {kck.get('arch')}); not loaded (unused heads): {len(missing.missing_keys)} keys", flush=True)
# ---- pass 1: per-frame geometric features with the frozen CNN (chunk by chunk: RAM stays small)
GF, EGO, _Z, _POS, _VEL, _ENV, _EP, _TICK = [], [], [], [], [], [], [], []; t0 = time.time(); model.eval()
with torch.no_grad():
    for k, (fd, lb) in enumerate(SETS):
        files = sorted(glob.glob(os.path.join(fd, "chunk_*.npz"))); lab = np.load(os.path.join(fd, lb)); n_ = 0
        for f in files:
            ch = np.load(f); rgb = ch["rgb"]; dep = ch["depth"].view(np.int16); EGO.append(ch["ego"].astype(np.float32)); n_ += rgb.shape[0]
            for i in range(0, rgb.shape[0], 512):
                x = torch.from_numpy(rgb[i:i + 512]).to(dev).float().div_(255).permute(0, 3, 1, 2)
                d = depth_to_channel(torch.from_numpy(dep[i:i + 512]).to(dev).float() / 1000.0)[:, None]
                GF.append(model.frame_feats(torch.cat([x, d], 1)).cpu())
        print(f"[seq] set {k}: {fd} frames {n_} labels {lb}", flush=True)
        _Z.append(lab["z"][:n_]); _POS.append(lab["pos"][:n_]); _VEL.append(lab["vel"][:n_]); _ENV.append(lab["env"][:n_] + 10000 * k); _EP.append(lab["ep"][:n_]); _TICK.append(lab["tick"][:n_])
FEAT = np.concatenate([torch.cat(GF).numpy(), np.concatenate(EGO)], 1).astype(np.float32) if SETS else np.zeros((0, 17), np.float32)
_FS = [f for f in a.feat_sets.split(",") if f]
for k, f in enumerate(_FS):
    q = np.load(f); _ft = q["feat"].astype(np.float32).copy(); n_ = _ft.shape[0]
    if a.gt_feats:
        import math
        P_ = q["pos"].astype(np.float32); X_ = np.maximum(P_[:, 0], 1e-3); tan_ = math.tan(math.radians(53.0)); asp_ = 144.0 / 192.0
        un_ = -(P_[:, 1] / X_) / tan_; vn_ = -(P_[:, 2] / X_) / (tan_ * asp_); vis_ = ((P_[:, 0] > 0.25) & (np.abs(un_) < 1.0) & (np.abs(vn_) < 1.0)).astype(np.float32)
        g_ = np.stack([un_.clip(-1.5, 1.5), vn_.clip(-1.5, 1.5), X_.clip(0, 8), X_.clip(0, 8), vis_, P_[:, 0] / 4.0, P_[:, 1], P_[:, 2]], 1) * vis_[:, None]; g_[:, 4] = vis_
        _ft[:, :8] = g_
    FEAT = np.concatenate([FEAT, _ft])
    _Z.append(q["z"]); _POS.append(q["pos"]); _VEL.append(q["vel"]); _ENV.append(q["env"] + 1000000 * (k + 1)); _EP.append(q["ep"]); _TICK.append(q["tick"])
    print(f"[seq] feature set {k}: {f} frames {n_}", flush=True)
if a.no_ego: FEAT[:, 8:17] = 0.0; print('[seq] ABLATION: ego-motion inputs zeroed', flush=True)
N = FEAT.shape[0]
print(f"[seq] per-frame features {FEAT.shape} in {time.time()-t0:.0f}s", flush=True)
Z = np.concatenate(_Z); POS = np.concatenate(_POS); VEL = np.concatenate(_VEL); ENV = np.concatenate(_ENV); EP = np.concatenate(_EP); TICK = np.concatenate(_TICK)
key = ENV.astype(np.int64) * 100000 + EP.astype(np.int64); order = np.lexsort((TICK, key)); key_s = key[order]
starts = np.r_[0, np.where(np.diff(key_s) != 0)[0] + 1, len(key_s)]; windows = []
for s, e in zip(starts[:-1], starts[1:]):
    seq = order[s:e]
    for j in range(a.win - 1, len(seq)): windows.append(seq[j - a.win + 1: j + 1])
windows = np.stack(windows); ep_of = key[windows[:, -1]]; uniq = np.unique(ep_of); rng = np.random.RandomState(a.seed)
test_eps = set(rng.choice(uniq, size=max(1, int(0.2 * len(uniq))), replace=False).tolist()); is_te = np.array([k in test_eps for k in ep_of]); tr_w, te_w = windows[~is_te], windows[is_te]
print(f"[seq] windows train {len(tr_w)} test {len(te_w)} (episodes {len(uniq)})", flush=True)
fm, fs = FEAT[np.unique(tr_w)].mean(0), FEAT[np.unique(tr_w)].std(0) + 1e-6; model.f_mean.copy_(torch.from_numpy(fm)); model.f_std.copy_(torch.from_numpy(fs))
FEAT_t = torch.from_numpy(FEAT).to(dev); Zt = torch.from_numpy(Z).float().to(dev)
pm, ps = torch.tensor(POS.mean(0)), torch.tensor(POS.std(0) + 1e-6); vm, vs = torch.tensor(VEL.mean(0)), torch.tensor(VEL.std(0) + 1e-6)
AUXt = torch.cat([(torch.from_numpy(POS) - pm) / ps, (torch.from_numpy(VEL) - vm) / vs], 1).float().to(dev)
zsd = torch.from_numpy(Z.std(0) + 1e-6).float().to(dev)
_zwp = os.path.join(os.path.dirname(a.out), "zw_v10.npz") if a.zw == "auto" else (None if a.zw == "none" else a.zw)
if _zwp and os.path.exists(_zwp):
    zw = torch.from_numpy(np.load(_zwp)["w"]).float().to(dev); zw = zw / zw.mean()
    print(f"[seq] TASK-AWARE z weights from {_zwp}: {[round(float(x), 2) for x in zw]}", flush=True)
else:
    zw = torch.ones(Z.shape[1], device=dev); print("[seq] uniform z weights", flush=True)
DIST = np.linalg.norm(POS, axis=1); FL = torch.from_numpy((DIST >= 0.5) & (DIST <= 2.5)).to(dev)
params = list(model.gru.parameters()) + list(model.head_z.parameters()) + list(model.head_aux.parameters())
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd); steps = a.epochs * math.ceil(len(tr_w) / a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
tr_t, te_t = torch.from_numpy(tr_w).to(dev), torch.from_numpy(te_w).to(dev)
def evaluate():
    model.eval(); zs, au = [], []
    with torch.no_grad():
        for i in range(0, len(te_t), 4096):
            z, aux = model.seq_forward(FEAT_t[te_t[i:i + 4096]]); zs.append(z); au.append(aux)
    model.train(); z = torch.cat(zs); aux = torch.cat(au).cpu(); t = te_t[:, -1]; d2 = ((z - Zt[t]) / zsd) ** 2; fl = FL[t]; d2w = d2 * zw
    pp = aux[:, :3] * ps + pm; vp = aux[:, 3:] * vs + vm; pt = torch.from_numpy(POS)[t.cpu()]; vt = torch.from_numpy(VEL)[t.cpu()]; flc = fl.cpu()
    return dict(z_mse_std=round(float(d2.mean()), 4), flt_z_mse_std=round(float(d2[fl].mean()), 4), flt_zw_mse=round(float(d2w[fl].mean()), 4), flt_z_dim0_5=[round(float(x), 3) for x in d2[fl][:, :6].mean(0)], flt_n=int(fl.sum()),
                flt_pos_rmse=[round(float(x), 3) for x in ((pp[flc] - pt[flc]) ** 2).mean(0).sqrt()], flt_vel_rmse=[round(float(x), 3) for x in ((vp[flc] - vt[flc]) ** 2).mean(0).sqrt()])
model.train()   # frame-feature extraction above ran in eval mode; cudnn GRU backward needs training mode
fs_t = torch.from_numpy(fs).to(dev); best = None; t0 = time.time()
for ep in range(a.epochs):
    perm = torch.randperm(len(tr_t), device=dev); tot = 0.0; nb = 0
    for i in range(0, len(perm), a.bs):
        w = tr_t[perm[i:i + a.bs]]; t = w[:, -1]; x = FEAT_t[w]
        if a.feat_noise > 0: x = x + a.feat_noise * fs_t * torch.randn_like(x)
        z, aux = model.seq_forward(x); sw = torch.where(FL[t], a.w_flight, 1.0).float(); sw = sw / sw.mean()
        loss = (((((z - Zt[t]) / zsd) ** 2) * zw).mean(-1) * sw).mean() + a.w_aux * (((aux - AUXt[t]) ** 2).mean(-1) * sw).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); tot += float(loss); nb += 1
    if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
        ev = evaluate(); print(f"[seq] epoch {ep+1}/{a.epochs} loss {tot/nb:.4f}  test {json.dumps(ev)}  {time.time()-t0:.0f}s", flush=True)
        if best is None or ev["flt_zw_mse"] < best["flt_zw_mse"]:
            best = dict(ev, epoch=ep + 1)
            torch.save({"model": model.state_dict(), "arch": "kpseq", "in_ch": 4, "win": a.win, "zdim": ZD, "hidden": a.hidden, "dropout": a.dropout, "eval": best, "kp_src": a.kp,
                        "aux_norm": dict(pmean=pm.tolist(), pstd=ps.tolist(), vmean=vm.tolist(), vstd=vs.tolist())}, a.out)
print(f"[seq] DONE best {json.dumps(best)} -> {a.out}", flush=True)
print(f"[det] DONE best {json.dumps(best)} -> {a.out}", flush=True)   # same completion marker as train_detector.py, for scripts that wait on it
