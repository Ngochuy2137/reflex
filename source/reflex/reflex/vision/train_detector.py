"""Stage 3, step 1: train an image-history (WIN frames) -> z model by regressing the z* labels.

For the released visual encoder this trains the per-frame RGB-D CNN + supervised box heatmap (--depth --arch kp,
--w_uv > 0, colour augmentation); train_seqdet.py then trains the temporal GRU on its frozen per-frame features.
The other architectures (detector_model.ARCHS) regress z directly and are kept for ablations.

Data: chunk_*.npz from render_frames.py + labels_<tag>.npz from make_labels.py (same frames_dir); comma-separated
lists of both are concatenated (DAgger aggregation). The heatmap labels of the kp archs are the projected box
position, so they need camera-frame labels (BOX_LATENT_PRIV_FRAME=camera at render time).
Windows: WIN consecutive ticks of the same (env, episode) sequence; target = z* at the last tick.
Model: shared per-frame CNN (192x144 input) -> temporal 1-D conv over WIN -> heads: z (tanh, ZDIM) + aux pos/vel (6).
Losses: MSE(z, z*) on live dims (standardised per dim) + aux MSE + denoising consistency (MSE between z of the clean
and of the augmented window) + KL(batch z || N(0, 1)), default weights 1 / 0.5 / 0.1 / 0.1, plus the heatmap BCE
(--w_uv) for the kp archs; flight windows can be up-weighted (--w_flight).
Split: 20 % of the (env, episode) sequences held out. Reports per-dim R2 of z and pos/vel R2 of the aux head.

Usage (recipe of the released front end):
  python -m reflex.vision.train_detector --frames_dir data/frames/<tag> --labels labels_<tag>.npz --out outputs/vision/det_kp.pt
      --epochs 20 --bs 32 --w_kl 0 --w_flight 8 --flight_select --arch kp --depth --w_uv 1.0 --color_aug 0.7
"""
import argparse, glob, json, math, os, time
import sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
p = argparse.ArgumentParser()
p.add_argument("--frames_dir", required=True); p.add_argument("--labels", required=True); p.add_argument("--out", required=True)
p.add_argument("--win", type=int, default=25); p.add_argument("--stride", type=int, default=2); p.add_argument("--epochs", type=int, default=15)
p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--max_chunks", type=int, default=0)
p.add_argument("--w_aux", type=float, default=0.5); p.add_argument("--w_den", type=float, default=0.1); p.add_argument("--w_kl", type=float, default=0.1)
p.add_argument("--device", default="cuda:0"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--depth", action="store_true", help="RGB-D: use the depth key of the chunks (uint16 mm) as a 4th input channel (clamp(m/8,0,1))")
p.add_argument("--w_uv", type=float, default=0.0, help="arch kp: weight of the box-heatmap loss (BCE vs Gaussian at the projected box pixel; camera-frame labels + pinhole fov_h 53 deg)")
p.add_argument("--fov_h", type=float, default=53.0); p.add_argument("--uv_sigma", type=float, default=0.08, help="heatmap target sigma in normalised image units")
p.add_argument("--arch", default="base", choices=["base", "ssm", "ssmd", "kp", "kp4", "kpm"], help="detector_model.ARCHS: base = avg-pool CNN + temporal conv; ssm = spatial-softmax keypoints + frame differences")
p.add_argument("--w_flight", type=float, default=1.0, help="loss weight for windows whose target frame has the box 0.5-2.5 m from the pelvis (flight); 1 = off")
p.add_argument("--flight_select", action="store_true", help="pick the best epoch by flight-window standardised z MSE instead of the global one")
p.add_argument("--w_dead", type=float, default=0.1)
p.add_argument("--color_aug", type=float, default=0.0, help="colour augmentation: with this probability per window, recolour the BOX pixels (chroma-keyed: the cardboard is the only saturated object in the renders) to a random RGB (25 %% dark, 25 %% grey, 50 %% random hue), keeping the per-pixel shading. RGB only; depth untouched.")
a = p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(a.device)

# ---------------- data ----------------
# --frames_dir and --labels accept comma-separated lists (DAgger aggregation): sets are concatenated with an
# env-id offset of 10000*k so (env, ep) keys never collide across sets; the held-out split is drawn over the union.
# Memory: np.concatenate of per-chunk lists transiently holds two copies (lists + result, ~33 GB for 200k RGB frames),
# so this uses two passes instead: count frames, preallocate once, fill per chunk.
_sets = []
for k, (fd, lb) in enumerate(zip(a.frames_dir.split(","), a.labels.split(","))):
    files = sorted(glob.glob(os.path.join(fd, "chunk_*.npz")))
    if a.max_chunks: files = files[: a.max_chunks]
    _sets.append((k, fd, lb, files))
_first = np.load(_sets[0][3][0]); _H, _W = _first["rgb"].shape[1:3]
_counts = [[int(np.load(f)["ep"].shape[0]) for f in files] for (_, _, _, files) in _sets]; N = int(sum(sum(c) for c in _counts))
RGB = np.empty((N, _H, _W, 3), dtype=np.uint8); DEP = np.empty((N, _H, _W), dtype=np.uint16) if a.depth else None
_Z, _POS, _VEL, _ENV, _EP, _TICK = [], [], [], [], [], []; _o = 0
for (k, fd, lb, files), cs in zip(_sets, _counts):
    lab = np.load(os.path.join(fd, lb)); n_ = int(sum(cs)); _o0 = _o
    for f, c in zip(files, cs):
        ch = np.load(f); RGB[_o:_o + c] = ch["rgb"]
        if a.depth: DEP[_o:_o + c] = ch["depth"]
        _o += c; del ch
    print(f"[det] set {k}: {fd} frames {n_} labels {lb}", flush=True)
    _Z.append(lab["z"][:n_]); _POS.append(lab["pos"][:n_]); _VEL.append(lab["vel"][:n_]); _ENV.append(lab["env"][:n_] + 10000 * k); _EP.append(lab["ep"][:n_]); _TICK.append(lab["tick"][:n_])
Z = np.concatenate(_Z); POS = np.concatenate(_POS); VEL = np.concatenate(_VEL); ENV = np.concatenate(_ENV); EP = np.concatenate(_EP); TICK = np.concatenate(_TICK)
live = Z.std(0) > 1e-6; ZD = Z.shape[1]
print(f"[det] frames {N} {RGB.shape[1:]}  z dims live {int(live.sum())}/{ZD}", flush=True)
# windows: rows are stored step-major (all envs at tick t, then t+1 ...); build per-(env,ep) tick-ordered index lists
key = ENV.astype(np.int64) * 100000 + EP.astype(np.int64)
order = np.lexsort((TICK, key)); key_s = key[order]
starts = np.r_[0, np.where(np.diff(key_s) != 0)[0] + 1, len(key_s)]
windows = []   # arrays of WIN global row indices (tick-ordered), last = target row
for s, e in zip(starts[:-1], starts[1:]):
    seq = order[s:e]
    for j in range(a.win - 1, len(seq), a.stride):
        windows.append(seq[j - a.win + 1: j + 1])
windows = np.stack(windows); ep_of_win = key[windows[:, -1]]
uniq = np.unique(ep_of_win); rng = np.random.RandomState(a.seed); test_eps = set(rng.choice(uniq, size=max(1, int(0.2 * len(uniq))), replace=False).tolist())
is_test = np.array([k in test_eps for k in ep_of_win]); tr_w, te_w = windows[~is_test], windows[is_test]
print(f"[det] windows train {len(tr_w)} test {len(te_w)} (episodes {len(uniq)}, test {len(test_eps)})", flush=True)
RGB_t = torch.from_numpy(RGB)   # keep uint8 on CPU; move per batch
if a.arch in ("kp", "kp4", "kpm"):
    _tan = math.tan(math.radians(a.fov_h)); _asp = RGB.shape[1] / RGB.shape[2]
    _xf = np.maximum(POS[:, 0], 1e-3); UVN = np.stack([-(POS[:, 1] / _xf) / _tan, -(POS[:, 2] / _xf) / (_tan * _asp)], 1).astype(np.float32)   # right, down in [-1,1]
    VISG = ((POS[:, 0] > 0.2) & (np.abs(UVN[:, 0]) < 1) & (np.abs(UVN[:, 1]) < 1)).astype(np.float32)
    UVN_t = torch.from_numpy(UVN); VISG_t = torch.from_numpy(VISG)
    print(f"[det] kp: projected-pixel labels, geometrically visible {VISG.mean()*100:.1f} % of frames", flush=True)
DEP_t = torch.from_numpy(DEP.view(np.int16)) if a.depth else None   # uint16 mm reinterpreted as int16 (max 20000 < 32767): zero-copy, 2 B/px
IN_CH = 4 if a.depth else 3
from reflex.vision.detector_model import depth_to_channel
zmean = torch.tensor(POS.mean(0)); zstd = torch.tensor(POS.std(0) + 1e-6); vmean = torch.tensor(VEL.mean(0)); vstd = torch.tensor(VEL.std(0) + 1e-6)

def batch(idx_w, aug):
    x = RGB_t[idx_w.reshape(-1)].to(dev, non_blocking=True).float() / 255.0     # (B*W,H,W,3)
    x = x.permute(0, 3, 1, 2)                                                    # (B*W,3,H,W)
    if a.depth:
        dch = depth_to_channel(DEP_t[idx_w.reshape(-1)].to(dev, non_blocking=True).float() / 1000.0)[:, None]   # (B*W,1,H,W)
        if aug: dch = (dch + 0.003 * torch.randn_like(dch)).clamp(0, 1)          # ~2.4 cm depth noise
        x = torch.cat([x, dch], 1)
    if aug and a.color_aug > 0:  # box recolour (same colour for all frames of a window)
        B = idx_w.shape[0]; rgb = x[:, :3]
        key = ((rgb[:, 0] - rgb[:, 2]) > 0.10) & (rgb[:, 0] > rgb[:, 1]) & (rgb[:, 0] > 0.12)          # (B*W,H,W) cardboard pixels
        kind = torch.rand(B, device=dev); col = torch.rand(B, 3, device=dev)
        col = torch.where((kind < 0.25)[:, None], col * 0.15 + 0.03, col)                                # dark
        col = torch.where(((kind >= 0.25) & (kind < 0.5))[:, None], col[:, :1].repeat(1, 3), col)          # grey
        do = (torch.rand(B, device=dev) < a.color_aug).float()[:, None]; col = col.repeat_interleave(a.win, 0); do = do.repeat_interleave(a.win, 0)
        v = rgb.max(1, keepdim=True)[0]; km = key[:, None].float(); vref = ((v * km).sum((2, 3), keepdim=True) / km.sum((2, 3), keepdim=True).clamp(min=1)).clamp(min=0.05)
        new = (col[:, :, None, None] * (v / vref)).clamp(0, 1)
        m_ = km * do[:, :, None, None]; x = torch.cat([rgb * (1 - m_) + new * m_, x[:, 3:]], 1)
    if aug:  # photometric + gaussian noise + small shift (same per window)
        B = idx_w.shape[0]; g = (0.7 + 0.6 * torch.rand(B, 1, 1, 1, device=dev)).repeat_interleave(a.win, 0)
        b = (0.1 * (torch.rand(B, 1, 1, 1, device=dev) - 0.5)).repeat_interleave(a.win, 0)
        if a.depth: x = torch.cat([(x[:, :3] * g + b + 0.02 * torch.randn_like(x[:, :3])).clamp(0, 1), x[:, 3:]], 1)
        else: x = (x * g + b + 0.02 * torch.randn_like(x)).clamp(0, 1)
        dx, dy = np.random.randint(-4, 5), np.random.randint(-3, 4); x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    return x.view(idx_w.shape[0], a.win, IN_CH, x.shape[-2], x.shape[-1])

# ---------------- model ----------------
from reflex.vision.detector_model import build_detector
model = build_detector(a.arch, a.win, ZD, IN_CH).to(dev); print(f"[det] arch {a.arch} in_ch {IN_CH}  params {sum(p.numel() for p in model.parameters())/1e6:.2f} M", flush=True); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
steps = a.epochs * math.ceil(len(tr_w) / a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
Zt = torch.from_numpy(Z).float(); POSt = torch.from_numpy(POS).float(); VELt = torch.from_numpy(VEL).float(); livem = torch.from_numpy(live).float().to(dev)
# loss weights per z dim: live dims standardised by their std (so small-variance dims are not drowned by dim 0),
# dead dims get a weak pull to their constant target so they do not drift (decoder ignores them anyway).
zsd = torch.from_numpy(np.where(live, Z.std(0), 1.0)).float().to(dev)
DIST = np.linalg.norm(POS, axis=1); FLIGHT = torch.from_numpy((DIST >= 0.5) & (DIST <= 2.5))   # box 0.5-2.5 m away: the ~12 % of frames where the catch happens
print(f"[det] flight frames (0.5-2.5 m) {float(FLIGHT.float().mean())*100:.1f} %  w_flight {a.w_flight}  select {'flight' if a.flight_select else 'global'}", flush=True)
wdim = torch.where(livem > 0, 1.0 / (zsd ** 2), torch.full_like(livem, a.w_dead))

def evaluate(idx_all):
    model.eval(); zs, zt, ps, pt, vs, vt = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(idx_all), a.bs):
            w = idx_all[i:i + a.bs]; _o = model(batch(w, False)); z, aux = _o[0], _o[1]; t = w[:, -1]
            zs.append(z.cpu()); zt.append(Zt[t]); ps.append(aux[:, :3].cpu() * zstd + zmean); pt.append(POSt[t]); vs.append(aux[:, 3:].cpu() * vstd + vmean); vt.append(VELt[t])
    model.train()
    def r2(pr, gt): pr, gt = torch.cat(pr), torch.cat(gt); return (1 - ((gt - pr) ** 2).sum(0) / ((gt - gt.mean(0)) ** 2).sum(0).clamp(min=1e-9)).tolist()
    zr = r2(zs, zt); pr = r2(ps, pt); vr = r2(vs, vt); d2 = (torch.cat(zs) - torch.cat(zt)) ** 2
    zmse = float(d2[:, torch.from_numpy(live)].mean()); zmse_std = float((d2[:, torch.from_numpy(live)] / (zsd.cpu()[torch.from_numpy(live)] ** 2)).mean())
    fl = FLIGHT[idx_all[:, -1]]; PT, PR, VT, VR = torch.cat(pt), torch.cat(ps), torch.cat(vt), torch.cat(vs)
    flt_pos = ((PR[fl] - PT[fl]) ** 2).mean(0).sqrt().tolist(); flt_vel = ((VR[fl] - VT[fl]) ** 2).mean(0).sqrt().tolist()
    flt_z = float((d2[fl][:, torch.from_numpy(live)] / (zsd.cpu()[torch.from_numpy(live)] ** 2)).mean())
    return dict(z_r2_live=[round(x, 3) for x, l in zip(zr, live) if l], z_mse=round(zmse, 5), z_mse_std=round(zmse_std, 4), pos_r2=[round(x, 3) for x in pr], vel_r2=[round(x, 3) for x in vr],
                flt_n=int(fl.sum()), flt_pos_rmse=[round(x, 3) for x in flt_pos], flt_vel_rmse=[round(x, 3) for x in flt_vel], flt_z_mse_std=round(flt_z, 4))

t0 = time.time(); best = None
for ep in range(a.epochs):
    perm = np.random.permutation(len(tr_w)); tot = 0.0; nb = 0
    for i in range(0, len(perm), a.bs):
        w = tr_w[perm[i:i + a.bs]]; t = w[:, -1]
        x_clean = batch(w, False); x_aug = batch(w, True)
        with torch.no_grad():
            z_c = model(x_clean)[0]          # denoise target only: no graph (halves activation memory)
        _out = model(x_aug); z_a, aux_a = _out[0], _out[1]
        zt = Zt[t].to(dev); pt = ((POSt[t] - zmean) / zstd).to(dev); vt = ((VELt[t] - vmean) / vstd).to(dev)
        sw = torch.where(FLIGHT[t], a.w_flight, 1.0).float().to(dev); sw = sw / sw.mean()          # per-sample weight, mean 1
        l_z = ((((z_a - zt) ** 2) * wdim).sum(-1) * sw).mean() / wdim.sum()
        l_aux = (((aux_a - torch.cat([pt, vt], -1)) ** 2).mean(-1) * sw).mean(); l_den = F.mse_loss(z_a, z_c.detach())
        mu, var = z_a.mean(0), z_a.var(0) + 1e-6; l_kl = 0.5 * (var + mu ** 2 - 1 - var.log())[live].mean()
        l_uv = torch.zeros((), device=dev)
        if a.arch in ("kp", "kp4", "kpm") and a.w_uv > 0:
            hl = _out[2]; Bh, Hh, Wh = hl.shape; wt = w.reshape(-1)             # heatmap for every frame of every window
            uv = UVN_t[wt].to(dev); vg = VISG_t[wt].to(dev)
            gy = torch.linspace(-1, 1, Hh, device=dev)[None, :, None]; gx = torch.linspace(-1, 1, Wh, device=dev)[None, None, :]
            tgt = torch.exp(-(((gx - uv[:, 0, None, None]) / a.uv_sigma) ** 2 + ((gy - uv[:, 1, None, None]) / a.uv_sigma) ** 2) / 2) * vg[:, None, None]
            l_uv = F.binary_cross_entropy_with_logits(hl, tgt, pos_weight=torch.tensor(20.0, device=dev))
        loss = l_z + a.w_aux * l_aux + a.w_den * l_den + a.w_kl * l_kl + a.w_uv * l_uv
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += float(loss); nb += 1
    ev = evaluate(te_w); print(f"[det] epoch {ep+1}/{a.epochs} loss {tot/max(nb,1):.4f}  test {json.dumps(ev)}  {time.time()-t0:.0f}s", flush=True)
    key = "flt_z_mse_std" if a.flight_select else "z_mse_std"
    if best is None or ev[key] < best[key]:
        best = dict(ev, epoch=ep + 1); torch.save({"model": model.state_dict(), "arch": a.arch, "in_ch": IN_CH, "win": a.win, "zdim": ZD, "live": live.tolist(), "eval": best,
                                                  "aux_norm": dict(pmean=zmean.tolist(), pstd=zstd.tolist(), vmean=vmean.tolist(), vstd=vstd.tolist())}, a.out)
print(f"[det] DONE best {json.dumps(best)} -> {a.out}", flush=True)
