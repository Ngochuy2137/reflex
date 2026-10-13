"""Stage 3 visual encoder networks ("detector" in the code).

The released visual encoder is DetectorKPSeq (arch 'kpseq'): a per-frame RGB-D CNN with a supervised box heatmap
(trained with train_detector.py, arch 'kp' or 'kpm') followed by a GRU over the frame history and the camera
ego-motion (trained with train_seqdet.py). The other architectures regress z directly with a CNN + temporal conv and
are kept for ablations. Shared by the trainers, the in-the-loop wrapper (detector_obs_wrapper.py) and the evaluator
(reflex.eval.per_throw_eval).
"""
import torch, torch.nn as nn, torch.nn.functional as F


DEPTH_NORM_M = 8.0   # depth channel = clamp(metres / 8, 0, 1); shared by every consumer of rendered depth (wrapper, evaluator, trainers)


def depth_to_channel(d_m: torch.Tensor) -> torch.Tensor:
    return (d_m / DEPTH_NORM_M).clamp(0.0, 1.0)


class Detector(nn.Module):
    """arch 'base': average-pooled per-frame CNN + temporal 1-D conv.
    Image history (B, WIN, C, H, W) in [0, 1] (C = 3 for RGB, 4 for RGB-D) -> (z = tanh(.) (B, ZDIM), aux (B, 6) = normalised box pos/vel)."""
    def __init__(self, win: int, zdim: int, in_ch: int = 3):
        super().__init__()
        self.in_ch = in_ch
        self.enc = nn.Sequential(nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(),
                                 nn.Conv2d(64, 96, 3, 2, 1), nn.ELU(), nn.Conv2d(96, 128, 3, 2, 1), nn.ELU(),
                                 nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Linear(128 * 12, 128), nn.ELU())
        self.temporal = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.ELU(), nn.Conv1d(128, 128, 5, padding=2), nn.ELU())
        self.head_z = nn.Linear(128 * 2, zdim); self.head_aux = nn.Linear(128 * 2, 6); self.win = win

    def forward(self, x):
        B, W = x.shape[:2]; f = self.enc(x.flatten(0, 1)).view(B, W, -1).transpose(1, 2)
        h = self.temporal(f); feat = torch.cat([h[:, :, -1], h.mean(-1)], -1)
        return torch.tanh(self.head_z(feat)), self.head_aux(feat)


class SpatialSoftmax(nn.Module):
    """per-channel soft-argmax over the feature map -> (B, C*3): expected x, expected y (in [-1,1]) and peak activation.
    Keeps the sub-cell localisation that AdaptiveAvgPool2d((3, 4)) discards (with average pooling only, the box position
    error during flight was 0.25-0.5 m)."""
    def __init__(self, temperature: float = 1.0):
        super().__init__(); self.log_t = nn.Parameter(torch.tensor(float(temperature)).log())
    def forward(self, f):
        B, C, H, W = f.shape
        ys = torch.linspace(-1, 1, H, device=f.device); xs = torch.linspace(-1, 1, W, device=f.device)
        a = torch.softmax((f / self.log_t.exp()).flatten(2), -1).view(B, C, H, W)
        ex = (a.sum(2) * xs).sum(-1); ey = (a.sum(3) * ys).sum(-1); pk = f.flatten(2).amax(-1)
        return torch.cat([ex, ey, pk], 1)


class DetectorSSM(nn.Module):
    """arch 'ssm': per-frame CNN -> spatial-softmax keypoints (64 ch x [x, y, peak]) + global pooled features; frame-to-frame
    feature differences are concatenated (an explicit motion signal for velocity) before the temporal conv.
    Same I/O contract as Detector: (B, WIN, C, H, W) in [0, 1] -> (z = tanh(.) (B, ZDIM), aux (B, 6))."""
    def __init__(self, win: int, zdim: int, in_ch: int = 3):
        super().__init__()
        self.in_ch = in_ch
        self.cnn = nn.Sequential(nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(),
                                 nn.Conv2d(64, 96, 3, 2, 1), nn.ELU(), nn.Conv2d(96, 64, 3, 1, 1), nn.ELU())   # 192x144 -> 24x18, 64 ch
        self.ssm = SpatialSoftmax(); self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(64 * 3 + 64, 128), nn.ELU())
        self.temporal = nn.Sequential(nn.Conv1d(256, 128, 5, padding=2), nn.ELU(), nn.Conv1d(128, 128, 5, padding=2), nn.ELU())
        self.head_z = nn.Linear(128 * 2, zdim); self.head_aux = nn.Linear(128 * 2, 6); self.win = win
    def forward(self, x):
        B, W = x.shape[:2]; fm = self.cnn(x.flatten(0, 1))
        f = self.proj(torch.cat([self.ssm(fm), self.gap(fm).flatten(1)], 1)).view(B, W, -1)          # (B, W, 128)
        d = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], 1)                          # frame-to-frame differences
        h = self.temporal(torch.cat([f, d], -1).transpose(1, 2)); feat = torch.cat([h[:, :, -1], h.mean(-1)], -1)
        return torch.tanh(self.head_z(feat)), self.head_aux(feat)


class DetectorSSMD(DetectorSSM):
    """arch 'ssmd': RGB-D-aware keypoints. Besides the expected (x, y, peak) per feature channel, the spatial-softmax
    attention of each channel also reads out the expected depth at its keypoint from the (downsampled) depth channel,
    E_att[depth]. This turns 'where is the box in the image' into 'where is the box in 3D' without asking the CNN to
    regress metric depth through pooling. Requires in_ch == 4 (depth as channel 3, normalised metres / 8)."""
    def __init__(self, win: int, zdim: int, in_ch: int = 4):
        assert in_ch == 4, "ssmd needs RGB-D input"
        super().__init__(win, zdim, in_ch)
        self.proj = nn.Sequential(nn.Linear(64 * 4 + 64, 128), nn.ELU())      # + 64 depth readouts
    def forward(self, x):
        B, W = x.shape[:2]; xf = x.flatten(0, 1); fm = self.cnn(xf)              # (B*W, 64, h, w)
        dmap = -F.adaptive_max_pool2d(-xf[:, 3:4], fm.shape[-2:])                # min-pooled depth per feature cell: a 17-px box
        # covers ~2 cells of the 24x18 grid and area-averaging would swamp it with the 20 m background; the nearest surface wins
        Bc, C, H, Wd = fm.shape
        a = torch.softmax((fm / self.ssm.log_t.exp()).flatten(2), -1)           # (B*W, C, H*W)
        ed = (a * dmap.flatten(2)).sum(-1)                                       # expected depth under each channel's attention
        f = self.proj(torch.cat([self.ssm(fm), ed, self.gap(fm).flatten(1)], 1)).view(B, W, -1)
        d = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], 1)
        h = self.temporal(torch.cat([f, d], -1).transpose(1, 2)); feat = torch.cat([h[:, :, -1], h.mean(-1)], -1)
        return torch.tanh(self.head_z(feat)), self.head_aux(feat)


class DetectorKP(DetectorSSMD):
    """arch 'kp': explicit box localisation. A 1-channel heatmap head on the feature map is supervised with the projected box
    pixel (train_detector.py --w_uv); its softmax gives the box's expected (u, v), its min-pooled expected depth and a
    visibility logit -> a 4-D geometric readout per frame, concatenated to the keypoint features before the temporal conv.
    forward() returns a third output (heatmap logits (B*W, h, w)) that train_detector.py uses for the loss; in-the-loop
    consumers unpack only the first two."""
    def __init__(self, win: int, zdim: int, in_ch: int = 4):
        super().__init__(win, zdim, in_ch)
        self.heat = nn.Conv2d(64, 1, 3, 1, 1)
        self.proj = nn.Sequential(nn.Linear(64 * 4 + 64 + 4, 128), nn.ELU())
    def forward(self, x):
        B, W = x.shape[:2]; xf = x.flatten(0, 1); fm = self.cnn(xf)
        dmap = -F.adaptive_max_pool2d(-xf[:, 3:4], fm.shape[-2:])
        Bc, C, H, Wd = fm.shape
        a = torch.softmax((fm / self.ssm.log_t.exp()).flatten(2), -1); ed = (a * dmap.flatten(2)).sum(-1)
        hl = self.heat(fm)[:, 0]                                                 # (B*W, h, w) logits
        ha = torch.softmax(hl.flatten(1), -1)                                    # box attention
        ys = torch.linspace(-1, 1, H, device=x.device); xs = torch.linspace(-1, 1, Wd, device=x.device)
        hu = (ha.view(Bc, H, Wd).sum(1) * xs).sum(-1); hv = (ha.view(Bc, H, Wd).sum(2) * ys).sum(-1)
        hd = (ha * dmap.flatten(2)[:, 0]).sum(-1); hvis = hl.flatten(1).amax(-1)
        geo = torch.stack([hu, hv, hd, hvis], 1)
        f = self.proj(torch.cat([self.ssm(fm), ed, self.gap(fm).flatten(1), geo], 1)).view(B, W, -1)
        d = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], 1)
        h = self.temporal(torch.cat([f, d], -1).transpose(1, 2)); feat = torch.cat([h[:, :, -1], h.mean(-1)], -1)
        return torch.tanh(self.head_z(feat)), self.head_aux(feat), hl


class DetectorKP4(DetectorKP):
    """arch 'kp4': same as 'kp' but the backbone downsamples 4x instead of 8x (192x144 -> 48x36 feature grid; one cell =
    4 px = 2.2 deg, ~7.5 cm at 2 m). Ablation on the resolution of the localisation grid."""
    def __init__(self, win: int, zdim: int, in_ch: int = 4):
        super().__init__(win, zdim, in_ch)
        self.cnn = nn.Sequential(nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(),
                                 nn.Conv2d(64, 96, 3, 1, 1), nn.ELU(), nn.Conv2d(96, 64, 3, 1, 1), nn.ELU())   # 192x144 -> 48x36


class DetectorKPM(DetectorKP):
    """arch 'kpm': 'kp' + a geometric depth readout and an explicit 3D lift.
    The supervised heatmap localises the box to 2-5 px (3-8 cm), but the attention-weighted expected depth is off by
    0.2-1 m (softmax mass on the 8 m background; front face vs centre). Here the depth is the minimum full-resolution depth
    in a window around the soft-argmax pixel (the box is the nearest surface there; two window sizes, 15 and 25 px), and
    the camera-frame 3D point (X, Y, Z) is computed from (u, v, d) with the pinhole model, so velocity becomes a difference
    of accurate 3D points. Per-frame geometric readout = [u, v, d15, d25, vis_logit, X, Y, Z] (8-D)."""
    TAN_H = 1.3270448216204098   # tan(53 deg): half horizontal FOV of the rendered camera
    def __init__(self, win: int, zdim: int, in_ch: int = 4):
        super().__init__(win, zdim, in_ch)
        self.proj = nn.Sequential(nn.Linear(64 * 4 + 64 + 8, 128), nn.ELU())
    def forward(self, x):
        B, W = x.shape[:2]; xf = x.flatten(0, 1); fm = self.cnn(xf); Hi, Wi = xf.shape[-2:]
        dmap = -F.adaptive_max_pool2d(-xf[:, 3:4], fm.shape[-2:]); Bc, C, H, Wd = fm.shape
        a = torch.softmax((fm / self.ssm.log_t.exp()).flatten(2), -1); ed = (a * dmap.flatten(2)).sum(-1)
        hl = self.heat(fm)[:, 0]; ha = torch.softmax(hl.flatten(1), -1).view(Bc, H, Wd)
        ys = torch.linspace(-1, 1, H, device=x.device); xs = torch.linspace(-1, 1, Wd, device=x.device)
        hu = (ha.sum(1) * xs).sum(-1); hv = (ha.sum(2) * ys).sum(-1); hvis = hl.flatten(1).amax(-1)
        g = torch.stack([hu, hv], -1)[:, None, None, :].detach()
        d15 = F.grid_sample(-F.max_pool2d(-xf[:, 3:4], 15, 1, 7), g, mode="nearest", align_corners=True)[:, 0, 0, 0]
        d25 = F.grid_sample(-F.max_pool2d(-xf[:, 3:4], 25, 1, 12), g, mode="nearest", align_corners=True)[:, 0, 0, 0]
        X = d15 * 8.0; Y = -hu * self.TAN_H * X; Z = -hv * self.TAN_H * (Hi / Wi) * X           # metres, camera frame (front face)
        geo = torch.stack([hu, hv, d15, d25, hvis, X / 4.0, Y, Z], 1)
        f = self.proj(torch.cat([self.ssm(fm), ed, self.gap(fm).flatten(1), geo], 1)).view(B, W, -1)
        d = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], 1)
        h = self.temporal(torch.cat([f, d], -1).transpose(1, 2)); feat = torch.cat([h[:, :, -1], h.mean(-1)], -1)
        return torch.tanh(self.head_z(feat)), self.head_aux(feat), hl


class DetectorKPSeq(nn.Module):
    """arch 'kpseq': the Stage 3 visual encoder.
    Image history (+ per-frame camera ego-motion, 9-D, IMU-like) -> z_hat, with no hand-written tracking logic:
      per frame : RGB-D CNN + supervised box heatmap -> soft-argmax (u, v), local-min depth, visibility logit, pinhole lift
                  to the camera-frame 3D point (fixed geometric layer, 8-D)                        [frame_feats]
      temporal  : GRU over WIN frames of [geo 8, ego 9] -> z (tanh) + aux pos/vel                  [seq_forward]
    Which frame is the latest one with the box visible, how long to hold an estimate when the box is out of view, and how
    to infer velocity are learned from the z* targets. Trained in two steps: the CNN + heatmap with train_detector.py,
    then the GRU on the frozen per-frame features with train_seqdet.py."""
    GEO, EGO = 8, 9
    def __init__(self, win: int, zdim: int, in_ch: int = 4, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.win, self.in_ch = win, in_ch
        self.kp = DetectorKPM(win, zdim, in_ch)                      # only its cnn / heat are used
        self.register_buffer("f_mean", torch.zeros(self.GEO + self.EGO)); self.register_buffer("f_std", torch.ones(self.GEO + self.EGO))
        self.gru = nn.GRU(self.GEO + self.EGO, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head_z = nn.Linear(hidden, zdim); self.head_aux = nn.Linear(hidden, 6)
    def frame_feats(self, xf):                                       # xf (N, 4, H, W) in [0,1] -> (N, 8)
        k = self.kp; fm = k.cnn(xf); Hi, Wi = xf.shape[-2:]; hl = k.heat(fm)[:, 0]; N, H, Wd = hl.shape
        ha = torch.softmax(hl.flatten(1), -1).view(N, H, Wd)
        hu = (ha.sum(1) * torch.linspace(-1, 1, Wd, device=xf.device)).sum(-1); hv = (ha.sum(2) * torch.linspace(-1, 1, H, device=xf.device)).sum(-1)
        g = torch.stack([hu, hv], -1)[:, None, None, :].detach()
        d15 = F.grid_sample(-F.max_pool2d(-xf[:, 3:4], 15, 1, 7), g, mode="nearest", align_corners=True)[:, 0, 0, 0]
        d25 = F.grid_sample(-F.max_pool2d(-xf[:, 3:4], 25, 1, 12), g, mode="nearest", align_corners=True)[:, 0, 0, 0]
        X = d15 * 8.0; Y = -hu * k.TAN_H * X; Z = -hv * k.TAN_H * (Hi / Wi) * X
        return torch.stack([hu, hv, d15, d25, hl.flatten(1).amax(-1) / 5.0, X / 4.0, Y, Z], 1)
    def seq_forward(self, feats):                                    # feats (B, W, 17) raw -> z, aux
        h, _ = self.gru((feats - self.f_mean) / self.f_std); h = h[:, -1]
        return torch.tanh(self.head_z(h)), self.head_aux(h)
    def forward(self, x, ego):                                       # x (B, W, 4, H, Wd), ego (B, W, 9)
        B, W = x.shape[:2]; gf = self.frame_feats(x.flatten(0, 1)).view(B, W, -1)
        return self.seq_forward(torch.cat([gf, ego], -1))


ARCHS = {"base": Detector, "ssm": DetectorSSM, "ssmd": DetectorSSMD, "kp": DetectorKP, "kp4": DetectorKP4, "kpm": DetectorKPM, "kpseq": DetectorKPSeq}


def build_detector(arch: str, win: int, zdim: int, in_ch: int = 3):
    return ARCHS[arch](win, zdim, in_ch)


def load_detector(path: str, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("arch") == "kpseq": m = DetectorKPSeq(ck["win"], ck["zdim"], int(ck.get("in_ch", 4)), int(ck.get("hidden", 128)))
    else: m = build_detector(ck.get("arch", "base"), ck["win"], ck["zdim"], int(ck.get("in_ch", 3)))
    m.load_state_dict(ck["model"]); m = m.to(device).eval()
    return m, ck
