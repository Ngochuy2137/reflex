"""Simulated perception front end used by the Stage 1 teacher (BOX_DP_EST=1).

Same pipeline as perception_estimator (synthetic detector -> Kalman filter ->
actor obs, plus blind-age and trust channels), with a fifth dirty-perception
arm, `delay`, and optional off-by-default models of real-robot error modes:
angle/distance-dependent dropout, range-proportional depth bias, per-frame
outliers, velocity loss while coasting, a grounded-box veto, and an external
detector input.
"""
from __future__ import annotations

import math
import os

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from reflex.envs.kalman import BoxKalmanBatch

# Why a filter in the loop rather than additive noise: a single detection is
# off by ~0.18 m (p50), and after filtering this becomes a temporally coherent
# tracking error (a wrong extrapolation that persists through a blind
# stretch).  A policy trained on white noise learns to time-average, which
# does nothing against a coherent error, so the pipeline is
#     true box pos
#       -> bursty detect/blind schedule (empirical run-length histogram)
#       -> on detect ticks: anisotropic camera-frame measurement noise
#       -> a batched copy of the robot's box Kalman filter (kalman.py)
#       -> the policy sees the filter's output, never the truth
# All BOX_DP_* defaults are measured real-robot numbers.

# ---------------------------------------------------------------- knobs ----
DP_EST = os.environ.get("BOX_DP_EST", "0") == "1"

# ---- dirty-perception curriculum ------------------------------------------
# Per-episode arm mix (fractions of episodes; remainder = clean):
#   noise = extra white scatter, jump = teleporting outliers,
#   freeze = measurement latched at one value, garbage = uniform junk.
V31_MIX_NOISE = float(os.environ.get("BOX_V31_MIX_NOISE", "0.15"))
V31_MIX_JUMP = float(os.environ.get("BOX_V31_MIX_JUMP", "0.10"))
V31_MIX_FREEZE = float(os.environ.get("BOX_V31_MIX_FREEZE", "0.10"))
V31_MIX_GARBAGE = float(os.environ.get("BOX_V31_MIX_GARBAGE", "0.05"))
V31_NOISE_LO = float(os.environ.get("BOX_V31_NOISE_LO", "0.05"))
V31_NOISE_HI = float(os.environ.get("BOX_V31_NOISE_HI", "0.30"))
V31_JUMP_P_LO = float(os.environ.get("BOX_V31_JUMP_P_LO", "0.03"))
V31_JUMP_P_HI = float(os.environ.get("BOX_V31_JUMP_P_HI", "0.15"))
V31_JUMP_R = float(os.environ.get("BOX_V31_JUMP_R", "3.0"))
# the freeze arm latches at a random flight tick in [FRZ_LO, FRZ_HI) (50 Hz)
V31_FRZ_LO = int(os.environ.get("BOX_V31_FRZ_LO", "30"))
V31_FRZ_HI = int(os.environ.get("BOX_V31_FRZ_HI", "220"))
# Delay arm (mode 5): measurements delivered N ticks late, N ~ U[DELAY_LO,
# DELAY_HI] per episode (20 ms ticks).  Moderate delays (total measurement age
# ~130 ms, just inside a 150 ms staleness limit) leave the robot safe but
# unable to catch; this arm makes the policy meet them in training.
V36_MIX_DELAY = float(os.environ.get("BOX_V36_MIX_DELAY", "0.10"))
V36_DELAY_LO = int(os.environ.get("BOX_V36_DELAY_LO", "3"))
V36_DELAY_HI = int(os.environ.get("BOX_V36_DELAY_HI", "8"))
V36_DELAY_MAX = 12   # ring size; HI must stay < MAX

# Trust channels (window lengths in samples, thresholds in m / s; tick = clock)
TR_WJ = 10
TR_WF = 15
TR_FROZEN_EPS = 0.03
TR_FROZEN_SPAN_S = 0.30
TR_FROZEN_MIN_DIST = 0.8
TR_JITTER_CLIP = 0.5
TR_FRESH_CLIP_S = 0.30

# Target fraction of 50 Hz flight ticks that carry a fresh detection.
# 0.409, not the detector's ~73% per-frame hit rate: that figure only counts
# frames in which the box is inside the field of view, and field-of-view loss
# dominates.  Independent measurements agree on a ~35-41% band (40.9% of 50 Hz
# flight ticks with a 30 fps camera; 34.3% in offline detector replay; ~40%
# box visibility over a flight in a camera-visibility trace), so 0.733 would
# train a policy that sees nearly twice the real information rate.
COVERAGE = float(os.environ.get("BOX_DP_COVERAGE", "0.409"))
# Off-axis detection dropout (all off by default): the keep probability ramps
# linearly from ANG_ON at boresight to ANG_OFF at ANG_DEG.  See _advance for
# why the dropout should depend on angle rather than be uniform.
DP_ANG_ON = float(os.environ.get("BOX_DP_ANG_ON", "0"))    # keep prob at boresight
DP_ANG_OFF = float(os.environ.get("BOX_DP_ANG_OFF", "0"))  # keep prob at/after ANG_DEG
DP_ANG_DEG = float(os.environ.get("BOX_DP_ANG_DEG", "35"))  # ramp end, degrees
# Degrees of full detection before the ramp starts.  A wide-FOV camera
# (~110 deg HFOV) detects nearly uniformly out to ~45 deg and then drops off
# at the frame edge; a pure linear ramp (FLAT=0) degrades from the centre
# outward and under-credits on-axis throws.  Matters mainly with an orbiting
# thrower (BOX_V38_ORBIT_DPS), where large off-axis angles occur.  0 = off.
DP_ANG_FLAT = float(os.environ.get("BOX_DP_ANG_FLAT", "0"))
# Distance-dependent dropout: keep probability FAR_COV beyond FAR_SPLIT m and
# NEAR_COV inside it (0 = off).
DP_FAR_COV = float(os.environ.get("BOX_DP_FAR_COV", "0"))
DP_NEAR_COV = float(os.environ.get("BOX_DP_NEAR_COV", "0"))
DP_FAR_SPLIT = float(os.environ.get("BOX_DP_FAR_SPLIT", "2.0"))

# Per-episode randomisation of the gap statistics, so the policy cannot
# overfit one fixed dropout signature: coverage jitter + blind-run length scale.
# Bimodal coverage (off by default): on real throws the detector either tracks
# most of the flight (in-flight coverage p50 ~85%) or produces essentially
# nothing (~21% of throws never form the 3 consecutive detections needed to
# open a track).  Over all throws: {0% x 21%, 57-100% x 79%}.  A uniform
# coverage draw makes every throw mediocre instead of most throws good and a
# fifth of them blind.  DEAD_FRAC episodes get coverage DEAD_COV; the rest draw
# around COVERAGE as before.
DP_DEAD_FRAC = float(os.environ.get("BOX_DP_DEAD_FRAC", "0"))
DP_DEAD_COV = float(os.environ.get("BOX_DP_DEAD_COV", "0.02"))
COV_JITTER = float(os.environ.get("BOX_DP_COV_JITTER", "0.15"))   # +-15%
# consecutive misses after which the velocity channel degrades (measured on the robot: p50 21)
DP_COAST_MAX = int(os.environ.get("BOX_DP_COAST_MAX", "0"))
DP_COAST_KEEP = float(os.environ.get("BOX_DP_COAST_KEEP", "0.0"))
# ---- grounded-box veto (off by default) -----------------------------------
# A box already lying on the floor but still tracked (static estimate at
# z ~0.3 m, ~0.45 m ahead) is a state the policy has no trained answer to, and
# it can bend toward the box and fall.  Detections lower than GROUNDED_Z for
# >= GROUNDED_S seconds are vetoed (not ingested) and the track is dropped, so
# the policy sees NO_TRACK_PRIOR exactly as before a throw; a detection above
# CLEAR_Z re-arms.  The same rule runs on the robot.
DP_GROUNDED_VETO = int(os.environ.get("BOX_DP_GROUNDED_VETO", "0"))
DP_GROUNDED_Z = float(os.environ.get("BOX_DP_GROUNDED_Z", "0.30"))          # m, measurement height
DP_GROUNDED_S = float(os.environ.get("BOX_DP_GROUNDED_S", "0.3"))           # s of low detections before the veto latches (the lunge starts ~0.75 s after landing, so 0.3 keeps a 0.45 s margin; a flying box is below 0.30 m for < 0.1 s)
DP_GROUNDED_CLEAR_Z = float(os.environ.get("BOX_DP_GROUNDED_CLEAR_Z", "0.60"))  # a detection above this clears the latch
if DP_GROUNDED_VETO:
    print(f"[robust] grounded-box veto active: meas z<{DP_GROUNDED_Z} m for >={DP_GROUNDED_S} s "
          f"-> track dropped (NO_TRACK_PRIOR), re-arm on z>{DP_GROUNDED_CLEAR_Z} m", flush=True)
# fraction of flight ticks whose velocity channel reads zero (measured on the robot: 19-35%)
DP_VZERO_P = float(os.environ.get("BOX_DP_VZERO_P", "0.0"))
LEN_SCALE_LO = float(os.environ.get("BOX_DP_LEN_LO", "0.8"))
LEN_SCALE_HI = float(os.environ.get("BOX_DP_LEN_HI", "1.3"))

# Empirical blind-run-length histogram, 50 Hz ticks, inside the flight window
# (n=201 runs, 1088 blind ticks).  p50 3, p90 14, max 88.  The real process is
# BURSTY -- a memoryless Bernoulli would give p50 1 and no long tail, and the
# long tail is precisely what breaks a filter.
BLIND_HIST = {
    1: 63, 2: 26, 3: 29, 4: 22, 5: 7, 6: 9, 7: 8, 8: 2, 9: 3, 10: 5,
    11: 3, 12: 1, 13: 2, 14: 2, 15: 2, 16: 3, 17: 4, 18: 1, 20: 4,
    21: 1, 29: 1, 30: 1, 49: 1, 88: 1,
}

# ---- measurement noise, decomposed in the CAMERA frame --------------------
# Radial = along the line of sight (depth).  Tangential = perpendicular.
# Measured: radial bias -0.072 m, |radial| p50 0.064; tangential p50 0.158;
# variance ratio radial/tangential 0.20 -> TANGENTIAL DOMINATES.  A fixed
# world-frame sigma would be flatly wrong, hence the camera-frame rotation.
RAD_BIAS = float(os.environ.get("BOX_DP_RAD_BIAS", "0.072"))   # m, TOWARD cam
RAD_SIGMA = float(os.environ.get("BOX_DP_RAD_SIGMA", "0.060"))  # m
# RAD_BIAS_JITTER: per-episode uniform jitter added to the radial bias (m).
# The measured -0.072 m is the raw unprojection bias; with a face-to-centre
# depth correction applied, the net bias is about +0.06 m: same magnitude,
# opposite sign, and the aggregate error cannot separate the two because
# tangential error dominates 2.5:1.  A coherent depth bias is exactly what a
# Kalman filter cannot average away, so rather than guess the sign,
# JITTER >= ~0.09 spans both signs and makes the policy robust to it.
RAD_BIAS_JITTER = float(os.environ.get("BOX_DP_RAD_BIAS_JITTER", "0.0"))
# ---- far-depth plateau (off by default) -----------------------------------
# Beyond DEPTH_FLAT_R the measured range is replaced by a constant plateau
# DEPTH_FLAT_D (+ DEPTH_FLAT_S noise): no range information plus a large
# positive bias.  Kept as a stress test; the real-camera reading that
# motivated it was later traced to a region-of-interest artefact (see the
# proportional bias below for the corrected measurement).
DEPTH_FLAT_R = float(os.environ.get("BOX_DP_DEPTH_FLAT_R", "0"))    # m, cliff
DEPTH_FLAT_D = float(os.environ.get("BOX_DP_DEPTH_FLAT_D", "4.81"))  # m, plateau
DEPTH_FLAT_S = float(os.environ.get("BOX_DP_DEPTH_FLAT_S", "0.04"))  # m, residual
# ---- range-proportional depth bias (off by default) -----------------------
# The box is reported RAD_BIAS_PCT x range farther than it is.  A real-camera
# check measured +17-18% at 3.5-4 m, whereas the absolute RAD_BIAS is ~2% at
# 3.5 m.  A proportional bias is still ~0.18-0.27 m at 1.0-1.5 m, the size of
# the box itself, right where the arms close.
RAD_BIAS_PCT = float(os.environ.get("BOX_DP_RAD_BIAS_PCT", "0"))  # frac of range
# ---- per-frame outlier mixture (off by default) --------------------------
# The V31_MIX_* arms are episode-level modes.  Fitting real flight trajectories
# under gravity and measuring per-frame residuals instead shows a well-behaved
# ~7 cm core with a small fraction of wild frames mixed in (tail out to ~1.8
# m); the fraction of frames off by > 30 cm ranged from ~1% to ~38% across
# detector configurations.  A policy meets those one frame at a time, so the
# mixture is per frame: with probability DIRTY_P a frame gets an extra
# isotropic N(0, DIRTY_SIG) displacement.
DIRTY_P = float(os.environ.get("BOX_DP_DIRTY_P", "0"))      # per-frame prob
DIRTY_SIG = float(os.environ.get("BOX_DP_DIRTY_SIG", "0.35"))  # m, isotropic
# ---- range-dependent radial sigma (off by default) ------------------------
# RAD_SIGMA is a constant at every range, and its value is a global average
# dominated by far ranges.  Per distance band (free-flight frames only) the
# real radial error is
#     0.8-1.2m 1.9cm | 1.2-1.6 2.0 | 1.6-2.0 2.6 | 2.0-2.5 3.6 | 2.5-3.5 6.5
# so in the 1-2 m window where the arms close it is 2.0-2.6 cm and a constant
# 6.6 cm injects ~3x the real error where it matters most.  Log-log fit over
# the five bands: sigma = 1.57 * z^1.12 cm.  A=0 falls back to RAD_SIGMA.
RAD_SIG_A = float(os.environ.get("BOX_DP_RAD_SIG_A", "0"))    # cm at z = 1 m
RAD_SIG_B = float(os.environ.get("BOX_DP_RAD_SIG_B", "1.12"))  # exponent
if RAD_SIG_A > 0.0:
    # log it, so training and evaluation noise settings can be compared from the logs
    print(f"[robust] radial noise sigma(z) = {RAD_SIG_A} z^{RAD_SIG_B} cm active (dirty_p {DIRTY_P})", flush=True)

# Tangential error is generated mechanistically, because it is a PIXEL effect:
# the bbox centre sits p50 46 px (p90 187) from the true 3D-centre projection,
# systematically biased TOWARD the image centre, growing with off-axis angle
# (r=-0.72): 0-10deg -> 21 px, 20-30deg -> 45 px, 40-55deg -> 121 px.
# Metres = (px / f_px) * range, so the same pixel error is a bigger metric
# error further away -- a fixed metre sigma would lose that.
# f_px 346.2 is the f-theta fisheye focal length of the head camera
# (OV9782 sensor); do not re-derive it from an FOV.
F_PX = float(os.environ.get("BOX_DP_F_PX", "346.2"))
# Calibration: the raw pixel model at scale 1.0 produced a tangential p50 of
# 0.331 m against the measured 0.158 m, a factor 2.1 too large.  The cause is
# a range mismatch, not a bad pixel model: metres = (px/f)*range, and 46 px
# <-> 0.158 m implies a detection-weighted range of ~1.2 m, because the real
# detector rarely fires on a box far enough away to subtend few pixels.  The
# simulated schedule has no range-dependent detection probability, so it emits
# detections out to 3-4 m where the same pixel error buys 2-3x the metres.
# Rather than invent an unmeasured detection-vs-range curve, the pixel scale is
# calibrated so the realised metre-space error distribution matches the
# measured one.  Range-proportionality is retained (far measurements are still
# worse than near ones); only the overall gain is fitted.
TANG_BIAS_SCALE = float(os.environ.get("BOX_DP_TANG_BIAS_SCALE", "0.48"))
TANG_SCATTER_PX = float(os.environ.get("BOX_DP_TANG_SCATTER_PX", "13.4"))
TANG_SCATTER_SCALE = float(os.environ.get("BOX_DP_TANG_SCATTER_SCALE", "1.0"))
# off-axis angle (deg) -> inward pixel bias, piecewise linear through the
# measured bins (centres 5 / 25 / 47.5 deg).
PX_KNOTS_DEG = (0.0, 5.0, 25.0, 47.5, 90.0)
PX_KNOTS_VAL = (18.0, 21.0, 45.0, 121.0, 300.0)

# Optional: stereo depth error growing as z^2 (baseline 7.5 cm).  Not part of
# the measured error model above; off by default.
DEPTH_Z2 = float(os.environ.get("BOX_DP_DEPTH_Z2", "0.0"))     # m per m^2

# Fraction of episodes whose actor gets clean ground-truth box obs (and
# blind_age 0).  This clean anchor set keeps the policy's observation
# distribution from drifting under pure-corruption training, while most of
# the gradient still comes from the noisy regime.
MIX_CLEAN = float(os.environ.get("BOX_DP_MIX_CLEAN", "0.25"))

# Camera mount in the torso frame: pos (0.02, 0, 0.62), optical axis = +x
# pitched up 10 deg.
CAM_OFF = (0.02, 0.0, 0.62)
CAM_PITCH_UP = math.radians(10.0)
# Prior used when the filter holds no track (same as on the robot):
# "far ahead", pelvis + R*[3, 0, 0.6].
NO_TRACK_PRIOR = (3.0, 0.0, 0.6)
CHEST_OFF = (0.25, 0.0, 0.30)      # chest point for the filter's contact freeze

if DP_EST:
    print(f"[robust] Kalman-filter-in-the-loop perception on (with delay arm): coverage={COVERAGE} "
          f"(+-{COV_JITTER}), blind-hist p50 3/p90 14, radial bias {RAD_BIAS} "
          f"sigma {RAD_SIGMA}, tangential px-model f={F_PX}, "
          f"mix_clean={MIX_CLEAN}", flush=True)


# ------------------------------------------------------------------ state --
def _blind_table(device):
    lens, probs = [], []
    tot = sum(BLIND_HIST.values())
    for k, v in sorted(BLIND_HIST.items()):
        lens.append(float(k))
        probs.append(v / tot)
    return (torch.tensor(lens, device=device),
            torch.tensor(probs, device=device))


def _state(env) -> dict:
    if not hasattr(env, "_v31_est"):
        n, dev = env.num_envs, env.device
        lens, probs = _blind_table(dev)
        env._v31_est = {
            "kf": BoxKalmanBatch(n, dev, dt=env.step_dt),
            "blind_len": lens,
            "blind_p": probs,
            # schedule: countdown of the current run + which phase we are in
            "run_left": torch.zeros(n, dtype=torch.long, device=dev),
            "in_blind": torch.zeros(n, dtype=torch.bool, device=dev),
            # per-episode gap randomisation
            "len_scale": torch.ones(n, device=dev),
            "det_mean": torch.ones(n, device=dev),
            "clean_ep": torch.zeros(n, dtype=torch.bool, device=dev),
            "rad_bias": torch.full((n,), RAD_BIAS, device=dev),
            "stepped": torch.full((n,), -1, dtype=torch.long, device=dev),
            # diagnostics (not used by the policy)
            "diag_blind_runs": [],
            "det_ticks": torch.zeros(n, dtype=torch.long, device=dev),
            # grounded-box veto latch
            "gnd_ticks": torch.zeros(n, dtype=torch.long, device=dev),
            "gnd_veto": torch.zeros(n, dtype=torch.bool, device=dev),
            "tot_ticks": torch.zeros(n, dtype=torch.long, device=dev),
            # ---- dirty-perception curriculum (per-episode arm) ----
            # 0=clean 1=noise 2=jump 3=freeze 4=garbage 5=delay
            "dirty_mode": torch.zeros(n, dtype=torch.long, device=dev),
            # delay arm: measurement ring + per-env delay ticks
            "dly_ticks": torch.zeros(n, dtype=torch.long, device=dev),
            "dly_meas": torch.zeros(n, V36_DELAY_MAX, 3, device=dev),
            "dly_det": torch.zeros(n, V36_DELAY_MAX, dtype=torch.bool,
                                   device=dev),
            "dirty_p1": torch.zeros(n, device=dev),   # sigma (noise) / P (jump)
            "frz_armed": torch.zeros(n, dtype=torch.bool, device=dev),
            "frz_at": torch.zeros(n, dtype=torch.long, device=dev),
            "frz_p": torch.zeros(n, 3, device=dev),
            # ---- trust channels: ring buffers (tick index as clock) ----
            "tr_raw": torch.full((n, TR_WF, 3), 1e9, device=dev),
            "tr_raw_t": torch.full((n, TR_WF), -1.0, device=dev),
            "tr_raw_d": torch.zeros(n, TR_WF, device=dev),
            "tr_acc": torch.full((n, TR_WJ, 3), 1e9, device=dev),
            "tr_acc_t": torch.full((n, TR_WJ), -1.0, device=dev),
            "trust": torch.zeros(n, 3, device=dev),   # jitter/fresh/frozen
        }
    return env._v31_est


def _sample_blind(s, idx):
    """Draw blind-run lengths from the empirical histogram (NOT geometric)."""
    k = torch.multinomial(s["blind_p"], len(idx), replacement=True)
    raw = s["blind_len"][k]
    return (raw * s["len_scale"][idx]).round().clamp(min=1).long()


def _sample_detect(s, idx):
    """Draw detect-run lengths ~ Geometric(mean = det_mean), which together
    with the blind histogram realises the target coverage."""
    m = s["det_mean"][idx].clamp(min=1.0)
    u = torch.rand(len(idx), device=m.device).clamp(min=1e-6)
    return (-m * torch.log(u)).round().clamp(min=1).long()


def dp_reset_estimator(env, env_ids: torch.Tensor,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")) -> None:
    """Reset event: clear the filter and re-roll this episode's gap statistics."""
    s = _state(env)
    if not torch.is_tensor(env_ids):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return
    dev = env.device
    n = len(env_ids)
    s["kf"].reset_idx(env_ids)

    # per-episode blind-length scale and coverage jitter
    s["len_scale"][env_ids] = (LEN_SCALE_LO
                               + (LEN_SCALE_HI - LEN_SCALE_LO)
                               * torch.rand(n, device=dev))
    cov = COVERAGE * (1.0 + COV_JITTER * (2.0 * torch.rand(n, device=dev) - 1.0))
    cov = cov.clamp(0.05, 0.95)
    if DP_DEAD_FRAC > 0.0:
        dead = torch.rand(n, device=dev) < DP_DEAD_FRAC
        cov = torch.where(dead, torch.full_like(cov, DP_DEAD_COV), cov)
    # mean blind run of THIS episode (histogram mean x its length scale)
    hist_mean = float((s["blind_len"] * s["blind_p"]).sum())
    blind_mean = hist_mean * s["len_scale"][env_ids]
    # coverage = det_mean / (det_mean + blind_mean)
    s["det_mean"][env_ids] = blind_mean * cov / (1.0 - cov)

    # start every episode detecting (the box hovers in view before release)
    s["in_blind"][env_ids] = False
    s["gnd_ticks"][env_ids] = 0
    s["gnd_veto"][env_ids] = False
    s["run_left"][env_ids] = _sample_detect(s, env_ids)
    s["clean_ep"][env_ids] = torch.rand(n, device=dev) < MIX_CLEAN
    # per-episode radial-bias roll (sign-agnostic when JITTER > RAD_BIAS)
    s["rad_bias"][env_ids] = RAD_BIAS + RAD_BIAS_JITTER * (
        2.0 * torch.rand(n, device=dev) - 1.0)
    s["stepped"][env_ids] = -1
    s["det_ticks"][env_ids] = 0
    s["tot_ticks"][env_ids] = 0

    # ---- roll this episode's dirty arm ------------------------------------
    u = torch.rand(n, device=dev)
    mode = torch.zeros(n, dtype=torch.long, device=dev)
    c1 = V31_MIX_NOISE
    c2 = c1 + V31_MIX_JUMP
    c3 = c2 + V31_MIX_FREEZE
    c4 = c3 + V31_MIX_GARBAGE
    c5 = c4 + V36_MIX_DELAY
    mode[u < c5] = 5
    mode[u < c4] = 4
    mode[u < c3] = 3
    mode[u < c2] = 2
    mode[u < c1] = 1
    # (clean_ep episodes stay clean regardless — they are the anchor set)
    mode[s["clean_ep"][env_ids]] = 0
    s["dirty_mode"][env_ids] = mode
    p1 = torch.zeros(n, device=dev)
    nz = mode == 1
    p1[nz] = V31_NOISE_LO + (V31_NOISE_HI - V31_NOISE_LO) \
        * torch.rand(int(nz.sum()), device=dev)
    jm = mode == 2
    p1[jm] = V31_JUMP_P_LO + (V31_JUMP_P_HI - V31_JUMP_P_LO) \
        * torch.rand(int(jm.sum()), device=dev)
    s["dirty_p1"][env_ids] = p1
    s["frz_armed"][env_ids] = False
    s["frz_at"][env_ids] = torch.randint(
        V31_FRZ_LO, V31_FRZ_HI, (n,), device=dev)
    # delay arm: per-episode delay + clear the ring
    s["dly_ticks"][env_ids] = torch.randint(
        V36_DELAY_LO, V36_DELAY_HI + 1, (n,), device=dev)
    s["dly_meas"][env_ids] = 0.0
    s["dly_det"][env_ids] = False
    s["tot_ticks"][env_ids] = 0

    # ---- clear trust ring buffers ------------------------------------------
    s["tr_raw"][env_ids] = 1e9
    s["tr_raw_t"][env_ids] = -1.0
    s["tr_raw_d"][env_ids] = 0.0
    s["tr_acc"][env_ids] = 1e9
    s["tr_acc_t"][env_ids] = -1.0
    s["trust"][env_ids] = 0.0


def _px_bias(theta_deg: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of the measured off-axis pixel bias."""
    out = torch.full_like(theta_deg, PX_KNOTS_VAL[-1])
    for i in range(len(PX_KNOTS_DEG) - 1):
        a, b = PX_KNOTS_DEG[i], PX_KNOTS_DEG[i + 1]
        va, vb = PX_KNOTS_VAL[i], PX_KNOTS_VAL[i + 1]
        m = (theta_deg >= a) & (theta_deg < b)
        out = torch.where(m, va + (vb - va) * (theta_deg - a) / (b - a), out)
    return out


# ---- external measurement (off by default) --------------------------------
# BOX_DP_EXT_MEAS=1: an outside detector (e.g. one run on the rendered torso
# camera) supplies the per-tick measurement through the EXT dict; the synthetic
# detect/blind schedule and noise model are bypassed and the estimator ingests
# exactly EXT["meas"] (env-origin-relative position) and EXT["det"] (detected
# flag).  Used for the detector-in-the-loop baseline with the state policy.
EXT_MEAS_ON = os.environ.get("BOX_DP_EXT_MEAS", "0") == "1"
EXT = {"meas": None, "det": None, "ticks": 0, "det_ticks": 0}
if EXT_MEAS_ON:
    print("[robust] external measurement active: estimator ingests detector output (synthetic schedule/noise bypassed)", flush=True)

def _advance(env, asset_cfg: SceneEntityCfg) -> dict:
    """One estimator tick.  Idempotent within a single env step (several obs
    terms call this)."""
    s = _state(env)
    step_now = int(env.common_step_counter)
    if int(s["stepped"][0].item()) == step_now:
        return s
    s["stepped"][:] = step_now

    dev = env.device
    n = env.num_envs
    robot = env.scene["robot"]
    box: RigidObject = env.scene[asset_cfg.name]
    kf: BoxKalmanBatch = s["kf"]

    # ---- work in a per-env local frame (relative to the env origin) so the
    # filter's guard thresholds behave as they do on the robot.
    org = env.scene.env_origins                                   # (n,3)
    gt = box.data.root_pos_w - org
    base_p = robot.data.root_pos_w - org
    base_q = robot.data.root_quat_w

    # ---- camera pose (torso frame mount) ----
    if not hasattr(env, "_dp_torso_i"):
        env._dp_torso_i = list(robot.data.body_names).index("torso_link")
    ti = env._dp_torso_i
    tpos = robot.data.body_pos_w[:, ti] - org
    tquat = robot.data.body_quat_w[:, ti]
    cam = tpos + quat_apply(tquat, torch.tensor(CAM_OFF, device=dev).expand(n, 3))
    axis_t = torch.tensor([math.cos(CAM_PITCH_UP), 0.0, math.sin(CAM_PITCH_UP)],
                          device=dev).expand(n, 3)
    axis = quat_apply(tquat, axis_t)                              # optical axis

    # ---- advance the detect/blind schedule -------------------------------
    s["run_left"] = s["run_left"] - 1
    flip = s["run_left"] <= 0
    if bool(flip.any()):
        idx = flip.nonzero(as_tuple=False).flatten()
        was_blind = s["in_blind"][idx]
        s["in_blind"][idx] = ~was_blind
        new_blind = idx[~was_blind]     # detect -> blind
        new_det = idx[was_blind]        # blind -> detect
        if new_blind.numel():
            s["run_left"][new_blind] = _sample_blind(s, new_blind)
        if new_det.numel():
            s["run_left"][new_det] = _sample_detect(s, new_det)
    detected = ~s["in_blind"]
    s["tot_ticks"] += 1
    s["det_ticks"] += detected.long()

    # ---- build the measurement on detect ticks ---------------------------
    d = gt - cam
    rng_ = d.norm(dim=-1).clamp(min=1e-4)
    r_hat = d / rng_[:, None]
    # Distance- and angle-dependent dropout (both off by default).  The
    # detect/blind schedule above is agnostic to geometry, but the real
    # detector's effective rate falls with range (~26% beyond 2 m vs ~43% at
    # 1-2 m) and with off-axis angle.  Against hand-counted real catch rates,
    # the real system behaves close to ground truth on straight throws and
    # close to this noise model only on offset ones, i.e. a uniform model
    # under-credits straight throws by ~30 pp while over-crediting early
    # warning at range.  The tangential pixel model already depends on the
    # off-axis angle (21 px at 0-10 deg vs 121 px at 40-55 deg); these gates add
    # the same dependence to detection itself.  The off-axis keep probability
    # ramps linearly from ANG_ON (after ANG_FLAT deg) to ANG_OFF at ANG_DEG.
    if DP_ANG_ON > 0.0 or DP_ANG_OFF > 0.0:
        # Printed once from the branch that applies the gate, so the log shows
        # the BOX_DP_ANG_* values actually in effect.
        global _ANG_PROOF
        if not globals().get("_ANG_PROOF", False):
            _ANG_PROOF = True
            print(f"[robust] off-axis detection gate active: keep {DP_ANG_ON} "
                  f"at boresight -> {DP_ANG_OFF} at {DP_ANG_DEG} deg "
                  f"(flat to {DP_ANG_FLAT} deg)", flush=True)
        _on = DP_ANG_ON if DP_ANG_ON > 0.0 else 1.0
        _off = DP_ANG_OFF if DP_ANG_OFF > 0.0 else 1.0
        _cos = (r_hat * axis).sum(-1).clamp(-1.0, 1.0)
        _th = torch.rad2deg(torch.acos(_cos))
        _t = ((_th - DP_ANG_FLAT)
              / max(DP_ANG_DEG - DP_ANG_FLAT, 1e-3)).clamp(0.0, 1.0)
        keep_a = _on + (_off - _on) * _t
        detected = detected & (torch.rand(n, device=dev) < keep_a)
    if DP_FAR_COV > 0.0 or DP_NEAR_COV > 0.0:
        far = DP_FAR_COV if DP_FAR_COV > 0.0 else 1.0
        near = DP_NEAR_COV if DP_NEAR_COV > 0.0 else 1.0
        # Printed once from the branch that applies the split, with the
        # numbers actually used (not the env var).  With both knobs at 0 the
        # sim gets early warning at range that the real robot does not have,
        # which matters most for offset throws.
        global _FARCOV_PROVEN
        try:
            _FARCOV_PROVEN
        except NameError:
            _FARCOV_PROVEN = True
            print("[box_v36] DISTANCE-DEPENDENT DETECTION ACTIVE: "
                  "far(>%.2f m)=%.3f  near=%.3f  (0 = OFF = sim gets free "
                  "early warning)" % (DP_FAR_SPLIT, far, near), flush=True)
        keep_p = torch.where(rng_ > DP_FAR_SPLIT,
                             torch.full_like(rng_, far),
                             torch.full_like(rng_, near))
        detected = detected & (torch.rand(n, device=dev) < keep_p)
    # radial: signed depth error, biased TOWARD the camera (face-vs-centre)
    _sig = (0.01 * RAD_SIG_A * rng_ ** RAD_SIG_B) if RAD_SIG_A > 0.0 \
        else torch.full_like(rng_, RAD_SIGMA)
    rad = -s["rad_bias"] + _sig * torch.randn(n, device=dev)
    if RAD_BIAS_PCT != 0.0:
        # +PCT of true range, i.e. the box is reported FARTHER than it is
        rad = rad + RAD_BIAS_PCT * rng_
    if DEPTH_Z2 > 0.0:
        rad = rad + DEPTH_Z2 * rng_ ** 2 * torch.randn(n, device=dev)
    if DEPTH_FLAT_R > 0.0:
        # past the cliff the measurement carries no range information: force the
        # measured range to the plateau, i.e. rad = D_flat - true_range.
        rad = torch.where(rng_ > DEPTH_FLAT_R,
                          (DEPTH_FLAT_D - rng_)
                          + DEPTH_FLAT_S * torch.randn(n, device=dev),
                          rad)
    # tangential: inward pixel bias + isotropic pixel scatter, px -> metres
    cos_t = (r_hat * axis).sum(-1).clamp(-1.0, 1.0)
    theta = torch.rad2deg(torch.acos(cos_t))
    inward = axis - cos_t[:, None] * r_hat            # points toward the axis
    inward = inward / inward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    bias_m = (_px_bias(theta) / F_PX) * rng_ * TANG_BIAS_SCALE
    scat = torch.randn(n, 3, device=dev) * (TANG_SCATTER_PX / F_PX) \
        * rng_[:, None] * TANG_SCATTER_SCALE
    scat = scat - (scat * r_hat).sum(-1, keepdim=True) * r_hat   # project out radial
    if DIRTY_P > 0.0:
        # a wild frame: large isotropic displacement on top of the clean model
        _d = (torch.rand(n, device=dev) < DIRTY_P)
        scat = scat + _d[:, None].float() * (
            DIRTY_SIG * torch.randn(n, 3, device=dev))
    meas = gt + rad[:, None] * r_hat + bias_m[:, None] * inward + scat

    # ---- dirty-perception injection (per-episode arm, detect ticks only) ---
    md = s["dirty_mode"]
    if bool((md > 0).any()):
        # noise: white world-frame scatter, sigma per episode
        nz = (md == 1) & detected
        if bool(nz.any()):
            meas[nz] = meas[nz] + torch.randn(
                int(nz.sum()), 3, device=dev) * s["dirty_p1"][nz, None]
        # jump: with per-tick prob P, teleport into a 3 m ball around pelvis
        jm = (md == 2) & detected \
            & (torch.rand(n, device=dev) < s["dirty_p1"])
        if bool(jm.any()):
            k = int(jm.sum())
            u_ = torch.randn(k, 3, device=dev)
            u_ = u_ / u_.norm(dim=-1, keepdim=True).clamp(min=1e-9) \
                * V31_JUMP_R * torch.rand(k, 1, device=dev) ** (1 / 3)
            mj = base_p[jm] + u_
            mj[:, 2] = mj[:, 2].clamp(min=0.05)
            meas[jm] = mj
        # freeze: latch the measurement at tick frz_at, feed it forever after
        fz = md == 3
        arm_now = fz & ~s["frz_armed"] & (s["tot_ticks"] >= s["frz_at"]) \
            & detected
        if bool(arm_now.any()):
            s["frz_p"][arm_now] = meas[arm_now]
            s["frz_armed"][arm_now] = True
        live = fz & s["frz_armed"]
        if bool(live.any()):
            meas[live] = s["frz_p"][live]
            # a frozen track "detects" every tick — that is the failure mode
            detected = detected | live
        # garbage: uniform box around the pelvis
        gb = (md == 4) & detected
        if bool(gb.any()):
            mg = base_p[gb] + (torch.rand(int(gb.sum()), 3,
                                          device=dev) * 5.0 - 2.5)
            mg[:, 2] = mg[:, 2].clamp(min=0.05)
            meas[gb] = mg
        # delay: ring-push every tick and deliver the N-ticks-ago sample.  The
        # ring is pushed for all delay envs (even blind ticks, det=False) so
        # the delivered (meas, detected) pair replays the true schedule late.
        dl = md == 5
        if bool(dl.any()):
            s["dly_meas"][dl] = torch.roll(s["dly_meas"][dl], -1, dims=1)
            s["dly_det"][dl] = torch.roll(s["dly_det"][dl], -1, dims=1)
            s["dly_meas"][dl, -1] = meas[dl]
            s["dly_det"][dl, -1] = detected[dl]
            idx = (V36_DELAY_MAX - 1 - s["dly_ticks"]).clamp(min=0)
            ar = torch.arange(env.num_envs, device=dev)
            warm = s["tot_ticks"] > s["dly_ticks"]      # ring warm-up guard
            use = dl & warm
            meas = torch.where(use[:, None],
                               s["dly_meas"][ar, idx], meas)
            detected = torch.where(use, s["dly_det"][ar, idx], detected)
            # cold ring: report blind (age grows) rather than a zero sample
            detected = detected & ~(dl & ~warm)

    if EXT_MEAS_ON and EXT["meas"] is not None:
        meas = EXT["meas"].to(dev); detected = EXT["det"].to(dev)
        EXT["ticks"] += 1; EXT["det_ticks"] += int(detected.float().mean().item() * 1000)   # mille per tick (avg detection rate)
    # ---- run the Kalman filter ---------------------------------------------
    kf.chest = base_p + quat_apply(
        base_q, torch.tensor(CHEST_OFF, device=dev).expand(n, 3))
    if DP_GROUNDED_VETO:
        # meas is env-origin relative (z = height); only detect ticks count.
        _low = detected & (meas[:, 2] < DP_GROUNDED_Z)
        _high = detected & (meas[:, 2] > DP_GROUNDED_CLEAR_Z)
        s["gnd_ticks"] = torch.where(
            _low, s["gnd_ticks"] + 1,
            torch.where(_high, torch.zeros_like(s["gnd_ticks"]), s["gnd_ticks"]))
        _need = max(1, int(round(DP_GROUNDED_S / float(env.step_dt))))
        s["gnd_veto"] = (s["gnd_veto"] | (s["gnd_ticks"] >= _need)) & ~_high
        detected = detected & ~s["gnd_veto"]      # vetoed = not ingested (trust sees no measurement)
        kf.have = kf.have & ~s["gnd_veto"]        # track dropped -> est falls back to NO_TRACK_PRIOR, age 1.0
    kf.predict()
    kf.update(meas, detected)
    kf.runaway_guard()
    kf.contact_freeze()

    # ---- estimate handed to the policy -----------------------------------
    prior = base_p + quat_apply(
        base_q, torch.tensor(NO_TRACK_PRIOR, device=dev).expand(n, 3))
    hv = kf.have[:, None]
    est_p = torch.where(hv, kf.p, prior)
    est_v = torch.where(hv, kf.v, torch.zeros_like(kf.v))
    # ---- coast degradation (off by default) ------------------------------
    # On the robot, the velocity the policy sees often reads as (near)
    # stationary during flight.  Those ticks are long measurement gaps (filter
    # miss count p50 21, i.e. ~420 ms without a new measurement) and the onboard
    # filter frequently stops coasting, whereas this filter only clears its
    # track on reset and keeps coasting a clean ballistic model however sparse
    # the measurements get.  Past COAST_MAX consecutive misses the velocity
    # channel is scaled by COAST_KEEP.
    if DP_COAST_MAX > 0:
        _stale = (kf.miss > DP_COAST_MAX)[:, None]
        est_v = torch.where(_stale, est_v * DP_COAST_KEEP, est_v)
    # Per-tick velocity dropout (off by default).  With a fixed 0.6 s
    # post-release window the median observed speed on the robot matches
    # simulation; what differs is the tail: the velocity reads as stationary
    # on 19-35% of real flight ticks versus ~0% in simulation.  So zero the
    # velocity on a random VZERO_P fraction of ticks, rather than decaying it
    # after a gap (which over-injects).
    if DP_VZERO_P > 0.0:
        _vz = (torch.rand(n, device=dev) < DP_VZERO_P)[:, None]
        est_v = torch.where(_vz, torch.zeros_like(est_v), est_v)

    s["est_p_w"] = est_p + org          # back to world for the pelvis transform
    s["est_v_w"] = est_v
    s["age"] = kf.blind_age()
    s["meas_w"] = meas + org
    s["detected"] = detected

    # ---- trust channels -----------------------------------------------------
    now = s["tot_ticks"].float() * env.step_dt              # per-env clock (s)
    det_i = detected.nonzero(as_tuple=False).flatten()
    if det_i.numel():
        # push RAW ring (frozen uses raw: a frozen stream passes every gate)
        s["tr_raw"][det_i] = torch.roll(s["tr_raw"][det_i], -1, dims=1)
        s["tr_raw_t"][det_i] = torch.roll(s["tr_raw_t"][det_i], -1, dims=1)
        s["tr_raw_d"][det_i] = torch.roll(s["tr_raw_d"][det_i], -1, dims=1)
        s["tr_raw"][det_i, -1] = meas[det_i]
        s["tr_raw_t"][det_i, -1] = now[det_i]
        s["tr_raw_d"][det_i, -1] = (meas[det_i] - base_p[det_i]).norm(dim=-1)
        # push ACCEPTED ring (jitter uses what the filter actually ate)
        acc = detected & (kf.miss == 0)
        acc_i = acc.nonzero(as_tuple=False).flatten()
        if acc_i.numel():
            s["tr_acc"][acc_i] = torch.roll(s["tr_acc"][acc_i], -1, dims=1)
            s["tr_acc_t"][acc_i] = torch.roll(s["tr_acc_t"][acc_i], -1, dims=1)
            s["tr_acc"][acc_i, -1] = meas[acc_i]
            s["tr_acc_t"][acc_i, -1] = now[acc_i]
    if os.environ.get("BOX_V31_DEBUG") == "1" and step_now % 100 == 0:
        fz_on = s["frz_armed"]
        div = (est_p - gt).norm(dim=-1)
        print(f"[v31dbg] step {step_now} frz_armed {int(fz_on.sum())}/{n} "
              f"est-gt p50 {div.median():.3f} "
              f"frz-env est-gt p50 "
              f"{div[fz_on].median():.3f}" if fz_on.any() else
              f"[v31dbg] step {step_now} frz_armed 0/{n} "
              f"est-gt p50 {div.median():.3f}", flush=True)

    # jitter: RMS innovation vs pairwise constant-velocity extrapolation
    ap, at = s["tr_acc"], s["tr_acc_t"]
    dt01 = at[:, 1:-1] - at[:, :-2]
    dt12 = at[:, 2:] - at[:, 1:-1]
    ok = (at[:, :-2] >= 0) & (dt01 > 1e-4) & (dt12 > 1e-4) & (dt12 <= 0.5)
    v_ = (ap[:, 1:-1] - ap[:, :-2]) / dt01.clamp(min=1e-4)[..., None]
    pred = ap[:, 1:-1] + v_ * dt12[..., None]
    e2 = ((ap[:, 2:] - pred) ** 2).sum(-1)
    e2 = torch.where(ok, e2, torch.zeros_like(e2))
    cnt = ok.sum(-1).clamp(min=1)
    jit = (e2.sum(-1) / cnt).sqrt()
    jit = torch.where(ok.sum(-1) > 0, jit, torch.zeros_like(jit))
    s["trust"][:, 0] = jit.clamp(max=TR_JITTER_CLIP) / TR_JITTER_CLIP
    # freshness: time since last accepted measurement
    last_t = at[:, -1]
    fresh = torch.where(last_t >= 0, (now - last_t).clamp(min=0.0),
                        torch.full_like(now, TR_FRESH_CLIP_S))
    s["trust"][:, 1] = fresh.clamp(max=TR_FRESH_CLIP_S) / TR_FRESH_CLIP_S
    # frozen: full raw window, enough time span, static, and NOT near-chest
    rp, rt, rd = s["tr_raw"], s["tr_raw_t"], s["tr_raw_d"]
    full = (rt[:, 0] >= 0) & (rt[:, -1] - rt[:, 0] >= TR_FROZEN_SPAN_S)
    ctr = rp.mean(dim=1, keepdim=True)
    spread = (rp - ctr).norm(dim=-1).max(dim=1).values
    far = rd.mean(dim=1) >= TR_FROZEN_MIN_DIST
    s["trust"][:, 2] = (full & far & (spread < TR_FROZEN_EPS)).float()
    # diagnostics (not used by the policy)
    s["gt_local"] = gt
    s["cam_local"] = cam
    s["theta_deg"] = theta
    s["range_m"] = rng_
    s["accepted"] = detected & (kf.miss == 0)
    s["kf_err"] = (kf.p - gt).norm(dim=-1)
    s["tracked"] = kf.have
    return s


# ------------------------------------------------------------ obs terms ----
def balloon_pos_pelvis_frame_dp(env,
                                asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")
                                ) -> torch.Tensor:
    """ACTOR obs: filtered box position in pelvis frame.  clean_ep envs get GT."""
    s = _advance(env, asset_cfg)
    robot = env.scene["robot"]
    box: RigidObject = env.scene[asset_cfg.name]
    pos_w = torch.where(s["clean_ep"][:, None], box.data.root_pos_w, s["est_p_w"])
    return quat_apply_inverse(robot.data.root_quat_w,
                              pos_w - robot.data.root_pos_w)


def balloon_lin_vel_pelvis_frame_dp(env,
                                    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")
                                    ) -> torch.Tensor:
    """ACTOR obs: filtered box velocity in pelvis frame.  clean_ep envs get GT."""
    s = _advance(env, asset_cfg)
    robot = env.scene["robot"]
    box: RigidObject = env.scene[asset_cfg.name]
    bv = torch.nan_to_num(box.data.root_lin_vel_w, nan=0.0,
                          posinf=10.0, neginf=-10.0)
    vel_w = torch.where(s["clean_ep"][:, None], bv, s["est_v_w"])
    return quat_apply_inverse(robot.data.root_quat_w, vel_w)


def box_blind_age(env,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")
                  ) -> torch.Tensor:
    """Blind-age channel (obs 166): 1.0 with no track, else min(miss, 15)/15,
    0 on a tick that ingested a measurement (same definition as on the robot).
    Clean episodes report 0 (perfect track).
    """
    s = _advance(env, asset_cfg)
    age = torch.where(s["clean_ep"], torch.zeros_like(s["age"]), s["age"])
    return age[:, None]


def box_trust(env,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")
              ) -> torch.Tensor:
    """Trust channels (obs 167-169): [jitter, freshness, frozen] in [0, 1].

    Same windows/thresholds/clips as the on-robot implementation.  Clean
    episodes report the channels as computed (not zeroed): a clean stream
    legitimately produces low jitter / low freshness / frozen 0, and the
    policy must learn the channels' honest statistics, not a sim-only shortcut.
    """
    s = _advance(env, asset_cfg)
    return s["trust"]


def leg_violence_penalty(env,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                         ) -> torch.Tensor:
    """Leg-speed penalty: sum(relu(|leg dq| - 30)) in rad/s.

    The 30 rad/s threshold sits above every measured healthy catch (peak
    28-29, p99.9 17-27), so nominal behaviour pays zero and the term never
    competes with the ~0.17/step living reward.  Flailing dives under bad
    perception (100-200 rad/s) pay 0.7-1.7/step at weight 0.01, so standing
    still always beats a flail.  Without it the reward prices nothing about
    HOW the policy fails, and dirty episodes teach "try harder" instead of
    "fail calmly".
    """
    robot = env.scene[asset_cfg.name]
    if not hasattr(env, "_v31_leg_ids"):
        names = robot.data.joint_names
        env._v31_leg_ids = [i for i, n in enumerate(names)
                            if any(k in n for k in
                                   ("hip", "knee", "ankle"))]
    dq = robot.data.joint_vel[:, env._v31_leg_ids].abs()
    return torch.relu(dq - 30.0).sum(dim=-1)
