"""Stage 2: rsl_rl ActorCritic with a privileged latent bottleneck (encoder E_phi + action decoder).

Actor (action decoder) input = obs[0:KEEP] ⊕ z,   z = tanh(A·priv + b + g(priv)) ∈ [-1,1]^ZDIM
  priv  obs[PRIV_LO:PRIV_HI]  (latent_obs.box_priv_stateless)
  A     linear map (``e_priv_lin``).  ENC_INIT=identity: identity-style init on pos/vel, so a
        warm start from the Stage 1 teacher behaves like the teacher at iteration 0 (see
        warm_start_latent).  ENC_INIT=random: random orthogonal init; what z encodes is learned
        (behavior cloning in bc_init.py, then joint PPO).
  g     2x64 ELU residual MLP (``e_priv_res``), last layer zero-initialised (learned by PPO only)
  The legacy estimator channels obs[KEEP:PRIV_LO] (estimated pos/vel, blind age, trust) are NOT
  seen by the actor; they stay in the observation so index-based environment code and the
  two-stage arm trigger keep working unchanged.
The critic is unchanged (privileged 292-D critic observation group).

Robustness of the decoder to z during PPO: Gaussian noise with std Z_NOISE in tanh space plus
whole-vector dropout with probability Z_DROP, both in training mode only.  `z_external` lets an
evaluation feed z directly (E_phi bypassed); with BOX_LATENT_Z_FROM_OBS_LO >= 0 the actor reads z
from the observation, where the Stage 3 visual encoder wrapper writes it.

There is no auxiliary decoding head; pos/vel probes, if needed, are fitted post hoc on z.
Knobs (defaults): BOX_LATENT_ZDIM (16; the released configs use 8) BOX_LATENT_ZNOISE (0.1)
       BOX_LATENT_ZDROP (0.1) BOX_LATENT_KEEP (160) BOX_LATENT_PRIV_LO (170) BOX_LATENT_PRIV_HI (181)
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP

ZDIM = int(os.environ.get("BOX_LATENT_ZDIM", "16"))
Z_NOISE = float(os.environ.get("BOX_LATENT_ZNOISE", "0.10"))
# Residual replay: per-dim Gaussian std on z measured from a trained visual encoder's held-out
# residuals, e.g. "0.033,0.047,0.033,0.052,0.046,0.054,0,...".  Overrides the scalar Z_NOISE when set.
Z_NOISE_VEC = [float(x) for x in os.environ.get("BOX_LATENT_ZNOISE_VEC", "").split(",") if x.strip()]
Z_DROP = float(os.environ.get("BOX_LATENT_ZDROP", "0.10"))
# Ramp noise/drop from 0 to full over this many act() calls (24 per PPO iteration at
# num_steps_per_env=24), so the warm-started teacher behaviour is not drowned at iteration 0.
Z_RAMP_CALLS = int(os.environ.get("BOX_LATENT_ZRAMP_ITERS", "200")) * 24
# Optionally freeze the warm-started decoder for the first N iterations so only E_phi learns:
# the encoder first learns to provide what the decoder expects (Kalman-style coasting from the
# last visible pos/vel/age) before joint training is allowed to reshape the decoder.
FREEZE_DEC_CALLS = int(os.environ.get("BOX_LATENT_FREEZE_DEC_ITERS", "0")) * 24
# Residual replay must keep E_phi fixed (the visual encoder was trained on its labels): freeze the encoder for the whole run.
FREEZE_ENC = os.environ.get("BOX_LATENT_FREEZE_ENC", "0") == "1"
# Visual-encoder-in-the-loop PPO: z is written into the observation by DetectorObsWrapper at columns
# [Z_FROM_OBS_LO, Z_FROM_OBS_LO + ZDIM); the actor reads it directly (E_phi bypassed), so rollout and update see the same z.
Z_FROM_OBS_LO = int(os.environ.get("BOX_LATENT_Z_FROM_OBS_LO", "-1"))
# Evaluation-only robustness probes (inference mode): Gaussian noise on z (BOX_LATENT_EVAL_ZNOISE_VEC, per-dim std)
# or on the privileged pos/vel before E_phi (BOX_LATENT_EVAL_PRIV_NOISE); BOX_LATENT_EVAL_ZNOISE_RHO > 0 makes the
# noise AR(1)-correlated in time.
_ev = os.environ.get("BOX_LATENT_EVAL_ZNOISE_VEC", "")
_EVAL_ZNOISE_VEC = torch.tensor([float(x) for x in _ev.split(",")]) if _ev else None
_EVAL_ZNOISE_RHO = float(os.environ.get("BOX_LATENT_EVAL_ZNOISE_RHO", "0"))
_ep = os.environ.get("BOX_LATENT_EVAL_PRIV_NOISE", "")
_EVAL_PRIV_NOISE = torch.tensor([float(x) for x in _ep.split(",")]) if _ep else None
if _EVAL_PRIV_NOISE is not None: print(f"[latent] EVAL PRIV NOISE (inference only, before E_phi): pos/vel std {_EVAL_PRIV_NOISE.tolist()} rho={_EVAL_ZNOISE_RHO}", flush=True)
if _EVAL_ZNOISE_VEC is not None: print(f"[latent] EVAL Z NOISE (inference only): per-dim std {_EVAL_ZNOISE_VEC.tolist()} rho={_EVAL_ZNOISE_RHO} ({'AR(1) correlated' if _EVAL_ZNOISE_RHO > 0 else 'iid'})", flush=True)
KEEP = int(os.environ.get("BOX_LATENT_KEEP", "160"))
PRIV_LO = int(os.environ.get("BOX_LATENT_PRIV_LO", "170"))
PRIV_HI = int(os.environ.get("BOX_LATENT_PRIV_HI", "181"))
# identity-init scales: z[0:3] = tanh(pos/POS_SCALE), z[3:6] = tanh(vel/VEL_SCALE)
POS_SCALE = 4.0
E_PRIV_EXTRA_INIT = float(os.environ.get("BOX_LATENT_EXTRA_INIT", "0.1"))   # std of the random init of encoder rows z[6:] (0 = zero init: dead dims, see __init__)
VEL_SCALE = 8.0
# BOX_LATENT_ENC_INIT=random: no designed identity pos/vel mapping.  The encoder starts as a random orthogonal
# projection of the (unit-scaled) privileged vector, the decoder's z columns start random, and the decoder body
# warm-started from the Stage 1 teacher is initially blind to the box: what z encodes is learned, not designed
# (behavior cloning in bc_init.py, then joint PPO).
ENC_INIT = os.environ.get("BOX_LATENT_ENC_INIT", "identity")
assert ENC_INIT in ("identity", "random"), ENC_INIT
DEC_Z_INIT_FRAC = float(os.environ.get("BOX_LATENT_DEC_Z_INIT_FRAC", "0.05"))   # z-column std as a fraction of the legacy box-column rms
# Legacy box-channel layout of the Stage 1 teacher's 170-D policy observation (the warm-start source)
LEG_POS, LEG_VEL, LEG_BLIND, LEG_TRUST = slice(160, 163), slice(163, 166), 166, slice(167, 170)


class LatentActorCritic(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, actor_hidden_dims=[256, 256, 256],
                 activation="elu", **kwargs):
        super().__init__(obs, obs_groups, num_actions, actor_hidden_dims=actor_hidden_dims,
                         activation=activation, **kwargs)
        n_in = self.actor[0].in_features
        assert n_in >= PRIV_HI and PRIV_HI - PRIV_LO > 0, (n_in, PRIV_LO, PRIV_HI)
        self.keep, self.plo, self.phi, self.zdim = KEEP, PRIV_LO, PRIV_HI, ZDIM
        pdim = PRIV_HI - PRIV_LO
        self.e_priv_lin = nn.Linear(pdim, ZDIM)
        self.e_priv_res = MLP(pdim, ZDIM, [64, 64], activation)
        with torch.no_grad():
            self.e_priv_lin.weight.zero_()
            self.e_priv_lin.bias.zero_()
            if ENC_INIT != "random":   # identity pos/vel rows need ZDIM >= 6; the random-init path skips them and overwrites the whole matrix below
                for i in range(3):
                    self.e_priv_lin.weight[i, i] = 1.0 / POS_SCALE
                    self.e_priv_lin.weight[3 + i, 3 + i] = 1.0 / VEL_SCALE
            self.e_priv_res[-1].weight.zero_()
            self.e_priv_res[-1].bias.zero_()
            # Break the zero-zero symmetry for z dims beyond pos/vel: with both the encoder rows and the
            # decoder's z columns at exactly 0, the gradients on both sides are identically 0 and those dims
            # never move (observed: z[6:16] variance 0.0 after 600 iterations).  Small random rows here plus
            # small random decoder columns in warm_start_latent give PPO a nonzero gradient path.
            if ZDIM > 6:
                self.e_priv_lin.weight[6:].normal_(0.0, E_PRIV_EXTRA_INIT)
            if ENC_INIT == "random":
                # generic unit scaling of the raw inputs (metres, m/s -> O(1)); NOT a mapping onto z dims
                sc = torch.ones(pdim); sc[0:3] = 1.0 / POS_SCALE; sc[3:6] = 1.0 / VEL_SCALE
                self.register_buffer("priv_scale", sc)
                nn.init.orthogonal_(self.e_priv_lin.weight, gain=1.0)
                print(f"[latent] ENC_INIT=random: e_priv_lin orthogonal {tuple(self.e_priv_lin.weight.shape)}, no identity rows, "
                      f"input unit-scaling buffer, residual MLP output zero-init", flush=True)
        if ENC_INIT != "random":
            self.priv_scale = None
        # replace the actor built by the base class (n_in wide) with the bottlenecked one
        self.actor = MLP(KEEP + ZDIM, num_actions, actor_hidden_dims, activation)
        self.z_noise, self.z_drop = Z_NOISE, Z_DROP
        if Z_NOISE_VEC:
            v = torch.zeros(ZDIM); v[: min(len(Z_NOISE_VEC), ZDIM)] = torch.tensor(Z_NOISE_VEC[:ZDIM])
            self.register_buffer("z_noise_vec", v, persistent=False); self.z_noise = float(v.max())
            print(f"[latent] RESIDUAL REPLAY: per-dim z noise std {[round(x, 3) for x in v.tolist()]}", flush=True)
        else:
            self.register_buffer("z_noise_vec", None, persistent=False)
        self.z_calls = 0
        self.act_calls = 0
        self.dec_frozen = False
        self.z_external = None
        if FREEZE_DEC_CALLS > 0:
            self._set_decoder_frozen(True)
        if FREEZE_ENC:
            for p_ in list(self.e_priv_lin.parameters()) + list(self.e_priv_res.parameters()):
                p_.requires_grad_(False)
            print("[latent] ENCODER FROZEN for the whole run (E_phi fixed; visual-encoder labels stay valid)", flush=True)
        print(f"[latent] LatentActorCritic ACTIVE: obs {n_in} -> keep [0:{KEEP}) + z{ZDIM} "
              f"from priv [{PRIV_LO}:{PRIV_HI}); legacy [{KEEP}:{PRIV_LO}) masked; "
              f"z noise {Z_NOISE} drop {Z_DROP} ramped over {Z_RAMP_CALLS // 24} iters; "
              f"decoder frozen for first {FREEZE_DEC_CALLS // 24} iters; actor in={KEEP + ZDIM}"
              + (f"; Z FROM OBS cols [{Z_FROM_OBS_LO}:{Z_FROM_OBS_LO + ZDIM}) (detector-in-loop)" if Z_FROM_OBS_LO >= 0 else ""), flush=True)

    # ---- decoder freeze (training detail, not architecture) -----------------
    def _set_decoder_frozen(self, frozen: bool):
        for p in self.actor.parameters():
            p.requires_grad_(not frozen)
        self.dec_frozen = frozen
        print(f"[latent] decoder {'FROZEN' if frozen else 'UNFROZEN'} at act call {self.act_calls}"
              f" (~iter {self.act_calls // 24})", flush=True)

    def act(self, obs, **kwargs):
        # count ROLLOUT calls only (rsl_rl collects under inference_mode; the PPO update
        # re-calls act() on minibatches with grad enabled -- those must not advance clocks)
        if torch.is_inference_mode_enabled():
            self.act_calls += 1
        if self.dec_frozen and self.act_calls > FREEZE_DEC_CALLS:
            self._set_decoder_frozen(False)
        return super().act(obs, **kwargs)

    # ---- latent -----------------------------------------------------------
    def encode(self, priv: torch.Tensor) -> torch.Tensor:
        if self.priv_scale is not None:
            priv = priv * self.priv_scale
        return torch.tanh(self.e_priv_lin(priv) + self.e_priv_res(priv))

    def latent(self, obs: torch.Tensor) -> torch.Tensor:
        if Z_FROM_OBS_LO >= 0:
            return obs[:, Z_FROM_OBS_LO:Z_FROM_OBS_LO + self.zdim]          # visual-encoder z written into obs (no noise, no E_phi)
        if _EVAL_PRIV_NOISE is not None and not self.training and self.z_external is None:
            # Physical-space probe: AR(1) noise (rho = _EVAL_ZNOISE_RHO) on the privileged pos/vel (cols 0:6) BEFORE E_phi,
            # with per-channel std from BOX_LATENT_EVAL_PRIV_NOISE (e.g. a visual encoder's flight-window pos/vel RMSE),
            # applied only when the window has a visible frame (col 7).  A coherent physical error is what a real visual
            # estimator makes; independent per-dim z noise is not.
            priv = obs[:, self.plo:self.phi].clone(); sig = _EVAL_PRIV_NOISE.to(obs.device)
            st = getattr(self, "_eval_priv_state", None)
            if st is None or st.shape != priv[:, :6].shape: st = torch.zeros_like(priv[:, :6])
            rho = _EVAL_ZNOISE_RHO; st = rho * st + math.sqrt(max(1e-6, 1.0 - rho ** 2)) * torch.randn_like(st); self._eval_priv_state = st
            priv[:, :6] = priv[:, :6] + sig * st * (priv[:, 7:8] > 0.5).float()
            z = self.encode(priv)
        else:
            z = self.z_external if self.z_external is not None else self.encode(obs[:, self.plo:self.phi])
        if _EVAL_ZNOISE_VEC is not None and not self.training:
            # Precision probe: E_phi z + per-dim Gaussian noise (BOX_LATENT_EVAL_ZNOISE_VEC="s0,s1,...", e.g. a visual
            # encoder's flight-window residual std), inference only.  BOX_LATENT_EVAL_ZNOISE_RHO > 0 makes it AR(1) in time
            # (rho per 20 ms step); the visual encoder's residual had a lag-1 autocorrelation of 0.86, so rho = 0.86
            # reproduces its temporal structure.
            sig = _EVAL_ZNOISE_VEC.to(z.device)
            if _EVAL_ZNOISE_RHO > 0.0:
                st = getattr(self, "_eval_noise_state", None)
                if st is None or st.shape != z.shape: st = torch.zeros_like(z)
                st = _EVAL_ZNOISE_RHO * st + math.sqrt(1.0 - _EVAL_ZNOISE_RHO ** 2) * torch.randn_like(z)
                self._eval_noise_state = st; z = z + sig * st
            else:
                z = z + sig * torch.randn_like(z)
        if self.training and (self.z_noise > 0.0 or self.z_drop > 0.0):
            if torch.is_inference_mode_enabled():
                self.z_calls += 1
            ramp = min(1.0, self.z_calls / max(Z_RAMP_CALLS, 1))
            sig, drop = self.z_noise * ramp, self.z_drop * ramp
            if sig > 0.0:
                if self.z_noise_vec is not None:
                    z = z + ramp * self.z_noise_vec.to(z.device) * torch.randn_like(z)
                else:
                    z = z + sig * torch.randn_like(z)
            if drop > 0.0:
                keep = (torch.rand(z.shape[0], 1, device=z.device) >= drop).float()
                z = z * keep
            z = z.clamp(-1.0, 1.0)
        return z

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs[:, :self.keep], self.latent(obs)], dim=-1)

    # ---- rsl_rl hooks -------------------------------------------------------
    def update_distribution(self, obs):
        mean = self.actor(self.features(obs))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(self.features(obs))

    # ---- warm start from the Stage 1 teacher (170-D state policy) ----------
    @torch.no_grad()
    def warm_start_latent(self, old_state: dict) -> dict:
        """Map a Stage 1 teacher ActorCritic state dict (170-D policy input) onto this module.

        actor.0.weight (H,170) -> (H, KEEP+ZDIM): columns [0:KEEP) are copied.
        ENC_INIT=identity: the legacy pos/vel columns move onto z[0:6], scaled by
        POS/VEL_SCALE (tanh linearised); the blind_age/trust columns are dropped
        (their clean-stream value is 0, so there is no bias shift).
        ENC_INIT=random: the z columns are random (std = DEC_Z_INIT_FRAC x the rms
        of the legacy box columns) and the legacy pos/vel columns are not mapped.
        Every other tensor with equal shape is copied (a latent checkpoint loads 1:1);
        the encoder keeps its initialisation.  Returns counters for the log line.
        """
        new_state = self.state_dict()
        n_loaded = n_mapped = n_skipped = 0
        for k, old_v in old_state.items():
            if k not in new_state:
                n_skipped += 1
                continue
            new_v = new_state[k]
            if old_v.shape == new_v.shape:
                new_state[k] = old_v.clone().to(new_v.device).type_as(new_v)
                n_loaded += 1
            elif k == "actor.0.weight" and old_v.shape[1] == 170:
                w = torch.zeros_like(new_v)
                w[:, :self.keep] = old_v[:, :self.keep]
                if ENC_INIT == "random":
                    leg = torch.cat([old_v[:, LEG_POS] * POS_SCALE, old_v[:, LEG_VEL] * VEL_SCALE], dim=1)
                    std = float(leg.pow(2).mean().sqrt()) * DEC_Z_INIT_FRAC
                    w[:, self.keep:] = torch.randn_like(w[:, self.keep:]) * std
                    print(f"[latent][warm_start] ENC_INIT=random: decoder z columns ~ N(0, {std:.4f}) "
                          f"(= {DEC_Z_INIT_FRAC} x legacy box-column rms); legacy pos/vel columns NOT mapped", flush=True)
                else:
                    w[:, self.keep:self.keep + 3] = old_v[:, LEG_POS] * POS_SCALE
                    w[:, self.keep + 3:self.keep + 6] = old_v[:, LEG_VEL] * VEL_SCALE
                    if self.zdim > 6 and E_PRIV_EXTRA_INIT > 0.0:
                        w[:, self.keep + 6:] = torch.randn_like(w[:, self.keep + 6:]) * 0.01
                new_state[k] = w
                n_mapped += 1
                print(f"[latent][warm_start] mapped {k}: {tuple(old_v.shape)} -> {tuple(new_v.shape)}", flush=True)
            else:
                n_skipped += 1
                print(f"[latent][warm_start] skip {k}: {tuple(old_v.shape)} vs {tuple(new_v.shape)}", flush=True)
        self.load_state_dict(new_state)
        return dict(loaded=n_loaded, mapped=n_mapped, skipped=n_skipped)


def legacy_actor_from_state(old_state: dict, hidden=(512, 256, 128), num_actions=29) -> nn.Sequential:
    """Rebuild the Stage 1 teacher's 170-D actor (behavior-cloning teacher and warm-start equivalence check)."""
    layers, d = [], 170
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, num_actions))
    actor = nn.Sequential(*layers)
    actor.load_state_dict({k[len("actor."):]: v for k, v in old_state.items() if k.startswith("actor.")})
    return actor.eval()


@torch.no_grad()
def warm_start_equivalence(ac: "LatentActorCritic", old_state: dict, n=2048, device="cpu") -> tuple:
    """Self-check of the warm-start mapping; returns two numbers.

    ENC_INIT=identity:
      exact  = max|new_actor(obs) - legacy(obs')| where obs' carries pos'=P·tanh(pos/P),
               vel'=V·tanh(vel/V): the mapping is exact, so this must be ~1e-5.
      lin    = mean|new_actor(obs) - legacy(obs)|: the tanh-linearisation effect the
               PPO continuation has to absorb (information only).
    ENC_INIT=random (blind start): with z = 0 the new actor must equal the legacy actor
    with its box channels zeroed (exact), plus mean|a(z) - a(0)| (information only).
    Random obs with priv pos/vel == legacy estimated pos/vel (|pos|<1.5 m, |vel|<3 m/s,
    blind_age 0, trust 0, window visible)."""
    # TF32 matmul (enabled by the training script) turns an exact identity into ~5e-3 action
    # error; run the proof in full precision so "exact" means exact.
    _tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    legacy = legacy_actor_from_state(old_state).to(device)
    obs = torch.randn(n, PRIV_HI, device=device)
    pos = (torch.rand(n, 3, device=device) * 2 - 1) * 1.5
    vel = (torch.rand(n, 3, device=device) * 2 - 1) * 3.0
    obs[:, LEG_POS] = pos
    obs[:, LEG_VEL] = vel
    obs[:, LEG_BLIND] = 0.0
    obs[:, LEG_TRUST] = 0.0
    obs[:, PRIV_LO:PRIV_LO + 3] = pos
    obs[:, PRIV_LO + 3:PRIV_LO + 6] = vel
    obs[:, PRIV_LO + 6] = 0.0
    obs[:, PRIV_LO + 7] = 1.0
    obs[:, PRIV_LO + 8:PRIV_LO + 11] = torch.tensor([0.17, 0.13, 0.11], device=device)
    was_training = ac.training
    ac.eval()
    # mask the extra z dims (random-initialised, see E_PRIV_EXTRA_INIT) so the proof tests only the mapping
    z = ac.encode(obs[:, PRIV_LO:PRIV_HI]); z[:, 6:] = 0.0
    if ENC_INIT == "random":
        # blind-start proof: with z = 0 the new actor must equal the legacy actor with its box channels zeroed
        z = torch.zeros_like(z)
        a_new = ac.actor(torch.cat([obs[:, :KEEP], z], dim=-1))
        o0 = obs[:, :170].clone(); o0[:, KEEP:170] = 0.0
        a_old0 = legacy(o0)
        zz = ac.encode(obs[:, PRIV_LO:PRIV_HI])
        a_z = ac.actor(torch.cat([obs[:, :KEEP], zz], dim=-1))
        ac.train(was_training)
        torch.backends.cuda.matmul.allow_tf32 = _tf32
        print(f"[latent][warm_start] blind start: z std per dim {[round(float(x), 3) for x in zz.std(0)]}; "
              f"mean|a(z)-a(0)| = {float((a_z - a_new).abs().mean()):.4f}", flush=True)
        return float((a_new - a_old0).abs().max()), float((a_z - a_new).abs().mean())
    a_new = ac.actor(torch.cat([obs[:, :KEEP], z], dim=-1))
    a_old = legacy(obs[:, :170])
    obs_lin = obs[:, :170].clone()
    obs_lin[:, LEG_POS] = POS_SCALE * torch.tanh(pos / POS_SCALE)
    obs_lin[:, LEG_VEL] = VEL_SCALE * torch.tanh(vel / VEL_SCALE)
    a_lin = legacy(obs_lin)
    ac.train(was_training)
    torch.backends.cuda.matmul.allow_tf32 = _tf32
    return float((a_new - a_lin).abs().max()), float((a_new - a_old).abs().mean())
