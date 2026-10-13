"""AmpPPO: rsl_rl 3.0.1 PPO with an Adversarial Motion Prior discriminator (Peng et al. 2021).

Used by both Stage 1 (teacher) and Stage 2 (latent controller) training.  Hand-crafted style
terms alone did not yield a human-like catch; AMP supplies a learned style reward instead.
The discriminator is trained online (expert transition pairs from the AMP dataset vs. policy
pairs from rollouts) and its score becomes a dense style reward mixed into the task reward
BEFORE storage (hooked in process_env_step -- no upstream rsl_rl file is modified; the training
scripts inject the class into the runner's namespace).

Feature layout MUST match tools/motion/build_amp_dataset.py exactly:
  joint_pos(29) joint_vel(29) root_h(1) ori6d(6) lin_vel_loc(3) ang_vel_loc(3)
  key_body_rel(12: left/right wrist, left/right ankle) = 83
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO

KEY_BODY_IDX = [28, 29, 18, 19]   # left_wrist_yaw, right_wrist_yaw, left_ankle_roll, right_ankle_roll
PELVIS_IDX = 0


class AMPDiscriminator(nn.Module):
    def __init__(self, in_dim: int = 166, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


def _heading_inv_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = torch.zeros_like(q)
    out[:, 0] = torch.cos(-yaw / 2)
    out[:, 3] = torch.sin(-yaw / 2)
    return out


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[:, :1]
    u = q[:, 1:]
    uv = torch.cross(u, v, dim=-1)
    uuv = torch.cross(u, uv, dim=-1)
    return v + 2 * (w * uv + uuv)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _rot6d(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    c0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                      2 * (x * z - w * y)], dim=-1)
    c1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
                      2 * (y * z + w * x)], dim=-1)
    return torch.cat([c0, c1], dim=-1)


class AmpPPO(PPO):
    """PPO + AMP style reward. Call set_amp_context() before learn()."""

    amp_ready: bool = False

    def set_amp_context(self, env, dataset_path: str, w_style: float = 1.0,
                        w_task: float = 1.0, disc_lr: float = 1e-4,
                        disc_steps: int = 4, batch: int = 4096,
                        buf_size: int = 200_000, r1_coef: float = 5.0,
                        logit_reg: float = 0.05, style_ramp_iters: int = 300,
                        disc_skip_below: float = 0.3):
        self._amp_env = env  # ManagerBasedRLEnv (unwrapped)
        data = torch.load(dataset_path, map_location=self.device)
        self._expert = data["features"].to(self.device)          # (N,2,83)
        st = data["stats"]
        self._mu = st["mean"].to(self.device)
        self._sig = st["std"].to(self.device)
        self._disc = AMPDiscriminator().to(self.device)
        # Anti-runaway measures from the reference AMP implementation
        # (IsaacGymEnvs amp_continuous.py). Without them the discriminator
        # separates expert and policy almost perfectly (loss ~0.08) and the
        # style gradient vanishes; logit regularisation, weight decay and
        # interleaved updates keep the separation soft:
        self._disc_opt = torch.optim.Adam(self._disc.parameters(), lr=disc_lr,
                                          weight_decay=1e-4)
        self._logit_reg = logit_reg
        self._style_ramp_iters = style_ramp_iters
        self._disc_skip_below = disc_skip_below
        self._iter_count = 0
        self._last_disc_loss = 1.0
        self._w_style, self._w_task = w_style, w_task
        self._disc_steps, self._batch = disc_steps, batch
        self._r1 = r1_coef
        n = env.num_envs
        self._prev = torch.zeros(n, 83, device=self.device)
        self._prev_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._pbuf = torch.zeros(buf_size, 2, 83, device=self.device)
        self._pbuf_n = 0
        self._pbuf_ptr = 0
        self._style_ema = 0.0
        # ---- Optional multi-GPU discriminator sync (AMP_DDP=1,
        # default off -> single-GPU path unchanged). rsl_rl 3.0.1's own
        # multi-GPU path (PPO.broadcast_parameters / reduce_parameters) only
        # covers the actor-critic; the disc lives outside it, so without this
        # block each rank trains a private disc that silently diverges.
        # NOTE: the AMP feature normalizer (_mu/_sig) is STATIC, loaded from
        # the same dataset file on every rank -> no sync needed.
        self._ddp = (os.environ.get("AMP_DDP", "0") == "1"
                     and torch.distributed.is_available()
                     and torch.distributed.is_initialized()
                     and torch.distributed.get_world_size() > 1)
        self._ddp_probe = os.environ.get("AMP_DDP_PROBE", "0") == "1"
        self._ddp_probe_every = int(os.environ.get("AMP_DDP_PROBE_EVERY", "50"))
        self._ddp_rank = torch.distributed.get_rank() if self._ddp else 0
        self._ddp_world = torch.distributed.get_world_size() if self._ddp else 1
        if self._ddp:
            # per-rank seeds differ (the training script adds local_rank), so disc init
            # differs -> make rank0's weights authoritative before training
            for p in self._disc.parameters():
                torch.distributed.broadcast(p.data, src=0)
            print(f"[amp_ddp] rank {self._ddp_rank}/{self._ddp_world}: disc "
                  f"params broadcast from rank0; disc grads all-reduced per "
                  f"step; disc_loss + pbuf gate rank-synced", flush=True)
        if self._ddp_probe:
            print(f"[amp_ddp] probe enabled on rank {self._ddp_rank}: "
                  f"disc/actor-critic param sums at iter 1 then every "
                  f"{self._ddp_probe_every}",
                  flush=True)
        self.amp_ready = True
        print(f"[AMP] context set: expert pairs {tuple(self._expert.shape)}, "
              f"w_style {w_style} w_task {w_task} disc_steps {disc_steps}",
              flush=True)

    @torch.no_grad()
    def _features(self) -> torch.Tensor:
        robot = self._amp_env.scene["robot"]
        d = robot.data
        jp, jv = d.joint_pos, d.joint_vel
        rp = d.body_pos_w[:, PELVIS_IDX]
        rq = d.body_quat_w[:, PELVIS_IDX]
        hinv = _heading_inv_quat(rq)
        h = rp[:, 2:3]
        ori = _rot6d(_quat_mul(hinv, rq))
        lv = _quat_rotate(hinv, d.body_lin_vel_w[:, PELVIS_IDX])
        av = _quat_rotate(hinv, d.body_ang_vel_w[:, PELVIS_IDX])
        keys = [
            _quat_rotate(hinv, d.body_pos_w[:, bi] - rp) for bi in KEY_BODY_IDX
        ]
        return torch.cat([jp, jv, h, ori, lv, av] + keys, dim=-1)

    def _norm(self, pairs: torch.Tensor) -> torch.Tensor:
        # clamp ±5: policy features drifting outside the dataset stats' tails
        # must not hand the disc free separation dimensions
        return ((pairs - self._mu) / self._sig).clamp(-5.0, 5.0).flatten(start_dim=-2)

    def process_env_step(self, obs, rewards, dones, extras):
        if self.amp_ready:
            feats = self._features()
            pair = torch.stack([self._prev, feats], dim=1)      # (n,2,83)
            with torch.no_grad():
                logit = self._disc(self._norm(pair)).squeeze(-1)
                prob = torch.sigmoid(logit)
                style = -torch.log((1.0 - prob).clamp(min=1e-4))
                style = style.clamp(max=5.0)
            style = torch.where(self._prev_valid, style,
                                torch.zeros_like(style))
            r = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
            # w_style ramp-in (a guard for warm-started training that the AMP
            # paper does not need): early on, the policy must improve on the
            # task gradient alone; a full style reward from the first
            # iteration stalled task learning.
            ramp = min(1.0, self._iter_count / max(1, self._style_ramp_iters))
            mixed = self._w_task * r + (self._w_style * ramp) * style
            rewards = mixed.unsqueeze(-1) if rewards.dim() > 1 else mixed
            # stash policy pairs (only valid ones)
            vidx = self._prev_valid.nonzero(as_tuple=False).squeeze(-1)
            if len(vidx):
                m = len(vidx)
                idx = (self._pbuf_ptr + torch.arange(m, device=self.device)) \
                    % self._pbuf.shape[0]
                self._pbuf[idx] = pair[vidx]
                self._pbuf_ptr = int((self._pbuf_ptr + m) % self._pbuf.shape[0])
                self._pbuf_n = min(self._pbuf_n + m, self._pbuf.shape[0])
            self._style_ema = 0.99 * self._style_ema + 0.01 * style.mean().item()
            self._prev = feats
            done_f = dones.squeeze(-1).bool() if dones.dim() > 1 else dones.bool()
            self._prev_valid = ~done_f
        return super().process_env_step(obs, rewards, dones, extras)

    def _disc_update(self):
        losses = []
        for _ in range(self._disc_steps):
            ei = torch.randint(0, self._expert.shape[0], (self._batch,),
                               device=self.device)
            pi = torch.randint(0, self._pbuf_n, (self._batch,),
                               device=self.device)
            e = self._norm(self._expert[ei]).requires_grad_(True)
            p = self._norm(self._pbuf[pi])
            le = self._disc(e)
            lp = self._disc(p)
            # BCE: expert -> 1, policy -> 0 (matches IsaacGymEnvs AMP)
            loss = nn.functional.binary_cross_entropy_with_logits(
                le, torch.ones_like(le)) + \
                nn.functional.binary_cross_entropy_with_logits(
                lp, torch.zeros_like(lp))
            # R1 gradient penalty on expert
            grad = torch.autograd.grad(le.sum(), e, create_graph=True)[0]
            loss = loss + self._r1 * grad.pow(2).sum(dim=-1).mean()
            # logit L2 regularization (as in the reference implementation:
            # keeps the separation soft so the style gradient never dies)
            loss = loss + self._logit_reg * (le.pow(2).mean() + lp.pow(2).mean())
            self._disc_opt.zero_grad()
            loss.backward()
            if getattr(self, "_ddp", False):
                # Multi-GPU: average disc grads across ranks (mirrors what
                # PPO.reduce_parameters does for the actor-critic). Identical
                # params + identical averaged grads -> identical Adam steps
                # -> disc stays in lockstep on every rank.
                for p in self._disc.parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(
                            p.grad, op=torch.distributed.ReduceOp.SUM)
                        p.grad /= self._ddp_world
            self._disc_opt.step()
            losses.append(loss.item())
        avg = sum(losses) / len(losses)
        if getattr(self, "_ddp", False):
            # Multi-GPU: share one disc_loss value across ranks: the adaptive
            # skip gate in update() compares it, and a rank-divergent gate
            # would make some ranks enter _disc_update (collective ops) while
            # others skip -> NCCL deadlock.
            t = torch.tensor([avg], device=self.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            avg = float(t.item()) / self._ddp_world
        return avg

    def update(self):
        if self.amp_ready:
            self._iter_count += 1
            disc_ready = self._pbuf_n >= self._batch
            if getattr(self, "_ddp", False):
                # Multi-GPU: rank-consistent gate. pbuf fill rate differs per
                # rank (valid-pair counts depend on local dones); _disc_update
                # runs collectives, so all ranks must take the same branch.
                t = torch.tensor([1.0 if disc_ready else 0.0],
                                 device=self.device)
                torch.distributed.all_reduce(
                    t, op=torch.distributed.ReduceOp.MIN)
                disc_ready = bool(t.item() > 0.5)
            if disc_ready:
                # adaptive balance: when the disc is winning hard, let the
                # policy catch up (cheap stand-in for official interleaving)
                if self._last_disc_loss >= self._disc_skip_below:
                    self._last_disc_loss = self._disc_update()
                else:
                    self._last_disc_loss = min(1.0, self._last_disc_loss + 0.05)
                ramp = min(1.0, self._iter_count / max(1, self._style_ramp_iters))
                print(f"[AMP] disc_loss {self._last_disc_loss:.3f} "
                      f"style_ema {self._style_ema:.3f} ramp {ramp:.2f} "
                      f"pbuf {self._pbuf_n}", flush=True)
            if getattr(self, "_ddp_probe", False) and (
                    self._iter_count == 1
                    or self._iter_count % self._ddp_probe_every == 0):
                # Multi-GPU probe: cross-rank sync check. Param sums must be
                # IDENTICAL on every rank (disc after this iter's disc update,
                # actor-critic after the previous iter's synced PPO update).
                with torch.no_grad():
                    dsum = sum(p.double().sum().item()
                               for p in self._disc.parameters())
                    acsum = sum(p.double().sum().item()
                                for p in self.policy.parameters())
                print(f"[amp_ddp_probe] rank {getattr(self, '_ddp_rank', 0)} "
                      f"iter {self._iter_count} disc_sum {dsum:.10e} "
                      f"ac_sum {acsum:.10e}", flush=True)
        return super().update()
