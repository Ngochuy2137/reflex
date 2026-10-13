"""Batched torch Kalman filter for the thrown box (perception noise + Kalman filter layer).

A per-env vectorised port of the box filter that runs on the robot. The policy is
trained on this filter's output rather than on white-noise-corrupted ground truth:
deployed tracking error is the coherent, extrapolation-shaped output of a filter fed
sparse, noisy detections, which a policy trained on white noise does not learn to handle.
"""
from __future__ import annotations

import os
import torch

# ---- constants (identical to the filter running on the robot) --------------
G = 9.81                 # gravity [m/s^2]
BOX_K = 0.85             # linear drag F = -k*m*v (same k as the simulated box)
CTRL_DT = 0.02           # 50 Hz policy tick
SA2 = 40.0 ** 2          # acceleration process-noise variance
R_MEAS = 0.03            # default measurement std [m]
VMAX_BOX = 7.0           # physical bound on box speed [m/s]
KF_CHEST_R = 1.2         # near-chest zone [m]: coast instead of blind re-init
REINIT_MISS = 12         # ticks without a measurement before re-acquisition
GATE_MIN = 0.4           # innovation gate floor [m]
FD_MISS_MAX = 6          # launch finite-difference (FD) velocity init only within 6 ticks
FD_V_MIN = 3.0           # FD init fires above this implied speed [m/s]
FD_V_STATIC = 1.0        # ...and only while the filter believes the box is static
P_INIT = 0.5             # covariance on (re)acquisition
P_FD = 1.0               # velocity covariance after FD init
# ---- cold-start FD velocity init, variant 2 (BOX_KF_FD_INIT_V2=1) -----------
# Default rule (above): FD velocity init only when miss>0 and |fd|>3.0.  At 50 fps
# the second detection usually arrives on the very next tick (miss==0), so the rule
# never fires and the velocity estimate climbs from 0 under gravity
# (0.2/0.4/0.6 m/s per 20 ms), delaying the arm response.  v2 allows miss 0..6 and
# accepts KF_FD_INIT_VMIN < |fd| < KF_FD_INIT_VMAX while |v| < 1.0.  Off by default.
KF_FD_INIT_V2 = int(os.environ.get("BOX_KF_FD_INIT_V2", "0"))
KF_FD_INIT_VMIN = float(os.environ.get("BOX_KF_FD_INIT_VMIN", "2.0"))
KF_FD_INIT_VMAX = float(os.environ.get("BOX_KF_FD_INIT_VMAX", "10.0"))
if KF_FD_INIT_V2:
    print(f"[kalman] cold-start FD velocity init v2 active: miss 0..{FD_MISS_MAX}, |v|<{FD_V_STATIC}, "
          f"{KF_FD_INIT_VMIN} < |fd| < {KF_FD_INIT_VMAX}", flush=True)


class BoxKalmanBatch:
    """N independent box filters advanced in lockstep.

    `have` marks an initialised filter (a position estimate exists) and
    `have_zprev` marks that a previous accepted measurement exists.

    All positions are expected in a per-env LOCAL frame (world minus the env
    origin).  On the robot the filter runs in a frame whose origin is the
    robot's start pose, so the absolute guard thresholds (z<-0.5, z>6.0,
    |xy|>12) only carry their intended meaning in a robot-local frame -- in
    Isaac's spawn grid a world-frame filter would trip |xy|>12 on distant envs.
    """

    def __init__(self, n: int, device, dt: float = CTRL_DT):
        self.n = n
        self.dev = device
        self.dt = dt
        z3 = lambda: torch.zeros(n, 3, device=device)          # noqa: E731
        self.p = z3()
        self.v = z3()
        self.P = torch.zeros(n, 6, 6, device=device)
        self.z_prev = z3()
        self.chest = z3()
        self.have = torch.zeros(n, dtype=torch.bool, device=device)
        self.have_zprev = torch.zeros(n, dtype=torch.bool, device=device)
        self.miss = torch.zeros(n, dtype=torch.long, device=device)
        self.n_upd = torch.zeros(n, dtype=torch.long, device=device)
        # diagnostics (counters, not semantics)
        self.n_vbound = torch.zeros(n, dtype=torch.long, device=device)
        self.n_coast = torch.zeros(n, dtype=torch.long, device=device)
        self.n_gate = torch.zeros(n, dtype=torch.long, device=device)
        self.n_reinit = torch.zeros(n, dtype=torch.long, device=device)
        self._build_FQ()

    def _build_FQ(self):
        dt = self.dt
        F = torch.eye(6, device=self.dev)
        F[:3, 3:] = torch.eye(3, device=self.dev) * dt
        F[3:, 3:] *= (1.0 - BOX_K * dt)              # drag on velocity
        Q = torch.zeros(6, 6, device=self.dev)
        I3 = torch.eye(3, device=self.dev)
        Q[:3, :3] = I3 * (SA2 * dt ** 4 / 4)
        Q[:3, 3:] = Q[3:, :3] = I3 * (SA2 * dt ** 3 / 2)
        Q[3:, 3:] = I3 * (SA2 * dt ** 2)             # white-acceleration process noise
        self.F, self.Q = F, Q

    # ------------------------------------------------------------------
    def reset_idx(self, idx: torch.Tensor):
        """Full reset of the selected envs (episode boundary, runaway guard)."""
        if idx.numel() == 0:
            return
        self.have[idx] = False
        self.have_zprev[idx] = False
        self.miss[idx] = 0
        self.n_upd[idx] = 0
        self.p[idx] = 0.0
        self.v[idx] = 0.0
        self.P[idx] = 0.0
        self.z_prev[idx] = 0.0

    # ------------------------------------------------------------------
    def predict(self):
        """Ballistic-with-drag prediction step.

        Envs whose filter is not initialised (`have` False) are left untouched.
        """
        h = self.have
        if not bool(h.any()):
            return
        dt = self.dt
        self.miss = torch.where(h, self.miss + 1, self.miss)
        g = torch.tensor([0.0, 0.0, -G], device=self.dev)
        a = g - BOX_K * self.v                                  # gravity + drag
        p_new = self.p + self.v * dt + 0.5 * a * dt * dt
        v_new = self.v + a * dt
        P_new = self.F @ self.P @ self.F.T + self.Q
        hm = h[:, None]
        self.p = torch.where(hm, p_new, self.p)
        self.v = torch.where(hm, v_new, self.v)
        self.P = torch.where(h[:, None, None], P_new, self.P)

    # ------------------------------------------------------------------
    def update(self, z: torch.Tensor, valid: torch.Tensor, r: float = R_MEAS):
        """Measurement update with the deployed filter's gating logic.

        `valid` marks the envs that have a measurement `z` this tick.  Envs whose
        measurement the filter REJECTS simply stay blind (miss keeps growing).
        """
        dt = self.dt
        val = valid.clone()

        near_chest = (self.have
                      & ((self.p - self.chest).norm(dim=-1) < KF_CHEST_R))

        # ---- (a) measurement-to-measurement implied-speed bound ----------
        # Only for tracked pairs (0 < miss <= 12).
        dtm = (self.miss.clamp(min=1).float() * dt)
        v_impl = (z - self.z_prev).norm(dim=-1) / dtm
        rej_a = (val & self.have & self.have_zprev
                 & (self.miss > 0) & (self.miss <= REINIT_MISS)
                 & (v_impl > VMAX_BOX))
        self.n_vbound += rej_a.long()
        val = val & ~rej_a

        # ---- (b) near-chest: blind re-init DISABLED, coast instead -------
        coast_zone = val & self.have & (self.miss > REINIT_MISS) & near_chest
        rej_b = coast_zone & ((z - self.p).norm(dim=-1) > GATE_MIN)
        self.n_coast += rej_b.long()
        val = val & ~rej_b
        # accepted inside the zone: clamp miss to 12 so the re-init branch
        # below is bypassed but the normal gate/gain path runs
        clamp = coast_zone & val
        self.miss = torch.where(clamp, torch.full_like(self.miss, REINIT_MISS),
                                self.miss)

        # snapshot for the post-update ||v|| restore
        snap_have = self.have.clone()
        snap = (self.p.clone(), self.v.clone(), self.P.clone(),
                self.miss.clone(), self.z_prev.clone(),
                self.have_zprev.clone())

        # ---- (re)acquire branch ------------------------------------------
        reinit = val & (~self.have | (self.miss > REINIT_MISS))
        if bool(reinit.any()):
            rm = reinit[:, None]
            self.p = torch.where(rm, z, self.p)
            self.v = torch.where(rm, torch.zeros_like(self.v), self.v)
            eye6 = torch.eye(6, device=self.dev) * P_INIT
            self.P = torch.where(reinit[:, None, None], eye6, self.P)
            self.n_upd = torch.where(reinit, torch.ones_like(self.n_upd),
                                     self.n_upd)
            self.miss = torch.where(reinit, torch.zeros_like(self.miss),
                                    self.miss)
            self.z_prev = torch.where(rm, z, self.z_prev)
            self.have = self.have | reinit
            self.have_zprev = self.have_zprev | reinit
            self.n_reinit += reinit.long()

        normal = val & ~reinit

        # ---- innovation gate (active once n_upd > 5) ---------------------
        thresh = torch.maximum(
            torch.full_like(self.p[:, 0], GATE_MIN),
            3.0 * self.P[:, 0, 0].clamp(min=1e-9).sqrt())
        gate_rej = normal & (self.n_upd > 5) & ((z - self.p).norm(dim=-1) > thresh)
        self.n_gate += gate_rej.long()
        normal = normal & ~gate_rej

        # ---- launch-only FD velocity re-init -----------------------------
        fd_v = (z - self.z_prev) / dtm[:, None]
        if KF_FD_INIT_V2:
            _fdn = fd_v.norm(dim=-1)
            fd = (normal & self.have_zprev & (self.miss >= 0)
                  & (self.miss <= FD_MISS_MAX)
                  & (self.v.norm(dim=-1) < FD_V_STATIC)
                  & (_fdn > KF_FD_INIT_VMIN) & (_fdn < KF_FD_INIT_VMAX))
        else:
            fd = (normal & self.have_zprev & (self.miss > 0)
                  & (self.miss <= FD_MISS_MAX)
                  & (self.v.norm(dim=-1) < FD_V_STATIC)
                  & (fd_v.norm(dim=-1) > FD_V_MIN))
        if bool(fd.any()):
            self.v = torch.where(fd[:, None], fd_v, self.v)
            eye3 = torch.eye(3, device=self.dev) * P_FD
            Pvv = torch.where(fd[:, None, None], eye3, self.P[:, 3:, 3:])
            self.P = self.P.clone()
            self.P[:, 3:, 3:] = Pvv

        # ---- Kalman gain update ------------------------------------------
        if bool(normal.any()):
            I3 = torch.eye(3, device=self.dev)
            S = self.P[:, :3, :3] + I3 * (r ** 2)          # H P H^T + R
            PHt = self.P[:, :, :3]                          # P H^T  (n,6,3)
            K = PHt @ torch.linalg.inv(S)                   # (n,6,3)
            innov = (z - self.p)[:, :, None]                # (n,3,1)
            x = torch.cat([self.p, self.v], dim=-1)[:, :, None] + K @ innov
            I6 = torch.eye(6, device=self.dev)
            P_upd = (I6 - K @ self._H()) @ self.P
            nm = normal[:, None]
            self.p = torch.where(nm, x[:, :3, 0], self.p)
            self.v = torch.where(nm, x[:, 3:, 0], self.v)
            self.P = torch.where(normal[:, None, None], P_upd, self.P)
            self.z_prev = torch.where(nm, z, self.z_prev)
            self.have_zprev = self.have_zprev | normal
            self.n_upd = self.n_upd + normal.long()
            self.miss = torch.where(normal, torch.zeros_like(self.miss),
                                    self.miss)

        # ---- (a) post-update ||v|| sanity: restore on violation ----------
        bad = (normal & snap_have & (self.v.norm(dim=-1) > VMAX_BOX))
        if bool(bad.any()):
            bm = bad[:, None]
            self.p = torch.where(bm, snap[0], self.p)
            self.v = torch.where(bm, snap[1], self.v)
            self.P = torch.where(bad[:, None, None], snap[2], self.P)
            self.miss = torch.where(bad, snap[3], self.miss)
            self.z_prev = torch.where(bm, snap[4], self.z_prev)
            self.have_zprev = torch.where(bad, snap[5], self.have_zprev)
            self.n_vbound += bad.long()

    def _H(self):
        H = torch.zeros(3, 6, device=self.dev)
        H[:, :3] = torch.eye(3, device=self.dev)
        return H

    # ------------------------------------------------------------------
    def runaway_guard(self):
        """Reset filters whose estimate left the physically possible
        envelope.  Positions are env-local (see class docstring).
        """
        bad = self.have & ((self.p[:, 2] < -0.5) | (self.p[:, 2] > 6.0)
                           | (self.p[:, :2].norm(dim=-1) > 12.0))
        self.reset_idx(bad.nonzero(as_tuple=False).flatten())
        return bad

    def contact_freeze(self):
        """Contact assumption of the deployed filter: once the box is blind AND
        close to the chest AND not receding, hold its velocity at zero.
        Returns the frozen mask."""
        rel = self.chest - self.p
        frozen = (self.have & (self.miss > 2)
                  & (rel.norm(dim=-1) < 0.75)
                  & ((self.v * rel).sum(dim=-1) <= 0.5))
        self.v = torch.where(frozen[:, None], torch.zeros_like(self.v), self.v)
        return frozen

    def blind_age(self) -> torch.Tensor:
        """Blind-age observation channel (index 166 of the policy observation):
        1.0 if the filter is uninitialised, else min(miss, 15) / 15."""
        age = self.miss.clamp(max=15).float() / 15.0
        return torch.where(self.have, age, torch.ones_like(age))
