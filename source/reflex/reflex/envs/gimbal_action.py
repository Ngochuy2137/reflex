"""Wrist-roll reflex action term (BOX_GIMBAL=1): keeps the palms facing up by construction.

As a reward, palm-up competes with the shoulder chain's roll and never fully converges.
Here it is a controller guarantee instead: every physics step each wrist_roll PD target
is solved in closed form (see ``_solve_delta``) so the palm normal has no lateral tilt
about the forearm axis. The policy's action space is unchanged (wrist action scales are 0).
"""
from __future__ import annotations

import os

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

# per-step max target change (rad) — PD smoothing on top of this
GIMBAL_RATE_LIMIT = 0.15
# wrist_roll joint limit guard (G1 spec ±1.97; stay inside)
GIMBAL_ROLL_MIN, GIMBAL_ROLL_MAX = -1.95, 1.95


class GimbalJointPositionAction(JointPositionAction):
    """JointPositionAction + wrist-roll palm-up reflex overwrite."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        robot = self._asset
        jn = robot.joint_names
        bn = robot.body_names
        # indices in the FULL joint array
        self._lroll_j = jn.index("left_wrist_roll_joint")
        self._rroll_j = jn.index("right_wrist_roll_joint")
        # indices of these joints WITHIN this action term's joint set
        self._lroll_t = self._joint_ids.index(self._lroll_j) if isinstance(
            self._joint_ids, list) else self._lroll_j
        self._rroll_t = self._joint_ids.index(self._rroll_j) if isinstance(
            self._joint_ids, list) else self._rroll_j
        # bodies for FK
        self._lfore_b = bn.index("left_wrist_roll_link")
        self._rfore_b = bn.index("right_wrist_roll_link")
        self._lpalm_b = bn.index("left_wrist_yaw_link")
        self._rpalm_b = bn.index("right_wrist_yaw_link")
        self._robot = robot
        # BOX_GIMBAL_PALM: which wrist_yaw_link axis is "the palm normal" the
        # reflex points at world up.
        #   "z" (default): local -z on both hands.  On the rubber-hand meshes
        #       (main_hands.urdf, fixed joint rpy 0) -z is the PINKY EDGE
        #       (thumb at +z), so this mode keeps the hand edge-up, not the palm.
        #   "y": anatomical palm normal from the hand mesh -- fingers curl toward
        #       -y on the left hand and +y on the right (thumb at +z on both),
        #       so left = -local_y, right = +local_y.
        # Per-hand spec keyed by the wrist_yaw_link body index -> (col, sign).
        self._palm_mode = os.environ.get("BOX_GIMBAL_PALM", "z")
        if self._palm_mode == "z":
            self._palm_spec = {self._lpalm_b: (2, -1.0), self._rpalm_b: (2, -1.0)}
        elif self._palm_mode == "y":
            self._palm_spec = {self._lpalm_b: (1, -1.0), self._rpalm_b: (1, +1.0)}
        else:
            raise ValueError(f"BOX_GIMBAL_PALM must be 'z' or 'y', got {self._palm_mode!r}")
        _ax = lambda b: ("-" if self._palm_spec[b][1] < 0 else "+") + "xyz"[self._palm_spec[b][0]]
        print(f"[gimbal] palm axis = {self._palm_mode} (left {bn[self._lpalm_b]} "
              f"local {_ax(self._lpalm_b)}, right {bn[self._rpalm_b]} local "
              f"{_ax(self._rpalm_b)})", flush=True)
        self._palm_live_n = 0
        print("[gimbal] wrist reflex ACTIVE (roll DOF, closed-form, "
              f"rate {GIMBAL_RATE_LIMIT}/step)", flush=True)
        # BOX_LIMIT_CLAMP=1: clamp every PD target to the joint limits before
        # it reaches the actuators, matching the clamp applied by the onboard
        # controller.  Off by default.  Without it a policy can learn to
        # command targets far beyond the mechanical stop (e.g. ankles
        # 1.8-2.45 rad past the limit) to saturate torque, which on hardware
        # means end-stop impacts; training with the clamp teaches catching
        # within the limits instead.
        self._limit_clamp = os.environ.get("BOX_LIMIT_CLAMP", "0") == "1"
        if self._limit_clamp:
            d = robot.data
            lim = getattr(d, "joint_pos_limits", None)
            if lim is None:
                lim = getattr(d, "joint_limits", None)
            if lim is None:
                lim = d.soft_joint_pos_limits
            ids = self._joint_ids
            self._clamp_lo = lim[:, ids, 0].clone()
            self._clamp_hi = lim[:, ids, 1].clone()
            print("[gimbal] LIMIT CLAMP ACTIVE (deploy-parity PD-target "
                  "clamp to joint limits)", flush=True)

    @staticmethod
    def _quat_axis(q: torch.Tensor, col: int) -> torch.Tensor:
        """World direction of a link-local basis axis (col 0=x,1=y,2=z)."""
        w, x, y, z = q.unbind(-1)
        if col == 0:
            return torch.stack([1 - 2 * (y * y + z * z),
                                2 * (x * y + w * z),
                                2 * (x * z - w * y)], dim=-1)
        if col == 1:
            return torch.stack([2 * (x * y - w * z),
                                1 - 2 * (x * x + z * z),
                                2 * (y * z + w * x)], dim=-1)
        if col == 2:
            return torch.stack([2 * (x * z + w * y),
                                2 * (y * z - w * x),
                                1 - 2 * (x * x + y * y)], dim=-1)
        raise ValueError(col)

    def _solve_delta(self, fore_b: int, palm_b: int) -> torch.Tensor:
        """Roll correction about the forearm axis that brings the palm normal up.

        a = forearm x-axis from FK (roll axis), p = palm normal in world,
        z = world up:  delta = atan2((p x z) . a, p . z - (p . a)(z . a)),
        the rotation about `a` that best aligns the component of p
        perpendicular to `a` with z.  Faded out as the forearm turns vertical.
        """
        qd = self._robot.data.body_quat_w
        a = self._quat_axis(qd[:, fore_b], 0)          # roll axis (forearm x)
        col, sign = self._palm_spec[palm_b]            # "z": (2, -1) = local -z
        p = self._quat_axis(qd[:, palm_b], col)        # palm normal
        if sign < 0:
            p = -p
        zw = torch.zeros_like(p)
        zw[:, 2] = 1.0
        adotz = a[:, 2]
        num = (torch.cross(p, zw, dim=-1) * a).sum(dim=-1)
        den = (p * zw).sum(dim=-1) - (p * a).sum(dim=-1) * adotz
        delta = torch.atan2(num, den.clamp(min=-1.0, max=1.0) if False else den)
        # near-singular when forearm points straight up/down: fade out
        fade = (1.0 - adotz.abs()).clamp(0.0, 1.0)
        return delta * fade

    def apply_actions(self):
        # NOTE: Isaac Lab's hook is apply_actionS (plural) — a method with the
        # singular name would silently never be called.
        # Overwrite the wrist_roll entries of the processed targets (target =
        # measured roll + rate-limited delta, clamped to the joint limits), then
        # let the parent write them to sim.
        jp = self._asset.data.joint_pos
        for fore_b, palm_b, jfull, jterm in (
                (self._lfore_b, self._lpalm_b, self._lroll_j, self._lroll_t),
                (self._rfore_b, self._rpalm_b, self._rroll_j, self._rroll_t)):
            delta = self._solve_delta(fore_b, palm_b)
            delta = delta.clamp(-GIMBAL_RATE_LIMIT, GIMBAL_RATE_LIMIT)
            tgt = (jp[:, jfull] + delta).clamp(GIMBAL_ROLL_MIN, GIMBAL_ROLL_MAX)
            self._processed_actions[:, jterm] = tgt
        # sanity print: live world-z of the configured palm normal (env 0)
        # at the 1st and 2000th physics-step call
        self._palm_live_n += 1
        if self._palm_live_n in (1, 2000):
            qd = self._robot.data.body_quat_w
            _nz = []
            for _b in (self._lpalm_b, self._rpalm_b):
                _c, _s = self._palm_spec[_b]
                _nz.append(float(_s * self._quat_axis(qd[:1, _b], _c)[0, 2]))
            print(f"[gimbal] palm live (mode {self._palm_mode}, call "
                  f"{self._palm_live_n}, env0): normal world-z L {_nz[0]:+.3f} "
                  f"R {_nz[1]:+.3f}; roll L {float(jp[0, self._lroll_j]):+.3f} "
                  f"R {float(jp[0, self._rroll_j]):+.3f}", flush=True)
        if self._limit_clamp:
            _pre = self._processed_actions
            _post = torch.clamp(_pre, self._clamp_lo, self._clamp_hi)
            # Expose the clamped-away amount so training can PENALISE
            # commanding past the limit: every clamped radian is a sim/real
            # behaviour divergence even when torques stay moderate.  Recorded
            # here, consumed by the v38_clamp_excess reward term
            # (teacher_terms.py).
            self._env._clamp_excess = (_pre - _post).abs().sum(dim=-1)
            self._processed_actions = _post
        super().apply_actions()


@configclass
class GimbalJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = GimbalJointPositionAction
