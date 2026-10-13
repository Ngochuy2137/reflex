"""Cradle-stability reward terms (V5g layer; re-exported by `chest_v5h_terms`).

bilateral_enclosure_reward (farther wrist near the object, so both arms wrap),
ball_lateral_offset_penalty (object off the torso centerline beyond a dead-zone)
and low_hold_reward (object low in the chest pocket). All reuse
`compute_strict_components` and the per-env radius buffer `env._balloon_radius_per_env`.
"""
from __future__ import annotations

import torch

from reflex.envs.base.chest_v5_terms import compute_strict_components

_DEF_R = 0.1016


def _ball_r(env, B, device):
    """Per-env physical ball radius (same logic as compute_strict_components)."""
    rad = getattr(env, "_balloon_radius_per_env", None)
    if rad is not None and isinstance(rad, torch.Tensor) and rad.shape[0] == B:
        return rad.to(device).clamp_min(0.05)
    return torch.full((B,), _DEF_R, device=device)


def bilateral_enclosure_reward(env, sigma: float = 0.18):
    """+[0,1]: reward the FARTHER wrist being near the ball surface.

    d_far = max(d_lwrist, d_rwrist) (the worse arm). reward = exp(-((d_far - r)/σ)^2).
    A one-arm clutch → d_far large → ~0. Both arms wrapping → d_far ≈ r → ~1.
    Radius-aware (subtracts the ball radius so it generalizes across sizes).
    """
    c = compute_strict_components(env)
    B, device = env.num_envs, env.device
    r = _ball_r(env, B, device)
    d_far = torch.maximum(c["d_lwrist"], c["d_rwrist"])
    x = (d_far - r).clamp_min(0.0) / float(sigma)
    return torch.exp(-(x * x))


def ball_lateral_offset_penalty(env, deadzone: float = 0.08, near_max: float = 0.6):
    """-[0,inf): penalize the ball off the torso centerline beyond ±deadzone.

    by = ball lateral position in the torso frame. penalty = -relu(|by| - deadzone),
    active only while the ball is near the body (norm(ball_in_torso) < near_max) so
    it does not act during the far approach.
    """
    c = compute_strict_components(env)
    bit = c["ball_in_torso"]
    by = bit[:, 1]
    near = (bit.norm(dim=-1) < float(near_max)).float()
    excess = torch.clamp(by.abs() - float(deadzone), min=0.0)
    return -(excess * near)


def low_hold_reward(env, z_low: float = 0.10, z_high: float = 0.30, near_max: float = 0.6):
    """+[0,1]: reward the ball held LOW in the torso pocket (not riding up to the head).

    bz = ball height in the torso frame. reward = clamp((z_high - bz)/(z_high - z_low), 0, 1):
    1 at/below z_low (chest center), 0 at z_high (head/shoulder). Gated on the ball
    being near, so it shapes the descent into a deep cradle and accrues over the hold.
    """
    c = compute_strict_components(env)
    bit = c["ball_in_torso"]
    bz = bit[:, 2]
    near = (bit.norm(dim=-1) < float(near_max)).float()
    r = torch.clamp((float(z_high) - bz) / (float(z_high) - float(z_low)), 0.0, 1.0)
    return r * near
