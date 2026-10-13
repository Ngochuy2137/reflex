"""Torso uprightness penalty (V5h layer) plus re-exports of the cradle terms.

The pelvis (root) pitch under-reads the visible back-lean, which is a combined
root + hip + waist fold. This term measures the torso_link tilt from vertical
instead and penalizes only the excess beyond a dead-zone. It applies every step:
the over-lean forms during the reach, while the object is still in flight.
"""
from __future__ import annotations

import torch

from reflex.envs.base.chest_v5_terms import compute_strict_components
# Re-export the cradle terms unchanged (weights are set in the env cfg).
from reflex.envs.base.chest_v5g_terms import (  # noqa: F401
    bilateral_enclosure_reward,
    ball_lateral_offset_penalty,
    low_hold_reward,
)


def _torso_quat(env) -> torch.Tensor:
    """Torso (chest) body quaternion in world frame, (B,4) wxyz.

    Uses the same `torso_link` body as compute_strict_components, so this frame is
    the pocket frame.
    """
    robot = env.scene["robot"]
    # Reuse the cached body-id lookup from chest_v5_terms.
    from reflex.envs.base.chest_v5_terms import _get_body_ids
    ids = _get_body_ids(env)
    return robot.data.body_quat_w[:, ids["chest"], :]


def torso_uprightness_penalty(
    env,
    max_angle_rad: float = 0.349,  # 20°
) -> torch.Tensor:
    """Penalize TORSO tilt beyond max_angle_rad. Returns -excess (rad), <= 0.

    Uses the torso's projected gravity g_torso = R(torso_quat)^T · (0,0,-1), which is
    (0,0,-1) when upright. The tilt atan2(||g_xy||, -g_z) is the total deviation from
    vertical, so forward and backward over-lean are both penalized.

    The dead-zone keeps the modest lean a chest cradle needs; only the excess is
    penalized, linearly.
    """
    q = _torso_quat(env)  # (B,4) wxyz
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Projected gravity in the body frame: g_b = R^T (0,0,-1) = -(third row of R).
    # Third row of R (world z axis expressed in the body frame), wxyz quaternion:
    #   gx = 2(xz - wy) ; gy = 2(yz + wx) ; gz = 1 - 2(x^2 + y^2)   [this is R[2,:]]
    gx = 2.0 * (x * z - w * y)
    gy = 2.0 * (y * z + w * x)
    gz = 1.0 - 2.0 * (x * x + y * y)
    # Gravity points "down" in the body frame: -R[2,:]
    gbx, gby, gbz = -gx, -gy, -gz
    tilt = torch.atan2(torch.sqrt(gbx * gbx + gby * gby).clamp_min(1e-8), (-gbz).clamp_min(1e-8))
    excess = torch.clamp(tilt - float(max_angle_rad), min=0.0)
    return -torch.nan_to_num(excess, nan=0.0, posinf=0.0, neginf=0.0)
