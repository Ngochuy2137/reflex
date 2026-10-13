"""Torso-upright layer: the last base layer, subclassed by the box-catch env.

Replaces the pelvis lean penalty with `torso_uprightness_penalty` (torso_link tilt
from vertical, 20 deg dead-zone, weight from env var M5_TORSO_W, default 6.0) and
raises the cradle weights (bilateral_enclosure 2.0, low_hold 2.0,
ball_lateral_offset 1.5) so the object is held by the arms rather than by leaning back.
"""
from __future__ import annotations

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from reflex.envs.base.chest_v5g_env_cfg import G1CatchChestV5gEnvCfg
from reflex.envs.base import chest_v5h_terms as mdp_v5h


# Torso uprightness penalty weight beyond 20 deg (env var M5_TORSO_W, default 6.0).
_TORSO_W = float(os.environ.get("M5_TORSO_W", "6.0"))


@configclass
class G1CatchChestV5hEnvCfg(G1CatchChestV5gEnvCfg):
    def __post_init__(self):
        super().__post_init__()  # chest hold + pelvis lean + cradle terms (w=1.0)

        # ---- 1. Remove the pelvis pitch penalty (replaced by the torso term). ----
        self.rewards.pelvis_pitch_penalty = None

        # ---- 2. Torso uprightness penalty: tilt > 20° -> linear -excess. ----
        self.rewards.torso_uprightness_penalty = RewTerm(
            func=mdp_v5h.torso_uprightness_penalty,
            weight=_TORSO_W,
            params={"max_angle_rad": 0.349},  # 20°
        )

        # ---- 3. Stronger cradle, so the object is held by a tighter arm wrap
        #         instead of by leaning back. ----
        self.rewards.bilateral_enclosure.weight = 2.0
        self.rewards.low_hold.weight = 2.0
        self.rewards.ball_lateral_offset.weight = 1.5

        print(f"[env] torso_uprightness_penalty w={_TORSO_W} thr=20deg; "
              f"cradle bilateral=2.0 low_hold=2.0 lateral=1.5; pelvis_pitch OFF",
              flush=True)
