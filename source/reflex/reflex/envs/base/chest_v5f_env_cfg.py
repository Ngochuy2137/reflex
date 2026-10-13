"""Pelvis-lean layer: adds a pelvis pitch penalty on top of the chest-hold layer.

Penalizes |pelvis pitch| beyond a 30 deg dead-zone, linearly in the excess (rad).
The weight is read from the env var M3_LEAN_W (default 6.0). Observation and
action spaces are unchanged (166-D / 29-D).
"""
from __future__ import annotations

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from reflex.envs.base.chest_v5c_env_cfg import G1CatchChestV5cEnvCfg
from reflex.envs.base import chest_v5_terms as mdp_v5


# Penalty weight on |pelvis pitch| beyond 30 deg (env var M3_LEAN_W, default 6.0).
_LEAN_W = float(os.environ.get("M3_LEAN_W", "6.0"))


@configclass
class G1CatchChestV5fEnvCfg(G1CatchChestV5cEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- Pelvis pitch penalty ----
        # Zero inside the 30 deg dead-zone, linear in the excess beyond it. The
        # weight is high enough to bite on extreme (70 deg+) folds but stays below
        # the hold-time catch reward (~9), so the policy keeps the catch and leans
        # less instead of dropping it.
        self.rewards.pelvis_pitch_penalty = RewTerm(
            func=mdp_v5.pelvis_pitch_penalty,
            weight=_LEAN_W,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_angle_rad": 0.524,  # 30°
            },
        )
        print(f"[env] pelvis_pitch_penalty weight = {_LEAN_W} (M3_LEAN_W)", flush=True)
