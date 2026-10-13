"""Cradle-stability layer: adds three cradle-shape rewards on top of the lean layer.

  - bilateral_enclosure  (+1.0)  farther wrist near the object: both arms wrap
  - ball_lateral_offset  (+1.0)  object off the torso centerline beyond 8 cm
  - low_hold             (+1.0)  object held low in the chest pocket
Terms live in `chest_v5g_terms`. Observation and action spaces are unchanged.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from reflex.envs.base.chest_v5f_env_cfg import G1CatchChestV5fEnvCfg
from reflex.envs.base import chest_v5g_terms as mdp_v5g


@configclass
class G1CatchChestV5gEnvCfg(G1CatchChestV5fEnvCfg):
    def __post_init__(self):
        super().__post_init__()  # chest-hold rewards + pelvis lean penalty

        # ---- Cradle-stability terms (weight 1.0 each) ----
        self.rewards.bilateral_enclosure = RewTerm(
            func=mdp_v5g.bilateral_enclosure_reward,
            weight=1.0,
            params={"sigma": 0.18},
        )
        self.rewards.ball_lateral_offset = RewTerm(
            func=mdp_v5g.ball_lateral_offset_penalty,
            weight=1.0,
            params={"deadzone": 0.08, "near_max": 0.6},
        )
        self.rewards.low_hold = RewTerm(
            func=mdp_v5g.low_hold_reward,
            weight=1.0,
            params={"z_low": 0.10, "z_high": 0.30, "near_max": 0.6},
        )
        print("[env] added cradle terms: bilateral_enclosure, ball_lateral_offset, "
              "low_hold (all weight 1.0)", flush=True)
