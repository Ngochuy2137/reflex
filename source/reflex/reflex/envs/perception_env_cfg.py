"""Perception-noise layer of the Stage 1 teacher environment.

With BOX_DP_EST=1 the actor's box position/velocity come from a simulated
detector feeding a Kalman filter (perception_estimator) instead of ground
truth, and the actor also sees the filter's blind age and trust channels.
The critic keeps ground truth (asymmetric actor-critic).
"""
from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from reflex.envs import perception_estimator as v31
from reflex.envs.catch_env_cfg import G1BoxCatchV1EnvCfg

# Observation contract (policy, 170-D):
#     channel 166       blind_age
#     channels 167-169  trust = [jitter, freshness, frozen]
# The critic observation (292-D) is unchanged.  To warm-start from a 167-D
# actor, copy actor.0.weight[:, :167] and zero-init columns 167-169, so the
# first iteration reproduces the parent policy exactly.


@configclass
class G1BoxCatchSafetyDREnvCfg(G1BoxCatchV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # channel 166: blind age of the box track
        self.observations.policy.blind_age = ObsTerm(func=v31.box_blind_age)
        # channels 167-169: trust scalars.  Always present, so the observation
        # width is 170 whether or not the estimator is enabled.
        self.observations.policy.box_trust = ObsTerm(func=v31.box_trust)

        # Optional leg-speed penalty (BOX_V31_VIOLENCE_W > 0 enables it; off by
        # default).  See leg_violence_penalty for the scale reasoning.
        _vw = float(os.environ.get("BOX_V31_VIOLENCE_W", "0"))
        if _vw > 0:
            self.rewards.leg_violence = RewTerm(
                func=v31.leg_violence_penalty, weight=-_vw)
            print(f"[perception] leg-violence penalty enabled, w={_vw}", flush=True)

        if os.environ.get("BOX_DP_EST", "0") == "1":
            self.observations.policy.balloon_pos_b.func = \
                v31.balloon_pos_pelvis_frame_dp
            self.observations.policy.balloon_lin_vel_b.func = \
                v31.balloon_lin_vel_pelvis_frame_dp
            self.events.dp_reset_estimator = EventTerm(
                func=v31.dp_reset_estimator, mode="reset",
                params={"asset_cfg": SceneEntityCfg("balloon")},
            )
            print("[perception] actor box obs <- simulated detector + Kalman "
                  "filter, dirty-perception curriculum "
                  f"(noise {v31.V31_MIX_NOISE} jump {v31.V31_MIX_JUMP} "
                  f"freeze {v31.V31_MIX_FREEZE} garbage {v31.V31_MIX_GARBAGE})"
                  "; trust channels 167-169", flush=True)
        else:
            print("[perception] estimator off: 170-D obs, box obs are ground truth",
                  flush=True)
