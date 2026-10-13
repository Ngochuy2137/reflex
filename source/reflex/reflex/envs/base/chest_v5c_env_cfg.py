"""Chest-hold layer: strict chest-cradle catch rewards on top of the catch env.

Extends the BeyondMimic catch env (`G1CatchEnvCfg`) with reset-time augmentation
(perturbed start state, motion-phase mixture), a randomized drop height, and the
strict chest-hold reward suite from `chest_v5_terms`. First of the base layers the
box-catch env inherits from: V5c (chest hold) -> V5f (pelvis lean) -> V5g (cradle
stability) -> V5h (torso upright) -> box catch.
"""
from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from reflex.envs.base import chest_v5_terms as mdp_v5
from reflex.envs.base import catch_terms_v2 as mdp_v2

from whole_body_tracking.tasks.tracking.config.g1.catch_env_cfg import G1CatchEnvCfg


@configclass
class G1CatchChestV5cEnvCfg(G1CatchEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # No radius randomization and no radius observation (166-D policy obs):
        # one physical radius, the 0.1016 m spawn default. Changing a sphere's
        # radius at runtime resizes only the visual mesh, not the PhysX collider.

        # ---- Reset augmentation: perturbed start state + motion-phase mixture ----
        self.events.randomize_init_state = EventTerm(
            func=mdp_v2.randomize_init_state,
            mode="reset",
            params={"joint_pose_std": 0.3, "root_vel_xy_max": 1.0},
        )
        self.events.randomize_cmd_time_steps = EventTerm(
            func=mdp_v2.randomize_cmd_time_steps_mixture,
            mode="reset",
            params={
                "motion_total_frames": 1500,
                "p_normal": 0.5,
                "p_catch_window": 0.3,
                "p_global": 0.2,
                "catch_window": (250, 800),
            },
        )

        # ---- Clear the strict-hold persistence counter on reset ----
        self.events.reset_chest_v5_persist = EventTerm(
            func=mdp_v5.reset_chest_v5_persist,
            mode="reset",
        )

        # ---- Drop spawn: 0.8-2.0 m ahead, +-0.5 m lateral, height 4-7 m ----
        if hasattr(self.events, "reset_balloon"):
            params = self.events.reset_balloon.params
            params["spawn_height"] = 5.5
            params["forward_distance_range"] = (0.8, 2.0)
            params["lateral_range"] = (-0.5, 0.5)
        self.events.randomize_balloon_height = EventTerm(
            func=mdp_v5.randomize_balloon_height,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("balloon"),
                "height_min": 4.0,
                "height_max": 7.0,
            },
        )

        # ====================================================================
        # REWARDS: zero the loose catch terms (names the parent does not define
        # are skipped), then add the strict chest-hold suite.
        # ====================================================================
        for name in (
            "reactive_hold",
            "reactive_hold_persist",
            "chest_funnel",
            "upright_recover",
            "catch_success_bonus",   # loose proximity + slow bonus
            "leg_stance_penalty",
            "bilateral_wrist_enclosure",
        ):
            if hasattr(self.rewards, name):
                term = getattr(self.rewards, name)
                if term is not None and hasattr(term, "weight"):
                    term.weight = 0.0

        self.rewards.chest_pocket = RewTerm(
            func=mdp_v5.chest_pocket_reward,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        self.rewards.wrist_proximity = RewTerm(
            func=mdp_v5.wrist_proximity_reward,
            weight=1.5,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        self.rewards.wrist_flank = RewTerm(
            func=mdp_v5.wrist_flank_reward,
            weight=1.5,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        self.rewards.head_shoulder_support_penalty = RewTerm(
            func=mdp_v5.head_shoulder_support_penalty,
            weight=-3.0,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        self.rewards.strict_hold_sustained = RewTerm(
            func=mdp_v5.strict_hold_sustained_reward,
            weight=4.0,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        self.rewards.catch_success_strict_bonus = RewTerm(
            func=mdp_v5.catch_success_strict_bonus,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        self.rewards.lean_during_intake = RewTerm(
            func=mdp_v2.lean_during_intake_reward,
            weight=0.2,
            params={},
        )
