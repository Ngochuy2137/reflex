"""Catch env variant with a random per-env balloon radius and a leg-stance penalty.

Extends G1CatchEnvCfg: samples a balloon radius in [0.10, 0.25] m at every reset
(spawned at BALLOON_RADIUS_SPAWN), adds the normalized radius as an observation
for actor and critic, and adds a leg-stance penalty (anti-lunge, anti-overspread).
Buoyancy, drag and wind keep their default-radius values.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from .catch_env_cfg import G1CatchEnvCfg


def _make_balloon_cfg_spawn(spawn_radius: float) -> RigidObjectCfg:
    """Same physics as catch_env_cfg._make_balloon_cfg, but spawned at the given
    radius (randomize_balloon_radius resizes it per env at reset).
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Balloon",
        spawn=sim_utils.SphereCfg(
            radius=spawn_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=5.0,
                angular_damping=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mdp.BALLOON_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=5.0,
                dynamic_friction=5.0,
                friction_combine_mode="max",
                restitution=0.0,
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 2.5)),
    )


@configclass
class G1CatchRadiusFinetuneEnvCfg(G1CatchEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Spawn the balloon at the mid-range radius; resized per env at reset.
        self.scene.balloon = _make_balloon_cfg_spawn(mdp.BALLOON_RADIUS_SPAWN)

        # Randomize the radius at reset. Added after super().__post_init__(), so
        # it is registered (and runs) after the parent's reset events, including
        # reset_balloon.
        self.events.randomize_balloon_radius = EventTerm(
            func=mdp.randomize_balloon_radius,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("balloon"),
                "radius_range": (mdp.BALLOON_RADIUS_MIN, mdp.BALLOON_RADIUS_MAX),
                "spawn_radius": mdp.BALLOON_RADIUS_SPAWN,
            },
        )

        # Balloon radius observation (normalized to [-1, +1]) for both
        # actor and critic.
        self.observations.policy.balloon_radius = ObsTerm(func=mdp.balloon_radius_obs)
        self.observations.critic.balloon_radius = ObsTerm(func=mdp.balloon_radius_obs)

        # Leg-stance penalty (anti-lunge + anti-overspread), moderate weight.
        self.rewards.leg_stance_penalty = RewTerm(
            func=mdp.leg_stance_penalty,
            weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "lateral_max": 0.30,
            },
        )
