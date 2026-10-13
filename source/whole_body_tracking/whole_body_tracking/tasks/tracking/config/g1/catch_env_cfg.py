"""G1 walk-then-catch env: the motion-tracking env plus a falling balloon.

Extends G1FlatEnvCfg with the balloon object and its physics (buoyancy, drag,
wind), a reset event that spawns it along the yaw-randomized motion direction,
catch rewards (wrist-midpoint proximity, sparse catch bonus) and balloon
position/velocity observations in the pelvis frame.
Reference motion: walk_then_catch__U001 (13 s: walk along -Y, then catch and hold).
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

from .flat_env_cfg import G1FlatEnvCfg


def _make_balloon_cfg() -> RigidObjectCfg:
    """Sphere balloon: radius BALLOON_RADIUS, mass BALLOON_MASS, high friction, no bounce."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Balloon",
        spawn=sim_utils.SphereCfg(
            radius=mdp.BALLOON_RADIUS,
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
class G1CatchEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Episode length must cover the 13-s walk_then_catch motion.
        self.episode_length_s = 13.0

        # Add balloon to scene.
        self.scene.balloon = _make_balloon_cfg()

        # Per-step buoy + drag + wind force.
        self.events.apply_balloon_force = EventTerm(
            func=mdp.apply_balloon_force,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=False,
            params={"asset_cfg": SceneEntityCfg("balloon"), "buoy_scale": 0.7},
        )

        # Random per-episode yaw of the reference motion, full ±π, so the
        # object can arrive from any direction.
        self.events.randomize_motion_yaw = EventTerm(
            func=mdp.randomize_motion_yaw,
            mode="reset",
            params={"yaw_range": (-3.14159, 3.14159)},
        )

        # Spawn balloon in motion-walking direction (rotated by per-env yaw),
        # 0.8-2.0 m ahead so it lands within reach.
        self.events.reset_balloon = EventTerm(
            func=mdp.reset_balloon_in_motion_direction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("balloon"),
                "spawn_height": 2.5,
                "forward_distance_range": (0.8, 2.0),
                "lateral_range": (-0.3, 0.3),
                "motion_direction": (0.0, -1.0),
            },
        )

        # Catch rewards (additive on top of motion-tracking rewards).
        # σ=0.5 keeps a usable gradient at range (d=1m → 0.02, d=0.3m → 0.7);
        # with σ=0.2 the reward is ~0 beyond 0.5 m and PPO cannot bootstrap.
        self.rewards.catch_eef_to_balloon = RewTerm(
            func=mdp.catch_eef_to_balloon,
            weight=2.0,
            params={"sigma": 0.5},
        )
        # Weight kept comparable to the motion-tracking rewards (sum ≈ 5): the
        # policy outputs full joint actions, and a much larger bonus (e.g. 40)
        # dominates the tracking terms and destabilizes training.
        self.rewards.catch_success_bonus = RewTerm(
            func=mdp.catch_success_bonus,
            weight=5.0,
            params={"trap_radius": 0.25, "speed_thresh": 0.5},
        )

        # Balloon observation: position + velocity in pelvis frame.
        # Add to BOTH policy and critic groups.
        self.observations.policy.balloon_pos_b = ObsTerm(
            func=mdp.balloon_pos_pelvis_frame,
        )
        self.observations.policy.balloon_lin_vel_b = ObsTerm(
            func=mdp.balloon_lin_vel_pelvis_frame,
        )
        self.observations.critic.balloon_pos_b = ObsTerm(
            func=mdp.balloon_pos_pelvis_frame,
        )
        self.observations.critic.balloon_lin_vel_b = ObsTerm(
            func=mdp.balloon_lin_vel_pelvis_frame,
        )
