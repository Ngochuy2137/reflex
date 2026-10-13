"""Stage 2 environment config: the Stage 1 teacher environment plus the stateless privileged box term.

Additive subclass only.  The policy observation grows from 170 to 170 + PRIV_DIM (term 12 =
box_priv_stateless: 181-D, or 196-D with the rich variant).  Everything else in the Stage 1 teacher
environment (state estimator, two-stage arm trigger from perception, rewards, curriculum) is
inherited unchanged; LatentActorCritic decides what the actor actually reads (obs[0:160] ⊕ z).
"""
from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from reflex.envs.teacher_env_cfg import G1BoxCatchStepCatchEnvCfg
from reflex.latent import latent_obs as lo


@configclass
class G1BoxCatchLatentEnvCfg(G1BoxCatchStepCatchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.box_priv_stateless = ObsTerm(func=lo.box_priv_stateless)
        self.events.latent_reset = EventTerm(func=lo.latent_reset, mode="reset")
        print(f"[latent_env] policy obs += box_priv_stateless ({lo.PRIV_DIM}) -> expect {170 + lo.PRIV_DIM}; "
              f"win={lo.LAT_WIN} delay=[{lo.LAT_DELAY_LO},{lo.LAT_DELAY_HI}] fov={lo.LAT_FOV_DEG} "
              f"drop_p={lo.LAT_DROP_P}", flush=True)
