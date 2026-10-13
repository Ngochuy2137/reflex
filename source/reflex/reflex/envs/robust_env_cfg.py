"""Safety/robustness layer of the Stage 1 teacher environment.

Builds on the perception-noise env: adds a delay arm to the dirty-perception
curriculum (robust_estimator), trust-gated catch rewards, low-trust stillness,
soft-fall shaping, a pre-clamp command-limit penalty, a one-shot fall penalty,
and a walking thrower (robust_terms).  The observation contract is unchanged.
"""
from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from reflex.envs import robust_estimator as v36e
from reflex.envs import robust_terms as v36x
from reflex.envs.perception_env_cfg import G1BoxCatchSafetyDREnvCfg

# Weight defaults follow the scale accounting in the robust_terms header.
# A long pre-throw hold can be enabled with BOX_HOVER_LO/HI.


@configclass
class G1BoxCatchSafeRobustEnvCfg(G1BoxCatchSafetyDREnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- estimator: point every perception hook at robust_estimator ---
        self.observations.policy.blind_age.func = v36e.box_blind_age
        self.observations.policy.box_trust.func = v36e.box_trust
        if os.environ.get("BOX_DP_EST", "0") == "1":
            self.observations.policy.balloon_pos_b.func = \
                v36e.balloon_pos_pelvis_frame_dp
            self.observations.policy.balloon_lin_vel_b.func = \
                v36e.balloon_lin_vel_pelvis_frame_dp
            self.events.dp_reset_estimator.func = v36e.dp_reset_estimator
            print(f"[robust] estimator -> robust_estimator (delay arm "
                  f"{v36e.V36_MIX_DELAY}, ticks {v36e.V36_DELAY_LO}-"
                  f"{v36e.V36_DELAY_HI})", flush=True)

        # ---- 1. trust-gate the catch rewards ------------------------------
        # Capture whatever func the parent config left on each term (some env
        # knobs reroute them in the parent __post_init__), then dispatch
        # through the module-level gate so RewTerm stays yaml-serializable.
        if os.environ.get("BOX_V36_TRUSTGATE", "1") == "1":
            for key, attr in (("sustained", "strict_hold_sustained"),
                              ("strict", "catch_success_strict_bonus"),
                              ("threept", "threepoint_bonus"),
                              ("on_bed", "box_on_bed")):
                term = getattr(self.rewards, attr, None)
                if term is None:
                    continue
                v36x.register_inner(key, term.func)
                term.func = v36x.trust_gated_reward
                term.params = {**term.params, "inner": key}
            print("[robust] trust gate on catch rewards "
                  f"(floor {v36x.V36_GATE_FLOOR})", flush=True)

        # ---- optional: trust-gate ALL box-conditioned dense rewards and
        # penalise action amplitude under distrust.  Ungated dense shaping can
        # pay the policy to thrash toward a phantom box (|a| up to the clip)
        # even while the frozen-trust channel fires.  Off by default.
        if os.environ.get("BOX_V37_GATE_ALL", "0") == "1":
            _gated = []
            for attr in ("catch_eef_to_balloon", "low_hold", "cradle_ready",
                         "cradle_pose", "face_box", "hands_under_box",
                         "chest_pocket", "upright_hold", "hold_still",
                         "chest_delivery", "chest_contact",
                         "wrist_proximity", "hold_feet_gather",
                         "palm_up"):
                term = getattr(self.rewards, attr, None)
                if term is None or term.func is v36x.trust_gated_reward:
                    continue
                if not set(term.params.keys()) <= {
                        "asset_cfg", "hold_sec", "sigma",
                        "z_low", "z_high", "near_max"}:
                    print(f"[robust] skip trust gate on {attr} (params "
                          f"{list(term.params.keys())})", flush=True)
                    continue
                v36x.register_inner(attr, term.func)
                term.func = v36x.trust_gated_reward
                term.params = {**term.params, "inner": attr}
                _gated.append(attr)
            print(f"[robust] trust gate on {len(_gated)} dense terms: "
                  f"{_gated}", flush=True)
        _aw = float(os.environ.get("BOX_V37_AMP_W", "0"))
        if _aw > 0:
            self.rewards.distrust_action_amp = RewTerm(
                func=v36x.distrust_action_amp, weight=-_aw)
            print(f"[robust] distrust action-amplitude penalty w=-{_aw}",
                  flush=True)

        # ---- 2-5. safety terms (weights env-tunable) ----------------------
        _sw = float(os.environ.get("BOX_V36_STILL_W", "1.5"))
        if _sw > 0:
            self.rewards.low_trust_stillness = RewTerm(
                func=v36x.low_trust_stillness, weight=_sw)
        _fw = float(os.environ.get("BOX_V36_SOFTFALL_W", "2.0"))
        if _fw > 0:
            self.rewards.soft_fall_shaping = RewTerm(
                func=v36x.soft_fall_shaping, weight=-_fw)
        _cw = float(os.environ.get("BOX_V36_CMDLIM_W", "2.0"))
        if _cw > 0:
            self.rewards.cmd_limit_excess = RewTerm(
                func=v36x.cmd_limit_excess, weight=-_cw)
        _tw = float(os.environ.get("BOX_V36_FALLTERM_W", "50.0"))
        if _tw > 0:
            self.rewards.fall_terminated = RewTerm(
                func=v36x.fall_terminated, weight=-_tw)
        print(f"[robust] terms: still +{_sw} softfall -{_fw} "
              f"cmdlim -{_cw} fallterm -{_tw}", flush=True)

        # ---- 6. walking thrower -------------------------------------------
        if os.environ.get("BOX_V36_WALK_THROWER", "1") == "1":
            self.events.box_hover_launch.func = \
                v36x.v36_walking_hover_and_launch
            print(f"[robust] walking thrower (v {v36x.V36_WALK_V_LO}-"
                  f"{v36x.V36_WALK_V_HI} m/s, R {v36x.V36_WALK_R} m); set "
                  "BOX_HOVER_LO/HI for long holds", flush=True)
