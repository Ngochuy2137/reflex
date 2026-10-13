"""Stage 1 teacher environment: the stepping-catch task.

Builds on the safety/robustness env and adds per-episode aim offsets (including
far "abandon" throws the robot should let go), cradle perturbations with an
anchor reward, and the reactive two-stage arm reference.  Most additions are
opt-in BOX_V38_* knobs (terms in teacher_terms); the obs contract stays 170-D.
"""
from __future__ import annotations

import os
import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from reflex.envs import teacher_terms as v38x
from reflex.envs.robust_env_cfg import G1BoxCatchSafeRobustEnvCfg


@configclass
class G1BoxCatchStepCatchEnvCfg(G1BoxCatchSafeRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Optional URDF override (G1_URDF_OVERRIDE).  The stock training URDF
        # has no collision geometry on the rubber hands (only wrist-yaw r=0.05
        # and forearm r=0.035 cylinders), so the policy learns a handless,
        # chest-and-forearm catch; main_hands.urdf adds a box per hand from the
        # mesh bounding box.  Unset = stock asset.  Applies to training and
        # evaluation alike (same cfg path).
        _urdf = os.environ.get("G1_URDF_OVERRIDE", "")
        if _urdf:
            self.scene.robot.spawn.asset_path = _urdf
            print(f"[teacher] G1 URDF override active: {_urdf}", flush=True)

        # ---- aim-offset curriculum --------------------------------------
        self.events.box_hover_launch.func = v38x.v38_walking_hover_and_launch
        self.events.v38_aim_offsets = EventTerm(
            func=v38x.sample_v38_aim_offsets, mode="reset")
        print(f"[teacher] aim-offset curriculum: P(off)={v38x.V38_P_OFFSET} "
              f"P(abandon)={v38x.V38_P_ABANDON} "
              f"abandon {v38x.V38_ABANDON_LO}-{v38x.V38_ABANDON_HI} m",
              flush=True)

        # ---- optional flight reward gate (teacher_terms._apply_flight_rgate):
        # scales selected tracking rewards while a throw is in flight.
        # Wrapping cfg.func here does not work: the config round-trip
        # re-resolves each term by its function name and silently drops the
        # wrapper.  So the gate patches env.reward_manager at runtime, after
        # env construction, and prints a FIRED line the first time it
        # actually scales a reward.
        if os.environ.get("BOX_V38_FLIGHT_RGATE", "0") == "1":
            print("[teacher] flight reward gate: reward manager will be patched "
                  "at runtime (watch for the FIRED line)", flush=True)

        # ---- optional wide box-mass randomisation (startup event; off by default)
        if os.environ.get("BOX_V38_MASS_DR", "0") == "1":
            self.events.v38_wide_mass_dr = EventTerm(
                func=v38x.v38_wide_mass_dr, mode="startup")
            print(f"[teacher] wide mass DR enabled: "
                  f"U({v38x.V38_MASS_LO},{v38x.V38_MASS_HI}) kg", flush=True)

        # ---- optional throw-bearing randomisation (so the robot learns to turn
        # toward the thrower).  The hover point p0 is placed by
        # throw.sample_throws() up to three times per episode start:
        #   1. the reset event, which runs before the motion command teleports
        #      the robot, i.e. off a stale facing;
        #   2. first_throw_realign (interval event, BOX_THROW_REALIGN=1) at
        #      episode step <= 1, which re-samples the throw off the
        #      post-teleport facing and overwrites any earlier rotation;
        #   3. in-episode rethrows (reset_box_throw from the hover event).
        # The fan half-width is a module global read at call time, so widening
        # it here makes all three placement sites draw bearing ~ U(+-B) around
        # the robot's live facing.  B = 0 (default) leaves the sampler as is.
        if float(os.environ.get("BOX_V38_BEARING_DEG", "0")) > 0:
            from reflex.envs import throw as _bte
            _old_fan = _bte.BEARING_RANGE_DEG
            _bte.BEARING_RANGE_DEG = v38x.V38_BEARING_DEG
            print(f"[teacher] bearing randomisation: fan {_old_fan:g} -> "
                  f"±{v38x.V38_BEARING_DEG:g} deg (module-global patch; "
                  "reset + realign + rethrows all inherit)", flush=True)

        # ---- optional mass curriculum (per-episode re-roll, ramping ceiling)
        if float(os.environ.get("BOX_V38_MASS_RAMP", "0")) > 0:
            self.events.v38_mass_ramp = EventTerm(
                func=v38x.v38_mass_dr_reset, mode="reset")
            print(f"[teacher] mass curriculum: {v38x.V38_MASS_HI} -> "
                  f"{v38x.V38_MASS_HI_FINAL} kg over {v38x.V38_MASS_RAMP:g} steps",
                  flush=True)

        # ---- optional clamp-excess penalty (keep commanded targets in range)
        _cw = float(os.environ.get("BOX_V38_CLAMP_W", "0"))
        if _cw > 0:
            self.rewards.v38_clamp_excess = RewTerm(
                func=v38x.v38_clamp_excess, weight=-_cw)
            print(f"[teacher] clamp-excess penalty enabled: -{_cw} per rad",
                  flush=True)

        # ---- optional: re-centre the motion clip so yaw_offsets rotate it in
        # place rather than about the env origin.  Registered before
        # anchor-face so the clip is already centred when the first slew happens.
        if os.environ.get("BOX_V38_RECENTRE", "0") == "1":
            self.events.v38_recentre = EventTerm(
                func=v38x.v38_recentre_clip_on_anchor, mode="startup")
            print("[teacher] clip re-centre enabled (yaw_offsets will rotate "
                  "in place instead of sweeping a 1.68 m arc)", flush=True)

        # ---- optional pre-launch arm-hang reward (see v38_hang_prelaunch_reward)
        if float(os.environ.get("BOX_V38_HANG_W", "0")) > 0:
            from isaaclab.managers import RewardTermCfg as _RewTerm
            self.rewards.v38_hang_prelaunch = _RewTerm(
                func=v38x.v38_hang_prelaunch_reward,
                weight=float(os.environ.get("BOX_V38_HANG_W", "0")))
            print("[box_v38] HANG-PRELAUNCH reward ON w=%s (armed; FIRED line "
                  "must appear at runtime)" % os.environ.get("BOX_V38_HANG_W"),
                  flush=True)

        # ---- optional: slew the reference-motion yaw toward the box --------
        if float(os.environ.get("BOX_V38_ANCHOR_FACE", "0")) > 0:
            self.events.v38_anchor_face = EventTerm(
                func=v38x.v38_anchor_face_box, mode="interval",
                interval_range_s=(0.0, 0.0))
            print(f"[teacher] anchor-face enabled: slew {v38x.V38_ANCHOR_FACE:g} deg/s",
                  flush=True)

        # ---- optional orbiting thrower: the throw origin walks an arc around
        # the robot.  The walking thrower wanders a disc around its own spawn
        # point, which at 3.0-3.5 m throw distance spans only +-23 deg of
        # bearing.  Registered after the walking thrower so it composes on
        # top of it.
        if float(os.environ.get("BOX_V38_ORBIT_DPS", "0")) > 0:
            self.events.v38_orbit_thrower = EventTerm(
                func=v38x.v38_orbit_thrower, mode="interval",
                interval_range_s=(0.0, 0.0))
            print(f"[teacher] orbiting thrower: {v38x.V38_ORBIT_DPS:g} deg/s "
                  f"around the robot, arc +-{v38x.V38_ORBIT_SPAN:g} deg",
                  flush=True)

        # ---- optional facing reward (turn toward the box) -----------------
        _fw = float(os.environ.get("BOX_V38_FACE_W", "0"))
        if _fw > 0:
            self.rewards.v38_face_box = RewTerm(
                func=v38x.v38_face_box, weight=_fw)
            print(f"[teacher] facing reward +{_fw} "
                  f"(dead {v38x.V38_FACE_DEAD} deg, sigma {v38x.V38_FACE_SIG})",
                  flush=True)

        # ---- reactive arms: penalty for raising the arms before the throw.
        # Gating the ready-pose rewards and the reference arms was not enough:
        # the arms still rose during the hover (17 cm vs 18 cm in flight)
        # because the catch bonus outweighs the tracking penalty.  The term
        # returns a negative deviation, so its weight is positive.
        if v38x.V38_ARM_EARLY_W > 0:
            self.rewards.v38_arm_early = RewTerm(
                func=v38x.v38_arm_early_raise_penalty,
                weight=v38x.V38_ARM_EARLY_W)
            print(f"[teacher] arm early-raise penalty registered w={v38x.V38_ARM_EARLY_W:g}",
                  flush=True)

        # ---- reactive arms: the reference motion itself.  The clip holds the
        # arms up for ~70% of its length and every catch lands inside that
        # stretch, so gating rewards alone cannot win against it.  While
        # waiting, serve the arm joints from the clip's own arms-down window.
        if v38x.V38_REF_ARMS_REACTIVE:
            v38x.patch_reference_arms()

        # reactive arms: two-stage arm POSITION TARGET.  Rewards and the
        # reference cannot move the arms on their own because the arm targets
        # are pinned by config (ARM_INIT_CRADLE default pose + ARM_SCALE 0.02
        # => the policy owns +-1.15 deg of the shoulder).
        if v38x.V38_ARM_TWO_STAGE:
            v38x.patch_arm_two_stage()

        # joint-limit target clamp, applied LAST so it sees the final target
        # (the real robot clamps targets to the URDF limits the same way).
        if v38x.V38_TGT_CLIP:
            v38x.patch_target_clip()

        # optional anchor-position error (odometry drift of the robot's own
        # anchor estimate).  Also drop the anchor_pos termination: on hardware
        # nothing resets the robot when the estimate drifts past 0.25 m, so
        # keeping it would end episodes on the first step and measure nothing.
        if (v38x.V38_AZ_RATE != 0.0 or v38x.V38_AZ_LO != 0.0
                or v38x.V38_AZ_HI != 0.0 or any(a != 0.0 for a in v38x.V38_AB)):
            v38x.patch_anchor_z_drift()
            if getattr(self.terminations, "anchor_pos", None) is not None:
                self.terminations.anchor_pos = None
                print("[teacher] anchor_pos termination disabled for anchor-error "
                      "injection (hardware has no such reset)", flush=True)

        # ---- evaluation-time plant-mismatch probes (all off by default) ----
        # BOX_FOOT_FRICTION pins foot-ground friction (training randomises it
        # over static 0.3-1.6 / dynamic 0.3-1.2) to test whether a slipping
        # support foot explains short real-robot steps.  BOX_LEG_EFFORT_SCALE
        # caps leg torque (see teacher_terms.apply_leg_effort_scale).
        # BOX_LEG_KP_SCALE scales the trained leg/foot kp (kd with it, at the
        # stock ratio): a policy trained on one plant, run on another.
        _fr = float(os.environ.get("BOX_FOOT_FRICTION", "0"))
        if _fr > 0 and getattr(self.events, "physics_material", None) is not None:
            _p = self.events.physics_material.params
            _p["static_friction_range"] = (_fr, _fr)
            _p["dynamic_friction_range"] = (_fr * 0.75, _fr * 0.75)
            print("[box_v38] FOOT FRICTION pinned: static %.2f / dynamic %.2f "
                  "(was %s / %s)" % (_fr, _fr * 0.75, "(0.3,1.6)", "(0.3,1.2)"),
                  flush=True)

        v38x.apply_leg_effort_scale(self.scene.robot)
        _leg_sc = float(os.environ.get("BOX_LEG_KP_SCALE", "0"))
        if _leg_sc > 0:
            for _grp in ("legs", "feet"):
                _a = self.scene.robot.actuators.get(_grp)
                if _a is None:
                    continue
                for _tbl in (_a.stiffness, _a.damping):
                    if isinstance(_tbl, dict):
                        for _j in list(_tbl.keys()):
                            _tbl[_j] = _tbl[_j] * _leg_sc
            print(f"[teacher] leg gain scale: legs+feet kp,kd x{_leg_sc} "
                  "(eval-time plant-mismatch probe; effort limits unchanged)",
                  flush=True)

        # ---- reactive arms: ready-pose rewards only once the throw is live.
        # Without this the robot raises both arms at the first step and holds
        # them all episode, which reads as a prop.  Gate the ready-pose rewards
        # on "throw in flight or box held" so the idle pose is owned by the
        # clip (arms down); BOX_ARM_SNAP must be off too, or the first step
        # pins the arms up regardless.
        if v38x.V38_ARMS_REACTIVE:
            _n = 0
            for _t in ("cradle_pose", "palm_up", "arm_idle_still",
                       "cradle_ready", "cradle_ready_dev"):
                _r = getattr(self.rewards, _t, None)
                if _r is not None:
                    _r.func = v38x.wrap_reactive(_r.func)
                    _n += 1
            print(f"[teacher] reactive arms: {_n} ready-pose rewards gated "
                  f"on box-in-flight (ARM_SNAP="
                  f"{os.environ.get('BOX_ARM_SNAP', '0')})", flush=True)

        # ---- cradle perturbation + anchor reward ---------------------------
        if os.environ.get("BOX_V38_PERTURB", "1") == "1":
            self.events.v38_cradle_perturb = EventTerm(
                func=v38x.v38_cradle_perturb, mode="interval",
                interval_range_s=(0.5, 1.0))
            print(f"[teacher] cradle perturb p={v38x.V38_PERTURB_P} "
                  f"v<={v38x.V38_PERTURB_V} w<={v38x.V38_PERTURB_W}",
                  flush=True)
        _aw = float(os.environ.get("BOX_V38_ANCHOR_W", "0.3"))
        if _aw > 0:
            self.rewards.v38_cradle_anchor = RewTerm(
                func=v38x.v38_cradle_anchor, weight=_aw)
            print(f"[teacher] cradle anchor reward +{_aw} "
                  f"(sigma {v38x.V38_ANCHOR_SIG})", flush=True)

        # ---- optional reward-parameter overrides (BOX_V38_EEF_SIG and
        # BOX_V38_ANCHOR_POS_STD, applied at the end of this method).  They
        # change no observation.  Under ground-truth perception the catch rate
        # drops sharply beyond a 0.4 m aim offset, and two mechanisms could
        # cause it:
        #   A  the pull toward the box is too short-range: catch_eef_to_balloon
        #      is exp(-d^2/sigma^2) with sigma 0.2-0.5, i.e. ~0 while the box
        #      is still 1 m away, so a failed chase earns nothing;
        #   B  the position anchor (motion_global_anchor_pos, std 0.3) keeps
        #      charging after the robot has stepped off the reference, so a
        #      half-finished step is a permanent loss.
        # ---- optional: predicted landing point as an observation (+3 dims,
        # obs 170 -> 173).  The training script pads the first actor layer when
        # the obs dim grows, so a 170-D checkpoint still warm-starts.
        if os.environ.get("BOX_V38_LAND_OBS", "0") == "1":
            from isaaclab.managers import ObservationTermCfg as ObsTerm
            self.observations.policy.balloon_landing_b = ObsTerm(
                func=v38x.balloon_landing_pelvis_frame)
            print(f"[teacher] landing-point obs enabled (+3 dims: xy in pelvis "
                  f"frame + t, plane {v38x.LAND_OBS_H} m)", flush=True)

        # ---- optional: anchor_ori yaw target -> box bearing.  Replaces the
        # func of motion_global_anchor_ori but keeps its weight (0.5) and std
        # (0.4), so the reward budget is unchanged and only the yaw target
        # moves.  No observation changes.
        if os.environ.get("BOX_V38_ORI_FACE", "0") == "1":
            _t = getattr(self.rewards, "motion_global_anchor_ori", None)
            if _t is not None:
                _t.func = v38x.v38_anchor_ori_facebox
                _yh = v38x.ORI_YAW_STD_HOVER
                print(f"[teacher] anchor_ori yaw target -> box "
                      f"bearing (weight {_t.weight}, std "
                      f"{_t.params.get('std')}, live above {v38x.ORI_FACE_H} m, "
                      f"hover yaw std "
                      f"{_yh if _yh > 0 else 'SAME (v3 fix OFF)'})", flush=True)
            else:
                print("[teacher] anchor_ori yaw-to-box requested but "
                      "motion_global_anchor_ori not found; not enabled",
                      flush=True)

        # ---- optional go-to-landing-point reward ------------------------
        _lw = float(os.environ.get("BOX_V38_LAND_W", "0"))
        if _lw > 0:
            self.rewards.v38_go_to_landing = RewTerm(
                func=v38x.v38_go_to_landing, weight=_lw)
            print(f"[teacher] go-to-landing reward +{_lw} "
                  f"(sigma {v38x.BOX_LAND_SIG}, catch_h {v38x.BOX_LAND_H})",
                  flush=True)

        _eefsig = os.environ.get("BOX_V38_EEF_SIG", "")
        if _eefsig and hasattr(self.rewards, "catch_eef_to_balloon"):
            _old = self.rewards.catch_eef_to_balloon.params.get("sigma")
            self.rewards.catch_eef_to_balloon.params["sigma"] = float(_eefsig)
            print(f"[teacher] catch_eef_to_balloon sigma "
                  f"{_old} -> {_eefsig}", flush=True)
        # ---- optional: relax the motion_body_lin_vel tracking term ---------
        # It is exp(-mean_b|v_ref - v_b|^2 / std^2) against a reference whose
        # body velocities are exactly zero for the last 851 frames of the clip
        # (57% of it, including the whole catch window), so there it carries
        # no information and only prices motion; it is the largest transient
        # brake on stepping.  std 1.0 -> 2.0 divides the exponent by 4: the
        # stepping factor goes 0.651 -> 0.898 while standing still is still
        # rewarded by the same exp shape.
        _lvstd = os.environ.get("BOX_V38_LINVEL_STD", "")
        if _lvstd and hasattr(self.rewards, "motion_body_lin_vel"):
            _old = self.rewards.motion_body_lin_vel.params.get("std")
            self.rewards.motion_body_lin_vel.params["std"] = float(_lvstd)
            print(f"[teacher] motion_body_lin_vel std {_old} -> {_lvstd}",
                  flush=True)

        _apstd = os.environ.get("BOX_V38_ANCHOR_POS_STD", "")
        if _apstd and hasattr(self.rewards, "motion_global_anchor_pos"):
            _old = self.rewards.motion_global_anchor_pos.params.get("std")
            self.rewards.motion_global_anchor_pos.params["std"] = float(_apstd)
            print(f"[teacher] motion_global_anchor_pos std "
                  f"{_old} -> {_apstd}", flush=True)
