"""Base box-catch task config: a thrown cardboard box on the G1 motion-tracking env.

Subclasses the chest-catch env and swaps the caught object for a cardboard box (scene
entity still named "balloon" so inherited terms bind unchanged) launched by the
ballistic thrower in ``throw.py``; adds forearm-bed hold shaping, posture terms and
optional wrist/arm mechanics, all switched by BOX_* environment variables.  The Stage 1
teacher builds on it: perception_env_cfg -> robust_env_cfg -> teacher_env_cfg.
"""
from __future__ import annotations

import os

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from reflex.envs import throw as box_mdp

from reflex.envs.base.chest_v5h_env_cfg import G1CatchChestV5hEnvCfg


@configclass
class G1BoxCatchV1EnvCfg(G1CatchChestV5hEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- box replaces balloon asset (same entity name, see module doc)
        # BOX_MODEL selects the box model(s) (see throw.BOX_MODELS / MIX_SETS).
        # Note: the mini box (19 cm wide) can wedge BETWEEN the forearms
        # instead of lying on them.
        self.scene.balloon = box_mdp.make_box_cfg(
            model=os.environ.get("BOX_MODEL", "random"))
        # Heterogeneous per-env box models (MultiAssetSpawner) require real
        # per-env physics — with replicate_physics=True env_0 gets cloned and
        # every env ends up with the same model.
        self.scene.replicate_physics = False

        # ---- balloon-specific physics/drop events removed
        self.events.apply_balloon_force = None
        if hasattr(self.events, "randomize_balloon_height"):
            self.events.randomize_balloon_height = None

        # ---- ballistic thrower (reset: sample + park hovering)
        self.events.reset_balloon = EventTerm(
            func=box_mdp.reset_box_throw,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("balloon"),
                "motion_direction": (0.0, -1.0),
            },
        )
        # ---- every-step hover hold + one-shot launch
        self.events.box_hover_launch = EventTerm(
            func=box_mdp.box_hover_and_launch,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=False,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )

        # ---- fast-object contact: the box arrives at ~6-8 m/s, i.e. several
        # cm per physics step, so continuous collision detection is enabled.
        self.sim.physx.enable_ccd = True

        # Wrist reflex gimbal: palms up BY CONSTRUCTION (controller), not by
        # reward.  Kept as a TOP-LEVEL block, deliberately NOT nested in any
        # other env-var branch (a nested switch can silently become a no-op).
        if os.environ.get("BOX_GIMBAL", "0") == "1":
            from reflex.envs.gimbal_action import GimbalJointPositionAction
            self.actions.joint_pos.class_type = GimbalJointPositionAction
            print("[env] wrist gimbal reflex ON (roll DOF, closed-form)",
                  flush=True)

        # ---- reward/termination fixes for the box task ----
        # F1: the wrist z-deviation termination killed any catch attempt outside
        # the clip's catch window (the box arrives at a random clip phase).
        # Keep the fall proxy on the ankles only.
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link", "right_ankle_roll_link",
        ]
        # F2: undesired_contacts penalized chest/forearm contact — dormant for
        # a light balloon (below the force threshold) but ACTIVE for a box,
        # i.e. it penalized exactly the contact catching requires.  The anti-shelf
        # guard remains via head_shoulder_support_penalty.
        self.rewards.undesired_contacts.weight = 0.0
        # CRADLE over CLAMP: real-robot latency makes a bilateral squeeze
        # fragile, so the box is received into a chest cradle and supported
        # from BELOW.  Side-support terms pay for clamping and are zeroed;
        # under-support terms are boosted.
        if hasattr(self.rewards, "bilateral_enclosure") and self.rewards.bilateral_enclosure:
            self.rewards.bilateral_enclosure.weight = 0.0
        if hasattr(self.rewards, "wrist_flank") and self.rewards.wrist_flank:
            self.rewards.wrist_flank.weight = 0.0
        if hasattr(self.rewards, "low_hold") and self.rewards.low_hold:
            self.rewards.low_hold.weight = 3.0              # was 2.0
        if hasattr(self.rewards, "chest_pocket") and self.rewards.chest_pocket:
            self.rewards.chest_pocket.weight = 3.5          # was 2.0; 3.5 anti-pinch
        # Anti-pinch: both wrists under the box (see hands_under_box_reward).
        self.rewards.hands_under_box = RewTerm(
            func=box_mdp.hands_under_box_reward,
            weight=1.5,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        # Cradle geometry: parallel forward forearms as the bed; elbows tucked
        # (the spread allowance adapts to box size via eff_radius).
        self.rewards.forearm_cradle = RewTerm(
            func=box_mdp.forearm_cradle_reward,
            weight=1.5,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        self.rewards.elbow_spread = RewTerm(
            func=box_mdp.elbow_spread_penalty,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        # Post-catch stabilization: freeze into the carry so the box doesn't
        # get shaken off by continued clip tracking.
        self.rewards.hold_still = RewTerm(
            func=box_mdp.hold_still_reward,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        # THE bed criterion: box lying ON both near-horizontal forearms (a
        # parallel-forearm proxy alone let the forearms rotate and the box ride
        # the chest/shoulder).  Dominant hold-shaping weight.
        self.rewards.box_on_bed = RewTerm(
            func=box_mdp.box_on_forearm_bed_reward,
            weight=2.5,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        # Face the box: throws are always frontal and the robot stays squared
        # up to the box (prevents heading drift).
        self.rewards.face_box = RewTerm(
            func=box_mdp.face_box_reward,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        # HOLD-phase arm-pose prior = the reference clip's catch-window pose
        # (a joint-space anchor; purely geometric proxies were exploitable).
        self.rewards.cradle_pose = RewTerm(
            func=box_mdp.cradle_pose_reward,
            weight=2.5,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )
        # Goalkeeper draw-in: after the bed catch, pull the box to the chest.
        self.rewards.tuck_in = RewTerm(
            func=box_mdp.tuck_in_reward,
            weight=0.5,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )

        # Human-like impact absorption: yield backward + squat during the brief
        # intake window.
        self.rewards.compliant_intake = RewTerm(
            func=box_mdp.compliant_intake_reward,
            weight=1.5,
            params={"asset_cfg": SceneEntityCfg("balloon")},
        )

        # BOX_V19: reward rebalance.  Style-agnostic chest terms can fund an
        # asymmetric top/bottom grip; cut the chest reward and make the bed +
        # symmetric pose prior the dominant hold reward.  Also switches
        # throw._BED_MIX.
        if os.environ.get("BOX_V19", "0") == "1":
            self.rewards.chest_pocket.weight = 1.2   # 3.5 -> 1.2
            self.rewards.low_hold.weight = 1.2       # 3.0 -> 1.2
            self.rewards.box_on_bed.weight = 4.0     # 2.5 -> 4.0
            self.rewards.cradle_pose.weight = 3.5    # 2.5 -> 3.5
            print("[env] reward rebalance ON: chest_pocket 1.2, "
                  "low_hold 1.2, box_on_bed 4.0, cradle_pose 3.5, "
                  "bed mix (0.15,0.45,0.15,0.25)", flush=True)

        # Optional hold-symmetry terms:
        if os.environ.get("BOX_ASYM_PEN", "0") == "1":
            # anti clamp-recovery: asymmetric arms are penalised while HELD
            # (never during the reach); deadzones 5 cm / ~32 deg.
            self.rewards.forearm_asym = RewTerm(
                func=box_mdp.forearm_asym_penalty,
                weight=-0.6,
                params={"asset_cfg": SceneEntityCfg("balloon")},
            )
            print("[env] forearm_asym_penalty ON (-0.6, held-gated)",
                  flush=True)
        if os.environ.get("BOX_BED_STRICT", "0") == "1":
            # Only a bed hold counts as a hold: the sustained reward is routed
            # through the tiered "above" gate; clamped holds earn 0 sustained.
            self.rewards.strict_hold_sustained.func = \
                box_mdp.bed_gated_hold_sustained
            print("[env] strict_hold_sustained -> bed_gated (above gate)",
                  flush=True)

        # BOX_V19C: palm-up reward (palms drift off up-facing via shoulder-chain
        # twist), stronger torso lean limits, and the optional terms below.
        if os.environ.get("BOX_V19C", "0") == "1":
            self.rewards.palm_up = RewTerm(
                func=box_mdp.palm_up_reward,
                weight=1.5,
                params={"asset_cfg": SceneEntityCfg("balloon")},
            )
            # NOTE: the torso_asym override lives in the BOX_POSTURE block
            # below — torso_asym is CREATED there, later in this __post_init__,
            # so an override here would be a silent no-op.
            self.rewards.compliant_intake.weight = 1.0   # 1.5 -> 1.0
            # Optional chest/tuck weights: without a bring-it-close reward the box
            # can stay parked on straight arms far from the chest ("waiter
            # carry"); clamping is fenced off by forearm_asym + bed-gated
            # sustained + palm_up.
            _chest_w = float(os.environ.get("BOX_CHEST_W", "0"))
            if _chest_w > 0:
                self.rewards.chest_pocket.weight = _chest_w
            _tuck_w = float(os.environ.get("BOX_TUCK_W", "0"))
            if _tuck_w > 0:
                self.rewards.tuck_in.weight = _tuck_w
            # pay for the upright, slightly leaned-back tuck (see
            # upright_hold_reward docstring)
            _upr_w = float(os.environ.get("BOX_UPRIGHT_W", "0"))
            if _upr_w > 0:
                self.rewards.upright_hold = RewTerm(
                    func=box_mdp.upright_hold_reward,
                    weight=_upr_w,
                    params={"asset_cfg": SceneEntityCfg("balloon")},
                )
            # All-phase torso-lean penalty beyond a dead-zone (default 30 deg).
            # Unlike BOX_UPRIGHT_W (a held-gated reward that starts the instant
            # the box is held and so discourages the catch lunge), this is
            # silent below the dead-zone and bites only on sustained post-catch
            # bows / a backward sit.  Default OFF (weight 0).
            _lean_w = float(os.environ.get("BOX_LEAN_PEN_W", "0"))
            if _lean_w > 0:
                _lean_deg = float(os.environ.get("BOX_LEAN_PEN_DEG", "30"))
                self.rewards.lean_excess = RewTerm(
                    func=box_mdp.lean_excess_penalty,
                    weight=-_lean_w,
                    params={"asset_cfg": SceneEntityCfg("robot"),
                            "max_angle_rad": _lean_deg * 3.141592653589793 / 180.0},
                )
                print(f"[env] lean excess penalty active: w={_lean_w} "
                      f"dead-zone={_lean_deg} deg torso={os.environ.get('BOX_GATE_TORSO', '0')}")
            # Base + leg stillness penalty while the box is far AND not flying,
            # against fidgeting when no box is in sight (see
            # base_idle_still_penalty).  Default OFF.
            _bidle_w = float(os.environ.get("BOX_BASE_IDLE_W", "0"))
            if _bidle_w > 0:
                self.rewards.base_idle_still = RewTerm(
                    func=box_mdp.base_idle_still_penalty,
                    weight=-_bidle_w,
                    params={"asset_cfg": SceneEntityCfg("balloon")},
                )
                print(f"[env] base idle-still penalty active: w={_bidle_w} "
                      f"(far>1.5 m & box speed<1.0; base xy speed dead-zone 0.03 + 0.2*leg joint speed)")
            # Idle-calm arm damper + deadzoned ready-pose penalty, against
            # anticipatory arm twitching (positive rewards alone did not
            # prevent it).  See arm_idle_still_penalty / cradle_ready_dev_penalty.
            _idle_w = float(os.environ.get("BOX_IDLE_STILL_W", "0"))
            if _idle_w > 0:
                self.rewards.arm_idle_still = RewTerm(
                    func=box_mdp.arm_idle_still_penalty,
                    weight=-_idle_w,
                    params={"asset_cfg": SceneEntityCfg("balloon")},
                )
            _rdev_w = float(os.environ.get("BOX_READY_DEV_W", "0"))
            if _rdev_w > 0:
                self.rewards.cradle_ready_dev = RewTerm(
                    func=box_mdp.cradle_ready_dev_penalty,
                    weight=-_rdev_w,
                    params={"asset_cfg": SceneEntityCfg("balloon")},
                )
            _spin_w = float(os.environ.get("BOX_HOLD_SPIN_W", "0"))
            if _spin_w > 0:
                self.rewards.box_hold_spin = RewTerm(
                    func=box_mdp.box_hold_spin_penalty,
                    weight=-_spin_w,
                    params={"asset_cfg": SceneEntityCfg("balloon")},
                )
            if os.environ.get("BOX_UPRIGHT_GATE", "0") == "1" or \
               os.environ.get("BOX_PALM_ALLPHASE", "0") == "1":
                print("[env] upright-gated sustained + all-phase palm ON; "
                      f"hold_spin_w={_spin_w}", flush=True)
            # Idle stance -> "delivery-worker" ready pose (far-gated CRADLE_POSE
            # pull; the reach window stays unconstrained).  At BOX_READY_W=0.4
            # the clip-tracking pull wins about half the idle frames; 1.2 makes
            # the ready pose dominant.
            self.rewards.cradle_ready = RewTerm(
                func=box_mdp.cradle_ready_reward,
                weight=float(os.environ.get("BOX_READY_W", "0.4")),
                params={"asset_cfg": SceneEntityCfg("balloon")},
            )
            print(f"[env] palm_up 1.5 ON (min={os.environ.get('BOX_PALM_MIN','0')}); "
                  f"compliant_intake 1.0; cradle_ready w="
                  f"{os.environ.get('BOX_READY_W','0.4')}; chest_w="
                  f"{os.environ.get('BOX_CHEST_W','-')}; tuck_w="
                  f"{os.environ.get('BOX_TUCK_W','-')}", flush=True)

        # WRIST HARD LOCK (receive, don't clamp; palms always up): wrists are
        # removed from policy control (action scale 0) and PD-pinned at the
        # calibrated palm-up pose (L roll -1.57 / R +1.57).  Deterministic — no
        # reward conflicts.  Action dim unchanged (warm-start compatible).
        # BOX_WRISTLOCK=0 disables the lock.
        # An abrupt full lock breaks a policy warm-started without it (obs
        # shift of ±1.57 on the wrist dims + 6 frozen action dims), so
        # BOX_WRISTLOCK_SCALE / _ANGLE allow a two-step lock curriculum:
        # step A scale 0.3 / angle 0.8, then step B scale 0.0 / angle 1.57
        # (full lock) warm-started from step A.
        if os.environ.get("BOX_WRISTLOCK", "1") == "1":
            _wl_scale = float(os.environ.get("BOX_WRISTLOCK_SCALE", "0.0"))
            _wl_angle = float(os.environ.get("BOX_WRISTLOCK_ANGLE", "1.57"))
            # Preserve the calibrated per-joint G1_ACTION_SCALE; scale ONLY
            # the wrist entries — a blanket 1.0 would amplify all other joints
            # several-fold (immediate flailing and falls).
            from whole_body_tracking.robots.g1 import G1_ACTION_SCALE
            _scale = dict(G1_ACTION_SCALE)
            for _k in list(_scale.keys()):
                if "_wrist_" in _k:
                    _scale[_k] = _scale[_k] * _wl_scale if isinstance(_scale[_k], float) else _wl_scale
            self.actions.joint_pos.scale = _scale
            self.scene.robot.init_state.joint_pos["left_wrist_roll_joint"] = -_wl_angle
            self.scene.robot.init_state.joint_pos["right_wrist_roll_joint"] = _wl_angle
            # Stiffen wrist PD so the lock HOLDS under box impacts (stock
            # 14-17 Nm/rad + 5 Nm pitch/yaw effort caps let hits knock palms
            # ~67° off).  Note: real G1 wrists are weaker than this sim lock.
            _arms = self.scene.robot.actuators["arms"]
            for _j, _k in ((".*_wrist_roll_joint", 60.0),
                           (".*_wrist_pitch_joint", 40.0),
                           (".*_wrist_yaw_joint", 40.0)):
                _arms.stiffness[_j] = _k
                _arms.damping[_j] = 0.12 * _k
            _arms.effort_limit_sim[".*_wrist_pitch_joint"] = 15.0
            _arms.effort_limit_sim[".*_wrist_yaw_joint"] = 15.0
            # WRIST LIP (BOX_WRIST_LIP): with palms locked FLAT, an upright
            # hold has no forward retention and the policy learns to recline
            # as a gravity backstop.  Pin wrist_pitch fingers-up (negative
            # pitch on both sides, e.g. -0.45) so the bed has a physical lip
            # and an upright tuck can retain the box.  Lever torque of a
            # 0.18 kg box on the lip ≈ 0.2 Nm ≪ the 15 Nm effort cap.
            _lip = float(os.environ.get("BOX_WRIST_LIP", "0"))
            if _lip > 0:
                self.scene.robot.init_state.joint_pos["left_wrist_pitch_joint"] = -_lip
                self.scene.robot.init_state.joint_pos["right_wrist_pitch_joint"] = -_lip
                print(f"[env] wrist lip ON: pitch pinned -{_lip} both "
                      "(fingers-up ledge)", flush=True)

        # BOX_ARM_SCALE: make "arms don't move much while holding" an ACTUATOR
        # constraint instead of a reward signal (reward gradients alone did
        # not move the policy out of a clamp-hug grasp).  Sets the ABSOLUTE
        # action scale of the shoulder+elbow joints (stock G1_ACTION_SCALE
        # arms ~0.44).  Wrists stay under WRISTLOCK above.  Per-joint edit
        # exactly like the wristlock block — NEVER a blanket multiply.
        _arm_sc = float(os.environ.get("BOX_ARM_SCALE", "0"))
        if _arm_sc > 0:
            from whole_body_tracking.robots.g1 import G1_ACTION_SCALE as _G1AS
            _cur = self.actions.joint_pos.scale
            _sc = dict(_cur) if isinstance(_cur, dict) else dict(_G1AS)
            _hit = sorted(k for k in _sc if "_shoulder_" in k or "_elbow_" in k)
            for _k in _hit:
                _sc[_k] = _arm_sc
            self.actions.joint_pos.scale = _sc
            print(f"[env] ARM_SCALE ON: {len(_hit)} shoulder/elbow "
                  f"joints -> {_arm_sc} (stock "
                  f"{ {k: round(float(_G1AS[k]), 3) for k in _hit} })", flush=True)

        # BOX_ELBOW_SCALE: elbow-only authority (shoulders stay locked).  The
        # G1 elbow is a single flexion DoF per arm; giving ONLY the elbows
        # action scale lets the pinned bed curl toward the chest and nothing
        # else.  Must run AFTER the BOX_ARM_SCALE block (overrides its elbow
        # entries).
        _elb_sc = float(os.environ.get("BOX_ELBOW_SCALE", "0"))
        if _elb_sc > 0:
            from whole_body_tracking.robots.g1 import G1_ACTION_SCALE as _G1AS2
            _cur3 = self.actions.joint_pos.scale
            _sc3 = dict(_cur3) if isinstance(_cur3, dict) else dict(_G1AS2)
            _ehit = [k for k in _sc3 if "_elbow_" in k]
            for _k in _ehit:
                _sc3[_k] = _elb_sc
            self.actions.joint_pos.scale = _sc3
            print(f"[env] ELBOW_SCALE ON: {_ehit} -> {_elb_sc} "
                  "(scoop authority; shoulders stay pinned)", flush=True)

        # Optional three-point-carry reward terms (functions + rationale in throw.py)
        _cd_w = float(os.environ.get("BOX_CHEST_DELIVERY_W", "0"))
        if _cd_w > 0:
            self.rewards.chest_delivery = RewTerm(
                func=box_mdp.chest_delivery_reward,
                weight=_cd_w,
                params={"asset_cfg": SceneEntityCfg("balloon")},
            )
        _cc_w = float(os.environ.get("BOX_CHEST_CONTACT_W", "0"))
        if _cc_w > 0:
            self.rewards.chest_contact = RewTerm(
                func=box_mdp.chest_contact_reward,
                weight=_cc_w,
                params={"asset_cfg": SceneEntityCfg("balloon")},
            )
        if os.environ.get("BOX_ROBOT_MASS_DR", "0") == "1":
            self.events.randomize_robot_mass = EventTerm(
                func=isaac_mdp.randomize_rigid_body_mass, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                        "mass_distribution_params": (0.9, 1.1),
                        "operation": "scale", "distribution": "uniform",
                        "recompute_inertia": True})
            print("[env] ROBOT_MASS_DR ON: link mass xU[0.9,1.1]", flush=True)

        _fg_w = float(os.environ.get("BOX_FEET_GATHER_W", "0"))
        if _fg_w > 0:
            self.rewards.hold_feet_gather = RewTerm(
                func=box_mdp.hold_feet_gather_reward,
                weight=_fg_w,
                params={"asset_cfg": SceneEntityCfg("balloon")},
            )

        # BOX_ARM_INIT_CRADLE "soft bed": the arms ARE the catching bed, not a
        # rigid fixture — bed SHAPE pinned at CRADLE_POSE (default_joint_pos =
        # cradle, so with BOX_ARM_SCALE≈0.02 the position targets sit at the
        # cradle and the policy keeps only a ±0.02 rad wiggle), while arm PD
        # stays STOCK (deliberately NOT stiffened, unlike the wrist lock) so
        # the springs yield on box impact and receive it.  Locomotion + waist
        # do the catch.  BOX_BED_SOFT (<1.0) optionally softens the bed further.
        # Reads box_mdp.CRADLE_POSE AFTER its BOX_POSE_SYM / BOX_ELBOW_SCOOP
        # corrections (see throw.py).
        if os.environ.get("BOX_ARM_INIT_CRADLE", "0") == "1":
            import re as _re
            # G1 cfg uses REGEX keys ('.*_elbow_joint'); adding exact keys on
            # top raises 'Multiple matches' at articulation init. Drop the
            # shoulder/elbow regex entries our exact keys supersede.
            _jp = self.scene.robot.init_state.joint_pos
            for _k in list(_jp.keys()):
                if ("_shoulder_" in _k or "_elbow_" in _k) and any(
                        _re.fullmatch(_k, _cj) for _cj in box_mdp.CRADLE_POSE):
                    del _jp[_k]
            _jp.update(box_mdp.CRADLE_POSE)
            _soft = float(os.environ.get("BOX_BED_SOFT", "1.0"))
            if _soft != 1.0:
                _arms_b = self.scene.robot.actuators["arms"]
                if isinstance(_arms_b.stiffness, dict):
                    for _j in list(_arms_b.stiffness.keys()):
                        if "_wrist_" not in _j:
                            _arms_b.stiffness[_j] = _arms_b.stiffness[_j] * _soft
            print(f"[env] SOFT BED ON: arm default pose = "
                  f"corrected CRADLE_POSE, bed_soft={_soft} (stock PD, "
                  "compliant intake). Pair with BOX_ARM_SCALE~0.02.", flush=True)

        # Posture pack: steady palm-up wrists; settle = slight lean-BACK (an
        # asymmetric torso term replaces the direction-agnostic inherited one —
        # set M5_TORSO_W=0); no leg splits.
        # Weights are scale-checked against this env's living reward (~0.17/step
        # mean): each term must stay well under that at typical violation, or
        # terminating early beats living and the policy learns to fall.
        # BOX_POSTURE=0 disables the pack.
        if os.environ.get("BOX_POSTURE", "1") == "1":
            # At a typical posture these cost ~0.05-0.06/step vs the
            # ~0.17/step living reward — pressure without an incentive to
            # terminate early.
            self.rewards.wrist_steady = RewTerm(
                func=box_mdp.wrist_steady_penalty, weight=-0.02, params={})
            self.rewards.leg_split = RewTerm(
                func=box_mdp.leg_split_penalty, weight=-0.2, params={"max_split": 0.8})
            self.rewards.torso_asym = RewTerm(
                func=box_mdp.torso_asym_penalty, weight=-0.2, params={})
            if os.environ.get("BOX_V19C", "0") == "1":
                # Lean limit, applied HERE (after torso_asym is created):
                # back allowance 28° -> 18°, weight -0.2 -> -0.5.  A deep
                # lean costs ~-0.2/step, fully avoidable by standing upright.
                self.rewards.torso_asym.params = {"back_max": 0.31}
                self.rewards.torso_asym.weight = -0.5
                print("[env] torso_asym override applied: back 18deg "
                      "w-0.5", flush=True)

        # ---- BOX_DR=1: physical domain randomization of the box (real boxes
        # vary).  Mass at STARTUP (once per env) + contact material per reset.
        if os.environ.get("BOX_DR", "0") == "1":
            self.events.randomize_box_mass = EventTerm(
                func=isaac_mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("balloon"),
                    "mass_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                    "distribution": "uniform",
                    "recompute_inertia": True,
                },
            )
            # BOX_MU_STATIC / BOX_MU_DYNAMIC: this reset-time randomisation
            # OVERRIDES the spawn material, so the friction knob must live here
            # too.  Set => the range is pinned to that value.  Unset => the
            # default U(0.3,0.7) / U(0.25,0.6) bands.
            _mu_s = os.environ.get("BOX_MU_STATIC"); _mu_d = os.environ.get("BOX_MU_DYNAMIC")
            _sf = (float(_mu_s), float(_mu_s)) if _mu_s else (0.3, 0.7)
            _df = (float(_mu_d), float(_mu_d)) if _mu_d else (0.25, 0.6)
            if _mu_s or _mu_d:
                print(f"[env] box material pinned: static {_sf} dynamic {_df} "
                      f"(was U(0.3,0.7)/U(0.25,0.6); robot material unchanged)", flush=True)
            self.events.randomize_box_material = EventTerm(
                func=isaac_mdp.randomize_rigid_body_material,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("balloon"),
                    "static_friction_range": _sf,
                    "dynamic_friction_range": _df,
                    "restitution_range": (0.05, 0.2),
                    "num_buckets": 64,
                },
            )

        # ---- HOLD-INIT curriculum (top-level block, own print): a fraction
        # of episodes start already holding the box in CRADLE_POSE on the bed
        # (see throw.py hold_init_* for the rationale).  Registration order
        # matters: hold_init_apply must come AFTER box_hover_launch (same-step
        # write order).
        _hif = float(os.environ.get("BOX_HOLD_INIT_FRAC", "0"))
        if _hif > 0:
            self.events.hold_init_mark = EventTerm(
                func=box_mdp.hold_init_mark,
                mode="reset",
                params={"frac": _hif},
            )
            self.events.hold_init_apply = EventTerm(
                func=box_mdp.hold_init_apply,
                mode="interval",
                interval_range_s=(0.0, 0.0),
                is_global_time=False,
                params={
                    "x_off": float(os.environ.get("BOX_HOLD_INIT_X", "0.32")),
                    "z_off": float(os.environ.get("BOX_HOLD_INIT_Z", "0.35")),
                },
            )
            print(f"[env] HOLD_INIT events registered: frac={_hif} "
                  f"x={os.environ.get('BOX_HOLD_INIT_X', '0.32')} "
                  f"z={os.environ.get('BOX_HOLD_INIT_Z', '0.35')}", flush=True)

        # ---- ARM STIFFNESS (BOX_ARM_STIFF; top-level block, own print).
        # kp/kd are commanded at deploy time (low-level command), not taken
        # from the URDF; the sim-to-real invariant is the TORQUE CAP (25 Nm,
        # untouched).  The stock kp 14.25 sags 12-24 deg under box load
        # (tau/kp), physically twisting the palms; kp 40 is the common G1 arm
        # gain.  The gains are part of the plant spec shared by sim and robot.
        _arm_kp = float(os.environ.get("BOX_ARM_STIFF", "0"))
        if _arm_kp > 0:
            _arms2 = self.scene.robot.actuators["arms"]
            for _j in (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                       ".*_shoulder_yaw_joint", ".*_elbow_joint"):
                _arms2.stiffness[_j] = _arm_kp
                _arms2.damping[_j] = 0.0637 * _arm_kp   # stock ratio kept
            print(f"[env] ARM_STIFF ON: shoulder/elbow kp -> {_arm_kp} "
                  f"(kd ratio 0.0637, effort caps UNTOUCHED)", flush=True)

        # ---- ARM SNAP (pin arms to the cradle pose after reset; top-level block)
        if os.environ.get("BOX_ARM_SNAP", "0") == "1":
            self.events.arm_snap = EventTerm(
                func=box_mdp.arm_snap_apply,
                mode="interval",
                interval_range_s=(0.0, 0.0),
                is_global_time=False,
                params={},
            )
            print("[env] ARM_SNAP event registered", flush=True)

        # ---- THROW REALIGN (top-level block): first throw re-sampled on the
        # first post-reset step, after the MotionCommand teleport — MUST be
        # registered after hold_init_apply (skips launched envs).
        if os.environ.get("BOX_THROW_REALIGN", "0") == "1":
            self.events.first_throw_realign = EventTerm(
                func=box_mdp.first_throw_realign,
                mode="interval",
                interval_range_s=(0.0, 0.0),
                is_global_time=False,
                params={},
            )
            print("[env] THROW_REALIGN event registered", flush=True)

        # ---- THREE-POINT hold success definition (top-level block)
        if os.environ.get("BOX_3POINT", "0") == "1":
            self.rewards.threepoint_bonus = RewTerm(
                func=box_mdp.threepoint_hold_bonus,
                weight=float(os.environ.get("BOX_3POINT_BONUS", "10.0")),
                params={"asset_cfg": SceneEntityCfg("balloon"),
                        "hold_sec": float(os.environ.get("BOX_3POINT_SEC", "3.0"))},
            )
            _fsw = float(os.environ.get("BOX_FOOT_STEP_W", "0.2"))
            if _fsw > 0:
                self.rewards.hold_foot_step = RewTerm(
                    func=box_mdp.hold_foot_step_penalty,
                    weight=-_fsw,
                    params={"asset_cfg": SceneEntityCfg("balloon")},
                )
            print(f"[env] 3POINT terms: bonus w="
                  f"{os.environ.get('BOX_3POINT_BONUS', '10.0')} @ "
                  f"{os.environ.get('BOX_3POINT_SEC', '3.0')}s; foot_step w=-{_fsw}",
                  flush=True)

        # ---- GATE_ALL (top-level block, own print): route ALL hold-reward
        # terms through the upright gate.  Captures whatever func the
        # inheritance chain installed into box_mdp._UGATE_INNER, then swaps in
        # the named wrapper (see the GATE_ALL section of throw.py for the
        # rationale).
        if os.environ.get("BOX_GATE_ALL", "0") == "1":
            _gated = []
            for _name in ["box_on_bed", "chest_pocket", "hold_still",
                          "hands_under_box", "low_hold"]:
                _t = getattr(self.rewards, _name, None)
                if _t is not None and _t.weight != 0.0:
                    box_mdp._UGATE_INNER[_name] = _t.func
                    _t.func = getattr(box_mdp, f"ugated_{_name}")
                    _gated.append(_name)
            print(f"[env] GATE_ALL: upright-gated {_gated} "
                  f"(start={os.environ.get('BOX_GATE_START','-0.9')} "
                  f"end={os.environ.get('BOX_GATE_END','-0.31')} "
                  f"floor={os.environ.get('BOX_GATE_ALL_FLOOR','0.1')})",
                  flush=True)

        # ---- AMP SLIM (LAST in __post_init__ by design, so it sees every
        # term created above): when the AMP discriminator owns "looks human",
        # the hand-crafted style terms are zeroed.  The task core stays: catch
        # bonuses, bed/upright-gated sustained, box_on_bed, chest/low_hold,
        # face_box, hands_under_box, hold_still (anti-drop), leg_split
        # (safety), and the MECHANICAL layer (wrist lock + lip + gimbal reflex).
        if os.environ.get("BOX_AMP_SLIM", "0") == "1":
            # BOX_SLIM_KEEP (comma list) exempts terms from the slim.  AMP style
            # reward is additive-only (-log(1-p) >= 0), so it cannot PUNISH a
            # degenerate absorbing state: with upright_hold/torso_asym removed,
            # the uprightness-blind hold terms pay a gravity-backstopped LYING
            # hold best (catch, then lie down).  Keep the anti-degenerate
            # structure terms in that case.
            _keep = set(filter(None,
                               os.environ.get("BOX_SLIM_KEEP", "").split(",")))
            _zeroed = []
            for _name in ["cradle_pose", "palm_up", "tuck_in",
                          "compliant_intake", "torso_asym", "elbow_spread",
                          "forearm_asym", "cradle_ready", "cradle_ready_dev",
                          "arm_idle_still", "upright_hold", "wrist_steady",
                          "lean_during_intake", "forearm_cradle"]:
                if _name in _keep:
                    continue
                _t = getattr(self.rewards, _name, None)
                if _t is not None:
                    _t.weight = 0.0
                    _zeroed.append(_name)
            print(f"[env] AMP SLIM: zeroed {_zeroed} "
                  f"kept {sorted(_keep)}", flush=True)
