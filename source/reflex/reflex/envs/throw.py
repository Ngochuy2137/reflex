"""Box asset, ballistic thrower events and catch reward terms used by all Reflex task configs.

The scene entity keeps the name "balloon" so the inherited reward/obs stack binds unchanged;
the asset is a cardboard-box RigidObject.  Default throws: distance U(3, 4.2) m, bearing within
±45° of the robot's live facing, aim error N(0, 6°) clipped to ±15°, ±8% speed jitter, arc time
rejection-sampled for a 5-7.6 m/s launch under linear drag, release height U(1.1, 1.6) m, aim
height U(0.6, 1.4) m, spin U(0, 1.5) rev/s, and a 0.5-0.75 s hover before launch.
"""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ---------------------------------------------------------------- box asset
# Box models are assigned per env at scene build (collision geometry and mass
# exact per model, not edited at runtime).
# Empty boxes are shells: mass ∝ surface area ∝ size², and drag ∝ area too, so
# drag/mass stays constant → ONE BOX_LIN_DAMPING fits every model.
# eff_radius = mean half-extent, fed to the radius-aware reward thresholds.
BOX_MODELS = (
    dict(name="small", size=(0.34, 0.26, 0.22), mass=0.18, eff_radius=0.14),
    dict(name="mini", size=(0.25, 0.19, 0.13), mass=0.10, eff_radius=0.10),
    # "large": FedEx-style box (54x42x33 cm); mass estimated for an empty box.
    # Its 54 cm long edge is about twice the robot's shoulder width, so it may
    # need a different catch style (pressing to the chest rather than cradling).
    # "largeold" (42x32x27 cm, ~1.25x `small`) is the earlier large size; its
    # mass stays in the same payload band (the box mass DR decouples mass from
    # size anyway).
    dict(name="large", size=(0.54, 0.42, 0.33), mass=0.35, eff_radius=0.22),
    dict(name="largeold", size=(0.42, 0.32, 0.27), mass=0.26, eff_radius=0.175),
    # Measured real-box sizes for the box-size sweep; mass held at the small nominal 0.18 kg so only size varies.
    dict(name="real_small", size=(0.22, 0.21, 0.15), mass=0.18, eff_radius=0.095),
    dict(name="real_regular", size=(0.31, 0.26, 0.21), mass=0.18, eff_radius=0.128),
    dict(name="real_large", size=(0.46, 0.41, 0.32), mass=0.18, eff_radius=0.196),
)
# BOX_MODEL=mix3 / mix:<a>,<b>,...: several box sizes in ONE run, so that size is a real hidden variable the
# Stage 2 latent z must carry.  mix3 = mini + small + largeold (the FedEx-size `large` needs a different catch
# style and stays out).  Single-model values (small / mini / random) keep the original code path.
MIX_SETS = {"mix3": ("mini", "small", "largeold")}


def mix_models(model: str):
    """Return the tuple of BOX_MODELS dicts for a mix spec, or None for a legacy spec."""
    if model in MIX_SETS:
        names = MIX_SETS[model]
    elif model.startswith("mix:"):
        names = tuple(x for x in model[4:].split(",") if x)
    else:
        return None
    out = tuple(next(m for m in BOX_MODELS if m["name"] == nm) for nm in names)
    assert len(out) >= 2, model
    return out


def infer_models_from_geometry(box, models, device):
    """Per-env (eff_radius, half_extents, model index) from the spawned prim's BOUNDING BOX (longest edge).
    Mass-based identification is unreliable under the box mass DR (x0.8-1.2 at startup, BEFORE the first reset),
    which can move a model's mass closer to another model's nominal mass."""
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    paths = list(box.root_physx_view.prim_paths)
    longest = []
    for p_ in paths:
        r_ = cache.ComputeUntransformedBound(stage.GetPrimAtPath(p_)).ComputeAlignedRange().GetSize()
        longest.append(max(float(r_[0]), float(r_[1]), float(r_[2])))
    lg = torch.tensor(longest, device=device)
    ml = torch.tensor([max(m["size"]) for m in models], device=device)
    err = (lg[:, None] - ml[None, :]).abs()
    idx = err.argmin(dim=1)
    worst = float(err.min(dim=1).values.max())
    assert worst < 0.01, f"box geometry does not match any mix model (worst longest-edge error {worst:.3f} m)"
    rad = torch.tensor([m["eff_radius"] for m in models], device=device)[idx]
    he = torch.tensor([m["size"] for m in models], device=device)[idx] * 0.5
    cnt = torch.bincount(idx, minlength=len(models)).tolist()
    print("[box_mix] per-env box models from prim GEOMETRY (worst edge error %.4f m): " % worst
          + ", ".join(f"{m['name']}={c}" for m, c in zip(models, cnt)), flush=True)
    return rad, he, idx


def infer_models_nearest_mass(masses: torch.Tensor, models, device):
    """Per-env (eff_radius (N,), half_extents (N,3), model index (N,)) by nearest BUILD mass.  Only valid on the
    original spawn masses (call before any mass DR overwrites them)."""
    mm = torch.tensor([m["mass"] for m in models], device=device)
    idx = (masses.reshape(-1, 1).to(device) - mm[None, :]).abs().argmin(dim=1)
    rad = torch.tensor([m["eff_radius"] for m in models], device=device)[idx]
    he = torch.tensor([m["size"] for m in models], device=device)[idx] * 0.5
    cnt = torch.bincount(idx, minlength=len(models)).tolist()
    print("[box_mix] per-env box models by nearest build mass: "
          + ", ".join(f"{m['name']}={c}" for m, c in zip(models, cnt)), flush=True)
    return rad, he, idx


# Aliases for BOX_MODELS[0] (the regular "small" box)
BOX_SIZE = BOX_MODELS[0]["size"]
BOX_MASS = BOX_MODELS[0]["mass"]
BOX_EFFECTIVE_RADIUS = BOX_MODELS[0]["eff_radius"]

GRAVITY = 9.81
# Air-drag proxy for an EMPTY box: real empty boxes visibly decelerate (their
# terminal velocity is only ~6 m/s, 0.18 kg over a ~0.075 m² mean face, i.e.
# right at typical arrival speeds, so the late fall is heavily damped).
# Modelled as PhysX linear damping k: F = -k·m·v.  The ballistic solve below
# uses the matching linear-drag closed form, so aim stays exact under drag.
BOX_LIN_DAMPING = 0.85  # linear damping k [1/s]
# Launch speed band [m/s] (BOX_SPEED_LO / BOX_SPEED_HI); lofty arcs get a
# higher cap (LOFTY_SPEED_BONUS).
LAUNCH_SPEED_RANGE = (float(os.environ.get("BOX_SPEED_LO", "5.0")), float(os.environ.get("BOX_SPEED_HI", "7.6")))   # far throws (>4.2 m) need a higher cap
# Throw distance [m] (BOX_DIST_LO / BOX_DIST_HI).  With k=0.85 drag, throws
# beyond ~4.2 m exceed even the lofty speed cap and systematically undershoot;
# widen moderately and never past 5.0 without revisiting the speed caps.
DIST_RANGE = (float(os.environ.get("BOX_DIST_LO", "3.0")),
              float(os.environ.get("BOX_DIST_HI", "4.2")))
# Frontal fan: throwers stand within ±BEARING_RANGE_DEG of the robot's live
# facing (BOX_BEARING_DEG); aim error std / clip via BOX_AIM_STD_DEG /
# BOX_AIM_MAX_DEG.
BEARING_RANGE_DEG = float(os.environ.get("BOX_BEARING_DEG", "45.0"))
T_MULT_RANGE = (0.75, 1.35)
# Arc-time multipliers, tried in order lofty -> mid -> flat: the rejection
# loop prefers HIGH arcs and only falls back to flatter ones when the speed
# cap forbids them (far throws).  The sampler pins the flight time T (drag
# only redistributes speed within the same T), so airtime is set here.
T_MULT_TRY_RANGES = ((1.15, 1.45), (0.95, 1.20), (0.80, 1.00))
# Throws look fast because of FLAT liners, not lobs, so the speed cap relaxes
# with arc height.
LOFTY_SPEED_BONUS = 2.0   # cap += bonus*max(t_mult-1,0); loftiest ~8.5 m/s
AIM_ERR_STD_DEG = float(os.environ.get("BOX_AIM_STD_DEG", "6.0"))  # defaults: std 6°, clip ±15°
AIM_ERR_MAX_DEG = float(os.environ.get("BOX_AIM_MAX_DEG", "15.0"))
SPEED_JITTER = float(os.environ.get("BOX_SPEED_JITTER", "0.08"))
# ---- real-throw library replay (BOX_THROW_LIB=<json>) --------------------
# Synthetic throws aim at the PELVIS xy at catch_h 0.6-1.4 m and leave at
# >= 5.0 m/s, so most pass the hand plane above 1.3 m; fitted real human
# throws (k=0.85 drag fit) aim at the HANDS, leave at ~4.7 m/s median and
# pass the hand plane at ~0.65 m.  The two distributions barely overlap.
# BOX_THROW_LIB replaces a fraction of throws with library entries (p0, v0 in
# the robot's level frame: x = robot forward toward the thrower, y = robot
# left, z = floor) injected verbatim at launch.  Unset = off.
THROW_LIB = os.environ.get("BOX_THROW_LIB", "")
THROW_LIB_KEY = os.environ.get("BOX_THROW_LIB_KEY", "all")        # all | sample60
THROW_LIB_FRAC = float(os.environ.get("BOX_THROW_LIB_FRAC", "1.0"))  # share of throws from the library
THROW_LIB_MIRROR = os.environ.get("BOX_THROW_LIB_MIRROR", "1") == "1"  # random y-mirror (augment)
THROW_LIB_JITTER = float(os.environ.get("BOX_THROW_LIB_JITTER", "0.0"))  # +-frac speed jitter
# Reachability probe (BOX_THROW_LIB_ALIGN; default "" = off).  Translates
# each library launch point in the robot's level frame so the fitted z=1.0
# descending crossing (arr_dx, arr_dy) lands on the catch sweet spot (ALIGN_X
# ahead of the pelvis, where the Stage 1 teacher's successful real catches
# cross, ~0.31-0.35 m; lateral 0).  v0 is untouched, so the height profile,
# descent angle and flight time are the real throw's -- only WHERE it arrives
# changes.  "x" aligns forward only (keeps the library lateral offset); "xy"
# aligns both.  Separates "the legs cannot get there" from "cannot catch a
# low/steep box delivered to the hands".  Assumes JITTER=0 (jitter would move
# the crossing).
THROW_LIB_ALIGN = os.environ.get("BOX_THROW_LIB_ALIGN", "")            # "" | x | xy | reach
THROW_LIB_ALIGN_X = float(os.environ.get("BOX_THROW_LIB_ALIGN_X", "0.35"))
# "reach" = training curriculum: per throw, the crossing is placed
# ALIGN_X + U(REACH_LO, REACH_HI) m ahead of the live pelvis (lateral =
# library), so the policy is asked to step forward a graded 0..R m instead of
# being handed the raw real distribution (~30 % unreachable from a standstill).
# Cannot be gamed by pre-positioning: the target tracks the LIVE robot every
# hover step and at launch.
THROW_LIB_REACH_LO = float(os.environ.get("BOX_THROW_LIB_REACH_LO", "0.0"))
THROW_LIB_REACH_HI = float(os.environ.get("BOX_THROW_LIB_REACH_HI", "0.8"))
# BACKTRACK: library p0 is the FIRST MEASUREMENT (real perception horizon ~2.5 m; many entries are first seen
# at 1.3-2.5 m), so a verbatim replay parks the hover box 1.3-2.5 m from the robot and launches it with only
# 0.2-0.7 s of flight -- out of distribution (synthetic throwers stand 2.5-5 m away).
# BACKTRACK>0 extrapolates each entry BACKWARDS along its own drag ballistic to horizontal distance
# BACKTRACK metres from the anchor (if it is closer than that), so the sim sees the unobserved early
# flight; the fitted part of the trajectory and the arrival point are unchanged.  0 = off = verbatim.
THROW_LIB_BACKTRACK = float(os.environ.get("BOX_THROW_LIB_BACKTRACK", "0"))
_THROW_LIB_CACHE: dict = {}


def _throw_lib(device):
    """Lazy-load the library once per device -> dict of tensors."""
    key = str(device)
    if key not in _THROW_LIB_CACHE:
        import json
        with open(THROW_LIB) as f:
            entries = json.load(f)[THROW_LIB_KEY]
        p0 = torch.tensor([e["p0"] for e in entries], dtype=torch.float32, device=device)
        v0 = torch.tensor([e["v0"] for e in entries], dtype=torch.float32, device=device)
        arr = torch.tensor([[e["arr_dx"], e["arr_dy"], e["arr_T"]] for e in entries],
                           dtype=torch.float32, device=device)
        _THROW_LIB_CACHE[key] = dict(p0=p0, v0=v0, arr=arr)
        print(f"[throw] throw library armed: {THROW_LIB} key={THROW_LIB_KEY} n={len(entries)} "
              f"frac={THROW_LIB_FRAC} mirror={int(THROW_LIB_MIRROR)} jitter={THROW_LIB_JITTER} "
              f"(launch injects library v0 verbatim; walking thrower frozen for lib throws)",
              flush=True)
    return _THROW_LIB_CACHE[key]
RELEASE_H_RANGE = (float(os.environ.get("BOX_RELEASE_H_LO", "1.1")),
                   float(os.environ.get("BOX_RELEASE_H_HI", "1.6")))
# Aim height [m] (BOX_CATCH_H_LO / BOX_CATCH_H_HI): the synthetic thrower aims at the PELVIS xy at
# this height; real human throws aim at the hands.
CATCH_H_RANGE = (float(os.environ.get("BOX_CATCH_H_LO", "0.6")),
                 float(os.environ.get("BOX_CATCH_H_HI", "1.4")))
# BOX_SPIN_MAX_REV = max release spin (rev/s, U(0,max)); BOX_RELEASE_LEVEL=1 = level release
# orientation (yaw only) instead of a uniformly random quaternion (an idealised human throw).
SPIN_MAX_RAD_S = float(os.environ.get("BOX_SPIN_MAX_REV", "1.5")) * 2.0 * math.pi
RELEASE_LEVEL = os.environ.get("BOX_RELEASE_LEVEL", "0") == "1"
HOVER_RANGE_S = (float(os.environ.get("BOX_HOVER_LO", "0.5")),
                 float(os.environ.get("BOX_HOVER_HI", "0.75")))  # hover before launch [s]
# Auto re-throw after the box lands: multi-attempt episodes + bad-physics guard.
RETHROW_GRACE_STEPS = 30  # ~0.6 s resting on the ground before the next throw
# BOX_RETHROW=0 disables the auto re-throw -> exactly one throw per env per episode (evaluation protocol:
# fixed denominator, no outcome-dependent extra trials).  Default 1 = re-throw on.
RETHROW_ENABLED = os.environ.get("BOX_RETHROW", "1") == "1"
# Compliant-intake shaping: humans instinctively yield backward and squat to absorb a caught box.
INTAKE_GATE_DIST = 0.9    # m, box-to-pelvis gate for the intake window
# bed partial-credit mix (horiz, above, over, level) — BOX_V19 shifts weight
# onto above+level (the anti-asymmetric-grip terms); otherwise the default mix.
_BED_MIX = (0.15, 0.45, 0.15, 0.25) if os.environ.get("BOX_V19", "0") == "1" \
    else (0.25, 0.30, 0.25, 0.20)
INTAKE_GATE_SPEED = 1.0   # m/s, box must still be flying (no reward for held box)
YIELD_V_CAP = 0.5         # m/s, velocity normalization cap


def _one_box_spawn(model: dict, color=None) -> "sim_utils.CuboidCfg":
    return sim_utils.CuboidCfg(
        size=model["size"],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=BOX_LIN_DAMPING,
            angular_damping=0.05,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=model["mass"]),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        # BOX_MU_STATIC / BOX_MU_DYNAMIC: box surface friction (default 0.45 / 0.40).
        # With BOX_DR=1 the reset-time material randomisation in catch_env_cfg
        # overrides these.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=float(os.environ.get("BOX_MU_STATIC", "0.45")),
            dynamic_friction=float(os.environ.get("BOX_MU_DYNAMIC", "0.40")),
            restitution=0.10,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(_box_color() if color is None else color), roughness=0.9),
    )


# BOX_COLOR: visual colour of the box, default cardboard brown.  Names below or "r,g,b" in [0,1].  Visual only: mass / size /
# physics untouched; only the rendered RGB-D frames for the Stage 3 visual encoder are affected.
_BOX_COLORS = {"cardboard": (0.72, 0.52, 0.30), "white": (0.90, 0.90, 0.90), "black": (0.05, 0.05, 0.05), "red": (0.80, 0.10, 0.10),
               "blue": (0.10, 0.20, 0.80), "grey": (0.50, 0.50, 0.50), "green": (0.10, 0.60, 0.15), "yellow": (0.90, 0.80, 0.10)}
_COLOR_PROOF = [False]


def _random_palette(n=16, seed=20260924):
    """BOX_COLOR=random (colour augmentation for the visual encoder): a fixed, seeded palette of n RGB colours spread over hue
    and luminance (including dark and grey-ish entries) + cardboard.  Deliberately NOT the named test colours."""
    import numpy as _np
    rng = _np.random.RandomState(seed); out = [_BOX_COLORS["cardboard"]]
    for k in range(n):
        h = rng.uniform(0, 1); sat = rng.uniform(0.0, 1.0); val = rng.uniform(0.08, 0.95)
        i = int(h * 6) % 6; f = h * 6 - int(h * 6); p_, q_, t_ = val * (1 - sat), val * (1 - sat * f), val * (1 - sat * (1 - f))
        r, g, b = [(val, t_, p_), (q_, val, p_), (p_, val, t_), (p_, q_, val), (t_, p_, val), (val, p_, q_)][i]
        out.append((round(float(r), 3), round(float(g), 3), round(float(b), 3)))
    return out


def box_spawn_variants(model: dict):
    """One spawn cfg per colour when BOX_COLOR=random (each env then draws model x colour at scene build), else a single cfg."""
    if os.environ.get("BOX_COLOR", "cardboard") == "random":
        pal = _random_palette()
        if not _COLOR_PROOF[0]:
            _COLOR_PROOF[0] = True
            print(f"[box_color] BOX_COLOR=random -> {len(pal)} colours per model (non-default box colour, per-env random)", flush=True)
        return [_one_box_spawn(model, color=c) for c in pal]
    return [_one_box_spawn(model)]


def _box_color():
    spec = os.environ.get("BOX_COLOR", "cardboard")
    c = _BOX_COLORS.get(spec) or tuple(float(x) for x in spec.split(","))
    assert len(c) == 3, spec
    if not _COLOR_PROOF[0]:
        _COLOR_PROOF[0] = True
        print(f"[box_color] BOX_COLOR={spec} -> diffuse {tuple(round(v, 2) for v in c)}" + ("" if spec == "cardboard" else "   (NON-DEFAULT box colour)"), flush=True)
    return c


def make_box_cfg(prim_path: str = "{ENV_REGEX_NS}/Balloon", model: str = "random") -> RigidObjectCfg:
    """Cardboard box RigidObject.  model="random" assigns one of the original
    four BOX_MODELS (not the "real_*" sizes) per env at scene build;
    "mix3" / "mix:a,b,..." mixes the named models; any other model name
    forces that model."""
    _mix = mix_models(model)
    if _mix is not None:
        from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
        spawn = MultiAssetSpawnerCfg(assets_cfg=[v for m in _mix for v in box_spawn_variants(m)], random_choice=True)
        print(f"[box_mix] BOX_MODEL={model}: MultiAsset over {[m['name'] for m in _mix]}", flush=True)
    elif model == "random":
        from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
        spawn = MultiAssetSpawnerCfg(
            assets_cfg=[v for m in BOX_MODELS if not m["name"].startswith("real_") for v in box_spawn_variants(m)],   # "random" keeps the original 4 models
            random_choice=True,
        )
    else:
        chosen = next(m for m in BOX_MODELS if m["name"] == model)
        _vars = box_spawn_variants(chosen)
        if len(_vars) > 1:   # BOX_COLOR=random on a single model: per-env random colour via MultiAsset
            from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
            spawn = MultiAssetSpawnerCfg(assets_cfg=_vars, random_choice=True)
        else:
            spawn = _vars[0]
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 8.0)),
    )


# ---------------------------------------------------------------- helpers
def _rot2d(vec: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Rotate (n,2) vectors by (n,) angles."""
    c, s = torch.cos(ang), torch.sin(ang)
    return torch.stack([vec[:, 0] * c - vec[:, 1] * s, vec[:, 0] * s + vec[:, 1] * c], dim=-1)


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


def _rand_quat(n: int, device) -> torch.Tensor:
    q = torch.randn(n, 4, device=device)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _init_buffers(env: "ManagerBasedEnv", device) -> None:
    n = env.num_envs
    env._box_p0 = torch.zeros((n, 3), device=device)
    env._box_q0 = torch.zeros((n, 4), device=device)
    env._box_q0[:, 0] = 1.0
    env._box_v0 = torch.zeros((n, 3), device=device)
    env._box_omega = torch.zeros((n, 3), device=device)
    env._box_launch_step = torch.zeros(n, dtype=torch.long, device=device)
    env._box_launched = torch.zeros(n, dtype=torch.bool, device=device)
    # per-throw scalars kept for the launch-time re-aim (see box_hover_and_launch)
    env._box_aim_err = torch.zeros(n, device=device)
    env._box_catch_h = torch.zeros(n, device=device)
    env._box_tmult = torch.zeros(n, device=device)
    env._box_jitter = torch.ones(n, device=device)


# ---------------------------------------------------------------- sampler
def sample_throws(n: int, origins: torch.Tensor, fwd: torch.Tensor, device,
                  step_dt: float = 0.02) -> dict:
    """Pure throw sampler (shared by the training env AND validation renders so
    videos show EXACTLY the training distribution).

    origins: (n,3) robot anchor points; fwd: (n,2) unit frontal direction.
    Returns dict of (n,*) tensors: p0, q0, v0, omega, hover_steps + meta.
    """
    # Thrower position in the frontal sector.
    bearing = _uniform(n, -1.0, 1.0, device) * math.radians(BEARING_RANGE_DEG)
    d = _uniform(n, *DIST_RANGE, device)
    thrower_dir = _rot2d(fwd, bearing)                       # robot -> thrower
    thrower_xy = origins[:, :2] + thrower_dir * d[:, None]
    release_h = _uniform(n, *RELEASE_H_RANGE, device)
    p0 = torch.cat([thrower_xy, (origins[:, 2] + release_h)[:, None]], dim=-1)

    # Aim: nominal target = robot spot at catch height, rotated by aim error.
    catch_h = _uniform(n, *CATCH_H_RANGE, device)
    aim_err = (torch.randn(n, device=device) * math.radians(AIM_ERR_STD_DEG)).clamp(
        -math.radians(AIM_ERR_MAX_DEG), math.radians(AIM_ERR_MAX_DEG))
    to_robot = _rot2d(origins[:, :2] - thrower_xy, aim_err)  # (n,2), length d
    target = torch.cat([thrower_xy + to_robot, (origins[:, 2] + catch_h)[:, None]], dim=-1)

    # Ballistic solve WITH linear drag (closed form matching PhysX damping):
    #   dv/dt = g - k v  =>  p(T) = p0 + v_term T + (v0 - v_term)(1 - e^{-kT})/k
    #   =>  v0 = v_term + k (target - p0 - v_term T) / (1 - e^{-kT})
    # Rejection on launch speed band.
    k = BOX_LIN_DAMPING
    g = torch.tensor([0.0, 0.0, -GRAVITY], device=device)
    v_term = (g / k)[None, :]                                # terminal velocity
    jitter = 1.0 + _uniform(n, -SPEED_JITTER, SPEED_JITTER, device)
    t_mult = _uniform(n, *T_MULT_TRY_RANGES[0], device)
    v0 = torch.zeros((n, 3), device=device)
    T = torch.zeros(n, device=device)
    need = torch.ones(n, dtype=torch.bool, device=device)
    for i in range(9):
        if not need.any():
            break
        T = torch.where(need, torch.sqrt(d / 4.0) * t_mult, T)
        expk = torch.exp(-k * T)[:, None]
        cand = v_term + k * (target - p0 - v_term * T[:, None]) / (1.0 - expk)
        cand = cand * jitter[:, None]
        v0 = torch.where(need[:, None], cand, v0)
        speed = v0.norm(dim=-1)
        cap = LAUNCH_SPEED_RANGE[1] + LOFTY_SPEED_BONUS * (t_mult - 1.0).clamp(min=0.0)
        need = (speed < LAUNCH_SPEED_RANGE[0]) | (speed > cap)
        # lofty first (3 tries), then mid, then flat — see T_MULT_TRY_RANGES
        rng = T_MULT_TRY_RANGES[min(i // 3, 2)]
        t_mult = torch.where(need, _uniform(n, *rng, device), t_mult)
    # Rare stragglers: clamp magnitude, keep direction (slightly short/long).
    speed = v0.norm(dim=-1).clamp_min(1e-6)
    cap = LAUNCH_SPEED_RANGE[1] + LOFTY_SPEED_BONUS * (t_mult - 1.0).clamp(min=0.0)
    clamped = torch.minimum(speed.clamp_min(LAUNCH_SPEED_RANGE[0]), cap)
    v0 = v0 * (clamped / speed)[:, None]

    # Spin + release orientation.
    axis = torch.randn(n, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    omega = axis * _uniform(n, 0.0, SPIN_MAX_RAD_S, device)[:, None]
    q0 = _rand_quat(n, device)
    if RELEASE_LEVEL:  # yaw-only: q = (cos(a/2), 0, 0, sin(a/2))
        _a = _uniform(n, -math.pi, math.pi, device) * 0.5
        q0 = torch.stack([torch.cos(_a), torch.zeros_like(_a), torch.zeros_like(_a), torch.sin(_a)], dim=-1)

    # Hover duration in env steps.
    hover_steps = (_uniform(n, *HOVER_RANGE_S, device) / step_dt).long()

    # ---- library replay override (see THROW_LIB above) ---------------------
    lib_mask = torch.zeros(n, dtype=torch.bool, device=device)
    lib_arrvec_w = torch.zeros(n, 2, device=device)
    lib_arr_dy = torch.zeros(n, device=device)
    lib_tgt_x = torch.full((n,), THROW_LIB_ALIGN_X, device=device)
    if THROW_LIB:
        L = _throw_lib(device)
        lib_mask = _uniform(n, 0.0, 1.0, device) < THROW_LIB_FRAC
        idx = torch.randint(0, L["p0"].shape[0], (n,), device=device)
        lp0, lv0, larr = L["p0"][idx], L["v0"][idx], L["arr"][idx]
        if THROW_LIB_BACKTRACK > 0:
            # solve x(-tau) = R along x(t) = x0 + vx0 (1-e^{-kt})/k  =>  e^{k tau} = 1 + k (R - x0)/(-vx0)
            _k = BOX_LIN_DAMPING
            _vt = torch.tensor([0.0, 0.0, -GRAVITY / _k], device=device)
            _r = lp0[:, :2].norm(dim=-1)
            _vx = -(lp0[:, :2] * lv0[:, :2]).sum(dim=-1) / _r.clamp_min(1e-6)   # closing speed toward anchor (>0)
            _need = (_r < THROW_LIB_BACKTRACK) & (_vx > 0.3)
            _arg = (1.0 + _k * (THROW_LIB_BACKTRACK - _r) / _vx.clamp_min(0.3)).clamp_min(1.0)
            _tau = torch.where(_need, torch.log(_arg) / _k, torch.zeros_like(_r)).clamp(max=1.5)
            _ek = torch.exp(_k * _tau)[:, None]
            # p(-tau) = p0 - vt*tau - (v0 - vt)(e^{k tau} - 1)/k ;  v(-tau) = vt + (v0 - vt) e^{k tau}
            lp0 = lp0 - _vt[None, :] * _tau[:, None] - (lv0 - _vt[None, :]) * (_ek - 1.0) / _k
            lv0 = _vt[None, :] + (lv0 - _vt[None, :]) * _ek
            if not _THROW_LIB_CACHE.get("_bt_printed"):
                _THROW_LIB_CACHE["_bt_printed"] = True
                print(f"[throw] BACKTRACK armed: R={THROW_LIB_BACKTRACK} m -- entries first seen closer than R are "
                      f"extrapolated back along their own drag ballistic (tau<=1.5 s); arrival unchanged", flush=True)
        if THROW_LIB_MIRROR:
            flip = torch.where(_uniform(n, 0.0, 1.0, device) < 0.5,
                               torch.full((n,), -1.0, device=device), torch.ones(n, device=device))
            lp0 = lp0 * torch.stack([torch.ones_like(flip), flip, torch.ones_like(flip)], dim=-1)
            lv0 = lv0 * torch.stack([torch.ones_like(flip), flip, torch.ones_like(flip)], dim=-1)
            larr = larr * torch.stack([torch.ones_like(flip), flip, torch.ones_like(flip)], dim=-1)
        if THROW_LIB_JITTER > 0:
            lv0 = lv0 * (1.0 + _uniform(n, -THROW_LIB_JITTER, THROW_LIB_JITTER, device))[:, None]
        lib_arr_dy = larr[:, 1].clone()          # library lateral (level frame, pre-shift)
        if THROW_LIB_ALIGN:
            # reachability probe: move the launch point so the fitted crossing
            # sits at (ALIGN_X, 0 | arr_dy); computed AFTER the mirror so the
            # shift is consistent with the mirrored arrival.  This only aligns
            # the HOVER position; lib_align_track() re-aligns to the robot's
            # live pose every hover step and at launch (the robot drifts
            # +-0.3 m during the 1-6 s hover).
            tgt_x = torch.full_like(larr[:, 0], THROW_LIB_ALIGN_X)
            if THROW_LIB_ALIGN == "reach":
                tgt_x = tgt_x + _uniform(n, THROW_LIB_REACH_LO, THROW_LIB_REACH_HI, device)
            lib_tgt_x = tgt_x
            tgt_y = torch.zeros_like(larr[:, 1]) if THROW_LIB_ALIGN == "xy" else larr[:, 1]
            shift = torch.stack([tgt_x - larr[:, 0], tgt_y - larr[:, 1]], dim=-1)
            lp0 = torch.cat([lp0[:, :2] + shift, lp0[:, 2:3]], dim=-1)
            larr = torch.cat([larr[:, :2] + shift, larr[:, 2:3]], dim=-1)
            if not _THROW_LIB_CACHE.get("_align_printed"):
                _THROW_LIB_CACHE["_align_printed"] = True
                _reach = (f" reach=U({THROW_LIB_REACH_LO},{THROW_LIB_REACH_HI})m beyond target_x (curriculum)"
                          if THROW_LIB_ALIGN == "reach" else "")
                print(f"[throw] ALIGN armed: mode={THROW_LIB_ALIGN} target_x={THROW_LIB_ALIGN_X}{_reach} "
                      f"(reachability probe: launch point re-anchored to the LIVE robot every hover step "
                      f"and at launch so the z=1.0 crossing lands on the hand plane; v0 verbatim; expect [HAND] hz~1.0"
                      f"{', |hy|~0' if THROW_LIB_ALIGN == 'xy' else ', hy = library lateral'})", flush=True)
        left = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=-1)          # robot-left unit
        def _lvl2w(xy):
            return origins[:, :2] + fwd * xy[:, :1] + left * xy[:, 1:2]
        lp0_w = torch.cat([_lvl2w(lp0[:, :2]), (origins[:, 2] + lp0[:, 2])[:, None]], dim=-1)
        lv0_w = torch.cat([fwd * lv0[:, :1] + left * lv0[:, 1:2], lv0[:, 2:3]], dim=-1)
        ltarget = torch.cat([_lvl2w(larr[:, :2]), (origins[:, 2] + 1.0)[:, None]], dim=-1)
        # world-frame vector launch point -> z=1.0 crossing (translation-invariant
        # because v0 is fixed in world); lib_align_track() uses it
        lib_arrvec_w = torch.where(lib_mask[:, None], ltarget[:, :2] - lp0_w[:, :2],
                                   torch.zeros_like(ltarget[:, :2]))
        m = lib_mask[:, None]
        p0 = torch.where(m, lp0_w, p0)
        v0 = torch.where(m, lv0_w, v0)
        target = torch.where(m, ltarget, target)
        T = torch.where(lib_mask, larr[:, 2], T)
        d = torch.where(lib_mask, lp0[:, :2].norm(dim=-1), d)
        aim_err = torch.where(lib_mask, torch.zeros_like(aim_err), aim_err)
        catch_h = torch.where(lib_mask, torch.ones_like(catch_h), catch_h)
        t_mult = torch.where(lib_mask, torch.ones_like(t_mult), t_mult)
        jitter = torch.where(lib_mask, torch.ones_like(jitter), jitter)

    # Arrival velocity under linear drag: v(T) = v_term + (v0 - v_term) e^{-kT}
    v_arr = v_term + (v0 - v_term) * torch.exp(-k * T)[:, None]
    return dict(
        lib_mask=lib_mask, lib_arrvec_w=lib_arrvec_w, lib_arr_dy=lib_arr_dy, lib_tgt_x=lib_tgt_x,
        p0=p0, q0=q0, v0=v0, omega=omega, hover_steps=hover_steps,
        # per-throw scalars needed by the launch-time re-aim
        aim_err=aim_err, catch_h=catch_h, t_mult=t_mult, jitter=jitter,
        # meta (for manifests/cameras; not used by the env)
        d=d, T=T, target=target, launch_speed=v0.norm(dim=-1),
        arrival_speed=v_arr.norm(dim=-1),
        spin_rev_s=omega.norm(dim=-1) / (2 * math.pi),
    )


def lib_align_track(env, hovering: torch.Tensor, launching: torch.Tensor) -> None:
    """Reachability probe (no-op unless BOX_THROW_LIB_ALIGN is set).
    Re-anchor aligned library throws to the robot's LIVE pose on every hover
    step and at the launch instant, so the z=1.0 crossing lands on the hand
    plane no matter how the robot drifts during the 1-6 s hover (aligning only
    at reset lets a +-0.3 m hover drift smear the arrival back to the
    library's own spread).  Same idea as the synthetic thrower's launch-time
    re-aim, applied to a fixed v0:
    p0_xy = target_xy(live robot) - (world vector launch -> crossing)."""
    if not THROW_LIB_ALIGN or not hasattr(env, "_box_lib_arrvec_w"):
        return
    m = (hovering | launching) & env._box_lib_mask
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    qw, qx, qy, qz = q.unbind(-1)
    fwd = torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz)], dim=-1)
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=-1)
    ty = env._box_lib_arr_dy if THROW_LIB_ALIGN == "x" else torch.zeros_like(env._box_lib_arr_dy)
    tgt = robot.data.root_pos_w[:, :2] + fwd * env._box_lib_tgt_x[:, None] + left * ty[:, None]
    new_xy = tgt - env._box_lib_arrvec_w
    env._box_p0[:, :2] = torch.where(m[:, None], new_xy, env._box_p0[:, :2])


# ---------------------------------------------------------------- events
def reset_box_throw(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    motion_direction: tuple[float, float] = (0.0, -1.0),
) -> None:
    """Reset event: sample a ballistic throw per env, park the box hovering at
    the release point. box_hover_and_launch() releases it later."""
    if env_ids is None or len(env_ids) == 0:
        return
    box: RigidObject = env.scene[asset_cfg.name]
    device = box.device
    n = len(env_ids)

    if not hasattr(env, "_box_p0"):
        _init_buffers(env, device)
    # Radius-aware reward thresholds read this buffer.  The per-env box model
    # is fixed at scene build; infer it once and store the matching effective
    # radius and half-extents: from prim geometry for mixes and single
    # non-default models, otherwise from the PhysX masses (small/mini split at
    # 0.132 kg, robust to ±20% mass DR: small 0.144-0.216, mini 0.08-0.12).
    _mix = mix_models(os.environ.get("BOX_MODEL", "random"))
    if not hasattr(env, "_balloon_radius_per_env") and _mix is not None:
        masses = box.root_physx_view.get_masses().reshape(-1).to(device)
        env._balloon_radius_per_env, env._box_half_extents, env._box_model_idx = infer_models_from_geometry(box, _mix, device)
    _one = next((m for m in BOX_MODELS if m["name"] == os.environ.get("BOX_MODEL", "random")), None)
    if not hasattr(env, "_balloon_radius_per_env") and _one is not None and _one["name"] not in ("small", "mini"):
        env._balloon_radius_per_env, env._box_half_extents, env._box_model_idx = infer_models_from_geometry(box, (_one,), device)
        print(f"[box_model] single model {_one['name']} {_one['size']} eff_r {_one['eff_radius']} (radius from geometry)", flush=True)
    if not hasattr(env, "_balloon_radius_per_env"):
        masses = box.root_physx_view.get_masses().reshape(-1).to(device)  # (N,)
        is_big = (masses > 0.132)
        radii = torch.where(
            is_big,
            torch.full((env.num_envs,), BOX_MODELS[0]["eff_radius"], device=device),
            torch.full((env.num_envs,), BOX_MODELS[1]["eff_radius"], device=device),
        )
        env._balloon_radius_per_env = radii
        # per-env box HALF-extents (for the orientation-aware bottom-face
        # "offside" bed check)
        big = torch.tensor(BOX_MODELS[0]["size"], device=device) * 0.5
        small = torch.tensor(BOX_MODELS[1]["size"], device=device) * 0.5
        env._box_half_extents = torch.where(is_big.unsqueeze(-1), big[None, :], small[None, :])

    origins = env.scene.env_origins[env_ids]  # (n, 3)

    # Anchor the throw on the robot's LIVE position, not the env origin — the
    # reference clip + phase randomization can put the robot up to ~2 m away
    # from the origin.  z stays at origin ground level for height references.
    robot = env.scene["robot"]
    anchor = origins.clone()
    anchor[:, :2] = robot.data.root_pos_w[env_ids, :2]

    # Frontal direction = the ROBOT'S LIVE facing: the box must always come
    # from in front of the robot as it currently faces, not from the
    # clip-frame "front" (which drifts once the robot rotates).
    q = robot.data.root_quat_w[env_ids]
    qw, qx, qy, qz = q.unbind(-1)
    fwd = torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz)], dim=-1)
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    s = sample_throws(n, anchor, fwd, device, step_dt=env.step_dt)

    # ---- INTAKE DRILL (BOX_V38_INTAKE_FRAC; default 0 = off) ---------------
    # Boxes arriving dead-centre are dropped disproportionately often; the
    # failure is the SEATING instant, which a normal throw exercises only once.
    # This mode replaces a fraction of throws with a SHORT DROP: the box appears
    # 0.35 m in front of the chest at 1.55 m and falls onto the bed in ~0.4 s,
    # drilling the seating moment at ~3x density.
    _ifr = float(os.environ.get("BOX_V38_INTAKE_FRAC", "0"))
    if _ifr > 0:
        _m = torch.rand(n, device=device) < _ifr
        if _m.any():
            _fw3 = torch.cat([fwd, torch.zeros(n, 1, device=device)], dim=-1)
            _pp = anchor + 0.35 * _fw3
            _pp[:, 2] = anchor[:, 2] + 1.55
            s["p0"] = torch.where(_m[:, None], _pp, s["p0"])
            _vv = torch.zeros_like(s["v0"]); _vv[:, 2] = -0.5
            s["v0"] = torch.where(_m[:, None], _vv, s["v0"])
        if not hasattr(env, "_intake_printed"):
            env._intake_printed = True
            print(f"[throw] intake drill armed: frac={_ifr} "
                  "(short drop 0.35m ahead @1.55m, seats in ~0.4s)", flush=True)
    env._box_p0[env_ids] = s["p0"]
    env._box_q0[env_ids] = s["q0"]
    env._box_v0[env_ids] = s["v0"]
    env._box_omega[env_ids] = s["omega"]
    env._box_launch_step[env_ids] = s["hover_steps"]
    env._box_launched[env_ids] = False
    env._box_aim_err[env_ids] = s["aim_err"]
    env._box_catch_h[env_ids] = s["catch_h"]
    env._box_tmult[env_ids] = s["t_mult"]
    env._box_jitter[env_ids] = s["jitter"]
    if not hasattr(env, "_box_lib_mask"):
        env._box_lib_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env._box_lib_mask[env_ids] = s["lib_mask"]
    if THROW_LIB_ALIGN:
        if not hasattr(env, "_box_lib_arrvec_w"):
            env._box_lib_arrvec_w = torch.zeros(env.num_envs, 2, device=device)
            env._box_lib_arr_dy = torch.zeros(env.num_envs, device=device)
            env._box_lib_tgt_x = torch.full((env.num_envs,), THROW_LIB_ALIGN_X, device=device)
        env._box_lib_arrvec_w[env_ids] = s["lib_arrvec_w"]
        env._box_lib_arr_dy[env_ids] = s["lib_arr_dy"]
        env._box_lib_tgt_x[env_ids] = s["lib_tgt_x"]

    box.write_root_pose_to_sim(torch.cat([s["p0"], s["q0"]], dim=-1), env_ids=env_ids)
    box.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)


def compliant_intake_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Human-like impact absorption: while the box is ARRIVING (within
    INTAKE_GATE_DIST of the pelvis and still flying faster than
    INTAKE_GATE_SPEED), reward the pelvis yielding along the box's travel
    direction and moving downward (squat).  Both components are
    velocity-based and capped; the gate closes once the box is held (slow), so
    there is nothing to farm outside the brief intake window."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    bp = box.data.root_pos_w
    bv = box.data.root_lin_vel_w
    rp = robot.data.root_pos_w
    rv = robot.data.root_lin_vel_w

    dist = (bp - rp).norm(dim=-1)
    speed = bv.norm(dim=-1)
    gate = (dist < INTAKE_GATE_DIST) & (speed > INTAKE_GATE_SPEED)

    # yield: pelvis velocity component along the box's horizontal travel dir
    dir_xy = bv[:, :2] / bv[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-3)
    yield_v = (rv[:, :2] * dir_xy).sum(dim=-1).clamp(0.0, YIELD_V_CAP) / YIELD_V_CAP
    # squat: downward pelvis velocity
    squat_v = (-rv[:, 2]).clamp(0.0, YIELD_V_CAP) / YIELD_V_CAP

    r = 0.6 * yield_v + 0.4 * squat_v
    return torch.where(gate, r, torch.zeros_like(r))


# ---------------------------------------------------------------- posture
# Wrists stay palm-up and steady; settle pose = slight LEAN-BACK, no forward
# crouch; legs may step in any direction but NO full splits.

def _get_posture_ids(env):
    if not hasattr(env, "_box_posture_ids"):
        robot = env.scene["robot"]
        jnames = robot.joint_names
        wrist = [i for i, n in enumerate(jnames) if "wrist" in n]
        bnames = robot.body_names
        env._box_posture_ids = dict(
            wrist=torch.tensor(wrist, dtype=torch.long, device=robot.device),
            torso=bnames.index("torso_link"),
            la=bnames.index("left_ankle_roll_link"),
            ra=bnames.index("right_ankle_roll_link"),
        )
    return env._box_posture_ids


def wrist_steady_penalty(env) -> torch.Tensor:
    """Wrist churn damper: penalize wrist joint VELOCITY only.  Pinning the
    wrists to the default pose instead fights reference tracking with an
    UNAVOIDABLE penalty, which teaches the policy to terminate early.
    Velocity-only is avoidable and scale-safe.  Returns positive badness;
    apply with a small negative weight."""
    ids = _get_posture_ids(env)["wrist"]
    robot = env.scene["robot"]
    return robot.data.joint_vel[:, ids].abs().sum(dim=-1)


def leg_split_penalty(env, max_split: float = 0.8) -> torch.Tensor:
    """Excess foot-to-foot horizontal distance beyond max_split (m).
    Free stepping below the threshold, splits punished linearly above."""
    d = _get_posture_ids(env)
    feet = env.scene["robot"].data.body_pos_w
    split = (feet[:, d["la"], :2] - feet[:, d["ra"], :2]).norm(dim=-1)
    return (split - max_split).clamp(min=0.0)


def torso_asym_penalty(env, fwd_max: float = 0.21, back_max: float = 0.49,
                       lat_max: float = 0.35) -> torch.Tensor:
    """Asymmetric torso attitude: forward lean past ~12° punished, BACKWARD
    allowed to ~28° (a slight lean-back settle is desired), lateral past ~20°
    punished.  gbx > 0 == forward lean."""
    d = _get_posture_ids(env)
    robot = env.scene["robot"]
    q = robot.data.body_quat_w[:, d["torso"]]
    w, x, y, z = q.unbind(-1)
    # projected gravity in torso frame: R^T @ (0,0,-1)
    gbx = -(2.0 * (x * z - w * y))
    gby = -(2.0 * (y * z + w * x))
    gbz = -(1.0 - 2.0 * (x * x + y * y))
    pitch = torch.atan2(gbx, (-gbz).clamp_min(1e-4))
    roll = torch.atan2(gby.abs(), (-gbz).clamp_min(1e-4))
    return ((pitch - fwd_max).clamp(min=0.0)
            + (-pitch - back_max).clamp(min=0.0)
            + (roll - lat_max).clamp(min=0.0))


def hands_under_box_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """TRUE-cradle shaping, against a vertical PINCH (one palm under + the
    other hand pressing on top).  Reward BOTH wrists being BELOW the box
    center while the box is in the hold zone: the top-pressing hand loses
    reward, pushing toward two-forearm under-support with the chest as
    backstop."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    # reuse posture id cache; wrist LINK heights via body indices
    if "lwrist" not in d:
        bnames = robot.body_names
        d["lwrist"] = bnames.index("left_wrist_yaw_link")
        d["rwrist"] = bnames.index("right_wrist_yaw_link")
    bz = box.data.root_pos_w[:, 2]
    lz = robot.data.body_pos_w[:, d["lwrist"], 2]
    rz = robot.data.body_pos_w[:, d["rwrist"], 2]
    under = 0.5 * (torch.sigmoid((bz - lz) / 0.08) + torch.sigmoid((bz - rz) / 0.08))
    near = (box.data.root_pos_w - robot.data.root_pos_w).norm(dim=-1) < INTAKE_GATE_DIST
    return torch.where(near, under, torch.zeros_like(under))


def forearm_cradle_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Cradle geometry: both forearms in FRONT of the chest, near-horizontal
    and PARALLEL — the forearm top surface is the bed.  Reward = (parallelism
    of the two forearm direction vectors) x (both pointing forward-horizontal),
    gated on the box being in the hold zone.  Bounded positive."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "lelbow" not in d:
        bnames = robot.body_names
        d["lelbow"] = bnames.index("left_elbow_link")
        d["relbow"] = bnames.index("right_elbow_link")
        d["lwrist"] = bnames.index("left_wrist_yaw_link")
        d["rwrist"] = bnames.index("right_wrist_yaw_link")
    bp = robot.data.body_pos_w
    dl = bp[:, d["lwrist"]] - bp[:, d["lelbow"]]
    dr = bp[:, d["rwrist"]] - bp[:, d["relbow"]]
    dl = dl / dl.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    dr = dr / dr.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    parallel = (0.5 * (1.0 + (dl * dr).sum(dim=-1))).clamp(0.0, 1.0)
    # forward = torso frame +fwd horizontal; use root yaw forward
    q = robot.data.root_quat_w
    w, x, y, z = q.unbind(-1)
    yaw_fwd = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)], dim=-1)
    yaw_fwd = yaw_fwd / yaw_fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    fwdness = 0.5 * ((dl[:, :2] * yaw_fwd).sum(-1) + (dr[:, :2] * yaw_fwd).sum(-1))
    fwdness = fwdness.clamp(0.0, 1.0)
    r = parallel * fwdness
    near = (box.data.root_pos_w - robot.data.root_pos_w).norm(dim=-1) < INTAKE_GATE_DIST
    return torch.where(near, r, torch.zeros_like(r))


def elbow_spread_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    base_spread: float = 0.45,
) -> torch.Tensor:
    """Elbows flaring sideways = side-clamp geometry.  Penalize elbow lateral
    spread beyond (base + 2*eff_radius) — the allowance ADAPTS to box size,
    so the opening is learned per box.  Gated on the hold zone."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "lelbow" not in d:
        bnames = robot.body_names
        d["lelbow"] = bnames.index("left_elbow_link")
        d["relbow"] = bnames.index("right_elbow_link")
    bp = robot.data.body_pos_w
    spread = (bp[:, d["lelbow"], :2] - bp[:, d["relbow"], :2]).norm(dim=-1)
    rad = getattr(env, "_balloon_radius_per_env", None)
    allow = base_spread + 2.0 * (rad if rad is not None else 0.14)
    excess = (spread - allow).clamp(min=0.0)
    near = (box.data.root_pos_w - robot.data.root_pos_w).norm(dim=-1) < INTAKE_GATE_DIST
    return torch.where(near, excess, torch.zeros_like(excess))


def forearm_asym_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    dz_deadzone: float = 0.05,
    ang_deadzone: float = 0.15,
) -> torch.Tensor:
    """Against asymmetric CLAMP recovery after a catch (box slides, then is
    pinched between one high and one low hand).  The level/parallel REWARDS
    can be traded away against the catch reward during the recovery; a penalty
    cannot.  Penalize (a) forearm height difference beyond 5 cm and
    (b) forearm direction misalignment beyond ~32 deg (1-cos > 0.15), gated
    on HELD (box near + slow, same gate as cradle_pose) — NOT the intake gate:
    asymmetric reach is the correct move for off-center throws and must never
    be penalized.  Fully avoidable via a symmetric hold; a typical clamp state
    (dz 0.15, ~45 deg) costs ~0.17 raw -> ~-0.10/step at weight -0.6, under
    the living reward (~0.17/step)."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "lelbow" not in d:
        bnames = robot.body_names
        d["lelbow"] = bnames.index("left_elbow_link")
        d["relbow"] = bnames.index("right_elbow_link")
        d["lwrist"] = bnames.index("left_wrist_yaw_link")
        d["rwrist"] = bnames.index("right_wrist_yaw_link")
    bp = robot.data.body_pos_w
    dl = bp[:, d["lwrist"]] - bp[:, d["lelbow"]]
    dr = bp[:, d["rwrist"]] - bp[:, d["relbow"]]
    dl = dl / dl.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    dr = dr / dr.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    misalign = (1.0 - (dl * dr).sum(dim=-1)).clamp(min=0.0)
    lmid_z = 0.5 * (bp[:, d["lelbow"], 2] + bp[:, d["lwrist"], 2])
    rmid_z = 0.5 * (bp[:, d["relbow"], 2] + bp[:, d["rwrist"], 2])
    dz = (lmid_z - rmid_z).abs()
    pen = (dz - dz_deadzone).clamp(min=0.0) \
        + 0.5 * (misalign - ang_deadzone).clamp(min=0.0)
    rel = box.data.root_pos_w - robot.data.root_pos_w
    held = (rel.norm(dim=-1) < 0.65) & (box.data.root_lin_vel_w.norm(dim=-1) < 1.2)
    return torch.where(held, pen, torch.zeros_like(pen))


def palm_up_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Wrists are PINNED, but shoulder-chain twist (shoulder_yaw + elbow)
    rotates the whole forearm, so the palms drift off up-facing — humans keep
    palms up by not twisting the arm.  Reward the world-up component of each
    wrist link's local -z axis (the same axis as the default
    BOX_GIMBAL_PALM="z"; see gimbal_action.py for the hand-mesh convention).
    Near-gated (reach + hold both want palms up)."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "lwrist" not in d:
        bnames = robot.body_names
        d["lwrist"] = bnames.index("left_wrist_yaw_link")
        d["rwrist"] = bnames.index("right_wrist_yaw_link")
    def up(idx):
        q = robot.data.body_quat_w[:, idx]
        w, x, y, z = q.unbind(-1)
        r22 = 1.0 - 2.0 * (x * x + y * y)   # world-z of local +z
        return (-r22).clamp(0.0, 1.0)       # local -z pointing world-up
    if os.environ.get("BOX_PALM_MIN", "0") == "1":
        # BOX_PALM_MIN=1: with the mean, the policy can bank the reward on one
        # palm while the other stays twisted.  Score by the WORSE hand — all
        # gradient lands on the laggard.
        r = torch.minimum(up(d["lwrist"]), up(d["rwrist"]))
    else:
        r = 0.5 * (up(d["lwrist"]) + up(d["rwrist"]))
    if os.environ.get("BOX_PALM_ALLPHASE", "0") == "1":
        # BOX_PALM_ALLPHASE=1: far from the box, shoulder roll could otherwise
        # twist the palms unpenalized; palms-up applies in EVERY phase — pay
        # everywhere, no gate.
        return r
    near = (box.data.root_pos_w - robot.data.root_pos_w).norm(dim=-1) < INTAKE_GATE_DIST
    return torch.where(near, r, torch.zeros_like(r))


def _bed_above_mean(env, box: RigidObject, robot) -> torch.Tensor:
    """Two-arm mean of the tiered offside 'above' score.  Same math as
    box_on_forearm_bed_reward (kept in sync by hand)."""
    d = _get_posture_ids(env)
    if "lelbow" not in d:
        bnames = robot.body_names
        d["lelbow"] = bnames.index("left_elbow_link")
        d["relbow"] = bnames.index("right_elbow_link")
        d["lwrist"] = bnames.index("left_wrist_yaw_link")
        d["rwrist"] = bnames.index("right_wrist_yaw_link")
    bp = robot.data.body_pos_w
    box_p = box.data.root_pos_w
    qw, qx, qy, qz = box.data.root_quat_w.unbind(-1)
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    he = env._box_half_extents if hasattr(env, "_box_half_extents") else \
        torch.full((box_p.shape[0], 3), 0.13, device=box_p.device)
    drop = r20.abs() * he[:, 0] + r21.abs() * he[:, 1] + r22.abs() * he[:, 2]
    box_bottom_z = box_p[:, 2] - drop

    def one(e_idx, w_idx):
        mid_z = 0.5 * (bp[:, e_idx, 2] + bp[:, w_idx, 2])
        a = (box_bottom_z - (mid_z + 0.045)).abs()

        def _ramp(lo, hi):
            return 1.0 - ((a - lo) / (hi - lo)).clamp(0.0, 1.0)

        return (0.2 + 0.3 * _ramp(0.05, 0.06) + 0.5 * _ramp(0.10, 0.12)) \
            * _ramp(0.20, 0.32)

    return 0.5 * (one(d["lelbow"], d["lwrist"]) + one(d["relbow"], d["rwrist"]))


def bed_gated_hold_sustained(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Only a bed hold counts as a hold — clamped / hand-carry holds earn NO
    sustained reward.  The inherited strict persistence term
    (chest_v5_terms.strict_hold_sustained_reward, unchanged) x a smooth gate
    on the tiered 'above' score.  The gate keeps gradient everywhere the tiers
    do (trickle at 15 cm offside, ~full from the 0.7 tier up), so there is no
    sparse-AND desert: gate = clamp((above - 0.15) / 0.6, 0, 1)."""
    from reflex.envs.base.chest_v5_terms import strict_hold_sustained_reward
    base = strict_hold_sustained_reward(env, asset_cfg)
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    gate = ((_bed_above_mean(env, box, robot) - 0.15) / 0.6).clamp(0.0, 1.0)
    if os.environ.get("BOX_UPRIGHT_GATE", "0") == "1":
        # BOX_UPRIGHT_GATE=1: a reclined hold does not count either (the same
        # gating idea that made the bed hold work where penalties and positive
        # rewards had failed): route the sustained reward through an upright
        # gate too.  Full-range ramp — 0 at gbx -0.9 (deep recline), 1 at
        # -0.31 (18° back); a moderate recline still earns a trickle so the
        # gradient reaches it.  gbx comes from _posture_gbx (BOX_GATE_TORSO=1
        # reads the TORSO quat so waist-pitch recline is visible; default = root).
        gbx = _posture_gbx(env)
        # Floor (BOX_GATE_FLOOR, default 0.35): with a zero floor, transition
        # holds earn too little to be worth completing and the catch rate
        # collapses.
        # Ramp start (BOX_GATE_START, default -0.9) is configurable: the wrist
        # gimbal decouples palm orientation from posture, removing an implicit
        # anti-recline pressure; tightening the ramp start restores it.
        _gs = float(os.environ.get("BOX_GATE_START", "-0.9"))
        _ge = float(os.environ.get("BOX_GATE_END", "-0.31"))
        _gf = float(os.environ.get("BOX_GATE_FLOOR", "0.35"))
        # A deep recline can also live off the floor + other rewards, so the
        # floor is configurable: e.g. 0.1 nearly starves deep recline while
        # the accepted band stays comfortable (ramp to 1.0 by _ge).
        gate_u = _gf + (1.0 - _gf) * ((gbx - _gs) / (_ge - _gs)).clamp(0.0, 1.0)
        gate = gate * gate_u
    if os.environ.get("BOX_3POINT", "0") == "1":
        # BOX_3POINT: a true catch = caught + stable three-point hold (two
        # forearms as the bed + chest) + standing still.  Three more gates
        # multiply into the sustained reward: chest contact, box stillness,
        # feet planted.  Floors keep gradient everywhere (a zero floor starves
        # transition holds).
        if os.environ.get("BOX_3POINT_CHEST", "reward") == "gap":
            _gap = _box_chest_gap(env, asset_cfg)
            cc = 1.0 - ((_gap - 0.02) / 0.06).clamp(0.0, 1.0)  # full credit <= 2 cm, zero >= 8 cm
        else:
            cc = chest_contact_reward(env, asset_cfg).clamp(0.0, 1.0)
        gate_c = 0.15 + 0.85 * cc
        bav = box.data.root_ang_vel_w.norm(dim=-1)
        blv = box.data.root_lin_vel_w.norm(dim=-1)
        still = (1.0 - ((bav - 1.0) / 2.0).clamp(0.0, 1.0)) * \
                (1.0 - ((blv - 0.3) / 0.5).clamp(0.0, 1.0))
        gate_s = 0.15 + 0.85 * still
        d = _get_posture_ids(env)
        feet_v = robot.data.body_lin_vel_w[:, [d["la"], d["ra"]]].norm(
            dim=-1).amax(dim=-1)
        planted = 1.0 - ((feet_v - 0.10) / 0.4).clamp(0.0, 1.0)
        gate_f = 0.15 + 0.85 * planted
        gate = gate * gate_c * gate_s * gate_f
    return base * gate


def threepoint_hold_bonus(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    hold_sec: float = 3.0,
) -> torch.Tensor:
    """Success bonus for a true catch: one-shot bonus on the step a
    fully-qualified hold has persisted hold_sec continuously — qualified =
    bed above + upright torso + chest contact + box still + feet planted, ALL
    as booleans at moderate cuts.  Dense shaping lives in the gated sustained
    term; this is the success DEFINITION."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    if not hasattr(env, "_tp_count"):
        env._tp_count = torch.zeros(env.num_envs, dtype=torch.long,
                                    device=env.device)
    above = _bed_above_mean(env, box, robot)
    gbx = _posture_gbx(env)
    if os.environ.get("BOX_3POINT_CHEST", "reward") == "gap":
        cc = (_box_chest_gap(env, asset_cfg) < 0.02).float()
    else:
        cc = chest_contact_reward(env, asset_cfg)
    bav = box.data.root_ang_vel_w.norm(dim=-1)
    blv = box.data.root_lin_vel_w.norm(dim=-1)
    d = _get_posture_ids(env)
    feet_v = robot.data.body_lin_vel_w[:, [d["la"], d["ra"]]].norm(
        dim=-1).amax(dim=-1)
    ok = ((above > 0.4) & (gbx > -0.45) & (cc > 0.5)
          & (bav < 1.5) & (blv < 0.4) & (feet_v < 0.2))
    env._tp_count = torch.where(ok, env._tp_count + 1,
                                torch.zeros_like(env._tp_count))
    T = max(1, int(hold_sec / env.step_dt))
    fire = env._tp_count == T          # exactly-once at crossing
    return fire.float()


def hold_foot_step_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    deadzone: float = 0.12,
) -> torch.Tensor:
    """Shuffling penalty — foot speed above deadzone while HOLDING the box
    costs linearly (vs the living reward ~0.17/step: a stepping swing of
    ~0.6 m/s x w0.2 = -0.10/step, avoidable by standing still)."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    feet_v = robot.data.body_lin_vel_w[:, [d["la"], d["ra"]]].norm(
        dim=-1).amax(dim=-1)
    pen = (feet_v - deadzone).clamp(min=0.0)
    rel = box.data.root_pos_w - robot.data.root_pos_w
    held = (rel.norm(dim=-1) < 0.65) & (box.data.root_lin_vel_w.norm(dim=-1) < 1.2)
    return torch.where(held, pen, torch.zeros_like(pen))


# ---- GATE_ALL: upright gate over ALL hold-reward terms ----
# The posture-blind hold terms (box_on_bed + chest_pocket + hold_still +
# hands_under_box + low_hold, ~10/step) out-pay any posture reward/penalty at
# the margin: deep-recline "shelf" holds are gravity-backstopped and
# risk-free, so the policy gives up upright_hold, accepts the torso_asym
# penalty and keeps reclining.  Gating only the sustained term is not enough
# (the lean lives off these other terms, see bed_gated_hold_sustained).
# GATE_ALL routes every hold-reward term through the same upright ramp, so
# recline starves across the board while the ramp keeps a gradient back to
# upright.
#
# Wrappers are module-level named functions (clean env.yaml dump); the
# original funcs are captured at cfg time into _UGATE_INNER by the
# BOX_GATE_ALL block in catch_env_cfg.py — whatever variant the inheritance
# chain installed is what gets wrapped.

_UGATE_INNER: dict = {}


def _posture_gbx(env) -> torch.Tensor:
    """The gbx (x component of gravity in the body frame; > 0 = forward lean)
    that every upright gate/reward reads.  BOX_GATE_TORSO=1 → TORSO body
    quat; default → ROOT (pelvis) quat.

    Why the torso: in a reclined hold the lean lives in the WAIST joint while
    the pelvis stays near-upright (e.g. torso gbx −0.82 vs pelvis −0.42), so
    root-quat gates/rewards pay almost full credit to a ~55° torso recline."""
    robot = env.scene["robot"]
    if os.environ.get("BOX_GATE_TORSO", "0") == "1":
        d = _get_posture_ids(env)
        q = robot.data.body_quat_w[:, d["torso"]]
    else:
        q = robot.data.root_quat_w
    w, x, y, z = q.unbind(-1)
    return -(2.0 * (x * z - w * y))


def _posture_gravity_b(env) -> torch.Tensor:
    """Full gravity vector in the posture body frame (same body selection as
    _posture_gbx: torso with BOX_GATE_TORSO=1, else pelvis). (B, 3), unit."""
    robot = env.scene["robot"]
    if os.environ.get("BOX_GATE_TORSO", "0") == "1":
        d = _get_posture_ids(env)
        q = robot.data.body_quat_w[:, d["torso"]]
    else:
        q = robot.data.root_quat_w
    w, x, y, z = q.unbind(-1)
    gx = -(2.0 * (x * z - w * y))
    gy = -(2.0 * (y * z + w * x))
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return torch.stack([gx, gy, gz], dim=-1)


def lean_excess_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_angle_rad: float = 0.5236,
) -> torch.Tensor:
    """ALL-PHASE penalty on torso lean beyond a dead-zone, returned as
    -(tilt - max_angle) [rad], 0 inside.

    Why not BOX_UPRIGHT_W: upright_hold is a held-gated reward that starts
    paying the instant the box is held, so the policy shortens the catch
    lunge to collect it sooner.  This term is silent below the dead-zone (a
    <30 deg lunge costs nothing) and bites only on sustained post-catch bows
    and a backward sit onto a dropped box.  Not gated on the box, so a bow
    toward a box on the floor is penalized too.  Avoidable => no incentive to
    terminate early (5 deg excess = 0.087 raw per step)."""
    g = _posture_gravity_b(env)
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    return -(tilt - max_angle_rad).clamp(min=0.0)


def _upright_gate_factor(env) -> torch.Tensor:
    """Same gbx ramp + env vars as bed_gated_hold_sustained's gate_u.
    Separate floor knob BOX_GATE_ALL_FLOOR (default 0.1): a zero floor
    starves transition holds and collapses the catch rate — deep recline
    must be starved, not the straighten-up path."""
    gbx = _posture_gbx(env)
    _gs = float(os.environ.get("BOX_GATE_START", "-0.9"))
    _ge = float(os.environ.get("BOX_GATE_END", "-0.31"))
    _gf = float(os.environ.get("BOX_GATE_ALL_FLOOR", "0.1"))
    return _gf + (1.0 - _gf) * ((gbx - _gs) / (_ge - _gs)).clamp(0.0, 1.0)


# NOTE explicit signatures: Isaac Lab's manager introspection treats
# `**kwargs` as a mandatory parameter literally named 'kwargs' and rejects
# the term ("expects mandatory parameters: ['kwargs']"); in a headless run
# the resulting error can leave the app hanging without a traceback.  Each
# wrapper mirrors its inner term's params.


def ugated_box_on_bed(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")
) -> torch.Tensor:
    return _UGATE_INNER["box_on_bed"](env, asset_cfg) * _upright_gate_factor(env)


def ugated_chest_pocket(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")
) -> torch.Tensor:
    return _UGATE_INNER["chest_pocket"](env, asset_cfg) * _upright_gate_factor(env)


def ugated_hold_still(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")
) -> torch.Tensor:
    return _UGATE_INNER["hold_still"](env, asset_cfg) * _upright_gate_factor(env)


def ugated_hands_under_box(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")
) -> torch.Tensor:
    return _UGATE_INNER["hands_under_box"](env, asset_cfg) * _upright_gate_factor(env)


def ugated_low_hold(
    env, z_low: float = 0.10, z_high: float = 0.30, near_max: float = 0.6
) -> torch.Tensor:
    return _UGATE_INNER["low_hold"](env, z_low=z_low, z_high=z_high,
                                    near_max=near_max) * _upright_gate_factor(env)


def face_box_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Keep the robot SQUARED UP to the box (prevents heading drift; the box
    always comes from the front and the robot should always face it).
    Reward = cos(yaw error to box), gated on the box being airborne or in
    the hold zone."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    w, x, y, z = q.unbind(-1)
    fwd = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)], dim=-1)
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    to_box = box.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist = to_box.norm(dim=-1)
    to_box = to_box / dist.clamp_min(1e-6).unsqueeze(-1)
    align = ((fwd * to_box).sum(dim=-1)).clamp(0.0, 1.0)
    z_rel = box.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    active = (z_rel > 0.35) | (dist < INTAKE_GATE_DIST)
    return torch.where(active, align, torch.zeros_like(align))


def box_on_forearm_bed_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """THE bed criterion: the box lying ON both forearms (a parallel-forearm
    proxy alone let the policy rotate its forearms and park the box on the
    chest/shoulder instead).  Per arm:
      horiz    - forearm direction near-horizontal (top surface usable as bed)
      above    - box BOTTOM face on the forearm top surface (tiered tolerance)
      over     - box horizontally covers the forearm midpoint
    plus `level` (both forearms at the same height).  Reward = 0.6 x additive
    partial credit (weights _BED_MIX) + 0.4 x full-bed product (geometric
    mean over the two arms x level), gated on the hold zone: BOTH arms must
    support for full credit."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "lelbow" not in d:
        bnames = robot.body_names
        d["lelbow"] = bnames.index("left_elbow_link")
        d["relbow"] = bnames.index("right_elbow_link")
        d["lwrist"] = bnames.index("left_wrist_yaw_link")
        d["rwrist"] = bnames.index("right_wrist_yaw_link")
    bp = robot.data.body_pos_w
    box_p = box.data.root_pos_w

    # Orientation-aware BOTTOM-FACE height ("offside rule"): a center-height
    # check pays straight-arm END-CLAMPS because the box CENTER sits at hand
    # height too.  Support means the box's LOWEST point rests ON the forearm
    # top surface, ±5 cm tolerance.
    bq = box.data.root_quat_w
    qw, qx, qy, qz = bq.unbind(-1)
    # third row of the rotation matrix = world-z components of body axes
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    he = env._box_half_extents if hasattr(env, "_box_half_extents") else \
        torch.full((box_p.shape[0], 3), 0.13, device=box_p.device)
    drop = (r20.abs() * he[:, 0] + r21.abs() * he[:, 1] + r22.abs() * he[:, 2])
    box_bottom_z = box_p[:, 2] - drop

    def arm_parts(e_idx, w_idx):
        e, wr = bp[:, e_idx], bp[:, w_idx]
        seg = wr - e
        seg_dir = seg / seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        horiz = torch.exp(-(seg_dir[:, 2] / 0.35) ** 2)
        mid = 0.5 * (e + wr)
        forearm_top_z = mid[:, 2] + 0.045
        # Tiered offside score (catches come at all angles, exact surface
        # contact is rare): <=5cm full, 6-10cm 0.7,
        # 12-20cm 0.2, ->0 by ~32cm. Short ramps between tiers keep gradient.
        a = (box_bottom_z - forearm_top_z).abs()

        def _ramp(lo, hi):
            return 1.0 - ((a - lo) / (hi - lo)).clamp(0.0, 1.0)

        above = (0.2 + 0.3 * _ramp(0.05, 0.06) + 0.5 * _ramp(0.10, 0.12)) * _ramp(0.20, 0.32)
        over = torch.exp(-((box_p[:, :2] - mid[:, :2]).norm(dim=-1) / 0.25) ** 2)
        return horiz, above, over

    lh, la, lo = arm_parts(d["lelbow"], d["lwrist"])
    rh, ra, ro = arm_parts(d["relbow"], d["rwrist"])
    # BOTH forearms at the SAME HEIGHT: each arm horizontal + parallel is not
    # enough — a left-high/right-low pair is a side ramp and the box slides
    # off.  Same 5 cm tolerance as the offside rule.
    lmid_z = 0.5 * (bp[:, d["lelbow"], 2] + bp[:, d["lwrist"], 2])
    rmid_z = 0.5 * (bp[:, d["relbow"], 2] + bp[:, d["rwrist"], 2])
    level = torch.exp(-((lmid_z - rmid_z) / 0.05) ** 2)
    # ADDITIVE partial credit: a product of narrow gaussians has no gradient
    # until nearly solved (sparse-AND trap).  The full-bed product is kept as
    # a smaller bonus.  The BOX_V19 mix shifts weight onto the two terms that
    # actually encode "box ON level forearms" (above, level), since horiz/over
    # partial credit can be collected while the box sits 20-30 cm off the bed.
    horiz = 0.5 * (lh + rh)
    above = 0.5 * (la + ra)
    over = 0.5 * (lo + ro)
    wh, wa, wo, wl = _BED_MIX
    partial = wh * horiz + wa * above + wo * over + wl * level
    full = torch.sqrt((lh * la * lo) * (rh * ra * ro) + 1e-9) * level
    r = 0.6 * partial + 0.4 * full
    near = (box_p - robot.data.root_pos_w).norm(dim=-1) < INTAKE_GATE_DIST
    return torch.where(near, r, torch.zeros_like(r))


# Cradle arm pose, extracted from the reference clip's catch-window frames
# (shoulders raised FORWARD ~57°, elbows near-straight, forearms converged
# into the bed).
CRADLE_POSE = {
    "left_shoulder_pitch_joint": -1.005,
    "right_shoulder_pitch_joint": -0.919,
    "left_shoulder_roll_joint": 0.187,
    "right_shoulder_roll_joint": -0.170,
    "left_shoulder_yaw_joint": -0.361,
    "right_shoulder_yaw_joint": 0.376,
    "left_elbow_joint": 0.537,
    "right_elbow_joint": 0.386,
}
# BOX_POSE_SYM=1: the raw clip pose above is (1) ASYMMETRIC (right elbow
# 0.15 rad straighter -> box rests tilted -> right forearm twists to shim it)
# and (2) puts the bed at SHOULDER height (shoulders at -1.0), where "box on
# bed + against chest + upright torso" is geometrically impossible, so the
# policy reclines ~45° instead.  Symmetrize L/R and LOWER the bed to
# mid-chest so an upright tuck exists mechanically.
if os.environ.get("BOX_POSE_SYM", "0") == "1":
    CRADLE_POSE = {
        "left_shoulder_pitch_joint": -0.85,
        "right_shoulder_pitch_joint": -0.85,
        "left_shoulder_roll_joint": 0.179,
        "right_shoulder_roll_joint": -0.179,
        "left_shoulder_yaw_joint": -0.369,
        "right_shoulder_yaw_joint": 0.369,
        "left_elbow_joint": 0.462,
        "right_elbow_joint": 0.462,
    }
# BOX_ELBOW_SCOOP: with wrists LOCKED flat, an upright torso leaves the bed
# with NO forward lip — the box can roll out, so the policy learns to recline
# as a gravity backstop (the reference clip itself is upright).  More elbow
# flexion tilts the forearms up toward the hands: the bed slopes INTO the
# chest, giving gravity retention with an upright torso — the human carry.
_scoop = float(os.environ.get("BOX_ELBOW_SCOOP", "0"))
if _scoop > 0:
    CRADLE_POSE["left_elbow_joint"] = _scoop
    CRADLE_POSE["right_elbow_joint"] = _scoop
    # deep flexion + full fingertip convergence self-collides (e.g. at 1.1
    # rad); relax the yaw convergence so the hands rise cleanly
    CRADLE_POSE["left_shoulder_yaw_joint"] = -0.25
    CRADLE_POSE["right_shoulder_yaw_joint"] = 0.25


def _cradle_hold_target(env, d):
    """HOLD member of the arm pose family (BOX_HOLD_POSE_CURL=1) = ready pose
    with elbows curled (BOX_HOLD_ELBOW, default 0.95) so the bed tilts in and
    the box can rest against the chest.  Shoulders unchanged (held near the
    ready pose by BOX_ARM_SCALE anyway).  Aligned with d["cradle_ids"]."""
    if "cradle_hold_target" not in d:
        _he = float(os.environ.get("BOX_HOLD_ELBOW", "0.95"))
        vals = []
        for name, val in CRADLE_POSE.items():
            vals.append(_he if "_elbow_" in name else val)
        d["cradle_hold_target"] = torch.tensor(
            vals, device=env.scene["robot"].device)
    return d["cradle_hold_target"]


def cradle_pose_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """HOLD-phase arm-pose prior = the reference clip's own catch pose
    (geometric proxies proved exploitable, so the hold pose is anchored
    directly in joint space).  Gated on held (box near + slow); intake stays
    unconstrained."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "cradle_ids" not in d:
        jn = robot.joint_names
        ids, targets = [], []
        for name, val in CRADLE_POSE.items():
            ids.append(jn.index(name))
            targets.append(val)
        d["cradle_ids"] = torch.tensor(ids, dtype=torch.long, device=robot.device)
        d["cradle_target"] = torch.tensor(targets, device=robot.device)
    dist = (box.data.root_pos_w - robot.data.root_pos_w).norm(dim=-1)
    bspeed = box.data.root_lin_vel_w.norm(dim=-1)
    held = (dist < 0.65) & (bspeed < 1.2)
    _tgt = d["cradle_target"]
    if os.environ.get("BOX_HOLD_POSE_CURL", "0") == "1":
        _tgt = _cradle_hold_target(env, d)
    dev = (robot.data.joint_pos[:, d["cradle_ids"]] - _tgt[None, :]).abs().mean(dim=-1)
    if os.environ.get("BOX_CRADLE_RAMP", "0") == "1":
        # BOX_CRADLE_RAMP=1: the gaussian has ~zero gradient far from the
        # target (dev ~1.0 → r=0.05), so a policy holding a different shape
        # never feels it.  Full-range linear ramp: every radian toward the
        # cradle pose pays immediately; 0.15 rad deadzone so physically
        # wrapping the box isn't fighting the anchor.
        r = 1.0 - ((dev - 0.15) / 1.0).clamp(0.0, 1.0)
    else:
        r = torch.exp(-dev / 0.35)
    return torch.where(held, r, torch.zeros_like(r))


def box_hold_spin_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Nothing else prices the BOX'S OWN rotation — every hold term watches
    the box CENTER (pocket/lateral/bed), so a box spinning in place on the
    cradle is free, and throws arrive with up to 1.5 rev/s of spin that the
    hold never damps.  Penalize box angular speed while held; a settled box
    costs nothing."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    rel = box.data.root_pos_w - robot.data.root_pos_w
    held = (rel.norm(dim=-1) < 0.65) & (box.data.root_lin_vel_w.norm(dim=-1) < 1.2)
    spin = box.data.root_ang_vel_w.norm(dim=-1)
    return torch.where(held, spin, torch.zeros_like(spin))


def arm_idle_still_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    far_dist: float = 1.5,
) -> torch.Tensor:
    """Against anticipatory arm flailing while waiting: a slow 2-3 m/s
    balloon leaves time to raise the arms during flight, but a 5-7.6 m/s box
    with a randomly timed launch teaches permanent hedge-twitching.  Calm
    waiting must be BOUGHT: penalize arm joint speed while the box is far
    (> far_dist).  The reach and hold stay unpenalized."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "arm_ids" not in d:
        jn = robot.joint_names
        d["arm_ids"] = torch.tensor(
            [i for i, n in enumerate(jn)
             if ("shoulder" in n or "elbow" in n) and "wrist" not in n],
            dtype=torch.long, device=robot.device)
    v = robot.data.joint_vel[:, d["arm_ids"]].abs().mean(dim=-1)
    far = (box.data.root_pos_w - robot.data.root_pos_w).norm(dim=-1) > far_dist
    return torch.where(far, v, torch.zeros_like(v))


def base_idle_still_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    far_dist: float = 1.5,
    box_speed_max: float = 1.0,
    deadzone_mps: float = 0.03,
    leg_gain: float = 0.2,
) -> torch.Tensor:
    """Against fidgeting with no box in sight (observed on hardware as idle
    base-speed RMS of 0.08-0.17 m/s and leg joint speeds of 0.6-1.6 rad/s,
    vs ~0.03 m/s and ~0.12 rad/s for a calm policy).  Calm waiting must be
    bought (cf. arm_idle_still_penalty for the arms): penalize base xy speed
    (above a small dead-zone) plus leg/waist joint speed while the box is FAR
    and NOT FLYING (hover / pre-throw / a box resting on the floor).  A
    flying box -- even 3 m out -- lifts the gate, so the early step toward
    the box is unpenalized.  Avoidable => no incentive to terminate early."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "leg_waist_ids" not in d:
        jn = robot.joint_names
        d["leg_waist_ids"] = torch.tensor(
            [i for i, n in enumerate(jn)
             if ("hip" in n or "knee" in n or "ankle" in n or "waist" in n)],
            dtype=torch.long, device=robot.device)
    far = (box.data.root_pos_w - robot.data.root_pos_w).norm(dim=-1) > far_dist
    calm_box = box.data.root_lin_vel_w.norm(dim=-1) < box_speed_max
    base_v = robot.data.root_lin_vel_w[:, :2].norm(dim=-1)
    leg_v = robot.data.joint_vel[:, d["leg_waist_ids"]].abs().mean(dim=-1)
    pen = (base_v - deadzone_mps).clamp(min=0.0) + leg_gain * leg_v
    return torch.where(far & calm_box, pen, torch.zeros_like(pen))


def cradle_ready_dev_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    deadzone: float = 0.3,
) -> torch.Tensor:
    """The cradle_ready reward alone (even at 1.2, all-phase) does not stop
    anticipation twitching — gates and penalties bind where positive rewards
    don't.  Deadzoned PENALTY on mean arm-joint deviation from CRADLE_POSE
    while not held: within 0.3 rad free (micro adjustments unpenalized),
    beyond costs linearly."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "cradle_ids" not in d:
        jn = robot.joint_names
        ids, targets = [], []
        for name, val in CRADLE_POSE.items():
            ids.append(jn.index(name))
            targets.append(val)
        d["cradle_ids"] = torch.tensor(ids, dtype=torch.long, device=robot.device)
        d["cradle_target"] = torch.tensor(targets, device=robot.device)
    dev = (robot.data.joint_pos[:, d["cradle_ids"]] - d["cradle_target"][None, :]).abs().mean(dim=-1)
    if os.environ.get("BOX_HOLD_POSE_CURL", "0") == "1":
        _hdev = (robot.data.joint_pos[:, d["cradle_ids"]]
                 - _cradle_hold_target(env, d)[None, :]).abs().mean(dim=-1)
        _rel = env.scene[asset_cfg.name].data.root_pos_w - robot.data.root_pos_w
        _held = (_rel.norm(dim=-1) < 0.65) & \
            (env.scene[asset_cfg.name].data.root_lin_vel_w.norm(dim=-1) < 1.2)
        dev = torch.where(_held, _hdev, dev)
    pen = (dev - deadzone).clamp(min=0.0)
    rel = box.data.root_pos_w - robot.data.root_pos_w
    held = (rel.norm(dim=-1) < 0.65) & (box.data.root_lin_vel_w.norm(dim=-1) < 1.2)
    if os.environ.get("BOX_READY_DEV_ALLPHASE", "0") == "1":
        # BOX_READY_DEV_ALLPHASE=1: otherwise the arms can collapse into a
        # clamp-hug after the catch (held dev ~1.0) without penalty.  The
        # deadzoned penalty applies in EVERY phase incl. held — ready / catch /
        # hold are ONE pose family; deviation beyond 0.3 rad costs everywhere.
        return pen
    return torch.where(~held, pen, torch.zeros_like(pen))


def upright_hold_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    target_gbx: float = -0.15,
    sigma: float = 0.2,
) -> torch.Tensor:
    """POSITIVE reward for holding the box with an upright, slightly
    leaned-back torso (peak at gbx -0.15 ≈ 8.6° back).  A sigma-0.2 gaussian
    has ~ZERO gradient at a deep recline (gbx -0.8), so a full-range ramp is
    used instead (`sigma` is unused): full credit within ±0.1 of target,
    linear decay to 0 at 0.8 away — every degree of straightening pays
    immediately.  Held-gated."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    # _posture_gbx: with BOX_GATE_TORSO=1 the reward sees the waist-pitch
    # recline (the root quat alone pays r=0.76 to a 55° torso).
    gbx = _posture_gbx(env)
    dev = (gbx - target_gbx).abs()
    r = 1.0 - ((dev - 0.1) / 0.7).clamp(0.0, 1.0)
    rel = box.data.root_pos_w - robot.data.root_pos_w
    held = (rel.norm(dim=-1) < 0.65) & (box.data.root_lin_vel_w.norm(dim=-1) < 1.2)
    return torch.where(held, r, torch.zeros_like(r))


def cradle_ready_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Idle-stance shaping: without it the learned WAITING pose can be a boxer
    guard (elbows fully curled, hands at the face — rational under the
    action-rate penalty and ~1.2 s flights, but unnatural).  Shape the idle
    stance into a delivery-worker ready pose instead: same CRADLE_POSE
    targets (shoulders forward, elbows near-straight), small weight, active
    only while the box is FAR (> INTAKE_GATE_DIST).  The reach window (near
    but not held) stays entirely unconstrained, so catching itself is
    untouched."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "cradle_ids" not in d:
        jn = robot.joint_names
        ids, targets = [], []
        for name, val in CRADLE_POSE.items():
            ids.append(jn.index(name))
            targets.append(val)
        d["cradle_ids"] = torch.tensor(ids, dtype=torch.long, device=robot.device)
        d["cradle_target"] = torch.tensor(targets, device=robot.device)
    dev = (robot.data.joint_pos[:, d["cradle_ids"]] - d["cradle_target"][None, :]).abs().mean(dim=-1)
    r = torch.exp(-dev / 0.35)
    if os.environ.get("BOX_READY_ALLPHASE", "0") == "1":
        # BOX_READY_ALLPHASE=1: ready / catch / hold all use this pose, so
        # anchor every non-held frame — idle, flight AND the reach window.
        # Soft gaussian => micro-adjust intercepts stay cheap; catch bonuses
        # dwarf the anchor cost.  Held frames belong to cradle_pose.
        rel = box.data.root_pos_w - robot.data.root_pos_w
        held = (rel.norm(dim=-1) < 0.65) & (box.data.root_lin_vel_w.norm(dim=-1) < 1.2)
        return torch.where(~held, r, torch.zeros_like(r))
    far = (box.data.root_pos_w - robot.data.root_pos_w).norm(dim=-1) > INTAKE_GATE_DIST
    return torch.where(far, r, torch.zeros_like(r))


def hold_still_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Post-catch stabilization (failure mode: a clean forearm+chest intake,
    then the body keeps performing the reference clip, the box slides off and
    the policy rescues it with a side clamp).  While the
    box is HELD (near + slow), reward (a) box velocity matching the root
    (carried, not sliding) and (b) low arm joint velocity (freeze into the
    carry). Positive, bounded, gated."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    d = _get_posture_ids(env)
    if "arm_joints" not in d:
        jn = robot.joint_names
        d["arm_joints"] = torch.tensor(
            [i for i, n in enumerate(jn) if ("shoulder" in n or "elbow" in n)],
            dtype=torch.long, device=robot.device)
    dist = (box.data.root_pos_w - robot.data.root_pos_w).norm(dim=-1)
    bspeed = box.data.root_lin_vel_w.norm(dim=-1)
    held = (dist < 0.65) & (bspeed < 1.2)
    rel_v = (box.data.root_lin_vel_w - robot.data.root_lin_vel_w).norm(dim=-1)
    arm_vel = robot.data.joint_vel[:, d["arm_joints"]].abs().mean(dim=-1)
    # Arm-freeze only AFTER the box is tucked at the chest (<0.38 m) — during
    # the goalkeeper draw-in (0.38-0.65 m) the arms MUST move (the bed
    # retracts to press the box onto the chest).
    tucked = dist < 0.38
    freeze = torch.where(tucked, torch.exp(-arm_vel / 2.0), torch.ones_like(arm_vel))
    r = torch.exp(-rel_v / 0.25) * freeze
    return torch.where(held, r, torch.zeros_like(r))


def tuck_in_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> torch.Tensor:
    """Goalkeeper draw-in: after the bed catch, reward the
    box moving TOWARD the chest until tucked — bed retracts, box ends pressed
    between forearms and chest."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    to_chest = robot.data.root_pos_w - box.data.root_pos_w
    dist = to_chest.norm(dim=-1)
    dirn = to_chest / dist.clamp_min(1e-6).unsqueeze(-1)
    rel_vel = box.data.root_lin_vel_w - robot.data.root_lin_vel_w
    # GENTLE gesture only (cardboard crushes — shape the motion, don't reward
    # force): slow closing cap, stop at a light-touch standoff instead of
    # pressing into the chest.
    closing = (rel_vel * dirn).sum(dim=-1).clamp(0.0, 0.25) / 0.25
    bspeed = box.data.root_lin_vel_w.norm(dim=-1)
    drawing = (dist > 0.42) & (dist < 0.65) & (bspeed < 2.0)
    return torch.where(drawing, closing, torch.zeros_like(closing))


def box_hover_and_launch(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
) -> None:
    """Every-step event (interval (0,0)): hold hovering boxes at the release
    point until each env's launch step, then write the launch velocity once."""
    # Flight reward gate (BOX_V38_FLIGHT_RGATE): idempotent runtime patch of
    # the reward manager, see teacher_terms._apply_flight_rgate.
    try:
        from reflex.envs.teacher_terms import _apply_flight_rgate
        _apply_flight_rgate(env)
    except Exception as _e:
        if not getattr(env, "_frg_err_once", False):
            env._frg_err_once = True
            print(f"[throw] flight reward gate apply FAILED: {_e}", flush=True)
    del env_ids  # operate on all envs every step
    if not hasattr(env, "_box_p0"):
        return
    box: RigidObject = env.scene[asset_cfg.name]
    device = box.device
    step = env.episode_length_buf  # (N,)

    hovering = (~env._box_launched) & (step < env._box_launch_step)
    launching = (~env._box_launched) & (step >= env._box_launch_step)

    # PERF: per-subset writes with .any() + .nonzero() cost several GPU->CPU
    # syncs every env step (~8 ms/step).  Full-batch masked writes have ZERO
    # syncs: untouched envs get their own current state written back
    # (physically a no-op).
    robot = env.scene["robot"]
    lib_align_track(env, hovering, launching)   # no-op unless BOX_THROW_LIB_ALIGN is set
    origins_all = env.scene.env_origins
    p0 = env._box_p0
    # launch-time RE-AIM at the robot's live position (thrower watches the
    # walking robot); computed for ALL envs, applied only where `launching`.
    to_robot = robot.data.root_pos_w[:, :2] - p0[:, :2]
    d_live = to_robot.norm(dim=-1).clamp_min(0.5)
    to_robot = _rot2d(to_robot, env._box_aim_err)
    target = torch.cat(
        [p0[:, :2] + to_robot, (origins_all[:, 2] + env._box_catch_h)[:, None]], dim=-1)
    k = BOX_LIN_DAMPING
    g = torch.tensor([0.0, 0.0, -GRAVITY], device=device)
    v_term = (g / k)[None, :]
    T = torch.sqrt(d_live / 4.0) * env._box_tmult
    expk = torch.exp(-k * T)[:, None]
    v0 = v_term + k * (target - p0 - v_term * T[:, None]) / (1.0 - expk)
    v0 = v0 * env._box_jitter[:, None]
    speed = v0.norm(dim=-1).clamp_min(1e-6)
    cap = LAUNCH_SPEED_RANGE[1] + LOFTY_SPEED_BONUS * (env._box_tmult - 1.0).clamp(min=0.0)
    clamped = torch.minimum(speed.clamp_min(LAUNCH_SPEED_RANGE[0]), cap)
    v0 = v0 * (clamped / speed)[:, None]
    if THROW_LIB and hasattr(env, "_box_lib_mask"):
        # library throws: no re-aim, no clamp -- the fitted real v0 verbatim
        v0 = torch.where(env._box_lib_mask[:, None], env._box_v0, v0)
    # publish the ACTUAL launch state for evaluation (per_throw_eval.py derives
    # the contact-free arrival point from it): read-only, no effect on the sim
    if not hasattr(env, "_box_v0_pub"):
        env._box_v0_pub = torch.zeros_like(v0)
        env._box_p0_pub = torch.zeros_like(p0)
    env._box_v0_pub = torch.where(launching[:, None], v0, env._box_v0_pub)
    env._box_p0_pub = torch.where(launching[:, None], p0, env._box_p0_pub)

    stored_pose = torch.cat([p0, env._box_q0], dim=-1)
    cur_pose = torch.cat([box.data.root_pos_w, box.data.root_quat_w], dim=-1)
    hold_or_launch = (hovering | launching).unsqueeze(-1)
    pose = torch.where(hold_or_launch, stored_pose, cur_pose)

    launch_vel = torch.cat([v0, env._box_omega], dim=-1)
    cur_vel = torch.cat([box.data.root_lin_vel_w, box.data.root_ang_vel_w], dim=-1)
    vel = torch.where(launching.unsqueeze(-1), launch_vel,
                      torch.where(hovering.unsqueeze(-1), torch.zeros_like(cur_vel), cur_vel))

    box.write_root_pose_to_sim(pose)
    box.write_root_velocity_to_sim(vel)
    env._box_launched = env._box_launched | launching

    # ---- auto re-throw: landed (or blown-up) boxes get a fresh throw after a
    # short grace, so every episode contains multiple catch attempts. The
    # single remaining .any() sync is throttled to every 5th step.
    if not hasattr(env, "_box_regrace"):
        env._box_regrace = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    bp = box.data.root_pos_w
    bv = box.data.root_lin_vel_w
    z = bp[:, 2] - env.scene.env_origins[:, 2]
    bspeed = bv.norm(dim=-1)
    bad = env._box_launched & (
        (z < 0.2) | (z > 30.0) | (bspeed > 30.0) | torch.isnan(z) | torch.isnan(bspeed))
    env._box_regrace = torch.where(bad, env._box_regrace + 1, torch.zeros_like(env._box_regrace))
    if RETHROW_ENABLED and int(env.common_step_counter) % 5 == 0:
        rethrow = env._box_regrace >= RETHROW_GRACE_STEPS
        if rethrow.any():
            ids = rethrow.nonzero(as_tuple=False).squeeze(-1)
            reset_box_throw(env, ids)
            # reset_box_throw stores hover_steps relative to episode start; shift to now.
            env._box_launch_step[ids] += env.episode_length_buf[ids]
            env._box_regrace[ids] = 0


# ---- HOLD-INIT curriculum (BOX_HOLD_INIT_FRAC) -------------------------
# Reward shaping alone barely moves a learned clamp hold toward the cradle:
# every deformation from the clamp hold crosses "box falls" territory (a
# grasp-topology basin), so reward signals can't carry the policy to the
# cradle.  Fix = change the STARTING STATE instead: a fraction of episodes
# begin already holding the box on the forearm bed in CRADLE_POSE.
# Stabilizing an in-place hold is a LOCAL problem; normal episodes then learn
# to funnel catches into that attractor.
#
# Two-event design forced by Isaac Lab reset order: reset-mode events run
# BEFORE command_manager.reset, and MotionCommand._resample_command
# teleports the whole robot from the clip — so the reset event only MARKS
# envs; the state write happens on the first post-reset step via an
# interval event registered AFTER box_hover_launch (same-step write order
# matters: hover writes first, ours lands last, next step it sees
# launched=True and no-ops). Throw cancel = env._box_launched[ids]=True.
# Dropped practice boxes hit z<0.2 → the existing rethrow turns the episode
# into a normal catch episode after the grace (graceful degradation).


def hold_init_mark(env, env_ids, frac: float = 0.25,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")):
    """Reset event (register after reset_balloon): mark `frac` of resets as
    hold-practice episodes."""
    if not hasattr(env, "_hold_init_pending"):
        env._hold_init_pending = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device)
    if not torch.is_tensor(env_ids):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    sel = torch.rand(len(env_ids), device=env.device) < frac
    env._hold_init_pending[env_ids] = sel
    # stale BOX_HOLD_INIT_CLAMP flags must not leak into the next episode
    if hasattr(env, "_hold_init_active"):
        env._hold_init_active[env_ids] = False


def hold_init_apply(env, env_ids,
                    x_off: float = 0.32, z_off: float = 0.35,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")):
    """Every-step interval event: on the first step after reset (after the
    MotionCommand teleport), snap marked envs into the ready hold: arms =
    CRADLE_POSE + wrist lock pose, zero arm joint vel, box flat on the
    forearm bed (x_off forward / z_off above root in robot yaw frame), zero
    box velocity, throw cancelled."""
    del env_ids
    if not hasattr(env, "_hold_init_pending") or not hasattr(env, "_box_launched"):
        return
    # BOX_HOLD_INIT_CLAMP: kinematic arm lock for the first N steps of a
    # hold-init episode — after a one-step snap alone the policy tears the
    # cradle pose down within a step; a sustained clamp lets it collect the
    # stable-hold reward before control of the arms is handed back.  Legs
    # stay policy-driven.  Same kinematic-write mechanism as the snap below,
    # arms-only joint set.
    _clamp = int(os.environ.get("BOX_HOLD_INIT_CLAMP", "0"))
    if _clamp > 0 and hasattr(env, "_hold_init_active"):
        c = (env._hold_init_active
             & (env.episode_length_buf > 1)
             & (env.episode_length_buf <= _clamp))
        if bool(c.any()):
            cids = c.nonzero(as_tuple=False).squeeze(-1)
            robot_c = env.scene["robot"]
            d_c = _get_posture_ids(env)
            if "hold_init_ids" in d_c:
                jp_c = robot_c.data.joint_pos[cids].clone()
                jv_c = robot_c.data.joint_vel[cids].clone()
                jp_c[:, d_c["hold_init_ids"]] = d_c["hold_init_target"][None, :]
                jv_c[:, d_c["hold_init_ids"]] = 0.0
                robot_c.write_joint_state_to_sim(jp_c, jv_c, env_ids=cids)
    m = env._hold_init_pending & (env.episode_length_buf <= 1)
    if not bool(m.any()):
        return
    ids = m.nonzero(as_tuple=False).squeeze(-1)
    robot = env.scene["robot"]
    box: RigidObject = env.scene[asset_cfg.name]
    device = robot.device
    d = _get_posture_ids(env)
    if "hold_init_ids" not in d:
        jn = robot.joint_names
        targets = dict(CRADLE_POSE)
        _lip = float(os.environ.get("BOX_WRIST_LIP", "0.0"))
        targets.update({
            "left_wrist_roll_joint": -1.57, "right_wrist_roll_joint": 1.57,
            "left_wrist_pitch_joint": -_lip, "right_wrist_pitch_joint": -_lip,
            "left_wrist_yaw_joint": 0.0, "right_wrist_yaw_joint": 0.0,
        })
        if os.environ.get("BOX_HOLD_POSE_CURL", "0") == "1":
            _he = float(os.environ.get("BOX_HOLD_ELBOW", "0.95"))
            for _k in list(targets):
                if "_elbow_" in _k:
                    targets[_k] = _he
        d["hold_init_ids"] = torch.tensor(
            [jn.index(k) for k in targets], dtype=torch.long, device=device)
        d["hold_init_target"] = torch.tensor(
            list(targets.values()), device=device)
    jp = robot.data.joint_pos[ids].clone()
    jv = robot.data.joint_vel[ids].clone()
    _tgt = d["hold_init_target"][None, :].expand(len(ids), -1)
    _rsi = float(os.environ.get("BOX_HOLD_INIT_RSI", "0"))
    if _rsi > 0:  # reference-state randomization of the hold pose
        _tgt = _tgt + (torch.rand_like(_tgt) * 2 - 1) * _rsi
    jp[:, d["hold_init_ids"]] = _tgt
    jv[:, d["hold_init_ids"]] = 0.0
    robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
    q = robot.data.root_quat_w[ids]
    qw, qx, qy, qz = q.unbind(-1)
    fwd = torch.stack([1 - 2 * (qy * qy + qz * qz),
                       2 * (qx * qy + qw * qz),
                       torch.zeros_like(qw)], dim=-1)
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pos = robot.data.root_pos_w[ids] + fwd * x_off
    pos[:, 2] = robot.data.root_pos_w[ids, 2] + z_off
    if float(os.environ.get("BOX_HOLD_INIT_RSI", "0")) > 0:
        pos = pos + (torch.rand_like(pos) * 2 - 1) * 0.025
    quat = torch.zeros((len(ids), 4), device=device)
    quat[:, 0] = 1.0
    box.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
    box.write_root_velocity_to_sim(
        torch.zeros((len(ids), 6), device=device), env_ids=ids)
    env._box_launched[ids] = True     # cancel this episode's throw
    env._hold_init_pending[ids] = False
    # arm the sustained clamp window (BOX_HOLD_INIT_CLAMP) for these envs
    if int(os.environ.get("BOX_HOLD_INIT_CLAMP", "0")) > 0:
        if not hasattr(env, "_hold_init_active"):
            env._hold_init_active = torch.zeros(
                env.num_envs, dtype=torch.bool, device=device)
        env._hold_init_active[ids] = True


def arm_snap_apply(env, env_ids,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")):
    """BOX_ARM_SNAP: the MotionCommand reset teleports ALL joints to the clip
    pose + ±0.1 rad noise AFTER reset events run, so an init_state cradle
    (BOX_ARM_INIT_CRADLE) never survives the teleport (arms rotated, palms
    facing the wrong way).  Every-step interval event: on the first
    post-reset step pin the arm joints (CRADLE_POSE + wrist lock pose) and
    zero their velocity — every episode starts palms-up BY CONSTRUCTION.
    Same first-step-write pattern as hold_init_apply."""
    del env_ids
    m = env.episode_length_buf <= 1
    if not bool(m.any()):
        return
    ids = m.nonzero(as_tuple=False).squeeze(-1)
    robot = env.scene["robot"]
    device = robot.device
    d = _get_posture_ids(env)
    if "arm_snap_ids" not in d:
        jn = robot.joint_names
        targets = dict(CRADLE_POSE)
        _lip = float(os.environ.get("BOX_WRIST_LIP", "0.0"))
        targets.update({
            "left_wrist_roll_joint": -1.57, "right_wrist_roll_joint": 1.57,
            "left_wrist_pitch_joint": -_lip, "right_wrist_pitch_joint": -_lip,
            "left_wrist_yaw_joint": 0.0, "right_wrist_yaw_joint": 0.0,
        })
        d["arm_snap_ids"] = torch.tensor(
            [jn.index(k) for k in targets], dtype=torch.long, device=device)
        d["arm_snap_target"] = torch.tensor(
            list(targets.values()), device=device)
    jp = robot.data.joint_pos[ids].clone()
    jv = robot.data.joint_vel[ids].clone()
    jp[:, d["arm_snap_ids"]] = d["arm_snap_target"][None, :]
    jv[:, d["arm_snap_ids"]] = 0.0
    robot.write_joint_state_to_sim(jp, jv, env_ids=ids)


# ---- switch self-announcement: every enabled env-var feature prints a line ----
if os.environ.get("BOX_CRADLE_RAMP", "0") == "1":
    print("[throw] CRADLE_RAMP ON: cradle_pose gaussian -> full-range linear ramp",
          flush=True)
if os.environ.get("BOX_READY_DEV_ALLPHASE", "0") == "1":
    print("[throw] READY_DEV_ALLPHASE ON: dev penalty in ALL phases incl held",
          flush=True)
if float(os.environ.get("BOX_HOLD_INIT_FRAC", "0")) > 0:
    print(f"[throw] HOLD_INIT ON: frac={os.environ['BOX_HOLD_INIT_FRAC']} "
          "(mark at reset, apply on first step, throw cancelled)", flush=True)
if int(os.environ.get("BOX_HOLD_INIT_CLAMP", "0")) > 0:
    print(f"[throw] HOLD_INIT_CLAMP ON: kinematic arm lock for first "
          f"{os.environ['BOX_HOLD_INIT_CLAMP']} steps of hold-init episodes",
          flush=True)


# ---- three-point carry + stance terms.  All positive rewards (no penalties,
# so they never create an incentive to terminate early). ----

def chest_delivery_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    x_anchor: float = 0.13, z_anchor: float = 0.32, sigma: float = 0.12,
) -> torch.Tensor:
    """Once the box is ON the bed (both forearms supporting), pay for
    scooping it against the chest — two forearms + chest = a three-point
    carry that stays stable while walking.  Gate = box_on_forearm_bed score,
    so the reward only exists post-catch (catching stays strictly upstream).
    Shaping = gaussian of box-center distance to a chest anchor in the
    root-yaw frame (anchor deliberately CLOSER than the bed rest x≈0.26 —
    chasing it produces the elbow curl and a slight lean by itself; the torso
    gate still bounds the recline)."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    bed = box_on_forearm_bed_reward(env, asset_cfg)
    q = robot.data.root_quat_w
    w, x, y, z = q.unbind(-1)
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    n = (fx * fx + fy * fy).sqrt().clamp_min(1e-6)
    fx, fy = fx / n, fy / n
    rel = box.data.root_pos_w - robot.data.root_pos_w
    fwd_d = rel[:, 0] * fx + rel[:, 1] * fy
    lat_d = -rel[:, 0] * fy + rel[:, 1] * fx
    d2 = (fwd_d - x_anchor) ** 2 + lat_d ** 2 + (rel[:, 2] - z_anchor) ** 2
    r = bed * torch.exp(-d2 / (2 * sigma * sigma))
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def hold_feet_gather_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    split_ref: float = 0.35, width: float = 0.25,
) -> torch.Tensor:
    """BONUS for gathering the feet once the hold has SETTLED
    (>= BOX_FEET_GATHER_SETTLE consecutive bed-support steps; 30 steps =
    0.6 s @ 50 Hz). Lunging into the catch stays free — the term is zero
    until the box has been resting on the bed for the settle window, and it
    is positive-only so catching never becomes unprofitable."""
    robot = env.scene["robot"]
    bed = box_on_forearm_bed_reward(env, asset_cfg)
    settle = int(os.environ.get("BOX_FEET_GATHER_SETTLE", "30"))
    if not hasattr(env, "_bed_hold_steps"):
        env._bed_hold_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=robot.device)
    held = bed > float(os.environ.get("BOX_FEET_GATHER_BED_TH", "0.5"))
    env._bed_hold_steps = torch.where(
        held, env._bed_hold_steps + 1,
        torch.zeros_like(env._bed_hold_steps))
    d = _get_posture_ids(env)
    if "la_gather" not in d:
        bnames = robot.body_names
        d["la_gather"] = bnames.index("left_ankle_roll_link")
        d["ra_gather"] = bnames.index("right_ankle_roll_link")
    bp = robot.data.body_pos_w
    split = (bp[:, d["la_gather"], :2] - bp[:, d["ra_gather"], :2]).norm(dim=-1)
    if os.environ.get("BOX_FEET_GATHER_RAMP", "0") == "1":
        # BOX_FEET_GATHER_RAMP=1: the gaussian pays exp(-11.6)~0 at a typical
        # lunge split of 1.2 m, so the policy never experiences the reward.
        # The linear ramp pays gradient from split 1.3 all the way down to
        # split_ref.
        gather = ((1.3 - split) / (1.3 - split_ref)).clamp(0.0, 1.0)
    else:
        gather = torch.exp(-((split - split_ref).clamp_min(0.0) / width) ** 2)
    bed_th = float(os.environ.get("BOX_FEET_GATHER_BED_TH", "0.5"))
    held2 = bed > bed_th
    r = (env._bed_hold_steps >= settle).float() * held2.float() * gather
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _box_chest_gap(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
                   torso_surf_x: float = 0.10) -> torch.Tensor:
    """Geometric box-face-to-chest gap (same math as chest_contact_reward).
    Negative = pressing in.  Separate so the 3POINT gates can test CONTACT
    directly — the tilt-into factor inside chest_contact_reward is a style
    term, not part of the success definition (a box lying flat against the
    chest is a valid three-point hold)."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    w, x, y, z = q.unbind(-1)
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    n = (fx * fx + fy * fy).sqrt().clamp_min(1e-6)
    fx, fy = fx / n, fy / n
    rel = box.data.root_pos_w - robot.data.root_pos_w
    fwd_d = rel[:, 0] * fx + rel[:, 1] * fy
    bq = box.data.root_quat_w
    bw, bx, by, bz = bq.unbind(-1)
    r00 = 1 - 2 * (by * by + bz * bz); r10 = 2 * (bx * by + bw * bz)
    r01 = 2 * (bx * by - bw * bz);     r11 = 1 - 2 * (bx * bx + bz * bz)
    r02 = 2 * (bx * bz + bw * by);     r12 = 2 * (by * bz - bw * bx)
    axf_x = r00 * fx + r10 * fy
    axf_y = r01 * fx + r11 * fy
    axf_z = r02 * fx + r12 * fy
    he = env._box_half_extents if hasattr(env, "_box_half_extents") else \
        torch.stack([torch.full_like(fwd_d, 0.17), torch.full_like(fwd_d, 0.13),
                     torch.full_like(fwd_d, 0.11)], dim=-1)
    he_fwd = axf_x.abs() * he[:, 0] + axf_y.abs() * he[:, 1] + axf_z.abs() * he[:, 2]
    return fwd_d - he_fwd - torso_surf_x


def chest_contact_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon"),
    torso_surf_x: float = 0.10, contact_tol: float = 0.005, decay: float = 0.02,
    z_lo: float = 0.15, z_hi: float = 0.50,
) -> torch.Tensor:
    """The chest-delivery gaussian saturates centimeters short of contact —
    the last-cm marginal reward is too thin to beat posture penalties.  This
    term pays ONLY
    for actual chest contact: geometric gap = (box-center forward dist in
    root-yaw frame − box half-extent projected on forward) − torso surface
    offset; full pay at gap<=5mm, e-folding 2cm. Gated by bed support AND
    box at chest height.  BOX_CHEST_GAP_PROBE=1 prints gap statistics."""
    box: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    bed = box_on_forearm_bed_reward(env, asset_cfg)
    q = robot.data.root_quat_w
    w, x, y, z = q.unbind(-1)
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    n = (fx * fx + fy * fy).sqrt().clamp_min(1e-6)
    fx, fy = fx / n, fy / n
    rel = box.data.root_pos_w - robot.data.root_pos_w
    fwd_d = rel[:, 0] * fx + rel[:, 1] * fy
    # box half-extent along robot-forward: |R^T f|·he (per-env extents if set)
    bq = box.data.root_quat_w
    bw, bx, by, bz = bq.unbind(-1)
    r00 = 1 - 2 * (by * by + bz * bz); r10 = 2 * (bx * by + bw * bz)
    r01 = 2 * (bx * by - bw * bz);     r11 = 1 - 2 * (bx * bx + bz * bz)
    r02 = 2 * (bx * bz + bw * by);     r12 = 2 * (by * bz - bw * bx)
    axf_x = r00 * fx + r10 * fy
    axf_y = r01 * fx + r11 * fy
    axf_z = r02 * fx + r12 * fy
    he = env._box_half_extents if hasattr(env, "_box_half_extents") else \
        torch.stack([torch.full_like(fwd_d, 0.17), torch.full_like(fwd_d, 0.13),
                     torch.full_like(fwd_d, 0.11)], dim=-1)
    he_fwd = axf_x.abs() * he[:, 0] + axf_y.abs() * he[:, 1] + axf_z.abs() * he[:, 2]
    gap = fwd_d - he_fwd - torso_surf_x
    z_ok = ((rel[:, 2] > z_lo) & (rel[:, 2] < z_hi)).float()
    contact = torch.exp(-(gap - contact_tol).clamp_min(0.0) / decay)
    # tilt-into factor: an edge graze can pass the gap test while the box
    # face leans AWAY — a human carry has the box top leaning INTO the chest.
    # u = box top axis in world; full credit when u tilts >=0.20 toward the
    # robot, zero when leaning away >0.05.
    u_x = 2 * (bx * bz + bw * by)
    u_y = 2 * (by * bz - bw * bx)
    u_dot_f = u_x * fx + u_y * fy
    tilt_in = ((-u_dot_f + 0.05) / 0.25).clamp(0.0, 1.0)
    r = bed * z_ok * contact * (0.3 + 0.7 * tilt_in)
    if os.environ.get("BOX_CHEST_GAP_PROBE", "0") == "1":
        if not hasattr(env, "_gap_probe_n"):
            env._gap_probe_n = 0
        env._gap_probe_n += 1
        if env._gap_probe_n % 100 == 0:
            m = bed > 0.5
            if bool(m.any()):
                g = gap[m]
                print(f"[chest_gap] n={int(m.sum())} gap mean {g.mean():.3f} "
                      f"p10 {g.quantile(0.1):.3f} p90 {g.quantile(0.9):.3f} "
                      f"contact<1cm {(g < 0.01).float().mean():.2f}", flush=True)
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


if float(os.environ.get("BOX_CHEST_DELIVERY_W", "0")) > 0:
    print(f"[throw] CHEST_DELIVERY ON: w={os.environ['BOX_CHEST_DELIVERY_W']} "
          "(bed-gated scoop-to-chest reward, 3-point carry)", flush=True)
if float(os.environ.get("BOX_CHEST_CONTACT_W", "0")) > 0:
    print(f"[throw] CHEST_CONTACT ON: w={os.environ['BOX_CHEST_CONTACT_W']} "
          "(pays ONLY at real chest contact, gap<=5mm full)", flush=True)
if float(os.environ.get("BOX_FEET_GATHER_W", "0")) > 0:
    print(f"[throw] FEET_GATHER ON: w={os.environ['BOX_FEET_GATHER_W']} "
          f"settle={os.environ.get('BOX_FEET_GATHER_SETTLE', '30')} steps "
          "(positive-only, post-settle stance gather)", flush=True)

if float(os.environ.get("BOX_HOLD_INIT_RSI", "0")) > 0:
    print(f"[throw] HOLD_INIT_RSI ON: +-{os.environ['BOX_HOLD_INIT_RSI']} rad arm, +-2.5cm box", flush=True)
if os.environ.get("BOX_HOVER_LO", "0.5") != "0.5" or os.environ.get("BOX_HOVER_HI", "0.75") != "0.75":
    print(f"[throw] HOVER RANGE NON-DEFAULT: {HOVER_RANGE_S}", flush=True)
if float(os.environ.get("BOX_BEARING_DEG", "45")) != 45.0 or \
        float(os.environ.get("BOX_AIM_MAX_DEG", "15")) != 15.0:
    print(f"[throw] FRONTAL FAN: bearing +-{BEARING_RANGE_DEG} deg, "
          f"aim std {AIM_ERR_STD_DEG} max {AIM_ERR_MAX_DEG} deg", flush=True)
if RELEASE_LEVEL or SPIN_MAX_RAD_S != 1.5 * 2.0 * math.pi:
    print(f"[throw] RELEASE NON-DEFAULT: level={int(RELEASE_LEVEL)} spin_max={SPIN_MAX_RAD_S/(2*math.pi):.2f} rev/s "
          f"(default level=0 spin_max=1.50)", flush=True)
if LAUNCH_SPEED_RANGE != (5.0, 7.6):
    print(f"[throw] SPEED RANGE NON-DEFAULT: {LAUNCH_SPEED_RANGE} m/s (default (5.0, 7.6))", flush=True)
if not RETHROW_ENABLED:
    print("[throw] RETHROW OFF: one throw per env per episode (BOX_RETHROW=0)", flush=True)
if CATCH_H_RANGE != (0.6, 1.4):
    print(f"[throw] CATCH_H NON-DEFAULT: aim height U{CATCH_H_RANGE} m at the pelvis xy "
          f"(default (0.6, 1.4))", flush=True)
if os.environ.get("BOX_3POINT", "0") == "1":
    print("[throw] 3POINT ON: sustained gated by chest-contact x box-still "
          "x feet-planted; threepoint_hold_bonus armed", flush=True)
if os.environ.get("BOX_ARM_SNAP", "0") == "1":
    print("[throw] ARM_SNAP ON: arms pinned to cradle+lock pose on first "
          "post-reset step (all envs)", flush=True)


def first_throw_realign(env, env_ids,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("balloon")):
    """BOX_THROW_REALIGN: the episode's FIRST throw is sampled by
    reset_box_throw in the event phase — BEFORE MotionCommand teleports the
    robot to a random clip frame with an arbitrary heading — so the thrower
    is placed relative to the PREVIOUS episode's facing (a wide ±45° bearing
    hides this; a narrow frontal fan exposes it as side throws).  Same
    first-step-write remedy as arm_snap/hold_init: on the first post-reset
    step, re-sample the throw from the robot's NOW-correct facing
    (reset_box_throw is idempotent).
    Hold-init envs are already _box_launched=True by the time this runs
    (registration order) and are skipped."""
    del env_ids
    if not hasattr(env, "_box_launched"):
        return
    m = (env.episode_length_buf <= 1) & (~env._box_launched)
    if not bool(m.any()):
        return
    ids = m.nonzero(as_tuple=False).squeeze(-1)
    reset_box_throw(env, ids)


if os.environ.get("BOX_THROW_REALIGN", "0") == "1":
    print("[throw] THROW_REALIGN ON: first throw re-sampled post-teleport "
          "(fixes stale-facing side throws)", flush=True)

if os.environ.get("BOX_HOLD_POSE_CURL", "0") == "1":
    print("[box_v24d] HOLD_POSE_CURL ON: held anchor = ready pose with elbows " + os.environ.get("BOX_HOLD_ELBOW", "0.95"), flush=True)

if os.environ.get("BOX_3POINT_CHEST", "reward") == "gap":
    print("[throw] 3POINT chest gate = geometric gap<2cm (tilt-into stays as style reward only)", flush=True)
