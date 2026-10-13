# SPDX-License-Identifier: MIT
"""Rebuild the catch-policy reference motion ``walk_then_catch_v17_30s.npz``.

The reference motion is 6 s of walking from the BONES-SEED clip
``210707/walk_forward_grab_normal_003__A006_M`` (licensed; you must download
it yourself), a 1 s linear blend into our own catch pose, a hold of that pose,
and a hold extension to 30 s. We do not redistribute any BONES-SEED data;
this script regenerates the file from your copy of the clip.

Inputs
  --bones_csv   BONES-SEED G1 retarget CSV of the clip above
                (``g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv``)
  --catch_pose  our catch pose (shipped: assets/motions/catch_pose_g1_dof.csv)

Outputs (in --out_dir)
  walk_then_catch_v17_30s.npz   50 fps, 1500 frames  (used by training/eval)
  walk_then_catch_v17.npz       50 fps,  649 frames  (only with --save_13s)

Needs Isaac Lab and the ``whole_body_tracking`` package (for the G1 model):

  python tools/motion/build_reference_motion.py --headless \\
      --bones_csv /path/to/bones_seed/g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv \\
      --out_dir assets/motions/generated

Use --verify <dir> to compare the result key by key against existing files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import motion_common as mc  # noqa: E402

BONES_CLIP_SHA256 = "7979e92da40a2241341d02609a37381b92fe00af3f755b20453a65dc3339a8af"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(name: str, new: dict, ref_path: str) -> float:
    ref = mc.load_npz(ref_path)
    worst = 0.0
    print(f"[verify] {name} vs {ref_path}")
    for k in mc.NPZ_KEYS:
        if new[k].shape != ref[k].shape:
            print(f"    {k:15s} SHAPE MISMATCH {new[k].shape} vs {ref[k].shape}")
            worst = float("inf")
            continue
        d = float(np.abs(new[k].astype(np.float64) - ref[k].astype(np.float64)).max())
        worst = max(worst, d)
        print(f"    {k:15s} max|diff| = {d:.3e}  {'(bit-identical)' if np.array_equal(new[k], ref[k]) else ''}")
    print(f"    fps            {int(new['fps'][0])} vs {int(ref['fps'][0])}")
    return worst


def build_track(bones_csv: str, catch_pose: str, csv_out: str | None = None) -> np.ndarray:
    """NumPy stage: BONES CSV + catch pose -> (390,36) 30 fps trajectory (after the
    "%.6f" CSV round trip used by the original pipeline)."""
    walk = mc.load_bones_g1_csv(bones_csv)
    track = mc.build_walk_then_catch(walk, mc.load_catch_pose(catch_pose))
    return mc.round_trip_csv(mc.motion_to_bm_array(track), "%.6f", path=csv_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bones_csv", required=True, help="BONES-SEED G1 CSV of 210707/walk_forward_grab_normal_003__A006_M")
    parser.add_argument("--catch_pose", default=os.path.join(REPO_ROOT, "assets", "motions", "catch_pose_g1_dof.csv"))
    parser.add_argument("--out_dir", default=os.path.join(os.environ.get("REFLEX_DATA_DIR", os.path.join(REPO_ROOT, "data")), "motions"))
    parser.add_argument("--save_13s", action="store_true", help="also write the 13 s walk_then_catch_v17.npz")
    parser.add_argument("--save_csv", action="store_true", help="also write the intermediate 30 fps CSV")
    parser.add_argument("--robot_cfg", default="whole_body_tracking.robots.g1:G1_CYLINDER_CFG",
                        help="module:attribute of the Isaac Lab ArticulationCfg used for the replay")
    parser.add_argument("--verify", default=None, help="directory with reference npz files to compare against")
    parser.add_argument("--skip_checksum", action="store_true", help="do not abort if the CSV checksum differs")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # ---- NumPy stage (fails fast before Isaac boots) ------------------------
    digest = sha256(args.bones_csv)
    if digest != BONES_CLIP_SHA256:
        msg = f"BONES CSV sha256 {digest} != expected {BONES_CLIP_SHA256}"
        if not args.skip_checksum:
            sys.exit(msg + " (wrong file? use --skip_checksum to continue anyway)")
        print("WARNING:", msg)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_out = os.path.join(args.out_dir, "walk_then_catch_v17_30fps.csv") if args.save_csv else None
    traj = build_track(args.bones_csv, args.catch_pose, csv_out)
    print(f"[build] 30 fps trajectory: {traj.shape}")

    # ---- Isaac stage: 30 -> 50 fps + forward kinematics ---------------------
    app = AppLauncher(args).app
    import fk_replay as fk

    sim, scene = fk.make_scene(fk.load_robot_cfg(args.robot_cfg), mc.OUTPUT_FPS, args.device)
    motion13 = fk.replay_motion(sim, scene, traj, mc.BUILD_FPS, mc.OUTPUT_FPS, 0.0)
    motion30 = mc.hold_extend(motion13, mc.REF_TOTAL_FRAMES_50)

    out30 = os.path.join(args.out_dir, "walk_then_catch_v17_30s.npz")
    mc.save_npz(out30, motion30)
    print(f"[build] wrote {out30}  ({motion30['joint_pos'].shape[0]} frames @ {mc.OUTPUT_FPS} fps)")
    if args.save_13s:
        out13 = os.path.join(args.out_dir, "walk_then_catch_v17.npz")
        mc.save_npz(out13, motion13)
        print(f"[build] wrote {out13}  ({motion13['joint_pos'].shape[0]} frames)")

    if args.verify:
        compare("walk_then_catch_v17_30s", motion30, os.path.join(args.verify, "walk_then_catch_v17_30s.npz"))
        p13 = os.path.join(args.verify, "walk_then_catch_v17.npz")
        if os.path.exists(p13):
            compare("walk_then_catch_v17", motion13, p13)
    # Files are written; skip app.close(), which can hang in some Isaac Sim containers.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
