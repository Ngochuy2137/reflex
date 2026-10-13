# SPDX-License-Identifier: MIT
"""Rebuild one AMASS-derived walking clip used by the AMP style dataset.

Source: "Retargeted AMASS for Robotics" (Hugging Face dataset
``fleaven/Retargeted_AMASS_for_robotics``), Unitree G1 retarget of
AMASS / ACCAD ``Male2Walking_c3d/B10 - Walk turn left 45`` (120 fps).
We do not redistribute it; download it yourself and run this script twice:

  # male2_walk_turn_left_45.npz  (root kept at the source height, z_offset 0)
  python tools/motion/build_amass_motion.py --headless --z_offset 0.0 \\
      --amass_npy "/path/g1/ACCAD/Male2Walking_c3d/B10 -  Walk turn left 45_poses_120_jpos.npy" \\
      --out assets/motions/generated/male2_walk_turn_left_45.npz

  # male2_walk_v2.npz  (root lifted to standing height, z_offset 0.76)
  python tools/motion/build_amass_motion.py --headless --z_offset 0.76 \\
      --amass_npy "/path/g1/ACCAD/Male2Walking_c3d/B10 -  Walk turn left 45_poses_120_jpos.npy" \\
      --out assets/motions/generated/male2_walk_v2.npz

Run one clip per process: the PhysX velocity read-back is only bit-identical
to the original files when the replay is the first one in a fresh app.

Pipeline (identical to the original): (T,36) array -> CSV text with "%.6f"
-> Isaac Lab kinematic replay, 120 -> 50 fps (fk_replay.py).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import motion_common as mc  # noqa: E402

AMASS_B10_SHA256 = "f7be749b1f59500c48a4c1421c00f6fefa36052b154b97dcdb70658ce031404f"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amass_npy", required=True, help="G1 retarget .npy (T,36: pos, quat xyzw, 29 joints)")
    parser.add_argument("--input_fps", type=int, default=120)
    parser.add_argument("--z_offset", type=float, required=True, help="0.0 for *_turn_left_45, 0.76 for *_walk_v2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--robot_cfg", default="whole_body_tracking.robots.g1:G1_CYLINDER_CFG")
    parser.add_argument("--verify", default=None, help="existing npz to compare against")
    parser.add_argument("--skip_checksum", action="store_true")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    with open(args.amass_npy, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != AMASS_B10_SHA256:
        msg = f"AMASS npy sha256 {digest} != expected {AMASS_B10_SHA256} (B10 - Walk turn left 45)"
        if not args.skip_checksum:
            sys.exit(msg + " (use --skip_checksum to continue anyway)")
        print("WARNING:", msg)
    traj = mc.round_trip_csv(np.load(args.amass_npy), "%.6f")

    app = AppLauncher(args).app
    import fk_replay as fk

    sim, scene = fk.make_scene(fk.load_robot_cfg(args.robot_cfg), mc.OUTPUT_FPS, args.device)
    motion = fk.replay_motion(sim, scene, traj, args.input_fps, mc.OUTPUT_FPS, args.z_offset)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mc.save_npz(args.out, motion)
    print(f"[build] wrote {args.out}  ({motion['joint_pos'].shape[0]} frames @ {mc.OUTPUT_FPS} fps)")
    if args.verify:
        ref = mc.load_npz(args.verify)
        for k in mc.NPZ_KEYS:
            d = float(np.abs(motion[k].astype(np.float64) - ref[k].astype(np.float64)).max())
            print(f"[verify] {k:15s} max|diff| = {d:.3e} {'(bit-identical)' if np.array_equal(motion[k], ref[k]) else ''}")
    # Files are written; skip app.close(), which can hang in some Isaac Sim containers.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
