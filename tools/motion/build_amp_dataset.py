# SPDX-License-Identifier: MIT
"""Rebuild the AMP discriminator dataset ``amp_dataset.pt``.

No Isaac Sim needed (NumPy + PyTorch only). Inputs are the motions rebuilt by
build_reference_motion.py / build_amass_motion.py plus our own capture:

  --reference_motion   walk_then_catch_v17_30s.npz        (build_reference_motion.py)
  --male2_walk_v2      male2_walk_v2.npz                  (build_amass_motion.py, z_offset 0.76)
  --male2_turn_left_45 male2_walk_turn_left_45.npz        (build_amass_motion.py, z_offset 0.0)
  --ready_pose         ready_pose_capture.npy             (shipped, own data)

Two further clips are cut from the reference motion (they are not separate
data sources):
  walk_only.npz  = reference frames [0, 300) + 50-frame hold of frame 299
  catch_only.npz = reference frames [300, 649) + 151-frame hold of frame 648
(holds keep the pose and zero all velocities). Use --save_motions to write them.

AMP features per frame (83): joint_pos 29 + joint_vel 29 + root height 1 +
heading-free root orientation 6D 6 + root lin/ang velocity in the heading
frame 3+3 + wrist/ankle positions in the heading frame 12.

  python tools/motion/build_amp_dataset.py \\
      --reference_motion assets/motions/generated/walk_then_catch_v17_30s.npz \\
      --male2_walk_v2 assets/motions/generated/male2_walk_v2.npz \\
      --male2_turn_left_45 assets/motions/generated/male2_walk_turn_left_45.npz \\
      --out assets/motions/generated/amp_dataset.pt
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import motion_common as mc  # noqa: E402

# Clip order matters (it fixes the row order of the dataset).
CLIPS = [
    "walk_only.npz",
    "male2_walk_v2.npz",
    "male2_walk_turn_left_45.npz",
    "catch_only.npz",
    "walk_then_catch_v17_30s.npz",
]
# Body indices in the 30-body arrays of the npz files (Isaac Lab G1 order).
KEY_BODY_IDX = [28, 29, 18, 19]  # left_wrist_yaw, right_wrist_yaw, left_ankle_roll, right_ankle_roll
PELVIS_IDX = 0
# Extra copies of the catch / hold section (pair-index ranges per clip).
CATCH_WINDOWS = {
    "walk_then_catch_v17_30s.npz": (520, 700),
    "catch_only.npz": (0, 499),
}
UPSAMPLE = 4

WALK_ONLY = (0, 300, 350)     # (start, stop, total frames) cut from the reference
CATCH_ONLY = (300, 649, 500)


def heading_quat_inv(q):
    """Yaw-only inverse quaternion (w,x,y,z) of root quats (N,4)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = np.zeros_like(q)
    out[:, 0] = np.cos(-yaw / 2)
    out[:, 3] = np.sin(-yaw / 2)
    return out


def quat_rotate(q, v):
    """Rotate vectors v (N,3) by quats q (N,4) wxyz."""
    w = q[:, :1]
    u = q[:, 1:]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2 * (w * uv + uuv)


def quat_mul(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def rot6d_from_quat(q):
    """First two columns of the rotation matrix, (N,6)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    c0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
    c1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)
    return np.concatenate([c0, c1], axis=1)


def clip_features(d):
    jp, jv = d["joint_pos"], d["joint_vel"]
    bp, bq = d["body_pos_w"], d["body_quat_w"]
    blv, bav = d["body_lin_vel_w"], d["body_ang_vel_w"]
    root_p = bp[:, PELVIS_IDX]
    root_q = bq[:, PELVIS_IDX]
    hinv = heading_quat_inv(root_q)
    root_h = root_p[:, 2:3]
    ori6d = rot6d_from_quat(quat_mul(hinv, root_q))
    lv = quat_rotate(hinv, blv[:, PELVIS_IDX])
    av = quat_rotate(hinv, bav[:, PELVIS_IDX])
    keys = np.concatenate([quat_rotate(hinv, bp[:, bi] - root_p) for bi in KEY_BODY_IDX], axis=1)
    return np.concatenate([jp, jv, root_h, ori6d, lv, av, keys], axis=1).astype(np.float32)  # (T,83)


def build(motions: dict, ready_pose: np.ndarray | None):
    all_pairs, names = [], []
    for c in CLIPS:
        f = clip_features(motions[c])
        pairs = np.stack([f[:-1], f[1:]], axis=1)  # (T-1, 2, 83)
        all_pairs.append(pairs)
        names += [c] * len(pairs)
        print(f"{c}: {len(pairs)} pairs")
        if c in CATCH_WINDOWS:
            a, b = CATCH_WINDOWS[c]
            b = min(b, len(pairs))
            win = pairs[a:b]
            for _ in range(UPSAMPLE - 1):
                all_pairs.append(win)
                names += [c + ":catch_up"] * len(win)
            print(f"  upsampled catch window [{a}:{b}] x{UPSAMPLE}")
    if ready_pose is not None:
        pairs = np.stack([ready_pose[:-1], ready_pose[1:]], axis=1)
        for _ in range(UPSAMPLE):
            all_pairs.append(pairs)
            names += ["ready_pose_capture"] * len(pairs)
        print(f"ready_pose_capture: {len(pairs)} pairs x{UPSAMPLE}")
    feats = np.concatenate(all_pairs, axis=0)
    flat = feats.reshape(-1, feats.shape[-1])
    stats = {"mean": torch.tensor(flat.mean(0)), "std": torch.tensor(flat.std(0) + 1e-5)}
    return {"features": torch.tensor(feats), "clip_names": names, "stats": stats}


def verify(new: dict, ref_path: str) -> None:
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    print(f"[verify] vs {ref_path}")
    print(f"    keys           {sorted(new)} vs {sorted(ref)}")
    a, b = new["features"], ref["features"]
    if a.shape != b.shape:
        print(f"    features       SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
    else:
        print(f"    features       {tuple(a.shape)} max|diff| = {float((a.double() - b.double()).abs().max()):.3e}"
              f" {'(bit-identical)' if torch.equal(a, b) else ''}")
        # per clip block
        names = np.asarray(new["clip_names"])
        for c in dict.fromkeys(new["clip_names"]):
            m = torch.from_numpy(names == c)
            d = float((a[m].double() - b[m].double()).abs().max())
            print(f"      {c:40s} {int(m.sum()):5d} pairs  max|diff| = {d:.3e}")
    print(f"    clip_names     identical = {new['clip_names'] == ref['clip_names']}")
    for k in ("mean", "std"):
        d = float((new["stats"][k].double() - ref["stats"][k].double()).abs().max())
        print(f"    stats.{k:8s} max|diff| = {d:.3e} {'(bit-identical)' if torch.equal(new['stats'][k], ref['stats'][k]) else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference_motion", required=True, help="walk_then_catch_v17_30s.npz")
    parser.add_argument("--male2_walk_v2", required=True)
    parser.add_argument("--male2_turn_left_45", required=True)
    parser.add_argument("--ready_pose", default=os.path.join(REPO_ROOT, "assets", "motions", "ready_pose_capture.npy"))
    parser.add_argument("--out", default=os.path.join(os.environ.get("REFLEX_DATA_DIR", os.path.join(REPO_ROOT, "data")), "amp", "amp_dataset.pt"))
    parser.add_argument("--save_motions", default=None, help="directory to also write walk_only.npz / catch_only.npz")
    parser.add_argument("--verify", default=None, help="existing amp_dataset.pt to compare against")
    args = parser.parse_args()

    ref = mc.load_npz(args.reference_motion)
    if ref["joint_pos"].shape[0] != mc.REF_TOTAL_FRAMES_50:
        sys.exit(f"reference motion has {ref['joint_pos'].shape[0]} frames, expected {mc.REF_TOTAL_FRAMES_50}")
    motions = {
        "walk_only.npz": mc.segment_then_hold(ref, *WALK_ONLY),
        "male2_walk_v2.npz": mc.load_npz(args.male2_walk_v2),
        "male2_walk_turn_left_45.npz": mc.load_npz(args.male2_turn_left_45),
        "catch_only.npz": mc.segment_then_hold(ref, *CATCH_ONLY),
        "walk_then_catch_v17_30s.npz": ref,
    }
    if args.save_motions:
        os.makedirs(args.save_motions, exist_ok=True)
        for c in ("walk_only.npz", "catch_only.npz"):
            mc.save_npz(os.path.join(args.save_motions, c), motions[c])
            print(f"wrote {os.path.join(args.save_motions, c)}")
    ready = np.load(args.ready_pose).astype(np.float32) if args.ready_pose else None
    data = build(motions, ready)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(data, args.out)
    print(f"saved {args.out}: {tuple(data['features'].shape)}")
    if args.verify:
        verify(data, args.verify)


if __name__ == "__main__":
    main()
