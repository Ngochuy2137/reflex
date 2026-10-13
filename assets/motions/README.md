# Motion assets

This folder ships **only data we created ourselves**. Files derived from
third-party motion datasets (BONES-SEED, AMASS) are not redistributed; the
scripts in `tools/motion/` rebuild them from your own downloads, bit for bit.

## Shipped (own data, MIT)

| File | What it is | Provenance |
|---|---|---|
| `catch_pose_g1_dof.csv` | 29 G1 joint angles [rad] of the catch pose (MuJoCo joint order) | One frame of a video of one of the authors, lifted with GVHMR and retargeted to the G1 with GMR. The two wrist rolls were offset by ±π/2 and wrist pitch/yaw set to 0 by hand. |
| `ready_pose_capture.npy` | (400, 83) float32 AMP features of the "ready" pose | Recorded in Isaac Lab: the G1 held a hand-set ready pose (from photos of an author) under PD control with a small sinusoidal wobble, after 100 settling steps. Same 83-dim layout as `amp_dataset.pt`. |

## Rebuilt from public data (not shipped)

| Output | Needs | Script |
|---|---|---|
| `walk_then_catch_v17_30s.npz` (50 fps, 1500 frames): reference motion of the catch policy, needed for training **and** evaluation | BONES-SEED clip `210707/walk_forward_grab_normal_003__A006_M` + `catch_pose_g1_dof.csv` | `build_reference_motion.py` (Isaac Lab) |
| `male2_walk_v2.npz`, `male2_walk_turn_left_45.npz` (50 fps, 299 frames) | AMASS/ACCAD `Male2Walking_c3d/B10 - Walk turn left 45`, G1 retarget | `build_amass_motion.py` (Isaac Lab, once per file) |
| `amp_dataset.pt` (6576 × 2 × 83): AMP style data | the three files above + `ready_pose_capture.npy` | `build_amp_dataset.py` (NumPy + PyTorch only) |

### Downloads

1. **BONES-SEED** (gated, accept the licence first): Hugging Face dataset
   [`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed).
   Extract only the G1 retarget of one clip from the `g1` archive:
   `g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv`
   (sha256 `7979e92da40a2241341d02609a37381b92fe00af3f755b20453a65dc3339a8af`).
   The BONES-SEED licence forbids redistributing the data or any part of it,
   so do not commit or share the generated reference motion. Credit: "Motion Data by Bones Studio".
2. **Retargeted AMASS for Robotics** (CC BY 4.0; AMASS/ACCAD terms also apply): Hugging Face dataset
   [`fleaven/Retargeted_AMASS_for_robotics`](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics),
   file `g1/ACCAD/Male2Walking_c3d/B10 -  Walk turn left 45_poses_120_jpos.npy`
   (two spaces after the dash; sha256 `f7be749b1f59500c48a4c1421c00f6fefa36052b154b97dcdb70658ce031404f`).

```bash
huggingface-cli download fleaven/Retargeted_AMASS_for_robotics --repo-type dataset \
    "g1/ACCAD/Male2Walking_c3d/B10 -  Walk turn left 45_poses_120_jpos.npy" --local-dir data/amass_g1
```

### Rebuild

Run with the Isaac Lab Python and the `whole_body_tracking` package installed.

```bash
OUT=data/motions
B10="data/amass_g1/g1/ACCAD/Male2Walking_c3d/B10 -  Walk turn left 45_poses_120_jpos.npy"
python tools/motion/build_reference_motion.py --headless --out_dir $OUT \
    --bones_csv data/bones_seed/g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv
python tools/motion/build_amass_motion.py --headless --z_offset 0.0  --amass_npy "$B10" --out $OUT/male2_walk_turn_left_45.npz
python tools/motion/build_amass_motion.py --headless --z_offset 0.76 --amass_npy "$B10" --out $OUT/male2_walk_v2.npz
python tools/motion/build_amp_dataset.py --reference_motion $OUT/walk_then_catch_v17_30s.npz \
    --male2_walk_v2 $OUT/male2_walk_v2.npz --male2_turn_left_45 $OUT/male2_walk_turn_left_45.npz \
    --out data/amp/amp_dataset.pt
```

Each script takes `--verify <reference file or dir>` and prints the max
absolute difference per array.

### How the reference motion is built

1. BONES CSV (120 fps; cm, degrees, Euler "xyz") → metres, quaternion,
   radians, every 4th frame (30 fps), as in NVIDIA SONIC's motion_lib converter.
2. Walk frames 40–219 (6 s); root x locked to 0, root y relative to the first
   walk frame. Then 30 frames of linear blend from the last walk pose to the
   catch pose and 180 frames holding the catch pose; the root stays at the
   last walk frame (13 s, 390 frames).
3. Written as BeyondMimic CSV text with `%.6f` and read back (the original
   pipeline did this, and the rounding is part of the result).
4. Isaac Lab kinematic replay (port of BeyondMimic `csv_to_npz.py`): 30 → 50 fps
   (lerp / slerp), finite-difference velocities, PhysX forward kinematics for
   all 30 bodies (649 frames).
5. Extended to 1500 frames by holding the last pose with zero velocities.
