# Third-party notices

Reflex is released under the MIT License (see `LICENSE`). It includes or builds on the following.

## Code included in this repository

| Component | Where | License |
|---|---|---|
| [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) (motion-tracking environment, trimmed and extended) | `source/whole_body_tracking/` | MIT (`source/whole_body_tracking/LICENSE`); its Isaac Lab template notice is kept in `source/whole_body_tracking/LICENCE` |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) training-script template | `source/reflex/reflex/teacher/train.py`, `source/reflex/reflex/latent/train.py`, `source/reflex/reflex/utils/cli_args.py` | BSD-3-Clause (headers kept in the files) |

## Dependencies (installed separately, not redistributed)

| Component | License |
|---|---|
| [Isaac Sim](https://developer.nvidia.com/isaac/sim) 5.1.0 | NVIDIA Isaac Sim additional software and materials license |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) v2.3.1 | BSD-3-Clause |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 3.0.1 | BSD-3-Clause |
| PyTorch, NumPy, and other Python packages | their respective licenses |

## Robot model

The Unitree G1 description (`unitree_description`, downloaded by `scripts/setup_assets.sh` from the
BeyondMimic distribution) is BSD-licensed. `assets/robots/main_hands.urdf` is a modified copy of its
`urdf/g1/main.urdf` with box-shaped collision geometry added to the hands.

## Motion data (not redistributed)

- **BONES-SEED** ([bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed)). The first
  six seconds of the reference motion are a G1 retarget of the clip
  `210707/walk_forward_grab_normal_003__A006_M`. The BONES-SEED license does not permit redistributing
  the data, so `tools/motion/build_reference_motion.py` rebuilds the motion from your own download.
  Training data includes Motion Data by Bones Studio (https://bones.studio/).
- **AMASS / ACCAD** via [Retargeted AMASS for Robotics](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics).
  One walking clip (`ACCAD/Male2Walking_c3d/B10 - Walk turn left 45`) is part of the AMP style data,
  rebuilt by `tools/motion/build_amass_motion.py`. AMASS: Mahmood et al., "AMASS: Archive of Motion
  Capture as Surface Shapes", ICCV 2019; use is subject to the AMASS license.
- `assets/motions/catch_pose_g1_dof.csv` and `assets/motions/ready_pose_capture.npy` were created by
  the authors and are released under the MIT License.
