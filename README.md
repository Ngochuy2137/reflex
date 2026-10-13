<h1 align="center"><code>Reflex</code>:<br>Visually Grounded Reactive Humanoid Control</h1>

<p align="center">
  <a href="https://toberj.github.io/reflex/static/paper/reflex.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper"></a>
  <a href="https://toberj.github.io/reflex/"><img src="https://img.shields.io/badge/Project%20Page-Reflex-6b4fa0.svg" alt="Project page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://toberj.github.io/">Taoyang Jia</a><sup>*1,2</sup> ·
  <a href="https://weikaih04.github.io/">Weikai Huang</a><sup>*1</sup> ·
  <a href="https://linxins.net/">Linxin Song</a><sup>*3</sup> ·
  <a href="https://jareddarlington.github.io/">Jared Darlington</a><sup>1</sup> ·
  <a href="https://jyzhao.net/">Jieyu Zhao</a><sup>3</sup> ·
  <a href="https://yuewang.xyz/">Yue Wang</a><sup>3</sup> ·
  <a href="https://jiafei1224.github.io/">Jiafei Duan</a><sup>4</sup> ·
  <a href="https://jason718.github.io/">Zhongzheng Ren</a><sup>5,6</sup> ·
  <a href="https://ranjaykrishna.com/index.html">Ranjay Krishna</a><sup>1</sup>
  <br>
  <sup>1</sup>University of Washington &nbsp; <sup>2</sup>Stanford University &nbsp;
  <sup>3</sup>University of Southern California &nbsp; <sup>4</sup>National University of Singapore
  <br>
  <sup>5</sup>University of North Carolina at Chapel Hill &nbsp; <sup>6</sup>Allen Institute for AI
  <br>
  <sup>*</sup>Equal contribution
</p>

<p align="center">
  <img src="assets/teaser.webp" width="100%" alt="A Unitree G1 humanoid catches human-thrown boxes of different sizes and payloads from 2.5 to 7 m using onboard RGB-D sensing.">
</p>

This repository contains the simulation code of Reflex: training of all three stages, the evaluation
protocol behind the paper's simulation results, and the released checkpoints. It runs in
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) and builds on the
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) motion-tracking code.

<p align="center">
  <img src="assets/method.png" width="100%" alt="Three-stage framework: a state-based teacher learns whole-body catching with PPO, a privileged encoder and policy decoder learn a latent of the box dynamics, and an egocentric RGB-D encoder learns to recover that latent from images while the controller stays fixed.">
</p>

| Stage | What it learns | Code | Checkpoint |
|---|---|---|---|
| 1 | State-based teacher: whole-body catching from (noisy, delayed) box state, PPO + AMP | `reflex/envs`, `reflex/teacher` | `teacher_stage1.pt` |
| 2 | Privileged encoder → 8-D latent `z` → action decoder; behaviour cloning from the teacher, then joint PPO | `reflex/latent` | `latent_stage2.pt` |
| 3 | Egocentric RGB-D visual encoder that predicts `z` from a 25-frame history and camera ego-motion; the controller stays fixed | `reflex/vision` | `visual_encoder_stage3.pt` |

The real-robot deployment stack is not part of this release.

## Requirements

- Linux with an NVIDIA **RTX** GPU. Stage 3 and visual evaluation render cameras, which needs RT cores,
  so data-centre GPUs without them (A100, H100) cannot run those parts. About 14 GB of GPU memory for
  visual evaluation with 320 environments; about 6 GB for the other evaluations and for training.
- NVIDIA driver ≥ 570.169, Docker with the NVIDIA Container Toolkit (for the Docker route).
- About 40 GB of free disk (the Isaac Sim image is ~15 GB).

The paper's results were produced with Isaac Sim 5.1.0, Isaac Lab v2.3.1, rsl-rl-lib 3.0.1 and PyTorch 2.7.0 (CUDA 12.8).

## Installation

**Docker (recommended).** The image pins the versions above.

```bash
git clone https://github.com/ToberJ/reflex.git && cd reflex
docker build -f docker/Dockerfile -t reflex .
docker/run.sh          # shell in /reflex with the GPU; ./data, ./outputs and ./logs are mounted
```

**Existing Isaac Lab v2.3.1 install.** Use the Isaac Lab Python (`./isaaclab.sh -p` or `/isaac-sim/python.sh`):

```bash
export ISAACLAB_PY=/isaac-sim/python.sh            # or: export ISAACLAB_PY="/path/to/IsaacLab/isaaclab.sh -p"
$ISAACLAB_PY -m pip install rsl-rl-lib==3.0.1 tensordict==0.12.0 toml joblib scipy matplotlib "imageio[ffmpeg]" onnxscript psutil wandb
$ISAACLAB_PY -m pip install --no-deps -e source/whole_body_tracking -e source/reflex
scripts/setup_assets.sh                            # Unitree G1 description + the hand-collider URDF
```

## Data

All downloaded or generated files go to `data/` (or `$REFLEX_DATA_DIR`).

**1. Checkpoints** (≈ 25 MB, verified by sha256):

```bash
scripts/download_checkpoints.sh
```

**2. Reference motion** — needed for evaluation *and* training. The policy tracks a reference motion
whose first six seconds come from the [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)
dataset, which may not be redistributed, so you rebuild it from your own copy:

1. Request access to `bones-studio/seed` on Hugging Face and download the G1 archive.
   Only one file is needed: `g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv`.
2. Rebuild (the script checks the input checksum; the output is bit-identical to the file used in the paper):

```bash
$ISAACLAB_PY tools/motion/build_reference_motion.py --headless \
    --bones_csv /path/to/g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv
# -> data/motions/walk_then_catch_v17_30s.npz
```

**3. AMP style data** — training only. See [`assets/motions/README.md`](assets/motions/README.md):
one AMASS clip from [Retargeted AMASS for Robotics](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics),
two Isaac replays, and `tools/motion/build_amp_dataset.py` → `data/amp/amp_dataset.pt`.

**4. Stage 3 frame set** (2.1 GB, verified by sha256) — only for retraining the heatmap CNN of Stage 3:

```bash
scripts/download_stage3_frames.sh
```

## Evaluation

```bash
scripts/eval.sh visual                   # Stage 3: visual encoder in the loop (paper Table I, "Reflex")
scripts/eval.sh true_z                   # Stage 2: privileged latent z
scripts/eval.sh teacher                  # Stage 1: state policy with ideal box measurements
scripts/eval.sh visual --condition dist_5m --seed 9
```

Each run simulates 320 environments with one observed throw each (the paper's protocol: static thrower
3.0–3.5 m away, 640 steps, no curriculum events), writes `outputs/eval/<mode>_<condition>_s<seed>.json`
and prints a summary line with the paper's metric. Conditions (`configs/eval/conditions/`) cover the
columns of Table I and the lateral-offset sweep: `regular`, `small`, `large`, `mass_{0.5kg,1kg,2kg}`,
`dist_{2.5m,5m,7m}`, `color_{white,grey,red,green,blue,black}`, `offset_{0.3,0.6,0.9,1.2,1.5}m`.

Expected catch success on the `regular` condition (paper: evaluation seeds 8 and 9, 640 throws):

| Mode | Paper | Seed 8 | Seed 9 |
|---|---|---|---|
| `visual` | 88.8 | 89.1 | 88.4 |
| `true_z` | 88.4 | 90.0 | 86.9 |
| `teacher` | 89.4 | 88.8 | 90.0 |

`true_z` and `teacher` do not render; with this code they reproduced the paper's per-throw results
exactly (checked on seed 8, all 320 throws). Camera
rendering in Isaac Sim is not deterministic, so `visual` varies by a few points between runs (a rerun
of seed 8 with this code gave 90.3).

## Training

How the released checkpoints were produced and what a rerun is expected to give: [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

```bash
scripts/train_stage1.sh --warm_start data/checkpoints/teacher_stage1.pt   # Stage 1 (see note)
scripts/train_stage2.sh bc                                              # Stage 2: behaviour cloning + DAgger (~15 min)
scripts/train_stage2.sh ppo                                             # Stage 2: joint PPO, 600 iterations (~1 h on an RTX 3090)
scripts/train_stage3.sh                                                 # Stage 3: visual encoder + its evaluation (~1 h 50 min)
```

- Outputs go to `logs/rsl_rl/reflex/<date>_<run_name>/` (TensorBoard logs and `model_<it>.pt`).
- `scripts/train_stage2.sh bc` writes `data/stage2/bc_init.pt`; the released `data/checkpoints/bc_init_stage2.pt`
  is left untouched. `scripts/train_stage2.sh ppo` starts from the released file by default; pass
  `--init data/stage2/bc_init.pt` to use your own behaviour-cloning result.
- **Stage 1 note.** The released teacher was trained through a sequence of warm-started runs; the
  earlier runs of that sequence are not reproducible. `configs/train/stage1_teacher.env` is the
  configuration of the final run (600 iterations, 4096 environments). Training from scratch with it
  has not been tested.
- **Stage 3.** `scripts/train_stage3.sh` trains the visual encoder for the fixed Stage 2 controller and evaluates it:
  heatmap CNN on 96k rendered RGB-D frames (70 min on an RTX 3090), feature collection with the controller driving
  on its privileged latent (15 min), temporal encoder and two DAgger rounds with the encoder in the loop (20 min),
  `scripts/eval.sh visual --seed 8` (4 min). The result is `data/stage3/default/visual_encoder.pt`; a full run gave
  90.3 / 90.9 % on seeds 8 / 9 (released encoder: 89.1 / 88.4). Rerunning the command resumes after the last
  completed step. The first step reads the rendered frame set (2.1 GB) from `data/stage3/frames/v10_s7_rgbd/`
  (`chunk_0000.npz` … `chunk_0011.npz` + `labels_v10.npz`), downloaded by `scripts/download_stage3_frames.sh`;
  see [`docs/STAGE3.md`](docs/STAGE3.md), which also lists the reference numbers of every step. `--cnn data/checkpoints/stage3_heatmap_cnn.pt` skips the first
  step and needs no frames (~40 min in total).

## Repository layout

```
configs/eval/        evaluation protocol (*.env) and per-condition overlays (conditions/*.env)
configs/train/       training configurations of Stage 1 and Stage 2
scripts/             eval.sh, train_stage{1,2,3}.sh, setup_assets.sh, download_checkpoints.sh, download_stage3_frames.sh
source/reflex/       the reflex package: envs (catch task), teacher, latent, vision, amp, eval
source/whole_body_tracking/   BeyondMimic (motion tracking), trimmed to what Reflex uses
tools/motion/        rebuild the reference motion and AMP data from public datasets
assets/              robot URDF with hand colliders, our own motion data, checkpoint and frame-set checksums
docker/              Dockerfile and run script
```

**Configuration.** The environments are configured through environment variables (`BOX_*`), which the
launch scripts load from `configs/`. Later files override earlier ones, so a condition overlay only
lists what it changes.

## License and acknowledgements

The code is released under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the licences of the code and data this project builds on, in particular
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) (MIT),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause) and
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) (BSD-3-Clause).
Training data includes Motion Data by [Bones Studio](https://bones.studio/) and motions from
[AMASS](https://amass.is.tue.mpg.de/) (ACCAD).

## Citation

```bibtex
@misc{jia2026reflex,
  title   = {Reflex: Visually Grounded Reactive Humanoid Control},
  author  = {Jia, Taoyang and Huang, Weikai and Song, Linxin and Darlington, Jared and
             Zhao, Jieyu and Wang, Yue and Duan, Jiafei and Ren, Zhongzheng and Krishna, Ranjay},
  year    = {2026}
}
```
