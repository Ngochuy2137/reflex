<h1 align="center"><code>Reflex</code>:<br>Visually Grounded Reactive Humanoid Control</h1>

<p align="center">
  <a href="https://toberj.github.io/reflex/static/paper/reflex.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2610.12772"><img src="https://img.shields.io/badge/arXiv-2610.12772-b31b1b.svg" alt="arXiv"></a>
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

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [GUI](#gui)
- [Data](#data)
- [Evaluation](#evaluation)
  - [Interactive catching demo](#interactive-catching-demo)
- [Training](#training)
  - [Online logging with W&B](#online-logging-with-wb)
  - [Train your own Stage 2 and 3](#train-your-own-stage-2-and-3)
- [Repository layout](#repository-layout)
- [License and acknowledgements](#license-and-acknowledgements)
- [Citation](#citation)

## Requirements

- Linux with an NVIDIA **RTX** GPU. Stage 3 and visual evaluation render cameras, which needs RT cores,
  so data-centre GPUs without them (A100, H100) cannot run those parts. About 14 GB of GPU memory for
  visual evaluation with 320 environments; about 6 GB for the other evaluations and for training.
- NVIDIA driver ≥ 570.169, Docker with the NVIDIA Container Toolkit (for the Docker route).
- About 40 GB of free disk (the Isaac Sim image is ~15 GB).

The paper's results were produced with Isaac Sim 5.1.0, Isaac Lab v2.3.1, rsl-rl-lib 3.0.1 and PyTorch 2.7.0 (CUDA 12.8).
The release was also run end to end on an RTX 4090 with a pip install of Isaac Sim 5.1 and Isaac Lab 2.3.1 (no Docker).

## Installation

Use the Docker image for the paper's environment. The host needs [Docker Compose](https://docs.docker.com/engine/install/ubuntu/)
and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). All host commands below run from this checkout's
root unless stated otherwise.

Clone the separate dataset checkout described in the
[dataset README](../reflex_dataset/README.md). To use it without copying,
set `REFLEX_DATA_DIR="$(pwd)"` from its root, then return to the Reflex root:

```bash
export REFLEX_ROOT="$(pwd)"
# Skip if data already points to the dataset checkout.
ln -sT "$(realpath --relative-to=. "$REFLEX_DATA_DIR")" data
docker compose -f docker/compose.yaml build
bash docker/run.sh
```

The shell opens inside the container. `exit` closes the shell; the container is
kept. On the host, `docker compose -f docker/compose.yaml stop` stops it without
removing it. `data`, `outputs`, and `logs` are mounted from the host. Files kept
only inside the container are lost if it is removed or recreated. Source changes
on the host need a new image build and container recreation. Prepare the motion
files using the dataset README after starting the container.

**Existing Isaac Lab v2.3.1 install.** Use the Isaac Lab Python (`./isaaclab.sh -p` or `/isaac-sim/python.sh`):

```bash
# Run from this checkout with your existing Isaac Lab Python active.
export ISAACLAB_PY=python
$ISAACLAB_PY -m pip install rsl-rl-lib==3.0.1 tensordict==0.12.0 toml joblib scipy matplotlib "imageio[ffmpeg]" onnxscript psutil wandb
$ISAACLAB_PY -m pip install --no-deps -e source/whole_body_tracking -e source/reflex
scripts/setup_assets.sh                            # Unitree G1 description + the hand-collider URDF
```

### GUI

Run from an X11 desktop terminal on the host with `DISPLAY`, `XAUTHORITY`, and
`xauth` available. **Before recreating an existing container, stop training and
save files stored only inside it.** Then enable the display mounts:

```bash
bash docker/prepare_gui.sh
docker compose -f docker/compose.yaml -f docker/compose.gui.yaml \
  up -d --no-build --pull never --force-recreate reflex
bash docker/run.sh --gui
```

The helper stores a private X11 cookie in `.docker-runtime/`, excluded from Git
and image builds. Only grant desktop access to a trusted container. Later,
`bash docker/run.sh --gui` reuses this container without recreating it.
After logging out of the desktop and back in, run `bash docker/run.sh --gui` command again from the
host to refresh X11 authorization before starting GUI programs.

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
    --bones_csv data/raw/bones_seed/g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv
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

Run the commands below from the Reflex root inside the prepared container.
Add `--gui` to open a simulator window, for example:

```bash
bash scripts/eval.sh teacher --gui --num_envs 16
```

Headless is the default. Enable the container display as described under
[GUI](#gui) first. The small GUI example is not the paper evaluation protocol.

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

`true_z` and `teacher` do not render; on our machine (RTX 3090, the Docker image) this code reproduced
the paper's per-throw results exactly (checked on seed 8, all 320 throws). On other GPUs or installs the
individual throws can differ while the pooled rates stay within a point or two. Camera
rendering in Isaac Sim is not deterministic, so `visual` varies by 1--3 points between runs of the same
encoder (seed 8 gave 86.6--90.3 across our and an external tester's runs).

### Interactive catching demo

The demo uses one robot, its simulated RGB-D camera, and your matching Stage 2/3
checkpoints. After the one-time [GUI setup](#gui), start it in two steps.

**1. On the host**, from the Reflex root, log in a container shell with GUI access:

```bash
bash docker/run.sh --gui
```

This refreshes desktop authorization and opens a shell; it does not start the demo.

**2. Inside that container shell**, from the Reflex root, start the demo:

```bash
# Replace TIMESTAMP with each existing checkpoint folder's timestamp.
STAGE2_CKPT="logs/experiment_01/stage2_ppo_<TIMESTAMP>/model_599.pt"
STAGE3_CKPT="logs/experiment_01/stage3_<TIMESTAMP>/visual_encoder.pt"
bash scripts/demo.sh \
  --ckpt "$STAGE2_CKPT" --detector "$STAGE3_CKPT"
```

If you are already inside the container and GUI access works, only step 2 is
needed. After logging out of the desktop and back in, repeat step 1 from the host.

The **Reflex Catching Demo** window has distance and object-seed fields:

- Empty distance: sample from the regular 3–3.5 m range. Enter a number to fix
  the distance in meters. Distances beyond 4.2 m use the existing 12 m/s speed
  cap from the 5 m/7 m evaluation presets; very distant throws can still fall short.
- Entered seed: Reset repeats the sampled object parameters. Empty seed: Reset
  chooses a new seed, shown in the status. The seed controls only object sampling,
  including release position, orientation, velocity-related parameters and spin.
- **Reset** applies the fields and prepares a trial without advancing physics.
  **Play** runs that trial once. Editing a field requires Reset before Play.
- A 0.6 s hold near the torso, a landed object, a fallen robot, an invalid state
  or the 12.8 s simulation limit freezes the scene. Inspect it, then press Reset.
  Play cannot restart a finished trial without Reset.

Optional starting values: `--distance 5 --seed 9`. Simulation is paced near real
time, but may run slower if rendering is expensive. The status shows simulation
time, wall-clock time and their speed ratio. `1.00x`
means real time; `0.25x` means one simulation second takes four wall-clock seconds.
This display slowdown does not change gravity or the physics/control timestep.
Completed trial JSON also contains per-step timing, posture, speed and runtime
settings for diagnosis. Only the sampler is reseeded;
robot resets keep their existing distribution. Throws are anchored to the robot,
and launch velocity is adjusted to its live position, just as in evaluation.
Thus the same object seed does not promise identical world trajectories when
the robot state changes, or the same trial as `eval.sh --seed` (which seeds the
whole environment). Reset clears camera/policy history between trials.

This viewer does not train or upload W&B runs. Its hold indicator is for visual
inspection, not the full benchmark success rule, which also checks later drops.
Completed trial details are saved in `outputs/demo/last_session.json`.
Use `eval.sh` for quantitative success rates.

## Training

Run these commands inside the container. Stage 2 BC/PPO and Stage 3 accept
`--gui` and `--num_envs N`; omit them to keep the original headless defaults.
Stage 3 opens windows only during simulation collection and final evaluation.
BC uses `--teacher` for a Stage 1 checkpoint. PPO uses `--init` for a Stage 2
BC checkpoint. See [Train your own Stage 2 and 3](#train-your-own-stage-2-and-3)
for a sequence that keeps your outputs separate from the released models.

How the released checkpoints were produced and what a rerun is expected to give: [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

```bash
scripts/train_stage1.sh --warm_start data/checkpoints/teacher_stage1.pt   # Stage 1 (see note)
scripts/train_stage2.sh bc                                              # Stage 2: behaviour cloning + DAgger (~15 min)
scripts/train_stage2.sh ppo                                             # Stage 2: joint PPO, 600 iterations (~1 h on an RTX 3090)
scripts/train_stage3.sh                                                 # Stage 3: visual encoder + its evaluation (~1 h 50 min)
```

All new training outputs default to `logs/`. Each run has its own folder, with
`stage1_<timestamp>`, `stage2_bc_<timestamp>`, `stage2_ppo_<timestamp>` and
`stage3_<timestamp>` inside it. Explicit `--out` paths
are still supported. Downloaded checkpoints and input frames stay under `data/`.

- Stage 1 PPO outputs go to `logs/<run_name>/stage1_<timestamp>/`.
- Stage 2 PPO outputs go to `logs/<run_name>/stage2_ppo_<timestamp>/`.
  Both folders contain TensorBoard logs, configuration files and `model_<it>.pt`.
  To resume, use the same `--run_name` and select a stage folder with `--load_run`.
  Automatic lookup only considers folders for the current stage.
- `scripts/train_stage2.sh bc` writes `logs/<run_name>/stage2_bc_<timestamp>/bc_init.pt`; the released `data/checkpoints/bc_init_stage2.pt`
  is left untouched. `scripts/train_stage2.sh ppo` starts from the released file by default; pass
  `--init logs/YOUR_RUN/stage2_bc_TIMESTAMP/bc_init.pt` to use your own behaviour-cloning result.
- **Stage 1 note.** The released teacher (`data/checkpoints/teacher_stage1.pt`) was trained through a
  sequence of warm-started runs (paper, Sec. Method); `configs/train/stage1_teacher.env` is the configuration
  of its final run (600 iterations, 4096 environments). Stages 2 and 3 and all evaluations start from the
  released teacher, which reproduces the paper's teacher numbers (see Evaluation). See `docs/REPRODUCE.md`.
- **Stage 3.** `scripts/train_stage3.sh` trains the visual encoder for the fixed Stage 2 controller and evaluates it:
  heatmap CNN on 96k rendered RGB-D frames (70 min on an RTX 3090), feature collection with the controller driving
  on its privileged latent (15 min), temporal encoder and two DAgger rounds with the encoder in the loop (20 min),
  `scripts/eval.sh visual --seed 8` (4 min). The result is `logs/default/stage3_<timestamp>/visual_encoder.pt`; a full run gave
  90.3 / 90.9 % on seeds 8 / 9 (released encoder: 89.1 / 88.4). Pass `--out logs/YOUR_RUN/stage3_TIMESTAMP` to resume an existing output folder. The first step reads the rendered frame set (2.1 GB) from `data/stage3/frames/v10_s7_rgbd/`
  (`chunk_0000.npz` … `chunk_0011.npz` + `labels_v10.npz`), downloaded by `scripts/download_stage3_frames.sh`;
  see [`docs/STAGE3.md`](docs/STAGE3.md), which also lists the reference numbers of every step. `--cnn data/checkpoints/stage3_heatmap_cnn.pt` skips the first
  step and needs no frames (~40 min in total).

### Online logging with W&B

Run inside the container, from the Reflex root. Log in once, then enable W&B
for the training and evaluation commands in this section:

```bash
"$ISAACLAB_PY" -m wandb login
export REFLEX_LOGGER=wandb
export WANDB_PROJECT=reflex
export WANDB_RUN_GROUP=experiment_01
```

Enter your API key in the login prompt. Do not put it in scripts or Git.
Set `WANDB_ENTITY` only if you want to use a specific account or team.
Each training process creates a separate run in the same group:

| Step | Logged values |
| --- | --- |
| Stage 2 BC | Training loss and held-out action error, including box-visible frames |
| Stage 2 PPO | Native RSL-RL rewards, losses, episode length and environment metrics |
| Stage 3 CNN and temporal encoder | Training loss and existing test errors; best model scores in the run summary |
| Evaluation | Throw/catch/fall counts and success rates after evaluation completes |

Stage 3 data collection does not create a separate W&B run. Its CNN and encoder
URLs appear in `logs/<run>/stage3_<timestamp>/logs/`. Skipped training steps create no new run.
Evaluation rates use fractions from 0 to 1; no rate is logged when no throws were observed.

You can also select logging per command. Create `RUN`, `BC_CKPT` and the BC checkpoint first,
as shown in the next section:

```bash
bash scripts/train_stage2.sh ppo \
  --init "$BC_CKPT" --run_name "$RUN" \
  --logger wandb --log_project_name reflex --wandb_group "$RUN"
```

| Option | Values and purpose |
| --- | --- |
| `--logger` | `wandb` enables online logging. PPO also supports `tensorboard` (default); other steps support `none` (default). `REFLEX_LOGGER` sets the shared default. |
| `--log_project_name` | W&B project; defaults to `WANDB_PROJECT` or `reflex`. |
| `--wandb_group` | Groups related runs; defaults to `WANDB_RUN_GROUP`, then the training run name or `evaluation`. |
| `--wandb_mode` | `online` (default) or `offline` for local testing without uploads. |

W&B prints the dashboard URL. BC, Stage 3 and evaluation also save run details
under `logs/wandb/links/`. PPO saves `wandb_run.json` in its training log folder
when training ends and keeps its local TensorBoard logs. Native RSL-RL W&B
logging also uploads PPO checkpoints. Existing running jobs keep their original logger;
these options apply when starting a new process.

### Train your own Stage 2 and 3

Run inside the container, from the Reflex root, with reference and AMP data ready.
This example keeps the released Stage 1 teacher and trains new Stage 2/3 models.
Choose a new experiment name for each fresh run:

```bash
RUN=experiment_01

# Stage 2 BC: learn from the Stage 1 teacher and save a new BC model.
# use wandb logger if you want
export REFLEX_LOGGER=wandb
export WANDB_PROJECT=reflex
export WANDB_RUN_GROUP="$RUN"

# Feed your trained policy of stage 1 to stage 2 BC training through parameter --teacher 
bash scripts/train_stage2.sh bc \
  --teacher data/checkpoints/teacher_stage1.pt --run_name "$RUN"

# Copy the actual BC output path printed by the script; replace TIMESTAMP below.
BC_CKPT="logs/$RUN/stage2_bc_TIMESTAMP/bc_init.pt"

# Stage 2 PPO: start from that BC model, not the released BC model.
# use wandb logger if you want
export REFLEX_LOGGER=wandb
export WANDB_PROJECT=reflex
export WANDB_RUN_GROUP="$RUN"

# Feed your trained policy of stage 2 BC to stage 2 PPO training through parameter --init 
bash scripts/train_stage2.sh ppo \
  --init "$BC_CKPT" --run_name "$RUN"
```

PPO saves models under `logs/experiment_01/stage2_ppo_<timestamp>/`.
After PPO finishes, set the actual path to the checkpoint you want to use:

```bash
# Replace TIMESTAMP with the timestamp of your Stage 2 PPO folder.
STAGE2_CKPT=logs/$RUN/stage2_ppo_TIMESTAMP/model_599.pt

# Stage 3 computes controller-specific weights and creates its output folder.
# Download dataset (Skip the download if the frame set is already present.)
bash scripts/download_stage3_frames.sh

# use wandb logger if you want
export REFLEX_LOGGER=wandb
export WANDB_PROJECT=reflex
export WANDB_RUN_GROUP="$RUN"

# Train
bash scripts/train_stage3.sh --run "$RUN" \
  --controller "$STAGE2_CKPT" --compute_weights
```

The scripts create timestamps automatically. You only choose existing input
checkpoint paths; do not create timestamped output folders in the shell.
Stage 3 prints its output folder, with `latent_weights.npz`, `latent_probe.json`
and the final `visual_encoder.pt` inside it.
Add `--gui` to the BC, PPO, or Stage 3 command for a window. `--num_envs` changes
training/collection sizes, not just how many robots are shown.
To reuse the released image network, skip the frame download and add
`--cnn data/checkpoints/stage3_heatmap_cnn.pt` to Stage 3.

Keep `--init` and `--controller` when connecting your own outputs. Use
`--compute_weights` for a new controller, or `--zw` to reuse its existing weights;
these two options cannot be combined. Without either option, Stage 3 uses the
released weights. The author did not include this custom-controller weight
calculation in their reproduction check; see
[Using another controller](docs/STAGE3.md#using-another-controller).
Stage 3 reuses existing outputs by filename, so use a new output folder after changing inputs. To resume, pass the same `--out` folder.

## Repository layout

```
configs/eval/        evaluation protocol (*.env) and per-condition overlays (conditions/*.env)
configs/train/       training configurations of Stage 1 and Stage 2
scripts/             eval.sh, train_stage{1,2,3}.sh, setup_assets.sh, download_checkpoints.sh, download_stage3_frames.sh
source/reflex/       the reflex package: envs (catch task), teacher, latent, vision, amp, eval
source/whole_body_tracking/   BeyondMimic (motion tracking), trimmed to what Reflex uses
tools/motion/        rebuild the reference motion and AMP data from public datasets
assets/              robot URDF with hand colliders, our own motion data, checkpoint and frame-set checksums
docker/              Dockerfile, Compose files, GUI helpers and runtime check
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
@misc{jia2026reflexvisuallygroundedreactive,
      title={Reflex: Visually Grounded Reactive Humanoid Control},
      author={Taoyang Jia and Weikai Huang and Linxin Song and Jared Darlington and Jieyu Zhao and Yue Wang and Jiafei Duan and Zhongzheng Ren and Ranjay Krishna},
      year={2026},
      eprint={2610.12772},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.12772},
}
```
