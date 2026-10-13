# Reproducing the paper from this release

This note says, stage by stage, what the released checkpoints are, how they were produced, and what a
rerun of the scripts in this repository is expected to give. "Exact" below means identical per-throw
outcomes; everything that renders camera images in Isaac Sim is non-deterministic and only matches to
within a few points.

## Checkpoints (`scripts/download_checkpoints.sh`)

| File | Stage | Produced by |
|---|---|---|
| `teacher_stage1.pt` | 1 | PPO + AMP on the state-based task; final run of a warm-started sequence (see below) |
| `bc_init_stage2.pt` | 2 | behaviour cloning + DAgger from the teacher (`train_stage2.sh bc`) |
| `latent_stage2.pt` | 2 | joint encoder/decoder PPO from `bc_init_stage2.pt` (`train_stage2.sh ppo`) |
| `stage3_heatmap_cnn.pt`, `stage3_feature_net.pt`, `stage3_latent_weights.npz`, `visual_encoder_stage3.pt` | 3 | the steps of `train_stage3.sh` (see `docs/STAGE3.md`) |

The checkpoints are the ones evaluated in the paper. Only path strings inside the files were changed
(laboratory paths removed); all tensors are identical, and `assets/checkpoints.sha256` pins them.

## Evaluation (`scripts/eval.sh`)

The protocol is the paper's: one observed throw per environment, static thrower 3.0–3.5 m away, 640
control steps, no curriculum events, 320 environments per seed, seeds 8 and 9 pooled (640 throws).

- `teacher` and `true_z` do not render. On our machine (RTX 3090, the Docker image) this code reproduced
  the paper's per-throw results exactly (checked on seed 8, all 320 throws, `regular` condition). A tester
  on a different machine saw individual throws differ while the pooled rates matched within a point or two;
  physics and PPO rollouts are not bit-reproducible across GPUs or driver versions.
- `visual` renders RGB-D frames. Reruns differ by a few points (paper 88.8 on `regular`; this code
  gave 89.1 / 88.4 on seeds 8 / 9 and 90.3 on a second seed-8 run).
- The conditions in `configs/eval/conditions/` are the columns of the paper's robustness table and the
  lateral-offset sweep. Paper numbers are listed in the README.

## Stage 1 — teacher

The released teacher came from a sequence of warm-started PPO runs; the run directories of the earlier
links of that sequence no longer exist, so the sequence cannot be replayed. `configs/train/stage1_teacher.env`
is the configuration of the final run (600 iterations, 4096 environments, seed 42).
`scripts/train_stage1.sh --warm_start data/checkpoints/teacher_stage1.pt` continues from the released
teacher with that configuration.

**From scratch (tested 2026-10-13, RTX 3090, release image).** `scripts/train_stage1.sh` without
`--warm_start` (600 iterations, 34 min, 28.5k steps/s), then resumed with
`--resume True --load_run <run> --checkpoint model_599.pt --max_iterations 1800` to 2400 iterations in
total. The PPO reward rose throughout (about 25 at iteration 600, 33 at 1200, 47 at 2400) and episodes
reached full length without falls, but catch success on `regular` (sustained hold, 320 throws per seed)
did not improve:

| Iterations | Seed 8 | Seed 9 | Interception |
|---|---|---|---|
| 600 | 13.8 | 12.8 | 32.5 / 30.9 |
| 1200 | 15.3 | — | 33.1 |
| 2400 | 12.5 | 10.6 | 32.2 / 29.1 |
| released teacher | 88.8 | 90.0 | 94.1 / 93.4 |

With this configuration alone the policy learns to stand and track the reference motion but the
catching skill does not emerge; the released teacher depends on the earlier, non-reproducible stages
of its warm-start sequence. Stages 2 and 3 start from the released teacher and are reproducible.

## Stage 2 — latent interface

`scripts/train_stage2.sh bc` and `scripts/train_stage2.sh ppo` run the two steps with the released
configuration (`configs/train/stage2_latent.env`, `stage2_bc_perception.env`). Both steps are stochastic
(PPO rollouts, DAgger data), so a rerun gives a different checkpoint, not the released one. To compare a
rerun against the paper, evaluate it with `scripts/eval.sh true_z --ckpt <your model_599.pt>`; the
released controller gives 88.4 on `regular` (640 throws).

`bc` writes to `data/stage2/bc_init.pt` and never overwrites the released `bc_init_stage2.pt`; `ppo`
starts from the released file unless `--init` is given.

## Stage 3 — visual encoder

`scripts/train_stage3.sh` retrains the visual encoder for the fixed Stage 2 controller: heatmap CNN on
the released 96k-frame set (`scripts/download_stage3_frames.sh`), feature collection, temporal encoder,
two DAgger rounds, evaluation. `docs/STAGE3.md` lists the reference number of every step. A full run in
the release Docker image took 1 h 49 min on an RTX 3090 and gave 90.3 / 90.9 on seeds 8 / 9 (released
encoder: 89.1 / 88.4). The feature collections of that run were identical to the laboratory ones; the CNN
step is not deterministic.

## Details not stated in the paper

- The Stage 3 heatmap CNN is trained on frames rendered while an earlier, unreleased 16-D latent
  controller drove the robot; its latent is the CNN's auxiliary target and is discarded afterwards
  (`docs/STAGE3.md`). Everything after that step uses the released Stage 2 controller.
- Stage 2 training samples the three box sizes 4:1:1 (training box : small : large;
  `BOX_MODEL=mix:small,small,small,small,mini,largeold` in `configs/train/stage2_latent.env`, where `small` is the
  internal name of the 34×26×22 cm training box). Stage 1 trains on the training box only.
- The two-stage arm reflex (arms down while waiting, raised when a throw is detected) is triggered by the
  simulated box-state estimator, and the reference arm pose in the observation switches on the simulator's
  launch flag (`BOX_V38_REF_ARMS`, `BOX_V38_TS_FROM_PERC` in the configs); neither is derived from the
  visual latent.
- The paper's torso-excursion statistic (38° threshold) is not computed by the released evaluator.

## What is not in this release

- The hardware deployment stack (onboard perception and control on the Unitree G1).
- The ablation and baseline runs of the paper (initialization, reward, visual-encoder ablations;
  detector + state-policy baselines).
- The reference motion and AMP data themselves (licence); `tools/motion/` rebuilds them from the public
  sources listed in the README.
