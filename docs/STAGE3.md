# Stage 3: training the visual encoder

The visual encoder (`DetectorKPSeq` in `reflex/vision/detector_model.py`) maps the egocentric RGB-D history to the
8-D latent `z` of the Stage 2 controller, which stays fixed:

- **per frame**: an RGB-D CNN with a supervised box heatmap → soft-argmax pixel, local-minimum depth, visibility
  logit and the pinhole lift to a camera-frame point (8-D), plus the camera ego-motion (angular and linear velocity,
  gravity direction; 9-D);
- **over time**: a 2-layer GRU over the last 25 frames of these 17-D features → `z` (and auxiliary box position/velocity).

It is trained in two parts: the CNN on a set of rendered frames, then the GRU on per-frame features logged while the
controller runs in simulation, first driven by its own privileged latent and then by the encoder (DAgger).
`scripts/train_stage3.sh` runs the whole recipe of the released encoder (`data/checkpoints/visual_encoder_stage3.pt`).

## Quick start

```bash
scripts/train_stage3.sh                    # all steps (~1 h 50 min on one RTX 3090) -> data/stage3/default/visual_encoder.pt
scripts/train_stage3.sh --run from_cnn --cnn data/checkpoints/stage3_heatmap_cnn.pt   # reuse the released CNN (~40 min, no frame set needed)
scripts/eval.sh visual --detector data/stage3/default/visual_encoder.pt --condition dist_5m --seed 9      # any other condition
```

A step whose output exists is skipped, so rerunning the same command resumes after the last completed step (outputs
are written under a temporary name and renamed when the step succeeds). Outputs are reused by name only: start a new
`--run` after changing an input. `scripts/train_stage3.sh --print_config` prints the configuration of the four
collection cells. Options: see the header of the script.

## Steps

| # | Step | Code | Output in `data/stage3/<run>/` | Time (RTX 3090) |
|---|---|---|---|---|
| 1 | Heatmap CNN on the rendered frame set: 20 epochs, colour augmentation of the box (p = 0.7), flight frames up-weighted, best epoch by flight-window latent error | `reflex.vision.train_detector` | `heatmap_cnn.pt` | 70 min |
| 2 | Feature extractor: the CNN wrapped into a `kpseq` checkpoint (its GRU is untrained and unused) | `reflex.vision.make_feature_net` | `feature_net.pt` | seconds |
| 3 | Teacher collection: the controller drives on its privileged latent, the encoder runs alongside (`--det_shadow`); per-frame features and the controller's latent z\* are logged (`--save_feats`). Seeds 11 and 12 × {`canon` 320 envs, `offset` 160 envs}, training box mix, random box colours | `reflex.eval.per_throw_eval` | `collect/teacher_*` | 15 min |
| 4 | Temporal encoder `it0` on the teacher features (GRU, 60 epochs, task-aware per-dimension weights `stage3_latent_weights.npz`) | `reflex.vision.train_seqdet` | `encoder_it0.pt` | 2 min |
| 4 | DAgger, 2 rounds: collect with `it<k>` driving the controller (seed 7, `canon` + `offset`), retrain `it<k+1>` from scratch on all features so far | both | `collect/dagger<k>_*`, `encoder_it{1,2}.pt` | 20 min |
| 5 | Final encoder `visual_encoder.pt` (= `encoder_it2.pt`); CLEAN2 evaluation, regular box, seed 8 (`scripts/eval.sh visual`) | `reflex.eval.per_throw_eval` | `eval/visual_regular_s8.json` | 4 min |

Logs of every step are in `data/stage3/<run>/logs/`. The `*.shadow.npz` files of the teacher collection are a
diagnostic by-product of the shadow mode and are not used.

**Resources.** Step 1 holds the whole frame set in memory (about 14 GB of RAM) and uses about 8 GB of GPU memory.
Steps 3–5 render 320 cameras and need about 14 GB of GPU memory and an RTX GPU (see Requirements in the README).
Outputs take about 0.3 GB.

## Frame set (input of step 1)

Step 1 trains on 96,000 rendered RGB-D frames (2.1 GB), expected at `data/stage3/frames/v10_s7_rgbd/`
(change with `--frames DIR`, `--labels FILE`):

```
data/stage3/frames/v10_s7_rgbd/
  chunk_0000.npz ... chunk_0011.npz   8,000 frames each: rgb (N,144,192,3) uint8, depth (N,144,192) uint16 [mm],
                                      pos / vel (box, camera frame), priv, obs, env, ep, tick
  labels_v10.npz                      z (96000,16), pos, vel, env, ep, tick, chunk, row, priv, ckpt
```

**Download:** `scripts/download_stage3_frames.sh` fetches the 13 files (2.1 GB) from the GitHub release
`stage3-frames-v1` and checks them against `assets/stage3_frames.sha256`. Steps 2–5 do not need the frames:
pass `--cnn data/checkpoints/stage3_heatmap_cnn.pt` to start from the released CNN.

The frames were rendered by the lab version of `reflex.vision.render_frames` (RGB-D, 192 × 144, 32 environments ×
3,000 control steps, seed 7) while an earlier 16-D latent controller drove the robot with the training box, and
labelled with `reflex.vision.make_labels` (`z` = that controller's latent, `pos`/`vel` = box state in the camera
frame). That controller is not part of the release, so the frame set is shipped as data. The CNN's heatmap is
supervised with the projected box position; its 16-D latent head is trained alongside and discarded. Everything
after step 1 uses the latent of the controller given with `--controller`.

## Collection configuration

The collection cells use the evaluator's default protocol, i.e. the training-time episode structure (a quarter of
the episodes start with the box already in the arms, automatic re-throws after a miss, a walking thrower, abandoned
throws, perturbations of the held box, random pushes; throws are scored from step 200 of 650), not the CLEAN2
protocol of `scripts/eval.sh`. The configuration is loaded from these files, in this order (later files win):

| File | Contents |
|---|---|
| `configs/train/stage3_collect.env` | Stage 2 recipe, ground-truth box observations, two-stage arm trigger, latent observation (25-frame window, camera frame, 2-frame delay), `BOX_COLOR=random` |
| `configs/train/stage3_throw_canon.env` / `stage3_throw_offset.env` | throw block: 3.0–3.5 m, aimed at the robot / lateral offset 0.30–0.80 m |
| `configs/train/stage3_teacher_boxmix.env` | teacher cells only: per-env box sizes from the Stage 2 training mix (4 × small, mini, largeold) |

The catch rates printed during collection (`[PERTHROW]`) count every throw of this protocol with the evaluator's
strict per-throw criterion (sized for the regular box), so they are far below the CLEAN2 numbers of the paper and
only serve as a reproduction check.

## Reference numbers

Lab run that produced the released encoder vs a full run of `scripts/train_stage3.sh` (Docker image of this
repository, RTX 3090). Training is stochastic and rendering is not deterministic, so the CNN, the features and the
encoders differ slightly between runs; the teacher collections reproduce the lab's rollouts.

| Step | Quantity | Lab run (released encoder) | `scripts/train_stage3.sh` |
|---|---|---|---|
| 1 CNN | held-out flight-window latent error `flt_z_mse_std`, best epoch | 0.2115 (epoch 18 of 20) | 0.2095 (epoch 19) |
| 1 CNN | held-out box position R² (x / y / z) | 0.974 / 0.774 / 0.873 | 0.973 / 0.762 / 0.862 |
| 3 teacher | `[PERTHROW]` canon s11 / offset s11 / canon s12 / offset s12 | 94/262, 50/154, 97/267, 33/152 | identical on our machine (an external run on another GPU gave e.g. 102/275: individual throws differ across GPUs, the rates match) |
| 3 teacher | feature frames per cell | 192k / 96k / 192k / 96k | identical on our machine |
| 4 it0 | best held-out `flt_zw_mse` (576k frames, 961 episodes) | 0.0524 | 0.0509 |
| 4 DAgger 0 | `[PERTHROW]` canon / offset (encoder it0 in the loop) | 102/285 = 35.8 % / 48/171 = 28.1 % | 110/276 = 39.9 % / 39/159 = 24.5 % |
| 4 it1 | best held-out `flt_zw_mse` | 0.0530 | 0.0479 |
| 4 DAgger 1 | `[PERTHROW]` canon / offset (encoder it1 in the loop) | 102/247 = 41.3 % / 44/148 = 29.7 % | 101/274 = 36.9 % / 50/149 = 33.6 % |
| 4 it2 | best held-out `flt_zw_mse` | 0.0379 | 0.0436 |
| 5 eval | CLEAN2 catch success, regular box, seed 8 / seed 9 | 89.1 / 88.4 (paper: 88.8) | 90.3 / 90.9 |

`flt_zw_mse` is the task-weighted, standardised latent error on held-out flight windows (box 0.5–2.5 m away). From
it1 on, the held-out set contains the run's own DAgger rollouts, so these values are not directly comparable between
runs; the closed-loop catch success is. Over seeds 8 and 9 this run reaches 90.6 % (580/640 throws) against 88.8 %
for the released encoder; for scale, the paper's six training seeds ranged from 81.6 to 91.7 % on this cell, and a
rerun of the released encoder with this code gave 90.3 % on seed 8.

## Using another controller

Steps 2–5 roll out the controller given with `--controller` and regress its latent; the CNN of step 1 is not tied to
it, so the released CNN can be reused: `--cnn data/checkpoints/stage3_heatmap_cnn.pt --controller <your Stage 2
checkpoint>`. The per-dimension loss weights (`--zw`) belong to the controller; `reflex.vision.latent_weights --zw_out`
computes them from the decoder's sensitivity to each latent dimension (this was not part of the reproduction check
above). The collection configuration matches the released controller's latent observation (`BOX_LATENT_*` in
`stage3_collect.env`); a controller trained with other latent settings needs them changed there too.
