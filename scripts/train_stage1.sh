#!/usr/bin/env bash
# Stage 1: train the state-based teacher (PPO + AMP) on the box-catching task.
#
# usage: scripts/train_stage1.sh [--warm_start CKPT] [--num_envs 4096] [--max_iterations 600] [--seed 42] [--run_name NAME]
#                                [further args are passed to reflex.teacher.train, e.g. --resume True --load_run RUN --checkpoint model_599.pt]
#
# The released teacher (data/checkpoints/teacher_stage1.pt) was produced by a sequence of warm-started
# runs; configs/train/stage1_teacher.env is the configuration of the final run (600 iterations, 4096
# envs, seed 42).  Use --warm_start to continue from a checkpoint; without it, PPO starts from scratch
# with this configuration (not the procedure used for the paper -- see docs/REPRODUCE.md).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
WARM=""; NENV=4096; ITERS=600; SEED=42; RUN=stage1_teacher; EXTRA=()
while [ $# -gt 0 ]; do case $1 in
  --warm_start) WARM=$2; shift 2 ;; --num_envs) NENV=$2; shift 2 ;; --max_iterations) ITERS=$2; shift 2 ;;
  --seed) SEED=$2; shift 2 ;; --run_name) RUN=$2; shift 2 ;; *) EXTRA+=("$1"); shift ;; esac; done
need "$MOTION" "$AMP_DATASET"
load_env "$ROOT/configs/train/stage1_teacher.env"
cd "$ROOT"
$PY -m reflex.teacher.train --task Reflex-Teacher-G1-v0 --headless --motion_file "$MOTION" \
  --num_envs "$NENV" --max_iterations "$ITERS" --seed "$SEED" --run_name "$RUN" \
  --experiment_name reflex --logger tensorboard ${WARM:+--warm_start_ckpt "$WARM"} "${EXTRA[@]}"
