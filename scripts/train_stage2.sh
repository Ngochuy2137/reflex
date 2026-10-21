#!/usr/bin/env bash
# Stage 2: learn the latent interface.  Two steps (paper Sec. Stage 2):
#   bc   initialise encoder + decoder by behaviour cloning the Stage 1 teacher (+ DAgger rounds)
#   ppo  train encoder and decoder jointly with PPO + AMP, latency randomised 20-60 ms
#
# usage: scripts/train_stage2.sh bc  [--gui] [--num_envs 512] [--teacher CKPT] [--out CKPT] [--seed 11]
#        scripts/train_stage2.sh ppo [--gui] [--init CKPT] [--num_envs 4096] [--max_iterations 600] [--seed 42] [--run_name NAME]
# Headless is the default. --gui needs a working desktop display connection.
# Logging: --logger wandb --log_project_name reflex --wandb_group NAME [--wandb_mode online|offline]
# Defaults reproduce the released controller (data/checkpoints/latent_stage2.pt):
#   bc  writes data/stage2/bc_init.pt (the released data/checkpoints/bc_init_stage2.pt is never overwritten);
#   ppo starts from the released data/checkpoints/bc_init_stage2.pt unless --init points at your own bc output.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
STEP=${1:-}; shift || true
TEACHER=$DATA/checkpoints/teacher_stage1.pt; BC_OUT=$DATA/stage2/bc_init.pt; BC_INIT=$DATA/checkpoints/bc_init_stage2.pt
NENV=""; ITERS=600; SEED=""; RUN=stage2_latent
LOGGER=${REFLEX_LOGGER:-none}; [ "$STEP" != ppo ] || LOGGER=${REFLEX_LOGGER:-tensorboard}
PROJECT=${WANDB_PROJECT:-reflex}; WB_GROUP=${WANDB_RUN_GROUP:-}; WB_MODE=online
VIEW_ARGS=(--headless); BC_VIEW_ARGS=(--headless)
while [ $# -gt 0 ]; do case $1 in
  --teacher) TEACHER=$2; shift 2 ;; --out) BC_OUT=$2; shift 2 ;; --init) BC_INIT=$2; shift 2 ;; --num_envs) NENV=$2; shift 2 ;;
  --max_iterations) ITERS=$2; shift 2 ;; --seed) SEED=$2; shift 2 ;; --run_name) RUN=$2; shift 2 ;;
  --gui) VIEW_ARGS=(); BC_VIEW_ARGS=(--gui); shift ;;
  --logger) LOGGER=$2; shift 2 ;; --log_project_name) PROJECT=$2; shift 2 ;;
  --wandb_group) WB_GROUP=$2; shift 2 ;; --wandb_mode) WB_MODE=$2; shift 2 ;;
  -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
  *) echo "unknown arg $1"; exit 1 ;; esac; done
if [ -n "$NENV" ] && { [[ ! "$NENV" =~ ^[1-9][0-9]*$ ]]; }; then
  echo '--num_envs must be a positive integer' >&2; exit 1
fi
need "$MOTION"
[ -n "$WB_GROUP" ] || WB_GROUP=$RUN
if [ "$STEP" = bc ] && [ "$LOGGER" = tensorboard ]; then
  echo 'BC supports --logger none or wandb' >&2; exit 1
fi
if [ "$STEP" = ppo ] && [ "$LOGGER" = none ]; then
  echo 'PPO supports --logger tensorboard or wandb' >&2; exit 1
fi
cd "$ROOT"
case $STEP in
  bc)
    need "$TEACHER"
    # the teacher is queried with ground-truth box channels during behaviour cloning
    load_env "$ROOT/configs/train/stage2_latent.env" "$ROOT/configs/train/stage2_bc_perception.env"
    setup_logging "$LOGGER" "$PROJECT" "$WB_GROUP" "$WB_MODE" "${RUN}_stage2_bc" stage2_bc
    mkdir -p "$(dirname "$BC_OUT")"
    $PY -m reflex.latent.bc_init "${BC_VIEW_ARGS[@]}" --motion_file "$MOTION" --num_envs "${NENV:-512}" --steps 800 --rounds 5 --epochs 8 \
      --seed "${SEED:-11}" --teacher_ckpt "$TEACHER" --out "$BC_OUT" ;;
  ppo)
    need "$BC_INIT" "$AMP_DATASET"
    load_env "$ROOT/configs/train/stage2_latent.env"
    setup_logging "$LOGGER" "$PROJECT" "$WB_GROUP" "$WB_MODE" "${RUN}_stage2_ppo" stage2_ppo
    $PY -m reflex.latent.train --task Reflex-Latent-G1-v0 "${VIEW_ARGS[@]}" --motion_file "$MOTION" \
      --num_envs "${NENV:-4096}" --max_iterations "$ITERS" --seed "${SEED:-42}" --run_name "$RUN" \
      --experiment_name reflex --logger "$LOGGER" --log_project_name "$PROJECT" --warm_start_ckpt "$BC_INIT" ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
