#!/usr/bin/env bash
# Evaluate a released controller under the paper's CLEAN2 protocol (one observed throw per environment,
# 3.0-3.5 m static thrower, 640 steps, no curriculum events).
#
# usage: scripts/eval.sh <visual|true_z|teacher> [--condition NAME] [--seed N] [--num_envs N] [--out FILE]
#                        [--ckpt FILE] [--detector FILE] [--overlay FILE.env ...] [extra evaluator args]
#   visual   Stage 3: RGB-D visual encoder in the loop -> latent decoder        (paper Table I "Reflex")
#   true_z   Stage 2: privileged encoder (true latent z) -> latent decoder
#   teacher  Stage 1: state policy with ideal box measurements through its Kalman filter
#   --condition  one of configs/eval/conditions/*.env (default: regular = the 34x26x22 cm training box)
#   --ckpt       controller checkpoint (default: the released one for the mode)
#   --detector   visual encoder checkpoint for `visual` (default: data/checkpoints/visual_encoder_stage3.pt)
#   --overlay    extra KEY=VALUE file applied after the protocol and condition (repeatable)
#   --gui        open the simulator window; headless is the default
#
# Environment:
#   ISAACLAB_PY      Python of your Isaac Lab install (default: python). e.g. "./isaaclab.sh -p" or /isaac-sim/python.sh
#   REFLEX_DATA_DIR  where checkpoints and the rebuilt reference motion live (default: <repo>/data)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${REFLEX_DATA_DIR:-$ROOT/data}
PY=${ISAACLAB_PY:-python}

MODE=${1:-}; shift || true
COND=regular; SEED=8; NENV=320; OUT=""; CKPT_ARG=""; DET_ARG=""; OVERLAYS=(); EXTRA=()
VIEW_ARGS=(--headless)
while [ $# -gt 0 ]; do
  case $1 in
    --condition) COND=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --num_envs) NENV=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --ckpt) CKPT_ARG=$2; shift 2 ;;
    --detector) DET_ARG=$2; shift 2 ;;
    --overlay) OVERLAYS+=("$2"); shift 2 ;;
    --gui) VIEW_ARGS=(--gui); shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

case $MODE in
  visual)  CFG=clean2_latent.env;  TASK=Reflex-Latent-G1-v0;  CKPT=$DATA/checkpoints/latent_stage2.pt
           EXTRA+=(--detector "${DET_ARG:-$DATA/checkpoints/visual_encoder_stage3.pt}") ;;
  true_z)  CFG=clean2_latent.env;  TASK=Reflex-Latent-G1-v0;  CKPT=$DATA/checkpoints/latent_stage2.pt ;;
  teacher) CFG=clean2_teacher.env; TASK=Reflex-Teacher-G1-v0; CKPT=$DATA/checkpoints/teacher_stage1.pt ;;
  *) awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 1 ;;
esac

[ -n "$CKPT_ARG" ] && CKPT=$CKPT_ARG
COND_FILE=$ROOT/configs/eval/conditions/$COND.env
MOTION=$DATA/motions/walk_then_catch_v17_30s.npz
URDF=$ROOT/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main_hands.urdf
for f in "$ROOT/configs/eval/$CFG" "$COND_FILE" "$CKPT" "$MOTION" "$URDF" "${OVERLAYS[@]}"; do
  [ -f "$f" ] || { echo "missing: $f  (see README: Setup)"; exit 1; }
done
[ -n "$OUT" ] || OUT=$ROOT/outputs/eval/${MODE}_${COND}_s${SEED}.json
mkdir -p "$(dirname "$OUT")"

# configuration = protocol file, then the condition overlay (later values win)
set -a
# shellcheck disable=SC1090
source "$ROOT/configs/eval/$CFG"
# shellcheck disable=SC1090
source "$COND_FILE"
for f in "${OVERLAYS[@]}"; do
  # shellcheck disable=SC1090
  source "$f"
done
set +a
export G1_URDF_OVERRIDE=$URDF   # G1 with box-shaped hand colliders (assets/robots/main_hands.urdf)

cd "$ROOT"
$PY -m reflex.eval.per_throw_eval "${VIEW_ARGS[@]}" --num_envs "$NENV" --seed "$SEED" \
  --eval_start_step 0 --episode_steps 640 --push_robot 0 --motion_phase 300,860 \
  --env_id "$TASK" --motion_file "$MOTION" --bm_ckpt "$CKPT" --out "$OUT" "${EXTRA[@]}"
$PY -m reflex.eval.summarize "$OUT" --condition "$COND"
