#!/usr/bin/env bash
# Single-robot RGB-D catching viewer with Play/Reset and object distance/seed fields.
# Run from the Reflex root inside the GUI-enabled container:
# bash scripts/demo.sh --ckpt STAGE2.pt --detector STAGE3.pt [--distance METERS] [--seed N]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CTRL=$DATA/checkpoints/latent_stage2.pt
DETECTOR=$DATA/checkpoints/visual_encoder_stage3.pt
DISTANCE=""; SEED=""; EXTRA=()
while [ $# -gt 0 ]; do case $1 in
  --ckpt) CTRL=$2; shift 2 ;; --detector) DETECTOR=$2; shift 2 ;;
  --distance) DISTANCE=$2; shift 2 ;; --seed) SEED=$2; shift 2 ;;
  -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
  *) EXTRA+=("$1"); shift ;;
esac; done
need "$CTRL" "$DETECTOR" "$MOTION"
$PY -m reflex.utils.desktop
load_env "$ROOT/configs/eval/clean2_latent.env" "$ROOT/configs/eval/conditions/regular.env"
export G1_URDF_OVERRIDE="$ROOT/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main_hands.urdf"
cd "$ROOT"
$PY -m reflex.eval.per_throw_eval --demo --gui --num_envs 1 \
  --env_id Reflex-Latent-G1-v0 --bm_ckpt "$CTRL" --detector "$DETECTOR" --motion_file "$MOTION" \
  --push_robot 0 --motion_phase 300,860 --episode_steps 640 --eval_start_step 0 \
  --demo_seed "$SEED" --demo_distance "$DISTANCE" --out outputs/demo/last_session.json "${EXTRA[@]}"
