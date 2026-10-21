#!/usr/bin/env bash
# Stage 3: learn the visual encoder (RGB-D image history + camera ego-motion -> latent z) for the fixed Stage 2
# controller (paper Sec. Stage 3). Steps (times on one RTX 3090):
#   1 cnn      per-frame RGB-D CNN + supervised box heatmap on the rendered frame set, colour augmentation  (~70 min)
#   2 featnet  wrap that CNN into a feature extractor for collection                                       (seconds)
#   3 teacher  roll out the controller on its privileged latent with the encoder in shadow mode; log the
#              per-frame features and the latent z* as labels: seeds 11, 12 x {canon 320 envs, offset 160 envs}  (~15 min)
#   4 dagger   train the temporal encoder (GRU) it0 on the teacher features; then per round k: roll out with
#              it<k> driving the controller (seed 7, canon + offset), retrain it<k+1> on all features so far   (~20 min)
#   5 eval     CLEAN2 evaluation of the final encoder: scripts/eval.sh visual --seed 8                           (~4 min)
# Every output is written under a temporary name and renamed when its step succeeds; a step whose output exists is
# skipped, so rerunning the same command resumes. Outputs are reused by name: use a new --run after changing inputs.
#
# usage: scripts/train_stage3.sh [--run NAME] [--out DIR] [--frames DIR] [--labels FILE] [--controller CKPT]
#                                [--cnn CKPT] [--zw NPZ] [--rounds N] [--no_eval] [--print_config] [--gui] [--num_envs N]
#   --gui           open simulator windows during collection and evaluation; default is headless
#   --num_envs      override both collection block sizes and the final evaluation size
#   --run           outputs go to $DATA/stage3/<run>/ (default: default)
#   --out           output directory (overrides --run)
#   --frames        rendered frame set of step 1: chunk_*.npz + label file (default: $DATA/stage3/frames/v10_s7_rgbd)
#   --labels        label file inside --frames (default: labels_v10.npz)
#   --controller    Stage 2 controller (default: data/checkpoints/latent_stage2.pt)
#   --cnn           use this heatmap CNN and skip step 1 (e.g. data/checkpoints/stage3_heatmap_cnn.pt)
#   --zw            per-dimension latent loss weights for --controller (default: data/checkpoints/stage3_latent_weights.npz)
#   --rounds        DAgger rounds (default: 2); the final encoder is it<rounds>, copied to <out>/visual_encoder.pt
#   --no_eval       skip step 5
#   --print_config  print the effective configuration of the four collection cells and exit
# Defaults reproduce the released encoder (data/checkpoints/visual_encoder_stage3.pt). See docs/STAGE3.md.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RUN=default; OUT=""; FRAMES=$DATA/stage3/frames/v10_s7_rgbd; LABELS=labels_v10.npz
CTRL=$DATA/checkpoints/latent_stage2.pt; CNN=""; ZW=$DATA/checkpoints/stage3_latent_weights.npz
ROUNDS=2; EVAL=1; PRINT=0
ENV_COUNT=""; VIEW_ARGS=(--headless); EVAL_ARGS=()
while [ $# -gt 0 ]; do case $1 in
  --run) RUN=$2; shift 2 ;; --out) OUT=$2; shift 2 ;; --frames) FRAMES=$2; shift 2 ;; --labels) LABELS=$2; shift 2 ;;
  --controller) CTRL=$2; shift 2 ;; --cnn) CNN=$2; shift 2 ;; --zw) ZW=$2; shift 2 ;; --rounds) ROUNDS=$2; shift 2 ;;
  --no_eval) EVAL=0; shift ;; --print_config) PRINT=1; shift ;;
  --gui) VIEW_ARGS=(--gui); EVAL_ARGS+=(--gui); shift ;;
  --num_envs) ENV_COUNT=$2; shift 2 ;;
  -h|--help) awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 0 ;;
  *) echo "unknown arg $1"; awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 1 ;; esac; done
if [ -n "$ENV_COUNT" ] && { [[ ! "$ENV_COUNT" =~ ^[1-9][0-9]*$ ]]; }; then
  echo '--num_envs must be a positive integer' >&2; exit 1
fi
[ -n "$OUT" ] || OUT=$DATA/stage3/$RUN
abspath () { case $1 in ""|/*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }   # the script works from the repo root
OUT=$(abspath "$OUT"); FRAMES=$(abspath "$FRAMES"); CTRL=$(abspath "$CTRL"); CNN=$(abspath "$CNN"); ZW=$(abspath "$ZW")
CFG=$ROOT/configs/train
URDF=$ROOT/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main_hands.urdf
TEACHER_SEEDS="11 12"; DAGGER_SEED=7
declare -A NENV=([canon]=320 [offset]=160)     # environments per throw block
if [ -n "$ENV_COUNT" ]; then
  NENV[canon]=$ENV_COUNT; NENV[offset]=$ENV_COUNT
  EVAL_ARGS+=(--num_envs "$ENV_COUNT")
fi

# configuration of a collection cell, in load order (later files override earlier ones):
#   kind teacher|dagger, block canon|offset
layers () {
  echo "$CFG/stage3_collect.env $CFG/stage3_throw_$2.env"
  if [ "$1" = teacher ]; then echo "$CFG/stage3_teacher_boxmix.env"; fi
}

if [ $PRINT = 1 ]; then
  for kind in teacher dagger; do for blk in canon offset; do
    files=$(layers $kind $blk | xargs)
    echo "== $kind $blk: ${files//$ROOT\//}"
    ( load_env $files; export G1_URDF_OVERRIDE=$URDF
      for k in $(grep -vh '^#' $files | cut -d= -f1 | sort -u) G1_URDF_OVERRIDE; do echo "$k=${!k}"; done )
  done; done
  exit 0
fi

need "$CTRL" "$ZW" "$MOTION" "$URDF"
mkdir -p "$OUT/collect" "$OUT/logs"
cd "$ROOT"
log () { echo "[stage3 $(date +%H:%M)] $*"; }

# ---- 1) heatmap CNN (skipped with --cnn)
if [ -z "$CNN" ]; then
  CNN=$OUT/heatmap_cnn.pt
  if [ -f "$CNN" ]; then log "1 cnn: exists, skipped ($CNN)"; else
    [ -f "$FRAMES/$LABELS" ] && ls "$FRAMES"/chunk_*.npz > /dev/null 2>&1 \
      || { echo "missing frame set: $FRAMES/{chunk_*.npz,$LABELS} (scripts/download_stage3_frames.sh; --cnn CKPT skips step 1)"; exit 1; }
    log "1 cnn: training on $FRAMES (log: $OUT/logs/heatmap_cnn.log)"; t0=$SECONDS
    $PY -m reflex.vision.train_detector --frames_dir "$FRAMES" --labels "$LABELS" --out "$CNN.partial" \
      --epochs 20 --bs 32 --w_kl 0 --w_flight 8 --flight_select --arch kp --depth --w_uv 1.0 --color_aug 0.7 \
      > "$OUT/logs/heatmap_cnn.log" 2>&1 || { log "1 cnn: FAILED, see $OUT/logs/heatmap_cnn.log"; exit 1; }
    grep -aq "\[det\] DONE" "$OUT/logs/heatmap_cnn.log" || { log "1 cnn: no DONE line, see $OUT/logs/heatmap_cnn.log"; exit 1; }
    mv "$CNN.partial" "$CNN"
    log "1 cnn: done in $(( (SECONDS - t0) / 60 )) min: $(grep -a '\[det\] DONE' "$OUT/logs/heatmap_cnn.log" | cut -c1-160)"
  fi
else
  need "$CNN"; log "1 cnn: using $CNN"
fi

# ---- 2) feature extractor
FEATNET=$OUT/feature_net.pt
if [ -f "$FEATNET" ]; then log "2 featnet: exists, skipped"; else
  $PY -m reflex.vision.make_feature_net --cnn "$CNN" --out "$FEATNET.partial" > "$OUT/logs/feature_net.log" 2>&1 \
    || { log "2 featnet: FAILED, see $OUT/logs/feature_net.log"; exit 1; }
  mv "$FEATNET.partial" "$FEATNET"; log "2 featnet: $(grep -a '\[featnet\]' "$OUT/logs/feature_net.log" | cut -c1-80)"
fi

# collect KIND BLOCK SEED ENCODER TAG: one rollout (evaluator default protocol) logging per-frame features + z* labels
# to $OUT/collect/TAG.npz.  teacher: the controller drives on its privileged latent, the encoder runs in shadow mode;
# dagger: the encoder's z drives the controller.
SETS=""
collect () {
  local kind=$1 blk=$2 seed=$3 enc=$4 tag=$5 c=$OUT/collect/$5 extra=()
  SETS="$SETS,$c.npz"
  if [ -f "$c.npz" ]; then log "  $tag: exists, skipped"; return; fi
  [ "$kind" = teacher ] && extra=(--det_shadow "$c.shadow.npz")
  log "  $tag: ${NENV[$blk]} envs, seed $seed (log: $OUT/logs/$tag.log)"; local t0=$SECONDS
  ( load_env $(layers "$kind" "$blk"); export G1_URDF_OVERRIDE=$URDF
    $PY -m reflex.eval.per_throw_eval "${VIEW_ARGS[@]}" --num_envs "${NENV[$blk]}" --seed "$seed" --env_id Reflex-Latent-G1-v0 \
      --motion_file "$MOTION" --bm_ckpt "$CTRL" --detector "$enc" "${extra[@]}" \
      --save_feats "$c.partial.npz" --out "$c.json" ) > "$OUT/logs/$tag.log" 2>&1 \
    || { log "  $tag: FAILED, see $OUT/logs/$tag.log"; exit 1; }
  grep -aq "^\[PERTHROW\] catch rate" "$OUT/logs/$tag.log" && [ -f "$c.partial.npz" ] \
    || { log "  $tag: incomplete, see $OUT/logs/$tag.log"; exit 1; }
  mv "$c.partial.npz" "$c.npz"
  log "  $tag: done in $(( (SECONDS - t0) / 60 )) min | $(grep -a '^\[PERTHROW\] catch rate' "$OUT/logs/$tag.log" | tail -1)"
}

# train_encoder N: GRU it<N> on all feature sets collected so far
train_encoder () {
  local enc=$OUT/encoder_it$1.pt lg=$OUT/logs/encoder_it$1.log
  if [ -f "$enc" ]; then log "  it$1: exists, skipped"; return; fi
  log "  it$1: training on $(echo "${SETS#,}" | tr ',' '\n' | wc -l) feature sets (log: $lg)"; local t0=$SECONDS
  $PY -m reflex.vision.train_seqdet --frames_dir none --labels none --feat_sets "${SETS#,}" --kp "$CNN" --zdim 8 \
    --zw "$ZW" --out "$enc.partial" > "$lg" 2>&1 || { log "  it$1: FAILED, see $lg"; exit 1; }
  grep -aq "\[seq\] DONE" "$lg" || { log "  it$1: no DONE line, see $lg"; exit 1; }
  mv "$enc.partial" "$enc"
  log "  it$1: done in $(( (SECONDS - t0) / 60 )) min | $(grep -a '\[seq\] DONE' "$lg" | sed 's/ -> .*//' | cut -c1-170)"
}

# ---- 3) teacher-driven feature collection
log "3 teacher collection"
for sd in $TEACHER_SEEDS; do for blk in canon offset; do
  collect teacher $blk $sd "$FEATNET" teacher_${blk}_s$sd
done; done

# ---- 4) temporal encoder + DAgger rounds
log "4 temporal encoder + $ROUNDS DAgger rounds"
for ((k = 0; k <= ROUNDS; k++)); do
  train_encoder $k
  if [ $k -lt $ROUNDS ]; then for blk in canon offset; do
    collect dagger $blk $DAGGER_SEED "$OUT/encoder_it$k.pt" dagger${k}_${blk}_s$DAGGER_SEED
  done; fi
done
cp "$OUT/encoder_it$ROUNDS.pt" "$OUT/visual_encoder.pt"
log "final encoder: $OUT/visual_encoder.pt (= encoder_it$ROUNDS.pt)"

# ---- 5) CLEAN2 evaluation (paper protocol, regular box, seed 8)
if [ $EVAL = 1 ]; then
  EV=$OUT/eval/visual_regular_s8.json
  if [ -f "$EV" ]; then log "5 eval: exists, skipped ($EV)"; else
    log "5 eval: CLEAN2 regular, seed 8 (log: $OUT/logs/eval_regular_s8.log)"; t0=$SECONDS
    "$ROOT/scripts/eval.sh" visual "${EVAL_ARGS[@]}" --ckpt "$CTRL" --detector "$OUT/visual_encoder.pt" --seed 8 --out "$EV.partial" \
      > "$OUT/logs/eval_regular_s8.log" 2>&1 || { log "5 eval: FAILED, see $OUT/logs/eval_regular_s8.log"; exit 1; }
    mv "$EV.partial" "$EV"
    log "5 eval: done in $(( (SECONDS - t0) / 60 )) min"
  fi
  grep -a "\[summary\]" "$OUT/logs/eval_regular_s8.log" || true
fi
