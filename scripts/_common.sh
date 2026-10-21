# Shared helpers for the Reflex launch scripts (sourced, not executed).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${REFLEX_DATA_DIR:-$ROOT/data}
PY=${ISAACLAB_PY:-python}                 # Isaac Lab Python, e.g. "./isaaclab.sh -p" or /isaac-sim/python.sh
MOTION=$DATA/motions/walk_then_catch_v17_30s.npz
export AMP_DATASET=${AMP_DATASET:-$DATA/amp/amp_dataset.pt}

# load_env FILE...  -- export KEY=VALUE lines (later files override earlier ones)
load_env () {
  local f
  for f in "$@"; do
    [ -f "$f" ] || { echo "missing config: $f"; exit 1; }
    set -a
    # shellcheck disable=SC1090
    source "$f"
    set +a
  done
}

need () { local f; for f in "$@"; do [ -f "$f" ] || { echo "missing: $f  (see README: Setup)"; exit 1; }; done; }

# Call after load_env so legacy WANDB_MODE=offline does not override the CLI.
setup_logging () {
  local logger=$1 project=$2 group=$3 mode=$4 name=$5 job=$6
  case "$logger" in none|tensorboard|wandb) ;; *) echo "Unsupported logger: $logger" >&2; exit 1 ;; esac
  export REFLEX_LOGGER=$logger
  [ "$logger" = wandb ] || return 0
  case "$mode" in online|offline) ;; *) echo '--wandb_mode must be online or offline' >&2; exit 1 ;; esac
  export WANDB_PROJECT=$project WANDB_RUN_GROUP=$group WANDB_MODE=$mode
  export WANDB_NAME=$name WANDB_JOB_TYPE=$job
  export WANDB_DIR=${WANDB_DIR:-$ROOT/logs/wandb}
  mkdir -p "$WANDB_DIR"
  # RSL-RL reads WANDB_USERNAME for the same account/team setting.
  if [ -n "${WANDB_ENTITY:-}" ]; then export WANDB_USERNAME=$WANDB_ENTITY; fi
}
