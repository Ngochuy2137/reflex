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
