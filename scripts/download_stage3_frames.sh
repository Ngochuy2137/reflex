#!/usr/bin/env bash
# Download the Stage 3 rendered frame set (96,000 RGB-D frames, 2.1 GB, 13 files) into
# data/stage3/frames/v10_s7_rgbd and verify the sha256 of every file.  Only step 1 of
# scripts/train_stage3.sh (heatmap CNN) needs it; `--cnn data/checkpoints/stage3_heatmap_cnn.pt` skips that step.
#
# usage: scripts/download_stage3_frames.sh
#   REFLEX_GH_REPO   GitHub repository holding the release (default: ToberJ/reflex)
#   REFLEX_DATA_TAG  release tag of the frame set (default: stage3-frames-v1)
# Uses the GitHub CLI when it is installed and logged in, otherwise curl.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DST=${REFLEX_DATA_DIR:-$ROOT/data}/stage3/frames/v10_s7_rgbd
REPO=${REFLEX_GH_REPO:-ToberJ/reflex}; TAG=${REFLEX_DATA_TAG:-stage3-frames-v1}
mkdir -p "$DST"
FILES=$(awk '{print $2}' "$ROOT/assets/stage3_frames.sha256")
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  for f in $FILES; do gh release download "$TAG" -R "$REPO" -p "$f" -D "$DST" --clobber; done
else
  for f in $FILES; do curl -fL -C - -o "$DST/$f" "https://github.com/$REPO/releases/download/$TAG/$f"; done
fi
(cd "$DST" && sha256sum -c "$ROOT/assets/stage3_frames.sha256")
