#!/usr/bin/env bash
# Download the released checkpoints into data/checkpoints and verify their sha256.
#
# usage: scripts/download_checkpoints.sh
#   REFLEX_GH_REPO  GitHub repository holding the release (default: ToberJ/reflex)
#   REFLEX_TAG      release tag (default: v1.0.0)
# Uses the GitHub CLI when it is installed and logged in (needed for private repositories), otherwise curl.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DST=${REFLEX_DATA_DIR:-$ROOT/data}/checkpoints
REPO=${REFLEX_GH_REPO:-ToberJ/reflex}; TAG=${REFLEX_TAG:-v1.0.0}
mkdir -p "$DST"
FILES=$(awk '{print $2}' "$ROOT/assets/checkpoints.sha256")
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  for f in $FILES; do gh release download "$TAG" -R "$REPO" -p "$f" -D "$DST" --clobber; done
else
  for f in $FILES; do curl -fL -o "$DST/$f" "https://github.com/$REPO/releases/download/$TAG/$f"; done
fi
(cd "$DST" && sha256sum -c "$ROOT/assets/checkpoints.sha256")
