#!/usr/bin/env bash
# Download the Unitree G1 description used by BeyondMimic and add the Reflex hand-collider URDF.
# usage: scripts/setup_assets.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DST=$ROOT/source/whole_body_tracking/whole_body_tracking/assets
URL=https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
if [ ! -d "$DST/unitree_description/urdf/g1" ]; then
  tmp=$(mktemp -d)
  curl -fL -o "$tmp/unitree_description.tar.gz" "$URL"
  tar -xzf "$tmp/unitree_description.tar.gz" -C "$DST"
  rm -rf "$tmp"
fi
# G1 with box-shaped colliders on the rubber hands (used for evaluation; see assets/robots/README.md)
cp "$ROOT/assets/robots/main_hands.urdf" "$DST/unitree_description/urdf/g1/main_hands.urdf"
echo "robot description ready: $DST/unitree_description"
