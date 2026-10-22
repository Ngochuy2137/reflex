#!/usr/bin/env bash
# Copy only the current display cookie for the local GUI container.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${DISPLAY:-}" ]; then
  echo 'DISPLAY is missing. Run this from a terminal in your desktop session.' >&2
  exit 1
fi
command -v xauth >/dev/null || { echo 'Install xauth on the host first.' >&2; exit 1; }
AUTH_ARGS=()
if [ -n "${XAUTHORITY:-}" ]; then
  AUTH_ARGS=(-f "$XAUTHORITY")
fi
RECORDS=$(xauth "${AUTH_ARGS[@]}" nlist "$DISPLAY")
if [ -z "$RECORDS" ]; then
  echo 'No display cookie found. Check DISPLAY and XAUTHORITY on the host.' >&2
  exit 1
fi
umask 077
mkdir -p "$ROOT/.docker-runtime"
chmod 700 "$ROOT/.docker-runtime"
COOKIE="$ROOT/.docker-runtime/Xauthority"
# xauth replaces its output file. Build a temporary file, then copy into the
# existing inode so the running container's read-only bind mount sees the update.
WORK_DIR=$(mktemp -d "$ROOT/.docker-runtime/xauth.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
touch "$WORK_DIR/Xauthority"
printf '%s\n' "$RECORDS" \
  | sed 's/^..../ffff/' \
  | xauth -f "$WORK_DIR/Xauthority" nmerge -
if [ ! -s "$WORK_DIR/Xauthority" ]; then
  echo 'No display cookie found. Check DISPLAY and XAUTHORITY on the host.' >&2
  exit 1
fi
touch "$COOKIE"
chmod 600 "$COOKIE"
cat "$WORK_DIR/Xauthority" > "$COOKIE"
echo 'Prepared the private display cookie under .docker-runtime.'
