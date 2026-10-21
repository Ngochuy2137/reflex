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
# Keep the same inode so an existing read-only bind mount sees refreshed cookies.
touch "$COOKIE"
chmod 600 "$COOKIE"
: > "$COOKIE"
printf '%s\n' "$RECORDS" \
  | sed 's/^..../ffff/' \
  | xauth -f "$COOKIE" nmerge -
if [ ! -s "$COOKIE" ]; then
  echo 'No display cookie found. Check DISPLAY and XAUTHORITY on the host.' >&2
  exit 1
fi
echo 'Prepared the private display cookie under .docker-runtime.'
