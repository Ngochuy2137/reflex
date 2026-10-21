#!/usr/bin/env bash
# Start or reuse the Reflex Compose container and open a shell or run a command.
# usage: docker/run.sh [--gui]         interactive shell
#        docker/run.sh "<command>"     run one command, e.g. docker/run.sh "scripts/eval.sh visual"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set -euo pipefail
if [ ! -d "$ROOT/data" ]; then
  echo 'Missing data directory or broken data link. Link the dataset to reflex/data first.' >&2
  exit 1
fi
mkdir -p "$ROOT/outputs" "$ROOT/logs"
COMPOSE=(docker compose -f "$ROOT/docker/compose.yaml")
GUI=0
if [ "${1:-}" = --gui ]; then
  GUI=1
  shift
  bash "$ROOT/docker/prepare_gui.sh"
  COMPOSE+=(-f "$ROOT/docker/compose.gui.yaml")
fi
# Reuse the existing container, including changes made inside it.
"${COMPOSE[@]}" up -d --no-recreate --no-build --pull never reflex
if [ "$GUI" = 1 ] && ! "${COMPOSE[@]}" exec -T reflex /bin/bash -c 'test -n "$DISPLAY" && test -r "$XAUTHORITY"'; then
  echo 'The existing container has no GUI setup. See the dataset README before recreating it.' >&2
  exit 1
fi
EXEC_ARGS=()
if [ ! -t 0 ] || [ ! -t 1 ]; then
  EXEC_ARGS=(-T)
fi
if [ $# -eq 0 ]; then
  exec "${COMPOSE[@]}" exec "${EXEC_ARGS[@]}" reflex /bin/bash
else
  exec "${COMPOSE[@]}" exec "${EXEC_ARGS[@]}" reflex /bin/bash -lc "$*"
fi
