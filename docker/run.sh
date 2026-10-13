#!/usr/bin/env bash
# Start the Reflex container with the GPU and ./data, ./outputs, ./logs mounted.
# usage: docker/run.sh                 interactive shell
#        docker/run.sh "<command>"     run one command, e.g. docker/run.sh "scripts/eval.sh visual"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/data" "$ROOT/outputs" "$ROOT/logs"
ARGS=(--rm --gpus all --network host -v "$ROOT/data:/reflex/data" -v "$ROOT/outputs:/reflex/outputs" -v "$ROOT/logs:/reflex/logs")
if [ $# -eq 0 ]; then
  exec docker run -it "${ARGS[@]}" reflex
else
  # files written to the mounted folders are handed back to the host user afterwards
  exec docker run "${ARGS[@]}" reflex -lc "$*; rc=\$?; chown -R $(id -u):$(id -g) /reflex/data /reflex/outputs /reflex/logs; exit \$rc"
fi
