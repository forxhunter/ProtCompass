#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Usage: run_queue.sh <env> <task> <encoder> [<encoder> ...]   (tasks may be a comma-separated list)
# Runs extract.py sequentially; one log per encoder and task. A failure is logged and the queue moves on.
ENV=$1; TASKS=$2; shift 2
cd "$PROTCOMPASS_ROOT/protcompass/extract"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
case "$ENV" in
  cdc5090)    export PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/extract/pylib_cdc5090;;
  gcpnet)     unset PYTHONPATH;;
  foldvision) export PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/pylib_foldvision;;
  *)          export PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/pylib;;
esac
PY=$HOME/miniconda3/envs/$ENV/bin/python
for enc in "$@"; do
  for task in ${TASKS//,/ }; do
    echo "$(date +%T) start $enc $task"
    $PY extract.py --encoder "$enc" --task "$task" > "logs/${task}_${enc}.log" 2>&1 && echo "$(date +%T) ok    $enc $task" || echo "$(date +%T) FAIL  $enc $task: $(grep -v -i warn logs/${task}_${enc}.log | tail -1 | cut -c1-200)"
  done
done
