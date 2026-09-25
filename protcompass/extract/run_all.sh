#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Master runner: everything still missing, in priority order, with a fixed number of concurrent workers.
# Priority: probing (the IIP profiles the IAH test needs) before the downstream tasks.
# Usage: run_all.sh [n_workers]   default 3
cd "$PROTCOMPASS_ROOT/protcompass/extract"
N=${1:-3}
LOG=logs/run_all.log
: > jobs.txt

# One line per job: <env> <encoder> <task>. Python decides what is still missing.
$HOME/miniconda3/envs/protein_embeddings/bin/python - <<'EOF' >> jobs.txt
import sys, pathlib
sys.path.insert(0, ".")
from common import SPEC
ROOT = pathlib.Path(__import__("os").environ["PROTCOMPASS_ROOT"]) / "protcompass"
# EGNN is excluded by the input audit: its output does not depend on the coordinates it declares.
SKIP = {"egnn", "geobpe"}
done_probe = {p.parent.name for p in (ROOT / "probing_embeddings").glob("*/meta.json")}
for k, s in SPEC.items():
    if k not in SKIP and k not in done_probe:
        print(s.env, k, "probing")
for task in ("stability", "ppi_affinity", "fluorescence", "solubility",
             "remote_homology", "ec_classification", "go_mf", "go_bp", "subcellular_localization", "membrane_soluble"):
    have = {p.name for p in (pathlib.Path("embeddings") / task).iterdir() if (p / "meta.json").exists()} if (pathlib.Path("embeddings") / task).exists() else set()
    for k, s in SPEC.items():
        if k not in SKIP and k not in have:
            print(s.env, k, task)
EOF

echo "$(date +%T) $(wc -l < jobs.txt) jobs queued, $N workers" | tee -a $LOG

run_one() {
  read -r env enc task <<< "$1"
  case "$env" in
    cdc5090)    export PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/extract/pylib_cdc5090;;
    gcpnet)     unset PYTHONPATH;;
    foldvision) export PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/pylib_foldvision;;
    *)          export PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/pylib;;
  esac
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  local py=$HOME/miniconda3/envs/$env/bin/python
  local log="logs/${task}_${enc}.log"
  # One retry after a pause: the GPU is shared, so an out-of-memory failure is usually transient.
  for attempt in 1 2; do
    if $py extract.py --encoder "$enc" --task "$task" > "$log" 2>&1; then
      echo "$(date +%T) ok    $enc $task" >> $LOG; return
    fi
    if grep -qi "out of memory" "$log" && [ $attempt = 1 ]; then sleep 420; else break; fi
  done
  echo "$(date +%T) FAIL  $enc $task: $(grep -v -i warn "$log" | tail -1 | cut -c1-160)" >> $LOG
}
export -f run_one
export LOG

# Group by env so that models sharing an environment do not reload it concurrently; -P bounds GPU pressure.
xargs -a jobs.txt -d '\n' -I{} -P "$N" bash -c 'run_one "$@"' _ {}
echo "$(date +%T) all jobs finished" | tee -a $LOG
