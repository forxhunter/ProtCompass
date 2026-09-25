#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Wait for the running erasure waves, then run one more wave over whatever cells are still missing, and stop when
# the sweep is complete or no further progress is possible. Coverage is recomputed from the part files each round,
# so a restart never repeats a finished (task, encoder).
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
while pgrep -f "erasure_v2[.]py" > /dev/null; do sleep 60; done
for round in 1 2 3; do
  missing=$($PY scripts/sweep_status.py --missing-tasks)
  [ -z "$missing" ] && break
  echo "round $round: $missing"
  TASKS="$missing" SHARDS=3 PAR=9 bash scripts/run_v2.sh
done
$PY scripts/sweep_status.py
