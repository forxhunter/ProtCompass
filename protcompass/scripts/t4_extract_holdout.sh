#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Extract the confirmation panel on the held-out downstream cohorts. TASKS is a space-separated task list.
cd "$PROTCOMPASS_ROOT/protcompass/extract" || exit 1
CD=$PROTCOMPASS_ROOT/protcompass/confirmation_downstream
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/pylib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for t in $TASKS; do
  [ -f "$CD/ids_$t.txt" ] || { echo "skip $t: no id file"; continue; }
  for enc in esm2_650m prottrans_t5_xl_u50 ankh_base esm3 esm_if1 proteinmpnn; do
    [ -f "$CD/embeddings/$t/$enc/meta.json" ] && continue
    echo "START $enc/$t"
    $PY extract.py --encoder "$enc" --task "$t" --ids-file "$CD/ids_$t.txt" --out-root "$CD/embeddings" 2>&1 | tail -1
  done
done
echo "T4 shard done: $TASKS"
