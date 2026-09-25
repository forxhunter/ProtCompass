#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# The corrected causal experiment across tasks, sharded over encoders.
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
SHARDS=${SHARDS:-4}; PAR=${PAR:-8}
TASKS=${TASKS:-"stability fluorescence solubility"}
ALL=$(ls extract/embeddings/stability)
run() {
  local t=$1 h=$2
  local e=$(echo "$ALL" | awk -v h=$h -v s=$SHARDS 'NR%s==h')
  [ -z "$e" ] && return 0
  OUT_SUFFIX="_${t}_${h}" $PY scripts/erasure_v2.py --tasks "$t" --encoders $e > logs/v2_${t}_${h}.log 2>&1
}
export -f run; export ALL PY SHARDS
for t in $TASKS; do for ((h=0;h<SHARDS;h++)); do echo "$t $h"; done; done | xargs -P "$PAR" -n 2 bash -c 'run "$0" "$1"'
echo "v2 done"
