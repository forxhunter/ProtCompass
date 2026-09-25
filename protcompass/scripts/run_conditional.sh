#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Conditional erasure and the positive control for every task that has embeddings, sharded over encoders.
# Each shard writes its own part file under erasure/parts/, which causal_demand.py concatenates.
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
SHARDS=${SHARDS:-3}
PAR=${PAR:-8}
TASKS=${TASKS:-"ec_classification remote_homology subcellular_localization go_mf go_bp membrane_soluble"}
ALL=$(ls extract/embeddings/stability)

run() {
  local t=$1 h=$2
  local e=$(echo "$ALL" | awk -v h=$h -v s=$SHARDS 'NR%s==h')
  [ -z "$e" ] && return 0
  OUT_SUFFIX="_${t}_${h}" $PY scripts/erasure_robustness.py --tasks "$t" --encoders $e --skip-mlp --control-ranks 8 \
    > logs/cond_${t}_${h}.log 2>&1
}
export -f run; export ALL PY SHARDS

for t in $TASKS; do for ((h=0; h<SHARDS; h++)); do echo "$t $h"; done; done \
  | xargs -P "$PAR" -n 2 bash -c 'run "$0" "$1"'
echo "conditional done"
