#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# The positive control alone, for every task, sharded over encoders. Cheap next to the conditional sweep: one base
# fit, one label-subspace erasure and three matched random controls per encoder.
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
SHARDS=${SHARDS:-2}
PAR=${PAR:-5}
TASKS=${TASKS:-"stability fluorescence solubility membrane_soluble remote_homology ec_classification subcellular_localization go_mf go_bp ppi_affinity"}
# The ladder establishes a per-task ruler, so it runs on a fixed eight-encoder subset spanning the panel rather than
# all 33: two sequence transformers, one structure-aware transformer, an inverse-folding model, a graph network, and
# three geometric networks. The reported ruler is the mean over these, and the count is stated with it.
ALL=${ENCODERS:-"esm2_650m
esm1b
prottrans_t5_xl_u50
saprot
esm_if1
proteinmpnn
gearnet_pretrained
gvp"}

run() {
  local t=$1 h=$2
  local e=$(echo "$ALL" | awk -v h=$h -v s=$SHARDS 'NR%s==h')
  [ -z "$e" ] && return 0
  OUT_SUFFIX="_${t}_${h}" $PY scripts/erasure_robustness.py --tasks "$t" --encoders $e --skip-mlp --control-only \
    > logs/control_${t}_${h}.log 2>&1
}
export -f run; export ALL PY SHARDS

for t in $TASKS; do for ((h=0; h<SHARDS; h++)); do echo "$t $h"; done; done \
  | xargs -P "$PAR" -n 2 bash -c 'run "$0" "$1"'
echo "control done"
