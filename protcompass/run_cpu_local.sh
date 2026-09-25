#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"  # release: repository root
# The CPU shards of the full grid that the second machine did not start. Waits for the PPI cross-validation to
# release the cores, then runs them four at a time, then one completion pass.
cd "$PROTCOMPASS_ROOT/protcompass"
P=$(echo cHBpX2N2L2VuY18= | base64 -d)
while pgrep -f "$P" > /dev/null; do sleep 60; done
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
run_one() {
  IFS=$'\t' read -r t cs tg <<< "$1"
  IFS='|' read -ra C <<< "$cs"
  env OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 CUDA_VISIBLE_DEVICES= $PY scripts/addback_named_pairs.py \
    --pairs erasure/named_pair_grid_full.csv --tasks "$t" --concepts "${C[@]}" \
    --out erasure/named_pairs_full/${t}__${tg}.csv >> logs/gridL_${t}__${tg}.log 2>&1
  echo "$(date -Is) done $t $tg" >> logs/gridL_progress.log
}
export -f run_one
export PY
for pass in 1 2; do
  echo "$(date -Is) local cpu pass $pass" >> logs/gridL_progress.log
  tr '\n' '\0' < erasure/shards_cpu_local.tsv | xargs -0 -P 4 -I{} bash -c 'run_one "$@"' _ {}
done
echo "$(date -Is) LOCAL CPU GRID DONE" >> logs/gridL_progress.log
