#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"  # release: repository root
# GPU tasks of the full grid on this machine, five at a time, then two completion passes (each shard resumes).
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
run_one() {
  IFS=$'\t' read -r t cs tg <<< "$1"
  IFS='|' read -ra C <<< "$cs"
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY scripts/addback_named_pairs.py \
    --pairs erasure/named_pair_grid_full.csv --tasks "$t" --concepts "${C[@]}" \
    --out erasure/named_pairs_full/${t}__${tg}.csv >> logs/gridL_${t}__${tg}.log 2>&1
  echo "$(date -Is) done $t $tg" >> logs/gridL_progress.log
}
export -f run_one
export PY
for pass in 1 2 3; do
  echo "$(date -Is) gpu pass $pass" >> logs/gridL_progress.log
  tr '\n' '\0' < erasure/shards_gpu_local.tsv | xargs -0 -P 5 -I{} bash -c 'run_one "$@"' _ {}
done
echo "$(date -Is) GPU GRID DONE" >> logs/gridL_progress.log
