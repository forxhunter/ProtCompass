#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Check 2: does the causal effect survive a nonlinear downstream head? LEACE removes linearly recoverable information,
# so a linear head could overstate the damage. fit_score_mlp handles regression and binary labels, which is the set of
# tasks whose conditional effects the document actually reports, so that is what this runs.
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
PAR=${PAR:-4}
TASKS=${TASKS:-"stability fluorescence solubility membrane_soluble"}
ENC=${ENCODERS:-"esm2_650m esm1b prottrans_t5_xl_u50 saprot esm_if1 proteinmpnn gearnet_pretrained gvp"}

run() {
  local t=$1 e=$2
  OUT_SUFFIX="_mlp_${t}_${e}" $PY scripts/erasure_robustness.py --tasks "$t" --encoders "$e" --control-ranks 8 \
    > logs/mlp_${t}_${e}.log 2>&1
}
export -f run; export PY

for t in $TASKS; do for e in $ENC; do echo "$t $e"; done; done | xargs -P "$PAR" -n 2 bash -c 'run "$0" "$1"'
echo "mlp check done"
