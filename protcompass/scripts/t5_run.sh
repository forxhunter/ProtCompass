#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
cd "$PROTCOMPASS_ROOT/protcompass" || exit 1
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
while pgrep -f "[t]4_extract_holdout.sh" >/dev/null; do sleep 30; done
echo "T4 complete: $(find confirmation_downstream/embeddings -name meta.json | wc -l) of 48"
export OMP_NUM_THREADS=8
$PY scripts/addback_holdout.py --tasks membrane_soluble subcellular_localization fluorescence stability remote_homology
echo "T5 finished"
