#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# The training-set-size sweep. The stage-1 manifests cap training at 6000 proteins, and the review asked whether the
# sequence-vs-coordinate ordering is an artifact of that budget. Re-extracting every encoder on the full splits is
# not affordable, so the budget is moved down instead: if the group gap is flat from 25% to 100% of the current
# training rows, the ordering is not a function of the budget over the range we can vary it.
cd "$PROTCOMPASS_ROOT/protcompass/extract"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
T="stability fluorescence ppi_affinity solubility membrane_soluble remote_homology ec_classification subcellular_localization go_mf go_bp"
for f in 0.25 0.5; do
  echo "=== train_frac $f ==="
  $PY score2.py --tasks $T --train-frac $f --part-suffix __frac$f
done
echo "TRAIN SIZE SWEEP COMPLETE"
