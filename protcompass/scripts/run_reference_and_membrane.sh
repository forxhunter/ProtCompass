#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Two scoring jobs the round-1 review asked for, run one after the other so they do not contend for the GPU.
#   1. membrane/soluble for all 33 encoders. It was the one extracted task with no downstream table, while the
#      interventional sweep already covered it, which the review flagged as an inconsistency.
#   2. the non-learned downstream reference (amino-acid composition and length, no encoder) on all ten tasks.
cd "$PROTCOMPASS_ROOT/protcompass/extract"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
echo "=== membrane/soluble, all encoders ==="
$PY score2.py --tasks membrane_soluble
echo
echo "=== non-learned reference, all tasks ==="
$PY score2.py --tasks stability fluorescence ppi_affinity solubility membrane_soluble remote_homology \
    ec_classification subcellular_localization go_mf go_bp --encoders ref_composition --part-suffix __reference
echo
echo "REFERENCE AND MEMBRANE COMPLETE"
