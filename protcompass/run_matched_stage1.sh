#!/usr/bin/env bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"  # release: repository root
# The variance-matched null on the released stage-1 cells that carry a naming claim, conditional arm only.
# Written to erasure/parts_matched so the released part files are untouched.
set -u
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
cd "$PROTCOMPASS_ROOT/protcompass"
COMMON="--matched-null --arms conditional --parts parts_matched"

OMP_NUM_THREADS=8 $PY scripts/erasure_v2.py $COMMON \
  --tasks fluorescence --concepts "amino-acid composition" "burial and packing"
OMP_NUM_THREADS=8 $PY scripts/erasure_v2.py $COMMON \
  --tasks membrane_soluble solubility subcellular_localization go_mf stability \
  --concepts "amino-acid composition"
OMP_NUM_THREADS=8 $PY scripts/erasure_v2.py $COMMON \
  --tasks ppi_affinity --concepts "secondary structure content" "contact order"
echo "MATCHED STAGE1 DONE"
