#!/usr/bin/env bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"  # release: repository root
set -u
cd "$PROTCOMPASS_ROOT/protcompass"
for p in 2709845 2709846 2709847 2709848 2709849 2709850; do while [ -e /proc/$p ]; do sleep 60; done; done
echo "SHARDS DONE $(date -u +%H:%M:%S)"
./run_matched_stage1.sh
OMP_NUM_THREADS=8 $HOME/miniconda3/envs/protein_embeddings/bin/python scripts/sham_variance_audit.py \
  --concepts "amino-acid composition" "burial and packing" "secondary structure content" "contact order"
echo "ALL DONE"
