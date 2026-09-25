#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# The four tasks whose head needs a GPU, after the main chain releases the card.
cd "$PROTCOMPASS_ROOT/protcompass" || exit 1
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PAT=$(echo cm91bmQ1X2NoYWlu | base64 -d)
while pgrep -f "$PAT" >/dev/null; do sleep 60; done
echo "main chain done, starting the large-head tasks $(date +%H:%M:%S)"
$PY scripts/bighead_holdout.py --tasks go_mf go_bp ec_classification remote_homology 2>&1 | tail -4
echo "bighead complete $(date +%H:%M:%S)"
