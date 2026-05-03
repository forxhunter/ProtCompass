#!/usr/bin/env bash
# run_pipeline.sh — full downstream task benchmark pipeline
#
# Usage:
#   bash downstream_tasks/run_pipeline.sh                # all encoders, all tasks
#   bash downstream_tasks/run_pipeline.sh schnet          # single encoder
#   bash downstream_tasks/run_pipeline.sh schnet remote_homology  # single pair
#
# Prerequisites:
#   - Data downloaded:    PYTHONPATH=. python downstream_tasks/download_data.py --tasks tape deeploc workshop
#   - Env active:         conda activate <your_env>
#   - PYTHONPATH set:     run from repo root

set -euo pipefail

ENCODER="${1:-all}"
TASK="${2:-all}"
SEEDS="0 1 2 3 4"

echo "========================================"
echo "Step 1: Extract embeddings"
echo "  encoder=$ENCODER  task=$TASK"
echo "========================================"
PYTHONPATH=. python downstream_tasks/embed_proteins.py \
    --encoder "$ENCODER" \
    --task "$TASK"

echo ""
echo "========================================"
echo "Step 2: Train probes (5 seeds)"
echo "========================================"
PYTHONPATH=. python downstream_tasks/evaluate_tasks.py \
    --encoder "$ENCODER" \
    --task "$TASK" \
    --seeds $SEEDS

echo ""
echo "========================================"
echo "Step 3: I-family × task correlation"
echo "========================================"
PYTHONPATH=. python downstream_tasks/analyze_ifamily_correlation.py \
    --n-bootstrap 2000

echo ""
echo "All done. Results in downstream_tasks/results/"
echo "Heatmaps and top correlations in downstream_tasks/results/heatmap_*.png"
