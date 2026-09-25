#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Start the contact task as soon as the card is free.
cd "$PROTCOMPASS_ROOT/protcompass" || exit 1
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]; do sleep 60; done
echo "GPU free at $(date +%H:%M:%S); starting T8"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
$HOME/miniconda3/envs/protein_embeddings/bin/python scripts/t8_contact_task.py
echo "T8 finished"
