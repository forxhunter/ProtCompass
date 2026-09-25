#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# The matched-capacity re-probe the reviews ask for, plus the two sources added after round 3. Runs on the CPU:
# another project's AlphaFold job holds the card, and the probes are small enough that waiting for the GPU costs
# more than not using it.
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=10   # another project restarted its AlphaFold job on the card

echo "=== the two sources added after round 3 ==="
$PY scripts/run_probing_suite.py ref_backbone_w0 ref_seq_global

echo
echo "=== every source at a common width of 116 ==="
# 116 is the median encoder width in the panel and is narrower than six of the ten floors, so the projection
# costs the floors more than it costs the encoders, which is the direction that makes the test honest.
PROJECT_K=116 $PY scripts/run_probing_suite.py
echo "CAPACITY MATCHED COMPLETE"
