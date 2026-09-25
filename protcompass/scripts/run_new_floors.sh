#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# The two probing sources added after round 3, run when the GPU has room. Another project's AlphaFold job holds
# most of the card, so this waits for space rather than competing with it.
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
need=6000
for i in $(seq 1 240); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge "$need" ] && break
  sleep 60
done
$PY scripts/run_probing_suite.py ref_backbone_w0 ref_seq_global
