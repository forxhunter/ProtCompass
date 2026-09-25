#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# E5 driver: the four equivariant encoders with frame-projected vector channels, in the env of the released run.
# Usage: run_vector_frame.sh <queue>   queue "pe" = gvp, mace (protein_embeddings); "gcp_r" / "gcp_p" = GCPNet random / pretrained
cd "$PROTCOMPASS_ROOT/protcompass/extract"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # env encoder
  if [ "$1" = gcpnet ]; then unset PYTHONPATH; else export PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/pylib; fi
  echo "$(date +%T) start $2"
  $HOME/miniconda3/envs/$1/bin/python extract_vector_frame.py --encoder "$2" > "logs/vframe_$2.log" 2>&1 && echo "$(date +%T) ok    $2" || echo "$(date +%T) FAIL  $2: $(grep -v -i warn logs/vframe_$2.log | tail -1 | cut -c1-200)"
}
case "$1" in
  pe)    run protein_embeddings gvp; run protein_embeddings mace;;
  gcp_r) run gcpnet gcpnet_random;;
  gcp_p) run gcpnet gcpnet_pretrained;;
esac
