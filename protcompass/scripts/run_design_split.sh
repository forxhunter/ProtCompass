#!/usr/bin/env bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Re-score every source on the design/score halves of the test split.
#
# The GPU is shared: another user's AlphaFold job holds 25 of 32 GB, and each CUDA context costs about 0.4 GB on top
# of the feature matrices, so three scoring processes do not fit. Three stages, never more than three processes and
# never more than two of them on the GPU:
#   stage A  three CPU-only processes build the reference descriptor cache (CUDA_VISIBLE_DEVICES= , no CUDA context)
#   stage B  two GPU lanes over every source with residue dimension <= 512
#   stage C  one GPU process for the nine large sources, esm2_3b (2560d) last
# Sources whose output CSV already exists are skipped, so the script is restartable.
set -u
cd "$PROTCOMPASS_ROOT/protcompass"
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
LOG=logs

run_lane() {
  local name=$1; shift
  for s in "$@"; do
    $PY scripts/run_probing_suite_split.py "$s" >> "$LOG/design_split_${name}.log" 2>&1
  done
  echo "lane $name done"
}

feature_lane() {
  local name=$1; shift
  CUDA_VISIBLE_DEVICES= $PY scripts/run_probing_suite_split.py --features-only "$@" \
      >> "$LOG/design_split_feat_${name}.log" 2>&1
  echo "feature lane $name done"
}

feature_lane 1 ref_global &
feature_lane 2 ref_pairhist &
feature_lane 3 ref_backbone ref_coords ref_seq_window ref_aa ref_position ref_length &
wait
echo "stage A complete: reference descriptors cached"

run_lane B1 ref_global ref_backbone ref_coords ref_aa ref_position ref_seq_window \
            foldseek_3di gvp gcpnet_pretrained gcpnet_random dimenet ipa ipa_pretrained mace &
run_lane B2 ref_pairhist pottsmpnn pottsmpnn_ft pottsmpnn_pretrained pottsmpnn_soluble proteinmpnn schnet \
            tfn tfn_pretrained cdconv esm2_8m esm2_35m esm_if1 gearnet gearnet_pretrained &
wait
echo "stage B complete"

run_lane C esm2_150m ankh_base foldvision prostt5 protbert_bfd prottrans_t5_xl_u50 esm1b esm2_650m saprot \
           esm3 esm2_3b
echo "stage C complete"
