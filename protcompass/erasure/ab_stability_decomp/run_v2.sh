#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"  # release: repository root
# Pass 1 (addendum rows A and B, C1 then C0) for all encoders, then pass 2 (other PRESPEC fits with per-parent
# scores), encoders the main run had not reached first. 3 BLAS threads: the setting that reproduces the released rows.
cd "$PROTCOMPASS_ROOT/protcompass/scripts"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
LOG=../erasure/ab_stability_decomp
ALL="gcpnet_random gcpnet_pretrained gvp tfn_pretrained tfn ipa_pretrained mace pottsmpnn pottsmpnn_ft ipa pottsmpnn_soluble proteinmpnn schnet dimenet pottsmpnn_pretrained cdconv esm2_8m esm2_35m gearnet_pretrained esm_if1 gearnet esm2_150m ankh_base foldvision prostt5 protbert_bfd prottrans_t5_xl_u50 esm2_650m saprot esm1b esm3 esm2_3b"
NEW="prostt5 protbert_bfd prottrans_t5_xl_u50 esm2_650m saprot esm1b esm3 esm2_3b"
echo "pass1 start $(date +%T)" >> $LOG/log_v2_times.txt
nice -n 5 $PY ab_stability_decomp_v2.py --pass 1 --encoders $ALL > $LOG/log_v2_p1.txt 2>&1
echo "pass1 end $(date +%T) rc=$?" >> $LOG/log_v2_times.txt
nice -n 5 $PY ab_stability_decomp_v2.py --pass 2 --encoders $NEW $ALL > $LOG/log_v2_p2.txt 2>&1
echo "pass2 end $(date +%T) rc=$?" >> $LOG/log_v2_times.txt
