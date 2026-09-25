#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Cohort B, end to end, after scripts/confirmation2_freeze.py and confirmation_build_labels.py have run.
#   1 extraction of the frozen panel on the cohort      extract.py --task probing
#   2 merge with the released fitting draw               confirmation_merge.py
#   3 the released scorer on the merged file             run_probing_suite.py        (H1-H4)
#   4 atom-matched floors on the merged file             atom_matched_floors.py      (H1b)
#   5 the same fits scored on the topology-novel part    confirmation_novelty.py     (H5)
# ATOM_SETS lists the atom sets of the panel's coordinate encoders, read from extract/encoder_atom_audit.csv.
set -e
R=$PROTCOMPASS_ROOT/protcompass
C=$R/confirmation2
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export CONFIRMATION_DIR=$C
cd $R/extract
for e in esm2_650m prottrans_t5_xl_u50 ankh_base esm3 esm_if1 proteinmpnn; do
  for attempt in 1 2 3; do
    [ -f $C/embeddings/$e/meta.json ] && break
    rm -f $C/embeddings/$e/residue_fp16.npy.partial
    $PY extract.py --encoder $e --task probing --probing-labels $C/labels_v44.h5 --probing-out $C/embeddings > $C/logs_extract_$e.log 2>&1 || sleep 120
  done
  echo "$(date +%H:%M) extracted $e: $(tail -1 $C/logs_extract_$e.log)"
done
cd $R/scripts
$PY confirmation_merge.py
PROBING_LABELS=$C/merged/labels.h5 PROBING_EMB=$C/merged PROBING_OUT=$C/probing_results $PY run_probing_suite.py \
  ref_length ref_position ref_aa ref_seq_window ref_seq_expert ref_seq_global ref_coords ref_backbone ref_global ref_pairhist \
  esm2_650m prottrans_t5_xl_u50 ankh_base esm3 esm_if1 proteinmpnn
ATOM_SETS=$($PY -c "
import pandas as pd
a = pd.read_csv('$R/extract/encoder_atom_audit.csv').set_index('encoder')
s = {a.loc[e, 'atom_set'] for e in ('esm3', 'esm_if1', 'proteinmpnn') if e in a.index}
print(' '.join(sorted(x for x in s if x not in ('N+CA+C+O+CB', 'none'))))")
echo "atom sets: $ATOM_SETS"
PROBING_LABELS=$C/merged/labels.h5 PROBING_EMB=$C/merged PROBING_OUT=$C/probing_results_atom $PY atom_matched_floors.py $ATOM_SETS
$PY confirmation_novelty.py
echo "$(date +%H:%M) cohort B pipeline done"
