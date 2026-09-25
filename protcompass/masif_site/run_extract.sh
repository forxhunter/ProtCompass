#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Residue-level extraction on the MaSIF-site chains, one encoder at a time, three passes so that an encoder that
# ran out of GPU memory next to the erasure grid is retried. The five encoders left out read a PDB file
# (SaProt, Foldseek 3Di, both GCPNets) or need a separate environment (CDConv); GeoBPE and FoldVision have no residue states.
cd "$PROTCOMPASS_ROOT/protcompass/extract"
L=$PROTCOMPASS_ROOT/protcompass/masif_site/labels.h5
O=$PROTCOMPASS_ROOT/protcompass/masif_site/embeddings
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
ENC="schnet tfn dimenet egnn ipa mace gearnet gvp pottsmpnn proteinmpnn pottsmpnn_pretrained pottsmpnn_soluble pottsmpnn_ft gearnet_pretrained tfn_pretrained ipa_pretrained esm_if1 esm3 esm2_8m esm2_35m esm2_150m esm2_650m esm1b protbert_bfd ankh_base prostt5 prottrans_t5_xl_u50 esm2_3b"
for pass in 1 2 3; do
  for e in $ENC; do
    [ -f "$O/$e/meta.json" ] && continue
    rm -f "$O/$e/residue_fp16.npy.partial"
    echo "$(date +%H:%M) pass $pass $e"
    timeout 5400 $PY extract.py --encoder $e --task probing --probing-labels "$L" --probing-out "$O" > ../masif_site/logs/$e.log 2>&1
    tail -1 ../masif_site/logs/$e.log
  done
  sleep 120
done
echo "$(date +%H:%M) extraction finished: $(ls $O/*/meta.json | wc -l) encoders"
