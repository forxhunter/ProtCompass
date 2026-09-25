#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"  # release: repository root
# Main-session driver for the remaining modality-ablation steps (written 2026-09-25 after the partial handback).
cd "$PROTCOMPASS_ROOT/protcompass/extract/ab_modality"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
LOG=logs/finish_pipeline.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "start"
# 1. saprot_seqonly probing run once its probing embeddings exist
until [ -f probing_embeddings/saprot_seqonly/meta.json ]; do sleep 30; done
say "saprot_seqonly probing embeddings ready; launching symmetric run"
PROBING_EMB=$PWD/probing_embeddings SYM_SPLIT=superfamily PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/pylib \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY ab_symmetric_run.py saprot_seqonly > logs/symmetric_saprot_seqonly.log 2>&1 &
SAP=$!
# 2. wait for downstream extraction, the score watcher and both symmetric runs
for p in 2906116 2906117 2906118 3117518 3134360; do while kill -0 $p 2>/dev/null; do sleep 30; done; say "pid $p finished"; done
wait $SAP; say "saprot_seqonly symmetric run finished (exit $?)"
# 3-5. merge, bootstrap, probing decisions, tables
$PY ab_score.py --merge-only >> $LOG 2>&1; say "merge exit $?"
$PY ab_bootstrap.py >> $LOG 2>&1; say "bootstrap exit $?"
PROBING_EMB=$PWD/probing_embeddings SYM_SPLIT=superfamily PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/pylib \
  $PY ab_probe_decide.py >> $LOG 2>&1; say "probe_decide exit $?"
$PY ab_latex.py >> $LOG 2>&1; say "latex exit $?"
say "ALL DONE"
