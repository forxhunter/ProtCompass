#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"  # release: repository root
# Shuffled-3Di control, probing (PRESPEC_shuf.md step 3; lower priority than downstream). Starts the probing extraction
# once one of the two downstream extraction processes has exited and the GPU has at least 8 GB free, then the symmetric
# probing run, then (once saprot_seqonly's symmetric results exist) the per-probe contrasts.
cd "$PROTCOMPASS_ROOT/protcompass/extract/ab_modality"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
export PYTHONPATH=$PROTCOMPASS_ROOT/protcompass/pylib
LOG=logs/shuf_probing_driver.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
PID_A=$1
say "start (downstream A pid $PID_A)"
if [ ! -f probing_embeddings/saprot_shuf3di/meta.json ]; then
  while true; do
    PID_B=$(grep -o "pid [0-9]*" logs/shuf_watch.log | tail -1 | awk '{print $2}')
    a_done=0; b_done=0
    kill -0 $PID_A 2>/dev/null || a_done=1
    if [ -n "$PID_B" ]; then kill -0 $PID_B 2>/dev/null || b_done=1; fi
    f=$(free_mb)
    if [ $((a_done + b_done)) -ge 1 ] && [ "$f" -ge 8192 ]; then break; fi
    sleep 30
  done
  say "probing extraction starts (free ${f} MiB)"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY ab_shuf_extract.py probing > logs/shuf_extract_probing.log 2>&1
  say "probing extraction exit $?"
fi
[ -f probing_embeddings/saprot_shuf3di/meta.json ] || { say "no probing embeddings; stop"; exit 1; }
until [ "$(free_mb)" -ge 8192 ]; do sleep 30; done
say "symmetric run starts (free $(free_mb) MiB)"
PROBING_EMB=$PWD/probing_embeddings SYM_SPLIT=superfamily OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  $PY ab_shuf_symmetric_run.py saprot_shuf3di > logs/symmetric_saprot_shuf3di.log 2>&1
say "symmetric run exit $?"
until [ -f symmetric/results_superfamily/by_source/saprot_seqonly.csv ]; do sleep 30; done
say "saprot_seqonly symmetric results present; contrasts"
PROBING_EMB=$PWD/probing_embeddings SYM_SPLIT=superfamily OMP_NUM_THREADS=6 $PY ab_shuf_probe_decide.py > logs/shuf_probe_decide.log 2>&1
say "probe decide exit $?"
$PY ab_shuf_latex.py >> $LOG 2>&1
say "latex exit $?; done"
