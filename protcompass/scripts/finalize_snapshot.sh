#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Wait for the erasure sweep to stop, then regenerate everything that reads it from one frozen snapshot.
#
# The reason this is one script rather than a sequence of calls is that the sweep's coverage changes while it runs,
# and a figure generated at 300 cells next to a table generated at 320 is how the numbers in this project have
# disagreed before. Nothing here starts until no worker is left.
set -u
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
MS=$PROTCOMPASS_ROOT/protcompass/paper
step() { echo; echo "=== $* ==="; }

while pgrep -f "erasure_v2[.]py" > /dev/null || pgrep -f "finish_sweep[.]sh" > /dev/null; do sleep 60; done

step "snapshot coverage"
$PY scripts/sweep_status.py | tee erasure/SNAPSHOT_COVERAGE.txt

step "exclusion bounds"
$PY scripts/exclusion_bounds.py

step "causal demand"
$PY scripts/causal_demand_v2.py > erasure/causal_demand_v2_report.txt 2>&1
tail -25 erasure/causal_demand_v2_report.txt

step "Figure 4"
$PY scripts/make_main_figures.py 2>&1 | tail -4

step "recompile"
cd "$MS"
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
grep -c "undefined" main.log || true
pdfinfo main.pdf | grep Pages

echo
echo "SNAPSHOT COMPLETE"
