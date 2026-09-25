#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Regenerate the LaTeX table bodies that must not drift from the release.
set -e
cd "$PROTCOMPASS_ROOT/protcompass"
PY=$HOME/miniconda3/envs/protein_embeddings/bin/python
G=$PROTCOMPASS_ROOT/protcompass/paper/generated
mkdir -p "$G"
$PY scripts/emit_inventory.py encoders          > "$G/encoder_inventory.tex"
$PY scripts/emit_inventory.py probes            > "$G/probe_inventory.tex"
$PY scripts/emit_floor_definitions.py           > "$G/floor_definitions.tex"
$PY scripts/determinacy_table.py --markdown     > "$G/determinacy.tex"
$PY scripts/emit_latex_tables.py downstream-reference > "$G/downstream_reference.tex"
$PY scripts/emit_latex_tables.py group-controls       > "$G/group_controls.tex"
$PY scripts/emit_latex_tables.py prediction-test      > "$G/prediction_test.tex"
$PY scripts/emit_latex_tables.py train-size           > "$G/train_size.tex"
$PY scripts/emit_latex_tables.py capacity             > "$G/capacity.tex"
$PY scripts/emit_latex_tables.py discard              > "$G/discard.tex"
$PY scripts/emit_latex_tables.py positive-control     > "$G/positive_control.tex"
$PY scripts/emit_latex_tables.py strong-floors        > "$G/strong_floors.tex"
$PY scripts/emit_latex_tables.py probing-restricted   > "$G/probing_restricted.tex"
$PY scripts/emit_latex_tables.py demand-restricted    > "$G/demand_restricted.tex"
$PY scripts/emit_latex_tables.py recoverability       > "$G/recoverability.tex" || true
$PY scripts/emit_latex_tables.py nonlinear-control    > "$G/nonlinear_control.tex" || true
$PY scripts/emit_downstream_table.py scores           > "$G/downstream_full.tex"
$PY scripts/emit_downstream_table.py sd               > "$G/downstream_full_sd.tex"
wc -l "$G"/*.tex
