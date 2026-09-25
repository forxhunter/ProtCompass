# Fixed-head erasure analysis: pre-specification

Written 2026-09-25 before any fixed-head score was computed. The hash of this file is stored in `PRESPEC.sha256`.

## Question

Does the fitted downstream predictor rely on the property? The released erasure grid
(`scripts/addback_named_pairs.py`) retrains the head on every arm. Here the head is trained once on the intact
representation, kept fixed, and scored on test inputs that have been intervened on in the original coordinate system.

## Interventions (original coordinates, no re-basing)

For each (task, property, encoder) cell, the projectors are built exactly as in `run_cell` of
`scripts/addback_named_pairs.py`, on the same training rows (`ok` mask) and the same residualised property
(`conditional`):

- erased: `mu, P, r, B = leace_eraser(X, Z)`
- sham s in 0..3: `leace_eraser(X, Z[perm])` with `perm = default_rng(100 + s).permutation(len(Z))`; a sham seed
  with rank 0 or fewer than 8 retained coordinates is skipped, as in `run_cell`
- null s in 0..3: `matched_null_projector(X, Z, r, target, seed=300 + s)`, with `target` the variance removed by
  the released erased arm

Each projector (mean `mu`, matrix `P`) is applied to a row x as `x' = mu + (x - mu) @ P.T`. The re-basing matrix `B`
is not applied. `B` is used only for the released inclusion criterion (`B.shape[1] >= 8`). Because B is an
orthonormal basis of range(P), the variance removed is the same with or without B.

Before the run, one cell is checked numerically: x' has the dimension of x, and the training cross-covariance between
x' and the residualised property is zero to numerical precision for the erased arm.

## Head

h is fitted once per cell on the intact training rows, with the same head class, the same standardiser (intact
training mean and sd) and the same penalty or stopping selection as `erasure.fit_score` for the task kind:

- reg: `RidgeCV(alphas=logspace(-2, 4, 13))` (its own efficient leave-one-out selection on train, as released), with
  the released jitter fallback on LinAlgError; metric Spearman
- binary: logistic regression with C in {0.01, 0.1, 1}, chosen by validation AUC on the intact validation split;
  metric AUC
- clf and multilabel: the released linear torch head (AdamW lr 3e-3, wd 1e-4, seed 0, batch 1024, at most 60
  epochs, early stopping at patience 5 on intact validation loss); metrics accuracy and protein-centric Fmax as in
  `fit_score`

The standardiser and the weights are frozen. Every arm is scored by applying h to the intervened test rows
(standardised with the intact training statistics). S is the released test metric.

## Quantities

Per cell: S_intact, S_erased, S_sham (mean over sham seeds), S_null (mean over null seeds), variance removed by the
erased arm and mean over null arms, property share removed, and

- Delta_fixed_var = mean_s S(h(x'_null_s)) - S(h(x'_erased))
- Delta_fixed_sham = mean_s S(h(x'_sham_s)) - S(h(x'_erased))

## Scope

Every pair in `erasure/named_pair_verdicts_full.csv` with `named_by_effect_rule` or `passes_R_rule` true. The file
gives 11 pairs:

| task | property | effect rule | repair rule |
|---|---|---|---|
| fluorescence | burial and packing | yes | yes |
| ppi_affinity | secondary structure content | yes | yes |
| solubility | amino-acid composition | yes | no |
| ppi_affinity | contact order | yes | yes |
| go_mf | amino-acid composition | yes | no |
| ppi_affinity | size | no | yes |
| ppi_affinity | global shape | no | yes |
| fluorescence | amino-acid composition | yes | yes |
| stability | amino-acid composition | no | yes |
| membrane_soluble | amino-acid composition | yes | no |
| subcellular_localization | amino-acid composition | yes | no |

Encoders: `panel()` restricted to those with `extract/embeddings/<task>/<encoder>/meta.json`, and the cell inclusion
conditions of `run_cell` (at least 200 usable training rows, the property present, a non-degenerate residualised
property, rank r > 0 and at least 8 retained coordinates). A cell whose null construction raises (for example a
target outside (0, 1)) is dropped, as in the released runner.

All cells are computed locally with the current code and data. The released fluorescence and PPI-affinity cells were
computed on another machine; the local recomputation of those pairs can differ from the released grid.

## Decision rule (fixed before the run)

The released effect rule applied to Delta_fixed_var, per pair:

- k = number of encoders with Delta_fixed_var > 0, n = number of cells
- p = two-sided sign test, `scipy.stats.binomtest(k, n, 0.5)`
- Holm correction over the 11 pairs tested
- the pair is named by the fixed-head rule if Holm p < 0.05, mean Delta_fixed_var >= 0.02, and the property varies on
  the task (the `varies` flag of the verdicts file)

For comparability, a second Holm adjustment over 60 pairs is reported, with the 49 untested pairs entered as p = 1.
The same test on Delta_fixed_sham is reported as a secondary, descriptive quantity. Nothing else is tested.

## Outputs

- `erasure/ab_fixed_head/cells.csv`: one row per cell
- `erasure/ab_fixed_head/summary.csv`: one row per pair, with the released retrained-head `eff_null` (Delta_var) and
  `R_null` (R_var) side by side
- `erasure/ab_fixed_head/fixed_head.tex`: LaTeX table emitted by `scripts/ab_fixed_head_summary.py`

Scripts: `scripts/ab_fixed_head.py` (cells), `scripts/ab_fixed_head_summary.py` (summary and table).
