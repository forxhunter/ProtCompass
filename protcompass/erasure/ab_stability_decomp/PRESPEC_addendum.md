# Addendum to PRESPEC.md: conditional rows and uncertainty

Written 2026-09-25 about 15:31 CDT, before any of the rows below was computed. PRESPEC.md (sha256 9ed95465...) is
unchanged and every row it specifies keeps its definition and its 10-row Holm family.

## What had been seen when this was written

These rows were requested by the coordinator after review. I did not choose them. Before this file was written I had
seen the following:
- the reproduction gate on the first 7 encoders (exact);
- a smoke test of the report code on those 7 encoders (R_var for all 10 original rows on 7 encoders);
- gvp run with 1 thread and with 3 threads (thread noise in R_var up to about 0.008);
- the no-representation fits for all descriptors with none, C0 and C1;
- the progress log of the main run, which prints each fit's pooled and within-parent score, up to foldvision.

## Additional rows (same erasure, nulls, heads, encoders, splits, permutations and decision rule)

- (A) dc given cparent: cparent joins the competitor block in every arm, and dc is the descriptor added back.
  R_var(dc | cparent, C) = [S(E_p X, C, cparent, dc) - S(E_p X, C, cparent)] - mean over the 4 nulls of the same
  difference, for C = C0 and C = C1. Permuted control: dc permuted (released permutations), cparent intact.
- (B) cparent given dc: dc joins the competitor block in every arm, and cparent is the descriptor added back.
  R_var(cparent | dc, C) = [S(E_p X, C, dc, cparent) - S(E_p X, C, dc)] - mean over the 4 nulls, for C = C0 and C1.
  Permuted control: cparent permuted, dc intact.
- The fits S(., C, cparent) and S(., C, dc) are the `cparent` and `dc` add-back fits of PRESPEC.md. The joint fit
  S(., C, cparent, dc) is computed once, with feature order [arm, C, cparent, dc], and used for both (A) and (B).
  A ridge fit on train-standardised columns does not depend on column order except through floating point.
- Holm: the 4 new rows (pooled metric) form their own Holm family. Holm over all 14 rows is reported as a
  sensitivity. The within-parent metric is corrected the same way, separately.

## Uncertainty for every row (old and new)

- Parent-cluster uncertainty: each fit records the Spearman inside each test parent with at least 20 variants (the
  17 test parents). The per-parent contrast for encoder e and parent g is R_var computed with that parent's
  Spearman in place of the pooled score. The panel value per parent is its mean over the 32 encoders. The panel
  within-parent contrast is the mean over parents of that value, which equals the reported within-parent R_var
  mean when every parent enters every fit. 95% interval: percentile bootstrap over parents (17 drawn with
  replacement, 2,000 draws, seed 0). Per-parent values go to `parent_contrasts.csv`.
- Family weighting: for every row and both metrics, the encoder-level result (mean, positives of 32, sign test) and
  the one-vote-per-family result. The family map is the one the released report uses
  (`family_weighted_demand.family`). The family result is the family-weighted mean (the mean of the 20 family means),
  the families with a positive mean, and a two-sided sign test over the 20 families.

## Execution

The main run of PRESPEC.md did not record per-parent Spearman, so it is stopped. A second script,
`scripts/ab_stability_decomp_v2.py`, refits with the same code path and 3 BLAS threads (the setting that reproduced
the released rows exactly) and records per-parent values. It runs in two passes. Pass 1 covers rows (A) and (B)
with C1 and then C0 for all encoders. Pass 2 covers the remaining PRESPEC fits, starting with the encoders the main
run had not reached. Where a fit appears in both the main run and v2, the two must agree to within 1e-9.
