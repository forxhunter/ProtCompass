# GFP competing-explanation panel: pre-specification

Written before any score with the covariates below was computed. Nothing in this file is changed after the first
result exists; later notes go in REPORT.md.

## Question

The repair rule names two properties on fluorescence: amino-acid composition and burial and packing. For each,
does the property descriptor still repair its own erasure, beyond a variance-matched null, once simple competing
explanations are present in every arm?

## Data and protocol (unchanged from the released repair rule)

Fluorescence stage-1 split (6,000 train, 1,500 validation, 3,000 test variants of avGFP), every panel encoder with a
fluorescence embedding, the conditional erasure of the property (LEACE on the property residualised on the other
independent blocks), four shams, four variance-matched nulls (seeds 300-303), the released regression head
(erasure.fit_score: RidgeCV on train-standardised features, Spearman on test), and the property descriptor q.
Everything is fitted on the training split.

## Competing explanations C (computed from the sequence and the manifest only)

Parent: the avGFP sequence of the TAPE fluorescence task, which is also the per-position mode of the 237-residue
training variants. Each variant is aligned to the parent globally with unit costs for substitutions and gaps
(Biopython PairwiseAligner, match 0, mismatch -1, gap -1).

- C1 mutation burden: number of edits to the parent (substitutions plus gaps), its square, and one-hot bins
  {0, 1, 2, 3, 4, 5, 6-7, 8+}.
- C2 substitution classes (counts over aligned substitutions): hydrophobic to non-hydrophobic, non-hydrophobic to
  hydrophobic (hydrophobic = A V I L M F W C), to Pro, from Pro, to Gly, from Gly, charge-altering substitutions
  (K R versus D E versus neutral), net charge change, aromatic gained (F W Y), aromatic lost, number of gaps.
- C3 composition change: the 20 amino-acid counts of the variant minus those of the parent.
- C4 structure quality: mean pLDDT of the predicted structure (manifest column mean_plddt).

Competing set per property:
- burial and packing: C1 + C2 + C3 + C4.
- amino-acid composition: C1 + C2 + C4. C3 is left out because for variants of the parent's length the composition
  descriptor is an affine function of C3; adding it would remove the descriptor by construction, which says nothing
  about the demand. (Variants with a deletion are 3.4% of the split.)

## Quantities

- R_var (no covariates): the released repair contrast against the variance-matched null, from the released cells.
- R_var|C (primary): the same contrast with C appended to every arm and in both terms of every difference:
  R = [S(E_p X + C + q) - S(E_p X + C)] - mean_null [S(E_c X + C + q) - S(E_c X + C)].
- Incremental value on the intact representation: S(X + C + q) - S(X + C).
- Incremental value without any representation: S(C + q) - S(C).
- Permuted-descriptor control: R with q's rows permuted across proteins, with C present.

## Decision rule (per property)

Applied to R_var|C over the panel, with Holm correction over the two properties:
- retained: two-sided sign test over encoders at Holm p < 0.05 and mean R_var|C >= 0.02 (the released rule);
- attenuated: the sign test passes and the mean is below 0.02;
- explained by the competitors: the sign test fails.
Also reported: the sign test over architecture families (one vote per family, the family mean), and the ratio of the
mean R_var|C to the mean R_var.

## Reproduction check

The no-covariate arms are recomputed for three encoders and must match the released cells to 1e-6 in score before
the released cells are used as the no-covariate reference.
2026-09-23T16:46:16-05:00

## Addendum written before any result with covariates was read (2026-09-23 16:53 CDT)

- The reproduction check failed: recomputed no-covariate erased-arm scores differ from the released cells by up to
  0.015 (esm2_8m, burial and packing), because the orthonormal basis of the erased arm is not unique (see
  leace_check/REPORT.md) and the head standardises each column. Per the check above, the released cells are not used;
  both the no-covariate and the covariate arms are computed in the same run for every encoder, so each comparison
  shares one basis.
- Deadline: the reports are due Sep 24, 10:00 CDT. The full panel is expected to finish well before that, so no
  subset is fixed. If it does not, the encoders finished by 06:00 CDT are reported, in panel order.
