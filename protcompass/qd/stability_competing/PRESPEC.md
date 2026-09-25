# Stability competing-explanation panel: pre-specification

Written before any score with the covariates below was computed. Nothing here is changed after the first result
exists; later notes go in the report.

## Question

Over the full 60-pair grid, amino-acid composition is named on stability by the repair rule (released cells:
R_var +0.066, 26 of 32 encoders positive). The stability test split is 2,998 single-site mutants of 17 parent
proteins (13 designed miniproteins, villin, Pin1, hYAP65) plus two parents. Does the composition descriptor still
repair its own erasure, beyond the variance-matched null, once simple features of the mutation, of the parent's
design and of the predicted structure are present in every arm? And does it repair within a parent, where the
descriptor cannot act as an identifier of the parent?

## Data and protocol (unchanged from the released repair rule)

Stability stage-1 split (6,000 train, 1,500 validation, 3,000 test), every panel encoder with a stability embedding
(composition is undefined for Foldseek 3Di), the conditional LEACE erasure of composition (residualised on the
other independent blocks), four shams (permuted concept, seeds 100-103), four variance-matched nulls (seeds
300-303), the released regression head (erasure.fit_score: RidgeCV on train-standardised features, Spearman on the
test split) and the released composition descriptor. Everything is fitted on the training split. Arms with and
without covariates are computed in the same run for every encoder, so each comparison shares one basis of the
erased arm (see leace_check: the basis is not unique and the head is not rotation invariant).

## Parent assignment (from the manifest id; no parent field exists)

- `<parent>_<X><pos><Y>` (a single substitution, X and Y amino acids): the parent is `<parent>`, kind "point".
- Designed proteins `<design>.pdb[_suffix]`: the parent is `<design>.pdb`; kind is the suffix (hp, random, PG_hp,
  buryD, other) or "original" without a suffix.
- Natural proteins with a suffix (`<name>_PG_hp`, ...): the parent is `<name>`.
- The parent sequence is the sequence of the parent's own entry in the full manifest (any split, any stage). All 17
  test parents have one. When a training parent has none (1.4% of variants), the variant is its own parent
  (edits 0) and a flag `parent_missing` is set.
- Edits: global alignment to the parent sequence, unit costs for substitutions and gaps (Biopython PairwiseAligner,
  match 0, mismatch -1, gap -1).

## Competing explanations C

- C1 edit burden: number of edits to the parent, its square, and one-hot bins {0, 1, 2, 3-5, 6-10, 11-20, 21+}.
- C2 substitution classes (counts over aligned substitutions, as for GFP): hydrophobic to non-hydrophobic,
  non-hydrophobic to hydrophobic (hydrophobic = A V I L M F W C), to Pro, from Pro, to Gly, from Gly,
  charge-altering, net charge change, aromatic gained, aromatic lost, number of gaps.
- C4 structure confidence: mean pLDDT of the predicted structure (manifest column mean_plddt).
- C5 design identity that transfers to unseen parents: topology one-hot {EEHEE, EHEE, HEEH, HHH, other}, design
  round one-hot {rd1, rd2, rd3, rd4, none}, variant kind one-hot {original, point, hp, random, PG_hp, buryD,
  other}, and parent_missing.
- Left out: composition-change counts (for single mutants of one parent they are the composition descriptor up to
  an affine map, as in the GFP specification), and a parent one-hot (13 of the 17 test parents never occur as a
  training parent, so a train-fitted parent coefficient cannot transfer; the within-parent metric below addresses
  parent identity instead).

## Quantities

- R_var and R_var|C: the repair contrast against the variance-matched nulls, without and with C in every arm:
  R = [S(E_p X + q) - S(E_p X)] - mean_null [S(E_c X + q) - S(E_c X)], C appended to every term for R_var|C.
- R_sham and R_sham|C: the same against the four shams.
- Rperm|C: R_var|C with the descriptor's rows permuted across proteins.
- inc|C: S(X + C + q) - S(X + C) on the intact representation; S(C), S(C + q), S(q) with no representation.
- Two test metrics from the same fitted heads: pooled Spearman over the test split (the released metric, primary),
  and within-parent Spearman, the mean over the 17 test parents of the Spearman inside each parent (secondary).

## Decision rule

Primary, on R_var|C with the pooled metric over the panel: retained if a two-sided sign test over encoders gives
p < 0.05 and the mean is at least 0.02 (the released rule); attenuated if the sign test passes and the mean is below
0.02; explained by the competitors if the sign test fails. The same rule is applied to R_var|C with the
within-parent metric as the secondary verdict. Reported alongside: the sign test over architecture families (one
vote per family, the family mean), the ratio of the mean R_var|C to the mean R_var, R_sham|C, and Rperm|C.
2026-09-23T18:09:52-05:00
