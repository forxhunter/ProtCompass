# Stability composition decomposition: pre-specification

Written before any score in this analysis was computed. Nothing here is changed after the first score exists; later
notes go in the report. Script: `scripts/ab_stability_decomp.py`. Outputs: `erasure/ab_stability_decomp/`.

## Question

The released repair contrast for amino-acid composition on stability, with the released competitors C0 in every arm
(`scripts/stability_competing.py`, `erasure/stability_competing/`), is R_var|C0 = +0.100 (30 of 32 encoders, 20 of
20 families). The test split is 2,998 single mutants of 17 parents plus 2 wild-type entries. The composition of a
variant is its parent's background composition plus the change made by its substitutions. Which part carries the
repair: the parent background or the substitution?

## Input facts checked before this file was written (no score computed)

- The released composition descriptor (structure-derived, `common_struct`, column order ALA ARG ASN ASP CYS GLN GLU
  GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL) equals the composition of the manifest sequence to within
  1.5e-8 on all 10,500 stage-1 rows.
- The parent sequence the released covariates use (the parent's own manifest entry) carries a C-terminal tag
  `LESSGGS` for 11 of the 17 test parents that the test variants do not carry. For those parents the variant is the
  parent minus 7 residues plus one substitution, so composition(variant) - composition(parent sequence) is not the
  +-1/L substitution vector: it contains the removed tag and a rescaled copy of the variant composition
  (counts x (1/43 - 1/50)).
- With the released aligner, 2,984 of the 2,998 test point mutants yield exactly one aligned substitution that
  matches the wild-type and mutant letters in the id. The other 14 are mutations of the last residue of a tagged
  parent, which the aligner absorbs into the tag gap (0 substitutions).
- The position number in a natural protein's id is native numbering, not a sequence index, so positions come from
  the alignment, not from the id.

## Protocol kept exactly as released

Stability stage-1 split (6,000 train, 1,500 valid, 3,000 test); the 32 panel encoders (`panel.panel()`); the
conditional LEACE erasure of composition (`addback_named_pairs.conditional`, `erasure.leace_eraser`); the four
variance-matched nulls (`matched_null.matched_null_projector`, seeds 300-303); the released head
(`stability_competing.head`: RidgeCV on train-standardised features, pooled test Spearman and within-parent test
Spearman = mean over test parents with at least 20 variants); the released parent mapping and C0 columns
(`erasure/stability_competing/covariates.csv`, all columns except `parent`, in file order); feature order in every
fit is [arm representation, competitor block, descriptor], as released. The four shams are not recomputed: they do
not enter R_var and do not change the null projectors. The permuted control uses the released permutations
(`np.random.default_rng(0)`, one permutation each for train, valid, test, drawn in that order), applied to every
descriptor.

## Substitutions (per stage-1 variant, relative to its released parent)

1. Global alignment of the parent sequence to the variant sequence with the released aligner (Biopython
   PairwiseAligner, match 0, mismatch -1, gap -1, first alignment). Substitutions are the aligned columns with
   different residues; the site index is the index in the variant sequence.
2. For kind "point" variants whose alignment does not give exactly one substitution equal to the id's letters
   (X, Y): compare the variant ungapped against the parent's first len(variant) residues, then against its last
   len(variant) residues; if one of these gives exactly one mismatch and it is X -> Y, use that site. Otherwise keep
   the alignment's substitutions.
3. Parent missing (released flag `parent_missing`): the variant is its own parent, no substitutions.

## Descriptors added back (each also scored with the released permutation)

- `full`: the released composition descriptor q (reproduces the released run).
- `cparent`: composition of the parent background on the variant's residues = composition of the variant sequence
  with every identified substitution reverted (Y -> X at each site). Same length as the variant. Equals q when there
  is no substitution or the parent is missing.
- `dc`: q - cparent. For every variant this is sum over substitutions of (e_Y - e_X)/L, the substitution vector.
- `cparent_seq` (as worded in the request): composition of the parent's full manifest sequence (tag included);
  the variant's own composition when the parent is missing.
- `dc_seq`: q - cparent_seq.

cparent + dc = cparent_seq + dc_seq = q exactly.

## Competitor sets (present in every arm, erased and null alike)

- C0: the released competitors (edit burden, substitution classes, mean pLDDT, topology, round, kind,
  parent_missing).
- C1: C0 plus, over the identified substitutions: wild-type residue counts (20, order ACDEFGHIKLMNPQRSTVWY), mutant
  residue counts (20), mean relative position mean((i+1)/L) with i the 0-based variant index and L the variant
  length, a 5-bin position histogram (counts, bin = min(floor(5 i / L), 4)), and the local environment on the
  variant's ESMFold structure (manifest `structure_path`): mean relative solvent accessibility over the sites
  (freesasa, residue total area / MAX_ASA of `probing_data/make_probing_data.py`, clipped to [0, 1], residue number
  = i + 1) and DSSP 3-state counts over the sites (mkdssp 3.0.0 as in `run_dssp`; H = G,H,I; E = B,E;
  C = T,S,blank,C,P). Two indicators: `no_sub` (no identified substitution; all site features 0) and `env_missing`
  (substitutions exist but no RSA or DSSP value was obtained at any site; RSA and DSSP features 0). Sites with a
  missing value are left out of that feature's mean or count. Site features are cached to `site_features.csv`.

## Quantities

For descriptor q' and competitor set C, from fits of one run per encoder:
R_var = [S(E_p X, C, q') - S(E_p X, C)] - mean over 4 nulls [S(E_c X, C, q') - S(E_c X, C)], with S the pooled test
Spearman (primary) or the within-parent Spearman (secondary). Rperm is the same with q' permuted. Also scored: the
intact representation with C (none and `full`), and, with no representation (released `no_representation`
protocol, labels and ids from esm2_8m): S(C0), S(C1), S(C + q') and S(q') for each descriptor.

## Decision rule (released rule, applied per row)

Rows: 5 descriptors x 2 competitor sets = 10. Per row: two-sided sign test over the 32 encoders on R_var
(binomtest, k positive of n); Holm correction over the 10 rows (pooled metric, primary). A row carries repair if the
Holm p < 0.05 and the mean R_var >= 0.02; attenuated if Holm p < 0.05, the mean is positive and below 0.02;
negative if Holm p < 0.05 and the mean is negative; not detected otherwise. The same rule, with its own Holm
correction over the 10 rows, is applied to the within-parent metric as the secondary reading. Family level, as
released: one vote per architecture family (`family_weighted_demand.family`, family mean of R_var), two-sided
binomial sign test over the 20 families, reported uncorrected. Rperm is reported per row with its uncorrected
sign-test p.

## Reproduction gate

The C0 rows with `full`, `full` permuted and no descriptor, on the intact, erased and null arms, must equal the
released rows of `erasure/stability_competing/cells_[ab].csv` (covariates C) to within 1e-6 in both the pooled and
the within-parent score, for every encoder. The first encoder is checked before the others run. If the gate fails,
the analysis stops and the failure is reported.

## Reading (not a decision rule)

R_var(dc | C0) > 0 while R_var(full | C1) falls toward 0: the composition signal is substitution identity.
R_var(cparent | C1) > 0: parent background composition carries signal. Neither outcome is assumed.
