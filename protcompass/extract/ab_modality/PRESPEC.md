# Within-model modality ablation: pre-specification

Written 2026-09-25 before any scoring of the new sources. The sha256 of this file is stored in `PRESPEC.sha256`.
Anything computed later that is not listed here is reported as exploratory.

## Question

With the same weights, what does the structure input add to ESM3 and to SaProt?

## Sources

| source | model and weights | input | module |
|---|---|---|---|
| `esm3` (released) | esm3_sm_open_v1, bfloat16 autocast | true sequence + VQ-VAE structure tokens from the real backbone + coordinates for geometric attention | `extract/patched/esm3_inproc.py` |
| `esm3_seqonly` (new) | same checkpoint, loader, dtype | `ESMProtein(sequence=...)` only: structure track = mask token at every position, no coordinates, per-residue pLDDT conditioning None | `extract/patched/esm3_seqonly.py` |
| `saprot` (released) | westlake-repl/SaProt_650M_AF2, revision d9b9ad00 | amino acid + real 3Di token (mini3di) per residue | `extract/patched/saprot_hf.py` |
| `saprot_seqonly` (new) | same checkpoint and tokenizer | amino acid + `#` per residue | `extract/patched/saprot_seqonly.py` |

Layer and pooling are those of the released modules: final trunk / last hidden state, special tokens removed, masked
mean over residues for the protein-level tasks, per-residue states for probing.

`#` check: the SaProt vocabulary (446 tokens) contains `A#` ... `Y#` for all 20 amino acids, `#` + each 3Di letter, and
`##`. `#` is the symbol SaProt uses for a masked (or low-confidence) position on either track. The tokenizer maps
`M#E#K#` to three single tokens with no `<unk>`; the module raises if any `<unk>` appears.

Environment: `~/miniconda3/envs/protein_embeddings/bin/python`, `PYTHONPATH=protcompass/pylib`
(the released esm3/saprot logs load `esm` from this env; `meta.json` records `env: base` because that is the
`CONDA_DEFAULT_ENV` of the launching shell).

Extraction runs through `extract/extract.py` (`run_downstream`, `run_probing`) unchanged, via the wrapper
`ab_modality/ab_extract.py`, which registers the two specs at run time (declared `uses_coords=False`,
`uses_sequence=True`), writes audit rows to `ab_modality/encoder_input_audit_ab.csv` and embeddings to
`ab_modality/embeddings/` and `ab_modality/probing_embeddings/`. `extract/common.py` is not edited.

## Step 1. Input audit (gate)

`extract/input_audit.py` (32 proteins, 16 equal-length pairs; threshold 1e-3 on the median relative change).
Each new source must be measured not to read coordinates (coordinate-swap change below 1e-3) and to read residue
identity (sequence-shuffle change above 1e-3), on the stage-1 audit set and on the probing audit set. extract.py's gate
refuses to save embeddings otherwise. If either source reads coordinates the ablation stops.

## Step 2. Downstream

Tasks and grouping by the structure the full models read:

- experimental structures: `ec_classification`, `go_mf`, `go_bp`, `remote_homology`
- predicted structures, AlphaFold DB: `subcellular_localization`, `membrane_soluble`
- predicted structures, ESMFold: `stability`, `fluorescence`, `solubility`, `ppi_affinity`

Rows: the stage-1 rows of `extract/manifest/<task>.csv` (6,000-protein training cap), identical ids and order to the
released esm3/saprot embeddings (checked before extraction and again after). PPI affinity: pair features are the
concatenation of the two chain embeddings (`extract/pairs/ppi_affinity_pairs.csv`).

Metrics (score2.TASKS): Spearman (stability, fluorescence, ppi_affinity), AUROC (solubility, membrane_soluble),
accuracy (remote_homology, ec_classification, subcellular_localization), protein-centric F-max (go_mf, go_bp).

Protocol: exactly score2.py (`ab_modality/ab_score.py` imports `score2.train`, `score2.evaluate` and copies the label
handling). Features standardised on train; linear and MLP-2 (512 hidden) readouts; seeds 0, 1, 2; early stopping on
validation loss. The readout of each source is selected independently on validation (lowest validation loss averaged
over the three seeds); the cell S is the three-seed mean of the selected readout's test score. Per-protein test
outputs of all six fits are saved.

Released sources: `esm3` and `saprot` are refit with the same code and seeds because their test predictions were not
saved. The refit reproduces the release if, for every task, the selected readout equals the released one and the
three-seed mean differs from `extract/scores2_selected.csv` by at most 0.005. Deltas and intervals use the refit
predictions in all cases; the released score is shown next to the refit, and any task that fails the reproduction
criterion is flagged.

Contrast: Delta = S(full) - S(seqonly), for (esm3, esm3_seqonly) and (saprot, saprot_seqonly), each task.

Interval: paired percentile bootstrap over test proteins (over test pairs for ppi_affinity), 2,000 draws. One index
matrix per task (`numpy.random.default_rng(sum(map(ord, task)))`, draws of n_test indices with replacement), shared by
all four sources of that task. In each draw each source's metric is computed on the resampled test set for each of its
three seeds and averaged over seeds; the draw's Delta is the difference of the two averages. 95% interval = 2.5th and
97.5th percentiles.

Comparison rule (per task and model): `structure adds` if the 95% interval lies above 0, `structure costs` if it lies
below 0, `unresolved` otherwise. Secondary: the same states at 99.75% (Bonferroni over the 20 downstream contrasts).
Group summary (secondary, descriptive): per model and per group (experimental; predicted), the count of tasks in each
state and the unweighted mean Delta over the group's tasks, with an interval from the same draws (tasks have
independent test sets, so the group mean of the per-task draws is a valid draw of the group mean).

## Step 3. Probing

Embeddings: `extract.py --task probing` logic (`run_probing`, last layer), same 19,461 domains and residue rows of
`probing_data/labels.h5` as the released esm3/saprot, written to `ab_modality/probing_embeddings/`.

Protocol: `scripts/symmetric_run.py` unchanged (superfamily split; 3,000 / 500 / 1,000 train / validation / test
domains; ridge or softmax linear readout and MLP-2 with seeds 0, 1, 2), run for `esm3_seqonly` and `saprot_seqonly`
only, with `PROBING_EMB=ab_modality/probing_embeddings` and its output directory redirected to
`ab_modality/symmetric/results_superfamily/by_source/` by the wrapper `ab_modality/ab_symmetric_run.py`. The released
`symmetric/` results (esm3, saprot, floors) are read, never written.

Probes: the 15 of symmetric_decide.py: determined `phi_circular`, `psi_circular`, `local_curvature`, `dssp_3`,
`dssp_8`, `core_surface`, `contact_density`, `distance_bins`, `contact_order_len_resid`,
`radius_of_gyration_len_resid`, `asphericity_len_resid`; undetermined `rsa`, `rsa_bins`, `bfactor`,
`architecture_label`. Metric: the released headline (Pearson r for regression, mean cosine for circular, balanced
accuracy for classification).

Selection: for every source the readout family (linear, or the mean of the three MLP seeds) with the higher
validation score, then the test score read once (symmetric_decide.choose).

Contrast 1: Delta = S(full) - S(seqonly) per probe for (esm3, esm3_seqonly) and (saprot, saprot_seqonly), with the
protein-level paired bootstrap of symmetric_decide.Boot (2,000 draws, one fixed draw per probe,
`default_rng(sum(map(ord, probe)))`, three MLP seeds averaged inside each draw). States at 95% and 99.9% as in the
downstream rule.

Contrast 2: each seq-only source against its input-matched floor. The input audit measures residue identity and no
coordinates, so the admissible floors (scripts/floor_rules.py rule) are the six sequence floors `ref_length`,
`ref_position`, `ref_aa`, `ref_seq_window`, `ref_seq_expert`, `ref_seq_global` (the `group` rule gives the same set
for a sequence-only reader). Floor descriptor and family chosen on validation; gap = S(source) - S(floor) with the same
bootstrap; `below` if the 95% upper end is below 0, `above` if the lower end is above 0, `unresolved` otherwise, and
the same at 99.9% (symmetric_decide logic). The released esm3 and saprot admissible-rule cells
(`symmetric/cells_superfamily.csv`) are shown next to them for reference, not recomputed.

## Step 4. Tables

`ab_modality/ab_latex.py` writes `ab_modality/table_downstream_deltas.tex` and `ab_modality/table_probing_deltas.tex`
from the CSV outputs.

## Outputs

`ab_modality/scores_ab_long.csv`, `scores_ab_selected.csv`, `predictions/`, `downstream_deltas.csv`,
`downstream_group_summary.csv`, `symmetric/results_superfamily/by_source/`, `probing_deltas.csv`,
`probing_floor_cells.csv`, `stage_times.csv`.
