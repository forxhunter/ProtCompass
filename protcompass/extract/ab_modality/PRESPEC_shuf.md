# Shuffled-3Di control for the SaProt modality ablation: pre-specification

Written 2026-09-25 before any extraction or scoring of the new source. The sha256 of this file is stored in
`PRESPEC_shuf.sha256`. Anything computed later that is not listed here is reported as exploratory. The parent design is
`PRESPEC.md` (not edited); this file adds one source and two contrasts.

## Question

The question is whether `saprot_seqonly` (every 3Di token replaced by `#`) is a valid counterfactual or an
out-of-distribution input. The control keeps a structure track that is present and in distribution for each protein
(its own 3Di alphabet and composition) and removes only the residue-to-structure correspondence. Three states:

| state | source | structure track |
|---|---|---|
| full | `saprot` (released) | the protein's real 3Di token at each residue |
| present, wrong alignment | `saprot_shuf3di` (new) | the protein's real 3Di string, permuted over positions |
| absent | `saprot_seqonly` | `#` at each residue |

## Source definition

Module `extract/patched/saprot_shuf3di.py`. Checkpoint, tokenizer, layer (last hidden state), special-token slicing and
pooling are those of `patched/saprot_hf.py`. The 3Di string is computed exactly as there (`patched/foldseek_3di.states_3di`,
mini3di on N, CA, C and virtual CB; masked states filled with state 0; no pLDDT masking). Then, for a protein of length L
with id `pid`:

    seed = int.from_bytes(sha256(pid.encode("utf-8")).digest()[:8], "big")
    perm = numpy.random.default_rng(seed).permutation(L)
    structure-aware token i = sequence[i] + 3Di[perm[i]]

The multiset of 3Di letters is asserted unchanged for every protein. `pid` is the manifest `id` of the row (downstream;
chain id for PPI affinity) or the domain id of `probing_data/labels.h5` (probing). The input audit has no ids; there the
key is `"audit:" + sequence of the unperturbed protein`, reused for its jitter, swap, shuffle and rigid variants, so the
permutation is fixed within each audited protein. The fraction of positions whose 3Di letter is unchanged by the
permutation is logged per task (descriptive).

Driver: `ab_modality/ab_shuf_extract.py` registers the spec at run time (`uses_coords=True`, `uses_sequence=True`) and
runs `extract.py` (`run_downstream`, `run_probing`) unchanged.

## Step 1. Input audit (gate)

`extract/input_audit.py` as in PRESPEC.md, rows to `ab_modality/encoder_input_audit_ab_shuf.csv`. Expected: reads
coordinates (coordinate-swap change above 1e-3; the 3Di multiset comes from the structure), reads residue identity
(sequence-shuffle change above 1e-3), invariant to rigid motion. extract.py's gate refuses to save embeddings otherwise.

## Step 2. Downstream

Tasks, groups, stage-1 rows (6,000-protein training cap), PPI pair features and metrics are those of PRESPEC.md step 2:

- experimental structures: `ec_classification`, `go_mf`, `go_bp`, `remote_homology`
- predicted structures, AlphaFold DB: `subcellular_localization`, `membrane_soluble`
- predicted structures, ESMFold: `stability`, `fluorescence`, `solubility`, `ppi_affinity`

Protocol: `ab_score.score_one` (score2 train / evaluate, standardised features, linear and MLP-2 readouts, seeds 0, 1, 2,
early stopping on validation loss), via `ab_modality/ab_shuf_score.py`. Per-protein test outputs of all six fits go to
`ab_modality/predictions/<task>__saprot_shuf3di.npz`; score rows to `ab_modality/score_parts_shuf/`. Readout of each
source selected independently on validation (lowest validation loss averaged over the three seeds); S = three-seed mean
of the selected readout's test score. For `saprot` and `saprot_seqonly` the selection and predictions are read from
`score_parts/ab_long_<task>__<source>.csv` and `predictions/<task>__<source>.npz` (read only).

Contrasts, per task:
- A = S(saprot) - S(saprot_shuf3di): what the correct alignment adds over a present, composition-matched track.
- B = S(saprot_shuf3di) - S(saprot_seqonly): what a present but misaligned track does relative to `#`.
(A + B equals the parent contrast S(saprot) - S(saprot_seqonly).)

Interval: paired percentile bootstrap over test proteins (test pairs for PPI affinity), 2,000 draws, the index matrix of
`ab_bootstrap.py` (`numpy.random.default_rng(sum(map(ord, task)))`, draws of n_test indices with replacement), shared by
the three sources of the task; in each draw each source's metric is computed for each of its three seeds with
`ab_bootstrap.boot_metric` and averaged over seeds. 95% interval = 2.5th and 97.5th percentiles. Test ids and labels
must be identical across the three sources (asserted).

States: A: `alignment adds` if the 95% interval lies above 0, `alignment costs` if below, `unresolved` otherwise.
B: `shuffled above #` / `shuffled below #` / `unresolved` by the same rule. Secondary: the same at 99.75% (Bonferroni
over the 20 contrasts). Group summary (descriptive): per contrast and group (experimental; predicted), the unweighted
mean over tasks with an interval from the same draws, and the count of tasks in each state.

## Step 3. Probing (if time allows; downstream has priority)

Embeddings: `run_probing` (last layer), same 19,461 domains and residue rows as the released saprot, to
`ab_modality/probing_embeddings/saprot_shuf3di`. Protocol: `scripts/symmetric_run.py` unchanged via
`ab_modality/ab_shuf_symmetric_run.py` (output `ab_modality/symmetric/results_superfamily/by_source/saprot_shuf3di.*`).
Probes and headline metrics: the 15 of PRESPEC.md step 3. Readout family chosen on validation for each source; per
probe, A = S(saprot) - S(saprot_shuf3di) and B = S(saprot_shuf3di) - S(saprot_seqonly) with the protein-level paired
bootstrap of `symmetric_decide.Boot` (2,000 draws, `default_rng(sum(map(ord, probe)))`), as `ab_probe_decide.py` does.
States at 95% and 99.9%. No floor contrast for this source.

## Outputs (new files only)

`PRESPEC_shuf.md`, `PRESPEC_shuf.sha256`, `encoder_input_audit_ab_shuf.csv`, `embeddings/<task>/saprot_shuf3di/`,
`predictions/<task>__saprot_shuf3di.npz`, `score_parts_shuf/`, `scores_ab_long_shuf.csv`, `scores_ab_selected_shuf.csv`,
`downstream_deltas_shuf.csv`, `downstream_group_summary_shuf.csv`, `probing_deltas_shuf.csv`, `table_shuf3di.tex`,
`stage_times_shuf.csv`, `logs/*shuf*`. No existing file in `ab_modality/` or `patched/` is edited.
