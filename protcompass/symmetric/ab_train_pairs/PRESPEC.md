# Pre-specification: pretrained versus randomly initialised pairs (same architecture, same input)

Written 2026-09-25 before any pair difference was computed. Its sha256 is stored in `PRESPEC.sha256`.
Analysis script: `scripts/ab_train_pairs.py` (probing and downstream), `scripts/ab_train_pairs_report.py` (table and figure).

## Question

Holding architecture and input fixed, what does pretraining change in the probing scores and in the downstream scores?

## Pairs (random -> pretrained)

Primary (five architectures):

| pair | random | pretrained |
|---|---|---|
| GCPNet | gcpnet_random | gcpnet_pretrained |
| GearNet | gearnet (7M configuration) | gearnet_pretrained (20M configuration) |
| IPA | ipa | ipa_pretrained |
| TFN | tfn | tfn_pretrained |
| PottsMPNN | pottsmpnn | pottsmpnn_pretrained |

Secondary (reported, not counted in the primary summary):

- PottsMPNN-soluble: pottsmpnn -> pottsmpnn_soluble
- PottsMPNN-ft: pottsmpnn -> pottsmpnn_ft
- GCPNet-vframe (probing only): gcpnet_random_vframe -> gcpnet_pretrained_vframe, the frame-projected re-extraction.

The GearNet pair differs in size as well as in training; any GearNet difference mixes the two.

## Probes and metric

The fifteen probes of the admissible floor comparison (`symmetric/cells_superfamily.csv`, rule == "admissible"):
the eleven determined labels (phi_circular, psi_circular, local_curvature, dssp_3, dssp_8, core_surface,
contact_density, distance_bins, contact_order_len_resid, radius_of_gyration_len_resid, asphericity_len_resid) and the
four undetermined labels (rsa, rsa_bins, bfactor, architecture_label). Superfamily split. Headline metric per probe as
in `scripts/symmetric_run.py`: circular cosine for phi and psi, Pearson r for regression probes, balanced accuracy for
classification probes.

## Score of one source

For each source separately, the readout family (linear, or the mean of the three MLP seeds) is the one with the higher
validation headline score (`symmetric_decide.choose` applied to that source alone). Its test score is read once from
`results_superfamily/by_source/<source>.csv`. The random and the pretrained version may end up with different families.

## Difference and interval

delta_train = S_test(pretrained) - S_test(random).

Interval: the protein-level paired bootstrap of `scripts/symmetric_decide.py` (`Boot`), reused without change: 2,000
draws, the resampling unit is the test protein (residue and pair probes are resampled by protein), one fixed weight
matrix per probe (generator seed = sum of the character codes of the probe name), the same weight matrix applied to
both sources, the per-draw score of a family computed from the saved test predictions (mean over MLP seeds for the
nonlinear family). The 95% interval is the 2.5th and 97.5th percentile of the per-draw differences; the 99.9% interval
(0.05th and 99.95th percentile) is reported as secondary.

Decision per (pair, probe):

- increase: the 95% interval lies entirely above zero;
- decrease: the 95% interval lies entirely below zero;
- unresolved: otherwise.

If the saved predictions of the two sources do not share the same test rows (labels and proteins) for a probe, no
interval is reported for that probe and only the point estimate is given.

## Floor context

For every (pair, probe) the input-matched admissible floor of each version is taken from `cells_superfamily.csv`
(floor, floor_test, gap, 95% interval, state), so each row reads floor -> random -> pretrained. For the vframe pair the
floor comes from `symmetric/vframe_superfamily.csv`. Per pair, the number of the eleven determined-label cells whose
state is "below" is reported for the random and for the pretrained version, with the number of determined cells that
change state between the two.

## Per-pair summary

Number of the fifteen probes with increase, decrease and unresolved; the same for the eleven determined labels and the
four undetermined labels separately; below-floor counts as above.

## Downstream

For each pair and each of the ten downstream tasks, delta_train = score(pretrained) - score(random) from
`extract/scores2_selected.csv` (three-seed mean of the validation-selected readout), with each version's seed standard
deviation. Per-protein downstream test predictions for the random encoders were searched for under `extract/` and are
not saved there (the only saved downstream predictions, in `confirmation_downstream/`, cover pretrained encoders only
and a different protocol), so no paired bootstrap is possible for downstream. Reported instead: the standard error of
the difference of the two three-seed means, sqrt(sd_r^2/n_r + sd_p^2/n_p), and a descriptive flag when |delta| exceeds
twice that value. This flag reflects readout-seed variation only, not test-set sampling, and is not a test.

Disclosure: while checking file structure before writing this file, the downstream selected scores of the twelve pair
encoders were printed once. No probing pair difference had been computed.

## Outputs

`symmetric/ab_train_pairs/`: probing_cells.csv, probing_summary.csv, downstream.csv, train_pairs.tex, train_pairs.png,
run log.
