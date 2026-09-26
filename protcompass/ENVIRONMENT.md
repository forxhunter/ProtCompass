# Computational environment of the release

Recorded 2026-09-25. Paths are relative to this directory (`protcompass/`). Machine names, user paths and scheduler
details of the original record are omitted from this copy.

## Main environment (local machine, all released tables unless listed below)

- Conda environment: `protein_embeddings`
  - full export: `environment_protein_embeddings_20260925.yml`
  - pip lock: `requirements_lock_protein_embeddings_20260925.txt`
- Python 3.10.20, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3, PyTorch 2.11.0 (CUDA 13.0),
  transformers 4.46.3
- GPU: NVIDIA RTX 5090, driver 610.57.04
- The analysis runs put a local `pylib/` directory on `PYTHONPATH`. It held a pip install of sentencepiece 0.2.2. It
  is not part of this release; install the package with `pip install sentencepiece==0.2.2` instead. The FoldVision
  and CDConv environments used two similar directories: `pylib_foldvision/` (h5py 3.16.0) and `extract/pylib_cdc5090/`
  (h5py 3.16.0, pandas 3.0.6, python-dateutil 2.9.0.post0, pytz 2026.3.post1). They are not released either.

## Outputs computed on a second machine

Part of the repair-grid shards ran on a second machine, from a copy of this directory whose scripts differ only in
the root path. That machine used Python 3.14, NumPy 2.5.3, SciPy 1.18.1 and scikit-learn 1.9.1.

The affected outputs are:
- `erasure/named_pairs_full_remote/` and `erasure/named_pairs_full_remote_partial/`: the fluorescence and PPI-affinity
  cells of the grid, and some membrane and solubility cells. They feed `erasure/named_pair_cells_full.csv` and
  `named_pair_verdicts_full.csv`.
- `erasure/ppi_cv_shape_remote/` and `erasure/ppi_cv_size_remote/`: part of the chain-disjoint PPI cross-validation.

**Consequence.** On fluorescence and PPI affinity, the erasure fitted on the second machine removes a different share
of variance from the local refit, by up to 0.34 on one cell. Code, embeddings, property caches, manifests and pairs
have identical md5 on both machines. The only identified difference is the numerical-library environment. Both runs
keep the same pairs under the matched-null effect rule (Appendix, matched-null subsection).

## Analyses added on 2026-09-25 (all local, main environment)

| Directory | Contents |
|---|---|
| `symmetric/ab_train_pairs/` | random against pretrained pairs |
| `erasure/ab_fixed_head/` | fixed-head erasure |
| `erasure/ab_stability_decomp/` | stability composition decomposition |
| `extract/ab_modality/` | ESM3 and SaProt within-model modality ablation, with the SaProt shuffled-3Di control |

Each directory holds `PRESPEC.md` and its sha256, both written before the scores were computed. The released copies
were edited for anonymity only; `PRESPEC_REDACTIONS.md` lists every changed line and explains how the checksums relate
to the originals.

## SaProt extraction cells computed on the second machine (2026-09-25, from 16:06)

To shorten the GPU path of the modality ablation, part of the SaProt extraction ran on the second machine (RTX PRO
6000 Blackwell) in a separate conda environment. That environment pins the same NumPy, SciPy, scikit-learn, pandas
and transformers versions as the local lock. PyTorch is 2.11.0+cu128 (a CUDA 12.8 build), because driver 570 cannot
run cu130. Coordinates come from copies of the local `extract/cache/*.npz`.

**Gate.** PPI affinity was recomputed on the second machine and compared with the local embeddings.

| Source | Relative Frobenius difference | Minimum row cosine | Where it runs |
|---|---|---|---|
| saprot_seqonly | 3.3e-7 | 1.000000 | may run remotely |
| esm3_seqonly | 3.3e-3 | 0.99936 | local only (bf16) |

**Remote cells.** They ran as one batch job on a shared GPU (one of four GPU shares).
- saprot_shuf3di: the probing embeddings, and downstream PPI affinity, stability and solubility.
- saprot_seqonly: membrane vs soluble.

A first attempt at 16:06 ran GPU processes outside the batch scheduler. The machine's GPU watchdog ended them at
16:11:30, and none of them wrote output.

Directories were copied back only when no local directory existed. Work on the second machine was paused at 16:30,
and the saprot_shuf3di probing extraction then ran locally.

**Final list.** Four downstream embedding directories were computed on the second machine and copied into the local
tree:
- `extract/ab_modality/embeddings/ppi_affinity/saprot_shuf3di` (md5 checked)
- `extract/ab_modality/embeddings/stability/saprot_shuf3di`
- `extract/ab_modality/embeddings/solubility/saprot_shuf3di`
- `extract/ab_modality/embeddings/membrane_soluble/saprot_seqonly`

No probing embeddings and no ESM3 output came from the second machine. Scoring, bootstraps and probing fits for every
cell ran locally.
