# Probing dataset schema

Source: CATH v4.3.0. Domains are the S35 representatives (`cath-domain-list-S35-v4_3_0.txt`, one domain per 35% sequence-identity
cluster) that have a coordinate file in the official S40 non-redundant domain PDB set
(`cath-dataset-nonredundant-S40-v4_3_0.pdb.tgz`). CATH ships domain coordinates only for its S20 and S40 sets, so S35
representatives outside the S40 set are not included; a subset of a 35%-non-redundant set is still non-redundant at 35%.

Build: `~/miniconda3/envs/protein_embeddings/bin/python make_probing_data.py --all`.
Reader: `probing_dataset.py`. Reference-row probes and statistics: `diagnose_probing_data.py` -> `DIAGNOSTICS.md`.

## Files

| Path | Content |
|---|---|
| `raw/` | Unmodified CATH files (lists, names, S40 tarball, extracted `dompdb/`) |
| `structures/<domain>.pdb` | Cleaned domain, heavy atoms, chain A, residues renumbered 1..L, element column added |
| `labels.h5` | Coordinates, sequence, all labels, splits (layout below) |
| `splits/<split>_<part>.txt` | Domain ids; `split` is `superfamily` or `topology`, `part` is `train`, `val`, `test` |
| `splits/subset_in_s20.txt` | Kept domains that are also in the official S20 set (pairwise identity at most 20%) |
| `exclusions.csv` | Every S35 representative that was not kept, with reason and detail |
| `work/shard_*.pkl` | Per-domain intermediate results; safe to delete after `assemble` |

## Cleaning and filters

- Only `ATOM` records; hydrogens dropped; for alternate locations the first atom record of each atom name is kept.
- `SEC` is renamed `CYS` and `MSE` is renamed `MET`, with the selenium atom renamed to the sulfur atom name. Other non-standard
  residues (mostly `UNK`) are removed. A domain is excluded if more than 2% of its residues are non-standard.
- Residues without all of N, CA, C, O are removed. A domain is excluded if more than 5% of its residues are incomplete.
- Length after cleaning must be 40 to 600.
- Domains for which mkdssp fails or assigns fewer than 95% of residues are excluded.
- A chain break is a C(i)-N(i+1) distance above 1.6 A. CATH domains are often discontinuous, so breaks are recorded
  (`residue/segment`, `protein/n_segments`) and not used as an exclusion criterion.

## `labels.h5`

Residue arrays are concatenated over domains. Domain `k` owns rows `offset[k] : offset[k] + length[k]`. Undefined labels are
`-1` for integer arrays and `NaN` for float arrays; they are never imputed.

### `protein/` (one row per domain)

| Dataset | Type | Meaning |
|---|---|---|
| `id` | S7 | CATH domain id |
| `length`, `offset` | int | Residues kept; first row in the residue arrays |
| `cath_C`, `cath_A`, `cath_T`, `cath_H` | int32 | CATH class, architecture, topology, homologous superfamily numbers |
| `resolution` | float32 | From the CATH list; 999 for NMR, 1000 for obsolete entries |
| `in_s20` | bool | Domain is also in the official S20 non-redundant set |
| `n_segments`, `n_virtual_cb`, `n_incomplete`, `n_nonstd` | int32 | Backbone segments; residues with a constructed CB; residues removed for incomplete backbone or non-standard type |
| `bfactor_ok` | bool | False when CA B-factors are constant or all zero (for example NMR models) |
| `split_superfamily`, `split_topology` | int8 | 0 train, 1 val, 2 test |
| `contact_order` | float32 | Relative contact order: sum of `|i-j|` over contacts, divided by (number of contacts x L); contact = CA-CA < 8 A and `|i-j| > 3`, each pair counted once |
| `radius_of_gyration` | float32 | A; root mean square distance of CA atoms from their centroid |
| `asphericity` | float32 | Relative shape anisotropy of the CA gyration tensor, `1.5 * sum(l_i^2) / (sum l_i)^2 - 0.5`, 0 for a sphere, 1 for a rod |
| `asphericity_legacy` | float32 | Formula of the original code, `(l1 - l3)^2 / (l1 + l2 + l3)^2` |
| `<target>_len_resid_<split>` | float32 | `y - (a + b log L + c (log L)^2)`, coefficients by OLS on the training part of that split (attribute `coef_a_b_c`) |
| `architecture_label_<split>` | int16 | Dense index of the CATH architecture `C.A`; `-1` when the architecture has fewer than 20 training domains. Attributes `classes` and `dropped` list the `C.A` codes |

### `residue/` (one row per residue)

| Dataset | Type | Probe (IIP) | Meaning |
|---|---|---|---|
| `protein_index`, `position` | int | | Row of the domain in `protein/`; 0-based index within the domain |
| `coords` | float32 (5,3) | | N, CA, C, O, CB in A. CB is built from ideal geometry when absent (`cb_virtual`), always for glycine |
| `aa` | int8 | | Residue type, alphabet `ACDEFGHIKLMNPQRSTVWY` (project convention) |
| `resseq`, `icode` | int32, S1 | | Residue number and insertion code in the original PDB numbering |
| `segment` | int16 | | Backbone segment index, increases at each chain break |
| `phi`, `psi` | float32 | | Radians in (-pi, pi]; NaN at termini and across breaks |
| `phi_sin`, `phi_cos`, `psi_sin`, `psi_cos` | float32 | | For circular regression |
| `phi_bin`, `psi_bin` | int8 | `phi_angle_bins`, `psi_angle_bins` (I1) | 36 bins of 10 degrees, bin 0 starts at -180 |
| `curvature` | float32 | `local_curvature` (I2) | Radians in [0, pi]: angle between `CA(i)-CA(i-1)` and `CA(i+1)-CA(i)`; NaN at termini and next to breaks |
| `dssp8` | int8 | `dssp_8` (I3) | mkdssp 3.0.0; classes `G,H,I,B,E,T,S,C` = 0..7 (blank is C) |
| `dssp3` | int8 | `dssp_3` (I3) | `G,H,I -> H (0)`, `B,E -> E (1)`, `T,S,C -> C (2)` |
| `sasa` | float32 | | A^2; freesasa, Lee-Richards, probe 1.4 A, ProtOr radii, heavy atoms, computed on the isolated domain |
| `rsa` | float32 | `rsa` (I4) | `sasa / max ASA` (Tien et al. 2013 theoretical values), clipped to [0, 1] |
| `rsa_bin` | int8 | `rsa_bins` (I4) | 0 buried `< 0.2`, 1 intermediate `0.2-0.5`, 2 exposed `> 0.5` |
| `bfactor_raw`, `bfactor_z` | float32 | `bfactor` (I5) | CA B-factor; `bfactor_z` is z-scored within the chain and NaN when `bfactor_ok` is false |
| `cb_neighbours` | int16 | | Number of other residues with CB within 10 A of this residue's CB |
| `core` | int8 | `core_surface` (I6) | 1 core if `cb_neighbours >= 20`, else 0 surface |
| `contact_count` | int16 | | Number of other residues with CA within 8 A (self excluded, sequence neighbours included) |
| `contact_density_bin` | int8 | `contact_density` (I6) | `contact_count` in 0-5, 6-10, 11-15, 16+ = 0..3 |
| `contact_count_sep4` | int16 | | Count of the original code: CA within 8 A and `|i-j| > 3`. With this definition the 16+ class is empty |
| `sequence_recovery` | int8 | `sequence_recovery` (I10) | Same as `aa` |

### `pair/` (fixed sample of residue pairs)

Up to 512 pairs per domain, drawn uniformly without replacement from all pairs with index separation `j - i >= 6`, with
`numpy.random.default_rng(crc32(domain id))`. Every encoder is probed on the same pairs.

| Dataset | Type | Meaning |
|---|---|---|
| `protein_index` | int32 | Row of the domain in `protein/` |
| `offset`, `count` | int | Per domain: first pair row and number of pairs |
| `i`, `j` | int16 | 0-based residue indices, `i < j` |
| `distance` | float32 | CA-CA distance in A |
| `distance_bin` | int8 | `distance_bins` (I7): 16 bins of 2.5 A over 0-40 A, distances of 40 A or more fall in bin 15 |

## Splits

Whole groups are assigned to train / val / test (target 80 / 10 / 10 of the domains), separately within each CATH class so the
class mix is the same in every part. Groups are processed from largest to smallest (ties broken with the seeded generator) and
each goes to the part with the largest relative deficit. Seed 20260919.

- `superfamily`: groups are CATH homologous superfamilies `C.A.T.H`. No superfamily appears in two parts.
- `topology`: groups are CATH topologies `C.A.T`. No fold appears in two parts. This is the stricter check.

Architecture classes are shared across parts in both splits by construction.

## Changes against the original label file

| Probe | Original (`src/data`, `src/probes`) | Here |
|---|---|---|
| `contact_density` | Raw neighbour count 0-18 stored as the label of a 4-class probe; count excluded `|i-j| <= 3`, which leaves the 16+ class empty | Count of all residues within 8 A, binned into the four classes of the manuscript |
| `core_surface` | Three labels (core, boundary, surface) from a CA-count accessibility proxy, probed as binary | Binary label from the CB neighbour count |
| `rsa`, `rsa_bins` | Neighbour-count proxy for accessibility | SASA from freesasa, normalised by maximum ASA |
| `dssp_3`, `dssp_8` | Source not recoverable on this machine | mkdssp 3.0.0 on the cleaned domain |
| `bfactor` | Raw values | Z-scored within the chain, masked when uninformative |
| `contact_order`, `radius_of_gyration`, `asphericity` | Raw values | Raw and length-residualised values |
| `asphericity` | `(l1 - l3)^2 / (sum l)^2` | Relative shape anisotropy; the legacy value is kept |
| `architecture_label` | Hard-coded map of 31 sparse labels | `C.A` classes with at least 20 training domains, mapping stored in the file |
| `distance_bins` | Pairs re-sampled at run time | Fixed pairs stored in the file |
| glycine CB | Zeros | Ideal-geometry virtual CB |
| splits | 1,000 / 200 / 200 proteins, validation score reported | Superfamily-disjoint and topology-disjoint 80 / 10 / 10 with a held-out test part |

The core threshold of 20 CB neighbours within 10 A is the midpoint of the 16 (surface) and 24 (core) neighbour-count cutoffs
used for layer assignment in protein design software. Check this convention against a citable source before it goes into the paper.
