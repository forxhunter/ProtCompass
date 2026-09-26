# Edits to the released pre-specification files

The pre-specification files (`PRESPEC*.md`) were written before the scores of their analyses were computed.
The copies released here were edited for anonymity only: a local path and wording that described how the
analysis was organised (who asked for a row or a question) were replaced. Statistics, thresholds, decision rules,
outputs and dates are unchanged in meaning. Every changed line is listed below. Where the original wording
described how the work was organised, it is summarised in square brackets instead of quoted.

The `.sha256` files are the checksums recorded before scoring. They refer to the unedited originals, so a
released copy listed below no longer matches its checksum. The originals will be released with the
de-anonymised version of the paper.

Checksum files: `erasure/ab_fixed_head/PRESPEC.sha256`, `erasure/ab_stability_decomp/PRESPEC.sha256`, `erasure/ab_stability_decomp/PRESPEC_addendum.sha256`, `extract/ab_modality/PRESPEC.sha256`, `extract/ab_modality/PRESPEC_shuf.sha256`, `symmetric/ab_train_pairs/PRESPEC.sha256`.

| File | Line | Original | Released |
|---|---:|---|---|
| `erasure/ab_stability_decomp/PRESPEC_addendum.md` | 8 | These rows were [a clause naming the role that requested the rows, and a sentence on who chose them] Before this file was written I had | These rows were added to the specification after the main run had started. Before this file was written I had |
| `extract/ab_modality/PRESPEC.md` | 26 | Environment: `<home>/miniconda3/envs/protein_embeddings/bin/python`, `PYTHONPATH=<working directory>/pylib` | Environment: `~/miniconda3/envs/protein_embeddings/bin/python`, `PYTHONPATH=protcompass/pylib` |
| `extract/ab_modality/PRESPEC_shuf.md` | 9 | [a clause naming the role that asked the question] whether `saprot_seqonly` (every 3Di token replaced by `#`) is a valid counterfactual or an | The question is whether `saprot_seqonly` (every 3Di token replaced by `#`) is a valid counterfactual or an |

Files not listed here are identical to the originals.
