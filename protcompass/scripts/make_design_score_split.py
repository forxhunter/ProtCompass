"""Split the probing TEST split in half by protein, into a `design` half and a `score` half.

Why this exists. The eight reference floors were built one at a time, each after the previous one was shown to be too
weak, and each was judged on the test split. The reported "best floor" is therefore a maximum over eight descriptors
read on the same data that scores it, with no held-out design set. This script gives the suite one: descriptor choice
happens on `design`, the number that is reported comes from `score`, and the two halves share no protein.

The unit of the split is the protein, never the residue. Residues of one protein stay together, and for the pair-level
probe `distance_bins` a pair follows the protein it came from (both residues of a sampled pair are in the same protein
by construction, see probing_data: pairs are drawn within a domain). One assignment covers all three probe levels:

  residue level   the 1,000 test domains sampled by run_probing_suite.Data (seed 0)
  pair level      pairs whose protein is one of those 1,000 domains
  protein level   all 1,948 test-split proteins (the protein probes use the whole split, not the sample)

Assignment is a stratified random half-split with a fixed seed. The strata are (is the protein in the 1,000-domain
residue sample, CATH architecture label), so that (a) the residue sample splits 500/500 rather than by chance, and
(b) every architecture class present in test is present in both halves, which matters because `architecture_label` is
a 25-class probe scored by balanced accuracy on ~960 proteins per half. Stratifying on the architecture label uses the
label, but the assignment is fixed once, before any score is read, and applies identically to every floor and every
encoder, so it cannot favour one source over another.

  python make_design_score_split.py

Writes probing_results/design_score_assignment.csv (one row per test-split protein).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R

SEED = 20260920
OUT = R.OUT / "design_score_assignment.csv"


def build():
    data = R.Data()
    test_prot = data.prot_idx["test"]                      # every protein in the test split
    residue_sample = set(int(i) for i in data.chosen["test"])   # the 1,000 domains the residue probes use
    arch = data.P[f"architecture_label_{R.SPLIT}"][:].astype(np.int64)
    pid = data.P["id"][:]

    rng = np.random.default_rng(SEED)
    half = {}
    # Stratify by (in residue sample, architecture label). Odd strata alternate which half gets the extra protein so
    # the totals stay balanced overall instead of all going one way.
    parity = 0
    keys = sorted({(int(i) in residue_sample, int(arch[i])) for i in test_prot})
    for key in keys:
        members = np.array([i for i in test_prot if (int(i) in residue_sample, int(arch[i])) == key])
        order = rng.permutation(len(members))
        n_design = len(members) // 2 + (1 if (len(members) % 2 and parity == 0) else 0)
        parity = 1 - parity if len(members) % 2 else parity
        for rank, m in enumerate(members[order]):
            half[int(m)] = "design" if rank < n_design else "score"

    rows = [{"protein_index": int(i), "protein_id": pid[i].decode(), "length": int(data.L[i]),
             "in_residue_sample": int(i) in residue_sample, "architecture_label": int(arch[i]),
             "half": half[int(i)]} for i in test_prot]
    df = pd.DataFrame(rows).sort_values("protein_index").reset_index(drop=True)
    return df


def main():
    df = build()
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} test-split proteins, seed {SEED})")
    print(df.groupby(["in_residue_sample", "half"]).size().to_string())
    print("\nresidue-sample residues per half:")
    print(df[df.in_residue_sample].groupby("half").length.agg(["count", "sum"]).to_string())
    print("\narchitecture classes present in both halves: "
          f"{len(set(df[df.half == 'design'].architecture_label) & set(df[df.half == 'score'].architecture_label))} of "
          f"{df.architecture_label.nunique()}")


if __name__ == "__main__":
    main()
