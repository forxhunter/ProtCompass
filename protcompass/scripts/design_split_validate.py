"""Check that the design/score rerun reproduces the published scores on the full test set.

The rerun exists only to partition the evaluation set. If it changed anything about training, the full-test column
would not match probing_results/by_source/<source>.csv. This script compares every (source, probe, readout, seed) cell
and prints the largest discrepancy per metric, plus any cell missing from either side.

It also checks the arithmetic of the partition: for a metric that is a mean over evaluation rows (accuracy), the
full-test value has to equal the row-count-weighted mean of the two halves.

  python design_split_validate.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path((PROTCOMPASS_ROOT + "/protcompass/probing_results"))
METRICS = ["pearson", "spearman", "r2", "circ_cos", "circ_mae_deg", "accuracy", "balanced_accuracy", "macro_f1"]
KEY = ["source", "probe", "readout", "seed"]


def main():
    old = pd.concat([pd.read_csv(f) for f in sorted((RES / "by_source").glob("*.csv"))], ignore_index=True)
    new = pd.concat([pd.read_csv(f) for f in sorted((RES / "by_source_split").glob("*.csv"))], ignore_index=True)
    print(f"published cells {len(old)}, rerun cells {len(new)} ({len(new[new.eval_set == 'test'])} on full test)")
    print(f"sources: published {old.source.nunique()}, rerun {new.source.nunique()}")
    miss = sorted(set(old.source) - set(new.source))
    if miss:
        print(f"MISSING from the rerun: {miss}")

    t = new[new.eval_set == "test"]
    m = old.merge(t, on=KEY, suffixes=("_old", "_new"), how="inner")
    print(f"\nmatched {len(m)} cells")
    unmatched = len(old[old.source.isin(set(new.source))]) - len(m)
    print(f"unmatched among shared sources: {unmatched}")
    worst = []
    for c in METRICS:
        if f"{c}_old" in m:
            d = (m[f"{c}_old"] - m[f"{c}_new"]).abs()
            if d.notna().any():
                i = d.idxmax()
                worst.append((c, float(np.nanmax(d)), int(d.notna().sum()), m.loc[i, "source"], m.loc[i, "probe"]))
    print(f"\n{'metric':20s} {'max |published - rerun|':>24s} {'cells':>7s}   where")
    for c, v, n, s, p in worst:
        print(f"{c:20s} {v:24.3e} {n:7d}   {s} / {p}")

    print("\npartition arithmetic (accuracy must be the row-weighted mean of the halves):")
    w = new.pivot_table(index=KEY + ["level"], columns="eval_set", values=["accuracy", "n_eval"])
    ok = w[("accuracy", "test")].notna()
    w = w[ok]
    recon = (w[("accuracy", "design")] * w[("n_eval", "design")] + w[("accuracy", "score")] * w[("n_eval", "score")]) \
        / (w[("n_eval", "design")] + w[("n_eval", "score")])
    err = (recon - w[("accuracy", "test")]).abs()
    print(f"  {len(err)} classification cells, max error {err.max():.3e}")
    n_sum = (w[("n_eval", "design")] + w[("n_eval", "score")] - w[("n_eval", "test")]).abs().max()
    print(f"  design rows + score rows == test rows on every cell: {'yes' if n_sum == 0 else f'NO, off by {n_sum}'}")


if __name__ == "__main__":
    main()
