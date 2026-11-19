"""Bootstrap over architecture families of the sequence-versus-coordinate score gap, per restriction of
scripts/group_controls.py.

The statistic is the one reported next to the rank permutation in the group comparison: per task, the mean over the
sequence-side families of their family mean score minus the same mean over the coordinate-side families, averaged
over the ten tasks. Families of each side are resampled with replacement (10,000 draws, seed 0) and the 2.5th and
97.5th percentiles are reported.

  python scripts/group_family_bootstrap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_controls as G  # noqa: E402

DRAWS, SEED = 10000, 0


def main():
    d = pd.read_csv(G.ROOT / "extract" / "scores2_selected.csv")
    w = d.pivot(index="encoder", columns="task", values="score")
    m = G.meta().reindex([e for e in w.index if e != G.REF])
    tasks = [t for t in G.ORDER if t in w.columns]
    rows = []
    for name, mask in G.restrictions(m).items():
        sel = m[mask]
        S = w.loc[sel.index[sel.group == "sequence"], tasks].groupby(sel.family[sel.group == "sequence"]).mean()
        C = w.loc[sel.index[sel.group == "coordinate-side"], tasks].groupby(sel.family[sel.group == "coordinate-side"]).mean()
        rng = np.random.default_rng(SEED)
        v = [float((S.iloc[rng.integers(0, len(S), len(S))].mean() - C.iloc[rng.integers(0, len(C), len(C))].mean()).mean())
             for _ in range(DRAWS)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append(dict(restriction=name, families_seq=len(S), families_crd=len(C),
                         family_gap=float((S.mean() - C.mean()).mean()), ci_low=float(lo), ci_high=float(hi)))
    out = pd.DataFrame(rows)
    out.to_csv(G.ROOT / "extract" / "group_family_bootstrap.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
