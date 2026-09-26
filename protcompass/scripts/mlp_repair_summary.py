"""Repair contrast with an MLP-2 head against the linear head, on the same 12 encoders and the 8 pairs of
erasure/named_pair_mlp_pairs.csv. Reads erasure/named_pair_cells_mlp.csv (named_pair_verdicts.py
--src erasure/named_pairs_mlp --tag mlp) and erasure/named_pair_cells.csv; writes erasure/named_pair_mlp_summary.csv.
"""
import pandas as pd
from scipy.stats import binomtest

L = pd.read_csv("erasure/named_pair_cells.csv"); M = pd.read_csv("erasure/named_pair_cells_mlp.csv")
V = pd.read_csv("erasure/named_pair_verdicts_mlp.csv").set_index(["task", "concept"])
encs = sorted(M.encoder.unique())
rows = []
for (t, c), g in M.groupby(["task", "concept"]):
    l = L[(L.task == t) & (L.concept == c) & L.encoder.isin(encs)]
    kl = int((l.R_null > 0).sum())
    rows.append(dict(task=t, concept=c, n=len(g), R_var_mlp=g.R_null.mean(), pos_mlp=int((g.R_null > 0).sum()),
                     p_holm_mlp=V.loc[(t, c), "p_null_holm"], passes_mlp=bool(V.loc[(t, c), "passes_R_rule"]),
                     R_var_linear=l.R_null.mean(), pos_linear=kl, p_linear=binomtest(kl, len(l)).pvalue))
pd.DataFrame(rows).to_csv("erasure/named_pair_mlp_summary.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
