"""Per-cell and per-task summary of the add-back study with variance-matched null arms.

Reads erasure/addback_rescue_matched.csv (scripts/addback_rescue.py --nulls) and writes

  erasure/addback_rescue_matched_cells.csv     one row per task and encoder
  erasure/addback_rescue_matched_summary.csv   one row per task

For a reference arm family (sham or null), the effect is the mean score of the reference arms minus the score of the
erased arm, both without a descriptor, and R is the repair of the real erasure minus the mean repair of the reference
arms. var_er and var_nu are the shares of representation variance removed by the erasure and by the null arms.

  python addback_matched_summary.py [--src FILE] [--no-write]
"""
import argparse
from pathlib import Path

import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
DESCS = ("named", "permuted", "other")


def cells(d):
    out = []
    for (task, enc), g in d.groupby(["task", "encoder"]):
        piv = g.pivot_table(index="arm", columns="descriptor", values="linear")
        var = g.groupby("arm").var_removed.mean()
        row = dict(task=task, encoder=enc, var_er=var["erased"],
                   var_nu=float(np.mean([var[a] for a in var.index if a.startswith("null")])))
        for fam in ("sham", "null"):
            arms = [a for a in piv.index if a.startswith(fam)]
            row[f"effect_{fam}"] = float(np.mean([piv.loc[a, "none"] for a in arms])) - piv.loc["erased", "none"]
            for q in DESCS:
                real = piv.loc["erased", q] - piv.loc["erased", "none"]
                ref = float(np.mean([piv.loc[a, q] - piv.loc[a, "none"] for a in arms]))
                row[f"R_{q}_{fam}"] = real - ref
        out.append(row)
    c = pd.DataFrame(out)
    return c[["task", "encoder", "var_er", "var_nu", "effect_sham", "R_named_sham", "R_permuted_sham", "R_other_sham",
              "effect_null", "R_named_null", "R_permuted_null", "R_other_null"]]


def summary(c):
    out = []
    for task, g in c.groupby("task"):
        out.append(dict(task=task, n=len(g), var_er=g.var_er.mean(), var_nu=g.var_nu.mean(),
                        eff_sham=g.effect_sham.mean(), n_eff_sham=int((g.effect_sham > 0).sum()),
                        eff_null=g.effect_null.mean(), n_eff_null=int((g.effect_null > 0).sum()),
                        R_sham=g.R_named_sham.mean(), n_R_sham=int((g.R_named_sham > 0).sum()),
                        R_null=g.R_named_null.mean(), n_R_null=int((g.R_named_null > 0).sum()),
                        Rperm_null=g.R_permuted_null.mean(), Roth_null=g.R_other_null.mean()))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "erasure" / "addback_rescue_matched.csv"))
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    c = cells(pd.read_csv(a.src))
    s = summary(c)
    pd.set_option("display.width", 220)
    print(s.round(4).to_string(index=False))
    if not a.no_write:
        stem = a.src[:-len(".csv")]
        c.to_csv(stem + "_cells.csv", index=False)
        s.to_csv(stem + "_summary.csv", index=False)
        print("written:", stem + "_cells.csv", stem + "_summary.csv")


if __name__ == "__main__":
    main()
