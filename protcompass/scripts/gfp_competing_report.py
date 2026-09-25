"""Summarise the GFP competing-explanation panel (scripts/gfp_competing.py) with the pre-specified decision rule.

Per encoder and property, from arms computed in one run (so both conditions share one basis of the erased arm):
  R_var      = [S(E_p X + q) - S(E_p X)] - mean_null [S(E_c X + q) - S(E_c X)]                 (no covariates)
  R_var|C    = the same with the competing covariates C in every arm                            (primary)
  Rperm|C    = R_var|C with the descriptor's rows permuted across proteins                      (control)
  inc|C      = S(X + C + q) - S(X + C) on the intact representation
Decision (PRESPEC.md): retained if R_var|C passes a two-sided sign test over encoders at Holm p < 0.05 (two
properties) with mean >= 0.02; attenuated if the sign test passes below 0.02; explained otherwise.

  python gfp_competing_report.py      # writes erasure/gfp_competing/{cells_summary,verdicts}.csv
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from family_weighted_demand import family, holm                         # noqa: E402
from panel import only_panel                                            # noqa: E402

DIR = Path((PROTCOMPASS_ROOT + "/protcompass/erasure/gfp_competing"))
FLOOR = 0.02


def per_encoder(d):
    rows = []
    for (concept, enc), g in d.groupby(["concept", "encoder"]):
        rec = dict(concept=concept, encoder=enc, family=family(enc))
        for cv in ("none", "C"):
            p = g[g.covariates == cv].pivot_table(index="arm", columns="descriptor", values="score")
            nulls = [a for a in p.index if a.startswith("null")]
            if "erased" not in p.index or not nulls:
                continue
            sfx = "" if cv == "none" else "_C"
            rec[f"R_var{sfx}"] = (p.loc["erased", "named"] - p.loc["erased", "none"]) - \
                (p.loc[nulls, "named"] - p.loc[nulls, "none"]).mean()
            rec[f"Rperm{sfx}"] = (p.loc["erased", "permuted"] - p.loc["erased", "none"]) - \
                (p.loc[nulls, "permuted"] - p.loc[nulls, "none"]).mean()
            rec[f"S_erased{sfx}"] = p.loc["erased", "none"]
            if cv == "C" and "named" in p.columns and not np.isnan(p.loc["intact", "named"]):
                rec["inc_C"] = p.loc["intact", "named"] - p.loc["intact", "none"]
                rec["S_intact_C"] = p.loc["intact", "none"]
            if cv == "none":
                rec["S_intact"] = p.loc["intact", "none"]
        rows.append(rec)
    return only_panel(pd.DataFrame(rows))


def main():
    d = pd.concat([pd.read_csv(f) for f in sorted(DIR.glob("cells_*_repro.csv"))], ignore_index=True)
    E = per_encoder(d).dropna(subset=["R_var", "R_var_C"])
    E.to_csv(DIR / "cells_summary.csv", index=False)
    rows = []
    for c, g in E.groupby("concept"):
        rec = dict(concept=c, n=len(g))
        for col in ("R_var", "R_var_C", "Rperm_C", "inc_C"):
            v = g[col].dropna()
            k = int((v > 0).sum())
            rec[f"{col}_mean"], rec[f"{col}_pos"], rec[f"{col}_p"] = float(v.mean()), k, binomtest(k, len(v), 0.5).pvalue
        fam = g.groupby("family")[["R_var", "R_var_C"]].mean()
        for col in ("R_var", "R_var_C"):
            k = int((fam[col] > 0).sum())
            rec[f"{col}_fam_pos"], rec["n_fam"], rec[f"{col}_fam_p"] = k, len(fam), binomtest(k, len(fam), 0.5).pvalue
        rec["ratio_C_to_none"] = rec["R_var_C_mean"] / rec["R_var_mean"] if rec["R_var_mean"] else np.nan
        nr = DIR / f"no_representation_{'comp' if c.startswith('amino') else 'burial'}.csv"
        if nr.exists():
            r = pd.read_csv(nr).iloc[0]
            rec["S_C"], rec["S_C_plus_q"], rec["S_q"] = r.S_C, r.S_C_plus_q, r.S_q
        rows.append(rec)
    V = pd.DataFrame(rows)
    for col in ("R_var", "R_var_C"):
        V[f"{col}_p_holm"] = holm(V[f"{col}_p"].to_numpy())
    V["verdict"] = np.where((V.R_var_C_p_holm < 0.05) & (V.R_var_C_mean >= FLOOR), "retained",
                            np.where(V.R_var_C_p_holm < 0.05, "attenuated", "explained by the competitors"))
    V.to_csv(DIR / "verdicts.csv", index=False)
    pd.set_option("display.width", 250)
    print(V.round(4).T.to_string())


if __name__ == "__main__":
    main()
