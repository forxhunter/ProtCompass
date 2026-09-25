"""Read the calibration of scripts/matched_null_calibration.py: what each rule names when the answer is known.

For every label and every erased property the script computes, over the encoders whose erasure verified, the effect
against the sham, the effect against the variance-matched null and the repair contrast against each, and then applies
two rules within each label with Holm correction over the properties:

  effect rule   the released rule: sign test on the effect against the sham, mean at least 0.02
  repair rule   the same thresholds on the repair contrast against the variance-matched null

A calibrated rule names the planted property, names nothing else, and names nothing under `none`.

  python matched_null_calibration_report.py [--src ...] [--latex]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
FLOOR = 0.02
SHORT = {"amino-acid composition": "composition", "secondary structure content": "secondary structure",
         "global shape": "global shape", "burial and packing": "burial and packing", "contact order": "contact order",
         "size": "size", "hydrophobic fraction": "hydrophobic fraction", "charge": "charge"}


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    out, run = np.empty_like(ps), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


def cells(d):
    rows = []
    for (enc, concept, label), g in d.groupby(["encoder", "concept", "label"]):
        p = g.pivot_table(index="arm", columns="descriptor", values="score")
        sh = [a for a in p.index if a.startswith("sham")]
        nu = [a for a in p.index if a.startswith("null")]
        if "erased" not in p.index or not sh or not nu:
            continue
        e0, en, ep = p.loc["erased", "none"], p.loc["erased", "named"], p.loc["erased", "permuted"]
        rec = dict(encoder=enc, concept=concept, label=label, erasure_ok=bool(g.erasure_ok.iloc[0]))
        for k, arms in (("sham", sh), ("null", nu)):
            rec[f"effect_{k}"] = p.loc[arms, "none"].mean() - e0
            rec[f"R_{k}"] = (en - e0) - (p.loc[arms, "named"] - p.loc[arms, "none"]).mean()
            rec[f"Rperm_{k}"] = (ep - e0) - (p.loc[arms, "permuted"] - p.loc[arms, "none"]).mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def rule(C, col, gate=True):
    out = []
    for label, g in C.groupby("label"):
        g = g[g.erasure_ok] if gate else g
        concepts, ps, means, ks, ns = [], [], [], [], []
        for concept, gg in g.groupby("concept"):
            k, n = int((gg[col] > 0).sum()), len(gg)
            concepts.append(concept); means.append(gg[col].mean()); ks.append(k); ns.append(n)
            ps.append(binomtest(k, n, 0.5).pvalue if n else 1.0)
        for concept, m, k, n, ph in zip(concepts, means, ks, ns, holm(ps)):
            out.append(dict(label=label, concept=concept, stat=col, mean=m, pos=k, n=n, p_holm=ph,
                            named=bool(ph < 0.05 and m >= FLOOR), sign=bool(ph < 0.05)))
    return pd.DataFrame(out)


def planted_of(label):
    return None if label == "none" else label.split(" | ", 1)[1]


def summary(C):
    E, R = rule(C, "effect_sham"), rule(C, "R_null")
    EN = rule(C, "effect_null")
    rows = []
    order = [f"{lvl} | {P}" for P in ("amino-acid composition", "secondary structure content", "global shape")
             for lvl in ("strong", "weak", "faint", "mixed")] + ["none"]
    for label in order:
        if label not in set(C.label):
            continue
        P = planted_of(label)
        r, e, en = R[R.label == label], E[E.label == label], EN[EN.label == label]
        rp = r[r.concept == P].iloc[0] if P else None
        ep = e[e.concept == P].iloc[0] if P else None
        enp = en[en.concept == P].iloc[0] if P else None
        others = r[r.concept != P] if P else r
        rows.append(dict(
            label=label,
            R_planted=rp["mean"] if P else np.nan, R_planted_pos=f"{int(rp.pos)}/{int(rp.n)}" if P else "",
            R_other_max=others["mean"].max(), R_other_min=others["mean"].min(),
            repair_named=", ".join(SHORT.get(c, c) for c in r.concept[r.named]) or "nothing",
            effect_sham_planted=ep["mean"] if P else np.nan,
            effect_null_planted=enp["mean"] if P else np.nan,
            effect_null_other_min=en[en.concept != P]["mean"].min() if P else en["mean"].min(),
            effect_named=", ".join(SHORT.get(c, c) for c in e.concept[e.named]) or "nothing",
            Rperm_absmax=float(C[C.label == label].Rperm_null.abs().groupby(C[C.label == label].concept).mean().max())))
    return pd.DataFrame(rows), E, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "erasure" / "calibration" / "part*.csv"))
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.src))], ignore_index=True)
    C = cells(d)
    S, E, R = summary(C)
    if not a.latex:
        C.to_csv(ROOT / "erasure" / "matched_null_calibration_cells.csv", index=False)
        S.to_csv(ROOT / "erasure" / "matched_null_calibration_summary.csv", index=False)
        pd.concat([E, R]).to_csv(ROOT / "erasure" / "matched_null_calibration_rules.csv", index=False)
        pd.set_option("display.width", 260)
        print(f"encoders {C.encoder.nunique()}, properties {C.concept.nunique()}, labels {C.label.nunique()}\n")
        print(S.round(4).to_string(index=False))
        return
    print("% generated by scripts/matched_null_calibration_report.py; do not edit by hand")
    print("\\begin{tabular}{lrrrlrrl}")
    print("\\toprule")
    print(" & \\multicolumn{3}{c}{repair contrast vs.\\ matched null} & & \\multicolumn{2}{c}{effect of the planted "
          "property} & \\\\")
    print("\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}")
    print("label & planted & positive & largest other & repair rule names & vs.\\ sham & vs.\\ null & "
          "effect rule names \\\\")
    print("\\midrule")
    for r in S.itertuples():
        lab = r.label.replace(" | ", " ").replace("amino-acid composition", "composition") \
                     .replace("secondary structure content", "sec.\\ structure")
        rp = "--" if np.isnan(r.R_planted) else f"${r.R_planted:+.3f}$"
        es = "--" if np.isnan(r.effect_sham_planted) else f"${r.effect_sham_planted:+.3f}$"
        en = "--" if np.isnan(r.effect_null_planted) else f"${r.effect_null_planted:+.3f}$"
        print(f"{lab} & {rp} & {r.R_planted_pos or '--'} & ${r.R_other_max:+.3f}$ & {r.repair_named} & {es} & {en} & "
              f"{r.effect_named} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
