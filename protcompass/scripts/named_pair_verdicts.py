"""Re-test the naming claims on the contrast that is identified.

The naming rule of Section 4.3 is written on the level contrast: a sign test over encoders on the conditional effect,
a magnitude floor of 0.02 on its mean, and variation of the property on the task. Appendix
sec:appendix_matched_null shows that the level contrast is not identified -- matched on the amount removed rather
than only on the rank, it loses sign consistency -- while the repair contrast is. This script applies a sign test to
the repair contrast against each null, over the candidate pairs of scripts/addback_named_pairs.py, so the naming set
can be read off the identified quantity.

The rule is the released naming rule with its contrast replaced and nothing else changed: a two-sided sign test over
encoders with Holm correction (scripts/causal_demand_v2.py uses scipy's binomtest at its two-sided default), a mean of
at least 0.02, which fixes the direction, and variation of the property on the task. The 0.02 floor is on the score
scale and was fixed before any repair contrast existed; the repair contrast estimates the share of the score the
property accounts for, which is what the floor was written to bound, so it applies without re-choosing. The sign
test alone is also reported, as a statement of presence without size: with 32 to 34 encoders it detects repairs of
a few thousandths, and on its own it would name nearly every candidate.

  python named_pair_verdicts.py [--latex]
  python named_pair_verdicts.py --src erasure/named_pairs_full erasure/named_pairs_full_remote --tag full [--latex]

With --src the cells are read from the listed directories (the full 60-pair grid, run on two machines, one machine
per task), Holm runs over every pair found there, and the outputs carry the tag: erasure/named_pair_cells_<tag>.csv
and erasure/named_pair_verdicts_<tag>.csv.
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from panel import only_panel                                            # noqa: E402

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
SRC = ROOT / "erasure" / "named_pairs"
DEM = ROOT / "erasure" / "causal_demand_v2.csv"
FLOOR = 0.02            # the released magnitude floor, on the score scale
LABEL = {"membrane_soluble": "membrane vs soluble", "subcellular_localization": "subcellular localisation",
         "fluorescence": "fluorescence", "stability": "stability", "ec_classification": "EC classification",
         "go_mf": "GO-MF", "go_bp": "GO-BP", "remote_homology": "remote homology", "solubility": "solubility",
         "ppi_affinity": "PPI affinity"}
SHORT = {"amino-acid composition": "composition", "burial and packing": "burial and packing",
         "secondary structure content": "secondary structure", "global shape": "global shape",
         "contact order": "contact order", "size": "size"}


def load(srcs=(SRC,)):
    frames = []
    for f in sorted(f for src in srcs for f in glob.glob(str(Path(src) / "*.csv"))):
        d = pd.read_csv(f)
        if len(d):
            frames.append(d)
    if not frames:
        raise SystemExit("no candidate-pair results yet")
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates(subset=["task", "concept", "encoder", "arm", "descriptor"], keep="last")
    return only_panel(d)                   # the composition reference has an embedding directory and is not an encoder


def cells(d):
    rows = []
    for (task, concept, enc), g in d.groupby(["task", "concept", "encoder"]):
        p = g.pivot_table(index="arm", columns="descriptor", values="score")
        fam = {k: [a for a in p.index if a.startswith(k)] for k in ("sham", "null")}
        if not fam["sham"] or not fam["null"] or "erased" not in p.index:
            continue
        if "named" not in p.columns or p.loc["erased", "named"] != p.loc["erased", "named"]:
            continue
        rec = dict(task=task, concept=concept, encoder=enc,
                   var_erased=g.loc[g.arm == "erased", "var_removed"].mean(),
                   var_sham=g.loc[g.arm.isin(fam["sham"]), "var_removed"].mean(),
                   var_null=g.loc[g.arm.isin(fam["null"]), "var_removed"].mean())
        real = p.loc["erased", "named"] - p.loc["erased", "none"]
        for k, arms in fam.items():
            rec[f"effect_{k}"] = p.loc[arms, "none"].mean() - p.loc["erased", "none"]
            rec[f"R_{k}"] = real - (p.loc[arms, "named"] - p.loc[arms, "none"]).mean()
            if "permuted" in p.columns and not p.loc[arms, "permuted"].isna().all():
                rec[f"Rperm_{k}"] = ((p.loc["erased", "permuted"] - p.loc["erased", "none"])
                                    - (p.loc[arms, "permuted"] - p.loc[arms, "none"]).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def verdicts(C, alpha=0.05):
    dem = pd.read_csv(DEM).set_index(["task", "concept"])
    rows = []
    for (task, concept), g in C.groupby(["task", "concept"]):
        n = len(g)
        rec = dict(task=task, concept=concept, n=n,
                   var_erased=g.var_erased.mean(), var_sham=g.var_sham.mean(), var_null=g.var_null.mean())
        for k in ("sham", "null"):
            pos = int((g[f"R_{k}"] > 0).sum())
            rec[f"R_{k}"] = g[f"R_{k}"].mean()
            rec[f"pos_{k}"] = pos
            rec[f"p_{k}"] = binomtest(pos, n, 0.5).pvalue
            rec[f"eff_{k}"] = g[f"effect_{k}"].mean()
            rec[f"eff_pos_{k}"] = int((g[f"effect_{k}"] > 0).sum())
            if f"Rperm_{k}" in g:
                rec[f"Rperm_{k}"] = g[f"Rperm_{k}"].mean()
        key = (task, concept)
        rec["named_by_effect_rule"] = bool(dem.loc[key, "demanded"]) if key in dem.index else False
        rec["effect_rule_mean"] = float(dem.loc[key, "mean_effect"]) if key in dem.index else np.nan
        rows.append(rec)
    V = pd.DataFrame(rows)
    # Holm over the candidate set, on the repair contrast against the matched null
    order = V.p_null.sort_values().index
    m = len(order)
    adj, running = {}, 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * V.loc[idx, "p_null"]))
        adj[idx] = running
    V["p_null_holm"] = pd.Series(adj)
    V["varies"] = [bool(dem.loc[(r.task, r.concept), "varies"]) if (r.task, r.concept) in dem.index else True
                   for r in V.itertuples()]
    V["passes_R_sign"] = V.p_null_holm < alpha
    V["passes_R_rule"] = V.passes_R_sign & (V.R_null >= FLOOR) & V.varies
    return V.sort_values(["p_null_holm", "task"]).reset_index(drop=True)


def fmt_p(p):
    """A p value for a table: three decimals down to 0.001, a power of ten below that."""
    if not np.isfinite(p):
        return "--"
    if p >= 1e-3:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--src", nargs="+", default=[str(SRC)], help="result directories; default the 18 candidates")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    C = cells(load([ROOT / s if not Path(s).is_absolute() else Path(s) for s in a.src]))
    C.to_csv(ROOT / "erasure" / f"named_pair_cells{sfx}.csv", index=False)
    V = verdicts(C)
    V.to_csv(ROOT / "erasure" / f"named_pair_verdicts{sfx}.csv", index=False)

    if a.latex:
        print("% generated by scripts/named_pair_verdicts.py; do not edit by hand")
        print("\\begin{tabular}{llrrrrrll}")
        print("\\toprule")
        print(" & & & \\multicolumn{2}{c}{effect vs.\\ null} & \\multicolumn{3}{c}{repair contrast vs.\\ null} & \\\\")
        print("\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}")
        print("task & property & $n$ & mean & positive & mean & positive & Holm $p$ & named \\\\")
        print("\\midrule")
        for r in V.itertuples():
            star = "$^{\\ast}$" if r.named_by_effect_rule else ""
            verdict = "yes" if r.passes_R_rule else ("sign only" if r.passes_R_sign else "")
            print(f"{LABEL.get(r.task, r.task)}{star} & {SHORT.get(r.concept, r.concept)} & {r.n} & "
                  f"${r.eff_null:+.4f}$ & {r.eff_pos_null} & ${r.R_null:+.4f}$ & {r.pos_null} & "
                  f"${fmt_p(r.p_null_holm)}$ & {verdict} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("% $^{\\ast}$ named by the effect rule. named: yes = sign test, mean >= 0.02 and variation; "
              f"sign only = the sign test alone. Holm over the {len(V)} pairs.")
        return

    pd.set_option("display.width", 250)
    cols = ["task", "concept", "n", "eff_sham", "eff_pos_sham", "eff_null", "eff_pos_null",
            "R_null", "pos_null", "p_null_holm", "Rperm_null", "varies",
            "named_by_effect_rule", "passes_R_sign", "passes_R_rule"]
    print(V[[c for c in cols if c in V.columns]].round(4).to_string(index=False))
    both = V[(V.named_by_effect_rule) & (V.passes_R_rule)]
    only_e = V[(V.named_by_effect_rule) & (~V.passes_R_rule)]
    only_r = V[(~V.named_by_effect_rule) & (V.passes_R_rule)]
    print(f"\nnamed by both rules: {len(both)}")
    for r in both.itertuples():
        print(f"   {LABEL.get(r.task, r.task)} / {SHORT.get(r.concept, r.concept)}")
    print(f"named by the effect rule only: {len(only_e)}")
    for r in only_e.itertuples():
        print(f"   {LABEL.get(r.task, r.task)} / {SHORT.get(r.concept, r.concept)}  Holm p {r.p_null_holm:.3g}")
    sign_only = V[V.passes_R_sign & ~V.passes_R_rule]
    print(f"passes the sign test on R but not the floor or variation: {len(sign_only)}")
    for r in sign_only.itertuples():
        print(f"   {LABEL.get(r.task, r.task)} / {SHORT.get(r.concept, r.concept)}  R {r.R_null:+.4f}  "
              f"{r.pos_null}/{r.n}  varies={r.varies}")
    print(f"passes the repair rule only: {len(only_r)}")
    for r in only_r.itertuples():
        print(f"   {LABEL.get(r.task, r.task)} / {SHORT.get(r.concept, r.concept)}  R {r.R_null:+.4f}")
    print(f"\nwritten: erasure/named_pair_cells{sfx}.csv and erasure/named_pair_verdicts{sfx}.csv")


if __name__ == "__main__":
    main()
