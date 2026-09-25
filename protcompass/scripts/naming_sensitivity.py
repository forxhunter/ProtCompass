"""The repair naming rule under task-calibrated magnitude floors, with only well-matched null cells, and with
architecture families as the unit.

The released rule (scripts/named_pair_verdicts.py) names a task-property pair when the repair contrast R_var passes a
two-sided sign test over encoders at Holm-corrected p < 0.05 over the 60 pairs, its mean is at least 0.02 on the
task's score scale, and the property varies on the task. The ten tasks use four metrics, so a fixed 0.02 means
different things on different tasks, and the variance-matched null meets its 1% target on only part of the cells.
This script recomputes the rule with one thing changed at a time:

  released    as released (0.02)
  mde         the floor of each task is its minimum detectable effect (alpha 0.05, power 0.8) from the sham spread,
              the largest over the properties erased on that task (erasure/exclusion_decisions.csv, worst_mde)
  reach5      the floor of each task is 5% of its reachable range, the loss from erasing the task label's own
              subspace at the property rank (erasure/exclusion_decisions.csv, reach); a task whose reachable range is
              not positive has no floor and names nothing
  matched     the released rule on the cells whose null removes the erased share of variance to within 1%
  family      one vote per architecture family, with leave-one-family-out (erasure/family_weighted_repair_full.csv)

  python naming_sensitivity.py [--latex]
Output: erasure/naming_sensitivity.csv (one row per pair and rule)
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from named_pair_verdicts import DEM                                      # noqa: E402

E = ROOT / "erasure"


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate((n - np.arange(n)) * p[o]).clip(max=1.0)
    out = np.empty(n); out[o] = adj
    return out


def rule(C, floor_of, varies, keep=None):
    rows = []
    for (task, concept), g in C.groupby(["task", "concept"]):
        if keep is not None:
            g = g[keep.loc[g.index]]
        n, pos = len(g), int((g.R_null > 0).sum())
        rows.append(dict(task=task, concept=concept, n=n, pos=pos, R=g.R_null.mean() if n else np.nan,
                         p=binomtest(pos, n, 0.5).pvalue if n else 1.0, floor=floor_of(task)))
    V = pd.DataFrame(rows)
    V["p_holm"] = holm(V.p.values)
    V["varies"] = [varies.get((r.task, r.concept), True) for r in V.itertuples()]
    V["passes"] = (V.p_holm < 0.05) & (V.R >= V.floor) & V.varies
    return V


def main():
    C = pd.read_csv(E / "named_pair_cells_full.csv")
    dem = pd.read_csv(DEM).set_index(["task", "concept"])
    varies = {k: bool(v) for k, v in dem["varies"].items()}
    X = pd.read_csv(E / "exclusion_decisions.csv").set_index("task")
    out = []
    specs = [("released", lambda t: 0.02, None),
             ("mde", lambda t: float(X.loc[t, "worst_mde"]), None),
             ("reach5", lambda t: 0.05 * float(X.loc[t, "reach"]) if X.loc[t, "reach"] > 0 else np.inf, None),
             ("matched", lambda t: 0.02, (C.var_null - C.var_erased).abs() / C.var_erased <= 0.01)]
    for name, f, keep in specs:
        out.append(rule(C, f, varies, keep).assign(rule=name))
    F = pd.read_csv(E / "family_weighted_repair_full.csv")
    out.append(F.rename(columns={"n_fam": "n", "pos_fam": "pos", "mean_fam": "R"})
               .assign(rule="family", floor=0.02)[["task", "concept", "n", "pos", "R", "p", "p_holm", "floor",
                                                    "varies", "passes", "rule", "loo_passes", "loo_total"]])
    V = pd.concat(out, ignore_index=True)
    V.to_csv(E / "naming_sensitivity.csv", index=False)
    matched = ((C.var_null - C.var_erased).abs() / C.var_erased <= 0.01)
    print(f"cells matched to within 1%: {int(matched.sum())} of {len(C)} ({matched.mean():.1%})")
    pd.set_option("display.width", 220)
    for name, g in V.groupby("rule", sort=False):
        p = g[g.passes]
        print(f"\n{name}: {len(p)} pairs named")
        print(p[["task", "concept", "n", "pos", "R", "floor", "p_holm"]].round(4).to_string(index=False))
    named = V[V.passes].groupby(["task", "concept"]).rule.apply(lambda s: ", ".join(s))
    print("\n", named.to_string())


if __name__ == "__main__" and "--latex" not in sys.argv:
    main()


def latex():
    """Every pair named under at least one rule, with the rules that name it."""
    V = pd.read_csv(E / "naming_sensitivity.csv")
    rules = ["released", "mde", "reach5", "matched", "family"]
    head = {"released": "released ($0.02$)", "mde": "task MDE", "reach5": "5\\% of reach", "matched": "matched cells",
            "family": "families"}
    lab = {"fluorescence": "fluorescence", "stability": "stability", "ppi_affinity": "PPI affinity", "go_mf": "GO-MF",
           "solubility": "solubility", "ec_classification": "EC"}
    short = {"amino-acid composition": "composition", "burial and packing": "burial and packing",
             "secondary structure content": "secondary structure", "contact order": "contact order",
             "global shape": "global shape", "size": "size"}
    named = V[V.passes][["task", "concept"]].drop_duplicates()
    print("% generated by scripts/naming_sensitivity.py --latex; do not edit by hand")
    print("\\begin{tabular}{ll" + "c" * len(rules) + "}")
    print("\\toprule")
    print("task & property & " + " & ".join(head[r] for r in rules) + " \\\\")
    print("\\midrule")
    order = sorted(named.itertuples(index=False), key=lambda r: (r.task != "stability", r.task, r.concept))
    for r in order:
        cells = []
        for rule in rules:
            g = V[(V.rule == rule) & (V.task == r.task) & (V.concept == r.concept)]
            if not len(g):
                cells.append("--"); continue
            g = g.iloc[0]
            mark = "\\textbf{yes}" if g.passes else "no"
            cells.append(f"{mark} (${g.R:+.3f}$)")
        print(f"{lab.get(r.task, r.task)} & {short.get(r.concept, r.concept)} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__" and "--latex" in sys.argv:
    latex()
