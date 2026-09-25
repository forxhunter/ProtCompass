"""Per-pair summary and LaTeX table of the fixed-head erasure analysis (scripts/ab_fixed_head.py).

Merges the shard outputs into erasure/ab_fixed_head/cells.csv, applies the pre-specified rule
(erasure/ab_fixed_head/PRESPEC.md) to Delta_fixed_var, and writes summary.csv and fixed_head.tex. The released
retrained-head numbers for the same pair (eff_null = Delta_var, R_null = R_var, from
erasure/named_pair_verdicts_full.csv) are placed side by side.

  python ab_fixed_head_summary.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from named_pair_verdicts import LABEL, SHORT, fmt_p                     # noqa: E402

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
D = ROOT / "erasure" / "ab_fixed_head"
VERDICTS = ROOT / "erasure" / "named_pair_verdicts_full.csv"
REL_CELLS = ROOT / "erasure" / "named_pair_cells_full.csv"
REL_SRC = [ROOT / "erasure" / "named_pairs_full", ROOT / "erasure" / "named_pairs_full_remote"]
FLOOR, ALPHA, M_ALL = 0.02, 0.05, 60


def holm(p, m=None):
    """Holm step-down adjusted p values. With m larger than len(p) the missing tests are entered as p = 1."""
    p = np.asarray(p, float)
    m = len(p) if m is None else m
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


def released_intact():
    """The released intact-arm score of every cell, to check that the frozen head reproduces the released fit."""
    frames = [pd.read_csv(f) for src in REL_SRC for f in sorted(glob.glob(str(src / "*.csv")))]
    d = pd.concat([f for f in frames if len(f)], ignore_index=True)
    d = d.drop_duplicates(subset=["task", "concept", "encoder", "arm", "descriptor"], keep="last")
    d = d[(d.arm == "intact") & (d.descriptor == "none")]
    return d[["task", "concept", "encoder", "score"]].rename(columns={"score": "S_intact_released"})


def main():
    parts = [pd.read_csv(f) for f in sorted(glob.glob(str(D / "parts" / "cells_shard*.csv")))]
    C = pd.concat([p for p in parts if len(p)], ignore_index=True)
    C = C.drop_duplicates(subset=["task", "concept", "encoder"], keep="last")
    C = C.merge(released_intact(), on=["task", "concept", "encoder"], how="left")
    rc = pd.read_csv(REL_CELLS)[["task", "concept", "encoder", "effect_null", "R_null"]]
    C = C.merge(rc.rename(columns={"effect_null": "Delta_var_released", "R_null": "R_var_released"}),
                on=["task", "concept", "encoder"], how="left")
    V = pd.read_csv(VERDICTS)
    V = V[V.named_by_effect_rule | V.passes_R_rule].reset_index(drop=True)
    order = {k: i for i, k in enumerate(zip(V.task, V.concept))}
    C["_o"] = [order[(t, c)] for t, c in zip(C.task, C.concept)]
    C = C.sort_values(["_o", "encoder"]).drop(columns="_o").reset_index(drop=True)
    lead = ["task", "concept", "encoder", "rank", "dim", "S_intact", "S_erased", "S_sham", "S_null",
            "var_erased", "var_null", "var_sham", "prop_null", "Delta_fixed_var", "Delta_fixed_sham",
            "S_intact_released", "Delta_var_released", "R_var_released"]
    C = C[lead + [c for c in C.columns if c not in lead]]
    C.to_csv(D / "cells.csv", index=False)

    rows = []
    for r in V.itertuples():
        g = C[(C.task == r.task) & (C.concept == r.concept)]
        g = g[np.isfinite(g.Delta_fixed_var)]
        n = len(g)
        k = int((g.Delta_fixed_var > 0).sum())
        ks = int((g.Delta_fixed_sham > 0).sum())
        ns = int(np.isfinite(g.Delta_fixed_sham).sum())
        rows.append(dict(
            task=r.task, concept=r.concept, n=n,
            S_intact=g.S_intact.mean(), S_erased=g.S_erased.mean(), S_null=g.S_null.mean(), S_sham=g.S_sham.mean(),
            var_erased=g.var_erased.mean(), var_null=g.var_null.mean(),
            Delta_fixed_var=g.Delta_fixed_var.mean(), pos_fixed_var=k,
            p_fixed_var=binomtest(k, n, 0.5).pvalue if n else np.nan,
            Delta_fixed_sham=g.Delta_fixed_sham.mean(), pos_fixed_sham=ks,
            p_fixed_sham=binomtest(ks, ns, 0.5).pvalue if ns else np.nan,
            max_abs_intact_vs_released=float((g.S_intact - g.S_intact_released).abs().max()),
            n_released=r.n, Delta_var_released=r.eff_null, pos_Delta_var_released=r.eff_pos_null,
            R_var_released=r.R_null, pos_R_var_released=r.pos_null, p_R_var_holm_released=r.p_null_holm,
            named_by_effect_rule=bool(r.named_by_effect_rule), passes_R_rule=bool(r.passes_R_rule),
            varies=bool(r.varies)))
    S = pd.DataFrame(rows)
    S["p_fixed_var_holm"] = holm(S.p_fixed_var)
    S["p_fixed_var_holm60"] = holm(S.p_fixed_var, M_ALL)
    S["p_fixed_sham_holm"] = holm(S.p_fixed_sham)
    S["named_fixed_head"] = (S.p_fixed_var_holm < ALPHA) & (S.Delta_fixed_var >= FLOOR) & S.varies
    S["named_fixed_head_holm60"] = (S.p_fixed_var_holm60 < ALPHA) & (S.Delta_fixed_var >= FLOOR) & S.varies
    S["sign_only_fixed_head"] = (S.p_fixed_var_holm < ALPHA) & ~S.named_fixed_head
    S.to_csv(D / "summary.csv", index=False)

    tex = ["% generated by scripts/ab_fixed_head_summary.py; do not edit by hand",
           "\\begin{tabular}{llrrrlrrl}",
           "\\toprule",
           " & & \\multicolumn{4}{c}{fixed head} & \\multicolumn{2}{c}{retrained head (released)} & \\\\",
           "\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}",
           "task & property & $n$ & $\\Delta^{\\mathrm{fix}}_{\\mathrm{var}}$ & positive & Holm $p$ & "
           "$\\Delta_{\\mathrm{var}}$ & $R_{\\mathrm{var}}$ & rules \\\\",
           "\\midrule"]
    for r in S.itertuples():
        rules = "+".join([x for x, f in (("E", r.named_by_effect_rule), ("R", r.passes_R_rule),
                                         ("F", r.named_fixed_head)) if f])
        tex.append(f"{LABEL.get(r.task, r.task)} & {SHORT.get(r.concept, r.concept)} & {r.n} & "
                   f"${r.Delta_fixed_var:+.4f}$ & {r.pos_fixed_var} & ${fmt_p(r.p_fixed_var_holm)}$ & "
                   f"${r.Delta_var_released:+.4f}$ & ${r.R_var_released:+.4f}$ & {rules} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            "% fixed head: head trained once on the intact representation and scored on test inputs projected in the",
            "% original coordinates; Delta^fix_var = mean over four variance-matched nulls of S(null) - S(erased).",
            f"% Holm over the {len(S)} pairs tested. rules: E = named by the released effect rule, R = named by the",
            "% released repair rule, F = named by the fixed-head rule (Holm p < 0.05, mean >= 0.02, property varies).",
            "% Retrained-head columns are eff_null and R_null of erasure/named_pair_verdicts_full.csv."]
    (D / "fixed_head.tex").write_text("\n".join(tex) + "\n")

    pd.set_option("display.width", 250)
    cols = ["task", "concept", "n", "Delta_fixed_var", "pos_fixed_var", "p_fixed_var_holm", "p_fixed_var_holm60",
            "Delta_fixed_sham", "pos_fixed_sham", "Delta_var_released", "R_var_released",
            "named_by_effect_rule", "passes_R_rule", "named_fixed_head", "max_abs_intact_vs_released"]
    print(S[cols].round(4).to_string(index=False))
    print(f"\nwritten: {D / 'cells.csv'} ({len(C)} cells), {D / 'summary.csv'}, {D / 'fixed_head.tex'}")


if __name__ == "__main__":
    main()
