"""The sequence-versus-coordinate comparison with multiplicity control, a metric-free aggregate, and a split by where
the coordinates come from.

scripts/group_controls.py tests each of the ten tasks separately (family-clustered Mann-Whitney). The ten tasks use
four metrics (accuracy, F-max, AUROC and Spearman), so a mean of raw gaps across tasks mixes scales, and ten separate
tests at p < 0.05 need a correction. This script adds, for every restriction of group_controls.py:

  holm        Holm-adjusted family-clustered p over the ten tasks
  rank gap    per task, every encoder of the panel ranked by score (1 = best); the gap is the mean rank of the
              coordinate side minus that of the sequence side (positive: the sequence side ranks higher), averaged
              over tasks, with a family-level permutation test (group labels permuted over architecture families,
              10,000 permutations, the statistic averaged over tasks)
  normalised  per task, (score - composition reference) / (best encoder - composition reference), then the mean
              group gap over tasks
  provenance  the mean raw gap over the four tasks scored on experimental structures (EC, GO-MF, GO-BP, remote
              homology) and over the six scored on predicted structures (Appendix: where the coordinates come from)

  python group_controls_multiplicity.py [--latex]
Output: extract/group_controls_multiplicity.csv (per restriction), extract/group_controls_holm.csv (per task)
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from group_controls import LABEL, ORDER, REF, meta, restrictions, summarize   # noqa: E402

def holm(p):
    """Holm-adjusted p values, in the input order."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate((n - np.arange(n)) * p[o]).clip(max=1.0)
    out = np.empty(n); out[o] = adj
    return out


EXPERIMENTAL = ["ec_classification", "go_mf", "go_bp", "remote_homology"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    d = pd.read_csv(ROOT / "extract" / "scores2_selected.csv")
    w = d.pivot(index="encoder", columns="task", values="score")
    ref = w.loc[REF] if REF in w.index else None
    w = w.drop(index=[REF], errors="ignore")
    m = meta().reindex(w.index)
    tasks = [t for t in ORDER if t in w.columns]
    per_task = summarize()
    rank = w[tasks].rank(ascending=False)                    # 1 = best, within the whole panel
    norm = (w[tasks] - ref[tasks]) / (w[tasks].max() - ref[tasks])
    rng = np.random.default_rng(0)
    rows, holm_rows = [], []
    for name, mask in restrictions(m).items():
        sel = m[mask]
        g = per_task[per_task.restriction == name].copy()
        g["fam_p_holm"] = holm(g.fam_p.values)
        holm_rows.append(g)
        seq = sel.index[sel.group == "sequence"]
        crd = sel.index[sel.group == "coordinate-side"]

        def fam_rank_gap(labels):
            fam_rank = rank.loc[sel.index].groupby(sel.family).mean()
            s = fam_rank.loc[labels == "sequence"].mean()
            c = fam_rank.loc[labels == "coordinate-side"].mean()
            return float((c - s).mean())

        fam_group = sel.groupby("family").group.first()
        obs = fam_rank_gap(fam_group)
        perm = np.array([fam_rank_gap(pd.Series(rng.permutation(fam_group.values), index=fam_group.index))
                         for _ in range(10_000)])
        p_rank = float((np.abs(perm) >= abs(obs)).mean())
        exp_t = [t for t in tasks if t in EXPERIMENTAL]
        pred_t = [t for t in tasks if t not in EXPERIMENTAL]
        raw_gap = w.loc[seq, tasks].mean() - w.loc[crd, tasks].mean()
        rows.append(dict(restriction=name, n_seq=len(seq), n_crd=len(crd),
                         fam_seq=int((fam_group == "sequence").sum()), fam_crd=int((fam_group != "sequence").sum()),
                         tasks_seq_leads=int(g.seq_leads.sum()), tasks=len(g),
                         tasks_p05=int((g.fam_p < 0.05).sum()), tasks_holm05=int((g.fam_p_holm < 0.05).sum()),
                         mean_raw_gap=float(raw_gap.mean()),
                         rank_gap=float((rank.loc[crd].mean() - rank.loc[seq].mean()).mean()),
                         fam_rank_gap=obs, fam_rank_gap_p=p_rank,
                         norm_gap=float((norm.loc[seq].mean() - norm.loc[crd].mean()).mean()),
                         gap_experimental=float(raw_gap[exp_t].mean()), gap_predicted=float(raw_gap[pred_t].mean()),
                         seq_leads_experimental=int((raw_gap[exp_t] > 0).sum()), n_experimental=len(exp_t),
                         seq_leads_predicted=int((raw_gap[pred_t] > 0).sum()), n_predicted=len(pred_t)))
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "extract" / "group_controls_multiplicity.csv", index=False)
    pd.concat(holm_rows).to_csv(ROOT / "extract" / "group_controls_holm.csv", index=False)
    if a.latex:
        print("% generated by scripts/group_controls_multiplicity.py; do not edit by hand")
        print("\\begin{tabular}{lcccccc}")
        print("\\toprule")
        print("restriction & tasks, $p<0.05$ & tasks, Holm & mean rank gap & permutation $p$ & "
              "normalised gap & experimental / predicted \\\\")
        print("\\midrule")
        for r in R.itertuples():
            pstr = "$<0.001$" if r.fam_rank_gap_p < 0.001 else f"{r.fam_rank_gap_p:.3f}"
            print(f"{r.restriction} & {r.tasks_p05} of {r.tasks} & {r.tasks_holm05} of {r.tasks} & "
                  f"{r.fam_rank_gap:+.1f} & {pstr} & {r.norm_gap:+.3f} & "
                  f"{r.gap_experimental:+.3f} / {r.gap_predicted:+.3f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        return
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    print(pd.concat(holm_rows)[["restriction", "task", "gap", "fam_p", "fam_p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
