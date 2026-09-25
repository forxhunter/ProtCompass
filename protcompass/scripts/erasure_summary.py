"""Turn the erasure runs into a causal task-demand matrix, with the checks that decide whether a cell counts.

effect = score after erasing a random subspace of the same rank, minus score after erasing the property.
A positive effect means the downstream head needed that property beyond needing that many directions.

A cell is counted only if the erasure worked: the property must be substantially less recoverable from the erased
representation than from the original. Cells where it did not are reported separately rather than averaged in.

Significance uses the random-subspace draws as the null for each cell, and a sign test across encoders for the
task-level statement, with Holm correction over the properties within a task.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "erasure"


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    d = pd.read_csv(OUT / "erasure_results.csv")
    d["erasure_ok"] = d.concept_after < 0.5 * d.concept_before
    print(f"{len(d)} cells; erasure verified on {int(d.erasure_ok.sum())} "
          f"({d.erasure_ok.mean():.0%}); properties still recoverable in the rest")
    bad = d[~d.erasure_ok].groupby("concept").size()
    if len(bad):
        print("cells where the erasure did not take, by property:"); print(bad.to_string())

    ok = d[d.erasure_ok]
    pd.set_option("display.width", 220)

    print("\n=== causal task demand: mean effect over encoders (positive = the task needed the property) ===")
    piv = ok.pivot_table(index="concept", columns="task", values="effect", aggfunc="mean")
    n = ok.pivot_table(index="concept", columns="task", values="effect", aggfunc="size")
    print(piv.round(3).to_string())
    print("\nencoders per cell:"); print(n.fillna(0).astype(int).to_string())

    print("\n=== sign test across encoders, Holm-corrected within a task ===")
    rows = []
    for task, g in ok.groupby("task"):
        ps, names = [], []
        for concept, gg in g.groupby("concept"):
            k = int((gg.effect > 0).sum())
            ps.append(binomtest(k, len(gg), 0.5).pvalue if len(gg) else 1.0)
            names.append((concept, k, len(gg), gg.effect.mean()))
        adj = holm(ps)
        for (concept, k, tot, mean), a in zip(names, adj):
            rows.append(dict(task=task, concept=concept, encoders_positive=k, n=tot,
                             mean_effect=mean, p_holm=a, significant=a < 0.05))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "task_demand_causal.csv", index=False)
    print(t.sort_values(["task", "p_holm"]).round(4).to_string(index=False))

    sig = t[t.significant & (t.mean_effect > 0)]
    print(f"\nproperties a task demands (positive effect, Holm p < 0.05): {len(sig)} of {len(t)} task-property pairs")
    for task, g in sig.groupby("task"):
        print(f"  {task:14s} {', '.join(f'{r.concept} ({r.mean_effect:+.3f})' for r in g.itertuples())}")

    print("\n=== does the demand differ between tasks? ===")
    w = t.pivot(index="concept", columns="task", values="mean_effect")
    if w.shape[1] > 1:
        print("correlation of demand vectors between tasks:")
        print(w.corr(method="spearman").round(3).to_string())

    print("\n=== is the random control flat? ===")
    print(f"mean |base - random| over all cells: {np.abs(d.base - d.after_random).mean():.4f} "
          f"(if this is large, removing directions hurts by itself and the effect is not attributable)")
    print(f"mean rank erased: {d['rank'].mean():.1f}; mean embedding dim: {d.dim.mean():.0f}")


if __name__ == "__main__":
    main()
