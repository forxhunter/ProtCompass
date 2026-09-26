"""The external prediction test, with the nuisance leakage removed and the contrast given an interval.

Two problems with the first version.

First, the nuisance summaries include the quantities being assessed: the encoder's general readout strength is the
mean readout over all eight properties, which contains the property under test, and its general downstream standing
is the mean rank over all tasks, which contains the task under test. Both are recomputed here leaving the target
property and the target task out of their own adjustment.

Second, the verdict was a ranking: a task passed when the demanded property's partial correlation exceeded every
undemanded property's. Ranking first among weak associations is not evidence. Membrane and soluble passed at a
partial correlation of +0.04 with the largest competitor at +0.02. Here the contrast between the demanded property
and the strongest undemanded one carries a bootstrap interval over encoders, and a task is reported as corroborated
only when that interval excludes zero.

  python demand_prediction_test_v2.py [--markdown] [--boot 5000]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from demand_prediction_test import LABEL, erasure, resid                # noqa: E402


def partial_rho(x, y, gr, gs):
    """Spearman between x and y after the two nuisance ranks are regressed out of both, as ranks."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(gr) & np.isfinite(gs)
    if ok.sum() < 8:
        return np.nan
    z = np.column_stack([pd.Series(gr[ok]).rank(), pd.Series(gs[ok]).rank()])
    r, _ = spearmanr(resid(x[ok], z), resid(y[ok], z))
    return float(r)


def build():
    """Per task: a frame of encoder x property readouts, the downstream score, and both leave-one-out nuisances."""
    d = erasure()
    cond = d[(d.arm == "conditional") & d.erasure_ok.fillna(False)]
    scores = pd.read_csv(ROOT / "extract" / "scores2_selected.csv")
    scores = scores[scores.encoder != "ref_composition"].set_index(["task", "encoder"]).score
    allsc = scores.unstack("task")
    ranks = allsc.rank(ascending=False)                                  # encoder x task

    st = pd.read_csv(ROOT / "erasure" / "causal_demand_v2.csv")
    dem = set(map(tuple, st.loc[st.demanded, ["task", "concept"]].values))

    per_task = {}
    for task, g in cond.groupby("task"):
        R = g.pivot_table(index="encoder", columns="concept", values="concept_before", aggfunc="mean")
        y = scores.reindex([(task, e) for e in R.index]).to_numpy()
        keep = np.isfinite(y)
        R, y = R[keep], y[keep]
        if len(R) < 8:
            continue
        # general downstream standing, leaving this task out of the mean rank
        others = [t for t in ranks.columns if t != task]
        gs = ranks.loc[R.index, others].mean(axis=1).to_numpy()
        per_task[task] = dict(R=R, y=y, gs=gs, demanded=[c for c in R.columns if (task, c) in dem])
    return per_task


def rhos(R, y, gs, cols):
    """Partial correlation of every property, each adjusted by the mean readout over the OTHER properties."""
    out = {}
    for c in cols:
        rest = [k for k in R.columns if k != c]
        gr = R[rest].mean(axis=1).to_numpy() if rest else np.full(len(R), np.nan)
        out[c] = partial_rho(R[c].to_numpy(), y, gr, gs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for task, D in build().items():
        R, y, gs, dem = D["R"], D["y"], D["gs"], D["demanded"]
        if not dem:
            continue
        base = rhos(R, y, gs, list(R.columns))
        others = [c for c in R.columns if c not in dem]
        for c in dem:
            best_other = max((base[o] for o in others if np.isfinite(base[o])), default=np.nan)
            who = max((o for o in others if np.isfinite(base[o])), key=lambda o: base[o], default=None)
            boot_r, boot_c = [], []
            n = len(R)
            for _ in range(a.boot):
                idx = rng.integers(0, n, n)
                Rb, yb, gsb = R.iloc[idx], y[idx], gs[idx]
                bb = rhos(Rb, yb, gsb, [c] + others)
                if not np.isfinite(bb[c]):
                    continue
                bo = max((bb[o] for o in others if np.isfinite(bb[o])), default=np.nan)
                boot_r.append(bb[c])
                if np.isfinite(bo):
                    boot_c.append(bb[c] - bo)
            q = lambda v, p: float(np.percentile(v, p)) if len(v) > 20 else np.nan
            rows.append(dict(task=task, label=LABEL.get(task, task), concept=c, n=n,
                             rho_partial=base[c], rho_lo=q(boot_r, 2.5), rho_hi=q(boot_r, 97.5),
                             best_other=best_other, best_other_name=who,
                             contrast=base[c] - best_other if np.isfinite(best_other) else np.nan,
                             contrast_lo=q(boot_c, 2.5), contrast_hi=q(boot_c, 97.5)))
    t = pd.DataFrame(rows)
    t["corroborated"] = (t.contrast_lo > 0) & (t.rho_lo > 0)
    t.to_csv(ROOT / "erasure" / "demand_prediction_test_v2.csv", index=False)

    if not a.markdown:
        pd.set_option("display.width", 240)
        print(t.round(3).to_string(index=False))
        ok = t.groupby("task").corroborated.all()
        print(f"\ncorroborated tasks ({int(ok.sum())} of {len(ok)}): {sorted(ok[ok].index)}")
        return

    print("% generated by scripts/demand_prediction_test_v2.py; do not edit by hand")
    print("\\begin{tabular}{llrrrl}")
    print("\\toprule")
    print("task & demanded property & partial $\\rho$ [95\\%] & strongest other & contrast [95\\%] & verdict \\\\")
    print("\\midrule")
    for r in t.sort_values(["task", "concept"]).itertuples():
        print(f"{r.label} ({r.n}) & {r.concept} & ${r.rho_partial:+.2f}$ $[{r.rho_lo:+.2f}, {r.rho_hi:+.2f}]$ & "
              f"${r.best_other:+.2f}$ & ${r.contrast:+.2f}$ $[{r.contrast_lo:+.2f}, {r.contrast_hi:+.2f}]$ & "
              f"{'corroborated' if r.corroborated else 'not corroborated'} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"% bootstrap over encoders, {a.boot} resamples; nuisance summaries exclude the target property and the "
          f"target task")


if __name__ == "__main__":
    main()
