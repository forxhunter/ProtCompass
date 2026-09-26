"""Exclusions as confidence bounds rather than as minimum detectable effects, with the rank stated.

Two problems with the way Section 4.3 stated its two null results.

First, a minimum detectable effect is a property of the design, not of the data. "The observed effect is below the
MDE" says the design would probably have missed an effect of that size; it does not bound the effect. A bound needs
an upper confidence limit on the mean effect, and an exclusion needs that limit to sit below a stated meaningful
effect. We use the magnitude floor the naming rule already carries, delta = 0.02, which was fixed on the sham spread
before this analysis, and report the limit both raw and Holm-adjusted over the six properties within a task.

Second, the reachable range was the largest effect the label's own subspace reached anywhere on the rank ladder, and
the ladder runs to rank 256 on two tasks and to rank 1 on five. That made the bars incomparable and put EC
classification into Figure 4 with a rank-256 reference bar while the text called it out of resolution. The reachable
range here is evaluated at the largest rank any property block occupies on that task, which is the rank the effects
being compared against it were measured at.

Three outcomes:

  detected      the largest effect clears the detection bound and the naming rule
  bounded       the reachable range is positive at the property rank and every Holm-adjusted upper limit is below
                delta, so a demand of delta or more is excluded for the properties erased
  inconclusive  the reachable range is not positive at the property rank, so the instrument has no resolution there,
                or an upper limit sits above delta

  python exclusion_decisions.py [--latex]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).parent))
from exclusion_bounds import TASK_LABEL, load, mde                      # noqa: E402

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "erasure"
DELTA = 0.02          # the magnitude floor the naming rule already uses
ALPHA = 0.05


def ucl(mean, sd, n, alpha=ALPHA):
    """One-sided upper confidence limit on the mean, from the per-encoder spread."""
    if not np.isfinite(sd) or n < 3:
        return np.nan
    return float(mean + student_t.ppf(1 - alpha, n - 1) * sd / np.sqrt(n))


def holm_alpha(k, m, alpha=ALPHA):
    """Holm step-down level for the k-th of m intervals, k = 1 the tightest."""
    return alpha / (m - k + 1)


def table():
    v2, ctrl = load("erasure_v2*.csv"), load("erasure_control*.csv")
    cond = v2[(v2.arm == "conditional") & v2.erasure_ok.fillna(False)]
    pc = ctrl[ctrl.check == "positive control"]
    ladder = pc.pivot_table(index="task", columns="rank", values="effect", aggfunc="mean")

    rows = []
    for (task, concept), g in cond.groupby(["task", "concept"]):
        rows.append(dict(task=task, concept=concept, n=len(g), mean_effect=g.effect.mean(), sd=g.effect.std(),
                         rank=int(g["rank"].max())))
    d = pd.DataFrame(rows)
    d["mde"] = [mde(r.sd, r.n) for r in d.itertuples()]
    d["ucl"] = [ucl(r.mean_effect, r.sd, r.n) for r in d.itertuples()]

    out = []
    for task, g in d.groupby("task"):
        cap = int(g["rank"].max())                         # the largest rank a property block occupies here
        cols = [c for c in ladder.columns if c <= cap] if task in ladder.index else []
        reach = float(ladder.loc[task, cols].max()) if cols else np.nan
        reach_rank = int(ladder.loc[task, cols].idxmax()) if cols else -1

        # Holm-adjusted upper limits over the properties of this task
        m = len(g)
        gg = g.sort_values("ucl")
        adj = []
        for k, r in enumerate(gg.itertuples(), start=1):
            adj.append(ucl(r.mean_effect, r.sd, r.n, alpha=holm_alpha(k, m)))
        gg = gg.assign(ucl_holm=adj)

        worst_mde, worst_ucl = g.mde.max(), gg.ucl_holm.max()
        biggest = g.mean_effect.max()
        detected = np.isfinite(worst_mde) and biggest > worst_mde
        if detected:
            decision = "detected"
        elif np.isfinite(reach) and reach > 0 and np.isfinite(worst_ucl) and worst_ucl < DELTA:
            decision = "bounded"
        else:
            decision = "inconclusive"
        out.append(dict(task=task, label=TASK_LABEL.get(task, task), n_encoders=int(g.n.max()),
                        property_rank=cap, reach=reach, reach_rank=reach_rank,
                        largest_effect=biggest, worst_mde=worst_mde,
                        worst_ucl=float(g.ucl.max()), worst_ucl_holm=float(worst_ucl),
                        ucl_frac=float(g.ucl.max() / reach) if np.isfinite(reach) and reach > 0 else np.nan,
                        ucl_holm_frac=float(worst_ucl / reach) if np.isfinite(reach) and reach > 0 else np.nan,
                        decision=decision))
    return d, pd.DataFrame(out).set_index("task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    d, o = table()
    d.to_csv(OUT / "exclusion_decisions_per_property.csv", index=False)
    o.to_csv(OUT / "exclusion_decisions.csv")

    if not a.latex:
        pd.set_option("display.width", 240)
        print(o.round(4).to_string())
        print(f"\ndelta = {DELTA}; upper limits are one-sided at {1 - ALPHA:.0%}, Holm-adjusted over the properties "
              f"of each task.")
        for task, r in o.iterrows():
            if r.decision == "bounded":
                print(f"- {r.label}: bounded. Every Holm-adjusted upper limit is at or below {r.worst_ucl_holm:+.3f}, "
                      f"which is {r.ucl_holm_frac:.0%} of the {r.reach:+.3f} that erasing the label's own subspace "
                      f"costs at rank {r.reach_rank}, so a demand of {DELTA} or more is excluded for the properties "
                      f"erased.")
            elif r.decision == "inconclusive":
                why = ("the label's own subspace costs nothing at the ranks the properties occupy, so the design has "
                       "no resolution there" if not (np.isfinite(r.reach) and r.reach > 0)
                       else f"the largest upper limit is {r.worst_ucl_holm:+.3f}, above delta")
                print(f"- {r.label}: inconclusive. {why}.")
        return

    print("% generated by scripts/exclusion_decisions.py; do not edit by hand")
    print("\\begin{tabular}{lrrrrrl}")
    print("\\toprule")
    print("task & encoders & property rank & erasing the label & largest effect & upper limit (Holm) & decision \\\\")
    print("\\midrule")
    for task, r in o.sort_values("decision").iterrows():
        reach = f"{r.reach:+.3f} at $r{{=}}{r.reach_rank}$" if np.isfinite(r.reach) else "--"
        frac = f" ({r.ucl_holm_frac * 100:.0f}\\%)" if np.isfinite(r.ucl_holm_frac) else ""
        lim = f"{r.worst_ucl_holm:+.3f}{frac}"
        print(f"{r.label} & {int(r.n_encoders)} & {int(r.property_rank)} & {reach} & "
              f"{r.largest_effect:+.3f} & {lim} & {r.decision} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"% delta = {DELTA}, one-sided {1 - ALPHA:.0%} limits, Holm-adjusted within a task")


if __name__ == "__main__":
    main()
