"""The causal task-demand matrix: which properties does each task need, measured by intervention.

Reads every erasure run (the parallel parts and the single-file runs), applies the checks that decide whether a cell
counts, and reports the demand per task with the statistics a reader needs to judge it.

  effect        score after erasing a random subspace of the same rank, minus score after erasing the property.
                Positive means the downstream head needed that property beyond needing that many directions.
  marginal      the property erased on its own.
  conditional   the property erased after the other properties have been projected out of it. This is the number to
                read: the eight properties are correlated, and the marginal and conditional effects have a Spearman
                correlation near zero, so a marginal effect does not attribute demand to a named property.

Cells are dropped when the erasure did not take, and a task is marked uninformative when its positive control fails.

  python causal_demand.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "erasure"
ERASURE_OK = 0.5          # the property must be at most half as recoverable after erasure as before
CONTROL_OK = 0.5          # the positive control must remove at least half of the base score
SENSITIVE_OK = 0.10       # the label ruler must move the score by a tenth of base somewhere on the rank ladder


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def load(pattern, extra=()):
    fs = sorted(OUT.glob(pattern)) + sorted((OUT / "parts").glob(pattern))
    fs += [OUT / f for f in extra if (OUT / f).exists()]
    if not fs:
        return pd.DataFrame()
    frames = []
    for f in fs:
        try:
            frames.append(pd.read_csv(f))      # a part still being written can be empty
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    key = [c for c in ("task", "encoder", "concept", "check", "requested_rank") if c in d]
    return d.drop_duplicates(subset=key, keep="last")


def sign_table(d, value="effect"):
    rows = []
    for task, g in d.groupby("task"):
        ps, meta = [], []
        for concept, gg in g.groupby("concept"):
            k, n = int((gg[value] > 0).sum()), len(gg)
            ps.append(binomtest(k, n, 0.5).pvalue if n else 1.0)
            meta.append((concept, k, n, gg[value].mean(), gg[value].std()))
        for (concept, k, n, mean, sd), a in zip(meta, holm(ps)):
            rows.append(dict(task=task, concept=concept, mean_effect=mean, sd=sd, positive=k, n=n,
                             p_holm=a, significant=bool(a < 0.05)))
    return pd.DataFrame(rows)


def main():
    marg = load("erasure_results*.csv", extra=["erasure_partial.csv"])
    cond_all = load("erasure_conditional*.csv", extra=["erasure_robustness.csv"])
    # the positive control has a dedicated pass; prefer it over control rows an older conditional run left behind
    ctrl = load("erasure_control*.csv")
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    if marg.empty:
        raise SystemExit("no erasure results found")
    marg["erasure_ok"] = marg.concept_after < ERASURE_OK * marg.concept_before
    print(f"marginal: {len(marg)} cells over {marg.task.nunique()} tasks and {marg.encoder.nunique()} encoders; "
          f"erasure verified on {marg.erasure_ok.mean():.0%}")
    print(f"random control flatness, mean |base - random|: {np.abs(marg.base - marg.after_random).mean():.4f}")

    # ---- positive control: is the instrument sensitive on this task?
    if not ctrl.empty:
        pc = ctrl[ctrl.check == "positive control"]
    else:
        pc = cond_all[cond_all.check == "positive control"] if "check" in cond_all else pd.DataFrame()
    if len(pc):
        print("\n=== positive control: erase the label subspace, at each rank the concepts are erased at ===")
        print("The effect at rank r is the most any rank-r erasure can cost this task, and the ruler for a concept "
              "effect at rank r. A task is informative where the ruler is above the noise.")
        lad = pc.pivot_table(index="task", columns="rank", values="effect", aggfunc="mean")
        if not lad.empty:
            print(lad.round(3).to_string())
        s = pc.groupby("task").agg(base=("base", "mean"), best=("effect", "max"), n=("effect", "size"))
        s["best_frac"] = s.best / s.base.replace(0, np.nan)
        s["sensitive"] = s.best_frac > SENSITIVE_OK
        print(s.round(3).to_string())
        blind = set(s.index[~s.sensitive.fillna(False)])
        if blind:
            print(f"tasks where a near-zero effect is uninformative: {', '.join(sorted(blind))}")
    else:
        lad, blind = pd.DataFrame(), set()

    ok = marg[marg.erasure_ok]
    print("\n=== marginal effect, mean over encoders ===")
    print(ok.pivot_table(index="concept", columns="task", values="effect", aggfunc="mean").round(3).to_string())

    cond = cond_all[cond_all.check == "conditional"] if "check" in cond_all else pd.DataFrame()
    if len(cond):
        print("\n=== conditional effect, the number to read ===")
        pivc = cond.pivot_table(index="concept", columns="task", values="effect", aggfunc="mean")
        print(pivc.round(3).to_string())
        st = sign_table(cond)
        st.to_csv(OUT / "causal_demand.csv", index=False)
        print("\n=== what each task demands: conditional effect, sign test over encoders, Holm within task ===")
        for task, g in st.sort_values(["task", "p_holm"]).groupby("task"):
            tag = "  [control failed, read with care]" if task in blind else ""
            print(f"\n{task}{tag}")
            for r in g.itertuples():
                mark = "*" if r.significant else " "
                print(f"  {mark} {r.concept:28s} {r.mean_effect:+.3f} +- {r.sd:.3f}   {r.positive:2d}/{r.n:2d}   p={r.p_holm:.2g}")

        both = ok.groupby(["task", "concept"]).effect.mean().rename("marginal").to_frame().join(
            cond.groupby(["task", "concept"]).effect.mean().rename("conditional")).dropna()
        if len(both) > 3:
            r, p = spearmanr(both.marginal, both.conditional)
            print(f"\nmarginal vs conditional across {len(both)} task-property pairs: Spearman {r:+.3f} (p={p:.3g})")
            print("A marginal effect therefore does not attribute demand to a named property.")

        print("\n=== do tasks demand different things? Spearman between conditional demand vectors ===")
        w = pivc.dropna(axis=1, how="any")
        if w.shape[1] > 1:
            c = w.corr(method="spearman")
            print(c.round(2).to_string())
            iu = np.triu_indices(len(c), 1)
            print(f"mean off-diagonal correlation: {c.values[iu].mean():.3f} over {len(iu[0])} task pairs")


if __name__ == "__main__":
    main()
