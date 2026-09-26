"""The causal demand matrix read against the sham null, with the later corrections applied.

Version 1 reported `rotation - erased`, where the rotation arm is a random rotation truncated to d - r axes. That arm
is not the same map as the erasure, removes an order of magnitude less of the representation's variance, and scores
above the unerased representation on some encoders, so the difference was inflated by an amount comparable to the
effects being claimed. Here the null is a LEACE erasure of the same property permuted across proteins: the same map,
the same rank, a target with the same marginal distribution and no relation to the protein.

Three gates, all of which version 1 was missing on at least one arm:

  erasure took      the property must be at most half as recoverable after erasure, checked on every arm including
                    the conditional one.
  above the sham    the effect is measured against the sham, and a cell must additionally exceed the largest sham
                    effect seen on that task before it counts, since the sham itself passes a bare sign test.
  magnitude floor   a significant sign test on a mean effect of +0.001 is not a demand. Cells below MIN_EFFECT are
                    reported as null regardless of their p-value.

  python causal_demand_v2.py
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
ERASURE_OK = 0.5
MIN_EFFECT = 0.02          # below this an effect is not reported as a demand whatever its p-value
MIN_CV = 0.02              # a property that barely varies on a task cannot be shown to be needed by it
ARM = "conditional"


def variation():
    """Coefficient of variation of each property within each task.

    Two of the ten tasks are mutational scans of a single fold: fluorescence is 10,500 GFP variants over 3 distinct
    lengths and stability 10,500 mini-proteins over 3. Several properties are then almost constant across the dataset,
    and `size` on fluorescence has a coefficient of variation of 1e-4 while the first version of this experiment
    reported it as significantly demanded at +0.022. A property with no variation cannot be shown to be needed, so the
    gate belongs with the statistics rather than in a caveat.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import common_struct
    rows = []
    for f in sorted((OUT / "concepts").glob("*.npz")):
        task = f.stem
        common_struct.set_task(task)
        try:
            tab = common_struct.build(task)
        except Exception:
            continue
        Z = np.stack(list(tab.values()))
        for name, (sl, _) in common_struct.BLOCKS.items():
            b = Z[:, sl]
            m = abs(np.nanmean(b))
            rows.append(dict(task=task, concept=name, sd=float(np.nanstd(b, 0).mean()),
                             cv=float(np.nanstd(b, 0).mean() / max(m, 1e-9))))
    return pd.DataFrame(rows).set_index(["task", "concept"]).cv


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def load():
    fs = sorted((OUT / "parts").glob("erasure_v2*.csv"))
    frames = []
    for f in fs:
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        raise SystemExit("no erasure_v2 parts found; run scripts/run_v2.sh")
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates(subset=["task", "encoder", "concept", "arm"], keep="last")


def main():
    d = load()
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print(f"{len(d)} rows, {d.task.nunique()} tasks, {d.encoder.nunique()} encoders, arms: "
          f"{', '.join(sorted(d.arm.dropna().unique()))}\n")

    print("=== how much the old control inflated the effect ===")
    both = d.dropna(subset=["effect", "effect_v1"])
    if len(both):
        print(f"mean effect against the sham null: {both.effect.mean():+.4f}")
        print(f"mean effect against the old rotation control: {both.effect_v1.mean():+.4f}")
        print(f"mean inflation: {(both.effect_v1 - both.effect).mean():+.4f} "
              f"over {len(both)} cells; the rotation arm scores above the unerased representation in "
              f"{(both.rotation > both.base).mean():.0%} of them")
        flip = ((both.effect_v1 > 0) & (both.effect < 0)).mean()
        print(f"cells positive under the old control and negative under the sham: {flip:.0%}\n")

    cond = d[(d.arm == ARM) & d.erasure_ok.fillna(False)]
    if cond.empty:
        raise SystemExit("no conditional cells with a verified erasure yet")

    # the sham ceiling per task: the largest sham-vs-erased effect any single property reaches by chance is not
    # available directly, so the conservative bar is the spread of the sham across its own permutation seeds
    bar = d.groupby("task").sham_sd.mean().rename("sham_sd")

    cv = variation()
    rows = []
    for task, g in cond.groupby("task"):
        ps, meta = [], []
        for concept, gg in g.groupby("concept"):
            k, n = int((gg.effect > 0).sum()), len(gg)
            ps.append(binomtest(k, n, 0.5).pvalue if n else 1.0)
            meta.append((concept, k, n, gg.effect.mean(), gg.effect.std(), gg["rank"].mean()))
        for (concept, k, n, mean, sd, rk), a in zip(meta, holm(ps)):
            v = float(cv.get((task, concept), np.nan))
            varies = bool(v >= MIN_CV) if np.isfinite(v) else True
            rows.append(dict(task=task, concept=concept, mean_effect=mean, sd=sd, positive=k, n=n, rank=rk, cv=v,
                             p_holm=a, passes_sign=bool(a < 0.05), passes_magnitude=bool(mean >= MIN_EFFECT),
                             varies=varies,
                             demanded=bool(a < 0.05 and mean >= MIN_EFFECT and varies)))
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "causal_demand_v2.csv", index=False)

    print(f"=== conditional demand against the sham null, arm '{ARM}' ===")
    piv = cond.pivot_table(index="concept", columns="task", values="effect", aggfunc="mean")
    print(piv.round(3).to_string())
    print(f"\nmean sham spread over permutation seeds, per task:")
    print(bar.round(4).to_string())

    n_dem = int(st.demanded.sum())
    print(f"\n{n_dem} of {len(st)} task-property pairs are demanded: sign test Holm p < 0.05, mean effect "
          f">= {MIN_EFFECT}, and the property varies on the task (cv >= {MIN_CV})")
    if n_dem:
        for task, g in st[st.demanded].sort_values(["task", "mean_effect"], ascending=[True, False]).groupby("task"):
            print(f"  {task:26s} " + ", ".join(f"{r.concept} ({r.mean_effect:+.3f}, rank {r.rank:.0f})"
                                               for r in g.itertuples()))
    for why, mask in (("fall below the magnitude floor", st.passes_sign & ~st.passes_magnitude),
                      ("are on a property that barely varies on that task", st.passes_sign & ~st.varies)):
        dropped = st[mask]
        if len(dropped):
            print(f"\n{len(dropped)} pairs pass the sign test but {why}, and are not reported:")
            for r in dropped.sort_values("mean_effect", ascending=False).itertuples():
                print(f"  {r.task:26s} {r.concept:28s} {r.mean_effect:+.4f}  p={r.p_holm:.2g}  cv={r.cv:.4f}")

    print("\n=== does the ordering survive rank matching? ===")
    for arm in sorted(a for a in d.arm.dropna().unique() if a.startswith("matched")):
        m = d[(d.arm == arm) & d.erasure_ok.fillna(False)]
        if m.empty:
            continue
        o = m.groupby("concept").effect.agg(["mean", "std", "size"]).sort_values("mean", ascending=False)
        print(f"\n{arm}, every property erased at the same rank:")
        print(o.round(4).to_string())

    ab = d[(d.arm == "conditional") & d["rank"].eq(0)]
    if len(ab):
        print(f"\n=== abstentions: the conditional residual was rounding error ===")
        print(ab.groupby("concept").size().to_string())
        print("These properties are exact linear functions of another and carry no independent conditional signal.")


if __name__ == "__main__":
    main()
