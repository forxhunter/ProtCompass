"""What size of causal demand can each task exclude? The null turned into a bound.

"We found no demand" is not a stable claim: anyone who adds a variant may find one. "On this task we can exclude a
demand larger than X" is stable, because it is a statement about the sensitivity of the instrument rather than about
the outcome of one run. Nothing a later reviewer adds can overturn a sensitivity measurement, and the bound only
tightens as coverage grows.

Three quantities, all already measured by the experiment rather than assumed:

  noise floor      The spread of the sham arm over its permutation seeds, and the spread of the effect across
                   encoders. This is how large an effect can appear when nothing is there.
  reachable range  The label's own subspace erased at the same rank, from the control ladder. This is the most any
                   rank-r property can cost the task, so it is the scale the bound is expressed against.
  detectable       The smallest true effect this design would have found, at the given number of encoders, from the
                   observed per-encoder spread. Reported both in raw metric units and as a fraction of the reachable
                   range.

A task whose reachable range is flat at the rank the properties occupy is out of resolution and is reported that
way. That is not an exclusion: a label-directed erasure is the most damaging one of its rank only for an optimal
linear readout under squared loss, so a flat reference at rank r does not rule out a rank-r direction the retrained
head needs (see scripts/rank_bound_check.py). Use scripts/exclusion_decisions.py for the confidence-bound version.

  python exclusion_bounds.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "erasure"
ALPHA = 0.05
POWER = 0.80
TASK_LABEL = {"stability": "stability", "fluorescence": "fluorescence", "solubility": "solubility",
              "membrane_soluble": "membrane", "remote_homology": "homology", "ec_classification": "EC",
              "subcellular_localization": "localisation", "go_mf": "GO-MF", "go_bp": "GO-BP",
              "ppi_affinity": "affinity"}


def load(pattern):
    frames = []
    for f in sorted((OUT / "parts").glob(pattern)):
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    # overlapping part files must not inflate n or distort the spread the bound is computed from
    key = [c for c in ("task", "encoder", "concept", "arm", "check", "requested_rank") if c in d]
    return d.drop_duplicates(subset=key, keep="last")


def mde(sd, n, alpha=ALPHA, power=POWER):
    """Minimum detectable effect for a one-sample t-test on n encoders with observed per-encoder sd.

    This is the smallest true mean effect the design would reject the null for, with probability `power`. An effect
    smaller than this is not absent from the protein; it is below what this many encoders can resolve, and saying so
    is the honest form of a null."""
    if not np.isfinite(sd) or n < 3:
        return np.nan
    df = n - 1
    return float((student_t.ppf(1 - alpha, df) + student_t.ppf(power, df)) * sd / np.sqrt(n))


def reachable_at_property_rank(ctrl, cond):
    """The label's own subspace erased at the largest rank a property block occupies on that task.

    An earlier version took the maximum over every rank on the ladder. The ladder runs to rank 256 on EC
    classification and remote homology and to rank 1 on five other tasks, so those bars were not comparable and EC
    appeared in Figure 4 with a rank-256 reference while the text called it out of resolution. The reference must be
    read at the rank the effects compared against it were measured at.
    """
    import numpy as _np
    import pandas as _pd
    if ctrl.empty or "check" not in ctrl:
        return _pd.Series(dtype=float), _pd.Series(dtype=float)
    pc = ctrl[ctrl.check == "positive control"]
    if not len(pc):
        return _pd.Series(dtype=float), _pd.Series(dtype=float)
    ladder = pc.pivot_table(index="task", columns="rank", values="effect", aggfunc="mean")
    cap = cond.groupby("task")["rank"].max()
    reach, at = {}, {}
    for t in ladder.index:
        cols = [c for c in ladder.columns if t not in cap.index or c <= cap[t]]
        if not cols or not _np.isfinite(ladder.loc[t, cols]).any():
            continue
        reach[t] = float(ladder.loc[t, cols].max())
        at[t] = int(ladder.loc[t, cols].idxmax())
    return _pd.Series(reach), _pd.Series(at)


def main():
    v2 = load("erasure_v2*.csv")
    ctrl = load("erasure_control*.csv")
    if v2.empty:
        raise SystemExit("no erasure_v2 parts yet; run scripts/run_v2.sh")
    cond = v2[(v2.arm == "conditional") & v2.erasure_ok.fillna(False)]
    if cond.empty:
        raise SystemExit("no conditional cells with a verified erasure yet")

    # reachable range per task, read at the rank the property blocks occupy (see reachable_at_property_rank)
    reach, reach_rank = reachable_at_property_rank(ctrl, cond)

    rows = []
    for (task, concept), g in cond.groupby(["task", "concept"]):
        sd, n = g.effect.std(), len(g)
        rows.append(dict(task=task, concept=concept, n=n, mean_effect=g.effect.mean(), sd=sd,
                         sham_sd=g.sham_sd.mean(), mde=mde(sd, n)))
    d = pd.DataFrame(rows)

    pd.set_option("display.width", 240)
    print("=== per task: what can be excluded ===\n")
    print("`worst mde` is the largest minimum detectable effect over the eight properties, so an effect above it "
          "would have been found for any of them.\n")
    out = []
    for task, g in d.groupby("task"):
        r = reach.get(task, np.nan)
        worst = g.mde.max()
        biggest = g.mean_effect.max()
        out.append(dict(task=task, n_encoders=int(g.n.max()), reachable=r, worst_mde=worst,
                        worst_mde_frac=worst / r if np.isfinite(r) and r > 0 else np.nan,
                        largest_observed=biggest, sham_sd=g.sham_sd.mean()))
    o = pd.DataFrame(out).set_index("task")
    print(o.round(4).to_string())

    print("\n=== the bound, one sentence per task ===\n")
    for task, r in o.iterrows():
        name = TASK_LABEL.get(task, task)
        if not np.isfinite(r.reachable) or r.reachable <= 0:
            print(f"- **{name}**: out of resolution. The label's own subspace erased at the ranks the properties "
                  f"occupy costs nothing, so there is no positive scale to state a bound against; the result is "
                  f"inconclusive rather than an exclusion.")
            continue
        if not np.isfinite(r.worst_mde):
            print(f"- **{name}**: too few encoders ({int(r.n_encoders)}) to state a bound.")
            continue
        print(f"- **{name}**: with {int(r.n_encoders)} encoders this design would have detected a conditional demand "
              f"of {r.worst_mde:+.3f} or more for any of the eight properties, which is "
              f"{r.worst_mde / r.reachable:.0%} of the {r.reachable:+.3f} that erasing the label itself costs. "
              f"The largest effect observed is {r.largest_observed:+.3f}. "
              f"{'That exceeds the bound, so it is a detection.' if r.largest_observed > r.worst_mde else 'Nothing reaches the bound, so demands above it are excluded.'}")

    d.to_csv(OUT / "exclusion_bounds.csv", index=False)
    print(f"\nwritten: {OUT / 'exclusion_bounds.csv'}")

    print("\n=== how the bound tightens with coverage ===")
    print("The bound scales as 1/sqrt(n) in the number of encoders, so it is reported with the coverage it was "
          "computed at and can only improve:")
    sd_typ = d.sd.median()
    for n in (8, 16, 24, 33):
        print(f"  {n:2d} encoders, at the median per-encoder sd of {sd_typ:.3f}: bound {mde(sd_typ, n):+.4f}")


def markdown():
    """The same bounds as a markdown block for FINDINGS.md, so the document cannot drift from them."""
    v2 = load("erasure_v2*.csv")
    ctrl = load("erasure_control*.csv")
    cond = v2[(v2.arm == "conditional") & v2.erasure_ok.fillna(False)]
    reach, _ = reachable_at_property_rank(ctrl, cond)
    rows = []
    for (task, concept), g in cond.groupby(["task", "concept"]):
        rows.append(dict(task=task, concept=concept, n=len(g), mean_effect=g.effect.mean(), sd=g.effect.std()))
    d = pd.DataFrame(rows)
    d["mde"] = [mde(r.sd, r.n) for r in d.itertuples()]
    print("<!-- generated by scripts/exclusion_bounds.py; do not edit by hand -->\n")
    print("| task | encoders | erasing the label costs | smallest demand detectable | as a share of that | "
          "largest observed | reading |")
    print("|---|---|---|---|---|---|---|")
    for task, g in d.groupby("task"):
        r = reach.get(task, np.nan)
        worst, big, n = g.mde.max(), g.mean_effect.max(), int(g.n.max())
        if not np.isfinite(r) or r <= 0:
            print(f"| {TASK_LABEL.get(task, task)} | {n} | {r if np.isfinite(r) else 0:+.3f} | — | — | "
                  f"{big:+.3f} | out of resolution; the bound is 0 |")
        elif not np.isfinite(worst):
            print(f"| {TASK_LABEL.get(task, task)} | {n} | {r:+.3f} | — | — | {big:+.3f} | too few encoders |")
        else:
            verdict = "a demand above the bound is present" if big > worst else "demands above the bound are excluded"
            print(f"| {TASK_LABEL.get(task, task)} | {n} | {r:+.3f} | {worst:+.3f} | {worst / r:.0%} | "
                  f"{big:+.3f} | {verdict} |")
    print(f"\nThe bound is the minimum detectable conditional effect for a one-sample test over encoders at "
          f"alpha {ALPHA}, power {POWER}, using the observed per-encoder spread. It scales as 1/sqrt(n), so it is "
          f"reported with the coverage it was computed at and can only tighten.\n")


if __name__ == "__main__":
    import sys
    markdown() if "--markdown" in sys.argv else main()
