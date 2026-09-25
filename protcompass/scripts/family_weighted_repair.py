"""The repair rule with one vote per architecture family, and with each family left out in turn.

The naming rule of Section 4.3 runs its sign test over encoders, and the 33 encoders are not independent draws: the
ESM-2 scales, the PottsMPNN variants, and random and pretrained runs of one network share most of what a sign test
counts. This script applies the same rule to the repair contrast against the variance-matched null with the encoders
first averaged inside an architecture family (the families of scripts/family_weighted_demand.py), and repeats it with
each family removed.

  rule      two-sided sign test over families, Holm over the candidate pairs, family-weighted mean at least 0.02,
            and variation of the property on the task
  interval  a percentile bootstrap over families of the family-weighted mean, 2,000 draws
  leave-one-family-out   the verdict with each family removed; a pair is stable if it passes every time

  python family_weighted_repair.py [--latex]
  python family_weighted_repair.py --tag full [--latex]      # the full 60-pair grid of named_pair_verdicts.py --tag full
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from family_weighted_demand import family, holm                         # noqa: E402
from panel import only_panel                                            # noqa: E402

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
FLOOR, ALPHA, BOOT = 0.02, 0.05, 2000
LABEL = {"membrane_soluble": "membrane vs soluble", "subcellular_localization": "subcellular localisation",
         "fluorescence": "fluorescence", "solubility": "solubility", "ppi_affinity": "PPI affinity",
         "go_mf": "GO-MF", "go_bp": "GO-BP", "stability": "stability", "ec_classification": "EC classification",
         "remote_homology": "remote homology"}
SHORT = {"amino-acid composition": "composition", "burial and packing": "burial and packing",
         "secondary structure content": "secondary structure", "global shape": "global shape",
         "contact order": "contact order", "size": "size"}


def verdicts(fam, varies):
    """fam: rows (task, concept, family, R) -> one verdict per pair, Holm over pairs."""
    rows = []
    for (t, c), g in fam.groupby(["task", "concept"]):
        v = g.R.to_numpy()
        v = v[np.isfinite(v) & (v != 0)]
        k, n = int((v > 0).sum()), len(v)
        rows.append(dict(task=t, concept=c, n_fam=n, pos_fam=k, mean_fam=float(v.mean()) if n else np.nan,
                         p=float(binomtest(k, n, 0.5).pvalue) if n else 1.0))
    V = pd.DataFrame(rows)
    V["p_holm"] = holm(V.p.to_numpy())
    V["varies"] = [varies.get((r.task, r.concept), True) for r in V.itertuples()]
    V["passes"] = (V.p_holm < ALPHA) & (V.mean_fam >= FLOOR) & V.varies
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the cell and verdict files to read, and of the output")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    C = only_panel(pd.read_csv(ROOT / "erasure" / f"named_pair_cells{sfx}.csv"))
    C["family"] = C.encoder.map(family)
    dem = pd.read_csv(ROOT / "erasure" / "causal_demand_v2.csv").set_index(["task", "concept"])
    varies = {k: bool(v) for k, v in dem.varies.items()}
    enc_rule = pd.read_csv(ROOT / "erasure" / f"named_pair_verdicts{sfx}.csv").set_index(["task", "concept"])

    fam = C.groupby(["task", "concept", "family"]).R_null.mean().rename("R").reset_index()
    V = verdicts(fam, varies)

    rng = np.random.default_rng(0)
    lo, hi = [], []
    for r in V.itertuples():
        v = fam[(fam.task == r.task) & (fam.concept == r.concept)].R.to_numpy()
        b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(BOOT)])
        lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
    V["lo"], V["hi"] = lo, hi

    families = sorted(fam.family.unique())
    stable, worst = {}, {}
    for f in families:
        Vf = verdicts(fam[fam.family != f], varies).set_index(["task", "concept"])
        for key, r in Vf.iterrows():
            stable.setdefault(key, []).append(bool(r.passes))
            worst[key] = max(worst.get(key, 0.0), float(r.p_holm))
    V["loo_passes"] = [sum(stable[(r.task, r.concept)]) for r in V.itertuples()]
    V["loo_total"] = len(families)
    V["loo_worst_p"] = [worst[(r.task, r.concept)] for r in V.itertuples()]
    V["encoder_rule"] = [bool(enc_rule.loc[(r.task, r.concept), "passes_R_rule"]) for r in V.itertuples()]
    V = V.sort_values(["passes", "p_holm"], ascending=[False, True]).reset_index(drop=True)
    V.to_csv(ROOT / "erasure" / f"family_weighted_repair{sfx}.csv", index=False)

    if a.latex:
        print("% generated by scripts/family_weighted_repair.py; do not edit by hand")
        print("\\begin{tabular}{llrrrlc}")
        print("\\toprule")
        print("task & property & families & mean $R_{\\mathrm{var}}$ [95\\% CI] & Holm $p$ & passes & "
              "leave-one-out \\\\")
        print("\\midrule")
        for r in V.itertuples():
            p = f"{r.p_holm:.3f}" if r.p_holm >= 1e-3 else f"{r.p_holm:.1e}".replace("e-0", "\\times 10^{-") + "}"
            print(f"{LABEL.get(r.task, r.task)} & {SHORT.get(r.concept, r.concept)} & {r.pos_fam}/{r.n_fam} & "
                  f"${r.mean_fam:+.4f}$ [${r.lo:+.4f}$, ${r.hi:+.4f}$] & ${p}$ & "
                  f"{'yes' if r.passes else 'no'} & {r.loo_passes}/{r.loo_total} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        return
    pd.set_option("display.width", 240)
    cols = ["task", "concept", "pos_fam", "n_fam", "mean_fam", "lo", "hi", "p_holm", "passes", "loo_passes",
            "loo_total", "loo_worst_p", "encoder_rule"]
    print(V[cols].round(4).to_string(index=False))
    print(f"\nfamilies: {len(families)}; pairs passing the family rule: {int(V.passes.sum())}; "
          f"the encoder rule: {int(V.encoder_rule.sum())}")


if __name__ == "__main__":
    main()
