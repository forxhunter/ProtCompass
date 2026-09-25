"""Does "this task demands property P" predict anything beyond the erasure that defines it?

The objection this answers: demand is defined as the drop in downstream score when P's subspace is erased, so
"the task demands P" restates the measurement. A demand claim should also constrain something the measurement did
not touch. The prediction tested here is the simplest one available:

    across encoders, a task's downstream score should track how well the demanded property can be read out of that
    encoder's representation, and should not track the readout of properties the same test did not name.

Both quantities are already recorded. `concept_before` in the erasure run is the cross-validated readout of the
property from the encoder's representation on that task's own proteins, measured before anything is erased.
Downstream score is the released three-seed mean. Neither was used in deciding which properties are demanded, so
this is a test rather than a restatement.

The raw correlation is confounded and we report both forms. An encoder that reads one property well tends to read
all of them well and to score well on everything, so every property correlates with every task and the ranking
mostly recovers which property's readout tracks general encoder strength. The partialled form removes that: the
encoder's mean readout over all eight properties and its mean downstream rank over all tasks are regressed out of
both sides first, so what is left is whether an encoder that is unusually good at P, for an encoder of its overall
strength, is unusually good at the task that demands P.

  python demand_prediction_test.py [--markdown]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from causal_demand_v2 import MIN_CV, MIN_EFFECT  # noqa: E402

LABEL = {"ec_classification": "EC", "go_mf": "GO-MF", "go_bp": "GO-BP",
         "subcellular_localization": "localisation", "remote_homology": "homology", "solubility": "solubility",
         "stability": "stability", "fluorescence": "fluorescence", "ppi_affinity": "affinity",
         "membrane_soluble": "membrane"}


def erasure():
    files = sorted(set(glob.glob(str(ROOT / "**" / "parts" / "erasure_v2*.csv"), recursive=True)))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates(subset=["task", "encoder", "concept", "arm"], keep="last")


def resid(v, Z):
    """v with the linear part of the covariate block Z removed, v as ranks, so this is a partial rank correlation."""
    v = pd.Series(v).rank().to_numpy()
    Z = np.column_stack([np.ones(len(v)), np.asarray(Z, dtype=float)])
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


def partial(x, y, encoders, general_readout, general_score, task):
    """The correlation after the encoder's general readout strength and general downstream standing are removed."""
    gr = general_readout.reindex([(task, e) for e in encoders]).to_numpy()
    gs = general_score.reindex(encoders).to_numpy()
    ok = np.isfinite(gr) & np.isfinite(gs)
    if ok.sum() < 8:
        return np.nan, np.nan
    z = np.column_stack([pd.Series(gr[ok]).rank(), pd.Series(gs[ok]).rank()])
    r, p = spearmanr(resid(x[ok], z), resid(y[ok], z))
    return float(r), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()

    d = erasure()
    cond = d[(d.arm == "conditional") & d.erasure_ok.fillna(False)]
    scores = pd.read_csv(ROOT / "extract" / "scores2_selected.csv")
    scores = scores[scores.encoder != "ref_composition"].set_index(["task", "encoder"]).score

    # general strength of each encoder, the thing both sides share: its mean readout over all properties on this
    # task, and its mean rank over all downstream tasks
    general_readout = cond.groupby(["task", "encoder"]).concept_before.mean()
    allsc = scores.unstack("task")
    general_score = allsc.rank(ascending=False).mean(axis=1)

    st = pd.read_csv(ROOT / "erasure" / "causal_demand_v2.csv")
    dem = set(map(tuple, st.loc[st.demanded, ["task", "concept"]].values))
    rows = []
    for task, g in cond.groupby("task"):
        for concept, h in g.groupby("concept"):
            h = h[["encoder", "concept_before"]].dropna()
            y = scores.reindex([(task, e) for e in h.encoder]).to_numpy()
            ok = np.isfinite(y) & np.isfinite(h.concept_before.to_numpy())
            if ok.sum() < 8:
                continue
            x, yy, encs = h.concept_before.to_numpy()[ok], y[ok], h.encoder.to_numpy()[ok]
            r, p = spearmanr(x, yy)
            rp, pp = partial(x, yy, encs, general_readout, general_score, task)
            rows.append(dict(task=task, concept=concept, n=int(ok.sum()), rho=float(r), p=float(p),
                             rho_partial=rp, p_partial=pp, demanded=(task, concept) in dem))
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "erasure" / "demand_prediction_test.csv", index=False)

    lines = []
    passed = 0
    for task, g in t.groupby("task"):
        if not g.demanded.any():
            continue
        dm, un = g[g.demanded], g[~g.demanded]
        ok = dm.rho.min() > un.rho.max()
        okp = dm.rho_partial.min() > un.rho_partial.max()
        passed += bool(okp)
        fmt = lambda v: f"{v.min():+.2f} to {v.max():+.2f}" if len(v) > 1 else f"{v.iloc[0]:+.2f}"
        lines.append(dict(task=LABEL.get(task, task), demanded=", ".join(sorted(dm.concept)),
                          rho_demanded=fmt(dm.rho), rho_others=fmt(un.rho), raw="yes" if ok else "no",
                          part_demanded=fmt(dm.rho_partial), part_others=fmt(un.rho_partial),
                          separates="yes" if okp else "no", n=int(g.n.max())))
    out = pd.DataFrame(lines)

    if a.markdown:
        print("<!-- generated by scripts/demand_prediction_test.py; do not edit by hand -->\n")
        print("| task | demanded property | partial correlation, demanded | partial, the other properties | "
              "demanded above all others |")
        print("|---|---|---|---|---|")
        for r in out.itertuples():
            print(f"| {r.task} ({r.n} encoders) | {r.demanded} | {r.part_demanded} | {r.part_others} | "
                  f"{r.separates} |")
        print(f"\nSpearman over encoders between the downstream score and the cross-validated readout of the "
              f"property from that encoder's representation before any erasure, after the encoder's mean readout "
              f"over all properties and its mean downstream rank are regressed out of both sides. The demanded "
              f"property outranks every undemanded one on {passed} of {len(out)} tasks that name one.")
        return
    print(out.to_string(index=False))
    print(f"\ndemanded property outranks every undemanded one on {passed} of {len(out)} tasks")
    print(f"thresholds in force: effect >= {MIN_EFFECT}, property cv >= {MIN_CV}")


if __name__ == "__main__":
    main()
