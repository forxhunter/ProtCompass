"""Selection-corrected floor comparison: choose the floor on one half of the test set, report on the other.

Two successive reviewers made the same objection to the headline table: the "best floor" is a maximum over eight
descriptors, each designed after seeing the previous one lose, and the maximum is read on the same test set the
comparison is reported on. The number is therefore a function of descriptor effort and of selection on test.

run_probing_suite_split.py rescored every source on a fixed, protein-level halving of the test set (`design` and
`score`), changing nothing about training. This script does the selection-corrected comparison:

  1. For each source and probe, the readout family (linear vs nonlinear) is chosen on the DESIGN half.
  2. The best floor for each probe is chosen on the DESIGN half.
  3. That floor's score, the best encoder, the gap and the count of encoders below the floor are all read on the
     SCORE half, which no selection has touched.

It also reports how often the design-half argmax floor differs from the full-test argmax floor: that count is the
size of the selection effect on the identity of the floor. Sources not yet rescored are simply absent, and the
coverage is printed, so the table can be read before the rerun is complete.

  python design_score_split.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from floor_vs_encoders import FLOORS, ORDER

RES = Path((PROTCOMPASS_ROOT + "/protcompass/probing_results"))
UND = {"rsa", "rsa_bins", "bfactor", "architecture_label"}
TAUT = {"contact_density", "contact_order_len_resid"}


def headline(d):
    return d["pearson"].fillna(d["circ_cos"] if "circ_cos" in d else np.nan).fillna(d["balanced_accuracy"])


def wide(d, eval_set, family_choice=None):
    """source x probe headline on one evaluation set; readout family fixed by `family_choice` if given."""
    x = d[d.eval_set == eval_set].copy()
    x["headline"] = headline(x)
    x["family"] = np.where(x.readout == "linear", "linear", "nonlinear")
    m = x.groupby(["source", "probe", "family"]).headline.mean().reset_index()
    if family_choice is None:
        best = m.loc[m.groupby(["source", "probe"]).headline.idxmax()]
        fam = best.set_index(["source", "probe"]).family
    else:
        fam = family_choice
        m = m.set_index(["source", "probe", "family"]).headline
        best = m.reindex([(s, p, fam.get((s, p))) for s, p in fam.index]).reset_index()
        best.columns = ["source", "probe", "family", "headline"]
    return best.pivot(index="source", columns="probe", values="headline"), fam


def main():
    files = sorted((RES / "by_source_split").glob("*.csv"))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sources = sorted(d.source.unique())
    floors_present = [f for f in FLOORS if f in sources]
    encs = [s for s in sources if not s.startswith("ref_")]
    print(f"{len(sources)} sources rescored: {len(floors_present)} of {len(FLOORS)} floors, {len(encs)} encoders")
    missing_floors = [f for f in FLOORS if f not in sources]
    if missing_floors:
        raise SystemExit(f"floors not yet rescored: {missing_floors}; the comparison needs all of them")

    W_design, fam = wide(d, "design")                       # family chosen on the design half
    W_score, _ = wide(d, "score", family_choice=fam)        # ... and applied unchanged to the score half
    W_full, _ = wide(d, "test")                             # the published rule: everything on the full test set
    pub = pd.read_csv(RES / "floor_vs_encoders.csv").set_index("probe")

    rows, flips = [], 0
    for p in [c for c in ORDER if c in W_score.columns]:
        fd = W_design.loc[floors_present, p].dropna()
        if fd.empty:
            continue
        chosen = fd.idxmax()                                # floor selected on the design half
        chosen_full = W_full.loc[floors_present, p].dropna().idxmax()
        flips += int(chosen != chosen_full)
        floor_score = W_score.loc[chosen, p]
        e = W_score.loc[[x for x in encs if x in W_score.index], p].dropna()
        rows.append(dict(probe=p, determined="no" if p in UND else "tautological" if p in TAUT else "yes",
                         floor_chosen_on_design=chosen, floor_chosen_on_full_test=chosen_full,
                         floor_score_on_score_half=floor_score,
                         best_encoder_on_score_half=e.idxmax() if len(e) else None,
                         best_encoder_score=e.max() if len(e) else np.nan,
                         gap_selection_corrected=(e.max() - floor_score) if len(e) else np.nan,
                         encoders_below=int((e < floor_score).sum()), n_encoders=len(e),
                         gap_published=pub.loc[p, "gap"] if p in pub.index else np.nan,
                         below_published=pub.loc[p, "encoders below best floor"] if p in pub.index else np.nan,
                         n_published=pub.loc[p, "n encoders"] if p in pub.index else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "design_score_split.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    print(f"\ndesign-half argmax floor differs from full-test argmax floor on {flips} of {len(t)} probes")
    # the same set the rest of the paper calls informative determined: determined, minus the two probes whose
    # floor contains the label by construction. Reporting a different denominator here is what produced the
    # "8 of 11" next to "six of nine" that a reviewer found.
    det = t[t.determined == "yes"]
    print(f"informative determined probes where no encoder beats the design-selected floor on the score half: "
          f"{int((det.gap_selection_corrected < 0).sum())} of {len(det)} "
          f"(published rule: {int((det.gap_published < 0).sum())} of {len(det)})")
    sign_flip = t[np.sign(t.gap_selection_corrected) != np.sign(t.gap_published)]
    print(f"probes whose gap changes sign under selection correction: "
          f"{', '.join(sign_flip.probe) if len(sign_flip) else 'none'}")
    print(f"\nwritten: {RES / 'design_score_split.csv'}")


if __name__ == "__main__":
    main()
