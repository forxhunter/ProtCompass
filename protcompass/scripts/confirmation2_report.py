"""Decide cohort B's frozen hypotheses (confirmation2/freeze.json) and write every cell behind each decision.

  H1-H4   cohort A's rules, unchanged, on cohort B's scores (confirmation_report.py logic)
  H1b     each coordinate-reading panel encoder against the floor admissible for its measured inputs, the descriptor
          and readout family chosen on the released validation split (scripts/floor_rules.py, rule 'both'), the
          chosen floor's score read on cohort B
  H2b     the same choice for the sequence floor
  H5      H1-H4 on the topology-novel part (confirmation_novelty.py on cohort B), if it holds at least 20 domains

  python confirmation2_report.py [--latex]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
C = ROOT / "confirmation2"
os.environ["CONFIRMATION_DIR"] = str(C)
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

import confirmation_report as cr                                         # noqa: E402
import floor_rules as fr                                                 # noqa: E402

PANEL_CRD = ["esm3", "esm_if1", "proteinmpnn"]


def released_rules(S, audit):
    enc = [s for s in S.index if not s.startswith("ref_")]
    reads = {e: bool(audit.loc[e, "uses_coordinates"]) for e in enc}
    rows = []
    for p in [p for p in cr.INFORMATIVE_DETERMINED + cr.UNDETERMINED + ["architecture_label"] if p in S.columns]:
        for e in enc:
            pool = cr.FLOORS_SEQ + (cr.FLOORS_COORD if reads[e] else [])
            fl = S.loc[[f for f in pool if f in S.index], p].max()
            rows.append(dict(probe=p, encoder=e, coordinates=reads[e], score=S.loc[e, p], floor=fl, below=S.loc[e, p] < fl))
    cells = pd.DataFrame(rows)
    det = cells[cells.probe.isin(cr.INFORMATIVE_DETERMINED)]
    per = det[det.coordinates].groupby("encoder").below.sum()
    seq = det[~det.coordinates]
    und = cells[cells.probe.isin(cr.UNDETERMINED)].assign(gap=lambda x: x.score - x.floor).groupby("probe").gap.max()
    h4 = {e: float(S.loc[e, "architecture_label"] - S.loc["ref_pairhist", "architecture_label"])
          for e in ("esm_if1", "esm3", "esm2_650m") if e in S.index}
    out = {"H1": (bool((per >= 7).all()), "per encoder below of 9: " + ", ".join(f"{e} {int(v)}" for e, v in per.items())),
           "H2": (bool(seq.below.sum() == 0), f"{int(seq.below.sum())} of {len(seq)} sequence-only cells below"),
           "H3": (bool((und > 0).all() and len(und) == 3), "best gap: " + ", ".join(f"{p} {v:+.3f}" for p, v in und.items())),
           "H4": (bool(h4.get("esm_if1", -1) > 0 and h4.get("esm3", -1) > 0 and h4.get("esm2_650m", 1) <= 0),
                  "vs ref_pairhist: " + ", ".join(f"{e} {v:+.3f}" for e, v in h4.items()))}
    return out, cells


def current_rule(audit):
    """H1b and H2b: admissible floors chosen on the released validation split, read on cohort B."""
    val = fr.scores([ROOT / "probing_results_val", ROOT / "probing_results_atom_val"])
    coh = fr.scores([C / "probing_results", C / "probing_results_atom"])
    enc = coh[~coh.source.str.startswith("ref_")].groupby(["source", "probe"]).score.max().unstack()
    M = fr.admissibility()
    rows = []
    for e in enc.index:
        for p in cr.INFORMATIVE_DETERMINED:
            elig = M.loc[e, "floors"]
            v = val[(val.probe == p) & val.source.isin(elig)]
            v = v[v.set_index(["source", "family"]).index.isin(
                coh[(coh.probe == p)].set_index(["source", "family"]).index)]
            ch = v.loc[v.score.idxmax()]
            fl = float(coh[(coh.probe == p) & (coh.source == ch.source) & (coh.family == ch.family)].score.iloc[0])
            rows.append(dict(probe=p, encoder=e, coordinates=bool(M.loc[e, "coordinates"]), score=enc.loc[e, p],
                             floor=fl, floor_source=ch.source, floor_family=ch.family, below=enc.loc[e, p] < fl))
    cells = pd.DataFrame(rows)
    per = cells[cells.coordinates].groupby("encoder").below.sum()
    seq = cells[~cells.coordinates]
    return {"H1b": (bool((per >= 7).all()), "per encoder below of 9: " + ", ".join(f"{e} {int(v)}" for e, v in per.items())),
            "H2b": (bool(seq.below.sum() == 0), f"{int(seq.below.sum())} of {len(seq)} sequence-only cells below")}, cells


def novelty():
    f = C / "novelty" / "novelty_verdicts.csv"
    if not f.exists():
        return {"H5": (None, "not computed")}
    R = pd.read_csv(f).set_index("subset")
    r = R.loc["topology_novel_split"]
    if r.n < 20:
        return {"H5": (None, f"not testable: {int(r.n)} topology-novel domains")}
    ok = bool(r.H1 and r.H2 and r.H3 and r.get("H4", True))
    return {"H5": (ok, f"{int(r.n)} domains; H1 {r.H1} ({r.per_encoder}), H2 {r.H2} ({r.seq_below} of {r.seq_cells}), "
                       f"H3 {r.H3} ({r.und_best_gap})" + (f", H4 {r.H4}" if "H4" in r else ""))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    freeze = json.loads((C / "freeze.json").read_text())
    audit = pd.read_csv(ROOT / "extract" / "encoder_input_audit.csv").query("dataset == 'probing'").set_index("encoder")
    S = cr.load_scores()
    V, cells = released_rules(S, audit)
    Vb, cells_b = current_rule(audit)
    V.update(Vb); V.update(novelty())
    cells.assign(rule="released").to_csv(C / "cells_released.csv", index=False)
    cells_b.assign(rule="admissible_validation").to_csv(C / "cells_current.csv", index=False)
    (C / "verdicts.json").write_text(json.dumps({k: {"confirmed": v[0], "detail": v[1]} for k, v in V.items()}, indent=2) + "\n")
    if a.latex:
        TEX = str.maketrans({"_": "\\_", "%": "\\%", "&": "\\&"})
        print("% generated by scripts/confirmation2_report.py; do not edit by hand")
        print("\\begin{tabular}{lp{0.40\\linewidth}p{0.34\\linewidth}l}")
        print("\\toprule")
        print("& prediction, frozen before cohort B was read & result on cohort B & verdict \\\\")
        print("\\midrule")
        for h in freeze["hypotheses"]:
            ok, det = V[h["id"]]
            verdict = "not testable" if ok is None else ("confirmed" if ok else "not confirmed")
            print(f"{h['id']} & {h['claim'].translate(TEX)} & {det.translate(TEX)} & {verdict} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        return
    for k, (ok, det) in V.items():
        print(f"{k:4s} {'confirmed' if ok else ('not testable' if ok is None else 'NOT confirmed'):14s} {det}")


if __name__ == "__main__":
    main()
