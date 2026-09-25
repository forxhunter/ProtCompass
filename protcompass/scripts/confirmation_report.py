"""Decide the frozen hypotheses on the confirmation cohort, and report every cell behind the decision.

The four hypotheses, their directions and their decision rules were written by scripts/confirmation_freeze.py before
any score on this cohort existed. This script reads that file, reads the scores, and applies the rules as written.
It does not choose thresholds, drop encoders or pick probes; if a hypothesis fails it is reported failed.

Aggregation is the released one, from floor_vs_encoders.py: the mean over seeds inside a readout family, then the
better of the two families. Floor assignment is the released one: an encoder measured not to read coordinates is
held to the sequence floors, one measured to read them to the best of all eight.

  python confirmation_report.py [--markdown]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
CONF = Path(os.environ.get("CONFIRMATION_DIR") or ROOT / "confirmation")   # cohort B: confirmation2
RES = CONF / "probing_results" / "by_source"

FLOORS_SEQ = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global"]
FLOORS_COORD = ["ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]
INFORMATIVE_DETERMINED = ["phi_circular", "psi_circular", "local_curvature", "dssp_3", "dssp_8", "core_surface",
                          "distance_bins", "radius_of_gyration_len_resid", "asphericity_len_resid"]
UNDETERMINED = ["rsa", "rsa_bins", "bfactor"]
LABEL = {"phi_circular": "backbone phi", "psi_circular": "backbone psi", "local_curvature": "local curvature",
         "dssp_3": "DSSP-3", "dssp_8": "DSSP-8", "core_surface": "core/surface", "distance_bins": "CA distance bins",
         "radius_of_gyration_len_resid": "radius of gyration", "asphericity_len_resid": "asphericity",
         "rsa": "RSA", "rsa_bins": "RSA bins", "bfactor": "B-factor", "architecture_label": "fold architecture"}


def headline(d):
    """The released headline metric per row: Pearson, else circular cosine, else balanced accuracy."""
    v = d["pearson"] if "pearson" in d else pd.Series(np.nan, index=d.index)
    if "circ_cos" in d:
        v = v.fillna(d["circ_cos"])
    return v.fillna(d["balanced_accuracy"])


def load_scores():
    frames = []
    for f in sorted(RES.glob("*.csv")):
        t = pd.read_csv(f)
        if "probe" not in t:
            continue
        frames.append(t)
    if not frames:
        raise SystemExit(f"no scores under {RES}; run the suite on the merged dataset first")
    d = pd.concat(frames, ignore_index=True)
    d["headline"] = headline(d)
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    m = d.groupby(["source", "probe", "family"]).headline.mean().reset_index()
    return m.groupby(["source", "probe"]).headline.max().unstack("probe")


def audit_groups(sources):
    a = pd.read_csv(ROOT / "extract" / "encoder_input_audit.csv").drop_duplicates("encoder").set_index("encoder")
    reads = {}
    for s in sources:
        if s.startswith("ref_"):
            continue
        reads[s] = bool(a.loc[s, "uses_coordinates"]) if s in a.index else None
    return reads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    freeze = json.loads((CONF / "freeze.json").read_text())
    S = load_scores()
    encoders = [s for s in S.index if not s.startswith("ref_")]
    reads = audit_groups(encoders)
    seq_enc = [e for e in encoders if reads.get(e) is False]
    crd_enc = [e for e in encoders if reads.get(e) is True]

    def floor_for(enc, probe):
        pool = FLOORS_SEQ if reads.get(enc) is False else FLOORS_SEQ + FLOORS_COORD
        vals = [S.loc[f, probe] for f in pool if f in S.index and np.isfinite(S.loc[f, probe])]
        return (max(vals), pool[int(np.argmax(vals))] if len(vals) == len(pool) else "") if vals else (np.nan, "")

    rows = []
    for probe in [p for p in INFORMATIVE_DETERMINED + UNDETERMINED + ["architecture_label"] if p in S.columns]:
        for enc in encoders:
            fl, _ = floor_for(enc, probe)
            sc = S.loc[enc, probe]
            if not np.isfinite(sc) or not np.isfinite(fl):
                continue
            rows.append(dict(probe=probe, encoder=enc, group="sequence-only" if reads.get(enc) is False else "coordinate",
                             score=sc, floor=fl, gap=sc - fl, below=sc < fl))
    cells = pd.DataFrame(rows)
    cells.to_csv(CONF / "confirmation_cells.csv", index=False)

    det = cells[cells.probe.isin(INFORMATIVE_DETERMINED)]
    crd = det[det.group == "coordinate"]
    seq = det[det.group == "sequence-only"]
    per_enc_below = crd.groupby("encoder").below.sum()
    h1 = bool((per_enc_below >= 7).all()) and len(per_enc_below) > 0
    h2 = bool(seq.below.sum() == 0)

    und = cells[cells.probe.isin(UNDETERMINED)]
    best_gap = und.groupby("probe").gap.max()
    h3 = bool((best_gap > 0).all()) and len(best_gap) == len(UNDETERMINED)

    arch = cells[cells.probe == "architecture_label"].set_index("encoder")
    def vs_pairhist(enc):
        if enc not in arch.index or "ref_pairhist" not in S.index:
            return np.nan
        return float(S.loc[enc, "architecture_label"] - S.loc["ref_pairhist", "architecture_label"])
    h4_parts = {e: vs_pairhist(e) for e in ("esm_if1", "esm3", "esm2_650m")}
    h4 = bool(h4_parts.get("esm_if1", -1) > 0 and h4_parts.get("esm3", -1) > 0
              and h4_parts.get("esm2_650m", 1) <= 0)

    verdicts = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}
    detail = {
        "H1": f"coordinate encoders below their floor on the nine informative determined probes: "
              + ", ".join(f"{e} {int(v)}/9" for e, v in per_enc_below.items()),
        "H2": f"sequence-only cells below the sequence floor: {int(seq.below.sum())} of {len(seq)}",
        "H3": "best encoder minus best floor: " + ", ".join(f"{LABEL.get(p, p)} {v:+.3f}" for p, v in best_gap.items()),
        "H4": "against ref_pairhist on fold architecture: "
              + ", ".join(f"{e} {v:+.3f}" for e, v in h4_parts.items() if np.isfinite(v)),
    }
    out = {"cohort": freeze["cohort"], "panel": freeze["panel"], "verdicts": verdicts, "detail": detail,
           "n_cells": int(len(cells))}
    (CONF / "confirmation_verdicts.json").write_text(json.dumps(out, indent=2) + "\n")

    if a.latex:
        print("% generated by scripts/confirmation_report.py; do not edit by hand")
        print("\\begin{tabular}{lp{0.40\\linewidth}p{0.34\\linewidth}l}")
        print("\\toprule")
        print("& prediction, fixed before the cohort was scored & result on the cohort & verdict \\\\")
        print("\\midrule")
        TEX = str.maketrans({"_": "\\_", "%": "\\%", "&": "\\&"})
        for h in freeze["hypotheses"]:
            print(f"{h['id']} & {h['claim'].translate(TEX)} & {detail[h['id']].translate(TEX)} & "
                  f"{'confirmed' if verdicts[h['id']] else 'not confirmed'} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print(f"% cohort {freeze['cohort']['n']} domains, panel of {len(freeze['panel'])} encoders, "
              f"{len(cells)} scored cells")
        return

    if a.markdown:
        print("<!-- generated by scripts/confirmation_report.py; do not edit by hand -->\n")
        print("| hypothesis | decision rule, fixed before the cohort was scored | result | verdict |")
        print("|---|---|---|---|")
        for h in freeze["hypotheses"]:
            print(f"| {h['id']} | {h['decision']} | {detail[h['id']]} | "
                  f"{'confirmed' if verdicts[h['id']] else 'not confirmed'} |")
        return

    print(f"cohort: {freeze['cohort']['n']} domains, panel {freeze['panel']}")
    print(f"sequence-only in panel: {seq_enc}\ncoordinate in panel:    {crd_enc}\n")
    for h in freeze["hypotheses"]:
        print(f"{h['id']}: {'CONFIRMED' if verdicts[h['id']] else 'NOT CONFIRMED'}")
        print(f"   rule:   {h['decision']}")
        print(f"   result: {detail[h['id']]}")
    print(f"\nper-probe detail written to {CONF / 'confirmation_cells.csv'}")
    print(pd.pivot_table(cells, index="probe", columns="encoder", values="gap").round(3).to_string())


if __name__ == "__main__":
    main()
