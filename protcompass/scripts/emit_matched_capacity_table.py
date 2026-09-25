"""The floor comparison with every source projected to a common width (PROJECT_K, default 116) before probing.

Reads probing_results/by_source_k<K>/*.csv, written by `PROJECT_K=<K> run_probing_suite.py`, and prints the LaTeX
body of one table: for each determined probe the coordinate floor and the count of coordinate encoders below it,
the sequence floor and the count of sequence-only encoders below it, next to the released (unprojected) counts; and
for each undetermined label the best encoder minus the best floor. The projected run scores the eight original
floors; the three sequence floors added after review are not projected.

  python emit_matched_capacity_table.py [K] > ../../manuscript/ICLR/ProtCompass_ICLR2027/generated/matched_capacity.tex
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
RES = ROOT / "probing_results"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 116
SEQ_FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global"]
ALL_FLOORS = SEQ_FLOORS + ["ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]
DET = ["phi_circular", "psi_circular", "local_curvature", "dssp_3", "dssp_8", "core_surface", "contact_density",
       "distance_bins", "contact_order_len_resid", "radius_of_gyration_len_resid", "asphericity_len_resid"]
TAUT = {"contact_density", "contact_order_len_resid"}
UND = ["rsa", "rsa_bins", "bfactor", "architecture_label"]  # the labels the stored atoms are not shown to fix
UNRESOLVED = {"architecture_label"}  # determinacy neither established nor refuted; see the determinacy table
LABEL = {"phi_circular": "backbone $\\phi$", "psi_circular": "backbone $\\psi$", "local_curvature": "local curvature",
         "dssp_3": "DSSP-3", "dssp_8": "DSSP-8", "core_surface": "core/surface", "contact_density": "contact density",
         "distance_bins": "CA distance bins", "contact_order_len_resid": "contact order",
         "radius_of_gyration_len_resid": "radius of gyration", "asphericity_len_resid": "asphericity",
         "rsa": "RSA", "rsa_bins": "RSA bins", "bfactor": "B-factor", "architecture_label": "fold architecture"}


def load(sub):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / sub / "*.csv"))], ignore_index=True)
    d["headline"] = d["pearson"].fillna(d["circ_cos"] if "circ_cos" in d else np.nan).fillna(d["balanced_accuracy"])
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    m = d.groupby(["source", "probe", "family"]).headline.mean().reset_index()
    best = m.loc[m.groupby(["source", "probe"]).headline.idxmax()]
    return best.pivot(index="source", columns="probe", values="headline")


def counts(wide):
    audit = pd.read_csv(RES / "floor_assignment.csv").rename(columns={"Unnamed: 0": "source"}).set_index("source")
    enc = [i for i in wide.index if not i.startswith("ref_")]
    seq = [e for e in enc if e in audit.index and not audit.loc[e, "uses_coordinates"]]
    crd = [e for e in enc if e in audit.index and audit.loc[e, "uses_coordinates"]]
    out = {}
    for p in DET + UND:
        sf = wide.loc[[f for f in SEQ_FLOORS if f in wide.index], p].max()
        af = wide.loc[[f for f in ALL_FLOORS if f in wide.index], p].max()
        s, c, e = wide.loc[seq, p].dropna(), wide.loc[crd, p].dropna(), wide.loc[enc, p].dropna()
        out[p] = dict(sf=sf, af=af, seq_below=int((s < sf).sum()), n_seq=len(s), crd_below=int((c < af).sum()),
                      n_crd=len(c), gap=e.max() - af, above=int((e > af).sum()), n_enc=len(e))
    return out


def main():
    r0, r1 = counts(load("by_source")), counts(load(f"by_source_k{K}"))
    print("% generated file: the whole tabular, so the caller never inputs rows into an open tabular")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print(f"probe & coord.\\ floor, released $\\to$ at {K} & coord.\\ below, released $\\to$ at {K} & "
          f"seq.\\ below at {K} & best enc.\\ $-$ floor at {K} \\\\")
    print("\\midrule")
    for p in DET:
        a, b = r0[p], r1[p]
        tag = " (floor contains the label)" if p in TAUT else ""
        print(f"{LABEL[p]}{tag} & {a['af']:.3f} $\\to$ {b['af']:.3f} & {a['crd_below']} / {a['n_crd']} $\\to$ "
              f"{b['crd_below']} / {b['n_crd']} & {b['seq_below']} / {b['n_seq']} & {b['gap']:+.3f} \\\\")
    print("\\midrule")
    for p in UND:
        a, b = r0[p], r1[p]
        tag = "not established" if p in UNRESOLVED else "undetermined"
        print(f"{LABEL[p]} ({tag}) & {a['af']:.3f} $\\to$ {b['af']:.3f} & {a['above']} / {a['n_enc']} above "
              f"$\\to$ {b['above']} / {b['n_enc']} above & -- & {b['gap']:+.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    det = [p for p in DET]; inf = [p for p in DET if p not in TAUT]
    print(f"% at {K}: coord below {sum(r1[p]['crd_below'] for p in det)} / {sum(r1[p]['n_crd'] for p in det)} "
          f"(informative {sum(r1[p]['crd_below'] for p in inf)} / {sum(r1[p]['n_crd'] for p in inf)}); "
          f"seq below {sum(r1[p]['seq_below'] for p in det)} / {sum(r1[p]['n_seq'] for p in det)}; "
          f"floor wins on {sum(r1[p]['gap'] < 0 for p in inf)} of {len(inf)} informative determined probes; "
          f"released: coord {sum(r0[p]['crd_below'] for p in det)} / {sum(r0[p]['n_crd'] for p in det)}, "
          f"floor wins on {sum(r0[p]['gap'] < 0 for p in inf)} of {len(inf)}")


if __name__ == "__main__":
    main()
