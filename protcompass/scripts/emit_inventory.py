"""The encoder inventory and the probe inventory, generated from the released files.

An earlier draft did not let a reader enumerate the evaluated models: the specification tables described the
predecessor panel, listed encoders that are excluded, and omitted two that are scored. Both tables here are
generated from what the release actually contains, so they cannot drift again.

  python emit_inventory.py encoders   LaTeX rows for the encoder inventory
  python emit_inventory.py probes     LaTeX rows for the probe inventory
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
EMB = ROOT / "extract" / "embeddings" / "stability"
sys.path.insert(0, str(ROOT / "scripts"))
from make_panels_original_style import DECLARED, DISP  # noqa: E402

CLASS = {"seq": "sequence", "struct": "structure", "multi": "multimodal"}
UND = {"rsa", "rsa_bins", "bfactor", "architecture_label"}  # labels the stored atoms are not shown to determine
UNRESOLVED = {"architecture_label"}  # a member of UND whose determinacy we neither establish nor refute
TAUT = {"contact_density", "contact_order_len_resid"}
PROBE_LABEL = {"phi_circular": "backbone $\\phi$", "psi_circular": "backbone $\\psi$",
               "local_curvature": "local curvature", "dssp_3": "DSSP-3", "dssp_8": "DSSP-8",
               "core_surface": "core/surface", "contact_density": "contact density",
               "distance_bins": "CA distance bins", "contact_order_len_resid": "contact order",
               "radius_of_gyration_len_resid": "radius of gyration", "asphericity_len_resid": "asphericity",
               "rsa": "RSA", "rsa_bins": "RSA bins", "bfactor": "B-factor",
               "architecture_label": "fold architecture", "sequence_recovery": "sequence recovery",
               "contact_prediction": "contact prediction", "phi_angle_bins": "$\\phi$ bins",
               "psi_angle_bins": "$\\psi$ bins", "radius_of_gyration": "radius of gyration (raw)",
               "asphericity": "asphericity (raw)", "contact_order": "contact order (raw)"}


def _open(cols, header):
    print("% generated file: the whole tabular, so the caller never inputs rows into an open tabular")
    print("\\begin{tabular}{" + cols + "}")
    print("\\toprule")
    print(header + " \\\\")
    print("\\midrule")


def _close():
    print("\\bottomrule")
    print("\\end{tabular}")


def human(p):
    if not p:
        return "--"
    return f"{p / 1e9:.1f}B" if p >= 1e9 else (f"{p / 1e6:.0f}M" if p >= 1e6 else f"{p / 1e3:.0f}K")


def width(e):
    """The width of the released embedding, read from the array rather than from what the wrapper declares."""
    import numpy as _np
    f = EMB / e / "train.npy"
    return int(_np.load(f, mmap_mode="r").shape[1]) if f.exists() else 0


def short(ck):
    """A checkpoint identifier a reader can look up: a hub id as given, a local path by its last component."""
    ck = str(ck)
    return ck.rsplit("/", 1)[-1] if ck.startswith("/") else ck


def encoders():
    audit = pd.read_csv(ROOT / "extract" / "encoder_input_audit.csv").drop_duplicates("encoder").set_index("encoder")
    scored = pd.read_csv(ROOT / "extract" / "scores2_selected.csv")
    n_tasks = scored[scored.encoder != "ref_composition"].groupby("encoder").task.nunique()
    rows = []
    for d in sorted(EMB.glob("*/meta.json")):
        e = d.parent.name
        if e == "ref_composition":
            continue                                   # the non-learned downstream reference, not an encoder
        m = json.loads(d.read_text())
        uses = audit.uses_coordinates.get(e, np.nan)
        rows.append(dict(encoder=DISP.get(e, e), key=e, cls=CLASS[DECLARED.get(e, "struct")],
                         weights=m.get("weights", "pretrained"), params=human(m.get("n_parameters")),
                         dim=width(e),
                         checkpoint=short(m.get("checkpoint") or m.get("target") or "--"),
                         reads_coords={True: "yes", False: "no"}.get(bool(uses) if uses == uses else None, "--"),
                         tasks=int(n_tasks.get(e, 0))))
    t = pd.DataFrame(rows).sort_values(["cls", "encoder"])
    _open("llrrrrl", "encoder & weights & parameters & width & reads coordinates & tasks scored & checkpoint")
    for cls in ("sequence", "structure", "multimodal"):
        g = t[t.cls == cls]
        print(f"\\multicolumn{{7}}{{l}}{{\\textit{{declared {cls} ({len(g)})}}}} \\\\")
        for r in g.itertuples():
            ck = str(r.checkpoint).replace("_", "\\_")
            ck = ck if len(ck) < 42 else ck[:39] + "..."
            print(f"{r.encoder} & {r.weights} & {r.params} & {r.dim} & {r.reads_coords} & {r.tasks} & "
                  f"\\texttt{{\\scriptsize {ck}}} \\\\")
    _close()
    print(f"% {len(t)} encoders; widths are the width of the released embedding, measured, not declared")


# Why a probe that was run does not enter the floor comparison. Every probe in the suite is listed, so the
# selection from the full suite to the comparison set is on the record rather than implicit.
NOT_ENTERED = {
    "sequence_recovery": "uninformative for any source given residue identity: 7 encoders and 2 floors score 1.000",
    "phi_angle_bins": "binned twin of backbone $\\phi$; the circular form is reported",
    "psi_angle_bins": "binned twin of backbone $\\psi$; the circular form is reported",
    "radius_of_gyration": "raw form; the length-residualised form is reported, since length alone predicts it",
    "asphericity": "raw form; the length-residualised form is reported",
    "contact_order": "raw form; the length-residualised form is reported",
}


PPG = {"phi_circular": "I1", "psi_circular": "I1", "phi_angle_bins": "I1", "psi_angle_bins": "I1",
       "local_curvature": "I2", "dssp_3": "I3", "dssp_8": "I3", "rsa": "I4", "rsa_bins": "I4", "bfactor": "I5",
       "core_surface": "I6", "contact_density": "I6", "distance_bins": "I7", "contact_order_len_resid": "I8",
       "radius_of_gyration_len_resid": "I8", "asphericity_len_resid": "I8", "contact_order": "I8",
       "radius_of_gyration": "I8", "asphericity": "I8", "architecture_label": "I9", "sequence_recovery": "I10"}


def probes():
    f = pd.read_csv(ROOT / "probing_results" / "floor_vs_encoders.csv").set_index("probe")
    every = sorted(pd.read_csv(ROOT / "probing_results" / "by_source" / "esm2_650m.csv").probe.unique())
    entered = list(f.index)
    _open("llllp{0.40\\linewidth}", "probe & PPG & label status & in the comparison & note")
    for p in entered:
        status = ("not established" if p in UNRESOLVED else "undetermined" if p in UND else "determined")
        note = ("the floor contains the label by construction" if p in TAUT else
                "neither shown to be a function of the encoder's input nor shown not to be" if p in UNRESOLVED else
                "not a function of the encoder's input" if p in UND else
                "a closed-form function of the coordinates")
        print(f"{PROBE_LABEL.get(p, p)} & {PPG[p]} & {status} & yes & {note} \\\\")
    for p in [q for q in every if q not in entered]:
        print(f"{PROBE_LABEL.get(p, p)} & {PPG[p]} & --- & no & {NOT_ENTERED.get(p, 'not in the comparison set')} \\\\")
    _close()
    print(f"% {len(every)} probes in the suite, {len(entered)} enter the floor comparison")


if __name__ == "__main__":
    {"encoders": encoders, "probes": probes}[sys.argv[1]]()
