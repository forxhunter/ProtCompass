"""Determinacy as a measured quantity, label by label, instead of a binary flag.

The paper marks each probe label determined or undetermined by the atoms an encoder receives. A possible
objection is that this is a gradient presented as a binary and is never argued label by label. Two labels make the
objection concrete: relative solvent accessibility is marked undetermined although a fixed descriptor of the same
atoms reaches 0.90 on it, and fold architecture is marked undetermined although structural alignment reaches 0.655
on the superfamily split.

The sense has to be stated precisely, which is what this table does, and it needs three statuses rather than two.
A label is *determined* when it is a closed-form function of the stored atom set: the same atoms give the same
label, always, with no reference to anything outside the file. A label is *undetermined* when two proteins with the
same stored atoms can carry different labels, because the label is fixed by something the file does not contain --
side-chain atoms beyond CB or a crystallographic measurement. A label's determinacy is *not established* when we
can neither exhibit the function nor exhibit two proteins that share an atom set and differ in label; fold
architecture is the one such case, because a CATH architecture describes an arrangement of secondary structures,
which the coordinates fix, while the assignment procedure also uses evidence that is not in the file.

The distinction is not about how well a descriptor predicts the label. A fixed descriptor reaching 0.90 on RSA
means RSA is largely predictable from the backbone, which is true and is why the floor is high. It does not make
RSA a function of the backbone, and the difference is what the argument turns on: on a determined label an encoder
that scores below the floor leaves the label less accessible than its own input does, while on an undetermined
label a score above the floor is an inference that no function of the input could have made. On a label whose
determinacy is not established neither reading is available, and the result stands as a comparison against the
floors we built.

The measured column is the floor score itself, which is the graded quantity the binary was standing in for.

  python determinacy_table.py [--markdown]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))

# label -> (status, what fixes the label, what the stored file holds)
LABELS = {
    "phi_circular": ("determined", "the dihedral of C($i-1$), N($i$), CA($i$), C($i$)",
                     "all four atoms are stored, so the angle is a function of them"),
    "psi_circular": ("determined", "the dihedral of N($i$), CA($i$), C($i$), N($i+1$)",
                     "all four atoms are stored"),
    "local_curvature": ("determined", "the angle between consecutive CA--CA vectors", "CA is stored"),
    "dssp_3": ("determined", "DSSP hydrogen-bond energies from N, CA, C, O",
               "all four atoms are stored; DSSP places H from them"),
    "dssp_8": ("determined", "the same, at eight states", "the same"),
    "core_surface": ("determined", "a CB neighbour count at a fixed radius", "CB is stored or virtual-CB is built"),
    "contact_density": ("determined", "a CA neighbour count at a fixed radius", "CA is stored"),
    "distance_bins": ("determined", "the binned CA--CA distance", "CA is stored"),
    "contact_order_len_resid": ("determined", "a function of the contact map", "CA is stored"),
    "radius_of_gyration_len_resid": ("determined", "the second moment of the CA coordinates", "CA is stored"),
    "asphericity_len_resid": ("determined", "an invariant of the gyration tensor", "CA is stored"),
    "rsa": ("undetermined", "solvent accessibility of the full side chain, computed by freesasa",
            "side-chain atoms beyond CB are not stored, and two residues with identical backbones and different "
            "side chains carry different RSA"),
    "rsa_bins": ("undetermined", "the same, binned", "the same"),
    "bfactor": ("undetermined", "a crystallographic refinement quantity",
                "no function of the coordinates yields it; the same fold refined twice gives different values"),
    "architecture_label": ("not established", "a CATH curator's assignment",
                           "the architecture level describes an arrangement of secondary structures, which the "
                           "stored atoms fix, while the assignment also uses evidence that is not in the file; we "
                           "exhibit neither the function nor a counterexample to it"),
}
LAB = {"phi_circular": "backbone $\\phi$", "psi_circular": "backbone $\\psi$", "local_curvature": "local curvature",
       "dssp_3": "DSSP-3", "dssp_8": "DSSP-8", "core_surface": "core/surface", "contact_density": "contact density",
       "distance_bins": "CA distance bins", "contact_order_len_resid": "contact order",
       "radius_of_gyration_len_resid": "radius of gyration", "asphericity_len_resid": "asphericity",
       "rsa": "RSA", "rsa_bins": "RSA bins", "bfactor": "B-factor", "architecture_label": "fold architecture"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    f = pd.read_csv(ROOT / "probing_results" / "floor_vs_encoders.csv").set_index("probe")
    rows = []
    for p, (status, fixed_by, why) in LABELS.items():
        if p not in f.index:
            continue
        rows.append(dict(probe=LAB.get(p, p), status=status, floor=f.loc[p, "best floor"],
                         best=f.loc[p, "max encoder"], gap=f.loc[p, "gap"], fixed_by=fixed_by, why=why))
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "probing_results" / "determinacy_table.csv", index=False)
    if not a.markdown:
        print(t[["probe", "status", "floor", "best", "gap"]].round(3).to_string(index=False))
        print(f"\ndetermined: floor beats the best encoder on "
              f"{int((t[t.status == 'determined'].gap < 0).sum())} of {int((t.status == 'determined').sum())}")
        print(f"undetermined: an encoder beats the floor on "
              f"{int((t[t.status == 'undetermined'].gap > 0).sum())} of {int((t.status == 'undetermined').sum())}")
        return
    print("% generated by scripts/determinacy_table.py; do not edit by hand")
    print("\\begin{tabular}{llrrp{0.44\\linewidth}}")
    print("\\toprule")
    print("probe & status & floor & best encoder & what fixes the label, and whether the stored atoms fix it \\\\")
    print("\\midrule")
    for r in t.itertuples():
        print(f"{r.probe} & {r.status} & {r.floor:.3f} & {r.best:.3f} & {r.fixed_by}: {r.why} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
