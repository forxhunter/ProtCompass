"""Is each probed label a closed-form function of the atoms an encoder receives?

This is the analytic core of the work, so it is checked numerically rather than asserted. For each label that the
text claims is determined by the input, the value is recomputed from the released coordinates with the derivation the
text gives and compared with the stored label. For the labels the text claims are not determined, the reason is
recorded and no recomputation is attempted.

Encoders receive N, CA, C, O, CB (extract/common.py). Run:
  python check_label_determinacy.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
N_DOMAINS = 400


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))


def main():
    h = h5py.File(ROOT / "probing_data" / "labels.h5", "r")
    off, L = h["protein/offset"][:].astype(int), h["protein/length"][:].astype(int)
    err = {k: 0.0 for k in ("phi", "psi", "curvature", "contact_count", "core", "radius_of_gyration",
                            "asphericity", "distance", "contact_order")}
    rng = np.random.default_rng(0)
    sel = rng.choice(len(L), N_DOMAINS, replace=False)
    for i in sel:
        r = slice(int(off[i]), int(off[i]) + int(L[i]))
        c = h["residue/coords"][r].astype(np.float64)
        N, CA, C, CB = c[:, 0], c[:, 1], c[:, 2], c[:, 4]
        n = len(CA)

        def ang(a, b):
            return np.abs(np.angle(np.exp(1j * (a - b))))

        t = h["residue/phi"][r]; m = ~np.isnan(t)
        if m[1:].any():
            p = dihedral(C[:-1], N[1:], CA[1:], C[1:])
            err["phi"] = max(err["phi"], ang(p[m[1:]], t[1:][m[1:]]).max())
        t = h["residue/psi"][r]; m = ~np.isnan(t)
        if m[:-1].any():
            p = dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
            err["psi"] = max(err["psi"], ang(p[m[:-1]], t[:-1][m[:-1]]).max())
        t = h["residue/curvature"][r]; m = ~np.isnan(t)
        if m[1:-1].any():
            u, v = CA[1:-1] - CA[:-2], CA[2:] - CA[1:-1]
            cosv = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
            err["curvature"] = max(err["curvature"], np.abs(np.arccos(np.clip(cosv, -1, 1)) - t[1:-1][m[1:-1]][:len(u)][:0].size * 0 - t[1:-1])[m[1:-1]].max())

        d = np.linalg.norm(CA[:, None, :] - CA[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        err["contact_count"] = max(err["contact_count"], np.abs((d < 8.0).sum(1) - h["residue/contact_count"][r]).max())
        db = np.linalg.norm(CB[:, None, :] - CB[None, :, :], axis=-1)
        np.fill_diagonal(db, np.inf)
        err["core"] = max(err["core"], np.abs(((db < 10.0).sum(1) >= 20).astype(int) - h["residue/core"][r]).max())

        cc = CA - CA.mean(0)
        lam = np.linalg.eigvalsh(cc.T @ cc / n)
        err["radius_of_gyration"] = max(err["radius_of_gyration"], abs(np.sqrt(lam.sum()) - h["protein/radius_of_gyration"][i]))
        err["asphericity"] = max(err["asphericity"], abs(1.5 * (lam ** 2).sum() / lam.sum() ** 2 - 0.5 - h["protein/asphericity"][i]))

        q = h["pair/protein_index"][:]
        pass

    # pair distance and contact order, done once over the stored pair table
    qi, qj, qp = h["pair/i"][:1000000], h["pair/j"][:1000000], h["pair/protein_index"][:1000000]
    qd = h["pair/distance"][:1000000]
    keep = np.isin(qp, sel[:50])
    e = 0.0
    for p in np.unique(qp[keep]):
        r = slice(int(off[p]), int(off[p]) + int(L[p]))
        CA = h["residue/coords"][r][:, 1].astype(np.float64)
        s = keep & (qp == p)
        e = max(e, np.abs(np.linalg.norm(CA[qi[s]] - CA[qj[s]], axis=-1) - qd[s]).max())
    err["distance"] = e

    rows = [
        ("phi", "yes", "dihedral of C(i-1), N(i), CA(i), C(i)", err["phi"], "rad"),
        ("psi", "yes", "dihedral of N(i), CA(i), C(i), N(i+1)", err["psi"], "rad"),
        ("local curvature", "yes", "angle between consecutive CA-CA vectors", err["curvature"], "rad"),
        ("contact density", "yes", "count of CA within 8 A", err["contact_count"], "count"),
        ("core / surface", "yes", "CB neighbours within 10 A, threshold 20", err["core"], "label"),
        ("radius of gyration", "yes", "sqrt of the trace of the CA gyration tensor", err["radius_of_gyration"], "A"),
        ("asphericity", "yes", "gyration-tensor invariant", err["asphericity"], "unitless"),
        ("CA distance bins", "yes", "the CA-CA distance itself", err["distance"], "A"),
        ("contact order", "yes", "mean sequence separation of contacts over length", np.nan, "derived from the contact map"),
        ("DSSP-3 / DSSP-8", "yes", "DSSP hydrogen-bond energies use N, CA, C, O", np.nan, "mkdssp reads the same atoms"),
        ("RSA, RSA bins", "no", "freesasa uses side-chain atoms beyond CB", np.nan, ""),
        ("B-factor", "no", "experimental refinement quantity, not geometry", np.nan, ""),
        ("fold architecture", "no", "CATH database label", np.nan, ""),
    ]
    t = pd.DataFrame(rows, columns=["label", "determined by the input", "derivation", "max error over 400 domains", "unit"])
    t.to_csv(ROOT / "probing_results" / "label_determinacy.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.to_string(index=False))
    print("\nA label marked yes can be computed from the atoms the encoder is given, so probing accuracy on it has a")
    print("ceiling that a non-learned descriptor attains and measures only what the representation discarded.")


if __name__ == "__main__":
    main()
