"""chi2 rotamer class for the pre-registered determinacy test, added to an existing prereg labels file.

chi2 is taken only where it is a well-defined, non-symmetric dihedral: CA-CB-CG-CD for Arg, Lys, Gln and Glu,
CA-CB-CG-SD for Met, CA-CB-CG1-CD1 for Ile, and CA-CB-CG-CD1 for Leu and Trp. Phe, Tyr and Asp (180-degree
symmetric), His and Asn (ring or amide flips that crystallography does not resolve) and Pro are left undefined (-1).
Classes as for chi1: g- [0, 120), t [120, 240), g+ [240, 360).

  python prereg_chi2.py --labels probing_data/labels.h5 --structures probing_data/structures --out prereg/labels_released.h5
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from prereg_labels import dihedral

CHI2 = {"ARG": ("CG", "CD"), "LYS": ("CG", "CD"), "GLN": ("CG", "CD"), "GLU": ("CG", "CD"), "MET": ("CG", "SD"),
        "ILE": ("CG1", "CD1"), "LEU": ("CG", "CD1"), "TRP": ("CG", "CD1")}


def one(args):
    pdb, L = args
    res, name = {}, {}
    for line in open(pdb):
        if not line.startswith("ATOM"):
            continue
        a = line[12:16].strip()
        if a[0] == "H":
            continue
        k = int(line[22:26])
        name[k] = line[17:20]
        res.setdefault(k, {})[a] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    out = np.full(L, -1, dtype=np.int8)
    for r, k in enumerate(sorted(res)[:L]):
        atoms = CHI2.get(name[k])
        if atoms and all(x in res[k] for x in ("CA", "CB") + atoms):
            t = np.degrees(dihedral(res[k]["CA"], res[k]["CB"], res[k][atoms[0]], res[k][atoms[1]])) % 360
            out[r] = 0 if t < 120 else (1 if t < 240 else 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--structures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    h = h5py.File(a.labels, "r")
    ids = [x.decode() for x in h["protein/id"][:]]
    L = h["protein/length"][:].astype(np.int64)
    with h5py.File(a.out, "r") as g:
        n = int(g.attrs["n_proteins"])
    with Pool(a.workers) as pool:
        out = pool.map(one, [(Path(a.structures) / f"{ids[i]}.pdb", L[i]) for i in range(n)], chunksize=16)
    v = np.concatenate(out)
    with h5py.File(a.out, "a") as g:
        assert len(v) == len(g["residue/chi1_class"]), (len(v), len(g["residue/chi1_class"]))
        if "residue/chi2_class" in g:
            del g["residue/chi2_class"]
        g["residue/chi2_class"] = v
    ok = v >= 0
    print(f"chi2_class rows {len(v)}  defined {ok.mean():.3f}  classes {np.bincount(v[ok], minlength=3) / ok.sum()}")


if __name__ == "__main__":
    main()
