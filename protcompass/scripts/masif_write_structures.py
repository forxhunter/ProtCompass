"""Write the all-atom structure file of every MaSIF-site chain of masif_site/labels.h5.

The encoders that read a structure file (both GCPNet variants and dMaSIF) need one file per chain. The source entry
stores 37 heavy-atom slots per residue in the graphein order; this script writes the residues that
scripts/masif_build_labels.py kept (same rules, imported from it), renumbered 1..L in one chain A, in the format of
probing_data/make_probing_data.py:write_clean_pdb. Row r of a written file is row r of the chain in labels.h5.

Checks, all of which raise:
  - the slots present for each residue are heavy atoms of its residue type (this also tests the slot order);
  - the N, CA, C, O and CB coordinates written equal residue/coords of labels.h5, and the residue types equal residue/aa.

  python scripts/masif_write_structures.py [--workers 8]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import masif_build_labels as B  # noqa: E402

OUT = B.OUT / "structures"
SLOTS = ["N", "CA", "C", "O", "CB", "OG", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OD1", "ND2", "CG1", "CG2", "CD",
         "CE", "NZ", "OD2", "OE1", "NE2", "OE2", "OH", "NE", "NH1", "NH2", "OG1", "SD", "ND1", "SG", "NE1", "CE3",
         "CZ2", "CZ3", "CH2", "OXT"]
BB = {"N", "CA", "C", "O", "OXT"}
HEAVY = {"ALA": {"CB"}, "ARG": {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}, "ASN": {"CB", "CG", "OD1", "ND2"},
         "ASP": {"CB", "CG", "OD1", "OD2"}, "CYS": {"CB", "SG"}, "GLN": {"CB", "CG", "CD", "OE1", "NE2"},
         "GLU": {"CB", "CG", "CD", "OE1", "OE2"}, "GLY": set(), "HIS": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
         "ILE": {"CB", "CG1", "CG2", "CD1"}, "LEU": {"CB", "CG", "CD1", "CD2"}, "LYS": {"CB", "CG", "CD", "CE", "NZ"},
         "MET": {"CB", "CG", "SD", "CE"}, "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
         "PRO": {"CB", "CG", "CD"}, "SER": {"CB", "OG"}, "THR": {"CB", "OG1", "CG2"},
         "TRP": {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
         "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}, "VAL": {"CB", "CG1", "CG2"}}


def write_one(job):
    entry, coords_ref, aa_ref = job
    import torch
    d = torch.load(B.SRC / "processed" / f"{entry}.pt", weights_only=False)
    target = str(d.graph_y)
    xyz = d.coords.numpy().astype(np.float64)
    present = ~np.all(np.abs(xyz - B.SENTINEL) < 1e-9, axis=-1)
    chain = np.array([r.split(":")[0] for r in d.residue_id])
    names = [B.M.REMAP.get(n, n) for n in d.residues]
    tgt = np.where(chain == target)[0]
    nonstd = np.array([names[i] not in B.M.AA_INDEX for i in tgt])
    incomplete = ~present[tgt][:, :4].all(1)
    keep = tgt[~nonstd & ~incomplete]
    if len(keep) != len(aa_ref):
        raise RuntimeError(f"{entry}: {len(keep)} kept residues, {len(aa_ref)} in labels.h5")
    serial, out, n_extra = 1, [], 0
    for k, i in enumerate(keep, start=1):
        name = names[i]
        if B.M.AA_INDEX[name] != aa_ref[k - 1]:
            raise RuntimeError(f"{entry}: residue {k} is {name}, labels.h5 has index {aa_ref[k - 1]}")
        for s in np.where(present[i])[0]:
            atom = SLOTS[s]
            if atom not in BB and atom not in HEAVY[name]:
                if d.residues[i] == name:          # a remapped residue (selenomethionine, for example) may differ
                    raise RuntimeError(f"{entry}: slot {atom} is present for {name}; the slot order is not as assumed")
                n_extra += 1
                continue
            x, y, z = xyz[i, s]
            if s < 4 and not np.allclose(xyz[i, s], coords_ref[k - 1, s], atol=1e-4):
                raise RuntimeError(f"{entry}: {atom} of residue {k} differs from labels.h5")
            if s == 4 and not np.allclose(xyz[i, s], coords_ref[k - 1, 4], atol=1e-4):
                raise RuntimeError(f"{entry}: CB of residue {k} differs from labels.h5")
            padded = f" {atom:<3s}" if len(atom) < 4 else atom
            out.append(f"ATOM  {serial:5d} {padded} {name:>3s} A{k:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                       f"  1.00  0.00          {atom[0]:>2s}")
            serial += 1
    out += ["TER", "END"]
    (OUT / f"{entry}.pdb").write_text("\n".join(out) + "\n")
    return entry, len(keep), serial - 1, n_extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with h5py.File(B.OUT / "labels.h5", "r") as h:
        ids = [x.decode() for x in h["protein/id"][:]]
        off, L = h["protein/offset"][:], h["protein/length"][:]
        coords, aa = h["residue/coords"][:], h["residue/aa"][:]
    jobs = [(e, coords[o:o + n], aa[o:o + n]) for e, o, n in zip(ids, off, L)]
    with Pool(a.workers) as pool:
        res = pool.map(write_one, jobs, chunksize=16)
    n_res, n_atoms, n_extra = (sum(r[k] for r in res) for k in (1, 2, 3))
    print(f"{len(res)} files in {OUT}: {n_res} residues, {n_atoms} atoms, {n_atoms / n_res:.2f} atoms per residue, "
          f"{n_extra} atoms of remapped residues dropped")


if __name__ == "__main__":
    main()
