"""Six residue labels for the pre-registered determinacy test, computed from the cleaned full-atom domain files.

Three are closed-form functions of the five atoms every coordinate encoder is handed (N, CA, C, O, CB); three need
side-chain atoms beyond CB, which only the encoders that read the structure file itself receive.

  determined by the five atoms
    hbond_count      backbone hydrogen bonds the residue takes part in, as donor or acceptor, DSSP electrostatic
                     energy below -0.5 kcal/mol with the amide H placed from the backbone; classes 0, 1, 2 or more
    hse_up           half-sphere exposure: CA atoms within 13 A in the half-sphere the CA->CB vector points into
    ca_dihedral      the CA(i-1), CA(i), CA(i+1), CA(i+2) virtual dihedral, as (sin, cos)
  not determined by the five atoms
    chi1_class       chi1 rotamer (N, CA, CB, gamma atom): g- [0, 120), t [120, 240), g+ [240, 360); -1 without gamma
    sc_rsa           side-chain relative accessible surface (freesasa, as the released RSA label); NaN for Gly
    sc_contacts      other residues with a side-chain heavy atom within 4.5 A of this residue's side chain; NaN for Gly

The domain files are the ones the released dataset wrote (residues renumbered 1..L in labels.h5 order), so row r of
every output array is row r of residue/* in the labels file. Output: <out>, arrays under residue/.

  python prereg_labels.py --labels probing_data/labels.h5 --structures probing_data/structures --out prereg/labels_released.h5
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

GAMMA = {"CG", "OG", "OG1", "SG", "CG1"}
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def parse(path):
    res = {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name[0] == "H":
            continue
        k = int(line[22:26])
        res.setdefault(k, {})[name] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return res


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))


def one(args):
    pdb, coords = args
    L = len(coords)
    N, CA, C, O, CB = (coords[:, k].astype(np.float64) for k in range(5))
    # backbone hydrogen bonds (Kabsch & Sander); H of residue i from the C=O of residue i-1
    H = N.copy()
    d = C[:-1] - O[:-1]
    H[1:] = N[1:] + d / np.linalg.norm(d, axis=1, keepdims=True)
    rON = np.linalg.norm(O[None, :, :] - N[:, None, :], axis=-1)       # [donor i, acceptor j]
    rCH = np.linalg.norm(C[None, :, :] - H[:, None, :], axis=-1)
    rOH = np.linalg.norm(O[None, :, :] - H[:, None, :], axis=-1)
    rCN = np.linalg.norm(C[None, :, :] - N[:, None, :], axis=-1)
    with np.errstate(divide="ignore"):
        E = 0.084 * 332.0 * (1 / rON + 1 / rCH - 1 / rOH - 1 / rCN)
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    bond = (E < -0.5) & (sep >= 2)
    bond[0, :] = False                                                  # the first residue has no placed H
    count = bond.sum(1) + bond.sum(0)
    hbond = np.minimum(count, 2).astype(np.int8)
    # half-sphere exposure (up), from CA and CB
    u = CB - CA
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    diff = CA[None, :, :] - CA[:, None, :]
    within = (np.linalg.norm(diff, axis=-1) < 13.0) & ~np.eye(L, dtype=bool)
    up = ((diff * u[:, None, :]).sum(-1) > 0) & within
    hse = up.sum(1).astype(np.float32)
    # CA virtual dihedral
    ang = np.full(L, np.nan)
    if L >= 4:
        ang[1:L - 2] = dihedral(CA[:-3], CA[1:-2], CA[2:-1], CA[3:])
    cad = np.stack([np.sin(ang), np.cos(ang)], 1).astype(np.float32)
    # side-chain labels from the full-atom file
    res = parse(pdb)
    keys = sorted(res)
    chi = np.full(L, -1, dtype=np.int8)
    sc_atoms = []
    for r, k in enumerate(keys[:L]):
        a = res[k]
        g = [x for x in GAMMA if x in a]
        if g and all(x in a for x in ("N", "CA", "CB")):
            t = np.degrees(dihedral(a["N"], a["CA"], a["CB"], a[sorted(g)[0]])) % 360
            chi[r] = 0 if t < 120 else (1 if t < 240 else 2)
        sc_atoms.append(np.array([v for n, v in a.items() if n not in BACKBONE and n != "CB"] +
                                 ([a["CB"]] if "CB" in a else [])))
    cont = np.full(L, np.nan, dtype=np.float32)
    owner = np.concatenate([np.full(len(x), i) for i, x in enumerate(sc_atoms)]) if L else np.zeros(0)
    allsc = np.concatenate([x for x in sc_atoms if len(x)]) if any(len(x) for x in sc_atoms) else np.zeros((0, 3))
    from scipy.spatial import cKDTree
    tree = cKDTree(allsc) if len(allsc) else None
    for i, x in enumerate(sc_atoms):
        if len(x) <= 1 or tree is None:                                   # Gly, or no side chain beyond CB
            continue
        hits = set()
        for lst in tree.query_ball_point(x, 4.5):
            hits.update(int(owner[j]) for j in lst)
        hits.discard(i)
        cont[i] = len(hits)
    import freesasa
    freesasa.setVerbosity(freesasa.silent)
    rsa = np.full(L, np.nan, dtype=np.float32)
    try:
        s = freesasa.Structure(str(pdb), options={"skip-unknown": True})
        for num, area in freesasa.calc(s).residueAreas().get("A", {}).items():
            idx = int(num.strip()) - 1
            if 0 <= idx < L and area.residueType != "GLY":
                rsa[idx] = area.relativeSideChain
    except Exception:
        pass
    return dict(hbond_count=hbond, hse_up=hse, ca_dihedral=cad, chi1_class=chi, sc_rsa=rsa, sc_contacts=cont)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--structures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    h = h5py.File(a.labels, "r")
    ids = [x.decode() for x in h["protein/id"][:]]
    L, off = h["protein/length"][:].astype(np.int64), h["protein/offset"][:].astype(np.int64)
    n = a.limit or len(ids)
    coords = h["residue/coords"]
    jobs = [(Path(a.structures) / f"{ids[i]}.pdb", coords[off[i]:off[i] + L[i]]) for i in range(n)]
    with Pool(a.workers) as pool:
        out = pool.map(one, jobs, chunksize=16)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as g:
        for k in out[0]:
            g[f"residue/{k}"] = np.concatenate([o[k] for o in out])
        g.attrs["n_proteins"] = n
        g.attrs["source_labels"] = str(a.labels)
    for k in out[0]:
        v = np.concatenate([o[k] for o in out])
        v = v.reshape(len(v), -1)[:, 0].astype(float)
        ok = np.isfinite(v) & (v >= 0) if k in ("hbond_count", "chi1_class") else np.isfinite(v)
        print(f"{k:12s} rows {len(v)}  defined {ok.mean():.3f}  mean {np.nanmean(np.where(ok, v, np.nan)):.3f}")


if __name__ == "__main__":
    main()
