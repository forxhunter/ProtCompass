"""Two generic reference descriptors added for the pre-registered determinacy test, before any new label is scored.

Neither descriptor is built for a particular label. Neither contains a chi angle, a hydrogen-bond energy or any
other quantity the new labels are computed from; both are fixed functions of the atoms an encoder reads.

  ref_atom_neighborhood@S   non-local atom-neighbour descriptor, built from the atom set S an encoder reads
                            (CA, N+CA+C, N+CA+C+O, N+CA+CB from the five stored atoms; full = every heavy atom of
                            the domain file). Atoms are grouped by element (C, N, O, S). For every pair of element
                            classes (query q, target t) and residue i: the K = 8 smallest distances from an atom of
                            class q in residue i to an atom of class t in another residue, the signed sequence
                            separation of those K partners (clipped to +-32, scaled to [-1, 1]), and the number of
                            class-t atoms of other residues within 3.5, 5, 7.5, 10 and 13 A; plus one presence flag
                            per query class. Missing partners are padded with 20 A and separation 0.
  ref_allatom_local         full-atom residue descriptor (full-atom readers only): residue identity (20), the
                            atom14 presence mask (14), the atom14 coordinates in the residue's N-CA-C frame (42),
                            all intra-residue atom14 distances (91), and the distances from every atom14 slot to
                            N, CA, C and O of residues i-1 and i+1 (112). The frame coordinates carry chirality, so no
                            separate signed volumes are needed.

The five stored atoms come from the labels file (residue/coords, N CA C O CB); full-atom geometry comes from the
domain files the dataset was built from, whose residues are numbered 1..L in labels order (as in prereg_labels.py).
Features are computed for the proteins the released draw uses (train, validation, test) and cached under
prereg/floor_cache/<labels hash>/.

  python prereg_floors.py --build [--workers 6]      # compute and cache every floor for the current PROBING_LABELS
  python prereg_floors.py --check                    # shapes, NaN, and alignment of the full-atom parse with labels
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import hashlib
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_probing_suite as rps                                          # noqa: E402
from atom_matched_floors import AtomData                                 # noqa: E402

if os.environ.get("PREREG_COHORT") == "B":
    rps.N_PROT["test"] = 10 ** 6          # every cohort B domain; train and validation are the released draw

ATOM_SETS = ["CA", "N+CA+C", "N+CA+C+O", "N+CA+CB", "full"]
FLOORS = [f"ref_atom_neighborhood@{s}" for s in ATOM_SETS] + ["ref_allatom_local"]
STORED = ["N", "CA", "C", "O", "CB"]
ELEMENTS = ["C", "N", "O", "S"]
K, SHELLS, PAD, SEP = 8, (3.5, 5.0, 7.5, 10.0, 13.0), 20.0, 32
STRUCT_DIRS = [ROOT / "probing_data" / "structures", ROOT / "confirmation2" / "structures"]
ATOM14 = {
    "ALA": "N CA C O CB", "ARG": "N CA C O CB CG CD NE CZ NH1 NH2", "ASN": "N CA C O CB CG OD1 ND2",
    "ASP": "N CA C O CB CG OD1 OD2", "CYS": "N CA C O CB SG", "GLN": "N CA C O CB CG CD OE1 NE2",
    "GLU": "N CA C O CB CG CD OE1 OE2", "GLY": "N CA C O", "HIS": "N CA C O CB CG ND1 CD2 CE1 NE2",
    "ILE": "N CA C O CB CG1 CG2 CD1", "LEU": "N CA C O CB CG CD1 CD2", "LYS": "N CA C O CB CG CD CE NZ",
    "MET": "N CA C O CB CG SD CE", "PHE": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ", "PRO": "N CA C O CB CG CD",
    "SER": "N CA C O CB OG", "THR": "N CA C O CB OG1 CG2", "TRP": "N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "N CA C O CB CG1 CG2"}
ATOM14 = {k: v.split() for k, v in ATOM14.items()}
RESTYPES = sorted(ATOM14)
IU = np.triu_indices(14, 1)


def structure_path(pid):
    for d in STRUCT_DIRS:
        p = d / f"{pid}.pdb"
        if p.exists():
            return p
    raise FileNotFoundError(pid)


def parse_full(path, L):
    """Heavy atoms of the first L residues, in file order: list of dicts atom name -> xyz, and residue names."""
    res, name = {}, {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        a = line[12:16].strip()
        if a[0] == "H":
            continue
        k = int(line[22:26])
        name[k] = line[17:20]
        res.setdefault(k, {})[a] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    keys = sorted(res)[:L]
    return [res[k] for k in keys], [name[k] for k in keys]


def element(atom_name):
    return atom_name[0]


def neighbourhood(xyz, res_of, elem, L, classes):
    """xyz (M, 3), residue index (M,), element letter (M,) -> (L, D) descriptor. `classes` is fixed per atom set, so
    the dimension does not depend on which elements a protein happens to contain (absent classes are padded)."""
    feats = []
    for q in classes:
        qm = elem == q
        present = np.zeros(L); present[np.unique(res_of[qm])] = 1.0
        feats.append(present[:, None])
        qx, qr = xyz[qm], res_of[qm]
        order = np.argsort(qr, kind="stable"); qx, qr = qx[order], qr[order]
        has = np.isin(np.arange(L), qr)
        for t in classes:
            tm = elem == t
            if not qm.any() or not tm.any():
                feats += [np.full((L, K), PAD, np.float32), np.zeros((L, K), np.float32),
                          np.zeros((L, len(SHELLS)), np.float32)]
                continue
            tx, tr = xyz[tm], res_of[tm]
            Rmin = np.full((L, len(tx)), np.inf, dtype=np.float32)
            for s in range(0, len(qx), 4096):
                D = np.linalg.norm(qx[s:s + 4096, None, :] - tx[None, :, :], axis=-1).astype(np.float32)
                r = qr[s:s + 4096]
                u, first = np.unique(r, return_index=True)
                Rmin[u] = np.minimum(Rmin[u], np.minimum.reduceat(D, first, axis=0))
            Rmin[tr[None, :] == np.arange(L)[:, None]] = np.inf      # other residues only
            k = min(K, len(tx))
            idx = np.argpartition(Rmin, k - 1, axis=1)[:, :k] if len(tx) > k else np.tile(np.arange(len(tx)), (L, 1))
            d = np.take_along_axis(Rmin, idx, 1)
            o = np.argsort(d, 1); d = np.take_along_axis(d, o, 1); idx = np.take_along_axis(idx, o, 1)
            sep = np.clip(tr[idx] - np.arange(L)[:, None], -SEP, SEP) / SEP
            sep = np.where(np.isfinite(d), sep, 0.0)
            d = np.where(np.isfinite(d), d, PAD)
            if k < K:
                d = np.pad(d, ((0, 0), (0, K - k)), constant_values=PAD); sep = np.pad(sep, ((0, 0), (0, K - k)))
            cnt = np.stack([(Rmin < r).sum(1) for r in SHELLS], 1).astype(np.float32)
            d[~has] = PAD; sep[~has] = 0.0; cnt[~has] = 0.0
            feats += [d, sep, cnt]
    return np.concatenate(feats, 1).astype(np.float32)


def frame(n, ca, c):
    x = c - ca; x /= np.linalg.norm(x) + 1e-8
    z = np.cross(x, n - ca); z /= np.linalg.norm(z) + 1e-8
    return np.stack([x, np.cross(z, x), z])                              # rows: x, y, z axes


def allatom_local(atoms, names, L):
    out = np.zeros((L, 20 + 14 + 42 + 91 + 112), dtype=np.float32)
    bb = np.full((L, 4, 3), np.nan)
    slots = np.full((L, 14, 3), np.nan)
    for i, (a, nm) in enumerate(zip(atoms, names)):
        for j, x in enumerate(ATOM14.get(nm, [])):
            if x in a:
                slots[i, j] = a[x]
        for j, x in enumerate(("N", "CA", "C", "O")):
            if x in a:
                bb[i, j] = a[x]
    for i, nm in enumerate(names):
        f, mask = [], ~np.isnan(slots[i, :, 0])
        onehot = np.zeros(20); onehot[RESTYPES.index(nm)] = 1.0 if nm in RESTYPES else 0.0
        s = np.nan_to_num(slots[i])
        if mask[:3].all():
            local = (s - s[1]) @ frame(s[0], s[1], s[2]).T
            local[~mask] = 0.0
        else:
            local = np.zeros((14, 3))
        D = np.linalg.norm(s[:, None] - s[None], axis=-1); D[~(mask[:, None] & mask[None])] = 0.0
        nb = []
        for j in (i - 1, i + 1):
            if 0 <= j < L and not np.isnan(bb[j]).any():
                e = np.linalg.norm(s[:, None] - bb[j][None], axis=-1); e[~mask] = 0.0
            else:
                e = np.zeros((14, 4))
            nb.append(e.reshape(-1))
        out[i] = np.concatenate([onehot, mask.astype(float), local.reshape(-1), D[IU], *nb])
    return out


def one_protein(args):
    pid, L, stored = args
    out = {}
    for s in ATOM_SETS[:-1]:
        names = s.split("+")
        ai = [STORED.index(a) for a in names]
        xyz = stored[:, ai].reshape(-1, 3).astype(np.float64)
        res_of = np.repeat(np.arange(L), len(ai))
        elem = np.array([element(a) for a in names] * L)
        classes = sorted({element(a) for a in names})
        out[f"ref_atom_neighborhood@{s}"] = neighbourhood(xyz, res_of, elem, L, classes)
    atoms, names = parse_full(structure_path(pid), L)
    assert len(atoms) == L, (pid, len(atoms), L)
    ca_file = np.array([a.get("CA", np.full(3, np.nan)) for a in atoms])
    assert np.nanmax(np.abs(ca_file - stored[:, 1])) < 1e-2, pid                  # the file is the labels' structure
    xyz = np.array([x for a in atoms for x in a.values()])
    res_of = np.array([i for i, a in enumerate(atoms) for _ in a])
    elem = np.array([element(n) for a in atoms for n in a])
    out["ref_atom_neighborhood@full"] = neighbourhood(xyz, res_of, elem, L, ELEMENTS)
    out["ref_allatom_local"] = allatom_local(atoms, names, L)
    return out


def cache_dir():
    return ROOT / "prereg" / "floor_cache" / hashlib.sha1(str(rps.LABELS).encode()).hexdigest()[:10]


class PreregData(AtomData):
    def residue_features(self, source):
        if source in FLOORS:
            return {p: np.load(cache_dir() / f"{source}__{p}.npy") for p in rps.PARTS}
        return super().residue_features(source)


def build(data, workers):
    ids = [x.decode() for x in data.P["id"][:]]
    coords = data.R["coords"]
    cache_dir().mkdir(parents=True, exist_ok=True)
    for p in rps.PARTS:
        if all((cache_dir() / f"{f}__{p}.npy").exists() for f in FLOORS):
            continue
        jobs = [(ids[i], int(data.L[i]), coords[data.offs[i]:data.offs[i] + data.L[i]]) for i in data.chosen[p]]
        with Pool(workers) as pool:
            res = pool.map(one_protein, jobs, chunksize=4)
        for f in FLOORS:
            a = np.concatenate([r[f] for r in res])
            assert len(a) == len(data.rows[p]) and np.isfinite(a).all(), (f, p, a.shape)
            np.save(cache_dir() / f"{f}__{p}.npy", a)
            print(f"{f:32s} {p:5s} {a.shape}", flush=True)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    data = PreregData()
    if "--build" in sys.argv:
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
        build(data, w)
    if "--check" in sys.argv:
        for f in FLOORS:
            X = data.residue_features(f)
            print(f, {p: X[p].shape for p in X}, "finite", all(np.isfinite(X[p]).all() for p in X))


if __name__ == "__main__":
    main()
