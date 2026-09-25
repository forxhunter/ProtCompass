"""Protein-level concepts for the erasure experiment, computed once per task from the structures.

The concepts are the protein-level summaries of the properties the probing suite measures, plus composition and size.
They are properties of the protein, so the same values are used for every encoder; a difference between encoders is a
difference in how their representation carries the property, not a difference in the property.

Cached under protcompass/erasure/concepts/<task>.npz, keyed by manifest id.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
CACHE = ROOT / "erasure" / "concepts"
AA3 = {"ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
       "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14, "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19}
# name -> (columns in the feature block, categorical)
BLOCKS = {
    "secondary structure content": (slice(0, 3), False),
    "burial and packing": (slice(3, 5), False),
    "global shape": (slice(5, 8), False),
    "contact order": (slice(8, 9), False),
    "size": (slice(9, 10), False),
    "amino-acid composition": (slice(10, 30), False),
    "hydrophobic fraction": (slice(30, 31), False),
    "charge": (slice(31, 33), False),
}

# `hydrophobic fraction` is the sum of eight entries of the composition block and `charge` the sums of three and of
# two, so both are exact linear functions of it: regressing either on the other seven blocks leaves only float32
# rounding error. A conditional erasure of one of them is therefore fitted to that rounding error, which an earlier
# version of this experiment reported as a demand. Conditional analysis uses INDEPENDENT_BLOCKS; the two dependent
# ones are only meaningful as contrasts inside composition, conditioned on everything except composition.
DEPENDS_ON = {"hydrophobic fraction": "amino-acid composition", "charge": "amino-acid composition"}
INDEPENDENT_BLOCKS = [b for b in BLOCKS if b not in DEPENDS_ON]


def one_protein(path, seq):
    """Backbone-derived summaries. Secondary-structure content is approximated from CA geometry, not DSSP, so that
    this file has no dependency on an external binary; the approximation is stated where the results are reported."""
    n_feat = 33
    try:
        xyz, names = [], []
        cur, atoms = None, {}
        for line in open(path):
            if not line.startswith("ATOM"):
                continue
            key = (line[21], line[22:27])
            if key != cur:
                if cur is not None and "CA" in atoms:
                    xyz.append(atoms["CA"]); names.append(res)
                cur, atoms = key, {}
                res = line[17:20].strip()
            atoms[line[12:16].strip()] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if cur is not None and "CA" in atoms:
            xyz.append(atoms["CA"]); names.append(res)
        ca = np.asarray(xyz, dtype=np.float64)
        n = len(ca)
        if n < 8:
            return np.full(n_feat, np.nan)
    except Exception:
        return np.full(n_feat, np.nan)

    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    # helix and strand content from the CA(i), CA(i+3) and CA(i+4) distances, a standard geometric approximation
    d3 = np.linalg.norm(ca[3:] - ca[:-3], axis=-1) if n > 3 else np.array([10.0])
    d4 = np.linalg.norm(ca[4:] - ca[:-4], axis=-1) if n > 4 else np.array([12.0])
    helix = float(((d3 < 6.5) & (d3 > 4.5)).mean())
    strand = float((d3 > 9.0).mean())
    coil = max(0.0, 1.0 - helix - strand)
    buried = float((np.isfinite(d) & (d < 10.0)).sum(1).mean())
    frac_buried = float(((np.isfinite(d) & (d < 10.0)).sum(1) >= 14).mean())
    c = ca - ca.mean(0)
    lam = np.sort(np.linalg.eigvalsh(c.T @ c / n))[::-1]
    rg = float(np.sqrt(lam.sum()))
    asph = float(1.5 * (lam ** 2).sum() / lam.sum() ** 2 - 0.5)
    iu = np.triu_indices(n, 1)
    contacts = d[iu] < 8.0
    sep = np.abs(iu[0] - iu[1])
    co = float(sep[contacts].mean() / n) if contacts.any() else 0.0
    comp = np.zeros(20)
    for r in names:
        if r in AA3:
            comp[AA3[r]] += 1.0
    comp /= max(comp.sum(), 1.0)
    hyd = float(comp[[0, 4, 9, 10, 12, 13, 17, 19]].sum())
    pos = float(comp[[1, 8, 11]].sum())
    neg = float(comp[[3, 6]].sum())
    return np.array([helix, strand, coil, buried, frac_buried, rg, asph, float(lam[0] / max(lam.sum(), 1e-9)),
                     co, np.log(n), *comp, hyd, pos, neg])


def build(task):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{task}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return dict(zip(z["ids"], z["feat"]))
    man = pd.read_csv(ROOT / "extract" / "manifest" / f"{task}.csv")
    man = man[man.stage1 == 1].reset_index(drop=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(12) as pool:
        feat = list(pool.map(lambda r: one_protein(r[0], r[1]), zip(man.structure_path, man.sequence), chunksize=64))
    feat = np.asarray(feat, dtype=np.float32)
    np.savez(f, ids=man.id.to_numpy(), feat=feat)
    print(f"[{task}] concepts for {len(feat)} proteins, {int(np.isnan(feat).any(1).sum())} unusable")
    return dict(zip(man.id.to_numpy(), feat))


_TASK = {"cache": None}


def protein_descriptors(ids_by_split, task=None):
    """Return {concept name: (Ztr, Zva, Zte, categorical)} aligned to the given id lists."""
    if task is None:
        task = _TASK["task"]
    table = _TASK["cache"] or build(task)
    _TASK["cache"] = table
    out = {}
    blocks = {s: np.stack([table.get(i, np.full(33, np.nan)) for i in ids]) for s, ids in ids_by_split.items()}
    for name, (sl, cat) in BLOCKS.items():
        Z = {s: np.atleast_2d(blocks[s][:, sl]) for s in blocks}
        if any(np.isnan(Z[s]).all() for s in Z):
            continue
        Z = {s: np.nan_to_num(Z[s], nan=float(np.nanmean(Z["train"]))) for s in Z}
        out[name] = (Z["train"], Z["valid"], Z["test"], cat)
    return out


def set_task(task):
    _TASK["task"] = task
    _TASK["cache"] = None
