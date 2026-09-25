"""Determinacy map and its executable checks for the pre-registered labels, written before any new label is scored.

For every (encoder, label) the status is one of
  determined        the label is reconstructed exactly from the atoms the encoder is measured to read
                    (agreement >= 0.99999 for a class label, maximum error <= 1e-4 for the angle)
  undetermined      an explicit construction changes the label while leaving every atom the encoder reads unchanged
  not_established   neither holds; such cells are reported descriptively and do not enter H1-H4

Effective inputs come from the audits: residue identity and coordinates (released input audit), the atom set among
N, CA, C, O, CB (extract/encoder_atom_audit.csv) and side-chain atoms beyond CB (extract/encoder_sidechain_audit.csv).
Checks run on 200 randomly drawn released domains.

  python prereg_determinacy.py
Outputs: prereg/determinacy_checks.csv (one row per check) and prereg/determinacy_map.csv (one row per cell).
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from panel import panel                                                   # noqa: E402
from prereg_labels import dihedral, parse                                  # noqa: E402
from prereg_chi2 import CHI2                                               # noqa: E402

LABELS = ["hbond_count", "chi1_class", "chi2_class", "ca_dihedral", "sc_contacts"]
GAMMA = {"CG", "OG", "OG1", "SG", "CG1"}


def hbond(N, CA, C, O):
    """Backbone H-bond count as in prereg_labels.py, from N, CA, C and O only."""
    L = len(N)
    H = N.copy()
    d = C[:-1] - O[:-1]
    H[1:] = N[1:] + d / np.linalg.norm(d, axis=1, keepdims=True)
    rON = np.linalg.norm(O[None] - N[:, None], axis=-1); rCH = np.linalg.norm(C[None] - H[:, None], axis=-1)
    rOH = np.linalg.norm(O[None] - H[:, None], axis=-1); rCN = np.linalg.norm(C[None] - N[:, None], axis=-1)
    with np.errstate(divide="ignore"):
        E = 0.084 * 332.0 * (1 / rON + 1 / rCH - 1 / rOH - 1 / rCN)
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None])
    bond = (E < -0.5) & (sep >= 2)
    bond[0, :] = False
    return np.minimum(bond.sum(1) + bond.sum(0), 2)


def ideal_O(N, CA, C):
    """Carbonyl O placed from N, CA, C alone: in the peptide plane, 1.231 A from C, bisecting CA-C-N(i+1)."""
    O = np.full_like(C, np.nan)
    u = CA[:-1] - C[:-1]; u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = N[1:] - C[:-1]; v /= np.linalg.norm(v, axis=1, keepdims=True)
    w = -(u + v); w /= np.linalg.norm(w, axis=1, keepdims=True)
    O[:-1] = C[:-1] + 1.231 * w
    return O


def rotate(points, a, b, deg):
    """Rotate points about the axis through a and b by deg degrees (Rodrigues)."""
    k = (b - a) / np.linalg.norm(b - a); t = np.radians(deg); p = points - a
    return a + p * np.cos(t) + np.cross(k, p) * np.sin(t) + np.outer(p @ k, k) * (1 - np.cos(t))


def cls(t):
    t = np.degrees(t) % 360
    return 0 if t < 120 else (1 if t < 240 else 2)


def checks(n_domains=200):
    f, g = h5py.File(ROOT / "probing_data" / "labels.h5", "r"), h5py.File(ROOT / "prereg" / "labels_released.h5", "r")
    ids = [x.decode() for x in f["protein/id"][:]]
    L, off = f["protein/length"][:], f["protein/offset"][:]
    pick = np.random.default_rng(0).choice(len(ids), n_domains, replace=False)
    agree = {"hbond@N+CA+C+O": [], "hbond@N+CA+C(ideal O)": [], "ca_dihedral@CA": [], "chi1@full": [], "chi2@full": []}
    flips = {"hbond@CA peptide-plane rotation": [], "chi1@N+CA+C+O+CB side-chain rotation": [],
             "chi2@N+CA+C+O+CB side-chain rotation": [], "ca_dihedral@sequence crankshaft": []}
    for i in pick:
        s = slice(off[i], off[i] + L[i])
        X = f["residue/coords"][s].astype(np.float64); N, CA, C, O = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        hb = g["residue/hbond_count"][s]
        agree["hbond@N+CA+C+O"].append((hbond(N, CA, C, O) == hb).mean())
        Oi = ideal_O(N, CA, C); ok = np.isfinite(Oi[:, 0]); Oi[~ok] = O[~ok]
        agree["hbond@N+CA+C(ideal O)"].append((hbond(N, CA, C, Oi) == hb).mean())
        cad = g["residue/ca_dihedral"][s]; m = np.isfinite(cad[:, 0])
        rec = np.full(L[i], np.nan); rec[1:L[i] - 2] = dihedral(CA[:-3], CA[1:-2], CA[2:-1], CA[3:])
        agree["ca_dihedral@CA"].append(float(np.max(np.abs(np.stack([np.sin(rec), np.cos(rec)], 1)[m] - cad[m]))))
        # construction: rotate peptide plane k (C_k, O_k, N_k+1) by 180 degrees about CA_k - CA_k+1; CA unchanged
        k = int(np.argmax(hb[1:-2])) + 1
        C2, O2, N2 = C.copy(), O.copy(), N.copy()
        C2[k], O2[k], N2[k + 1] = rotate(np.stack([C[k], O[k], N[k + 1]]), CA[k], CA[k + 1], 180.0)
        flips["hbond@CA peptide-plane rotation"].append(bool((hbond(N2, CA, C2, O2) != hb).any()))
        # construction: a crankshaft rotation of residues k+1.. about CA_k - CA_k+1 changes CA dihedrals; the sequence
        # is unchanged, so a sequence-only input does not determine the angle
        CA3 = CA.copy(); CA3[k + 2:] = rotate(CA[k + 2:], CA[k], CA[k + 1], 60.0)
        rec3 = dihedral(CA3[:-3], CA3[1:-2], CA3[2:-1], CA3[3:])
        flips["ca_dihedral@sequence crankshaft"].append(bool(np.nanmax(np.abs(rec3 - rec[1:L[i] - 2])) > 0.1))
        # side chains from the domain file
        atoms = parse(ROOT / "probing_data" / "structures" / f"{ids[i]}.pdb")
        names = {}
        for line in open(ROOT / "probing_data" / "structures" / f"{ids[i]}.pdb"):
            if line.startswith("ATOM"):
                names[int(line[22:26])] = line[17:20]
        keys = sorted(atoms)[:L[i]]
        c1, c2 = g["residue/chi1_class"][s], g["residue/chi2_class"][s]
        r1, r2, f1, f2 = [], [], [], []
        for r, key in enumerate(keys):
            a = atoms[key]
            gam = sorted(x for x in GAMMA if x in a)
            if gam and all(x in a for x in ("N", "CA", "CB")):
                r1.append(cls(dihedral(a["N"], a["CA"], a["CB"], a[gam[0]])) == c1[r])
                # rotate every atom beyond CB about CA-CB by 120 degrees: the five stored atoms do not move
                g2 = rotate(a[gam[0]][None], a["CA"], a["CB"], 120.0)[0]
                f1.append(cls(dihedral(a["N"], a["CA"], a["CB"], g2)) != c1[r])
            ch = CHI2.get(names.get(key, ""))
            if ch and all(x in a for x in ("CA", "CB") + ch):
                r2.append(cls(dihedral(a["CA"], a["CB"], a[ch[0]], a[ch[1]])) == c2[r])
                d2 = rotate(a[ch[1]][None], a["CB"], a[ch[0]], 120.0)[0]
                f2.append(cls(dihedral(a["CA"], a["CB"], a[ch[0]], d2)) != c2[r])
        agree["chi1@full"].append(np.mean(r1) if r1 else np.nan); agree["chi2@full"].append(np.mean(r2) if r2 else np.nan)
        flips["chi1@N+CA+C+O+CB side-chain rotation"].append(bool(np.all(f1)) if f1 else np.nan)
        flips["chi2@N+CA+C+O+CB side-chain rotation"].append(bool(np.all(f2)) if f2 else np.nan)
    rows = []
    for k, v in agree.items():
        v = np.array(v, float)
        stat = float(np.nanmax(v)) if k == "ca_dihedral@CA" else float(np.nanmin(v))
        passed = stat <= 1e-4 if k == "ca_dihedral@CA" else stat >= 0.99999
        rows.append(dict(check=k, kind="reconstruction", statistic="max abs error" if k == "ca_dihedral@CA" else
                         "min per-domain agreement", value=stat, mean=float(np.nanmean(v)), passes=passed, n=len(v)))
    for k, v in flips.items():
        v = np.array([x for x in v if x == x], bool)
        rows.append(dict(check=k, kind="construction", statistic="share of domains where the label changes",
                         value=float(v.mean()), mean=float(v.mean()), passes=bool(v.any()), n=len(v)))
    return pd.DataFrame(rows)


def status_map(C):
    ok = C.set_index("check").passes
    atoms = pd.read_csv(ROOT / "extract" / "encoder_atom_audit.csv").set_index("encoder").atom_set
    sc = pd.read_csv(ROOT / "extract" / "encoder_sidechain_audit.csv").set_index("encoder")
    rows = []
    for e in panel():
        s = atoms.get(e, "--")
        s = "sequence" if s in ("--", "none") or pd.isna(s) else s
        if e in sc.index and bool(sc.loc[e, "reads_sidechain"]):
            s = "full"
        A = set() if s in ("sequence", "full") else set(s.split("+"))
        for lab in LABELS:
            if lab == "hbond_count":
                if s == "full" or A >= {"N", "CA", "C", "O"}:
                    st = "determined" if ok["hbond@N+CA+C+O"] else "not_established"
                elif s == "sequence" or A == {"CA"}:
                    st = "undetermined" if ok["hbond@CA peptide-plane rotation"] else "not_established"
                elif A == {"N", "CA", "C"}:
                    st = "determined" if ok["hbond@N+CA+C(ideal O)"] else "not_established"
                else:
                    st = "not_established"
            elif lab in ("chi1_class", "chi2_class"):
                key = "chi1" if lab == "chi1_class" else "chi2"
                if s == "full":
                    st = "determined" if ok[f"{key}@full"] else "not_established"
                else:
                    st = "undetermined" if ok[f"{key}@N+CA+C+O+CB side-chain rotation"] else "not_established"
            elif lab == "ca_dihedral":
                st = ("determined" if ok["ca_dihedral@CA"] else "not_established") if s != "sequence" else \
                     ("undetermined" if ok["ca_dihedral@sequence crankshaft"] else "not_established")
            else:                                                                   # sc_contacts: descriptive
                st = "determined" if s == "full" else "not_established"
            rows.append(dict(encoder=e, input=s, label=lab, status=st))
    return pd.DataFrame(rows)


def main():
    C = checks()
    C.to_csv(ROOT / "prereg" / "determinacy_checks.csv", index=False)
    print(C.to_string(index=False))
    M = status_map(C)
    M.to_csv(ROOT / "prereg" / "determinacy_map.csv", index=False)
    print(M.pivot_table(index="input", columns="label", values="status", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
