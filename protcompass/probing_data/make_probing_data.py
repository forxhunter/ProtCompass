#!/usr/bin/env python
"""Rebuild the ProtCompass IIP probing dataset from CATH v4.3.0.

One command reproduces everything under protcompass/probing_data/:

    ~/miniconda3/envs/protein_embeddings/bin/python make_probing_data.py --all

Stages
  download   official CATH files into raw/ (FTP mirror; the https host truncates large files)
  labels     parse and clean every candidate domain, write structures/<domain>.pdb, compute labels
  assemble   filters, splits, length residualisation, architecture classes, labels.h5, exclusions.csv
  diagnose   statistics and reference-row probes -> DIAGNOSTICS.md

Tools: mkdssp 3.0.0 (binary taken from the GeoBPE-6 conda env, read only), freesasa (Lee-Richards,
probe 1.4 A, ProtOr radii), numpy/scipy. See schema.md for every array and formula.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
_HOME = _os.path.expanduser("~")
import argparse
import csv
import os
import pickle
import subprocess
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW, STRUCT, WORK, SPLITS = ROOT / "raw", ROOT / "structures", ROOT / "work", ROOT / "splits"
FTP = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_3_0"
RAW_FILES = [
    "cath-classification-data/cath-domain-list-S35-v4_3_0.txt",
    "cath-classification-data/cath-domain-list-v4_3_0.txt",
    "cath-classification-data/cath-names-v4_3_0.txt",
    "cath-classification-data/README-cath-list-file-format.txt",
    "non-redundant-data-sets/cath-dataset-nonredundant-S20-v4_3_0.list",
    "non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.list",
    "non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.pdb.tgz",
]
MKDSSP = (_HOME + "/miniconda3/envs/GeoBPE-6/bin/mkdssp")
SEED = 20260919
N_WORKERS = 7

# Residue alphabet of the project (src/data/preprocessing/pdb_parser.py): alphabetical by one-letter code.
AA3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
       "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
AA_INDEX = {a: i for i, a in enumerate(AA3)}
AA1 = "ACDEFGHIKLMNPQRSTVWY"
REMAP = {"SEC": "CYS", "MSE": "MET"}          # selenium analogues, side-chain SE renamed to the sulfur name
# Theoretical maximum ASA, Tien et al. 2013 (PLoS ONE 8:e80635), in A^2.
MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104,
           "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155,
           "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
DSSP8 = {"G": 0, "H": 1, "I": 2, "B": 3, "E": 4, "T": 5, "S": 6, " ": 7, "C": 7, "P": 7}
DSSP8_TO_3 = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int8)      # G,H,I -> H(0); B,E -> E(1); T,S,C -> C(2)

MIN_LEN, MAX_LEN = 40, 600
MAX_INCOMPLETE_FRAC = 0.05       # residues without a full N, CA, C, O backbone
MAX_NONSTANDARD_FRAC = 0.02      # residues outside the 20 standard types after REMAP
PEPTIDE_BREAK = 1.6              # C(i)-N(i+1) distance above this is a chain break (A)
CONTACT_CUTOFF, CONTACT_MIN_SEP = 8.0, 4      # CA-CA < 8 A; |i-j| > 3 applies to contact order and the legacy count only
CORE_RADIUS, CORE_MIN_NEIGHBOURS = 10.0, 20   # CB within 10 A; core if >= 20 neighbours
PAIR_MIN_SEP, PAIRS_PER_PROTEIN = 6, 512
DIST_BINS, DIST_MAX = 16, 40.0
MIN_ARCH_TRAIN = 20


# --------------------------------------------------------------------------------------------- download
def stage_download():
    RAW.mkdir(parents=True, exist_ok=True)
    for rel in RAW_FILES:
        out = RAW / Path(rel).name
        if out.exists() and out.stat().st_size > 0:
            continue
        for _ in range(20):
            rc = subprocess.call(["curl", "-s", "-m", "3000", "-C", "-", "-o", str(out), f"{FTP}/{rel}"])
            if rc == 0:
                break
        else:
            sys.exit(f"download failed: {FTP}/{rel}")
    if not (RAW / "dompdb").is_dir():
        subprocess.check_call(["tar", "-xzf", str(RAW / "cath-dataset-nonredundant-S40-v4_3_0.pdb.tgz"), "-C", str(RAW)])


def read_domain_list(path):
    """CATH list file: domain, C, A, T, H, S35, S60, S95, S100, count, length, resolution."""
    rows = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        rows[p[0]] = dict(C=int(p[1]), A=int(p[2]), T=int(p[3]), H=int(p[4]), cath_length=int(p[10]), resolution=float(p[11]))
    return rows


def candidates():
    s35 = read_domain_list(RAW / "cath-domain-list-S35-v4_3_0.txt")
    have = set(os.listdir(RAW / "dompdb"))
    s20 = {l.strip() for l in open(RAW / "cath-dataset-nonredundant-S20-v4_3_0.list")}
    return s35, have, s20


# --------------------------------------------------------------------------------------------- geometry
def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-9)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)


def virtual_cb(n, ca, c):
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def parse_domain(path):
    """Returns residues in file order: list of dict(name, resseq, icode, atoms{name: (xyz, bfac)}, lines)."""
    residues, seen, key_prev = [], None, None
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name[0] in "HD" or (name[0].isdigit() and len(name) > 1 and name[1] in "HD"):
            continue
        key = (line[21], line[22:26], line[26])
        if key != key_prev:
            residues.append(dict(name=line[17:20].strip(), resseq=int(line[22:26]), icode=line[26], atoms={}))
            key_prev = key
        atoms = residues[-1]["atoms"]
        if name in atoms:                                  # alternate location: keep the first one seen
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            bfac = float(line[60:66]) if line[60:66].strip() else 0.0
        except ValueError:
            continue
        atoms[name] = (xyz, bfac)
    return residues


def write_clean_pdb(path, residues):
    serial, out = 1, []
    for k, r in enumerate(residues, start=1):
        for name, (xyz, bfac) in r["atoms"].items():
            elem = "".join(ch for ch in name if ch.isalpha())[:1]
            padded = f" {name:<3s}" if len(name) < 4 else name
            out.append(f"ATOM  {serial:5d} {padded} {r['name']:>3s} A{k:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                       f"  1.00{bfac:6.2f}          {elem:>2s}")
            serial += 1
    out.append("TER")
    out.append("END")
    Path(path).write_text("\n".join(out) + "\n")


def run_dssp(pdb_path, L):
    with tempfile.NamedTemporaryFile(suffix=".dssp", delete=False, dir=str(WORK)) as tmp:
        out = tmp.name
    try:
        rc = subprocess.run([MKDSSP, "-i", str(pdb_path), "-o", out], capture_output=True, timeout=120).returncode
        if rc != 0 or not os.path.getsize(out):
            return None
        ss = np.full(L, -1, dtype=np.int8)
        started = False
        for line in open(out):
            if line.startswith("  #  RESIDUE"):
                started = True
                continue
            if not started or len(line) < 17 or line[13] == "!":
                continue
            idx = int(line[5:10]) - 1
            if 0 <= idx < L:
                ss[idx] = DSSP8.get(line[16], 7)
        return ss
    except Exception:
        return None
    finally:
        if os.path.exists(out):
            os.unlink(out)


def run_sasa(pdb_path, L):
    import freesasa
    freesasa.setVerbosity(freesasa.silent)
    s = freesasa.Structure(str(pdb_path), options={"skip-unknown": True})
    res = freesasa.calc(s).residueAreas()
    sasa = np.full(L, np.nan, dtype=np.float32)
    for num, area in res.get("A", {}).items():
        idx = int(num.strip()) - 1
        if 0 <= idx < L:
            sasa[idx] = area.total
    return sasa


def process_domain(dom):
    """Clean one domain, write structures/<dom>.pdb and return label arrays, or an exclusion reason."""
    try:
        raw = parse_domain(RAW / "dompdb" / dom)
        if not raw:
            return dom, "parse_failed", "no ATOM records", None
        n_raw = len(raw)
        kept, n_nonstd, n_incomplete = [], 0, 0
        for r in raw:
            r["name"] = REMAP.get(r["name"], r["name"])
            if "SE" in r["atoms"] and r["name"] in ("CYS", "MET"):
                r["atoms"] = {("SG" if (k == "SE" and r["name"] == "CYS") else "SD" if k == "SE" else k): v for k, v in r["atoms"].items()}
            if r["name"] not in AA_INDEX:
                n_nonstd += 1
                continue
            if not all(a in r["atoms"] for a in ("N", "CA", "C", "O")):
                n_incomplete += 1
                continue
            kept.append(r)
        if n_nonstd / n_raw > MAX_NONSTANDARD_FRAC:
            return dom, "nonstandard_residues", f"{n_nonstd}/{n_raw}", None
        if n_incomplete / n_raw > MAX_INCOMPLETE_FRAC:
            return dom, "incomplete_backbone", f"{n_incomplete}/{n_raw}", None
        L = len(kept)
        if L < MIN_LEN or L > MAX_LEN:
            return dom, "length_out_of_range", str(L), None

        bb = np.array([[r["atoms"][a][0] for a in ("N", "CA", "C", "O")] for r in kept], dtype=np.float64)   # (L,4,3)
        has_cb = np.array(["CB" in r["atoms"] for r in kept])
        cb = virtual_cb(bb[:, 0], bb[:, 1], bb[:, 2])
        for i, r in enumerate(kept):
            if has_cb[i]:
                cb[i] = r["atoms"]["CB"][0]
        coords = np.concatenate([bb, cb[:, None]], axis=1).astype(np.float32)                              # (L,5,3)
        aa = np.array([AA_INDEX[r["name"]] for r in kept], dtype=np.int8)
        N, CA, C = bb[:, 0], bb[:, 1], bb[:, 2]

        # chain breaks and segments
        link = np.linalg.norm(C[:-1] - N[1:], axis=1) <= PEPTIDE_BREAK       # True where i and i+1 are bonded
        segment = np.concatenate([[0], np.cumsum(~link)]).astype(np.int16)

        # I1 backbone dihedrals (radians); undefined at termini and across breaks
        phi = np.full(L, np.nan)
        psi = np.full(L, np.nan)
        phi[1:] = np.where(link, dihedral(C[:-1], N[1:], CA[1:], C[1:]), np.nan)
        psi[:-1] = np.where(link, dihedral(N[:-1], CA[:-1], C[:-1], N[1:]), np.nan)

        def bins36(a):
            b = np.floor((np.degrees(a) + 180.0) / 10.0)
            return np.where(np.isnan(a), -1, np.clip(b, 0, 35)).astype(np.int8)

        # I2 curvature: angle between consecutive CA-CA vectors, defined where both links exist
        curv = np.full(L, np.nan)
        v1, v2 = CA[1:-1] - CA[:-2], CA[2:] - CA[1:-1]
        cosang = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
        curv[1:-1] = np.where(link[:-1] & link[1:], np.arccos(np.clip(cosang, -1, 1)), np.nan)

        pdb_path = STRUCT / f"{dom}.pdb"
        write_clean_pdb(pdb_path, kept)

        ss8 = run_dssp(pdb_path, L)
        if ss8 is None or (ss8 < 0).mean() > 0.05:
            pdb_path.unlink(missing_ok=True)
            return dom, "dssp_failed", "", None
        ss3 = np.where(ss8 >= 0, DSSP8_TO_3[np.clip(ss8, 0, 7)], -1).astype(np.int8)

        sasa = run_sasa(pdb_path, L)
        max_asa = np.array([MAX_ASA[r["name"]] for r in kept], dtype=np.float32)
        rsa = np.clip(sasa / max_asa, 0.0, 1.0)
        rsa_bin = np.where(np.isnan(rsa), -1, np.digitize(rsa, [0.2, 0.5])).astype(np.int8)

        # I5 flexibility: CA B-factor, z-scored within the chain; masked when constant or zero
        b_raw = np.array([r["atoms"]["CA"][1] for r in kept], dtype=np.float32)
        b_ok = b_raw.std() > 1e-3 and np.abs(b_raw).max() > 1e-6
        b_z = ((b_raw - b_raw.mean()) / b_raw.std()).astype(np.float32) if b_ok else np.full(L, np.nan, dtype=np.float32)

        # I6 packing
        idx = np.arange(L)
        sep = np.abs(idx[:, None] - idx[None, :])
        d_ca = np.linalg.norm(CA[:, None] - CA[None, :], axis=-1)
        contact = (d_ca < CONTACT_CUTOFF) & (sep >= CONTACT_MIN_SEP)                   # used for contact order
        contact_count_sep4 = contact.sum(1).astype(np.int16)                           # legacy count, |i-j| > 3
        contact_count = ((d_ca < CONTACT_CUTOFF).sum(1) - 1).astype(np.int16)          # all residues within 8 A, self excluded
        contact_bin = np.digitize(contact_count, [6, 11, 16]).astype(np.int8)          # 0-5, 6-10, 11-15, 16+
        d_cb = np.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
        cb_nb = ((d_cb < CORE_RADIUS).sum(1) - 1).astype(np.int16)
        core = (cb_nb >= CORE_MIN_NEIGHBOURS).astype(np.int8)

        # I7 sampled residue pairs, identical for every encoder
        iu, ju = np.where(np.triu(sep >= PAIR_MIN_SEP, k=1))
        rng = np.random.default_rng(zlib.crc32(dom.encode()))
        pick = rng.choice(len(iu), size=min(PAIRS_PER_PROTEIN, len(iu)), replace=False)
        pick.sort()
        pi, pj = iu[pick].astype(np.int16), ju[pick].astype(np.int16)
        pd_ = d_ca[pi, pj].astype(np.float32)
        pbin = np.clip(np.floor(pd_ / (DIST_MAX / DIST_BINS)), 0, DIST_BINS - 1).astype(np.int8)

        # I8 global geometry (CA atoms)
        n_contacts = np.triu(contact, k=1).sum()
        rco = float((sep * np.triu(contact, k=1)).sum() / (max(n_contacts, 1) * L))
        centred = CA - CA.mean(0)
        rg = float(np.sqrt((centred ** 2).sum(1).mean()))
        lam = np.sort(np.linalg.eigvalsh(centred.T @ centred / L))[::-1]
        asph = float(1.5 * (lam ** 2).sum() / lam.sum() ** 2 - 0.5)               # relative shape anisotropy
        asph_legacy = float((lam[0] - lam[2]) ** 2 / lam.sum() ** 2)              # formula of the original code

        rec = dict(
            dom=dom, L=L, n_raw=n_raw, n_nonstd=n_nonstd, n_incomplete=n_incomplete, n_segments=int(segment[-1]) + 1,
            n_virtual_cb=int((~has_cb).sum()), bfactor_ok=bool(b_ok),
            coords=coords, aa=aa, resseq=np.array([r["resseq"] for r in kept], dtype=np.int32),
            icode=np.array([r["icode"] for r in kept], dtype="S1"), segment=segment, cb_virtual=~has_cb,
            phi=phi.astype(np.float32), psi=psi.astype(np.float32), phi_bin=bins36(phi), psi_bin=bins36(psi),
            curvature=curv.astype(np.float32), dssp8=ss8, dssp3=ss3, sasa=sasa, rsa=rsa.astype(np.float32), rsa_bin=rsa_bin,
            bfactor_raw=b_raw, bfactor_z=b_z, contact_count=contact_count, contact_count_sep4=contact_count_sep4,
            contact_density_bin=contact_bin,
            cb_neighbours=cb_nb, core=core, pair_i=pi, pair_j=pj, pair_dist=pd_, pair_bin=pbin,
            contact_order=rco, radius_of_gyration=rg, asphericity=asph, asphericity_legacy=asph_legacy,
        )
        return dom, None, "", rec
    except Exception as e:                                                           # noqa: BLE001
        return dom, "exception", f"{type(e).__name__}: {e}"[:200], None


def stage_labels(limit=None):
    STRUCT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    s35, have, _ = candidates()
    doms = sorted(d for d in s35 if d in have)
    if limit:
        doms = doms[:limit]
    shard_size, done = 1000, 0
    for s in range(0, len(doms), shard_size):
        shard = WORK / f"shard_{s // shard_size:03d}.pkl"
        if shard.exists():
            done += 1
            continue
        with Pool(N_WORKERS) as pool:
            out = pool.map(process_domain, doms[s:s + shard_size], chunksize=8)
        with open(shard, "wb") as f:
            pickle.dump(out, f, protocol=4)
        ok = sum(1 for o in out if o[1] is None)
        print(f"shard {s // shard_size:03d}: {ok}/{len(out)} kept", flush=True)
    print(f"labels stage complete ({done} shards were already present)")


# --------------------------------------------------------------------------------------------- assemble
def group_split(meta, level, seed):
    """Group-disjoint 80/10/10 split, stratified by CATH class. level: 4 = superfamily, 3 = topology."""
    rng = np.random.default_rng(seed)
    assign = {}
    by_class = defaultdict(lambda: defaultdict(list))
    for dom, m in meta.items():
        by_class[m["C"]][tuple(m[k] for k in "CATH"[:level])].append(dom)
    for c, groups in sorted(by_class.items()):
        keys = list(groups)
        order = sorted(range(len(keys)), key=lambda i: (-len(groups[keys[i]]), rng.random()))
        total = sum(len(v) for v in groups.values())
        target = {"train": 0.8 * total, "val": 0.1 * total, "test": 0.1 * total}
        have = {"train": 0, "val": 0, "test": 0}
        for i in order:
            g = groups[keys[i]]
            split = max(target, key=lambda s: (target[s] - have[s]) / target[s] if target[s] else -1)
            have[split] += len(g)
            for dom in g:
                assign[dom] = split
    return assign


def residualise(y, logL, train_mask):
    X = np.stack([np.ones_like(logL), logL, logL ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(X[train_mask], y[train_mask], rcond=None)
    return (y - X @ coef).astype(np.float32), coef


def stage_assemble():
    import h5py
    s35, have, s20 = candidates()
    records, excl = [], []
    for dom in sorted(s35):
        if dom not in have:
            excl.append((dom, "not_in_s40_tarball", "S35 representative without a file in the official S40 domain PDB set"))
    for shard in sorted(WORK.glob("shard_*.pkl")):
        for dom, reason, detail, rec in pickle.load(open(shard, "rb")):
            if reason:
                excl.append((dom, reason, detail))
            else:
                records.append(rec)
    records.sort(key=lambda r: r["dom"])
    meta = {r["dom"]: s35[r["dom"]] for r in records}
    n = len(records)
    L = np.array([r["L"] for r in records], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
    logL = np.log(L.astype(np.float64))

    splits = {"superfamily": group_split(meta, 4, SEED), "topology": group_split(meta, 3, SEED + 1)}
    SPLITS.mkdir(exist_ok=True)
    for name, assign in splits.items():
        for part in ("train", "val", "test"):
            (SPLITS / f"{name}_{part}.txt").write_text("\n".join(d for d in sorted(assign) if assign[d] == part) + "\n")
    (SPLITS / "subset_in_s20.txt").write_text("\n".join(r["dom"] for r in records if r["dom"] in s20) + "\n")

    with open(ROOT / "exclusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "reason", "detail"])
        w.writerows(excl)

    out = ROOT / "labels.h5"
    if out.exists():
        out.unlink()
    with h5py.File(out, "w") as h:
        h.attrs.update(dict(cath_release="v4_3_0", domain_set="S35 representatives with a file in the official S40 non-redundant domain PDB set",
                            seed=SEED, aa_alphabet=AA1, dssp8_classes="G,H,I,B,E,T,S,C", dssp3_classes="H,E,C",
                            rsa_bin_edges="0.2,0.5", contact_density_bins="0-5,6-10,11-15,16+", distance_bins="16 bins of 2.5 A over 0-40 A, last bin open",
                            core_rule=f"CB neighbours within {CORE_RADIUS} A >= {CORE_MIN_NEIGHBOURS}", mkdssp="3.0.0", sasa="freesasa Lee-Richards probe 1.4 A ProtOr radii, isolated domain"))
        gp, gr, gq = h.create_group("protein"), h.create_group("residue"), h.create_group("pair")
        ids = np.array([r["dom"] for r in records], dtype="S7")
        gp.create_dataset("id", data=ids)
        gp.create_dataset("length", data=L)
        gp.create_dataset("offset", data=offsets[:-1])
        for k in "CATH":
            gp.create_dataset(f"cath_{k}", data=np.array([meta[d.decode()][k] for d in ids], dtype=np.int32))
        gp.create_dataset("resolution", data=np.array([meta[d.decode()]["resolution"] for d in ids], dtype=np.float32))
        gp.create_dataset("in_s20", data=np.array([d.decode() in s20 for d in ids]))
        for k in ("n_segments", "n_virtual_cb", "n_incomplete", "n_nonstd"):
            gp.create_dataset(k, data=np.array([r[k] for r in records], dtype=np.int32))
        gp.create_dataset("bfactor_ok", data=np.array([r["bfactor_ok"] for r in records]))
        code = {"train": 0, "val": 1, "test": 2}
        for name, assign in splits.items():
            gp.create_dataset(f"split_{name}", data=np.array([code[assign[d.decode()]] for d in ids], dtype=np.int8))
            gp[f"split_{name}"].attrs["codes"] = "0=train,1=val,2=test"

        for k in ("contact_order", "radius_of_gyration", "asphericity", "asphericity_legacy"):
            y = np.array([r[k] for r in records], dtype=np.float64)
            gp.create_dataset(k, data=y.astype(np.float32))
            if k == "asphericity_legacy":
                continue
            for name in splits:
                res, coef = residualise(y, logL, gp[f"split_{name}"][:] == 0)
                ds = gp.create_dataset(f"{k}_len_resid_{name}", data=res)
                ds.attrs["fit"] = "y - (a + b*log(L) + c*log(L)^2), OLS on the training split"
                ds.attrs["coef_a_b_c"] = coef

        arch = np.array([(meta[d.decode()]["C"], meta[d.decode()]["A"]) for d in ids])
        for name in splits:
            tr = gp[f"split_{name}"][:] == 0
            counts = Counter(map(tuple, arch[tr]))
            kept = sorted(a for a, c in counts.items() if c >= MIN_ARCH_TRAIN)
            index = {a: i for i, a in enumerate(kept)}
            lab = np.array([index.get(tuple(a), -1) for a in arch], dtype=np.int16)
            ds = gp.create_dataset(f"architecture_label_{name}", data=lab)
            ds.attrs["classes"] = ",".join(f"{c}.{a}" for c, a in kept)
            ds.attrs["dropped"] = ",".join(f"{c}.{a}" for c, a in sorted(set(map(tuple, arch)) - set(kept)))

        def cat(key, dtype=None):
            a = np.concatenate([r[key] for r in records])
            return a.astype(dtype) if dtype else a

        gr.create_dataset("protein_index", data=np.repeat(np.arange(n, dtype=np.int32), L), compression="gzip")
        gr.create_dataset("position", data=np.concatenate([np.arange(l, dtype=np.int16) for l in L]), compression="gzip")
        for key in ("coords", "aa", "resseq", "icode", "segment", "cb_virtual", "phi", "psi", "phi_bin", "psi_bin", "curvature",
                    "dssp8", "dssp3", "sasa", "rsa", "rsa_bin", "bfactor_raw", "bfactor_z", "contact_count", "contact_count_sep4",
                    "contact_density_bin", "cb_neighbours", "core"):
            gr.create_dataset(key, data=cat(key), compression="gzip")
        for ang in ("phi", "psi"):
            a = gr[ang][:]
            gr.create_dataset(f"{ang}_sin", data=np.sin(a), compression="gzip")
            gr.create_dataset(f"{ang}_cos", data=np.cos(a), compression="gzip")
        gr.create_dataset("sequence_recovery", data=cat("aa"), compression="gzip")

        n_pairs = np.array([len(r["pair_i"]) for r in records], dtype=np.int64)
        gq.create_dataset("protein_index", data=np.repeat(np.arange(n, dtype=np.int32), n_pairs), compression="gzip")
        gq.create_dataset("offset", data=np.concatenate([[0], np.cumsum(n_pairs)])[:-1])
        gq.create_dataset("count", data=n_pairs.astype(np.int32))
        for src, dst in (("pair_i", "i"), ("pair_j", "j"), ("pair_dist", "distance"), ("pair_bin", "distance_bin")):
            gq.create_dataset(dst, data=cat(src), compression="gzip")

    kept_ids = {r["dom"] for r in records}
    for f in STRUCT.glob("*.pdb"):
        if f.stem not in kept_ids:
            f.unlink()
    print(f"assembled {n} domains, {int(L.sum())} residues, {int(n_pairs.sum())} pairs; {len(excl)} exclusions")
    for name, assign in splits.items():
        print(name, dict(Counter(assign.values())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stage", choices=["download", "labels", "assemble", "diagnose"])
    ap.add_argument("--limit", type=int, default=None, help="process only the first N candidate domains (pilot)")
    a = ap.parse_args()
    stages = ["download", "labels", "assemble", "diagnose"] if a.all else [a.stage]
    for st in stages:
        if st == "download":
            stage_download()
        elif st == "labels":
            stage_labels(a.limit)
        elif st == "assemble":
            stage_assemble()
        elif st == "diagnose":
            from diagnose_probing_data import main as diagnose
            diagnose()
