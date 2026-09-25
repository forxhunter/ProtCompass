"""Two stronger non-learned floors for fold architecture: structural alignment and the distance-matrix spectrum.

Fold architecture is the one probe where an encoder beats every floor we had built, so it is the result most in
need of a floor that was built to win. The floors in the main table are generic descriptors of the input pushed
through the same probe. The standard non-learned way to assign a fold, however, is not a probe on a descriptor: it
is structural alignment against the training set. A reviewer pointed out that omitting it leaves the paper's one
surviving positive result protected by the weakest of the available baselines.

Two floors are added here, both non-learned in the same sense as the others: nothing is fitted on the labels.

  alignment 1-NN   Foldseek searches each test domain against the training domains of the same split and the
                   architecture of the best hit is the prediction. Structures are written from the coordinates in
                   labels.h5, so the floor sees exactly the atoms the encoders were given.
  spectral probe   the sorted eigenvalue spectrum of the CA distance matrix and of the contact map at three
                   cutoffs, length-normalised and padded to a fixed width, pushed through the identical probe,
                   split, readout and seeds as every other floor.

  python architecture_alignment_floor.py --class-set common11
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R
from architecture_all_floors import SHARED_11

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "probing_results"
SCRATCH = Path(os.environ.get("ALIGN_SCRATCH",
                              (PROTCOMPASS_ROOT + "/protcompass/cache/scratch/align")))
def _foldseek():
    """The repository ships several builds; the one under external_repos is for another architecture and fails
    with an exec-format error, so the candidates are tried in order and the first one that runs is used."""
    import subprocess as sp
    cands = [Path(os.environ["FOLDSEEK"])] if os.environ.get("FOLDSEEK") else []
    cands += [SCRATCH.parent / "foldseek" / "bin" / "foldseek",
              Path((PROTCOMPASS_ROOT + "/protcompass/external_repos/foldseek/foldseek/bin/foldseek"))]
    for c in cands:
        try:
            if c.exists() and sp.run([str(c), "version"], capture_output=True).returncode == 0:
                return c
        except OSError:
            continue
    raise SystemExit("no runnable foldseek binary found; set FOLDSEEK")
ATOMS = ["N", "CA", "C", "O", "CB"]
AA3 = {0: "ALA", 1: "CYS", 2: "ASP", 3: "GLU", 4: "PHE", 5: "GLY", 6: "HIS", 7: "ILE", 8: "LYS", 9: "LEU",
       10: "MET", 11: "ASN", 12: "PRO", 13: "GLN", 14: "ARG", 15: "SER", 16: "THR", 17: "VAL", 18: "TRP", 19: "TYR"}


def write_pdb(path, coords, aa):
    """A backbone-plus-CB PDB of one domain, from the coordinates the probes and the encoders both received."""
    lines, n = [], 0
    for i, (c, a) in enumerate(zip(coords, aa), start=1):
        res = AA3.get(int(a), "GLY")
        for k, name in enumerate(ATOMS):
            x, y, z = c[k]
            if not np.isfinite([x, y, z]).all():
                continue
            n += 1
            # the atom name occupies columns 13-16 with the element symbol right-justified in 13-14, so a
            # one-letter element starts at column 14. Writing it at column 13 produces a file every text editor
            # shows as correct and foldseek's parser rejects: the first run ignored all 14,250 structures.
            lines.append(f"ATOM  {n:5d} {' ' + name:<4s} {res:>3s} A{i:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>2s}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def spectral_features(h, offs, L, idx, n_ev=24, ca_all=None):
    """Rotation-invariant global shape: the spectrum of the CA distance matrix and of three contact maps.

    `ca_all` is every CA in the file, read once. Reading each domain's slice from the HDF5 file inside the loop
    costs more than the eigendecompositions do."""
    if ca_all is None:
        ca_all = h["residue/coords"][:, 1].astype(np.float64)
    out = []
    for i in idx:
        r = slice(int(offs[i]), int(offs[i]) + int(L[i]))
        ca = ca_all[r]
        n = len(ca)
        d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        blocks = [np.sort(np.linalg.eigvalsh(d))[::-1][:n_ev] / max(n, 1)]
        for cut in (8.0, 10.0, 12.0):
            A = (d < cut).astype(np.float64)
            np.fill_diagonal(A, 0.0)
            w = np.sort(np.linalg.eigvalsh(A))
            blocks += [w[::-1][:n_ev] / max(n, 1), w[:n_ev // 2] / max(n, 1)]
        v = np.concatenate([np.pad(b, (0, max(0, (n_ev if len(b) >= n_ev // 2 + 1 else n_ev // 2) - len(b))))
                            for b in blocks])
        out.append(np.concatenate([v, [np.log(n)]]))
    w = max(len(v) for v in out)
    return np.stack([np.pad(v, (0, w - len(v))) for v in out]).astype(np.float32)


def labels(h, split, class_set):
    arch = h[f"protein/architecture_label_{split}"][:].astype(np.int64)
    s = h[f"protein/split_{split}"][:]
    idx = {p: np.where(s == c)[0] for p, c in R.PARTS.items()}
    n_all = int(arch.max()) + 1
    if class_set == "common11":
        vocab = h[f"protein/architecture_label_{split}"].attrs["classes"].split(",")
        pos = {c: i for i, c in enumerate(vocab)}
        keep = np.array([pos[c] for c in SHARED_11], dtype=np.int64)
        remap = -np.ones(n_all, dtype=np.int64)
        remap[keep] = np.arange(len(SHARED_11))
        y = np.where(arch >= 0, remap[np.clip(arch, 0, None)], -1)
        n_out = len(SHARED_11)
    else:
        y, n_out = arch, n_all
    sel = {p: idx[p][y[idx[p]] >= 0] for p in R.PARTS}
    return y, n_out, sel


def alignment_1nn(h, split, y, sel, tag):
    ids = h["protein/id"][:]
    offs, L = h["protein/offset"][:], h["protein/length"][:]
    base = SCRATCH / f"{split}_{tag}"
    if base.exists():
        shutil.rmtree(base)
    for part in ("train", "test"):
        (base / part).mkdir(parents=True, exist_ok=True)
        for i in sel[part]:
            r = slice(int(offs[i]), int(offs[i]) + int(L[i]))
            write_pdb(base / part / f"{ids[i].decode()}.pdb",
                      h["residue/coords"][r].astype(np.float64), h["residue/aa"][r])
        print(f"  wrote {len(sel[part])} {part} structures", flush=True)
    aln = base / "aln.tsv"
    cmd = [str(_foldseek()), "easy-search", str(base / "test"), str(base / "train"), str(aln), str(base / "tmp"),
           "--alignment-type", "1", "-e", "10", "--max-seqs", "300", "--threads", "8",
           "--format-output", "query,target,alntmscore,evalue,bits"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise SystemExit("foldseek failed")
    print(f"  foldseek: {time.time() - t0:.0f}s", flush=True)
    d = pd.read_csv(aln, sep="\t", names=["query", "target", "tm", "evalue", "bits"])
    d["query"] = d["query"].str.replace(".pdb", "", regex=False)
    d["target"] = d["target"].str.replace(".pdb", "", regex=False)
    lab = {ids[i].decode(): int(y[i]) for i in np.concatenate([sel["train"], sel["test"]])}
    best = d.sort_values("tm", ascending=False).drop_duplicates("query")
    truth = np.array([lab[q] for q in best["query"]])
    pred = np.array([lab[t] for t in best["target"]])
    n_test = len(sel["test"])
    hit = len(best)
    acc = float((truth == pred).sum()) / n_test          # a test domain with no hit counts as wrong
    per = pd.DataFrame({"t": truth, "p": pred})
    bal = float(per.assign(ok=per.t == per.p).groupby("t").ok.mean().mean())
    return dict(accuracy=acc, balanced_accuracy=bal, n_test=n_test, with_hit=hit,
                median_tm=float(best.tm.median()))


def tag_of(a):
    return "" if a.only is None else f"_{a.only}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-set", default="common11")
    ap.add_argument("--splits", nargs="+", default=["superfamily", "topology"])
    ap.add_argument("--skip-alignment", action="store_true")
    ap.add_argument("--only", choices=["spectral", "alignment"], default=None)
    a = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    h = h5py.File(ROOT / "probing_data" / "labels.h5", "r")
    offs, L = h["protein/offset"][:], h["protein/length"][:]
    ca_all, rows = None, []
    for split in a.splits:
        y, n_out, sel = labels(h, split, a.class_set)
        print(f"{split}/{a.class_set}: {n_out} classes, {len(sel['test'])} test domains", flush=True)

        X = None
        if a.only != "alignment":
            # the spectral block is the expensive half; cache it so a failure in the alignment half, which is what
            # happened on the first run, does not cost it again
            cache = SCRATCH / f"spectral_{split}_{a.class_set}.npz"
            if cache.exists():
                z = np.load(cache)
                X = {p: z[p] for p in R.PARTS}
                print(f"  spectral floor: cached, {X['train'].shape[1]} features", flush=True)
            else:
                if ca_all is None:
                    t0 = time.time()
                    ca_all = h["residue/coords"][:, 1].astype(np.float64)
                    print(f"  read {len(ca_all)} CA positions in {time.time() - t0:.0f}s", flush=True)
                t0 = time.time()
                X = {p: spectral_features(h, offs, L, sel[p], ca_all=ca_all) for p in R.PARTS}
                np.savez_compressed(cache, **X)
                print(f"  spectral floor: {X['train'].shape[1]} features in {time.time() - t0:.0f}s", flush=True)
            PX = dict(zip(R.PARTS, R.standardise(*[X[p] for p in R.PARTS])))
            yy = {p: y[sel[p]] for p in R.PARTS}
            for readout, seed, sc in R.run_probe(PX, yy, "clf", n_out):
                rows.append(dict(split=split, class_set=a.class_set, source="ref_spectral",
                                 dim=X["train"].shape[1], readout=readout, seed=seed, **sc))
            pd.DataFrame(rows).to_csv(OUT / f"architecture_strong_floors_{a.class_set}{tag_of(a)}.csv", index=False)
        if not a.skip_alignment and a.only != "spectral":
            r = alignment_1nn(h, split, y, sel, a.class_set)
            rows.append(dict(split=split, class_set=a.class_set, source="ref_align_1nn", dim=0,
                             readout="1-nn", seed=0, **r))
            print(f"  alignment 1-NN accuracy {r['accuracy']:.4f} "
                  f"(balanced {r['balanced_accuracy']:.4f}, hits for {r['with_hit']} of {r['n_test']})", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"architecture_strong_floors_{a.class_set}{tag_of(a)}.csv", index=False)
    print(pd.DataFrame(rows).groupby(["split", "source", "readout"]).accuracy.max().round(4).to_string())


if __name__ == "__main__":
    main()
