"""Settle the undecided fold-architecture cell with 15 optimiser seeds.

The cell is the strictest fold-architecture comparison: topology-disjoint split, matched MLP-2
readout, restricted class set. At three seeds the gap between the best encoder and the best floor was
-0.011 / +0.144 / +0.047 once `ref_pairhist` is in the floor set, so its sign was not resolved. This script reruns
that cell at 15 seeds for the whole 33-encoder panel and for the distinct protein-level floors, so the
"floor wins outright" fraction is comparable to the three-seed statement.

`run_probing_suite.MLP_SEEDS` is overridden in this process only; the imported module is not edited. Each invocation
writes its own part file under probing_results/architecture_seed_interval_parts/. `--merge` concatenates them into
probing_results/architecture_seed_interval.csv.

  python architecture_seed_interval.py --sources floors --tag floors
  python architecture_seed_interval.py --sources encoders:0:3 --tag enc0
  python architecture_seed_interval.py --merge
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R
import architecture_all_floors as A

ROOT = A.ROOT
PARTS_DIR = ROOT / "probing_results" / "architecture_seed_interval_parts"
SPLIT = "topology"
N_SEEDS = 15
# The distinct protein-level floors. ref_position, ref_seq_window and ref_backbone are bit-identical twins of
# ref_length, ref_aa and ref_coords at protein level, so scoring them again would add nothing.
FLOORS = ["ref_length", "ref_aa", "ref_coords", "ref_global", "ref_pairhist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources")
    ap.add_argument("--tag")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    if a.merge:
        d = pd.concat([pd.read_csv(f) for f in sorted(PARTS_DIR.glob("*.csv"))], ignore_index=True)
        d.to_csv(ROOT / "probing_results" / "architecture_seed_interval.csv", index=False)
        print(f"merged {len(d)} rows from {len(list(PARTS_DIR.glob('*.csv')))} parts")
        return

    R.MLP_SEEDS = tuple(range(N_SEEDS))       # this process only
    if a.sources == "floors":
        sources = list(FLOORS)
    elif a.sources.startswith("encoders"):
        bits = a.sources.split(":")
        names = A.encoder_names()
        sources = list(np.array_split(np.array(names), int(bits[2]))[int(bits[1])]) if len(bits) == 3 else names
    else:
        sources = a.sources.split(",")

    h = h5py.File(ROOT / "probing_data" / "labels.h5", "r")
    L = h["protein/length"][:].astype(np.int64)
    aa, pidx = h["residue/aa"][:], h["residue/protein_index"][:]
    comp = np.zeros((len(L), 20), dtype=np.float32)
    np.add.at(comp, (pidx, aa), 1.0)
    comp /= L[:, None]
    cache = {}
    if any(s.startswith("ref_") for s in sources):
        cache = {"global": A.load_cached("global"), "coord": A.load_cached("coord"),
                 "pairhist": A.load_cached("pairhist")}

    s = h[f"protein/split_{SPLIT}"][:]
    arch = h[f"protein/architecture_label_{SPLIT}"][:].astype(np.int64)
    idx = {p: np.where(s == c)[0] for p, c in R.PARTS.items()}
    n_all = int(arch.max()) + 1
    vocab = h[f"protein/architecture_label_{SPLIT}"].attrs["classes"].split(",")
    pos = {c: i for i, c in enumerate(vocab)}

    label_sets = {}
    keep11 = np.array([pos[c] for c in A.SHARED_11], dtype=np.int64)
    remap = -np.ones(n_all, dtype=np.int64); remap[keep11] = np.arange(len(A.SHARED_11))
    label_sets["common11"] = (np.where(arch >= 0, remap[np.clip(arch, 0, None)], -1), len(A.SHARED_11))
    te = arch[idx["test"]]
    keep13 = np.where(np.bincount(te[te >= 0], minlength=n_all) >= A.MIN_TEST)[0]
    remap = -np.ones(n_all, dtype=np.int64); remap[keep13] = np.arange(len(keep13))
    label_sets["common13"] = (np.where(arch >= 0, remap[np.clip(arch, 0, None)], -1), len(keep13))

    rows, t0 = [], time.time()
    for class_set, (y_all, n_out) in label_sets.items():
        sel = {p: idx[p][y_all[idx[p]] >= 0] for p in R.PARTS}
        print(f"{SPLIT}/{class_set}: n_out {n_out}, test proteins {len(sel['test'])}", flush=True)
        for src in sources:
            X = A.floor_features(src, L, comp, cache) if src.startswith("ref_") \
                else np.load(ROOT / "probing_embeddings" / src / "pooled_fp32.npy")
            PX = dict(zip(R.PARTS, R.standardise(*[X[sel[p]].astype(np.float32) for p in R.PARTS])))
            y = {p: y_all[sel[p]] for p in R.PARTS}
            for readout, seed, sc in R.run_probe(PX, y, "clf", n_out):
                rows.append(dict(split=SPLIT, class_set=class_set, source=src, dim=int(X.shape[1]),
                                 n_classes=n_out, readout=readout, seed=seed, **sc))
            del PX, X
            torch.cuda.empty_cache()
            print(f"  [{time.time() - t0:5.0f}s] {src}", flush=True)

    pd.DataFrame(rows).to_csv(PARTS_DIR / f"{a.tag}.csv", index=False)
    print(f"wrote {PARTS_DIR / (a.tag + '.csv')}, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
