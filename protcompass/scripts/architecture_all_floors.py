"""Fold architecture against all seven non-learned floors, on both splits and both class sets.

This script checks whether encoders exceed every non-learned floor on fold architecture.
`scripts/architecture_robustness.py` ran only `ref_aa` and `ref_global`. This script runs all seven reference
feature sets -- ref_length, ref_position, ref_aa, ref_seq_window, ref_coords, ref_backbone, ref_global -- through the
identical probe, split, readout and seed protocol as the encoders, so the best floor in each cell is measured rather
than assumed.

Two class sets are scored:
  common11  the 11 CATH architectures both restricted sets share, remapped to a common index across splits
  common13  the 13 classes with at least MIN_TEST test domains in that split (the section 3 restriction)
  full      every class present, which is what probing_results/architecture_by_split.csv reports

Each invocation writes its own part file under probing_results/architecture_all_floors_parts/, so two runs never
write the same path. `--merge` concatenates the parts into probing_results/architecture_all_floors.csv.

  python architecture_all_floors.py --class-set common13 --sources floors --tag floors_common13
  python architecture_all_floors.py --class-set full --sources encoders:0:3 --tag enc_full_0
  python architecture_all_floors.py --merge
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
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

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
PARTS_DIR = ROOT / "probing_results" / "architecture_all_floors_parts"
CACHE = Path((PROTCOMPASS_ROOT + "/protcompass/cache/scratch/feat"))
MIN_TEST = 20
FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_coords", "ref_backbone", "ref_global"]
# The CATH architectures present at the >=MIN_TEST cut in both splits. The two restricted sets share these eleven;
# the superfamily set additionally has 1.25 and 2.60, the topology set 3.50 and 3.60.
SHARED_11 = ["1.10", "1.20", "2.30", "2.40", "3.10", "3.20", "3.30", "3.40", "3.90", "4.10", "6.10"]


def load_cached(which):
    """Reassemble a protein-level descriptor from the shards written by precompute_floor_protein_features.py."""
    shards = sorted(CACHE.glob(f"{which}_shard*of*.npy"))
    shards = [s for s in shards if not s.name.endswith("_idx.npy")]
    if not shards:
        raise FileNotFoundError(f"no cached {which} descriptor in {CACHE}")
    xs = [np.load(s) for s in shards]
    idx = [np.load(s.with_name(s.stem + "_idx.npy")) for s in shards]
    n = sum(len(i) for i in idx)
    out = np.zeros((n, xs[0].shape[1]), dtype=np.float32)
    for x, i in zip(xs, idx):
        out[i] = x
    return out


def floor_features(source, L, comp, cache):
    """The same branches as run_probing_suite.Data.protein_features, over every protein."""
    if source == "ref_pairhist":
        return cache["pairhist"]
    if source == "ref_global":
        return cache["global"]
    if source in ("ref_coords", "ref_backbone"):      # protein_features maps both to coord_features_protein
        return cache["coord"]
    if source in ("ref_aa", "ref_seq_window"):        # at protein level both reduce to composition
        return comp
    if source in ("ref_length", "ref_position"):      # at protein level both reduce to the length features
        Lf = L.astype(np.float64)
        return np.stack([np.log(Lf), np.log(Lf) ** 2, Lf / 600.0], 1).astype(np.float32)
    raise ValueError(source)


def encoder_names():
    return sorted(d.name for d in (ROOT / "probing_embeddings").iterdir()
                  if (d / "meta.json").exists() and not d.name.endswith("_layers")
                  and (d / "pooled_fp32.npy").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-set", choices=["common11", "common13", "full"])
    ap.add_argument("--sources", help="'floors', 'encoders', 'encoders:<shard>:<n>' or a comma-separated list")
    ap.add_argument("--tag")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    if a.merge:
        d = pd.concat([pd.read_csv(f) for f in sorted(PARTS_DIR.glob("*.csv"))], ignore_index=True)
        d.to_csv(ROOT / "probing_results" / "architecture_all_floors.csv", index=False)
        print(f"merged {len(d)} rows from {len(list(PARTS_DIR.glob('*.csv')))} parts")
        return

    if a.sources == "floors":
        sources = list(FLOORS)
    elif a.sources.startswith("encoders"):
        bits = a.sources.split(":")
        names = encoder_names()
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
    if any(s in ("ref_coords", "ref_backbone", "ref_global") for s in sources):
        cache |= {"global": load_cached("global"), "coord": load_cached("coord")}
    if "ref_pairhist" in sources:
        cache |= {"pairhist": load_cached("pairhist")}

    rows, t0 = [], time.time()
    for split in ("superfamily", "topology"):
        s = h[f"protein/split_{split}"][:]
        arch = h[f"protein/architecture_label_{split}"][:].astype(np.int64)
        idx = {p: np.where(s == c)[0] for p, c in R.PARTS.items()}
        n_all = int(arch.max()) + 1
        if a.class_set == "common11":
            # The 11 CATH architectures both restricted class sets contain, remapped so that label k denotes the
            # same architecture in both splits. Each split has its own `classes` vocabulary (superfamily carries
            # 1.50 where topology carries 2.120), so the mapping has to go through the CATH string, not the index.
            vocab = h[f"protein/architecture_label_{split}"].attrs["classes"].split(",")
            pos = {c: i for i, c in enumerate(vocab)}
            keep = np.array([pos[c] for c in SHARED_11], dtype=np.int64)
            remap = -np.ones(n_all, dtype=np.int64)
            remap[keep] = np.arange(len(SHARED_11))            # common index, ordered by SHARED_11
            y_all = np.where(arch >= 0, remap[np.clip(arch, 0, None)], -1)
            n_out = len(SHARED_11)
        elif a.class_set == "common13":
            te = arch[idx["test"]]
            counts = np.bincount(te[te >= 0], minlength=n_all)
            keep = np.where(counts >= MIN_TEST)[0]
            remap = -np.ones(n_all, dtype=np.int64)
            remap[keep] = np.arange(len(keep))
            y_all = np.where(arch >= 0, remap[np.clip(arch, 0, None)], -1)
            n_out = len(keep)
        else:
            y_all, n_out = arch, n_all
        sel = {p: idx[p][y_all[idx[p]] >= 0] for p in R.PARTS}
        print(f"{split}/{a.class_set}: n_out {n_out}, test proteins {len(sel['test'])}", flush=True)
        for part in ("train", "test"):
            c = np.bincount(y_all[sel[part]], minlength=n_out)
            print(f"  {part} class counts {c.tolist()} (largest share {c.max() / c.sum():.3f})", flush=True)
        for src in sources:
            X = floor_features(src, L, comp, cache) if src.startswith("ref_") \
                else np.load(ROOT / "probing_embeddings" / src / "pooled_fp32.npy")
            PX = dict(zip(R.PARTS, R.standardise(*[X[sel[p]].astype(np.float32) for p in R.PARTS])))
            y = {p: y_all[sel[p]] for p in R.PARTS}
            for readout, seed, sc in R.run_probe(PX, y, "clf", n_out):
                rows.append(dict(split=split, class_set=a.class_set, source=src, dim=int(X.shape[1]),
                                 n_classes=n_out, n_train=len(sel["train"]), n_test=len(sel["test"]),
                                 readout=readout, seed=seed, **sc))
            del PX, X
            torch.cuda.empty_cache()
            print(f"  [{time.time() - t0:5.0f}s] {src}", flush=True)

    pd.DataFrame(rows).to_csv(PARTS_DIR / f"{a.tag}.csv", index=False)
    print(f"wrote {PARTS_DIR / (a.tag + '.csv')}, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
