"""Does the fold-architecture result survive the topology-disjoint split?

Fold architecture is the one probed property a fixed backbone descriptor does not recover. The main table uses the
superfamily-disjoint split. `labels.h5` also ships a topology-disjoint split, which is strictly harder: no CATH
topology is shared between train and test, so a probe cannot succeed by recognising a fold it has seen.
Protein level only, so this is cheap.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))


def run(split):
    R.SPLIT = split
    import h5py
    h = h5py.File(ROOT / "probing_data" / "labels.h5", "r")
    s = h[f"protein/split_{split}"][:]
    arch = h[f"protein/architecture_label_{split}"][:].astype(np.int64)
    idx = {p: np.where(s == c)[0] for p, c in R.PARTS.items()}
    L = h["protein/length"][:].astype(np.int64)
    offs = h["protein/offset"][:].astype(np.int64)
    aa, pidx = h["residue/aa"][:], h["residue/protein_index"][:]
    comp = np.zeros((len(L), 20), dtype=np.float32)
    np.add.at(comp, (pidx, aa), 1.0)
    comp /= L[:, None]
    out = {}
    sources = ["ref_aa", "ref_coords", "ref_global"] + sorted(d.name for d in (ROOT / "probing_embeddings").iterdir()
                                                if (d / "meta.json").exists() and not d.name.endswith("_layers"))
    data = R.Data()
    for src in sources:
        if src == "ref_aa":
            X = comp
        elif src in ("ref_coords", "ref_global"):
            fn = data.coord_features_protein if src == "ref_coords" else data.global_features_protein
            X = np.concatenate([fn(idx[p]) for p in R.PARTS])
            order = np.concatenate([idx[p] for p in R.PARTS])
            full = np.zeros((len(L), X.shape[1]), dtype=np.float32); full[order] = X; X = full
        else:
            f = ROOT / "probing_embeddings" / src / "pooled_fp32.npy"
            if not f.exists():
                continue
            X = np.load(f)
        ok = {p: idx[p][arch[idx[p]] >= 0] for p in R.PARTS}
        PX = dict(zip(R.PARTS, R.standardise(*[X[ok[p]].astype(np.float32) for p in R.PARTS])))
        y = {p: arch[ok[p]] for p in R.PARTS}
        sc = R.run_probe(PX, y, "clf", int(arch.max()) + 1)
        out[src] = max(s["balanced_accuracy"] for _, _, s in sc)
        torch.cuda.empty_cache()
    return pd.Series(out, name=split)


if __name__ == "__main__":
    a, b = run("superfamily"), run("topology")
    t = pd.concat([a, b], axis=1)
    t["drop"] = t["superfamily"] - t["topology"]
    t.round(3).to_csv(ROOT / "probing_results" / "architecture_by_split.csv")
    pd.set_option("display.width", 200)
    print(t.sort_values("superfamily", ascending=False).round(3).to_string())
