"""Is the fold-architecture gap robust to class imbalance, split and readout?

Fold architecture is the only probed property whose label is not a function of the encoder's input, and it is the only
one where encoders clearly exceed every non-learned floor. Three things could produce that gap without any real
advantage: rare classes inflating balanced accuracy, the two splits averaging over different class sets, and taking
the encoder's best readout while the floor gets its worst. All three are tested here on one common class set.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
MIN_TEST = 20


def main():
    h = h5py.File(ROOT / "probing_data" / "labels.h5", "r")
    data = R.Data()
    L = h["protein/length"][:].astype(np.int64)
    aa, pidx = h["residue/aa"][:], h["residue/protein_index"][:]
    comp = np.zeros((len(L), 20), dtype=np.float32)
    np.add.at(comp, (pidx, aa), 1.0)
    comp /= L[:, None]
    sources = ["ref_aa", "ref_global"] + sorted(d.name for d in (ROOT / "probing_embeddings").iterdir()
                                                if (d / "meta.json").exists() and not d.name.endswith("_layers"))
    rows = []
    for split in ("superfamily", "topology"):
        s = h[f"protein/split_{split}"][:]
        arch = h[f"protein/architecture_label_{split}"][:].astype(np.int64)
        idx = {p: np.where(s == c)[0] for p, c in R.PARTS.items()}
        te = arch[idx["test"]]
        counts = np.bincount(te[te >= 0], minlength=int(arch.max()) + 1)
        keep = np.where(counts >= MIN_TEST)[0]                 # the common class set
        remap = -np.ones(int(arch.max()) + 1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        y_all = np.where(arch >= 0, remap[np.clip(arch, 0, None)], -1)
        sel = {p: idx[p][y_all[idx[p]] >= 0] for p in R.PARTS}
        print(f"{split}: {len(keep)} classes with at least {MIN_TEST} test domains "
              f"(of {int((counts > 0).sum())} present); test proteins {len(sel['test'])}", flush=True)
        for src in sources:
            if src == "ref_aa":
                X = comp
            elif src == "ref_global":
                Xp = np.concatenate([data.global_features_protein(sel[p]) for p in R.PARTS])
                order = np.concatenate([sel[p] for p in R.PARTS])
                X = np.zeros((len(L), Xp.shape[1]), dtype=np.float32); X[order] = Xp
            else:
                f = ROOT / "probing_embeddings" / src / "pooled_fp32.npy"
                if not f.exists():
                    continue
                X = np.load(f)
            PX = dict(zip(R.PARTS, R.standardise(*[X[sel[p]].astype(np.float32) for p in R.PARTS])))
            y = {p: y_all[sel[p]] for p in R.PARTS}
            for readout, seed, sc in R.run_probe(PX, y, "clf", len(keep)):
                rows.append(dict(split=split, source=src, readout=readout, seed=seed, n_classes=len(keep), **sc))
            torch.cuda.empty_cache()
        print(f"  {split} done", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "probing_results" / "architecture_robustness.csv", index=False)
    pd.set_option("display.width", 220)
    for metric in ("balanced_accuracy", "macro_f1"):
        w = d.groupby(["split", "source", "readout"])[metric].mean().unstack(["split", "readout"])
        print(f"\n=== {metric}, common class set ===")
        print(w.round(3).sort_values(w.columns[0], ascending=False).to_string())
    # matched readout: floor and encoder both from the same family
    for fam in ("linear", "mlp2"):
        g = d[d.readout == fam].groupby(["split", "source"]).balanced_accuracy.mean().unstack(0)
        enc = g.drop(index=[i for i in g.index if i.startswith("ref_")])
        print(f"\nmatched readout {fam}: gap = best encoder minus ref_global")
        for sp in g.columns:
            print(f"  {sp:14s} best encoder {enc[sp].max():.3f} ({enc[sp].idxmax()}), ref_global {g.loc['ref_global', sp]:.3f}, "
                  f"gap {enc[sp].max() - g.loc['ref_global', sp]:+.3f}")


if __name__ == "__main__":
    main()
