"""On the second GPU machine: put the arrays of gap_worker.py where the scoring scripts read them.

The split files written here are the ones finalize_gap.py writes on the first machine (rows of pooled.npy by split, in
manifest order). The meta.json files are placeholders that mark the folders as extracted; the released metadata is
written by finalize_gap.py after its check.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

G = Path(__file__).resolve().parent
R = G / "2026_encoder" / "resubmission"
z = np.load(G / "data" / "remote_homology.npz")
X = np.load(G / "out" / "remote_homology" / "foldvision" / "pooled.npy")
man = pd.DataFrame({"id": z["ids"], "split": z["split"]})
assert len(man) == len(X) and np.isfinite(X).all() and (np.abs(X).sum(1) > 0).all()
out = R / "extract" / "embeddings" / "remote_homology" / "foldvision"
out.mkdir(parents=True, exist_ok=True)
for split, g in man.groupby("split"):
    np.save(out / f"{split}.npy", X[g.index.to_numpy()])
    (out / f"ids_{split}.txt").write_text("\n".join(g.id) + "\n")
(out / "meta.json").write_text(json.dumps({"encoder": "foldvision", "placeholder": True}))
for e in ("saprot", "gcpnet_pretrained", "gcpnet_random"):
    d = R / "masif_site" / "embeddings" / e
    d.mkdir(parents=True, exist_ok=True)
    for f in ("residue_fp16.npy", "pooled_fp32.npy"):
        if not (d / f).exists():
            os.symlink(G / "out" / "masif" / e / f, d / f)
    (d / "meta.json").write_text(json.dumps({"encoder": e, "placeholder": True}))
print("prepared", {s: int(n) for s, n in man.groupby("split").size().items()})
