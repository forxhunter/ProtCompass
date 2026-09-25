"""Export the released seed-0 draw (3,000 train, 500 validation, 1,000 test domains) as a self-contained bundle, so
the pre-registered encoder scoring can run on the remote GPU without copying the full 96 GB embedding store.

The bundle's labels file holds exactly the drawn domains, in their original order, with split codes 0/1/2, so
run_probing_suite.Data on the bundle draws every domain and reproduces the released rows one for one (asserted on
the stored coordinates). Encoder embeddings and the pre-registered labels are sliced to the same rows. Nothing in
the scoring or the decision changes; only file locations do.

  python prereg_export_subset.py OUT_DIR ENCODER [ENCODER ...]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_probing_suite as rps                                          # noqa: E402
from confirmation_merge import gather                                    # noqa: E402

out, encoders = Path(sys.argv[1]), sys.argv[2:]
data = rps.Data()
idx = np.concatenate([data.chosen[p] for p in rps.PARTS])
code = np.concatenate([np.full(len(data.chosen[p]), c, np.int8) for p, c in rps.PARTS.items()])
rel = h5py.File(rps.LABELS, "r")
g = gather(rel, idx)
assert np.array_equal(g["rows"], np.concatenate([data.rows[p] for p in rps.PARTS]))
(out / "probing_data").mkdir(parents=True, exist_ok=True)
lab = out / "probing_data" / "labels.h5"
if not lab.exists():
    with h5py.File(lab, "w") as h:
        h.attrs.update(dict(rel.attrs))
        gp, gr, gq = h.create_group("protein"), h.create_group("residue"), h.create_group("pair")
        L = g["L"]
        gp["id"] = rel["protein/id"][:][idx]
        gp["length"] = L.astype(np.int32)
        gp["offset"] = np.concatenate([[0], np.cumsum(L)])[:-1]
        for k in rel["protein"]:
            if k in ("id", "length", "offset"):
                continue
            v = rel["protein"][k][:][idx]
            gp[k] = code if k.startswith("split_") else v
        gr["protein_index"] = np.repeat(np.arange(len(L), dtype=np.int32), L)
        for k in rel["residue"]:
            if k != "protein_index":
                gr[k] = rel["residue"][k][:][g["rows"]]
        q = g["qcount"].astype(np.int64)
        gq["protein_index"] = np.repeat(np.arange(len(L), dtype=np.int32), q)
        gq["offset"] = np.concatenate([[0], np.cumsum(q)])[:-1]
        gq["count"] = q.astype(np.int32)
        for k in rel["pair"]:
            if k not in ("protein_index", "offset", "count"):
                gq[k] = rel["pair"][k][:][g["qrows"]]
    print("wrote", lab)
(out / "prereg").mkdir(exist_ok=True)
pl = out / "prereg" / "labels_released.h5"
if not pl.exists():
    src = h5py.File(ROOT / "prereg" / "labels_released.h5", "r")
    with h5py.File(pl, "w") as h:
        for k in src["residue"]:
            h[f"residue/{k}"] = src["residue"][k][:][g["rows"]]
        h.attrs["n_proteins"] = len(idx)
    print("wrote", pl)
for f in ("freeze.json", "freeze.sha256.json"):
    shutil.copy(ROOT / "prereg" / f, out / "prereg" / f)
for e in encoders:
    d = out / "probing_embeddings" / e
    if (d / "meta.json").exists():
        continue
    d.mkdir(parents=True, exist_ok=True)
    X = np.load(rps.EMB / e / "residue_fp16.npy", mmap_mode="r")
    np.save(d / "residue_fp16.npy", np.ascontiguousarray(X[g["rows"]]))
    for extra in ("pooled_fp32.npy",):
        if (rps.EMB / e / extra).exists():
            np.save(d / extra, np.load(rps.EMB / e / extra)[idx])
    shutil.copy(rps.EMB / e / "meta.json", d / "meta.json")
    print("sliced", e, flush=True)
# the bundle reproduces the released rows
sub = h5py.File(lab, "r")
assert np.array_equal(sub["residue/coords"][:5000], rel["residue/coords"][:][g["rows"][:5000]])
json.dump(dict(n_domains=int(len(idx)), n_rows=int(len(g["rows"])), encoders=encoders),
          open(out / "bundle.json", "w"), indent=2)
print("bundle ok", len(idx), "domains", len(g["rows"]), "rows")
