"""Build a residue-level interface-site task from the MaSIF-site chains, in the layout of probing_data/labels.h5.

The ProteinWorkshop copy of MaSIF-site (structure_encoder_data/proteinworkshop/masif_site) stores, for each listed
chain, the whole deposited structure (every chain, 37 heavy-atom slots per residue) and the name of the target chain.
It carries no residue labels, so the label is derived from geometry:

  interface   a residue of the target chain is an interface residue if any of its heavy atoms lies within 5.0 A of a
              heavy atom of any other chain in the same file.

The encoders see the target chain alone, so an entry whose target is several chains (an antibody H+L pair, for
example) is excluded. The cleaning rules are the released ones of probing_data/make_probing_data.py
(imported, not re-implemented): selenium analogues remapped, residues outside the 20 standard types and residues
without a full N, CA, C, O backbone dropped, a chain dropped if more than 2% of its residues are non-standard or more
than 5% incomplete, length 40 to 600, and a virtual CB where the file has none.

Split: the MaSIF-site training list is train, its testing list is test, and 10% of the training chains (seed 0) are
held out as validation for early stopping. Nothing is fitted on the test list.

  python masif_build_labels.py [--workers 16]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import importlib.util
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
SRC = ROOT.parent / "structure_encoder_data" / "proteinworkshop" / "masif_site"
OUT = ROOT / "masif_site"
CUTOFF = 5.0
SENTINEL = 1e-5            # ProteinWorkshop fills absent atom slots with this value
VAL_FRAC, SEED = 0.10, 0


def released_rules():
    spec = importlib.util.spec_from_file_location("mpd", ROOT / "probing_data" / "make_probing_data.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["mpd"] = m
    spec.loader.exec_module(m)
    return m


M = released_rules()


def process(entry):
    """Clean one target chain; return (entry, reason, detail, arrays or None)."""
    import torch
    from scipy.spatial import cKDTree
    f = SRC / "processed" / f"{entry}.pt"
    if not f.exists():
        return entry, "missing_file", "", None
    d = torch.load(f, weights_only=False)
    target = str(d.graph_y)
    xyz = d.coords.numpy().astype(np.float64)                        # (N, 37, 3); slots 0..4 are N, CA, C, O, CB
    present = ~np.all(np.abs(xyz - SENTINEL) < 1e-9, axis=-1)          # (N, 37)
    chain = np.array([r.split(":")[0] for r in d.residue_id])
    names = [M.REMAP.get(n, n) for n in d.residues]
    tgt = np.where(chain == target)[0]
    other = np.where(chain != target)[0]
    if len(target) > 1:                                                 # e.g. an antibody H+L target; each encoder here reads one chain
        return entry, "multi_chain_target", target, None
    if len(tgt) == 0:
        return entry, "no_target_chain", target, None
    if len(other) == 0:
        return entry, "no_partner_chain", target, None
    n_raw = len(tgt)
    nonstd = np.array([names[i] not in M.AA_INDEX for i in tgt])
    incomplete = ~present[tgt][:, :4].all(1)
    if nonstd.sum() / n_raw > M.MAX_NONSTANDARD_FRAC:
        return entry, "nonstandard_residues", f"{nonstd.sum()}/{n_raw}", None
    if (incomplete & ~nonstd).sum() / n_raw > M.MAX_INCOMPLETE_FRAC:
        return entry, "incomplete_backbone", f"{(incomplete & ~nonstd).sum()}/{n_raw}", None
    keep = tgt[~nonstd & ~incomplete]
    L = len(keep)
    if L < M.MIN_LEN or L > M.MAX_LEN:
        return entry, "length_out_of_range", str(L), None
    bb = xyz[keep][:, :4]
    cb = M.virtual_cb(bb[:, 0], bb[:, 1], bb[:, 2])
    has_cb = present[keep][:, 4]
    cb[has_cb] = xyz[keep][has_cb, 4]
    coords = np.concatenate([bb, cb[:, None]], 1).astype(np.float32)
    aa = np.array([M.AA_INDEX[names[i]] for i in keep], dtype=np.int8)
    partner = xyz[other][present[other]]
    tree = cKDTree(partner)
    interface = np.zeros(L, dtype=np.int8)
    for k, i in enumerate(keep):
        atoms = xyz[i][present[i]]
        if len(tree.query_ball_point(atoms, CUTOFF, return_length=True).nonzero()[0]):
            interface[k] = 1
    resseq = np.array([int(d.residue_id[i].split(":")[2]) for i in keep], dtype=np.int32)
    return entry, "kept", "", dict(coords=coords, aa=aa, interface=interface, resseq=resseq, cb_virtual=~has_cb,
                                   n_partner_chains=len(set(chain[other])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    lists = {part: (SRC / f"{part}.txt").read_text().split() for part in ("training", "testing")}
    entries = lists["training"] + lists["testing"]
    with Pool(a.workers) as pool:
        res = pool.map(process, entries, chunksize=8)
    excl = pd.DataFrame([(e, r, d) for e, r, d, _ in res if r != "kept"], columns=["id", "reason", "detail"])
    OUT.mkdir(exist_ok=True)
    excl.to_csv(OUT / "exclusions.csv", index=False)
    kept = [(e, arr) for e, r, _, arr in res if r == "kept"]
    train_ids = [e for e, _ in kept if e in set(lists["training"])]
    rng = np.random.default_rng(SEED)
    val_ids = set(rng.choice(train_ids, size=int(round(VAL_FRAC * len(train_ids))), replace=False).tolist())
    split = np.array([2 if e in set(lists["testing"]) else (1 if e in val_ids else 0) for e, _ in kept], dtype=np.int8)
    L = np.array([len(arr["aa"]) for _, arr in kept], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(L)[:-1]]).astype(np.int64)
    cat = lambda k: np.concatenate([arr[k] for _, arr in kept])
    with h5py.File(OUT / "labels.h5", "w") as h:
        p = h.create_group("protein")
        p["id"] = np.array([e.encode() for e, _ in kept])
        p["length"], p["offset"] = L, off
        p["split_superfamily"] = split           # the name the probe code reads; here it is the MaSIF-site split
        p["split_masif"] = split
        p["n_partner_chains"] = np.array([arr["n_partner_chains"] for _, arr in kept], dtype=np.int16)
        r = h.create_group("residue")
        r["coords"], r["aa"], r["interface"] = cat("coords"), cat("aa"), cat("interface")
        r["resseq"], r["cb_virtual"] = cat("resseq"), cat("cb_virtual")
        r["protein_index"] = np.repeat(np.arange(len(kept)), L).astype(np.int32)
        r["position"] = np.concatenate([np.arange(n) for n in L]).astype(np.int32)
        q = h.create_group("pair")               # no pair labels; empty arrays keep the reader's layout
        q["offset"], q["count"] = np.zeros(len(kept), dtype=np.int64), np.zeros(len(kept), dtype=np.int64)
        for k in ("protein_index", "i", "j", "distance_bin"):
            q[k] = np.zeros(0, dtype=np.int64)
        q["distance"] = np.zeros(0, dtype=np.float32)
        h.attrs["provenance"] = json.dumps(dict(source=str(SRC), label=f"heavy atom within {CUTOFF} A of another chain",
                                                cleaning="probing_data/make_probing_data.py rules", val_frac=VAL_FRAC, seed=SEED))
    iface = cat("interface")
    summary = dict(entries=len(entries), kept=len(kept), excluded=dict(Counter(excl.reason)),
                   chains_per_split={k: int((split == c).sum()) for k, c in (("train", 0), ("val", 1), ("test", 2))},
                   residues=int(L.sum()), interface_rate=float(iface.mean()),
                   interface_rate_per_split={k: float(iface[np.repeat(split, L) == c].mean()) for k, c in (("train", 0), ("val", 1), ("test", 2))},
                   length_median=float(np.median(L)))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "ids.txt").write_text("\n".join(e for e, _ in kept) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
