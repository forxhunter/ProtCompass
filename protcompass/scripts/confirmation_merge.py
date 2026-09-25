"""Merge the released fitting split with the confirmation cohort into one dataset the released scorer can read.

The confirmation design is: fit every probe on the proteins the released run fitted on, and evaluate on proteins no
part of this work has inspected. `run_probing_suite.py` fits and scores inside one labels file, so rather than
rewrite the scorer, this script builds a file whose train and validation parts are exactly the released ones and
whose test part is the cohort.

The train and validation domains are the seed-0 draw `run_probing_suite.Data` makes, reproduced here in the same
order, so the fitting set is identical to the released run's, not merely similar. The test part is every cohort
domain. Embeddings are concatenated the same way, row for row, so the alignment the released extractor guarantees
(row r of residue_fp16.npy is row r of residue/*) still holds in the merged file.

Nothing is refitted: the length-residualisation coefficients and the architecture class index already came from the
released file when the cohort labels were built, and the released rows are copied unchanged.

  python confirmation_merge.py [--encoders esm2_650m ...]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
PD = ROOT / "probing_data"
CONF = Path(os.environ.get("CONFIRMATION_DIR") or ROOT / "confirmation")   # cohort B: confirmation2
REL_EMB, COH_EMB = ROOT / "probing_embeddings", CONF / "embeddings"
OUT = CONF / "merged"
SPLIT, N_PROT, PARTS = "superfamily", {"train": 3000, "val": 500, "test": 1000}, {"train": 0, "val": 1, "test": 2}

RESIDUE_KEYS = ["coords", "aa", "resseq", "icode", "segment", "cb_virtual", "phi", "psi", "phi_sin", "phi_cos",
                "psi_sin", "psi_cos", "phi_bin", "psi_bin", "curvature", "dssp8", "dssp3", "sasa", "rsa", "rsa_bin",
                "bfactor_raw", "bfactor_z", "contact_count", "contact_count_sep4", "contact_density_bin",
                "cb_neighbours", "core", "sequence_recovery", "position"]
PAIR_KEYS = ["i", "j", "distance", "distance_bin"]


def released_fit_draw(h):
    """The train and validation domains run_probing_suite.Data draws, in its own order, with its own seed."""
    split = h["protein"][f"split_{SPLIT}"][:]
    rng = np.random.default_rng(0)
    chosen = {}
    for p, c in PARTS.items():                       # order matters: it advances the generator
        pool = np.where(split == c)[0]
        chosen[p] = np.sort(rng.choice(pool, size=min(N_PROT[p], len(pool)), replace=False))
    return chosen["train"], chosen["val"]


def gather(h, idx):
    """Per-protein slices of one labels file for the given protein indices, in the given order."""
    P, R, Q = h["protein"], h["residue"], h["pair"]
    L = P["length"][:].astype(np.int64)
    off = P["offset"][:].astype(np.int64)
    qcount = Q["count"][:].astype(np.int64)
    qoff = Q["offset"][:].astype(np.int64)
    rows = np.concatenate([np.arange(off[i], off[i] + L[i]) for i in idx]) if len(idx) else np.zeros(0, np.int64)
    qrows = np.concatenate([np.arange(qoff[i], qoff[i] + qcount[i]) for i in idx]) if len(idx) else np.zeros(0, np.int64)
    return dict(L=L[idx], rows=rows, qrows=qrows, qcount=qcount[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=None, help="default: whatever the cohort store holds")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((CONF / "freeze.json").read_text())

    rel = h5py.File(PD / "labels.h5", "r")
    coh = h5py.File(CONF / "labels_v44.h5", "r")
    tr, va = released_fit_draw(rel)
    n_coh = len(coh["protein"]["id"])
    coh_idx = np.arange(n_coh)
    print(f"released fitting draw: {len(tr)} train, {len(va)} val; cohort test: {n_coh}")

    rel_idx = np.concatenate([tr, va])
    g_rel, g_coh = gather(rel, rel_idx), gather(coh, coh_idx)
    L = np.concatenate([g_rel["L"], g_coh["L"]])
    offsets = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
    split = np.concatenate([np.zeros(len(tr), np.int8), np.ones(len(va), np.int8),
                            np.full(n_coh, 2, np.int8)])
    ids = np.concatenate([rel["protein"]["id"][:][rel_idx], coh["protein"]["id"][:]])
    assert len(set(ids)) == len(ids), "a domain appears in both parts"

    out = OUT / "labels.h5"
    if out.exists():
        out.unlink()
    with h5py.File(out, "w") as h:
        h.attrs.update(dict(rel.attrs))
        h.attrs["cath_release"] = "v4_3_0 fitting split + v4_4_0 confirmation cohort"
        h.attrs["domain_set"] = ("train and validation are the released seed-0 fitting draw; test is the frozen "
                                 "confirmation cohort, absent from the whole v4.3 release")
        h.attrs["cohort_manifest_sha256"] = freeze["cohort"]["manifest_sha256"]
        h.attrs["frozen_utc"] = freeze["written_utc"]
        gp, gr, gq = h.create_group("protein"), h.create_group("residue"), h.create_group("pair")
        gp.create_dataset("id", data=ids)
        gp.create_dataset("length", data=L.astype(np.int32))
        gp.create_dataset("offset", data=offsets[:-1])
        for name in ("superfamily", "topology"):
            ds = gp.create_dataset(f"split_{name}", data=split)
            ds.attrs["codes"] = "0=train,1=val,2=test"
        for k in rel["protein"]:
            if k in ("id", "length", "offset") or k.startswith("split_"):
                continue
            if k not in coh["protein"]:
                continue
            gp.create_dataset(k, data=np.concatenate([rel["protein"][k][:][rel_idx], coh["protein"][k][:]]))
            for ak, av in rel["protein"][k].attrs.items():
                gp[k].attrs[ak] = av

        n_res = int(L.sum())
        gr.create_dataset("protein_index", data=np.repeat(np.arange(len(L), dtype=np.int32), L), compression="gzip")
        for k in RESIDUE_KEYS:
            if k not in rel["residue"] or k not in coh["residue"]:
                print(f"  [skip residue] {k}")
                continue
            v = np.concatenate([rel["residue"][k][:][g_rel["rows"]], coh["residue"][k][:][g_coh["rows"]]])
            assert len(v) == n_res, f"{k}: {len(v)} != {n_res}"
            gr.create_dataset(k, data=v, compression="gzip")

        qcount = np.concatenate([g_rel["qcount"], g_coh["qcount"]]).astype(np.int64)
        gq.create_dataset("protein_index", data=np.repeat(np.arange(len(L), dtype=np.int32), qcount),
                          compression="gzip")
        gq.create_dataset("offset", data=np.concatenate([[0], np.cumsum(qcount)])[:-1])
        gq.create_dataset("count", data=qcount.astype(np.int32))
        for k in PAIR_KEYS:
            v = np.concatenate([rel["pair"][k][:][g_rel["qrows"]], coh["pair"][k][:][g_coh["qrows"]]])
            gq.create_dataset(k, data=v, compression="gzip")
    print(f"wrote {out}: {len(L)} domains, {n_res:,} residues, {int(qcount.sum()):,} pairs")

    (OUT / "ids.txt").write_text("\n".join(i.decode() for i in ids) + "\n")
    np.save(OUT / "offsets.npy", offsets[:-1])
    np.save(OUT / "lengths.npy", L.astype(np.int64))

    encoders = a.encoders or sorted(d.name for d in COH_EMB.iterdir() if (d / "meta.json").exists())
    for enc in encoders:
        d_out = OUT / enc
        if (d_out / "meta.json").exists():
            print(f"  [{enc}] already merged"); continue
        rel_f, coh_f = REL_EMB / enc / "residue_fp16.npy", COH_EMB / enc / "residue_fp16.npy"
        if not (rel_f.exists() and coh_f.exists()):
            print(f"  [{enc}] missing residue embeddings, skipped"); continue
        d_out.mkdir(parents=True, exist_ok=True)
        A = np.load(rel_f, mmap_mode="r")
        B = np.load(coh_f, mmap_mode="r")
        assert A.shape[1] == B.shape[1], f"{enc}: dim {A.shape[1]} vs {B.shape[1]}"
        merged = np.lib.format.open_memmap(d_out / "residue_fp16.npy", mode="w+",
                                           dtype=A.dtype, shape=(n_res, A.shape[1]))
        merged[:len(g_rel["rows"])] = A[g_rel["rows"]]
        merged[len(g_rel["rows"]):] = B[g_coh["rows"]]
        merged.flush()
        pa = np.load(REL_EMB / enc / "pooled_fp32.npy", mmap_mode="r")
        pb = np.load(COH_EMB / enc / "pooled_fp32.npy", mmap_mode="r")
        np.save(d_out / "pooled_fp32.npy", np.concatenate([np.asarray(pa)[rel_idx], np.asarray(pb)]))
        meta = json.loads((COH_EMB / enc / "meta.json").read_text())
        meta.update(dataset=str((OUT / "labels.h5").relative_to(ROOT)),
                    merged_from=["probing_embeddings/" + enc, "confirmation/embeddings/" + enc],
                    row_convention="row r of residue_fp16.npy is row r of residue/* in the merged labels.h5",
                    n_domains=int(len(L)))
        (d_out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"  [{enc}] merged: {n_res:,} x {A.shape[1]}")
    rel.close(); coh.close()


if __name__ == "__main__":
    main()
