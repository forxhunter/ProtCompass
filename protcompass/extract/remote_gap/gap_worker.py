"""Embeddings on a second GPU machine for encoder and dataset pairs that were missing or wrong on the first.

The worker uses extract/common.py:Encoder and the wrappers in extract/patched/ unchanged, apart from the root path, and
calls Encoder.residue with the same arguments as extract/extract.py. It writes arrays only; the first machine checks a
sample against its own computation and writes meta.json (finalize_gap.py).

  python gap_worker.py --encoder saprot --set masif               # residue_fp16.npy, pooled_fp32.npy
  python gap_worker.py --encoder foldvision --set remote_homology  # pooled.npy, one row per manifest row
  python gap_worker.py --encoder proteinmpnn --set ds_stability   # pooled.npy for a downstream task, from coordinates
  python gap_worker.py --encoder proteinmpnn --set probing        # residue_fp16.npy, pooled_fp32.npy on the probing set

Sets (data/<set>.npz, written on the first machine): residue-level sets `masif`, `probing`, `cohortA`, `cohortB`
(ids, lengths, offsets, coords, aa); pooled sets `ds_<task>`, `ho_<task>` and `remote_homology` (ids, sequence, xyz,
offsets, and `pdb` file names where the encoder reads a structure file from data/rh_pdb/).

A wrapper that offers `residue_many` (GearNet) is called on several chains at once. The chains are centred as
Encoder.residue centres them, and the wrapper builds edges within each chain, so the result equals the one-chain call.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "2026_encoder" / "resubmission" / "extract"))
from common import Encoder  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
RESIDUE_SETS = ("masif", "probing", "cohortA", "cohortB")
PACK_RESIDUES, PACK_CHAINS = 3000, 16


def packs(lengths):
    """Consecutive index groups of at most PACK_CHAINS chains and PACK_RESIDUES residues."""
    out, cur, tot = [], [], 0
    for i, n in enumerate(lengths):
        if cur and (tot + n > PACK_RESIDUES or len(cur) == PACK_CHAINS):
            out.append(cur); cur, tot = [], 0
        cur.append(i); tot += n
    if cur:
        out.append(cur)
    return out


def centred(c):
    c = c.astype(np.float32)
    return (c - c[:, 1].mean(0, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--set", required=True, help="masif, probing, cohortA, cohortB, remote_homology, ds_<task> or ho_<task>")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = HERE / "out" / a.set / a.encoder
    if (out / "done.json").exists():
        print(f"[{a.encoder}/{a.set}] already done"); return
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(HERE / "data" / f"{a.set}.npz")
    t0 = time.time()
    enc = Encoder(a.encoder)
    many = getattr(enc.impl, "residue_many", None)
    residue_set = a.set in RESIDUE_SETS
    if residue_set:
        ids, lengths, offsets = z["ids"], z["lengths"].astype(np.int64), z["offsets"].astype(np.int64)
        coords, aa = z["coords"], z["aa"]
        seq_of = lambda i: "".join(ALPHABET[k] for k in aa[int(offsets[i]):int(offsets[i]) + int(lengths[i])])
        xyz_of = lambda i: coords[int(offsets[i]):int(offsets[i]) + int(lengths[i])].astype(np.float32)
        pdb_of = lambda i: str(HERE / "data" / "structures" / f"{ids[i]}.pdb") if a.set == "masif" else None
        size = lambda i: int(lengths[i])
    else:
        ids, seqs, xyz, off = z["ids"], z["sequence"], z["xyz"], z["offsets"]
        files = z["pdb"] if "pdb" in z.files else None
        seq_of = lambda k: str(seqs[k])
        xyz_of = lambda k: xyz[off[k]:off[k + 1]]
        pdb_of = lambda k: str(HERE / "data" / "rh_pdb" / str(files[k])) if files is not None else None
        size = lambda k: int(off[k + 1] - off[k])
    n = a.limit or len(ids)
    groups = packs([size(i) for i in range(n)]) if many else [[i] for i in range(n)]
    emb = pooled = None
    done = 0
    for group in groups:
        if many:
            outs = many([(seq_of(i), centred(xyz_of(i))) for i in group])
        else:
            outs = [enc.residue(seq_of(i), coords=xyz_of(i), pdb_path=pdb_of(i)) for i in group]
        for i, h in zip(group, outs):
            L = size(i)
            assert np.isfinite(h).all() and np.abs(h).max() < 6.5e4, f"{ids[i]}: non-finite or float16 overflow"
            assert h.shape[-2] == L, f"{ids[i]}: {h.shape[-2]} residue rows for {L} residues"
            if pooled is None:
                pooled = np.zeros((len(ids), h.shape[-1]), dtype=np.float32)
                if residue_set:
                    emb = np.lib.format.open_memmap(out / "residue_fp16.npy.partial", mode="w+", dtype=np.float16,
                                                    shape=(int(lengths.sum()), h.shape[-1]))
            if residue_set:
                o = int(offsets[i]); emb[o:o + L] = h.astype(np.float16)
            pooled[i] = h.mean(0)
            done += 1
            if done % 2000 == 0:
                print(f"  [{a.encoder}] {done}/{n}, {time.time() - t0:.0f}s", flush=True)
    if residue_set:
        emb.flush()
    if a.limit:
        np.save(out / f"pooled_first{n}.npy", pooled[:n]); print("limit run finished"); return
    if residue_set:
        (out / "residue_fp16.npy.partial").rename(out / "residue_fp16.npy")
        np.save(out / "pooled_fp32.npy", pooled)
    else:
        np.save(out / "pooled.npy", pooled)
    import torch
    meta = dict(enc.finalize_meta(), seconds=round(time.time() - t0, 1), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                packed_chains=bool(many))
    (out / "done.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[{a.encoder}/{a.set}] done in {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
