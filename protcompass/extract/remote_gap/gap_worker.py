"""Embeddings on a second GPU machine for encoder and dataset pairs that were missing on the first.

The worker uses extract/common.py:Encoder and the wrappers in extract/patched/ unchanged, apart from the root path, and
calls Encoder.residue with the same arguments as extract/extract.py. It writes arrays only; the first machine checks a
sample against its own computation and writes meta.json (finalize_gap.py).

  python gap_worker.py --encoder saprot --set masif               # residue_fp16.npy, pooled_fp32.npy
  python gap_worker.py --encoder foldvision --set remote_homology  # pooled.npy, one row per manifest row

Datasets (data/): masif.npz and remote_homology.npz as written on the first machine, structure files under
data/structures/ and data/rh_pdb/.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--set", required=True, choices=["masif", "remote_homology"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = HERE / "out" / a.set / a.encoder
    if (out / "done.json").exists():
        print(f"[{a.encoder}/{a.set}] already done"); return
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(HERE / "data" / f"{a.set}.npz")
    t0 = time.time()
    enc = Encoder(a.encoder)
    if a.set == "masif":
        ids, lengths, offsets = z["ids"], z["lengths"].astype(np.int64), z["offsets"].astype(np.int64)
        coords, aa = z["coords"], z["aa"]
        n = a.limit or len(ids)
        emb = pooled = None
        for i in range(n):
            o, L = int(offsets[i]), int(lengths[i])
            seq = "".join(ALPHABET[k] for k in aa[o:o + L])
            h = enc.residue(seq, coords=coords[o:o + L].astype(np.float32), pdb_path=str(HERE / "data" / "structures" / f"{ids[i]}.pdb"))
            assert np.isfinite(h).all() and np.abs(h).max() < 6.5e4, f"{ids[i]}: non-finite or float16 overflow"
            assert h.shape[-2] == L, f"{ids[i]}: {h.shape[-2]} residue rows for {L} labelled residues"
            if emb is None:
                emb = np.lib.format.open_memmap(out / "residue_fp16.npy.partial", mode="w+", dtype=np.float16,
                                                shape=(int(lengths.sum()), h.shape[-1]))
                pooled = np.zeros((len(ids), h.shape[-1]), dtype=np.float32)
            emb[o:o + L] = h.astype(np.float16); pooled[i] = h.mean(0)
            if (i + 1) % 500 == 0:
                print(f"  [{a.encoder}] {i + 1}/{n} chains, {time.time() - t0:.0f}s", flush=True)
        emb.flush()
        if a.limit:
            np.save(out / f"pooled_first{n}.npy", pooled[:n]); print("limit run finished"); return
        (out / "residue_fp16.npy.partial").rename(out / "residue_fp16.npy")
        np.save(out / "pooled_fp32.npy", pooled)
    else:
        ids, seqs, files, xyz, offsets = z["ids"], z["sequence"], z["pdb"], z["xyz"], z["offsets"]
        n = a.limit or len(ids)
        X = None
        for k in range(n):
            h = enc.residue(str(seqs[k]), coords=xyz[offsets[k]:offsets[k + 1]], pdb_path=str(HERE / "data" / "rh_pdb" / str(files[k])))
            if X is None:
                X = np.zeros((len(ids), h.shape[1]), dtype=np.float32)
            X[k] = h.mean(0)
            if (k + 1) % 500 == 0:
                print(f"  [{a.encoder}] {k + 1}/{n}", flush=True)
        if a.limit:
            np.save(out / f"pooled_first{n}.npy", X[:n]); print("limit run finished"); return
        np.save(out / "pooled.npy", X)
    import torch
    meta = dict(enc.finalize_meta(), seconds=round(time.time() - t0, 1), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    (out / "done.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[{a.encoder}/{a.set}] done in {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
