"""Fill the dMaSIF cache from the worklist of build_worklist.py on a machine without the structure files.

  python worker.py --gate                 # recompute the gate sample and compare with the embeddings stored on the first machine
  python worker.py --part 0 --of 2        # embed the shards with index % 2 == 0

Environment: DMASIF_REPO, DMASIF_REDUCE, DMASIF_CACHE, DMASIF_TMP as read by patched/dmasif_official.py.
Each record set is checked against its SHA-1 key before use, so the cache entry is the one the first machine expects.
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from patched.dmasif_official import SEED, DMaSIFOfficial  # noqa: E402


def records(z):
    off = z["offsets"]
    res, name, resname, el, xyz = z["res"], z["name"], z["resname"], z["el"], z["xyz"]
    for k, key in enumerate(z["keys"]):
        s = slice(int(off[k]), int(off[k + 1]))
        rows = [(int(a), str(b), str(c), str(d), float(x), float(y), float(w))
                for a, b, c, d, (x, y, w) in zip(res[s], name[s], resname[s], el[s], xyz[s])]
        if hashlib.sha1(repr((SEED, rows)).encode()).hexdigest() != str(key):
            raise RuntimeError(f"record {key} does not reproduce its key")
        yield str(key), rows, int(z["n_res"][k])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    enc = DMaSIFOfficial("cuda")
    cache = Path(os.environ["DMASIF_CACHE"])
    t0 = time.time()
    if a.gate:
        z = np.load(HERE / "shards" / "gate.npz")
        eo, worst = z["emb_offsets"], 0.0
        for k, (key, rows, n_res) in enumerate(records(z)):
            f = cache / key[:2] / f"{key}.npz"
            if f.exists():
                f.unlink()                                  # the gate always recomputes
            new, ref = enc.residue_from_rows(rows, n_res), z["emb"][eo[k]:eo[k + 1]]
            d = float(np.abs(new - ref).max())
            worst = max(worst, d)
            print(f"gate {k:3d} {key[:10]} residues {n_res:5d} max_abs_diff {d:.3e} ref_max_abs {float(np.abs(ref).max()):.3f}", flush=True)
        print(f"gate finished: {k + 1} proteins, worst max_abs_diff {worst:.3e}, {time.time() - t0:.0f}s", flush=True)
        sys.exit(0)
    shards = sorted((HERE / "shards").glob("work_*.npz"))
    mine = [s for i, s in enumerate(shards) if i % a.of == a.part]
    done = 0
    for s in mine:
        for key, rows, n_res in records(np.load(s)):
            if not (cache / key[:2] / f"{key}.npz").exists():
                enc.residue_from_rows(rows, n_res)
            done += 1
        print(f"{time.strftime('%H:%M:%S')} part {a.part}/{a.of}: {s.name} finished, {done} structures, {time.time() - t0:.0f}s", flush=True)
    print(f"part {a.part}/{a.of} finished: {done} structures", flush=True)
