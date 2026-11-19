"""List the proteins of the unfinished dMaSIF extractions that are not in the local cache and write their atom records.

The local card was unavailable on 2026-11-18, so the remaining proteins are embedded on a second machine. This script
needs no GPU. It parses every structure of the given tasks exactly as patched/dmasif_official.py does, computes the cache
key, and writes the records of uncached proteins to shards/work_XXXX.npz. A sample of cached proteins goes to
shards/gate.npz, with their stored embeddings, for the cross-machine check in worker.py --gate.

  python build_worklist.py probing fluorescence stability ec_classification membrane_soluble subcellular_localization
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXTRACT = HERE.parent
sys.path.insert(0, str(EXTRACT))
from patched.dmasif_official import SEED, read_chain  # noqa: E402

CACHE = EXTRACT / "cache_dmasif"
SHARD, N_GATE = 400, 60


def task_inputs(task):
    if task == "probing":
        sys.path.insert(0, str(EXTRACT.parent / "probing_data"))
        from probing_dataset import ProbingData
        d = ProbingData()
        return [(str(Path(d.h.filename).parent / "structures" / f"{i}.pdb"), int(n)) for i, n in zip(d.ids, d.length)]
    man = pd.read_csv(EXTRACT / "manifest" / f"{task}.csv")
    man = man[man.stage1 == 1]
    return [(p, len(s)) for p, s in zip(man.structure_path, man.sequence)]


def parse(item):
    path, length = item
    rows, n_res = read_chain(path)
    if n_res != length:
        raise RuntimeError(f"{path}: {n_res} residues in the file for a sequence of length {length}")
    if sum(r[1] == "CA" for r in rows) != n_res:
        raise RuntimeError(f"{path}: CA count differs from {n_res} residues")
    key = hashlib.sha1(repr((SEED, rows)).encode()).hexdigest()
    return key, rows, n_res


def write(path, items, extra=None):
    keys = np.array([k for k, _, _ in items])
    n_res = np.array([n for _, _, n in items], dtype=np.int32)
    off = np.concatenate([[0], np.cumsum([len(r) for _, r, _ in items])]).astype(np.int64)
    flat = [a for _, r, _ in items for a in r]
    np.savez_compressed(path, keys=keys, n_res=n_res, offsets=off,
                        res=np.array([a[0] for a in flat], dtype=np.int32), name=np.array([a[1] for a in flat]),
                        resname=np.array([a[2] for a in flat]), el=np.array([a[3] for a in flat]),
                        xyz=np.array([a[4:] for a in flat], dtype=np.float64), **(extra or {}))


if __name__ == "__main__":
    out = HERE / "shards"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.npz"):
        f.unlink()
    todo, cached, seen = [], [], set()
    for task in sys.argv[1:]:
        inputs = task_inputs(task)
        n_new = n_hit = 0
        with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:   # the cores the queue assigned
            for key, rows, n_res in pool.map(parse, inputs, chunksize=64):
                if key in seen:
                    continue
                seen.add(key)
                if (CACHE / key[:2] / f"{key}.npz").exists():
                    n_hit += 1; cached.append((key, rows, n_res))
                else:
                    n_new += 1; todo.append((key, rows, n_res))
        print(f"{task}: {len(inputs)} rows, {n_hit} distinct structures cached, {n_new} to compute", flush=True)
    todo.sort(key=lambda t: len(t[1]))                      # similar sizes per shard; the two workers interleave shards
    for s in range(0, len(todo), SHARD):
        write(out / f"work_{s // SHARD:04d}.npz", todo[s:s + SHARD])
    pick = np.random.default_rng(0).choice(len(cached), N_GATE, replace=False)
    gate = [cached[i] for i in pick]
    emb = [np.load(CACHE / k[:2] / f"{k}.npz")["emb"] for k, _, _ in gate]
    write(out / "gate.npz", gate, {"emb": np.concatenate(emb), "emb_offsets": np.concatenate([[0], np.cumsum([len(e) for e in emb])])})
    print(f"total: {len(todo)} structures in {len(list(out.glob('work_*.npz')))} shards, {sum(len(r) for _, r, _ in todo)} atoms; gate sample {N_GATE}")
