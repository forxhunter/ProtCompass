"""Precompute the protein-level non-learned floor descriptors for every protein in labels.h5.

`scripts/architecture_all_floors.py` needs the same descriptor matrices for two splits and two class sets. The
coordinate descriptors cost a full pairwise CA distance matrix and a contact-map eigendecomposition per protein, so
they are computed once here, over the whole protein universe, and cached. Each worker writes its own shard file;
nothing is appended to a shared output.

  python precompute_floor_protein_features.py <descriptor> <shard index> <n shards> <out dir>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R


def main():
    which, shard, n_shards, out_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = R.Data()
    n_prot = len(data.L)
    idx = np.array_split(np.arange(n_prot), n_shards)[shard]
    fn = {"global": data.global_features_protein, "coord": data.coord_features_protein,
          "pairhist": data.pairhist_features_protein}[which]
    chunks, done = [], 0
    for block in np.array_split(idx, max(1, len(idx) // 200)):
        chunks.append(fn(block))
        done += len(block)
        print(f"{which} shard {shard}: {done}/{len(idx)}", flush=True)
    X = np.concatenate(chunks).astype(np.float32)
    np.save(out_dir / f"{which}_shard{shard}of{n_shards}.npy", X)
    np.save(out_dir / f"{which}_shard{shard}of{n_shards}_idx.npy", idx)
    print(f"{which} shard {shard} done, shape {X.shape}", flush=True)


if __name__ == "__main__":
    main()
