"""Side-chain input audit for the pre-registered determinacy test: does an encoder read side-chain atoms?

The atom audit (atom_audit.py) moves the five stored atoms. Encoders that read the domain file itself could also read
side-chain atoms beyond CB, which decides whether chi1 and chi2 are determined by their input. For the 32 proteins of
the released audit, the pooled embedding of the original full-atom domain file is compared with the embedding of the
same file in which every heavy atom beyond CB is moved by Gaussian noise of 1 A (backbone and CB untouched), and with
an unperturbed copy of the file (the determinism check). The input counts as read if the median relative change
exceeds the released threshold of 1e-3 and the unperturbed copy stays below it.

  python sidechain_audit.py --encoders gcpnet_random gcpnet_pretrained
Output rows are added to encoder_sidechain_audit.csv.
"""
import argparse
import fcntl
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import HERE, ROOT, Encoder
from input_audit import THRESH

OUT = HERE / "encoder_sidechain_audit.csv"
KEEP = {"N", "CA", "C", "O", "CB", "OXT"}


def rewrite(src, dst, rng, sigma):
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() not in KEEP and line[12:16].strip()[0] != "H" and sigma > 0:
                x = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]) + rng.normal(0, sigma, 3)
                line = line[:30] + "".join(f"{v:8.3f}" for v in x) + line[54:]
            g.write(line)


def audit(key):
    sys.path.insert(0, str(ROOT / "." / "probing_data"))
    from probing_dataset import ProbingData
    enc, data = Encoder(key), ProbingData()
    lengths = pd.Series(data.length)
    rng_pick = np.random.default_rng(0)
    pool = lengths[(lengths >= 60) & (lengths <= 250)].index.to_numpy()
    picks = rng_pick.choice(pool, 32, replace=False)
    rng, tmp = np.random.default_rng(3), Path(tempfile.mkdtemp(prefix=f"sc_audit_{key}_"))
    moved, copy = [], []
    for k in picks:
        p = data.protein(int(k))
        seq = p["sequence"] if isinstance(p["sequence"], str) else None
        from common import ALPHABET
        seq = seq or "".join(ALPHABET[a] for a in p["sequence"])
        xyz = p["coords"].astype(np.float32)
        emb = lambda path: enc.residue(seq, coords=xyz, pdb_path=str(path)).mean(0)
        base = emb(p["pdb_path"]); nb = np.linalg.norm(base) + 1e-12
        rewrite(p["pdb_path"], tmp / "copy.pdb", rng, 0.0)
        rewrite(p["pdb_path"], tmp / "moved.pdb", rng, 1.0)
        copy.append(np.linalg.norm(emb(tmp / "copy.pdb") - base) / nb)
        moved.append(np.linalg.norm(emb(tmp / "moved.pdb") - base) / nb)
    row = dict(encoder=key, n=len(picks), median_change_sidechain=float(np.median(moved)),
               median_change_copy=float(np.median(copy)), reads_sidechain=bool(np.median(moved) > THRESH),
               deterministic=bool(np.median(copy) <= THRESH))
    print(row, flush=True)
    with open(OUT.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
        d = pd.concat([d[d.encoder != key] if len(d) else d, pd.DataFrame([row])], ignore_index=True)
        d.to_csv(OUT, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    for e in ap.parse_args().encoders:
        audit(e)
