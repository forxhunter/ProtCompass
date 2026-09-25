"""Atom-level input audit: which of the five stored atoms (N, CA, C, O, CB) each encoder reads, and whether it reads
the absolute scale of the coordinates.

The released audit (input_audit.py) decides whether an encoder reads coordinates and residue identity at all. The
floors, though, are built from particular atoms: `ref_coords` and `ref_pairhist` from CA alone, `ref_backbone` and
`ref_global` from all five. Holding an encoder to a floor built from atoms it never receives would overstate the
floor, so this audit measures the atom set directly. For the 32 proteins of the released audit the pooled embedding is
recomputed with one atom type moved at a time by Gaussian noise of 1 A on every residue, and with every coordinate
scaled by 1.05 about the CA centroid; an input counts as read if the median relative change ||e' - e|| / ||e|| exceeds
the released threshold of 1e-3. Encoders that read a structure file get a file written from the perturbed coordinates,
as in the released audit.

Run once per conda environment; each run adds its encoders to encoder_atom_audit.csv.

  python atom_audit.py --encoders esm_if1 proteinmpnn ...
"""
import argparse
import fcntl
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import HERE, SPEC, SPECS, Encoder
from input_audit import THRESH, audit_proteins, write_backbone_pdb

OUT = HERE / "encoder_atom_audit.csv"
ATOMS = ["N", "CA", "C", "O", "CB"]
SCALE = 1.05


def audit(key):
    enc = Encoder(key)
    prots = audit_proteins("probing")
    rng = np.random.default_rng(2)
    tmp = Path(tempfile.mkdtemp(prefix=f"atom_audit_{key}_"))

    def pooled(seq, xyz, tag):
        path = tmp / f"{tag}.pdb"
        write_backbone_pdb(path, seq, xyz)
        return enc.residue(seq, coords=xyz, pdb_path=str(path)).mean(0)

    change = {k: [] for k in ATOMS + ["scale"]}
    t0 = time.time()
    for seq, xyz in prots:
        base = pooled(seq, xyz, "base")
        nb = np.linalg.norm(base) + 1e-12
        for a, name in enumerate(ATOMS):
            x2 = xyz.copy()
            x2[:, a] += rng.normal(scale=1.0, size=(len(seq), 3)).astype(np.float32)
            change[name].append(float(np.linalg.norm(pooled(seq, x2, name) - base) / nb))
        centre = xyz[:, 1].mean(0, keepdims=True)
        change["scale"].append(float(np.linalg.norm(pooled(seq, (xyz - centre) * SCALE + centre, "scale") - base) / nb))
    med = {k: float(np.median(v)) for k, v in change.items()}
    row = {"encoder": key, "n_proteins": len(prots), **{f"rel_change_{k}": v for k, v in med.items()},
           **{f"reads_{k}": med[k] > THRESH for k in ATOMS}, "reads_scale": med["scale"] > THRESH,
           "atom_set": "+".join(k for k in ATOMS if med[k] > THRESH) or "none", "seconds": round(time.time() - t0, 1)}
    with open(str(OUT) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
        if len(old):
            old = old[old.encoder != key]
        pd.concat([old, pd.DataFrame([row])]).sort_values("encoder").to_csv(OUT, index=False)
    print(f"[{key}] atoms {row['atom_set']}, scale {row['reads_scale']}  "
          + " ".join(f"{k}={v:.1e}" for k, v in med.items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=[s.key for s in SPECS if s.env == "protein_embeddings"])
    a = ap.parse_args()
    for k in a.encoders:
        if not SPEC[k].uses_coords:
            print(f"[{k}] reads no coordinates (released audit); skipped", flush=True)
            continue
        audit(k)
