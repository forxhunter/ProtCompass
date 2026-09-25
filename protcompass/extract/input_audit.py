"""Input-sensitivity audit: does each encoder respond to the inputs it is declared to use, and only to those?

For 32 proteins (16 equal-length pairs) the pooled embedding is recomputed under
  jitter   Gaussian noise of 1 A on every backbone atom
  swap     the coordinates of another protein of the same length (sequence unchanged)
  shuffle  a permuted sequence (coordinates unchanged)
  rigid    a random rotation and a 10 A translation
and the relative change ||e' - e|| / ||e|| is recorded (median over proteins).

Verdicts: uses_coordinates = swap change > 1e-3; uses_residue_identity = shuffle change > 1e-3; invariant = rigid change below
max(1e-3, 2% of the larger of the swap and jitter changes), which separates numerical noise from frame dependence.
An encoder passes only if the measured use of coordinates and of residue identity equals its declaration in common.SPECS and,
when it uses coordinates, its output is invariant (unless the spec allows a frame-dependent model). extract.py refuses to
save embeddings for an encoder that does not pass.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ALPHABET, HERE, ROOT, SPEC, SPECS, Encoder, load_backbone

AUDIT_CSV = HERE / "encoder_input_audit.csv"
THRESH = 1e-3
ONE_TO_THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
                "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
ALLOW_FRAME_DEPENDENT = {"foldvision"}  # voxel CNN on the raw frame; declared, not repaired


def write_backbone_pdb(path, sequence, coords):
    names, k = ["N", "CA", "C", "O", "CB"], 1
    with open(path, "w") as fh:
        for i, aa in enumerate(sequence):
            for a, name in enumerate(names):
                if name == "CB" and aa == "G":
                    continue
                x, y, z = coords[i, a]
                fh.write(f"ATOM  {k:5d}  {name:<3s} {ONE_TO_THREE[aa]} A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           {name[0]}\n")
                k += 1
        fh.write("END\n")


def audit_proteins(dataset, n_pairs=16, seed=0):
    """Equal-length pairs of real proteins: list of (sequence, coords)."""
    rng = np.random.default_rng(seed)
    pool = []
    if dataset == "probing":
        import sys
        sys.path.insert(0, str(ROOT / "." / "probing_data"))
        from probing_dataset import ProbingData
        data = ProbingData()
        lengths = pd.Series(data.length)
        for L, idx in lengths[(lengths >= 60) & (lengths <= 250)].groupby(lengths).groups.items():
            if len(idx) >= 2:
                pool.append(("probing", list(idx[:2])))
        picks = [pool[i] for i in rng.choice(len(pool), n_pairs, replace=False)]
        out = []
        for _, (i, j) in picks:
            for k in (i, j):
                p = data.protein(int(k))
                out.append(("".join(ALPHABET[a] for a in p["sequence"]), p["coords"].astype(np.float32)))
        return out
    for task, share in (("stability", n_pairs // 2), ("fluorescence", n_pairs - n_pairs // 2)):
        man = pd.read_csv(HERE / "manifest" / f"{task}.csv").query("stage1 == 1 and length <= 300")
        groups = [g for _, g in man.groupby("length") if len(g) >= 2]
        for n, gi in enumerate(rng.choice(len(groups), share, replace=len(groups) < share)):   # GFP variants all share one length
            g = groups[gi].sample(2, random_state=int(seed) + n)
            pool.extend((r.sequence, load_backbone(r.structure_path, r.sequence)[0]) for r in g.itertuples())
    return pool


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return (q * np.sign(np.linalg.det(q))).astype(np.float32)


def run_audit(key, dataset="stage1", enc=None, tmpdir=None):
    spec = SPEC[key]
    enc = enc or Encoder(key)
    prots = audit_proteins(dataset)
    rng = np.random.default_rng(1)
    tmp = Path(tmpdir or tempfile.mkdtemp(prefix=f"audit_{key}_"))

    def pooled(seq, xyz, tag):
        path = tmp / f"{tag}.pdb"
        write_backbone_pdb(path, seq, xyz)
        return enc.residue(seq, coords=xyz, pdb_path=str(path)).mean(0)

    change = {k: [] for k in ("jitter", "swap", "shuffle", "rigid")}
    t0 = time.time()
    for i in range(0, len(prots), 2):
        for a, b in ((i, i + 1), (i + 1, i)):
            seq, xyz = prots[a]
            base = pooled(seq, xyz, "base")
            nb = np.linalg.norm(base) + 1e-12
            perm = rng.permutation(len(seq))
            shuffled = "".join(seq[j] for j in perm)
            variants = {"jitter": (seq, xyz + rng.normal(scale=1.0, size=xyz.shape).astype(np.float32)),
                        "swap": (seq, prots[b][1]),
                        "shuffle": (shuffled, xyz),
                        "rigid": (seq, xyz @ random_rotation(rng).T + np.float32(10.0))}
            for name, (s2, x2) in variants.items():
                if name == "shuffle" and s2 == seq:
                    continue
                change[name].append(float(np.linalg.norm(pooled(s2, x2, name) - base) / nb))
    med = {k: float(np.median(v)) for k, v in change.items()}
    # Reduced-precision models (bfloat16) show rigid-motion noise of order 1e-3; frame dependence is of the order of the swap change.
    uses_c, uses_s, inv = med["swap"] > THRESH, med["shuffle"] > THRESH, med["rigid"] < max(THRESH, 0.02 * max(med["swap"], med["jitter"]))
    problems = []
    if uses_c != spec.uses_coords:
        problems.append(f"declared uses_coordinates={spec.uses_coords}, measured {uses_c}")
    if uses_s != spec.uses_sequence:
        problems.append(f"declared uses_residue_identity={spec.uses_sequence}, measured {uses_s}")
    if uses_c and not inv and key not in ALLOW_FRAME_DEPENDENT:
        problems.append(f"output changes under rigid motion (relative change {med['rigid']:.2e})")
    row = {"encoder": key, "dataset": dataset, "weights": spec.weights, "n_proteins": len(prots),
           "rel_change_jitter_1A": med["jitter"], "rel_change_coord_swap": med["swap"], "rel_change_seq_shuffle": med["shuffle"],
           "rel_change_rigid_motion": med["rigid"], "uses_coordinates": uses_c, "uses_residue_identity": uses_s, "invariant": inv,
           "declared_coordinates": spec.uses_coords, "declared_residue_identity": spec.uses_sequence,
           "verdict": "pass" if not problems else "fail", "problems": "; ".join(problems), "seconds": round(time.time() - t0, 1),
           "inputs_passed": enc.meta.get("inputs_passed", "coordinates, sequence indices and mask via ProteinInput")}
    import fcntl
    with open(str(AUDIT_CSV) + ".lock", "w") as lock:                 # several extraction queues write this table
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = pd.read_csv(AUDIT_CSV) if AUDIT_CSV.exists() else pd.DataFrame()
        if len(old):
            old = old[~((old.encoder == key) & (old.dataset == dataset))]
        pd.concat([old, pd.DataFrame([row])]).to_csv(AUDIT_CSV, index=False)
    return row, enc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=[s.key for s in SPECS if s.env == "protein_embeddings"])
    ap.add_argument("--dataset", default="stage1", choices=["stage1", "probing"])
    a = ap.parse_args()
    for key in a.encoders:
        try:
            row, enc = run_audit(key, a.dataset)
            del enc
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if k not in ("inputs_passed",)}), flush=True)
        except Exception as e:
            print(json.dumps({"encoder": key, "dataset": a.dataset, "verdict": "error", "problems": f"{type(e).__name__}: {str(e)[:300]}"}), flush=True)
