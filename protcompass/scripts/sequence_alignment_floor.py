"""The sequence-side floor the coordinate side already had: nearest-neighbour label transfer by sequence alignment.

This script answers the objection that decides this paper. The floors were not built with equal effort on the
two sides. The coordinate side has six descriptors, two of them added later, including a
structural-alignment nearest neighbour. The sequence side has four, which the paper itself calls trivial. The
headline is that the two groups separate in opposite directions, and the obvious competing explanation is that the
two sides received different amounts of design effort.

This builds the exact analogue of the structural-alignment floor on the sequence side. MMseqs2 searches each test
domain against the training domains of the same split and the label of the best hit is the prediction: for a
protein-level label, the hit's own label; for a residue-level label, the label of the residue the alignment maps
that position onto, so a residue with no aligned counterpart is a miss rather than a guess. Nothing is fitted.

It is the same 1-NN rule the structural floor uses, and it is applied to every probe in the comparison, not only
the ones where we expect it to win.

  python sequence_alignment_floor.py --split superfamily
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "probing_results"
SCRATCH = Path(os.environ.get("SEQALIGN_SCRATCH",
                              (PROTCOMPASS_ROOT + "/protcompass/cache/scratch/seqalign")))
MMSEQS = Path(os.environ.get("MMSEQS", SCRATCH.parent / "mmseqs" / "bin" / "mmseqs"))
AA = "ACDEFGHIKLMNPQRSTVWY"

# probe -> (dataset in labels.h5, level, kind). The labels are the ones the floor comparison uses.
PROBES = {
    "phi_circular": ("residue/phi", "residue", "circular"),
    "psi_circular": ("residue/psi", "residue", "circular"),
    "local_curvature": ("residue/curvature", "residue", "reg"),
    "dssp_3": ("residue/dssp3", "residue", "clf"),
    "dssp_8": ("residue/dssp8", "residue", "clf"),
    "rsa": ("residue/rsa", "residue", "reg"),
    "rsa_bins": ("residue/rsa_bin", "residue", "clf"),
    "bfactor": ("residue/bfactor_z", "residue", "reg"),
    "core_surface": ("residue/core", "residue", "clf"),
    "contact_density": ("residue/contact_density_bin", "residue", "clf"),
}
PROTEIN_PROBES = {
    "radius_of_gyration_len_resid": "protein/radius_of_gyration_len_resid_{split}",
    "asphericity_len_resid": "protein/asphericity_len_resid_{split}",
    "contact_order_len_resid": "protein/contact_order_len_resid_{split}",
    "architecture_label": "protein/architecture_label_{split}",
}


def sequences(h):
    aa, offs, L = h["residue/aa"][:], h["protein/offset"][:], h["protein/length"][:]
    idx = np.clip(aa, 0, 19)
    letters = np.array(list(AA))[idx]
    return [("".join(letters[int(o):int(o) + int(l)])) for o, l in zip(offs, L)]


def write_fasta(path, ids, seqs, keep):
    with open(path, "w") as f:
        for i in keep:
            f.write(f">{ids[i]}\n{seqs[i]}\n")


def search(split, ids, seqs, sel, tag):
    base = SCRATCH / f"{split}_{tag}"
    base.mkdir(parents=True, exist_ok=True)
    q, t = base / "query.fasta", base / "target.fasta"
    write_fasta(q, ids, seqs, sel["test"])
    write_fasta(t, ids, seqs, sel["train"])
    aln = base / "aln.tsv"
    if not aln.exists() or aln.stat().st_size == 0:
        cmd = [str(MMSEQS), "easy-search", str(q), str(t), str(aln), str(base / "tmp"),
               "-s", "7.5", "-e", "1000", "--max-seqs", "300", "--threads", "16", "-a",
               "--format-output", "query,target,fident,evalue,bits,qstart,qend,tstart,tend,qaln,taln"]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-1500:], p.stderr[-1500:])
            raise SystemExit("mmseqs failed")
        print(f"  mmseqs: {time.time() - t0:.0f}s", flush=True)
    d = pd.read_csv(aln, sep="\t", names=["query", "target", "fident", "evalue", "bits",
                                          "qstart", "qend", "tstart", "tend", "qaln", "taln"])
    return d.sort_values("bits", ascending=False).drop_duplicates("query")


def residue_map(best, pos):
    """For each (query, query position) the aligned target position, from the alignment strings."""
    out = {}
    for r in best.itertuples():
        qi, ti = int(r.qstart) - 1, int(r.tstart) - 1
        m = {}
        for cq, ct in zip(r.qaln, r.taln):
            if cq != "-" and ct != "-":
                m[qi] = ti
            qi += 0 if cq == "-" else 1
            ti += 0 if ct == "-" else 1
        out[r.query] = (r.target, m)
    return out


def score_probe(kind, truth, pred, n_total):
    """Two numbers per probe, because they answer different questions.

    `covered` counts every test position the probe is scored on, so a position the alignment cannot reach is a
    failure. That is the number comparable with a floor or an encoder, both of which produce a prediction
    everywhere. `on_aligned` is the same metric restricted to the positions that do get an aligned counterpart,
    which separates "the search found nothing" from "the transfer is wrong when it finds something"."""
    from scipy.stats import spearmanr
    ok = np.isfinite(truth) & np.isfinite(pred)
    if ok.sum() < 10:
        return np.nan, np.nan, int(ok.sum())
    t, p = truth[ok], pred[ok]
    if kind == "clf":
        hit = float((t == p).sum())
        return hit / n_total, hit / len(t), int(ok.sum())
    if kind == "circular":
        c = float(np.cos(t - p).sum())
        return c / n_total, c / len(t), int(ok.sum())
    r = float(spearmanr(t, p).statistic)
    return r * (len(t) / n_total), r, int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["superfamily", "topology"])
    ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    h = h5py.File(ROOT / "probing_data" / "labels.h5", "r")
    ids = [x.decode() for x in h["protein/id"][:]]
    offs, L = h["protein/offset"][:], h["protein/length"][:]
    seqs = sequences(h)
    rows = []
    for split in a.splits:
        s = h[f"protein/split_{split}"][:]
        sel = {p: np.where(s == c)[0] for p, c in R.PARTS.items()}
        print(f"{split}: {len(sel['train'])} train, {len(sel['test'])} test", flush=True)
        best = search(split, ids, seqs, sel, a.tag)
        print(f"  {len(best)} of {len(sel['test'])} test proteins have a hit; median identity "
              f"{best.fident.median():.3f}, upper quartile {best.fident.quantile(0.75):.3f}", flush=True)
        rows.append(dict(split=split, source="ref_seq_align_1nn", probe="_coverage", level="protein",
                         score=len(best) / len(sel["test"]), score_on_aligned=float(best.fident.median()),
                         n_scored=len(best), n_total=len(sel["test"])))
        pos = {ids[i]: i for i in range(len(ids))}
        amap = residue_map(best, pos)

        for probe, (ds, level, kind) in PROBES.items():
            v = h[ds][:]
            truth, pred, n_total = [], [], 0
            for i in sel["test"]:
                qid = ids[i]
                n = int(L[i]); n_total += n
                if qid not in amap:
                    continue
                tid, m = amap[qid]
                j = pos[tid]
                qo, to = int(offs[i]), int(offs[j])
                for qp, tp in m.items():
                    if 0 <= tp < int(L[j]):
                        truth.append(v[qo + qp]); pred.append(v[to + tp])
            truth, pred = np.asarray(truth, dtype=np.float64), np.asarray(pred, dtype=np.float64)
            sc, cond, n_ok = score_probe(kind, truth, pred, n_total)
            rows.append(dict(split=split, source="ref_seq_align_1nn", probe=probe, level=level,
                             score=sc, score_on_aligned=cond, n_scored=n_ok, n_total=n_total))
            print(f"    {probe:28s} covered {sc:+.4f}  on aligned {cond:+.4f}  "
                  f"({n_ok} of {n_total} positions)", flush=True)

        for probe, path in PROTEIN_PROBES.items():
            ds = path.format(split=split)
            if ds not in h:
                continue
            v = h[ds][:]
            kind = "clf" if probe == "architecture_label" else "reg"
            truth, pred = [], []
            for i in sel["test"]:
                qid = ids[i]
                if qid not in amap:
                    continue
                truth.append(v[i]); pred.append(v[pos[amap[qid][0]]])
            truth, pred = np.asarray(truth, dtype=np.float64), np.asarray(pred, dtype=np.float64)
            sc, cond, n_ok = score_probe(kind, truth, pred, len(sel["test"]))
            rows.append(dict(split=split, source="ref_seq_align_1nn", probe=probe, level="protein",
                             score=sc, score_on_aligned=cond, n_scored=n_ok, n_total=len(sel["test"])))
            print(f"    {probe:28s} covered {sc:+.4f}  on aligned {cond:+.4f}  "
                  f"({n_ok} of {len(sel['test'])} proteins)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "sequence_alignment_floor.csv", index=False)
    print(f"\nwritten: {OUT / 'sequence_alignment_floor.csv'}")


if __name__ == "__main__":
    main()
