"""A correctly indexed residue-pair task, to test the one regime the pooled downstream suite cannot reach.

Both tasks in the original suite that used a residue-level or residue-pair readout were removed as unusable: contact
prediction was computed from CA-only coordinates indexed as though they were N, CA and C, and interface-site
prediction was labelled by chain membership. Everything that survived reads a pooled protein-level representation,
so the downstream comparison of Section 4.2 cannot say whether a coordinate encoder carries residue-resolved
structure that pooling discards. That is the regime such encoders are built for, and it is the single
addition that would most change what the downstream side can conclude.

Nothing new has to be extracted. The rebuilt probing dataset already carries the pair labels, sampled at 512 pairs
per domain with a minimum sequence separation of 6, and residue-level embeddings are cached for every encoder. A
contact is a CA-CA distance below 8 A, which is 2.83% of the sampled pairs and is the same rate in all three splits.
The split is the released superfamily-disjoint one and the protein subsample is the released seed-0 draw, so this
task sits on the same footing as every probe in the suite.

The floors are deliberately absent. `ref_coords`, `ref_backbone`, `ref_global` and `ref_pairhist` are built from
CA distances, so each contains the contact label by construction and a floor comparison here would be tautological
in the sense Section 4.1 already defines. What this task measures is the comparison between encoder groups, read
against the base rate and against sequence separation alone.

  python t8_contact_task.py [--encoders esm2_650m ...] [--max-pairs 1500000]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
EMB, OUT = ROOT / "probing_embeddings", ROOT / "contact_task"
LABELS = ROOT / "probing_data" / "labels.h5"
SPLIT, N_PROT, PARTS = "superfamily", {"train": 3000, "val": 500, "test": 1000}, {"train": 0, "val": 1, "test": 2}
CONTACT_A = 8.0
HIDDEN, MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 128, 30, 3, 2048, 1e-3, 1e-4
EVAL_CHUNK = 8192          # small enough to share the card with another job
SEEDS = (0, 1, 2)
PANEL = ["esm2_650m", "prottrans_t5_xl_u50", "ankh_base", "esm3", "esm_if1", "proteinmpnn"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairHead(nn.Module):
    """A residue projection shared by both positions, then a head on the symmetric pair summary.

    The pair features are the sum and the absolute difference of the two projected residues, so the head cannot use
    the order of the pair, and a separation feature, because contact probability depends on |i-j| and a model that
    could not see it would be handicapped relative to the base rate.
    """

    def __init__(self, d_in, hidden=HIDDEN, linear=False):
        super().__init__()
        self.proj = nn.Identity() if linear else nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        d = d_in if linear else hidden
        self.out = nn.Linear(2 * d + 1, 1) if linear else nn.Sequential(
            nn.Linear(2 * d + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, zi, zj, sep):
        a, b = self.proj(zi), self.proj(zj)
        return self.out(torch.cat([a + b, (a - b).abs(), sep], -1)).squeeze(-1)


def load_pairs(max_pairs, rng):
    """Pair indices, labels and separations per split, on the released protein subsample."""
    h = h5py.File(LABELS, "r")
    P, Q = h["protein"], h["pair"]
    split = P[f"split_{SPLIT}"][:]
    offs = P["offset"][:].astype(np.int64)
    r = np.random.default_rng(0)                                   # the released draw
    chosen = {p: np.sort(r.choice(np.where(split == c)[0],
                                  size=min(N_PROT[p], int((split == c).sum())), replace=False))
              for p, c in PARTS.items()}
    qp, qi, qj = Q["protein_index"][:], Q["i"][:].astype(np.int64), Q["j"][:].astype(np.int64)
    dist = Q["distance"][:]
    out = {}
    for p in PARTS:
        sel = np.where(np.isin(qp, chosen[p]))[0]
        if max_pairs and len(sel) > max_pairs:
            sel = np.sort(rng.choice(sel, size=max_pairs, replace=False))
        gi = offs[qp[sel]] + qi[sel]
        gj = offs[qp[sel]] + qj[sel]
        y = (dist[sel] < CONTACT_A).astype(np.float32)
        sep = np.log1p(np.abs(qi[sel] - qj[sel])).astype(np.float32)[:, None]
        out[p] = dict(gi=gi, gj=gj, y=y, sep=sep, n_prot=len(chosen[p]))
    h.close()
    return out


def score_encoder(key, pairs, linear):
    f = EMB / key / "residue_fp16.npy"
    if not f.exists():
        return None
    X = np.load(f, mmap_mode="r")
    d_in = X.shape[1]
    data = {}
    for p, v in pairs.items():
        zi = torch.from_numpy(np.asarray(X[v["gi"]], dtype=np.float32))
        zj = torch.from_numpy(np.asarray(X[v["gj"]], dtype=np.float32))
        mu, sd = zi.mean(0, keepdim=True), zi.std(0, keepdim=True).clamp(min=1e-6)
        data[p] = ((zi - mu) / sd, (zj - mu) / sd, torch.from_numpy(v["sep"]), torch.from_numpy(v["y"]))
    aps = []
    for seed in (SEEDS if not linear else (0,)):
        torch.manual_seed(seed)
        net = PairHead(d_in, linear=linear).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        best, bad, best_state = -1.0, 0, None
        n = len(data["train"][3])
        for _ in range(MAX_EPOCHS):
            net.train()
            perm = torch.randperm(n)
            for s in range(0, n, BATCH):
                b = perm[s:s + BATCH]
                zi, zj, sep, y = (t[b].to(DEV) for t in data["train"])
                loss = nn.functional.binary_cross_entropy_with_logits(net(zi, zj, sep), y)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                zi, zj, sep, y = data["val"]
                pv = torch.cat([net(zi[s:s + EVAL_CHUNK].to(DEV), zj[s:s + EVAL_CHUNK].to(DEV), sep[s:s + EVAL_CHUNK].to(DEV)).cpu()
                                for s in range(0, len(y), EVAL_CHUNK)]).numpy()
            v = average_precision_score(y.numpy(), pv)
            if v > best + 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            zi, zj, sep, y = data["test"]
            pt = torch.cat([net(zi[s:s + EVAL_CHUNK].to(DEV), zj[s:s + EVAL_CHUNK].to(DEV), sep[s:s + EVAL_CHUNK].to(DEV)).cpu()
                            for s in range(0, len(y), EVAL_CHUNK)]).numpy()
        aps.append(float(average_precision_score(y.numpy(), pt)))
    return float(np.mean(aps)), float(np.std(aps)) if len(aps) > 1 else 0.0, d_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=PANEL)
    ap.add_argument("--max-pairs", type=int, default=1_500_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pairs = load_pairs(a.max_pairs, rng)
    base = float(pairs["test"]["y"].mean())
    sep_only = float(average_precision_score(pairs["test"]["y"], pairs["test"]["sep"][:, 0]))
    print(f"device {DEV}; pairs train {len(pairs['train']['y']):,} val {len(pairs['val']['y']):,} "
          f"test {len(pairs['test']['y']):,}")
    print(f"base rate {base:.4f}; sequence separation alone, AUPRC {sep_only:.4f}\n", flush=True)

    rows = []
    csv = OUT / "contact_auprc.csv"
    if csv.exists():
        rows = pd.read_csv(csv).to_dict("records")
    done = {(r["encoder"], r["readout"]) for r in rows}
    for enc in a.encoders:
        for linear in (True, False):
            tag = "linear" if linear else "mlp2"
            if (enc, tag) in done:
                continue
            r = score_encoder(enc, pairs, linear)
            if r is None:
                print(f"  [skip] {enc}: no residue embeddings", flush=True)
                continue
            ap_, sd, dim = r
            rows.append(dict(encoder=enc, readout=tag, dim=dim, auprc=ap_, auprc_sd=sd,
                             base_rate=base, sep_only=sep_only))
            print(f"  {enc:22s} {tag:7s} dim {dim:5d}  AUPRC {ap_:.4f} +- {sd:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False)
    d = pd.DataFrame(rows)
    d.to_csv(csv, index=False)
    json.dump(dict(base_rate=base, sep_only=sep_only, contact_angstrom=CONTACT_A, min_separation=6,
                   split=SPLIT, n_proteins=N_PROT, max_pairs=a.max_pairs),
              open(OUT / "protocol.json", "w"), indent=2)
    print(f"\nwritten: {csv}")


if __name__ == "__main__":
    main()
