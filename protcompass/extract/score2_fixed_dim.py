"""The released downstream protocol with every representation projected to one dimension first.

The released downstream table (score2.py) already uses the same two heads for every encoder: a linear probe and a
two-hidden-layer MLP of width 512, selected on validation, three seeds. What still varies with the encoder is the
input dimension, from 20 to 2,560, and with it the parameter count of each head's first layer. This script removes
that: every representation wider than 116 dimensions is projected onto its top 116 principal directions, fitted on
the training split alone, before the released standardisation and heads. 116 is the native width of GVP and GCPNet,
the narrowest encoders after Foldseek (20), and the width of the matched-capacity probing control. Foldseek is left at
20 dimensions and is reported as such.

Nothing else changes: the same splits, labels, heads, optimiser, early stopping, selection rule, seeds and metrics
are imported from score2.py. The output goes to score_fixed_dim/ and never touches the released tables.

  python score2_fixed_dim.py [--tasks ...] [--encoders ...] [--dim 116]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import score2 as S

HERE = Path(__file__).parent
OUT = HERE / "score_fixed_dim"


def project(X, k):
    """Top-k principal directions of the training split, applied to every split. Returns X unchanged when narrower."""
    d = X["train"].shape[1]
    if d <= k:
        return X, d
    mu = X["train"].mean(0, keepdims=True)
    V = np.linalg.svd((X["train"] - mu).astype(np.float64), full_matrices=False)[2][:k].T
    return {p: ((X[p] - mu) @ V).astype(np.float32) for p in X}, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(S.TASKS))
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--dim", type=int, default=116)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        kind, metric = S.TASKS[task]
        dest = OUT / f"scores_long_{task}{a.tag}.csv"
        rows = pd.read_csv(dest).to_dict("records") if dest.exists() else []
        done = {r["encoder"] for r in rows}
        encs = sorted(d.name for d in (HERE / "embeddings" / task).iterdir()) if (HERE / "embeddings" / task).exists() else []
        if a.encoders:
            encs = [e for e in encs if e in a.encoders]
        for enc in encs:
            if enc in done:
                continue
            got, why = S.load(task, enc)
            if got is None:
                print(f"  [skip] {task}/{enc}: {why}", flush=True)
                continue
            X, Y = (got["X"], got["Y"]) if isinstance(got, dict) else got
            if kind == "multilabel":
                terms = sorted({t for v in Y["train"] for t in json.loads(v)})
                idx = {t: i for i, t in enumerate(terms)}
                Y = {p: np.stack([np.isin(np.arange(len(terms)), [idx[t] for t in json.loads(v) if t in idx])
                                  for v in Y[p]]).astype(np.float32) for p in Y}
                n_out = len(terms)
            elif kind in ("clf", "binary"):
                classes = sorted(set(Y["train"].astype(int)))
                remap = {c: i for i, c in enumerate(classes)}
                keep = {p: np.array([v in remap for v in Y[p].astype(int)]) for p in Y}
                X = {p: X[p][keep[p]] for p in X}
                Y = {p: np.array([remap[v] for v in Y[p].astype(int)[keep[p]]]) for p in Y}
                n_out = len(classes)
            else:
                Y = {p: Y[p].astype(np.float64) for p in Y}
                n_out = 1
            native = X["train"].shape[1]
            X, used = project(X, a.dim)
            mu, sd = X["train"].mean(0), X["train"].std(0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            Xt = {p: torch.from_numpy(((X[p] - mu) / sd).astype(np.float32)).to(S.DEV) for p in X}
            best = None
            for hidden in (0, S.HIDDEN):
                for seed in S.SEEDS:
                    out, val = S.train(Xt, Y, kind, n_out, hidden, seed)
                    s = S.evaluate(kind, out, Y["test"])
                    rows.append(dict(task=task, encoder=enc, probe="mlp2" if hidden else "linear", seed=seed,
                                     metric=metric, score=s, val_loss=val, dim_native=native, dim=used,
                                     n_train=len(Y["train"]), n_test=len(Y["test"]), n_out=n_out))
                    best = (val, s) if best is None or val < best[0] else best
            del Xt
            torch.cuda.empty_cache()
            print(f"  {task:26s} {enc:22s} {native:5d}->{used:4d}  {metric:9s} {best[1]:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(dest, index=False)


if __name__ == "__main__":
    main()
