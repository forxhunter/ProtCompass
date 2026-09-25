"""Modality ablation, downstream scoring: the released protocol of extract/score2.py, with per-protein test predictions.

The training, evaluation and preprocessing functions are imported from score2.py unchanged (train, evaluate, TASKS,
SEEDS, HIDDEN), and the label handling below is a line-by-line copy of score2.main. Differences from score2.main:
  - embeddings are read from extract/embeddings/ for the released sources (esm3, saprot; read only) and from
    ab_modality/embeddings/ for the new ones (esm3_seqonly, saprot_seqonly);
  - the test output of every fit (2 readouts x 3 seeds) is saved with the test ids, so the selected readout's
    predictions can be bootstrapped (ab_bootstrap.py);
  - results go to ab_modality/score_parts/ and ab_modality/scores_ab_{long,selected}.csv; score2's released files
    (scores2_*.csv, score_parts/) are never written.
Selection (score2.merge): readout with the lowest validation loss averaged over the three seeds; the cell is the
three-seed mean of that readout's test score.

  python ab_score.py --tasks stability ... --encoders esm3 esm3_seqonly
  python ab_score.py --merge-only
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path((PROTCOMPASS_ROOT + "/protcompass/extract"))
AB = HERE / "ab_modality"
sys.path.insert(0, str(HERE))
import score2                                                            # noqa: E402
from score2 import HIDDEN, SEEDS, TASKS, evaluate, train                 # noqa: E402

ROOTS = {"esm3": HERE / "embeddings", "saprot": HERE / "embeddings",
         "esm3_seqonly": AB / "embeddings", "saprot_seqonly": AB / "embeddings"}
PARTS = AB / "score_parts"
PRED = AB / "predictions"


def load(task, enc):
    """score2.load with the embedding root as a parameter; returns (X, Y, test_ids)."""
    d = ROOTS[enc] / task / enc
    if not (d / "meta.json").exists():
        return None
    if task == "ppi_affinity":
        pairs = pd.read_csv(HERE / "pairs" / "ppi_affinity_pairs.csv")
        vec = np.load(d / "chains.npy").astype(np.float32)
        ids = {i: k for k, i in enumerate((d / "ids_chains.txt").read_text().split())}
        keep = pairs[pairs.chain_a.isin(ids) & pairs.chain_b.isin(ids)]
        X = np.concatenate([vec[[ids[i] for i in keep.chain_a]], vec[[ids[i] for i in keep.chain_b]]], 1)
        Xs, Ys, T = {}, {}, None
        for split in ("train", "valid", "test"):
            m = (keep.split == split).to_numpy()
            Xs[split], Ys[split] = X[m], keep.label.to_numpy()[m]
            if split == "test":
                T = np.array([f"{a}|{b}|{r}" for a, b, r in zip(keep.chain_a[m], keep.chain_b[m], np.where(m)[0])])
        return Xs, Ys, T
    man = pd.read_csv(HERE / "manifest" / f"{task}.csv")
    man = man[man.stage1 == 1] if "stage1" in man else man
    lab = man.set_index("id").label
    X, Y, T = {}, {}, None
    for split in ("train", "valid", "test"):
        ids = (d / f"ids_{split}.txt").read_text().split()
        v = lab.reindex(ids)
        assert not v.isna().any(), (task, enc, split)
        X[split], Y[split] = np.load(d / f"{split}.npy").astype(np.float32), v.to_numpy()
        if split == "test":
            T = np.array(ids)
    return X, Y, T


def score_one(task, enc):
    kind, metric = TASKS[task]
    got = load(task, enc)
    if got is None:
        raise SystemExit(f"{task}/{enc}: no embeddings")
    X, Y, T = got
    # ---- label handling copied from score2.main
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
        T = T[keep["test"]]
        n_out = len(classes)
    else:
        Y = {p: Y[p].astype(np.float64) for p in Y}
        n_out = 1
    mu, sd = X["train"].mean(0), X["train"].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xt = {p: torch.from_numpy(((X[p] - mu) / sd).astype(np.float32)).to(score2.DEV) for p in X}
    # ----
    rows, preds = [], {}
    for hidden in (0, HIDDEN):
        for seed in SEEDS:
            out, val = train(Xt, Y, kind, n_out, hidden, seed)
            s = evaluate(kind, out, Y["test"])
            probe = "mlp2" if hidden else "linear"
            rows.append(dict(task=task, encoder=enc, probe=probe, seed=seed, train_frac=1.0, metric=metric, score=s,
                             val_loss=val, n_train=len(Y["train"]), n_test=len(Y["test"]), dim=X["train"].shape[1],
                             n_out=n_out))
            preds[f"{probe}_{seed}"] = out.astype(np.float32)
    del Xt
    torch.cuda.empty_cache()
    PRED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED / f"{task}__{enc}.npz", y_test=Y["test"], test_ids=T, kind=kind, **preds)
    return rows


def merge():
    parts = [pd.read_csv(f) for f in sorted(PARTS.glob("ab_long_*.csv"))]
    d = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["task", "encoder", "probe", "seed"], keep="last")
    d.to_csv(AB / "scores_ab_long.csv", index=False)
    probe_val = d.groupby(["task", "encoder", "probe"]).val_loss.mean().reset_index()
    best = probe_val.loc[probe_val.groupby(["task", "encoder"]).val_loss.idxmin(), ["task", "encoder", "probe"]]
    sel = d.merge(best, on=["task", "encoder", "probe"], how="inner")
    agg = sel.groupby(["task", "encoder", "probe"]).score.agg(["mean", "std", "size"]).reset_index()
    agg = agg.rename(columns={"mean": "score", "std": "seed_sd", "size": "n_seeds"})
    agg.to_csv(AB / "scores_ab_selected.csv", index=False)
    print(agg.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--encoders", nargs="+", default=list(ROOTS))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.merge_only:
        return merge()
    PARTS.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        for enc in a.encoders:
            f = PARTS / f"ab_long_{task}__{enc}.csv"
            if f.exists() and (PRED / f"{task}__{enc}.npz").exists():
                print(f"{task}/{enc}: done already", flush=True); continue
            t0 = time.time()
            rows = score_one(task, enc)
            pd.DataFrame(rows).to_csv(f, index=False)
            d = pd.DataFrame(rows)
            pv = d.groupby("probe").val_loss.mean()
            sel = d[d.probe == pv.idxmin()].score.mean()
            print(f"  {task:26s} {enc:16s} selected {pv.idxmin():6s} {sel:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            with open(AB / "stage_times.csv", "a") as fh:
                fh.write(f"score,{enc}/{task},{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))},"
                         f"{time.strftime('%Y-%m-%d %H:%M:%S')},{time.time() - t0:.0f}\n")
    merge()


if __name__ == "__main__":
    main()
