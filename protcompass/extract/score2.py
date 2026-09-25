"""Uniform scoring of every re-extracted task, including the multiclass and multilabel ones.

One protocol for all ten tasks and all encoders: features standardised on the training split, a linear probe and an
MLP-2 probe, regularisation and the choice between the two made on validation, the test split used once for the
selected probe, three seeds. A cell that cannot be evaluated is NaN with a reason, never zero.

Metrics follow the task: Spearman for the regression tasks, AUROC for the binary ones, accuracy for multiclass,
F-max for multilabel. Runs on the GPU when one is free.

  python score2.py --tasks ec_classification go_mf
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
PARTS = HERE / "score_parts"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2)
HIDDEN, MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 512, 60, 5, 1024, 1e-3, 1e-4

TASKS = {
    "stability": ("reg", "spearman"), "fluorescence": ("reg", "spearman"), "ppi_affinity": ("reg", "spearman"),
    "solubility": ("binary", "AUROC"), "membrane_soluble": ("binary", "AUROC"),
    "remote_homology": ("clf", "accuracy"), "ec_classification": ("clf", "accuracy"),
    "subcellular_localization": ("clf", "accuracy"),
    "go_mf": ("multilabel", "Fmax"), "go_bp": ("multilabel", "Fmax"),
}


def fmax(y_true, scores):
    """Protein-centric F-max, the standard CAFA metric: the best F1 over a sweep of thresholds."""
    best = 0.0
    for t in np.linspace(0.01, 0.99, 25):
        pred = scores >= t
        tp = (pred & (y_true > 0)).sum(1)
        p_den, r_den = pred.sum(1), (y_true > 0).sum(1)
        has = p_den > 0
        if not has.any():
            continue
        prec = np.divide(tp, p_den, out=np.zeros_like(tp, dtype=float), where=p_den > 0)[has].mean()
        rec = np.divide(tp, r_den, out=np.zeros_like(tp, dtype=float), where=r_den > 0).mean()
        if prec + rec > 0:
            best = max(best, 2 * prec * rec / (prec + rec))
    return float(best)


def train(X, Y, kind, n_out, hidden, seed):
    torch.manual_seed(seed)
    d = X["train"].shape[1]
    net = (nn.Linear(d, n_out) if not hidden else
           nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    if kind == "multilabel":
        loss_fn = nn.functional.binary_cross_entropy_with_logits
        T = {p: torch.from_numpy(Y[p].astype(np.float32)).to(DEV) for p in ("train", "valid")}
    elif kind in ("clf", "binary"):
        loss_fn = nn.functional.cross_entropy
        T = {p: torch.from_numpy(Y[p].astype(np.int64)).to(DEV) for p in ("train", "valid")}
    else:
        mu, sd = Y["train"].mean(), Y["train"].std() + 1e-8
        loss_fn = lambda o, t: ((o.squeeze(-1) - t) ** 2).mean()
        T = {p: torch.from_numpy(((Y[p] - mu) / sd).astype(np.float32)).to(DEV) for p in ("train", "valid")}
    n = len(Y["train"])
    best, state, bad = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        for s in range(0, n, BATCH):
            idx = torch.randperm(n, device=DEV)[s:s + BATCH] if s == 0 else torch.randint(0, n, (BATCH,), device=DEV)
            loss = loss_fn(net(X["train"][idx]), T["train"][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(X["valid"]), T["valid"]))
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        out = net(X["test"]).float().cpu().numpy()
    if kind == "reg":
        return out.reshape(-1) * (Y["train"].std() + 1e-8) + Y["train"].mean(), best
    return out, best


def evaluate(kind, out, y):
    if kind == "reg":
        return float(spearmanr(y, out)[0])
    if kind == "binary":
        return float(roc_auc_score(y, out[:, 1] - out[:, 0]))
    if kind == "clf":
        return float((out.argmax(1) == y).mean())
    return fmax(y, 1.0 / (1.0 + np.exp(-out)))


def load(task, enc):
    """PPI affinity is a property of a chain pair, so its manifest lists chains and the label lives in the pair file.
    The feature of a pair is the concatenation of the two chain embeddings in the order the dataset gives."""
    d = HERE / "embeddings" / task / enc
    if not (d / "meta.json").exists():
        return None, "no embeddings"
    if task == "ppi_affinity":
        return load_pairs(d), None
    man = pd.read_csv(HERE / "manifest" / f"{task}.csv")
    man = man[man.stage1 == 1] if "stage1" in man else man
    lab = man.set_index("id").label
    X, Y = {}, {}
    for split in ("train", "valid", "test"):
        f = d / f"{split}.npy"
        if not f.exists():
            return None, f"missing {split}.npy"
        ids = (d / f"ids_{split}.txt").read_text().split()
        v = lab.reindex(ids)
        if v.isna().any():
            return None, "labels missing for some ids"
        X[split], Y[split] = np.load(f).astype(np.float32), v.to_numpy()
    return (X, Y), None


def load_pairs(d):
    """Pairs come from the stage-1 pair table written next to the manifests."""
    pairs = pd.read_csv(HERE / "pairs" / "ppi_affinity_pairs.csv")
    emb, ids = {}, {}
    for split in ("chains",):
        f = d / f"{split}.npy"
        if not f.exists():
            return None
        vec = np.load(f).astype(np.float32)
        ids = {i: k for k, i in enumerate((d / f"ids_{split}.txt").read_text().split())}
        emb = vec
    keep = pairs[pairs.chain_a.isin(ids) & pairs.chain_b.isin(ids)]
    X = np.concatenate([emb[[ids[i] for i in keep.chain_a]], emb[[ids[i] for i in keep.chain_b]]], 1)
    out = {}
    for split in ("train", "valid", "test"):
        m = (keep.split == split).to_numpy()
        out[split] = (X[m], keep.label.to_numpy()[m])
    return {"X": {k: v[0] for k, v in out.items()}, "Y": {k: v[1] for k, v in out.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--encoders", nargs="+", default=None, help="score only these sources")
    ap.add_argument("--part-suffix", default="", help="write to scores2_long_<task><suffix>.csv instead, so a run "
                                                      "over a subset of sources cannot replace the full part file")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="score on this fraction of the training rows, sampled once with a fixed seed. Used for the "
                         "training-set-size sweep: the stage-1 manifests cap training at 6000 proteins, so the "
                         "question of whether the group ordering depends on the data budget is answered by moving "
                         "the budget down rather than up")
    ap.add_argument("--merge-only", action="store_true", help="rebuild the tables from the existing parts")
    a = ap.parse_args()
    if a.merge_only:
        return merge()
    rows, missing = [], []
    for task in a.tasks:
        kind, metric = TASKS[task]
        encs = sorted(d.name for d in (HERE / "embeddings" / task).iterdir()) if (HERE / "embeddings" / task).exists() else []
        if a.encoders:
            encs = [e for e in encs if e in a.encoders]
        for enc in encs:
            got, why = load(task, enc)
            if got is None:
                missing.append(dict(task=task, encoder=enc, reason=why)); continue
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
            if a.train_frac < 1.0:
                n = len(Y["train"])
                keep_tr = np.random.default_rng(0).choice(n, size=max(int(round(n * a.train_frac)), 50), replace=False)
                X = {**X, "train": X["train"][keep_tr]}
                Y = {**Y, "train": Y["train"][keep_tr]}
            mu, sd = X["train"].mean(0), X["train"].std(0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            Xt = {p: torch.from_numpy(((X[p] - mu) / sd).astype(np.float32)).to(DEV) for p in X}
            best = None
            for hidden in (0, HIDDEN):
                for seed in SEEDS:
                    out, val = train(Xt, Y, kind, n_out, hidden, seed)
                    s = evaluate(kind, out, Y["test"])
                    rows.append(dict(task=task, encoder=enc, probe="mlp2" if hidden else "linear", seed=seed,
                                     train_frac=a.train_frac,
                                     metric=metric, score=s, val_loss=val, n_train=len(Y["train"]), n_test=len(Y["test"]),
                                     dim=X["train"].shape[1], n_out=n_out))
                    best = (val, s) if best is None or val < best[0] else best
            torch.cuda.empty_cache()
            print(f"  {task:26s} {enc:22s} {metric:9s} {best[1]:.4f}", flush=True)
        # one file per task, so a run over a subset of the tasks cannot replace the results for the others
        PARTS.mkdir(parents=True, exist_ok=True)
        t = pd.DataFrame([r for r in rows if r["task"] == task])
        if len(t):
            t.to_csv(PARTS / f"scores2_long_{task}{a.part_suffix}.csv", index=False)
        pd.DataFrame([m for m in missing if m["task"] == task]).to_csv(
            PARTS / f"scores2_missing_{task}{a.part_suffix}.csv", index=False)
    merge()


def merge():
    """Concatenate the per-task parts into the long, wide and missing tables the rest of the analysis reads."""
    long_parts = [pd.read_csv(f) for f in sorted(PARTS.glob("scores2_long_*.csv")) if f.stat().st_size > 0]
    if not long_parts:
        print("no score parts found")
        return
    d = pd.concat(long_parts, ignore_index=True)
    # The training-size sweep writes parts with the same (task, encoder, probe, seed) key at a smaller budget.
    # Those rows answer a different question and must never enter the released tables, which are the full-budget
    # scores; scripts/train_size_sweep.py reads them from the part files directly.
    if "train_frac" in d:
        d = d[d.train_frac.fillna(1.0) >= 1.0]
    d = d.drop_duplicates(subset=["task", "encoder", "probe", "seed"], keep="last")
    d.to_csv(HERE / "scores2_long.csv", index=False)

    # Select the probe on validation, then report the MEAN over that probe's seeds. Taking the argmin of val_loss
    # twice, as an earlier version did, selects the seed as well, so the published cell was one run rather than a
    # three-seed mean; on the noisiest tasks the two differ by up to 0.29. Selection on validation is legitimate,
    # selecting the seed is not, because the seed is not a hyperparameter.
    probe_val = d.groupby(["task", "encoder", "probe"]).val_loss.mean().reset_index()
    best_probe = probe_val.loc[probe_val.groupby(["task", "encoder"]).val_loss.idxmin(), ["task", "encoder", "probe"]]
    sel = d.merge(best_probe, on=["task", "encoder", "probe"], how="inner")
    agg = sel.groupby(["task", "encoder"]).score.agg(["mean", "std", "size"]).reset_index()
    wide = agg.pivot(index="encoder", columns="task", values="mean")
    wide.round(4).to_csv(HERE / "scores2_wide.csv")
    agg.rename(columns={"mean": "score", "std": "seed_sd", "size": "n_seeds"}).round(4).to_csv(
        HERE / "scores2_selected.csv", index=False)

    miss = []
    for f in sorted(PARTS.glob("scores2_missing_*.csv")):
        try:
            miss.append(pd.read_csv(f))          # a part with no missing cells is written header-less and empty
        except pd.errors.EmptyDataError:
            continue
    miss = pd.concat(miss, ignore_index=True) if miss else pd.DataFrame(columns=["task", "encoder", "reason"])
    # A cell can also be absent because it was never attempted, which the per-task missing files do not record.
    # The protocol promises a reason for every unevaluated cell, so the gaps are reconciled here rather than left blank.
    every = {(t, e) for t in d.task.unique() for e in d.encoder.unique()}
    have = set(map(tuple, agg[["task", "encoder"]].values)) | set(map(tuple, miss[["task", "encoder"]].values))
    extra = [dict(task=t, encoder=e, reason="not attempted: no run recorded in the per-task parts")
             for t, e in sorted(every - have)]
    if extra:
        miss = pd.concat([miss, pd.DataFrame(extra)], ignore_index=True)
    miss.to_csv(HERE / "scores2_missing.csv", index=False)

    print(f"\n{len(agg)} cells over {wide.shape[1]} tasks and {wide.shape[0]} encoders; "
          f"{len(miss)} unevaluated with a stated reason; median seed sd {agg['std'].median():.4f}")
    print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
