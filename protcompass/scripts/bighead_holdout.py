"""Held-out erasure and add-back for the tasks whose head needs a GPU, run on the second machine.

The local pass scored the four tasks a CPU head can fit. Remote homology carries 906 classes, EC classification 373,
and GO-MF and GO-BP are multilabel, so all four need the GPU multi-output head that the released runner uses. This
script is that head, and nothing else about the protocol changes: every map, standardiser and head is fitted on the
released stage-1 training split and applied once to the frozen held-out cohort, with four sham permutations and the
same three descriptors.

Self-contained by necessity: the second machine does not share storage with the first, so the descriptor blocks
travel as a precomputed cache rather than being recomputed from structures.

  python remote_holdout.py --tasks remote_homology ec_classification go_mf go_bp
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

R = Path((PROTCOMPASS_ROOT + "/protcompass/confirmation_downstream/bighead_payload"))
OUT = Path((PROTCOMPASS_ROOT + "/protcompass/confirmation_downstream/bighead"))
KIND = {"ec_classification": "clf", "remote_homology": "clf", "go_mf": "multilabel", "go_bp": "multilabel"}
PANEL = ["esm2_650m", "prottrans_t5_xl_u50", "ankh_base", "esm3", "esm_if1", "proteinmpnn"]
SHAM_SEEDS = (0, 1, 2, 3)
HIDDEN, EPOCHS, PATIENCE, BATCH, LR, WD = 128, 30, 3, 256, 1e-3, 1e-4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WITH_NULLS = False
# the eight protein descriptor blocks, in the released column order of common_struct.BLOCKS
BLOCKS = {"secondary structure content": slice(0, 3), "burial and packing": slice(3, 5),
          "global shape": slice(5, 8), "contact order": slice(8, 9), "size": slice(9, 10),
          "amino-acid composition": slice(10, 30)}
PROPERTY = "amino-acid composition"
INDEPENDENT = list(BLOCKS)
NULL_SEEDS = (0, 1, 2, 3)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matched_null import matched_null_projector, variance_removed, property_share_removed   # noqa: E402


# The eraser is imported rather than reimplemented. An earlier local copy differed from scripts/erasure.py in one
# place: it cut the rank of the whitened cross-covariance at 1e-8 of the leading singular value where erasure.py
# cuts at 1e-6. On these four tasks that kept one numerically null direction, and erasing a null direction with an
# oblique projection shears the representation instead of restricting it: the erased arm came out with 10 to 300
# times the variance of the unerased one. Sharing the released implementation removes the discrepancy by
# construction.
from erasure import leace_eraser                                        # noqa: E402


def descriptors(task, ids):
    z = np.load(R / "cache" / f"{task}.npz", allow_pickle=True)
    table = dict(zip([str(i) for i in z["ids"]], z["feat"]))
    miss = [i for i in ids if i not in table]
    if miss:
        raise SystemExit(f"{task}: {len(miss)} ids missing from the descriptor cache, first {miss[:3]}")
    M = np.stack([table[i] for i in ids]).astype(np.float64)
    return {b: M[:, sl] for b, sl in BLOCKS.items()}


def labels(task, ids, kind):
    man = pd.read_csv(R / "manifest" / f"{task}.csv").set_index("id").label
    v = man.reindex(ids)
    if v.isna().any():
        raise SystemExit(f"{task}: {int(v.isna().sum())} ids without a label")
    if kind == "multilabel":
        sets = [json.loads(x) for x in v]
        return sets
    return v.to_numpy()


def load_split(root, task, enc, splits):
    X, ids = [], []
    for s in splits:
        f, i = root / task / enc / f"{s}.npy", root / task / enc / f"ids_{s}.txt"
        if not (f.exists() and i.exists()):
            continue
        X.append(np.load(f).astype(np.float64))
        ids += i.read_text().split()
    return (np.concatenate(X), ids) if X else (None, None)


def fit_head(Xtr, Ytr, Xva, Yva, Xte, Yte, kind, n_out):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, B, C = (torch.tensor((Z - mu) / sd, dtype=torch.float32, device=DEV) for Z in (Xtr, Xva, Xte))
    net = nn.Sequential(nn.Linear(A.shape[1], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_out)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    if kind == "multilabel":
        yt = torch.tensor(Ytr, dtype=torch.float32, device=DEV)
        yv = torch.tensor(Yva, dtype=torch.float32, device=DEV)
        loss_fn = nn.functional.binary_cross_entropy_with_logits
    else:
        yt = torch.tensor(Ytr, dtype=torch.long, device=DEV)
        yv = torch.tensor(Yva, dtype=torch.long, device=DEV)
        loss_fn = nn.functional.cross_entropy
    best, bad, state = -1e9, 0, None
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(A), device=DEV)
        for s in range(0, len(A), BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            loss_fn(net(A[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = -loss_fn(net(B), yv).item()
        if v > best + 1e-6:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if state:
        net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        logits = net(C).cpu().numpy()
    if kind == "multilabel":
        return fmax(Yte, logits), logits
    return float((logits.argmax(1) == np.asarray(Yte)).mean()), logits


def fmax(Y, logits):
    """Protein-centric F-max, the released metric for the two GO tasks."""
    p = 1 / (1 + np.exp(-logits))
    Y = np.asarray(Y)
    best = 0.0
    for t in np.arange(0.05, 1.0, 0.05):
        pred = p >= t
        tp = (pred & (Y > 0)).sum(1)
        denom_p, denom_r = pred.sum(1), (Y > 0).sum(1)
        has = denom_p > 0
        if has.sum() == 0:
            continue
        prec = (tp[has] / denom_p[has]).mean()
        rec = (tp[denom_r > 0] / denom_r[denom_r > 0]).mean() if (denom_r > 0).any() else 0.0
        if prec + rec > 0:
            best = max(best, 2 * prec * rec / (prec + rec))
    return float(best)


def run(task, enc):
    kind = KIND[task]
    Xtr, itr = load_split(R / "released", task, enc, ("train",))
    Xva, iva = load_split(R / "released", task, enc, ("valid",))
    Xte, ite = load_split(R / "holdout", task, enc, ("train", "valid", "test"))
    if Xtr is None or Xva is None or Xte is None:
        print(f"  [skip] {task}/{enc}: missing embeddings", flush=True)
        return []
    ytr, yva, yte = (labels(task, i, kind) for i in (itr, iva, ite))
    if kind == "multilabel":
        terms = sorted({t for s in ytr for t in s})
        idx = {t: k for k, t in enumerate(terms)}
        to_m = lambda ss: np.array([[1.0 if t in set(s) else 0.0 for t in terms] for s in ss])
        Ytr, Yva, Yte = to_m(ytr), to_m(yva), to_m(yte)
        n_out = len(terms)
    else:
        classes = sorted(set(np.concatenate([ytr, yva, yte]).astype(int)))
        rm = {c: k for k, c in enumerate(classes)}
        Ytr, Yva, Yte = ([rm[int(v)] for v in y] for y in (ytr, yva, yte))
        n_out = len(classes)

    Dtr, Dva, Dte = (descriptors(task, i) for i in (itr, iva, ite))
    others = [b for b in INDEPENDENT if b != PROPERTY]
    O = np.concatenate([Dtr[b] for b in others], 1)
    coef = np.linalg.lstsq(O, Dtr[PROPERTY], rcond=None)[0]
    Ztr = Dtr[PROPERTY] - O @ coef

    rng = np.random.default_rng(0)
    desc = {"named": (Dtr[PROPERTY], Dva[PROPERTY], Dte[PROPERTY]),
            "permuted": (Dtr[PROPERTY][rng.permutation(len(Dtr[PROPERTY]))],
                         Dva[PROPERTY][rng.permutation(len(Dva[PROPERTY]))],
                         Dte[PROPERTY][rng.permutation(len(Dte[PROPERTY]))]),
            "other": tuple(np.concatenate([D[b] for b in others], 1) for D in (Dtr, Dva, Dte))}

    mu, P, r, Bb = leace_eraser(Xtr, Ztr)
    if r == 0 or Bb.shape[1] < 8:
        print(f"  [skip] {task}/{enc}: erasure rank {r}", flush=True)
        return []
    er = lambda Z: ((Z - mu) @ P.T) @ Bb
    arms = {"intact": (Xtr, Xva, Xte), "erased": (er(Xtr), er(Xva), er(Xte))}
    diag = {"intact": (0.0, 0.0),
            "erased": (variance_removed(Xtr, er), property_share_removed(Xtr, Ztr, er))}
    for s in SHAM_SEEDS:
        rs = np.random.default_rng(100 + s)
        mus, Ps, rr, Bs = leace_eraser(Xtr, Ztr[rs.permutation(len(Ztr))])
        if rr == 0 or Bs.shape[1] < 8:
            continue
        es = lambda Z, mus=mus, Ps=Ps, Bs=Bs: ((Z - mus) @ Ps.T) @ Bs
        arms[f"sham{s}"] = (es(Xtr), es(Xva), es(Xte))
        diag[f"sham{s}"] = (variance_removed(Xtr, es), property_share_removed(Xtr, Ztr, es))
    if WITH_NULLS:
        # B here is the full orthogonal rotation this runner uses, so the null keeps the same width as the erased arm
        target = diag["erased"][0]
        for s in NULL_SEEDS:
            mun, Pn, _, Bn, _, _ = matched_null_projector(Xtr, Ztr, r, target, seed=200 + s, full_basis=True)
            en = lambda Z, mun=mun, Pn=Pn, Bn=Bn: ((Z - mun) @ Pn.T) @ Bn
            arms[f"null{s}"] = (en(Xtr), en(Xva), en(Xte))
            diag[f"null{s}"] = (variance_removed(Xtr, en), property_share_removed(Xtr, Ztr, en))

    rows, preds = [], {}
    for arm, (A, B, C) in arms.items():
        for key in [None, "named", "permuted", "other"]:
            if key is None:
                A2, B2, C2 = A, B, C
            else:
                d = desc[key]
                A2, B2, C2 = np.hstack([A, d[0]]), np.hstack([B, d[1]]), np.hstack([C, d[2]])
            sc, logits = fit_head(A2, Ytr, B2, Yva, C2, Yte, kind, n_out)
            rows.append(dict(task=task, encoder=enc, arm=arm, descriptor=key or "none", rank=r,
                             dim=Xtr.shape[1], n_train=len(Ytr), n_test=len(Yte), score=sc,
                             var_removed=diag.get(arm, (np.nan, np.nan))[0],
                             prop_removed=diag.get(arm, (np.nan, np.nan))[1]))
            preds[f"{arm}|{key or 'none'}"] = logits.astype(np.float16)
            print(f"  {task:20s} {enc:20s} {arm:7s} {(key or 'none'):9s} {sc:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{task}__{enc}.npz", ids=np.array(ite), **preds)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(KIND))
    ap.add_argument("--encoders", nargs="+", default=PANEL)
    ap.add_argument("--nulls", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    global OUT, WITH_NULLS
    WITH_NULLS = a.nulls
    if a.out_dir:
        OUT = Path(a.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "remote_holdout.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["task"], r["encoder"]) for r in rows}
    print(f"device {DEV}; resuming with {len(done)} cells done", flush=True)
    for task in a.tasks:
        for enc in a.encoders:
            if (task, enc) in done:
                continue
            try:
                rows += run(task, enc)
            except Exception as e:                                        # noqa: BLE001
                print(f"  [failed] {task}/{enc}: {type(e).__name__}: {e}", flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nwritten {csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
