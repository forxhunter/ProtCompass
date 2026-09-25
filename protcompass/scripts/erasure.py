"""Does a downstream task need a given property? Erase the property and see what breaks.

The probing question "is property P present in this representation" cannot separate a learned representation from its
own input, because most probed labels are closed-form functions of that input. The causal question is different and
is not affected by that: take a trained representation, remove the linear subspace that carries P, retrain the
downstream head on what is left, and measure the loss in downstream score. Nothing here compares an encoder against a
descriptor of its input.

Erasure uses the closed-form LEACE projection (Belrose et al., 2023): in the whitened space, project out the column
space of the cross-covariance between the representation and the concept. The result is the least-damaging affine map
that makes the concept linearly unpredictable.

Two controls decide whether a drop means anything.
  random subspace  erase a random subspace of the same rank. The difference between the two drops is the effect
                   attributable to P rather than to losing that many directions.
  erasure check    re-probe P after erasure. If P is still predictable the erasure failed and the cell is void.

Everything is fitted on train, selected on validation, and evaluated once on test.

  python erasure.py --tasks stability fluorescence --encoders esm2_650m esm_if1
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
EMB, OUT = ROOT / "extract" / "embeddings", ROOT / "erasure"
ALPHAS = np.logspace(-2, 4, 13)
TASK_KIND = {"stability": "reg", "fluorescence": "reg", "ppi_affinity": "reg", "solubility": "binary",
             "membrane_soluble": "binary", "remote_homology": "clf", "ec_classification": "clf",
             "subcellular_localization": "clf", "go_mf": "multilabel", "go_bp": "multilabel"}
SEEDS = (0, 1, 2)


# ----------------------------------------------------------------------------- erasure
def leace_eraser(X, Z, ridge=1e-6):
    """Return (mu, P) so that (x - mu) @ P + mu makes Z linearly unpredictable from x.

    X: (n, d) representation. Z: (n, k) concept, one-hot for a categorical concept, centred inside.
    """
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    Zc = Z - Z.mean(0, keepdims=True)
    S = Xc.T @ Xc / len(Xc) + ridge * np.eye(X.shape[1])
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, ridge, None)
    W = vecs @ np.diag(vals ** -0.5) @ vecs.T           # whitening
    Wi = vecs @ np.diag(vals ** 0.5) @ vecs.T           # its inverse
    M = W @ (Xc.T @ Zc / len(Xc))                       # whitened cross-covariance
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int((s > s[0] * 1e-6).sum()) if s.size and s[0] > 0 else 0
    U = U[:, :r]
    P = np.eye(X.shape[1]) - Wi @ U @ U.T @ W           # remove the concept subspace, return to the original basis
    # P is an oblique projection, so it is applied as (x - mu) @ P.T for row-major data, and the erased rows lie in
    # range(P). Erasing leaves r directions with near-zero variance; standardising those back to unit variance would
    # resurrect the concept from numerical residue, so the erased data is expressed in an orthonormal basis of
    # range(P): (d - r) coordinates, full rank, nothing to rescale.
    q, sv, _ = np.linalg.svd(P, full_matrices=False)
    B = q[:, sv > sv[0] * 1e-8]
    return mu, P, r, B


def random_eraser(d, r, seed):
    """An orthonormal basis of a random (d - r)-dimensional subspace: the control for losing that many directions."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    return Q[:, r:]


# ----------------------------------------------------------------------------- heads
DEV = None


def _dev():
    global DEV
    if DEV is None:
        import torch
        DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEV


def fit_score(Xtr, ytr, Xva, yva, Xte, yte, kind):
    """One head for every task type. Multiclass and multilabel run on the GPU: EC has 1,195 classes and remote
    homology 906, which a CPU multinomial solver cannot fit inside the loop this script runs thousands of times."""
    import torch
    import torch.nn as nn
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, B, C = ((Z - mu) / sd for Z in (Xtr, Xva, Xte))
    if kind == "reg":
        try:
            m = RidgeCV(alphas=ALPHAS).fit(A, ytr)
        except np.linalg.LinAlgError:
            # RidgeCV decomposes the design matrix with LAPACK gesdd, which fails to converge on a few of the
            # PC-erased matrices: after the top principal directions are removed the remaining columns are close to
            # collinear. A jitter far below the standardised scale breaks the tie without changing any score to the
            # precision reported here. If it fails again the caller records the arm as failed rather than losing the
            # whole cell.
            rng = np.random.default_rng(0)
            A = A + 1e-6 * rng.standard_normal(A.shape)
            m = RidgeCV(alphas=ALPHAS).fit(A, ytr)
        return float(spearmanr(yte, m.predict(C))[0])
    if kind == "binary":
        best = None
        for c in (1e-2, 1e-1, 1.0):
            m = LogisticRegression(C=c, max_iter=1000).fit(A, ytr)
            v = roc_auc_score(yva, m.predict_proba(B)[:, 1])
            if best is None or v > best[0]:
                best = (v, m)
        return float(roc_auc_score(yte, best[1].predict_proba(C)[:, 1]))

    dev = _dev()
    Xt = {k: torch.from_numpy(v.astype(np.float32)).to(dev) for k, v in (("train", A), ("valid", B), ("test", C))}
    if kind == "multilabel":
        Y = {k: torch.from_numpy(np.stack(v).astype(np.float32)).to(dev) for k, v in (("train", ytr), ("valid", yva))}
        n_out, loss_fn = Y["train"].shape[1], nn.functional.binary_cross_entropy_with_logits
    else:
        classes = np.unique(np.concatenate([ytr, yva, yte]).astype(int))
        remap = {c: i for i, c in enumerate(classes)}
        Y = {k: torch.from_numpy(np.array([remap[int(x)] for x in v])).to(dev) for k, v in (("train", ytr), ("valid", yva))}
        n_out, loss_fn = len(classes), nn.functional.cross_entropy
    torch.manual_seed(0)
    net = nn.Linear(Xt["train"].shape[1], n_out).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    n, best, state, bad = len(Y["train"]), float("inf"), None, 0
    for _ in range(60):
        net.train()
        perm = torch.randperm(n, device=dev)
        for s0 in range(0, n, 1024):
            idx = perm[s0:s0 + 1024]
            loss = loss_fn(net(Xt["train"][idx]), Y["train"][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(Xt["valid"]), Y["valid"]))
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        out = net(Xt["test"]).float().cpu().numpy()
    del Xt, Y, net
    torch.cuda.empty_cache()
    if kind == "clf":
        pred = np.array(classes)[out.argmax(1)]
        return float((pred == np.asarray(yte).astype(int)).mean())
    Yte, S = np.stack(yte), 1.0 / (1.0 + np.exp(-out))
    best_f = 0.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = S >= t
        tp, pd_, rd = (pred & (Yte > 0)).sum(1), pred.sum(1), (Yte > 0).sum(1)
        if not (pd_ > 0).any():
            continue
        prec = np.divide(tp, pd_, out=np.zeros(len(tp), float), where=pd_ > 0)[pd_ > 0].mean()
        rec = np.divide(tp, rd, out=np.zeros(len(tp), float), where=rd > 0).mean()
        if prec + rec > 0:
            best_f = max(best_f, 2 * prec * rec / (prec + rec))
    return float(best_f)


def probe_concept(Xtr, Ztr, Xte, Zte, categorical):
    """Accuracy or Pearson of recovering the concept; used to verify that erasure worked."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, C = (Xtr - mu) / sd, (Xte - mu) / sd
    if categorical:
        y = Ztr.argmax(1)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(A, y)
        return float((m.predict(C) == Zte.argmax(1)).mean())
    m = RidgeCV(alphas=ALPHAS).fit(A, Ztr)            # a concept block may have several columns
    p = np.atleast_2d(m.predict(C).T).T
    Z = np.atleast_2d(Zte.T).T
    r = [float(np.corrcoef(p[:, j], Z[:, j])[0, 1]) if np.std(p[:, j]) > 0 and np.std(Z[:, j]) > 0 else 0.0
         for j in range(Z.shape[1])]
    return float(np.mean(r))


# ----------------------------------------------------------------------------- data
def load_pair_task(task, encoder):
    """A binding-affinity example is a pair of chains, so the embedding is the concatenation of the two chain
    embeddings in the order the dataset gives, and the id of an example is the two chain ids joined by a bar."""
    d = EMB / task / encoder
    f = d / "chains.npy"
    if not (d / "meta.json").exists() or not f.exists():
        return None
    vec = np.load(f).astype(np.float64)
    pos = {i: k for k, i in enumerate((d / "ids_chains.txt").read_text().split())}
    pairs = pd.read_csv(ROOT / "extract" / "pairs" / f"{task}_pairs.csv")
    pairs = pairs[pairs.chain_a.isin(pos) & pairs.chain_b.isin(pos)]
    X = np.concatenate([vec[[pos[i] for i in pairs.chain_a]], vec[[pos[i] for i in pairs.chain_b]]], 1)
    ids = (pairs.chain_a + "|" + pairs.chain_b).to_numpy()
    out = {}
    for split in ("train", "valid", "test"):
        m = (pairs.split == split).to_numpy()
        if m.sum() == 0:
            return None
        out[split] = (X[m], pairs.label.to_numpy(dtype=float)[m], list(ids[m]))
    return reduce_if_underdetermined(out)


def reduce_if_underdetermined(data, factor=1):
    """Project onto a train-fitted PCA basis when the training set is too small for the embedding dimension.

    Erasing a subspace and comparing against a random subspace of the same rank both assume the covariance is
    estimated, which needs more examples than dimensions. Binding affinity has 2,352 training pairs and a 2,560-
    dimensional concatenated feature, so without this the whitening is rank-deficient and the two arms are not
    comparable. The basis is fitted on train alone and applied to every split and to both arms, so it changes what
    can be erased (only what the retained directions carry) but not the comparison between erasure and control.

    It triggers only when the training set is smaller than the dimension, which in this benchmark is binding affinity
    with the widest encoders and nothing else: every other task has 6,000 training proteins against at most 2,560
    dimensions. Where it triggers the retained basis is n/2 directions, leaving the same examples-per-dimension ratio
    the widest encoder has on the other tasks."""
    (Xtr, ytr, itr) = data["train"]
    n, d = Xtr.shape
    if n >= factor * d:
        return data
    k = max(16, min(d, n // 2))
    mu = Xtr.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    V = Vt[:k].T
    return {s: ((X - mu) @ V, y, i) for s, (X, y, i) in data.items()}


def load_task(task, encoder):
    """Embeddings and labels per split. For a multilabel task the term vocabulary is built from the training split
    only and applied to the others, so every split has the same label dimension."""
    d = EMB / task / encoder
    if not (d / "meta.json").exists():
        return None
    if (d / "chains.npy").exists() and (ROOT / "extract" / "pairs" / f"{task}_pairs.csv").exists():
        return load_pair_task(task, encoder)
    man = pd.read_csv(ROOT / "extract" / "manifest" / f"{task}.csv")
    man = man[man.stage1 == 1].reset_index(drop=True)
    lab = man.set_index("id").label
    kind = TASK_KIND[task]
    raw, vecs, id_lists = {}, {}, {}
    for split in ("train", "valid", "test"):
        f = d / f"{split}.npy"
        if not f.exists():
            return None
        ids = (d / f"ids_{split}.txt").read_text().split()
        v = lab.reindex(ids)
        if v.isna().any():
            return None
        raw[split], vecs[split], id_lists[split] = v, np.load(f).astype(np.float64), ids
    if kind == "multilabel":
        sets = {p: [json.loads(x) for x in raw[p]] for p in raw}
        terms = sorted({t for s_ in sets["train"] for t in s_})
        idx = {t: i for i, t in enumerate(terms)}
        out = {}
        for p in raw:
            M = np.zeros((len(sets[p]), len(terms)), dtype=np.float64)
            for r, s_ in enumerate(sets[p]):
                for t in s_:
                    if t in idx:
                        M[r, idx[t]] = 1.0
            out[p] = (vecs[p], M, id_lists[p])
        return out
    return {p: (vecs[p], raw[p].to_numpy(dtype=float), id_lists[p]) for p in raw}


def concepts_for(ids_by_split):
    """Protein-level concepts computed from the structure, keyed by the task's own ids.

    These are properties of the protein, not of the encoder: the same values are used for every encoder, so a
    difference between encoders is a difference in how their representation carries the property."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from common_struct import protein_descriptors
    if any("|" in i for v in ids_by_split.values() for i in v[:1]):
        # a pair example: the property of the example is the property of both chains, so the concept for the pair is
        # the two chains' blocks side by side, matching how the two embeddings are concatenated
        a = protein_descriptors({p: [i.split("|")[0] for i in v] for p, v in ids_by_split.items()})
        b = protein_descriptors({p: [i.split("|")[1] for i in v] for p, v in ids_by_split.items()})
        return {n: (np.hstack([a[n][0], b[n][0]]), np.hstack([a[n][1], b[n][1]]),
                    np.hstack([a[n][2], b[n][2]]), a[n][3]) for n in a if n in b}
    return protein_descriptors(ids_by_split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASK_KIND))
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--concepts", nargs="+", default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    encoders = a.encoders or sorted(d.name for d in (EMB / "stability").iterdir() if (d / "meta.json").exists())
    rows = []
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import common_struct
    for task in a.tasks:
        kind = TASK_KIND[task]
        common_struct.set_task(task)
        for enc in encoders:
            data = load_task(task, enc)
            if data is None:
                continue
            (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
            ok = np.ones(len(ytr), bool) if ytr.ndim > 1 else ~np.isnan(ytr)
            if ok.sum() < 200:
                continue
            base = fit_score(Xtr[ok], ytr[ok], Xva, yva, Xte, yte, kind)
            conc = concepts_for({"train": itr, "valid": iva, "test": ite})
            for name, (Ztr, Zva, Zte, categorical) in conc.items():
                mu, P, r, B = leace_eraser(Xtr[ok], Ztr[ok])
                if r == 0 or B.shape[1] < 8:
                    continue
                er = lambda Z: ((Z - mu) @ P.T) @ B
                left = probe_concept(er(Xtr[ok]), Ztr[ok], er(Xte), Zte, categorical)
                before = probe_concept(Xtr[ok], Ztr[ok], Xte, Zte, categorical)
                after = fit_score(er(Xtr[ok]), ytr[ok], er(Xva), yva, er(Xte), yte, kind)
                ctrl = []
                for s in SEEDS:
                    Q = random_eraser(Xtr.shape[1], r, s)
                    cr = lambda Z: (Z - mu) @ Q
                    ctrl.append(fit_score(cr(Xtr[ok]), ytr[ok], cr(Xva), yva, cr(Xte), yte, kind))
                rows.append(dict(task=task, encoder=enc, concept=name, rank=r, dim=Xtr.shape[1],
                                 base=base, after_erasure=after, after_random=float(np.mean(ctrl)),
                                 random_sd=float(np.std(ctrl)), concept_before=before, concept_after=left,
                                 erasure_worked=bool(left < 0.5 * before),
                                 effect=float(np.mean(ctrl) - after)))
                print(f"  {task:14s} {enc:22s} {name:22s} rank {r:3d}  base {base:.3f}  erased {after:.3f}  "
                      f"random {np.mean(ctrl):.3f}  effect {np.mean(ctrl) - after:+.3f}  concept {before:.2f}->{left:.2f}", flush=True)
            pd.DataFrame(rows).to_csv(out_file(), index=False)
    pd.DataFrame(rows).to_csv(out_file(), index=False)
    print(f"\nwritten: {out_file()}")


def out_file():
    """Every run writes its own file under parts/; a run without OUT_SUFFIX gets one from its process id, so a small
    test run cannot overwrite a finished sweep. The summary concatenates parts/."""
    import os
    suffix = os.environ.get("OUT_SUFFIX") or f"_pid{os.getpid()}"
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    return OUT / "parts" / f"erasure_results{suffix}.csv"


if __name__ == "__main__":
    main()
