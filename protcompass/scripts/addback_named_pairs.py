"""The repair contrast for every candidate task-property pair, against both nulls.

Appendix~\\ref{sec:appendix_matched_null} shows that the level contrast the naming rule was written on is not
identified: matched on the amount removed rather than only on the rank, it loses sign consistency across encoders.
The repair contrast survives that change of null. This script computes the repair contrast for every pair that could
plausibly be named, so the rule can be restated on the quantity that is identified and the passing set re-derived
rather than merely re-checked.

    R = [ S(E_p X + q) - S(E_p X) ] - mean_c [ S(E_c X + q) - S(E_c X) ]

with E_p the erasure of the property, E_c a control erasure drawn from one of the two null families, q an
independently computed descriptor of the property, and every map, standardiser and head fitted on the training split
alone. Two descriptors are scored at every arm: the named property block, and the same columns with the proteins
permuted, which matches the width and the marginal and carries no relation to the protein.

The candidate set is every pair whose released conditional effect clears a Holm-corrected sign test at $p<0.05$ with
a positive mean, 18 of the 60 (\\texttt{erasure/named\\_pair\\_candidates.csv}). It is wider than the eight the effect
rule names, so a pair the effect rule missed can still enter, and narrower than all 60, which is a screening on the
released effect and is reported as one.

The head is \\texttt{erasure.fit_score}, the same head the released effect numbers use, so the two contrasts are
computed on one protocol.

  python addback_named_pairs.py --tasks fluorescence [--encoders ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from common_struct import DEPENDS_ON, INDEPENDENT_BLOCKS                # noqa: E402
from erasure import EMB, OUT, TASK_KIND, concepts_for, fit_score, leace_eraser, load_task   # noqa: E402
from matched_null import matched_null_projector, property_share_removed, variance_removed   # noqa: E402
from panel import panel                                                 # noqa: E402

SHAM_SEEDS = (0, 1, 2, 3)
HEAD = "linear"             # set from --head; "mlp" swaps in the released downstream MLP


def fit_score_mlp(Xtr, ytr, Xva, yva, Xte, yte, kind, seed=0):
    """The released downstream MLP (extract/score2.py): two hidden layers of width 512 with ReLU and dropout 0.1,
    AdamW at 1e-3 with weight decay 1e-4, batch 1024, at most 60 epochs, early stopping at patience 5 on the
    validation loss. The width is the same for every encoder, so this is the fixed-capacity nonlinear head the
    repair contrast is re-measured with. Metrics are the ones fit_score reports."""
    import torch
    import torch.nn as nn
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    X = {k: torch.from_numpy(((v - mu) / sd).astype(np.float32)).to(dev) for k, v in (("tr", Xtr), ("va", Xva), ("te", Xte))}
    if kind == "multilabel":
        Y = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(dev) for k, v in (("tr", ytr), ("va", yva))}
        n_out, loss_fn = Y["tr"].shape[1], nn.functional.binary_cross_entropy_with_logits
    elif kind in ("clf", "binary"):
        classes = np.unique(np.concatenate([ytr, yva, yte]).astype(int))
        rm = {c: i for i, c in enumerate(classes)}
        Y = {k: torch.from_numpy(np.array([rm[int(x)] for x in v])).to(dev) for k, v in (("tr", ytr), ("va", yva))}
        n_out, loss_fn = len(classes), nn.functional.cross_entropy
    else:
        m_, s_ = float(np.mean(ytr)), float(np.std(ytr)) + 1e-8
        Y = {k: torch.from_numpy(((np.asarray(v) - m_) / s_).astype(np.float32)).to(dev) for k, v in (("tr", ytr), ("va", yva))}
        n_out, loss_fn = 1, (lambda o, t: ((o.squeeze(-1) - t) ** 2).mean())
    torch.manual_seed(seed)
    d = X["tr"].shape[1]
    net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, n_out)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n, best, state, bad = len(Y["tr"]), float("inf"), None, 0
    for _ in range(60):
        net.train()
        perm = torch.randperm(n, device=dev)
        for s0 in range(0, n, 1024):
            idx = perm[s0:s0 + 1024]
            loss = loss_fn(net(X["tr"][idx]), Y["tr"][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(X["va"]), Y["va"]))
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        out = net(X["te"]).float().cpu().numpy()
    del X, Y, net
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if kind == "reg":
        return float(spearmanr(yte, out.reshape(-1))[0])
    if kind == "binary":
        return float(roc_auc_score(np.asarray(yte).astype(int) == classes.max(), out[:, -1] - out[:, 0]))
    if kind == "clf":
        return float((classes[out.argmax(1)] == np.asarray(yte).astype(int)).mean())
    Yte, S = np.asarray(yte), 1.0 / (1.0 + np.exp(-out))
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


def score(*a):
    return fit_score_mlp(*a) if HEAD == "mlp" else fit_score(*a)
NULL_SEEDS = (0, 1, 2, 3)
CAND = OUT / "named_pair_candidates.csv"


def conditional(conc, name, ok):
    """The property residualised on the other independent blocks: what the conditional arm of the sweep erases."""
    Ztr, _, Zte, _ = conc[name]
    others = ([c for c in conc if c != name and c != DEPENDS_ON[name] and c not in DEPENDS_ON]
              if name in DEPENDS_ON else [c for c in INDEPENDENT_BLOCKS if c != name and c in conc])
    if not others:
        return None
    O = np.concatenate([conc[c][0] for c in others], 1)[ok]
    coef = np.linalg.lstsq(O, Ztr[ok], rcond=None)[0]
    resid = Ztr[ok] - O @ coef
    if np.linalg.norm(resid) / max(np.linalg.norm(Ztr[ok] - Ztr[ok].mean(0)), 1e-12) < 1e-3:
        return None
    return resid


def run_cell(task, name, enc):
    kind = TASK_KIND[task]
    data = load_task(task, enc)
    if data is None:
        return []
    (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = np.ones(len(ytr), bool) if ytr.ndim > 1 else ~np.isnan(ytr)
    if ok.sum() < 200:
        return []
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    if name not in conc:
        return []
    Z = conditional(conc, name, ok)
    if Z is None:
        return []
    X = Xtr[ok]
    y = ytr[ok]

    mu, P, r, B = leace_eraser(X, Z)
    if r == 0 or B.shape[1] < 8:
        return []
    er = lambda W: ((W - mu) @ P.T) @ B
    arms = {"intact": (X, Xva, Xte), "erased": (er(X), er(Xva), er(Xte))}
    diag = {"intact": (0.0, 0.0), "erased": (variance_removed(X, er), property_share_removed(X, Z, er))}
    for s in SHAM_SEEDS:
        rs = np.random.default_rng(100 + s)
        mus, Ps, rr, Bs = leace_eraser(X, Z[rs.permutation(len(Z))])
        if rr == 0 or Bs.shape[1] < 8:
            continue
        es = lambda W, mus=mus, Ps=Ps, Bs=Bs: ((W - mus) @ Ps.T) @ Bs
        arms[f"sham{s}"] = (es(X), es(Xva), es(Xte))
        diag[f"sham{s}"] = (variance_removed(X, es), property_share_removed(X, Z, es))
    target = diag["erased"][0]
    for s in NULL_SEEDS:
        mun, Pn, _, Bn, _, _ = matched_null_projector(X, Z, r, target, seed=300 + s)
        en = lambda W, mun=mun, Pn=Pn, Bn=Bn: ((W - mun) @ Pn.T) @ Bn
        arms[f"null{s}"] = (en(X), en(Xva), en(Xte))
        diag[f"null{s}"] = (variance_removed(X, en), property_share_removed(X, Z, en))

    # the descriptor concatenated back: the property block as released, and the same columns with proteins permuted
    rng = np.random.default_rng(0)
    Qtr, Qva, Qte, _ = conc[name]
    Qtr = Qtr[ok]
    desc = {"named": (Qtr, Qva, Qte),
            "permuted": (Qtr[rng.permutation(len(Qtr))], Qva[rng.permutation(len(Qva))],
                         Qte[rng.permutation(len(Qte))])}

    rows = []
    for arm, (A, Bv, Cv) in arms.items():
        for key in [None, "named", "permuted"]:
            if arm == "intact" and key is not None:
                continue                      # the intact arm is scored once, for context only
            if key is None:
                A2, B2, C2 = A, Bv, Cv
            else:
                q = desc[key]
                A2, B2, C2 = np.hstack([A, q[0]]), np.hstack([Bv, q[1]]), np.hstack([Cv, q[2]])
            sc = score(A2, y, B2, yva, C2, yte, kind)
            rows.append(dict(task=task, concept=name, encoder=enc, arm=arm, descriptor=key or "none",
                             rank=r, dim=X.shape[1], score=sc,
                             var_removed=diag[arm][0], prop_removed=diag[arm][1]))
            print(f"  {task:22s} {name[:22]:22s} {enc:18s} {arm:7s} {(key or 'none'):8s} {sc:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--concepts", nargs="+", default=None,
                    help="restrict to these property blocks, so one (task, property) pair can be its own shard")
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear",
                    help="linear: the released erasure heads; mlp: the released downstream MLP of width 512")
    ap.add_argument("--pairs", default=str(CAND),
                    help="the task-property list to score; the default is the 18 screened candidates, "
                         "erasure/named_pair_grid_full.csv is every pair the released sweep erased")
    a = ap.parse_args()
    global HEAD
    HEAD = a.head
    cand = pd.read_csv(a.pairs)
    if a.tasks:
        cand = cand[cand.task.isin(a.tasks)]
    if a.concepts:
        cand = cand[cand.concept.isin(a.concepts)]
    if not len(cand):
        raise SystemExit("no candidate pair matches this shard")
    dest = Path(a.out) if a.out else OUT / "addback_named_pairs.csv"
    rows, done = [], set()
    if dest.exists():
        prev = pd.read_csv(dest)
        if len(prev):
            rows = prev.to_dict("records")
            done = set(map(tuple, prev[["task", "concept", "encoder"]].drop_duplicates().values))
            print(f"resuming with {len(done)} cells done", flush=True)
    for task, g in cand.groupby("task"):
        common_struct.set_task(task)
        encoders = a.encoders or [e for e in panel() if (EMB / task / e / "meta.json").exists()]
        for name in g.concept:
            for enc in encoders:
                if (task, name, enc) in done:
                    continue
                try:
                    rows += run_cell(task, name, enc)
                except Exception as e:                                    # noqa: BLE001
                    print(f"  [failed] {task}/{name}/{enc}: {type(e).__name__}: {e}", flush=True)
                pd.DataFrame(rows).to_csv(dest, index=False)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\nwritten: {dest} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
