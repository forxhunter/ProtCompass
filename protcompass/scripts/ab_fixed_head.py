"""Fixed-head erasure: does the fitted predictor rely on the property?

The released erasure grid (scripts/addback_named_pairs.py) retrains the downstream head on every arm, so it measures
whether the property can be replaced by other directions. This script asks a different question: train the head
once on the intact representation, freeze it, and score it on test inputs whose representation has been intervened
on. The intervention is applied in the original coordinate system,

    x' = mu + (x - mu) @ P.T

with (mu, P) the mean and projection matrix of the released LEACE eraser, of the permuted-property shams, and of the
variance-matched null projectors, all built exactly as in run_cell of addback_named_pairs.py. The released re-basing
matrix B is not applied; it is used only for the released inclusion criterion. B is an orthonormal basis of range(P),
so the re-based arm and the original-coordinate arm differ by an isometry and remove the same variance.

The head is the released head for the task kind (erasure.fit_score), fitted on the intact training rows with the
intact validation split, and frozen together with its standardiser.

Pre-specification: erasure/ab_fixed_head/PRESPEC.md (hash in PRESPEC.sha256).

  python ab_fixed_head.py --check                 # numerical check of the original-coordinate projection
  python ab_fixed_head.py --shard 0 --nshards 4   # one of four single-thread shards
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                         # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
from pathlib import Path                                                # noqa: E402

import numpy as np                                                      # noqa: E402
import pandas as pd                                                     # noqa: E402
from scipy.stats import spearmanr                                       # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeCV            # noqa: E402
from sklearn.metrics import roc_auc_score                               # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from addback_named_pairs import NULL_SEEDS, SHAM_SEEDS, conditional     # noqa: E402
from erasure import ALPHAS, EMB, OUT, TASK_KIND, _dev, concepts_for, leace_eraser, load_task   # noqa: E402
from matched_null import matched_null_projector, variance_removed       # noqa: E402
from panel import panel                                                 # noqa: E402

DEST = OUT / "ab_fixed_head"
VERDICTS = OUT / "named_pair_verdicts_full.csv"


def pairs():
    """Every pair the effect rule or the repair rule names in the released verdicts, in file order."""
    V = pd.read_csv(VERDICTS)
    V = V[V.named_by_effect_rule | V.passes_R_rule]
    return list(zip(V.task, V.concept))


# ----------------------------------------------------------------------------- the frozen head
def fit_head(Xtr, ytr, Xva, yva, yte, kind):
    """Fit the released head (erasure.fit_score) on intact rows and return a frozen predictor.

    The returned function maps raw representation rows to the head's output, standardising them with the intact
    training mean and sd. The selection of the penalty or of the stopping epoch is the released one."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, Bv = (Xtr - mu) / sd, (Xva - mu) / sd
    if kind == "reg":
        try:
            m = RidgeCV(alphas=ALPHAS).fit(A, ytr)
        except np.linalg.LinAlgError:
            rng = np.random.default_rng(0)
            A = A + 1e-6 * rng.standard_normal(A.shape)
            m = RidgeCV(alphas=ALPHAS).fit(A, ytr)
        return (lambda X: m.predict((X - mu) / sd)), None
    if kind == "binary":
        best = None
        for c in (1e-2, 1e-1, 1.0):
            m = LogisticRegression(C=c, max_iter=1000).fit(A, ytr)
            v = roc_auc_score(yva, m.predict_proba(Bv)[:, 1])
            if best is None or v > best[0]:
                best = (v, m)
        mdl = best[1]
        return (lambda X: mdl.predict_proba((X - mu) / sd)[:, 1]), None

    import torch
    import torch.nn as nn
    dev = _dev()
    Xt = {k: torch.from_numpy(v.astype(np.float32)).to(dev) for k, v in (("train", A), ("valid", Bv))}
    classes = None
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
    del Xt, Y

    def predict(X):
        with torch.no_grad():
            T = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(dev)
            return net(T).float().cpu().numpy()
    return predict, classes


def metric(yte, out, kind, classes):
    """The released test metric of erasure.fit_score, computed from the head's output."""
    if kind == "reg":
        return float(spearmanr(yte, out)[0])
    if kind == "binary":
        return float(roc_auc_score(yte, out))
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


# ----------------------------------------------------------------------------- one cell
def orig(m, Pm):
    """The projector applied in the original coordinates: x' = m + (x - m) @ Pm.T."""
    return lambda W: m + (W - m) @ Pm.T


def build(task, name, enc):
    """The released projectors of one cell, following run_cell of addback_named_pairs.py step for step.

    Returns None when run_cell would return no rows, else a dict with the data, the property and the projectors."""
    data = load_task(task, enc)
    if data is None:
        return None
    (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = np.ones(len(ytr), bool) if ytr.ndim > 1 else ~np.isnan(ytr)
    if ok.sum() < 200:
        return None
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    if name not in conc:
        return None
    Z = conditional(conc, name, ok)
    if Z is None:
        return None
    X, y = Xtr[ok], ytr[ok]

    mu, P, r, B = leace_eraser(X, Z)
    if r == 0 or B.shape[1] < 8:
        return None
    er_released = lambda W: ((W - mu) @ P.T) @ B                         # noqa: E731
    proj = {"erased": dict(mu=mu, P=P, B=B)}
    for s in SHAM_SEEDS:
        rs = np.random.default_rng(100 + s)
        mus, Ps, rr, Bs = leace_eraser(X, Z[rs.permutation(len(Z))])
        if rr == 0 or Bs.shape[1] < 8:
            continue
        proj[f"sham{s}"] = dict(mu=mus, P=Ps, B=Bs)
    target = variance_removed(X, er_released)
    for s in NULL_SEEDS:
        mun, Pn, _, Bn, vn, pn = matched_null_projector(X, Z, r, target, seed=300 + s)
        proj[f"null{s}"] = dict(mu=mun, P=Pn, B=Bn, prop_share=pn)
    return dict(X=X, y=y, Xva=Xva, yva=yva, Xte=Xte, yte=yte, Z=Z, r=r, target=target, proj=proj)


def run_cell(task, name, enc):
    kind = TASK_KIND[task]
    t0 = time.time()
    c = build(task, name, enc)
    if c is None:
        return None
    X, Xte, yte = c["X"], c["Xte"], c["yte"]
    head, classes = fit_head(X, c["y"], c["Xva"], c["yva"], yte, kind)
    S, var = {"intact": metric(yte, head(Xte), kind, classes)}, {}
    for arm, pj in c["proj"].items():
        f = orig(pj["mu"], pj["P"])
        S[arm] = metric(yte, head(f(Xte)), kind, classes)
        var[arm] = variance_removed(X, f)
    sh = [a for a in S if a.startswith("sham")]
    nu = [a for a in S if a.startswith("null")]
    rec = dict(task=task, concept=name, encoder=enc, rank=c["r"], dim=X.shape[1],
               S_intact=S["intact"], S_erased=S["erased"],
               S_sham=float(np.mean([S[a] for a in sh])) if sh else np.nan,
               S_null=float(np.mean([S[a] for a in nu])),
               n_sham=len(sh), n_null=len(nu),
               var_erased=var["erased"], var_null=float(np.mean([var[a] for a in nu])),
               var_sham=float(np.mean([var[a] for a in sh])) if sh else np.nan,
               var_target=c["target"],
               prop_null=float(np.mean([c["proj"][a]["prop_share"] for a in nu])))
    rec["Delta_fixed_var"] = rec["S_null"] - rec["S_erased"]
    rec["Delta_fixed_sham"] = rec["S_sham"] - rec["S_erased"]
    for a in sh + nu:
        rec[f"S_{a}"] = S[a]
    rec["seconds"] = time.time() - t0
    return rec


# ----------------------------------------------------------------------------- numerical check
def check(task, name, enc):
    """Verify on one cell that the original-coordinate arm has the dimension of x, that the erased arm has no
    training cross-covariance with the property, and that it is an isometric re-expression of the released arm."""
    common_struct.set_task(task)
    c = build(task, name, enc)
    X, Z, Xte = c["X"], c["Z"], c["Xte"]
    Zc = Z - Z.mean(0, keepdims=True)
    cross = lambda F: (F - F.mean(0, keepdims=True)).T @ Zc / len(F)    # noqa: E731
    base = np.linalg.norm(cross(X))
    lines = [f"cell: {task} / {name} / {enc}; n_train={len(X)}, d={X.shape[1]}, rank r={c['r']}, "
             f"property columns={Z.shape[1]}",
             f"||Cov(x, z)||_F on train = {base:.4e}"]
    rows = []
    for arm in ["erased", "sham0", "null0"]:
        pj = c["proj"][arm]
        f = orig(pj["mu"], pj["P"])
        Xp, Xtp = f(X), f(Xte)
        released = lambda W: ((W - pj["mu"]) @ pj["P"].T) @ pj["B"]     # noqa: E731
        iso_tr = np.abs((Xp - pj["mu"]) @ pj["B"] - released(X)).max()
        back = np.abs(((Xp - pj["mu"]) @ pj["B"]) @ pj["B"].T - (Xp - pj["mu"])).max()
        cc = np.linalg.norm(cross(Xp))
        rows.append(dict(arm=arm, dim_x=X.shape[1], dim_xprime=Xp.shape[1], dim_test_xprime=Xtp.shape[1],
                         dim_released=pj["B"].shape[1], crosscov_rel=cc / base,
                         var_removed_orig=variance_removed(X, f), var_removed_released=variance_removed(X, released),
                         max_abs_B_reexpression_diff=iso_tr, max_abs_range_residual=back,
                         max_abs_idempotence=np.abs(pj["P"] @ pj["P"] - pj["P"]).max()))
    D = pd.DataFrame(rows)
    lines.append(D.to_string(index=False))
    lines.append("crosscov_rel = ||Cov(x', z)||_F / ||Cov(x, z)||_F on the training rows the projector was fitted on.")
    lines.append("max_abs_B_reexpression_diff = max |(x' - mu) @ B - released arm|; max_abs_range_residual = "
                 "max |(x' - mu) B B^T - (x' - mu)|, zero when x' - mu lies in range(P) = span(B).")
    txt = "\n".join(lines)
    print(txt)
    (DEST / "projection_check.txt").write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-cell", nargs=3, default=["fluorescence", "burial and packing", "esm2_650m"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="score at most this many cells (timing runs)")
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if a.check:
        check(*a.check_cell)
        return
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    jobs = []
    for task, name in pairs():
        for enc in [e for e in panel() if (EMB / task / e / "meta.json").exists()]:
            jobs.append((task, name, enc))
    mine = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard][:a.limit]
    dest = DEST / "parts" / f"cells_shard{a.shard}of{a.nshards}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows, done = [], set()
    if dest.exists():
        prev = pd.read_csv(dest)
        rows = prev.to_dict("records")
        done = set(map(tuple, prev[["task", "concept", "encoder"]].values))
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} cells, {len(done)} done", flush=True)
    cur_task = None
    for task, name, enc in mine:
        if (task, name, enc) in done:
            continue
        if task != cur_task:
            common_struct.set_task(task)
            cur_task = task
        try:
            rec = run_cell(task, name, enc)
        except Exception as e:                                           # noqa: BLE001
            print(f"  [failed] {task}/{name}/{enc}: {type(e).__name__}: {e}", flush=True)
            continue
        if rec is None:
            print(f"  [no cell] {task}/{name}/{enc}", flush=True)
            continue
        rows.append(rec)
        print(f"  {task:24s} {name[:22]:22s} {enc:20s} intact {rec['S_intact']:.4f} erased {rec['S_erased']:.4f} "
              f"null {rec['S_null']:.4f} sham {rec['S_sham']:.4f}  dvar {rec['Delta_fixed_var']:+.4f}  "
              f"{rec['seconds']:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(dest, index=False)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"written: {dest} ({len(rows)} cells)", flush=True)


if __name__ == "__main__":
    main()
