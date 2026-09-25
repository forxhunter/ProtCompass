"""The four checks that decide whether the causal demand matrix survives review.

1. positive control     Erase the subspace that predicts the task label itself. If the score does not collapse, the
                        method is not sensitive on that task and every null result there is uninformative. This is
                        what tells a near-zero effect on solubility apart from a blind instrument.
2. nonlinear head       LEACE removes linearly recoverable information. A linear downstream head could therefore
                        overstate the damage. Rerun with an MLP head: if the effect survives, the property was not
                        merely linearly encoded for the head's benefit.
3. conditional erasure  The eight properties are correlated, so erasing one may remove another. Erase property A
                        after projecting out property B, and report the effect that survives conditioning.
4. effect size          Express the effect against two rulers a reader has: the spread of the base score across
                        encoders, and the drop from erasing the label subspace itself.

  python erasure_robustness.py --tasks stability solubility --encoders esm2_650m esm_if1 prottrans_t5_xl_u50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor

sys.path.insert(0, str(Path(__file__).parent))
import common_struct
from erasure import (OUT, ROOT, SEEDS, TASK_KIND, concepts_for, fit_score, leace_eraser, load_task,
                     probe_concept, random_eraser)

MLP_KW = dict(hidden_layer_sizes=(128, 128), max_iter=400, early_stopping=True, n_iter_no_change=10, random_state=0)


def fit_score_mlp(Xtr, ytr, Xva, yva, Xte, yte, kind):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, C = (Xtr - mu) / sd, (Xte - mu) / sd
    if kind == "binary":
        m = MLPClassifier(**MLP_KW).fit(A, ytr)
        return float(roc_auc_score(yte, m.predict_proba(C)[:, 1]))
    m = MLPRegressor(**MLP_KW).fit(A, ytr)
    return float(spearmanr(yte, m.predict(C))[0])


def label_subspace(X, y, kind, k=8, ridge=1e-6):
    """A rank-k subspace that carries the label, for the positive control.

    A regression label is one column and needs no reduction. A categorical or multilabel one gives a one-hot matrix
    with as many columns as classes, and erasing all of them on a task with hundreds of classes would empty the
    representation, leaving nothing for the matched-rank random control to be compared against. Reducing the one-hot
    matrix to its own top principal directions does not work either: those are the largest class contrasts, which on
    a near-uniform label carry almost none of the class identity, and the control then shows no drop on a
    representation that clearly does predict the label.

    The reduction used here is the top-k directions of the whitened representation-label cross-covariance, the PLS
    basis: the k label combinations most linearly recoverable from this representation. Erasing them asks whether the
    k directions the representation actually uses for the label are needed, which is the question the control is for.
    k is 8, inside the range of the concept ranks the experiment erases (1 to 19)."""
    if kind == "binary":
        Z = np.eye(2)[y.astype(int)]
    elif kind == "clf":
        classes = np.unique(y)
        Z = (y.reshape(-1, 1) == classes.reshape(1, -1)).astype(np.float64)
    elif kind == "multilabel":
        Z = np.asarray(y, dtype=np.float64)
    else:
        Z = y.reshape(-1, 1).astype(np.float64)
    if Z.shape[1] <= k:
        return Z
    Xc, Zc = X - X.mean(0, keepdims=True), Z - Z.mean(0, keepdims=True)
    S = Xc.T @ Xc / len(Xc) + ridge * np.eye(X.shape[1])
    vals, vecs = np.linalg.eigh(S)
    W = vecs @ np.diag(np.clip(vals, ridge, None) ** -0.5) @ vecs.T
    _, _, Vt = np.linalg.svd(W @ (Xc.T @ Zc / len(Xc)), full_matrices=False)
    return Zc @ Vt[:k].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASK_KIND))
    ap.add_argument("--encoders", nargs="+", default=["esm2_650m", "esm_if1", "prottrans_t5_xl_u50", "schnet", "saprot"])
    ap.add_argument("--concepts", nargs="+", default=None)
    ap.add_argument("--skip-mlp", action="store_true", help="conditional erasure and the positive control only")
    ap.add_argument("--control-only", action="store_true", help="the positive control alone, written to its own file")
    ap.add_argument("--control-ranks", nargs="+", type=int, default=[1, 2, 3, 5, 8, 13, 19],
                    help="ranks at which to erase the label subspace; the concepts occupy ranks 1 to 19")
    a = ap.parse_args()
    rows = []
    for task in a.tasks:
        kind = TASK_KIND[task]
        common_struct.set_task(task)
        for enc in a.encoders:
            data = load_task(task, enc)
            if data is None:
                continue
            (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
            ok = np.ones(len(ytr), bool) if ytr.ndim > 1 else ~np.isnan(ytr)
            if ok.sum() < 200 or (ytr.ndim == 1 and np.isnan(yte).all()):
                continue
            base = fit_score(Xtr[ok], ytr[ok], Xva, yva, Xte, yte, kind)
            base_mlp = np.nan if (a.skip_mlp or a.control_only) else fit_score_mlp(Xtr[ok], ytr[ok], Xva, yva, Xte, yte, kind)

            # 1. positive control: erase the label's own subspace, at each rank the concepts are erased at.
            # A control at one fixed rank is not interpretable: eight directions cannot be expected to destroy a
            # 538-class task, so a small drop there would be read as a blind instrument when it is only a small
            # subspace. Running the ladder gives, for every rank r, the largest effect any rank-r erasure can have on
            # this task and encoder, which is the ruler the concept effects at rank r are read against.
            seen_ranks = set()
            for k in a.control_ranks:
                mu, P, r, B = leace_eraser(Xtr[ok], label_subspace(Xtr[ok], ytr[ok], kind, k=k))
                if r == 0 or B.shape[1] < 8 or r in seen_ranks:
                    continue      # a scalar or binary label spans one rank, so the ladder above it repeats itself
                seen_ranks.add(r)
                er = lambda Z: ((Z - mu) @ P.T) @ B
                lab_after = fit_score(er(Xtr[ok]), ytr[ok], er(Xva), yva, er(Xte), yte, kind)
                ctrl = np.mean([fit_score(((Xtr[ok] - mu) @ random_eraser(Xtr.shape[1], r, s)), ytr[ok],
                                          ((Xva - mu) @ random_eraser(Xtr.shape[1], r, s)), yva,
                                          ((Xte - mu) @ random_eraser(Xtr.shape[1], r, s)), yte, kind)
                                for s in SEEDS[:2]])
                rows.append(dict(task=task, encoder=enc, check="positive control", concept="the task label itself",
                                 rank=r, requested_rank=k, base=base, after=lab_after, random=ctrl,
                                 effect=ctrl - lab_after))
                print(f"  [control] {task:12s} {enc:20s} rank {r:3d}  base {base:.3f} -> {lab_after:.3f} after "
                      f"erasing the label subspace (random {ctrl:.3f}, effect {ctrl - lab_after:+.3f})", flush=True)
                pd.DataFrame(rows).to_csv(out_file(a.skip_mlp, a.control_only), index=False)

            if a.control_only:
                continue

            conc = concepts_for({"train": itr, "valid": iva, "test": ite})
            names = [c for c in (a.concepts or conc) if c in conc]
            for name in names:
                Ztr, Zva, Zte, cat = conc[name]
                mu, P, r, B = leace_eraser(Xtr[ok], Ztr[ok])
                if r == 0 or B.shape[1] < 8:
                    continue
                er = lambda Z: ((Z - mu) @ P.T) @ B
                after_lin = fit_score(er(Xtr[ok]), ytr[ok], er(Xva), yva, er(Xte), yte, kind)
                ctrl_lin, ctrl_mlp = [], []
                for s in SEEDS[:2]:
                    Q = random_eraser(Xtr.shape[1], r, s)
                    cr = lambda Z: (Z - mu) @ Q
                    ctrl_lin.append(fit_score(cr(Xtr[ok]), ytr[ok], cr(Xva), yva, cr(Xte), yte, kind))
                if not a.skip_mlp:
                    after_mlp = fit_score_mlp(er(Xtr[ok]), ytr[ok], er(Xva), yva, er(Xte), yte, kind)
                    for s in SEEDS[:2]:
                        Q = random_eraser(Xtr.shape[1], r, s)
                        cr = lambda Z: (Z - mu) @ Q
                        ctrl_mlp.append(fit_score_mlp(cr(Xtr[ok]), ytr[ok], cr(Xva), yva, cr(Xte), yte, kind))
                    rows.append(dict(task=task, encoder=enc, check="nonlinear head", concept=name, rank=r,
                                     base=base_mlp, after=after_mlp, random=float(np.mean(ctrl_mlp)),
                                     effect=float(np.mean(ctrl_mlp)) - after_mlp,
                                     effect_linear=float(np.mean(ctrl_lin)) - after_lin))
                    print(f"  [mlp]     {task:12s} {enc:20s} {name:28s} linear {float(np.mean(ctrl_lin)) - after_lin:+.3f}  "
                          f"mlp {float(np.mean(ctrl_mlp)) - after_mlp:+.3f}", flush=True)

                # 3. conditional erasure: remove `name` after projecting out every other concept
                others = np.concatenate([conc[c][0] for c in names if c != name], 1)
                Zc = Ztr[ok] - others[ok] @ np.linalg.lstsq(others[ok], Ztr[ok], rcond=None)[0]
                if np.linalg.norm(Zc) < 1e-8:
                    continue
                mu2, P2, r2, B2 = leace_eraser(Xtr[ok], Zc)
                if r2 == 0 or B2.shape[1] < 8:
                    continue
                er2 = lambda Z: ((Z - mu2) @ P2.T) @ B2
                after_c = fit_score(er2(Xtr[ok]), ytr[ok], er2(Xva), yva, er2(Xte), yte, kind)
                ctrl_c = np.mean([fit_score(((Xtr[ok] - mu2) @ random_eraser(Xtr.shape[1], r2, s)), ytr[ok],
                                            ((Xva - mu2) @ random_eraser(Xtr.shape[1], r2, s)), yva,
                                            ((Xte - mu2) @ random_eraser(Xtr.shape[1], r2, s)), yte, kind) for s in SEEDS[:2]])
                rows.append(dict(task=task, encoder=enc, check="conditional", concept=name, rank=r2,
                                 base=base, after=after_c, random=ctrl_c, effect=ctrl_c - after_c))
                print(f"  [cond]    {task:12s} {enc:20s} {name:28s} conditional effect {ctrl_c - after_c:+.3f} "
                      f"(rank {r2})", flush=True)

                # The conditional effect is the one this experiment reports, so the nonlinear-head check has to be run
                # on it and not only on the marginal erasure. Scoring only the marginal arm with an MLP leaves the
                # reported number unchecked, which is what an earlier version of this script did.
                if not a.skip_mlp:
                    after_cm = fit_score_mlp(er2(Xtr[ok]), ytr[ok], er2(Xva), yva, er2(Xte), yte, kind)
                    ctrl_cm = np.mean([fit_score_mlp(((Xtr[ok] - mu2) @ random_eraser(Xtr.shape[1], r2, s)), ytr[ok],
                                                     ((Xva - mu2) @ random_eraser(Xtr.shape[1], r2, s)), yva,
                                                     ((Xte - mu2) @ random_eraser(Xtr.shape[1], r2, s)), yte, kind)
                                       for s in SEEDS[:2]])
                    rows.append(dict(task=task, encoder=enc, check="conditional nonlinear head", concept=name,
                                     rank=r2, base=base_mlp, after=after_cm, random=ctrl_cm,
                                     effect=ctrl_cm - after_cm, effect_linear=ctrl_c - after_c))
                    print(f"  [cond-mlp] {task:11s} {enc:20s} {name:28s} linear {ctrl_c - after_c:+.3f}  "
                          f"mlp {ctrl_cm - after_cm:+.3f}", flush=True)
            pd.DataFrame(rows).to_csv(out_file(a.skip_mlp, a.control_only), index=False)
    print(f"\nwritten: {out_file(a.skip_mlp, a.control_only)}")


def out_file(skip_mlp, control_only=False):
    """Every run writes its own file under parts/, never a shared one.

    A run without OUT_SUFFIX gets a suffix from its process id. Writing to one shared name is how a two-encoder test
    run once overwrote a finished 888-row sweep; the summary concatenates parts/ anyway, so nothing needs the shared
    name. The positive control has its own stem so that rerunning the control does not redo the conditional erasures,
    and so the summary can prefer the dedicated run over control rows an older pass left behind."""
    import os
    stem = "erasure_control" if control_only else ("erasure_conditional" if skip_mlp else "erasure_robustness")
    suffix = os.environ.get("OUT_SUFFIX") or f"_pid{os.getpid()}"
    (OUT / "parts").mkdir(exist_ok=True)
    return OUT / "parts" / f"{stem}{suffix}.csv"


if __name__ == "__main__":
    main()
