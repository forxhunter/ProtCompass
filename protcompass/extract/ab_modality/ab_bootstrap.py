"""Modality ablation, downstream contrasts: paired protein bootstrap of S(full) - S(seqonly) (PRESPEC.md, step 2).

Reads ab_modality/scores_ab_long.csv (selection) and ab_modality/predictions/<task>__<source>.npz (test outputs).
Writes ab_modality/downstream_deltas.csv, downstream_group_summary.csv and refit_check.csv.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path((PROTCOMPASS_ROOT + "/protcompass/extract"))
AB = HERE / "ab_modality"
sys.path.insert(0, str(HERE))
from score2 import TASKS, evaluate, fmax                                 # noqa: E402

DRAWS = 2000
PAIRS = [("esm3", "esm3_seqonly"), ("saprot", "saprot_seqonly")]
GROUP = {"ec_classification": "experimental", "go_mf": "experimental", "go_bp": "experimental",
         "remote_homology": "experimental", "subcellular_localization": "predicted (AFDB)",
         "membrane_soluble": "predicted (AFDB)", "stability": "predicted (ESMFold)",
         "fluorescence": "predicted (ESMFold)", "solubility": "predicted (ESMFold)", "ppi_affinity": "predicted (ESMFold)"}
ORDER = list(GROUP)


def boot_metric(kind, y, out, I, C):
    """Metric of one fit on every resampled test set. I: (draws, n) indices; C: (draws, n) counts of the same draws."""
    n = len(y)
    if kind == "clf":
        correct = (out.argmax(1) == y).astype(float)
        return C @ correct / n
    if kind == "multilabel":
        s = 1.0 / (1.0 + np.exp(-out))
        yt = y > 0
        r_den = yt.sum(1)
        A, H, R = [], [], []
        for t in np.linspace(0.01, 0.99, 25):            # the thresholds of score2.fmax
            pred = s >= t
            tp = (pred & yt).sum(1).astype(float)
            p_den = pred.sum(1)
            has = p_den > 0
            A.append(np.divide(tp, p_den, out=np.zeros_like(tp), where=has))
            H.append(has.astype(float))
            R.append(np.divide(tp, r_den, out=np.zeros_like(tp), where=r_den > 0))
        A, H, R = np.stack(A, 1), np.stack(H, 1), np.stack(R, 1)
        nh = C @ H
        prec = np.divide(C @ A, nh, out=np.zeros_like(nh), where=nh > 0)
        rec = C @ R / n
        f = np.where((nh > 0) & (prec + rec > 0), 2 * prec * rec / np.maximum(prec + rec, 1e-300), 0.0)
        return f.max(1)
    if kind == "binary":
        sc = out[:, 1] - out[:, 0]
        return np.array([roc_auc_score(y[i], sc[i]) for i in I])
    # Spearman: Pearson of the ranks within each draw (average ranks for ties, as scipy.stats.spearmanr)
    ry = rankdata(y[I], axis=1)
    rp = rankdata(out[I], axis=1)
    ry -= ry.mean(1, keepdims=True); rp -= rp.mean(1, keepdims=True)
    return (ry * rp).sum(1) / np.sqrt((ry ** 2).sum(1) * (rp ** 2).sum(1))


def main():
    long = pd.read_csv(AB / "scores_ab_long.csv")
    pv = long.groupby(["task", "encoder", "probe"]).val_loss.mean().reset_index()
    best = pv.loc[pv.groupby(["task", "encoder"]).val_loss.idxmin()].set_index(["task", "encoder"]).probe
    released = pd.read_csv(HERE / "scores2_selected.csv").set_index(["task", "encoder"]).score
    rel_long = pd.read_csv(HERE / "scores2_long.csv")
    rel_pv = rel_long.groupby(["task", "encoder", "probe"]).val_loss.mean().reset_index()
    rel_best = rel_pv.loc[rel_pv.groupby(["task", "encoder"]).val_loss.idxmin()].set_index(["task", "encoder"]).probe

    rows, draws_by = [], {}
    refit = []
    for task in ORDER:
        kind = TASKS[task][0]
        P = {s: dict(np.load(AB / "predictions" / f"{task}__{s}.npz", allow_pickle=False)) for p in PAIRS for s in p}
        ids0 = P["esm3"]["test_ids"]
        for s in P:
            assert np.array_equal(P[s]["test_ids"], ids0) and np.array_equal(P[s]["y_test"], P["esm3"]["y_test"]), (task, s)
        y = P["esm3"]["y_test"]
        n = len(y)
        rng = np.random.default_rng(sum(map(ord, task)))
        I = rng.integers(0, n, size=(DRAWS, n))
        C = np.stack([np.bincount(r, minlength=n) for r in I]).astype(float)
        S, B = {}, {}
        for s in P:
            probe = best[(task, s)]
            pts, bs = [], []
            for seed in (0, 1, 2):
                out = P[s][f"{probe}_{seed}"]
                pts.append(evaluate(kind, out, y))
                bs.append(boot_metric(kind, y, out, I, C))
            S[s], B[s] = float(np.mean(pts)), np.mean(bs, 0)
            # the bootstrap metric on the identity draw must equal score2.evaluate
            chk = boot_metric(kind, y, P[s][f"{probe}_0"], np.arange(n)[None], np.ones((1, n)))[0]
            assert abs(chk - evaluate(kind, P[s][f"{probe}_0"], y)) < 1e-6, (task, s, chk)
            lg = long[(long.task == task) & (long.encoder == s) & (long.probe == probe)].score.mean()
            assert abs(lg - S[s]) < 1e-6, (task, s, lg, S[s])
            if s in ("esm3", "saprot"):
                rsel = rel_best[(task, s)]
                rl = rel_long[(rel_long.task == task) & (rel_long.encoder == s)]
                ml = long[(long.task == task) & (long.encoder == s)]
                mg = rl.merge(ml, on=["probe", "seed"], suffixes=("_released", "_refit"))
                refit.append(dict(task=task, source=s, released_probe=rsel, refit_probe=probe,
                                  released_score=float(released[(task, s)]), refit_score=S[s],
                                  abs_diff=abs(float(released[(task, s)]) - S[s]),
                                  max_abs_diff_any_fit=float((mg.score_released - mg.score_refit).abs().max()),
                                  reproduces=bool(rsel == probe and abs(float(released[(task, s)]) - S[s]) <= 0.005)))
        for full, seq in PAIRS:
            d = B[full] - B[seq]
            draws_by[(task, full)] = d
            lo, hi = np.percentile(d, [2.5, 97.5])
            lo3, hi3 = np.percentile(d, [0.125, 99.875])
            state = "structure adds" if lo > 0 else "structure costs" if hi < 0 else "unresolved"
            state3 = "structure adds" if lo3 > 0 else "structure costs" if hi3 < 0 else "unresolved"
            rows.append(dict(task=task, group=GROUP[task], metric=TASKS[task][1], model=full, n_test=n,
                             released_full=float(released[(task, full)]), S_full=S[full], S_seqonly=S[seq],
                             probe_full=best[(task, full)], probe_seqonly=best[(task, seq)],
                             delta=S[full] - S[seq], lo=lo, hi=hi, lo9975=lo3, hi9975=hi3, state=state,
                             state_bonferroni=state3))
            print(f"{task:26s} {full:7s} S={S[full]:.4f} seq={S[seq]:.4f} delta={S[full] - S[seq]:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] {state}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(AB / "downstream_deltas.csv", index=False)
    R = pd.DataFrame(refit)
    R.to_csv(AB / "refit_check.csv", index=False)
    print(R.round(4).to_string(index=False))
    G = []
    for full, _ in PAIRS:
        for gname, tasks in (("experimental", [t for t in ORDER if GROUP[t] == "experimental"]),
                             ("predicted", [t for t in ORDER if GROUP[t] != "experimental"])):
            sub = D[(D.model == full) & D.task.isin(tasks)]
            m = np.mean([draws_by[(t, full)] for t in tasks], 0)
            lo, hi = np.percentile(m, [2.5, 97.5])
            G.append(dict(model=full, group=gname, n_tasks=len(tasks), mean_delta=sub.delta.mean(), lo=lo, hi=hi,
                          adds=int((sub.state == "structure adds").sum()),
                          costs=int((sub.state == "structure costs").sum()),
                          unresolved=int((sub.state == "unresolved").sum())))
    G = pd.DataFrame(G)
    G.to_csv(AB / "downstream_group_summary.csv", index=False)
    print(G.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
