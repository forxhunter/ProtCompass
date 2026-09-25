"""Leak-free evaluation of the Information Alignment Hypothesis (IAH).

Question: does an encoder's IIP probing profile predict its downstream scores beyond what is
already known from cheap encoder descriptors (modality, size, pretrained or not, mean probe score,
average downstream quality)?

Two settings
  new-encoder (LOEO): hold out one encoder entirely; fit per-task models on the other encoders;
      predict the held-out encoder on every task from descriptors that need no downstream runs.
  new-task (LOTO, m-shot): hold out one task entirely; the user has run m pilot encoders on it.
      Predict the remaining encoders. Neighbour tasks and task weights may only use the pilot scores.

Scores are z-scored within task using training rows only, so between-task offsets cannot inflate
any metric. All metrics are computed within task and then summarized across tasks.

Usage
  python iah_eval.py                       # manuscript tables as published
  python iah_eval.py --zero-as-missing     # exact 0.000 cells treated as failed runs
  python iah_eval.py --probing P.csv --downstream D.csv --meta M.csv   # clean matrices (wide CSV)
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")

TEX_DIR = Path((PROTCOMPASS_ROOT + "/protcompass/paper/appendices"))
OUT_DIR = Path((PROTCOMPASS_ROOT + "/protcompass/audit"))
ALPHAS = np.logspace(-2, 3, 11)
MODALITY_HEADERS = {"Sequence Encoders": "seq", "Structure Encoders": "struct", "Multimodal Encoders": "multi"}

# Parameter counts from appendices/EncoderSelection.tex; random-weight flags from downstream_tasks/config.py.
PARAMS_M = {"esm1b": 650, "prottrans": 3000, "protbertbfd": 420, "esm235m": 35, "esm2150m": 150, "esm2650m": 650,
            "ankhbase": 453, "schnet": 0.24, "tfn": 0.25, "gvp": 0.85, "dimenet": 0.36, "egnn": 0.85, "ipa": 0.65,
            "gearnet": 6.64, "esmif1": 142, "mace": 0.58, "proteinmpnn": 1.66, "gcpnet": 0.19, "pottsmpnn": 0.25,
            "pottsmpnnsol": 0.25, "pottsmpnnft": 0.25, "prostt5": 3000, "saprot": 650, "esm3": 1400, "foldvision": 123}
RANDOM_WEIGHTS = {"schnet", "tfn", "gvp", "dimenet", "egnn", "mace", "gearnet", "pottsmpnn", "cdconv"}


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ----------------------------------------------------------------------------- data
def parse_tex_table(label, n_cols):
    tex = (TEX_DIR / "SupportingTables.tex").read_text()
    block = tex.split("\\label{%s}" % label)[1].split("\\end{tabular}")[0]
    rows, modality = {}, None
    for line in block.splitlines():
        for header, tag in MODALITY_HEADERS.items():
            if header in line:
                modality = tag
        cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
        if len(cells) != n_cols + 2 or cells[0].startswith("\\textbf"):
            continue
        rows[norm(cells[0])] = (cells[0], modality, [np.nan if c == "--" else float(c) for c in cells[1:1 + n_cols]])
    return rows


def load_published(zero_as_missing):
    probe, down = parse_tex_table("tab:iip_probing_full", 10), parse_tex_table("tab:downstream_full", 14)
    keys = [k for k in down if k in probe]
    Z = pd.DataFrame([probe[k][2] for k in keys], index=keys, columns=[f"I{i}" for i in range(1, 11)])
    Y = pd.DataFrame([down[k][2] for k in keys], index=keys, columns=[f"T{i}" for i in range(1, 15)])
    if zero_as_missing:
        Y = Y.mask(Y == 0.0)
    meta = pd.DataFrame({"modality": [down[k][1] for k in keys]}, index=keys)
    return Z, Y, meta


def load_csv(probing, downstream, meta):
    Z, Y = pd.read_csv(probing, index_col=0), pd.read_csv(downstream, index_col=0)
    M = pd.read_csv(meta, index_col=0)
    keys = [k for k in Y.index if k in Z.index and k in M.index]
    return Z.loc[keys], Y.loc[keys], M.loc[keys]


def descriptor_blocks(Z, meta):
    """Feature blocks that need no downstream evaluation of the encoder."""
    keys = list(Z.index)
    onehot = pd.get_dummies(meta["modality"]).astype(float)
    if "log10_params" in meta:
        logp = meta["log10_params"].astype(float)
    else:
        logp = pd.Series([np.log10(PARAMS_M[norm(k)] * 1e6) if norm(k) in PARAMS_M else np.nan for k in keys], index=keys)
    logp = logp.fillna(logp.groupby(meta["modality"]).transform("median")).fillna(logp.median())
    pretrained = meta["pretrained"].astype(float) if "pretrained" in meta else \
        pd.Series([0.0 if norm(k) in RANDOM_WEIGHTS else 1.0 for k in keys], index=keys)
    cheap = pd.concat([onehot, logp.rename("log10_params"), pretrained.rename("pretrained")], axis=1)
    mean_probe = Z.mean(axis=1).rename("mean_probe").to_frame()
    return {
        "modality": onehot.values,
        "cheap (modality+size+pretrained)": cheap.values,
        "cheap + mean probe": np.hstack([cheap.values, mean_probe.values]),
        "IIP profile": Z.values,
        "cheap + IIP profile": np.hstack([cheap.values, Z.values]),
    }


# ----------------------------------------------------------------------------- helpers
def zscore_cols(Y, rows):
    mu, sd = np.nanmean(Y[rows], axis=0), np.nanstd(Y[rows], axis=0)
    sd[~(sd > 1e-9)] = np.nan
    return (Y - mu) / sd


def ridge_fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    model = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, ytr)
    return model.predict((Xte - mu) / sd)


def rho(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return spearmanr(a[m], b[m])[0]


def selection_metrics(y, p):
    """Regret of picking argmax prediction, in units of the task's score range; top-3 hit."""
    m = ~(np.isnan(y) | np.isnan(p))
    if m.sum() < 5 or np.ptp(y[m]) < 1e-12:
        return np.nan, np.nan
    y, p = y[m], p[m]
    return (y.max() - y[np.argmax(p)]) / np.ptp(y), float(np.argmax(y) in np.argsort(-p)[:3])


# ----------------------------------------------------------------------------- new-encoder setting
def loeo_predictions(X, Y):
    """Hold out each encoder; per-task ridge on the rest. Returns predictions in raw score units."""
    n, T = Y.shape
    P = np.full((n, T), np.nan)
    for e in range(n):
        tr = np.arange(n) != e
        for t in range(T):
            ok = tr & ~np.isnan(Y[:, t])
            if ok.sum() < 8 or np.isnan(Y[e, t]) or np.std(Y[ok, t]) < 1e-9:
                continue
            P[e, t] = ridge_fit_predict(X[ok], Y[ok, t], X[e:e + 1])[0]
    return P


def loeo_task_mean(Y):
    n, T = Y.shape
    P = np.full((n, T), np.nan)
    for e in range(n):
        P[e] = np.nanmean(Y[np.arange(n) != e], axis=0)
    P[np.isnan(Y)] = np.nan
    return P


def loeo_summary(Y, P, P_ref):
    """Per task: Spearman across encoders, and predictive R2 against the leave-one-out task mean."""
    rhos, r2s = [], []
    for t in range(Y.shape[1]):
        m = ~(np.isnan(Y[:, t]) | np.isnan(P[:, t]) | np.isnan(P_ref[:, t]))
        if m.sum() < 8:
            continue
        rhos.append(rho(Y[m, t], P[m, t]))
        sse, sse_ref = np.sum((Y[m, t] - P[m, t]) ** 2), np.sum((Y[m, t] - P_ref[m, t]) ** 2)
        r2s.append(1 - sse / sse_ref if sse_ref > 0 else np.nan)
    return np.nanmedian(rhos), np.nanmedian(r2s), int(np.sum(np.array(r2s) > 0)), len(r2s)


# ----------------------------------------------------------------------------- new-task setting
def loto_predict(X, Yz, t, pilots, mode, rank=2):
    """Predict z-scored scores of held-out task t for all encoders.

    mode: 'pooled'        ridge X -> z-score pooled over the other tasks (no pilot information)
          'encoder_mean'  mean z-score of each encoder over the other tasks (no X, no pilot information)
          'pilot_tasks'   mean z-score over the 3 tasks closest to t on the pilot encoders only (no X)
          'pilot_demand'  task weight vectors w_j fitted on other tasks, reduced to `rank` directions;
                          the new task's coordinates are fitted on the pilot encoders only (uses X)
    """
    n, T = Yz.shape
    others = [j for j in range(T) if j != t]
    if mode == "encoder_mean":
        return np.nanmean(Yz[:, others], axis=1)
    if mode == "pilot_tasks":
        d = []
        for j in others:
            m = ~(np.isnan(Yz[pilots, t]) | np.isnan(Yz[pilots, j]))
            d.append(np.linalg.norm(Yz[pilots, t][m] - Yz[pilots, j][m]) / np.sqrt(m.sum()) if m.sum() >= 2 else np.inf)
        nn = [others[i] for i in np.argsort(d)[:3]]
        return np.nanmean(Yz[:, nn], axis=1)
    Xs = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1.0, X.std(0))
    if mode == "pooled":
        rows = [(Xs[~np.isnan(Yz[:, j])], Yz[~np.isnan(Yz[:, j]), j]) for j in others]
        model = RidgeCV(alphas=ALPHAS).fit(np.vstack([r[0] for r in rows]), np.concatenate([r[1] for r in rows]))
        return model.predict(Xs)
    if mode == "pilot_demand":
        W = []
        for j in others:
            ok = ~np.isnan(Yz[:, j])
            W.append(RidgeCV(alphas=ALPHAS).fit(Xs[ok], Yz[ok, j]).coef_)
        W = np.array(W)
        _, _, Vt = np.linalg.svd(W - W.mean(0), full_matrices=False)
        basis = np.vstack([W.mean(0), Vt[:rank]])            # mean demand + leading demand directions
        F = Xs @ basis.T                                      # encoder scores along each direction
        ok = [p for p in pilots if not np.isnan(Yz[p, t])]
        if len(ok) < 2:
            return F[:, 0]
        coef = RidgeCV(alphas=ALPHAS).fit(F[ok], Yz[ok, t]).coef_
        return F @ coef
    raise ValueError(mode)


def loto_run(X, Y, m_pilot, n_draws, rng):
    n, T = Y.shape
    modes = ["encoder_mean", "pilot_tasks", "pooled", "pilot_demand"]
    res = {k: {"rho": [], "regret": [], "top3": []} for k in modes}
    for t in range(T):
        avail = np.where(~np.isnan(Y[:, t]))[0]
        if len(avail) < m_pilot + 8:
            continue
        per_mode = {k: {"rho": [], "regret": [], "top3": []} for k in modes}
        for _ in range(n_draws):
            pilots = rng.choice(avail, m_pilot, replace=False)
            test = np.setdiff1d(avail, pilots)
            Yz = zscore_cols(Y, np.arange(n))
            for k in modes:
                p = loto_predict(X, Yz, t, pilots, k)
                reg, hit = selection_metrics(Y[test, t], p[test])
                per_mode[k]["rho"].append(rho(Y[test, t], p[test]))
                per_mode[k]["regret"].append(reg)
                per_mode[k]["top3"].append(hit)
        for k in modes:
            for metric in per_mode[k]:
                res[k][metric].append(np.nanmean(per_mode[k][metric]))
    return {k: {metric: float(np.nanmean(v)) for metric, v in d.items()} | {"n_tasks": len(d["rho"])} for k, d in res.items()}


# ----------------------------------------------------------------------------- permutation test
def permute_rows(X, groups, rng):
    Xp = X.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        Xp[idx] = X[rng.permutation(idx)]
    return Xp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probing"); ap.add_argument("--downstream"); ap.add_argument("--meta")
    ap.add_argument("--zero-as-missing", action="store_true")
    ap.add_argument("--pilots", type=int, default=4)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    if args.probing:
        Zdf, Ydf, meta = load_csv(args.probing, args.downstream, args.meta)
        tag = args.tag or "clean"
    else:
        Zdf, Ydf, meta = load_published(args.zero_as_missing)
        tag = args.tag or ("published_zero_missing" if args.zero_as_missing else "published")
    blocks, Y = descriptor_blocks(Zdf, meta), Ydf.values.astype(float)
    print(f"[{tag}] {Y.shape[0]} encoders x {Y.shape[1]} tasks, {int((~np.isnan(Y)).sum())} scored cells")

    # ---- new-encoder setting
    print("\nNEW-ENCODER setting (leave-one-encoder-out, per-task ridge)")
    print(f"{'descriptor':36s} {'median rho':>10s} {'median R2 vs task mean':>23s} {'tasks with R2>0':>16s}")
    P_ref, rows = loeo_task_mean(Y), []
    for name, X in blocks.items():
        r, r2, npos, nt = loeo_summary(Y, loeo_predictions(X, Y), P_ref)
        rows.append({"setting": "new_encoder", "predictor": name, "median_rho": r, "median_r2": r2, "tasks_r2_pos": npos, "n_tasks": nt})
        print(f"{name:36s} {r:10.3f} {r2:23.3f} {npos:>10d} / {nt}")

    # Does the IIP profile add to the cheap descriptors? Shuffle IIP rows within modality.
    X_cheap, Zv, groups = blocks["cheap (modality+size+pretrained)"], Zdf.values, meta["modality"].values
    obs = loeo_summary(Y, loeo_predictions(np.hstack([X_cheap, Zv]), Y), P_ref)
    null = np.array([loeo_summary(Y, loeo_predictions(np.hstack([X_cheap, permute_rows(Zv, groups, rng)]), Y), P_ref)[:2]
                     for _ in range(args.perms)])
    p_rho, p_r2 = (np.sum(null[:, 0] >= obs[0]) + 1) / (args.perms + 1), (np.sum(null[:, 1] >= obs[1]) + 1) / (args.perms + 1)
    print(f"permutation (IIP rows shuffled within modality, {args.perms} draws): "
          f"rho obs {obs[0]:.3f} vs null {null[:, 0].mean():.3f} (p={p_rho:.3f}); "
          f"R2 obs {obs[1]:.3f} vs null {null[:, 1].mean():.3f} (p={p_r2:.3f})")
    rows.append({"setting": "new_encoder", "predictor": "perm test: cheap + IIP", "median_rho": obs[0], "median_r2": obs[1],
                 "null_rho": null[:, 0].mean(), "null_r2": null[:, 1].mean(), "p_rho": p_rho, "p_r2": p_r2})

    # ---- new-task setting
    print(f"\nNEW-TASK setting (leave-one-task-out, {args.pilots} pilot encoders, {args.draws} pilot draws per task)")
    print(f"{'predictor':46s} {'within-task rho':>15s} {'top-1 regret':>13s} {'top-3 hit':>10s}")
    label = {"encoder_mean": "no IIP: encoder mean over other tasks", "pilot_tasks": "no IIP: 3 nearest tasks on pilot scores",
             "pooled": "IIP: pooled ridge, no pilot information", "pilot_demand": "IIP: low-rank task demand fitted on pilots"}
    out = loto_run(Zv, Y, args.pilots, args.draws, rng)
    for k, v in out.items():
        rows.append({"setting": "new_task", "predictor": label[k], "median_rho": v["rho"], "regret": v["regret"], "top3": v["top3"], "n_tasks": v["n_tasks"]})
        print(f"{label[k]:46s} {v['rho']:15.3f} {v['regret']:13.3f} {v['top3']:10.3f}   (n_tasks={v['n_tasks']})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"iah_eval_{tag}.csv", index=False)
    print(f"\nwritten: {OUT_DIR / f'iah_eval_{tag}.csv'}")


if __name__ == "__main__":
    main()
