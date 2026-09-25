"""Erasure, sham and selective add-back on the held-out downstream cohorts, with per-protein predictions saved.

The Section 4.3 results were produced on the stage-1 subsample of each task. This script repeats the measurement on
rows of the same manifests that were never extracted and never scored, frozen by
scripts/downstream_holdout_freeze.py before any of them was touched.

Nothing is refitted on the held-out rows. Every head, every LEACE map, every standardiser and every descriptor
residualisation is fitted on the released training split and applied to the cohort, which is used once, as a test
set. The arms and the contrast are the ones of scripts/addback_rescue.py:

    R = [ S(E_p X + q) - S(E_p X) ] - mean_pi [ S(E_pi X + q) - S(E_pi X) ]

Per-protein predictions are written for every arm and descriptor so that the group bootstrap can run without a
second pass over the GPU.

  python addback_holdout.py [--tasks membrane_soluble ...] [--encoders esm2_650m ...]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from common_struct import INDEPENDENT_BLOCKS                            # noqa: E402
from erasure import EMB, leace_eraser                                   # noqa: E402
from matched_null import matched_null_projector, variance_removed, property_share_removed   # noqa: E402

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
CD = ROOT / "confirmation_downstream"
HOLD = CD / "embeddings"
PROPERTY = "amino-acid composition"
SHAM_SEEDS = (0, 1, 2, 3)
NULL_SEEDS = (0, 1, 2, 3)
C_GRID = (1e-2, 1e-1, 1.0)
KIND = {"membrane_soluble": "binary", "subcellular_localization": "clf", "fluorescence": "reg",
        "stability": "reg", "ec_classification": "clf", "go_mf": "multilabel", "go_bp": "multilabel",
        "remote_homology": "clf"}
# tasks this run scores: binary, regression and small multiclass only. EC and GO need the GPU multi-output head of
# the released runner and are left to a separate pass rather than scored with a different head here.
SUPPORTED = ("binary", "reg", "clf")


def load_released(task, enc):
    """Released stage-1 train and validation rows: the fitting set, unchanged."""
    d = EMB / task / enc
    man = pd.read_csv(ROOT / "extract" / "manifest" / f"{task}.csv")
    man = man[man.stage1 == 1].set_index("id").label
    out = {}
    for split in ("train", "valid"):
        f, i = d / f"{split}.npy", d / f"ids_{split}.txt"
        if not (f.exists() and i.exists()):
            return None
        ids = i.read_text().split()
        y = man.reindex(ids)
        if y.isna().any():
            return None
        out[split] = (np.load(f).astype(np.float64), y.to_numpy(dtype=float), ids)
    return out


def load_holdout(task, enc):
    """The frozen cohort, whatever source split its rows carry, concatenated as one test set."""
    d = HOLD / task / enc
    if not (d / "meta.json").exists():
        return None
    man = pd.read_csv(ROOT / "extract" / "manifest" / f"{task}.csv").set_index("id").label
    X, ids = [], []
    for f in sorted(d.glob("*.npy")):
        split = f.stem
        idf = d / f"ids_{split}.txt"
        if not idf.exists():
            continue
        X.append(np.load(f).astype(np.float64))
        ids += idf.read_text().split()
    if not X:
        return None
    y = man.reindex(ids)
    if y.isna().any():
        return None
    return np.concatenate(X), y.to_numpy(dtype=float), ids


def extend_concept_table(task, extra_ids):
    """The released descriptor cache covers stage-1 rows only; add the held-out rows without touching it.

    common_struct.build(task) computes and caches descriptors for the scored subsample. A held-out id is absent
    from that table, which makes its whole block NaN and silently drops the property. The extension is cached
    separately so the released cache stays byte-identical.
    """
    table = dict(common_struct.build(task))
    missing = [i for i in extra_ids if i not in table]
    if not missing:
        common_struct._TASK["cache"] = table
        return table
    cache = CD / "concept_cache"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{task}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        table.update(dict(zip(z["ids"], z["feat"])))
        missing = [i for i in extra_ids if i not in table]
    if missing:
        man = pd.read_csv(ROOT / "extract" / "manifest" / f"{task}.csv").set_index("id")
        rows = man.reindex(missing)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(12) as pool:
            feat = list(pool.map(lambda r: common_struct.one_protein(r[0], r[1]),
                                 zip(rows.structure_path, rows.sequence), chunksize=32))
        feat = np.asarray(feat, dtype=np.float32)
        bad = int(np.isnan(feat).any(1).sum())
        print(f"  [{task}] descriptors for {len(feat)} held-out proteins, {bad} unusable", flush=True)
        prev_ids, prev_feat = ([], [])
        if f.exists():
            z = np.load(f, allow_pickle=True)
            prev_ids, prev_feat = list(z["ids"]), list(z["feat"])
        np.savez(f, ids=np.array(prev_ids + list(missing), dtype=object),
                 feat=np.asarray(list(prev_feat) + list(feat), dtype=np.float32))
        table.update(dict(zip(missing, feat)))
    common_struct._TASK["cache"] = table
    return table


def concepts(ids_by_split, task):
    return common_struct.protein_descriptors(ids_by_split, task=task)


def conditional_target(conc, key="train"):
    """Composition residualised on the other independent blocks, the released conditional construction."""
    idx = {"train": 0, "valid": 1, "test": 2}[key]
    Z = conc[PROPERTY][0]
    others = [c for c in INDEPENDENT_BLOCKS if c != PROPERTY and c in conc]
    O = np.concatenate([conc[c][0] for c in others], 1)
    coef = np.linalg.lstsq(O, Z, rcond=None)[0]
    return Z - O @ coef, coef, others


def score_and_predict(Xtr, ytr, Xva, yva, Xte, yte, kind):
    """Fit on train, select on validation, return the test score and the per-protein prediction."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, B, C = ((Z - mu) / sd for Z in (Xtr, Xva, Xte))
    if kind == "reg":
        try:
            m = RidgeCV(alphas=(1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4)).fit(A, ytr)
        except np.linalg.LinAlgError:
            # RidgeCV decomposes the design with LAPACK gesdd, which fails to converge on a few of the restricted
            # matrices: once directions are removed the remaining columns are close to collinear, and which LAPACK
            # path runs depends on the BLAS thread count, so the same cell can converge in one run and not another.
            # The jitter is far below the standardised scale and is the remedy scripts/erasure.py already uses.
            rng = np.random.default_rng(0)
            A = A + 1e-6 * rng.standard_normal(A.shape)
            m = RidgeCV(alphas=(1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4)).fit(A, ytr)
        pred = m.predict(C)
        return float(spearmanr(yte, pred)[0]), pred
    best = None
    for c in C_GRID:
        m = LogisticRegression(C=c, max_iter=2000).fit(A, ytr)
        v = (roc_auc_score(yva, m.predict_proba(B)[:, 1]) if kind == "binary" else m.score(B, yva))
        if best is None or v > best[0]:
            best = (v, m)
    m = best[1]
    if kind == "binary":
        pred = m.predict_proba(C)[:, 1]
        return float(roc_auc_score(yte, pred)), pred
    pred = m.predict(C)
    return float((pred == yte).mean()), pred


def run_cell(task, enc, out_pred, with_nulls=False):
    kind = KIND[task]
    if kind not in SUPPORTED:
        print(f"  [skip] {task}/{enc}: head kind {kind} is not scored in this pass", flush=True)
        return []
    rel, hold = load_released(task, enc), load_holdout(task, enc)
    if rel is None or hold is None:
        print(f"  [skip] {task}/{enc}: released={rel is not None} holdout={hold is not None}", flush=True)
        return []
    (Xtr, ytr, itr), (Xva, yva, iva) = rel["train"], rel["valid"]
    Xte, yte, ite = hold
    ok = ~np.isnan(ytr)
    Xtr, ytr, itr = Xtr[ok], ytr[ok], [i for i, k in zip(itr, ok) if k]

    extend_concept_table(task, list(itr) + list(iva) + list(ite))
    conc = concepts({"train": itr, "valid": iva, "test": ite}, task)
    if PROPERTY not in conc:
        print(f"  [skip] {task}/{enc}: no {PROPERTY} block; present: {sorted(conc)}", flush=True)
        return []
    Ztr, coef, others = conditional_target(conc)
    rng = np.random.default_rng(0)
    Zc = conc[PROPERTY]
    desc = {"named": (Zc[0], Zc[1], Zc[2]),
            "permuted": (Zc[0][rng.permutation(len(Zc[0]))], Zc[1][rng.permutation(len(Zc[1]))],
                         Zc[2][rng.permutation(len(Zc[2]))])}
    if others:
        desc["other"] = tuple(np.concatenate([conc[c][i] for c in others], 1) for i in (0, 1, 2))

    mu, P, r, Bb = leace_eraser(Xtr, Ztr)
    if r == 0 or Bb.shape[1] < 8:
        print(f"  [skip] {task}/{enc}: erasure rank {r}, basis {Bb.shape[1]}", flush=True)
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
    if with_nulls:
        # the variance-matched null: same rank, same share of representation variance as the erased arm, in
        # principal directions carrying as little of the property as the search finds
        target = diag["erased"][0]
        for s in NULL_SEEDS:
            mun, Pn, _, Bn, got, _ = matched_null_projector(Xtr, Ztr, r, target, seed=200 + s)
            en = lambda Z, mun=mun, Pn=Pn, Bn=Bn: ((Z - mun) @ Pn.T) @ Bn
            arms[f"null{s}"] = (en(Xtr), en(Xva), en(Xte))
            diag[f"null{s}"] = (variance_removed(Xtr, en), property_share_removed(Xtr, Ztr, en))

    jobs = [(arm, key) for arm in arms for key in [None, "named", "permuted", "other"]
            if key is None or key in desc]

    def one(job):
        """Score one (arm, descriptor) pair. The concatenation happens here so that at most `workers` of the
        widened matrices exist at a time; building all of them up front costs several gigabytes per encoder."""
        arm, key = job
        A, B, C = arms[arm]
        if key is not None:
            d = desc[key]
            A, B, C = np.hstack([A, d[0]]), np.hstack([B, d[1]]), np.hstack([C, d[2]])
        return score_and_predict(A, ytr, B, yva, C, yte, kind)

    # The pairs are independent and every head here is deterministic, so running them concurrently changes the wall
    # clock and nothing else. One BLAS thread per worker beats a few wide fits: the multiclass solver on a 1,280-
    # dimensional representation spends its time in matrix-vector products that do not parallelise well.
    workers = int(os.environ.get("ADDBACK_WORKERS", "1"))
    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            scored = list(pool.map(one, jobs))
    else:
        scored = [one(j) for j in jobs]

    rows, preds = [], {}
    for (arm, key), (sc, pred) in zip(jobs, scored):
        rows.append(dict(task=task, encoder=enc, arm=arm, descriptor=key or "none", rank=r,
                         dim=Xtr.shape[1], n_train=len(ytr), n_test=len(yte), score=sc,
                         var_removed=diag.get(arm, (np.nan, np.nan))[0],
                         prop_removed=diag.get(arm, (np.nan, np.nan))[1]))
        preds[f"{arm}|{key or 'none'}"] = pred
        print(f"  {task:24s} {enc:20s} {arm:7s} {(key or 'none'):9s} {sc:.4f}", flush=True)
    np.savez_compressed(out_pred / f"{task}__{enc}.npz", ids=np.array(ite), y=yte,
                        **{k: v for k, v in preds.items()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=[t for t in KIND if KIND[t] in SUPPORTED])
    ap.add_argument("--encoders", nargs="+",
                    default=["esm2_650m", "prottrans_t5_xl_u50", "ankh_base", "esm3", "esm_if1", "proteinmpnn"])
    ap.add_argument("--out", default=str(CD / "addback_holdout.csv"))
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--nulls", action="store_true",
                    help="also score the variance-matched null arms; writes extra rows and extra prediction keys")
    a = ap.parse_args()
    pred_dir = Path(a.pred_dir) if a.pred_dir else CD / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    # resume: a (task, encoder) already in the output file is not recomputed, so the run can be stopped and
    # restarted with a different task list without losing finished cells
    rows, done = [], set()
    out_path = pathlib.Path(a.out)
    if out_path.exists():
        prev = pd.read_csv(out_path)
        if len(prev):
            rows = prev.to_dict("records")
            done = set(map(tuple, prev[["task", "encoder"]].drop_duplicates().values))
            print(f"resuming: {len(done)} cells already scored", flush=True)
    for task in a.tasks:
        common_struct.set_task(task)
        for enc in a.encoders:
            if (task, enc) in done:
                continue
            try:
                rows += run_cell(task, enc, pred_dir, with_nulls=a.nulls)
            except Exception as e:                                        # noqa: BLE001
                print(f"  [failed] {task}/{enc}: {type(e).__name__}: {e}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(f"\nwritten: {a.out} ({len(d)} rows); predictions in {pred_dir}")


if __name__ == "__main__":
    main()
