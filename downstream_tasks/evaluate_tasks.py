"""
evaluate_tasks.py — train probes on frozen embeddings, 5 seeds, all tasks.

One seed = one train/val split + probe weight init. Aggregates to mean ± std.

Saves per-seed results then aggregates:
  results/{encoder}/{task}/seed_{0-4}.json
  results/{encoder}/{task}/aggregated.json   {mean, std, seeds: [...]}

Usage:
  PYTHONPATH=. python downstream_tasks/evaluate_tasks.py --encoder schnet
  PYTHONPATH=. python downstream_tasks/evaluate_tasks.py --encoder all --task all
  PYTHONPATH=. python downstream_tasks/evaluate_tasks.py --encoder all --seeds 0 1 2 3 4
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from downstream_tasks.config import ALL_ENCODERS, EMB_DIR, RES_DIR, SEEDS, TASKS

RES_DIR.mkdir(parents=True, exist_ok=True)


# ── Probe definitions ─────────────────────────────────────────────────────────

def make_linear(task_type: str, seed: int):
    if task_type in ("clf", "clf_residue"):
        # SGDClassifier: O(n_samples) per pass, handles any n_classes efficiently.
        # modified_huber supports predict_proba; 20 passes is enough for relative ranking.
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss="modified_huber", max_iter=20, tol=1e-3,
                             n_jobs=1, random_state=seed)
    elif task_type == "reg":
        return Ridge(alpha=1.0)
    elif task_type == "multilabel":
        # Ridge multi-output: single closed-form solve, much faster than 1943 binary classifiers.
        # Continuous scores work fine with Fmax (threshold sweep over predictions).
        return Ridge(alpha=1.0)
    return None


def make_mlp2(task_type: str, seed: int, D: int):
    hidden = max(128, min(512, D // 2))
    if task_type in ("clf", "clf_residue"):
        return MLPClassifier(hidden_layer_sizes=(hidden, hidden), max_iter=500,
                             early_stopping=True, n_iter_no_change=15,
                             validation_fraction=0.1, random_state=seed)
    elif task_type == "reg":
        return MLPRegressor(hidden_layer_sizes=(hidden, hidden), max_iter=500,
                            early_stopping=True, n_iter_no_change=15,
                            validation_fraction=0.1, random_state=seed)
    elif task_type == "multilabel":
        return MLPClassifier(hidden_layer_sizes=(hidden, hidden), max_iter=500,
                             early_stopping=True, n_iter_no_change=15,
                             validation_fraction=0.1, random_state=seed)
    return None


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metric(y_true, y_pred, y_proba, task: str) -> float:
    cfg = TASKS[task]
    metric  = cfg["metric"]
    ttype   = cfg["type"]

    if metric == "Q3_accuracy":
        flat_t = np.concatenate(y_true) if isinstance(y_true, list) else y_true
        flat_p = np.concatenate(y_pred) if isinstance(y_pred, list) else y_pred
        return float(accuracy_score(flat_t, flat_p))

    elif metric == "spearman_rho":
        rho, _ = spearmanr(np.array(y_true).ravel(), np.array(y_pred).ravel())
        return float(rho)

    elif metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))

    elif metric == "AUROC":
        scores = y_proba[:, 1] if (y_proba is not None and y_proba.ndim == 2) else np.array(y_pred, dtype=float)
        return float(roc_auc_score(y_true, scores))

    elif metric == "AUPRC":
        scores = y_proba[:, 1] if (y_proba is not None and y_proba.ndim == 2) else np.array(y_pred, dtype=float)
        return float(average_precision_score(y_true, scores))

    elif metric == "Fmax":
        return _fmax(np.array(y_true), np.array(y_proba if y_proba is not None else y_pred))

    elif metric == "precision_L5":
        return _precision_at_L5(y_true, y_pred)

    return float("nan")


def _fmax(y_true: np.ndarray, y_scores: np.ndarray,
          thresholds: Optional[np.ndarray] = None) -> float:
    if thresholds is None:
        thresholds = np.linspace(0, 1, 51)
    best = 0.0
    for t in thresholds:
        yp = (y_scores >= t).astype(int)
        tp = (y_true * yp).sum(axis=1)
        fp = ((1 - y_true) * yp).sum(axis=1)
        fn = (y_true * (1 - yp)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = np.where((tp + fp) > 0, tp / (tp + fp), 0.0).mean()
            r = np.where((tp + fn) > 0, tp / (tp + fn), 0.0).mean()
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        best = max(best, f)
    return float(best)


def _precision_at_L5(tertiary_list, pred_list) -> float:
    """Precision@L/5 long-range contacts from CA coordinates."""
    prec_list = []
    for ca_full, pred in zip(tertiary_list, pred_list):
        ca = np.array(ca_full).reshape(-1, 3)
        L  = len(ca)
        if L < 12:
            continue
        diff = ca[:, None, :] - ca[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        gt   = ((dist < 8.0) & (np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= 6)).astype(int)
        triu = np.triu_indices(L, k=6)
        n_pred = max(1, L // 5)
        gt_flat = gt[triu]
        # Use pred as a score if it's an array, else random
        score_flat = pred[triu] if (hasattr(pred, '__len__') and
                                     hasattr(pred, 'ndim') and pred.ndim == 2) \
                     else np.random.rand(len(gt_flat))
        top = np.argsort(score_flat)[::-1][:n_pred]
        prec_list.append(gt_flat[top].mean())
    return float(np.mean(prec_list)) if prec_list else float("nan")


# ── Data loading from cached embeddings ───────────────────────────────────────

def load_split(task: str, encoder: str, split: str):
    """Load cached embeddings and labels for one split."""
    d = EMB_DIR / task / encoder
    emb_f = d / f"{split}_embeddings.npy"
    if not emb_f.exists():
        return None, None

    X = np.load(emb_f, allow_pickle=True)

    # Try primary labels file
    lab_f = d / f"{split}_labels.npy"
    if lab_f.exists():
        y = np.load(lab_f, allow_pickle=True)
    else:
        lab_f0 = d / f"{split}_labels_0.npy"
        lab_f1 = d / f"{split}_labels_1.npy"
        if task == "contact_prediction" and lab_f0.exists():
            # y = (tertiary_coords, valid_masks) — both ragged object arrays
            tertiary   = np.load(lab_f0, allow_pickle=True)
            valid_mask = np.load(lab_f1, allow_pickle=True) if lab_f1.exists() else None
            y = (tertiary, valid_mask)
        elif task == "subcellular_localization" and lab_f0.exists():
            y = np.load(lab_f0)
        elif task == "membrane_soluble" and lab_f1.exists():
            y = np.load(lab_f1)
        else:
            return None, None

    return X, y


def flatten_residue_data(X_obj: np.ndarray, y_obj: np.ndarray):
    """Flatten ragged residue-level arrays to flat numpy arrays."""
    X_flat, y_flat = [], []
    for emb, lab in zip(X_obj, y_obj):
        if emb is None or lab is None:
            continue
        emb = np.array(emb)
        lab = np.array(lab)
        min_len = min(len(emb), len(lab))
        X_flat.append(emb[:min_len])
        y_flat.append(lab[:min_len])
    if not X_flat:
        return None, None
    return np.concatenate(X_flat, axis=0), np.concatenate(y_flat, axis=0)


# ── Probe training ────────────────────────────────────────────────────────────

def run_probe(X_train, y_train, X_test, y_test, task: str, seed: int) -> Dict:
    """Train linear + MLP-2 probes. Return dict of scores."""
    cfg    = TASKS[task]
    ttype  = cfg["type"]
    metric = cfg["metric"]
    D      = X_train.shape[1]

    scaler  = StandardScaler()
    Xtr     = scaler.fit_transform(X_train)
    Xte     = scaler.transform(X_test)

    scores = {}
    for probe_name, model in [("linear", make_linear(ttype, seed)),
                               ("mlp2",   make_mlp2(ttype, seed, D))]:
        if model is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xtr, y_train)
                y_pred = model.predict(Xte)
                y_prob = (model.predict_proba(Xte)
                          if hasattr(model, "predict_proba") else None)

            score = compute_metric(y_test, y_pred, y_prob, task)
            scores[probe_name] = round(float(score), 6)
        except Exception as e:
            scores[probe_name] = None
            print(f"    [warn] {probe_name}: {e}")

    return scores


# ── Contact prediction probe ──────────────────────────────────────────────────

def _contact_pair_data(embs: np.ndarray, tertiaries: np.ndarray,
                        valid_masks, max_pairs: int, rng: np.random.RandomState):
    """Build (pair_features, contact_labels) from per-residue embeddings + CA coords.
    Stops collecting once max_pairs is reached to avoid OOM on large datasets."""
    X_parts, y_parts = [], []
    total_pairs = 0
    vm_iter = valid_masks if valid_masks is not None else [None] * len(embs)

    # Shuffle proteins so the max_pairs cap samples uniformly
    perm = rng.permutation(len(embs))

    for idx_p in perm:
        if total_pairs >= max_pairs:
            break
        emb_obj   = embs[idx_p]
        tert_obj  = tertiaries[idx_p]
        vmask_obj = vm_iter[idx_p] if not isinstance(vm_iter, type(None)) else None

        emb  = np.array(emb_obj, dtype=np.float32)
        tert = np.array(tert_obj, dtype=np.float32)   # [L*3, 3]: N/Cα/C per residue
        ca   = tert[1::3]                              # Cα atoms [L, 3]
        L    = min(len(emb), len(ca))
        if L < 12:
            continue
        emb, ca = emb[:L], ca[:L]

        vmask = np.array(vmask_obj, dtype=bool)[:L] if vmask_obj is not None else np.ones(L, dtype=bool)

        # Long-range pairs: |i-j| >= 6, both residues valid
        ii, jj = np.triu_indices(L, k=6)
        keep   = vmask[ii] & vmask[jj]
        ii, jj = ii[keep], jj[keep]
        if len(ii) == 0:
            continue

        # Subsample pairs from this protein if needed to stay within budget
        remaining = max_pairs - total_pairs
        if len(ii) > remaining:
            sel  = rng.choice(len(ii), remaining, replace=False)
            ii, jj = ii[sel], jj[sel]

        diff     = ca[ii] - ca[jj]
        dists    = np.sqrt((diff * diff).sum(-1))
        contacts = (dists < 8.0).astype(np.int32)
        pair_feat = np.abs(emb[ii] - emb[jj])

        X_parts.append(pair_feat)
        y_parts.append(contacts)
        total_pairs += len(ii)

    if not X_parts:
        return None, None

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def _precision_at_L5_from_embs(embs: np.ndarray, tertiaries: np.ndarray,
                                 valid_masks, model, scaler) -> float:
    """Evaluate precision@L/5 long-range contact prediction on test proteins."""
    prec_list = []
    for emb_obj, tert_obj, vmask_obj in zip(embs, tertiaries, valid_masks if valid_masks is not None else [None]*len(embs)):
        emb  = np.array(emb_obj, dtype=np.float32)
        tert = np.array(tert_obj, dtype=np.float32)
        ca   = tert[1::3]
        L    = min(len(emb), len(ca))
        if L < 12:
            continue
        emb, ca = emb[:L], ca[:L]

        ii, jj = np.triu_indices(L, k=6)
        if len(ii) == 0:
            continue

        diff  = ca[ii] - ca[jj]
        dists = np.sqrt((diff * diff).sum(-1))
        gt    = (dists < 8.0).astype(int)

        pair_feat = np.abs(emb[ii] - emb[jj])
        pair_feat = scaler.transform(pair_feat)
        prob      = model.predict_proba(pair_feat)[:, 1]

        n_pred = max(1, L // 5)
        top    = np.argsort(prob)[::-1][:n_pred]
        prec_list.append(float(gt[top].mean()))

    return float(np.mean(prec_list)) if prec_list else float("nan")


def run_contact_probe(X_tr: np.ndarray, y_tr: tuple,
                       X_te: np.ndarray, y_te: tuple,
                       seed: int, max_pairs: int = 200_000) -> Dict:
    """
    Train linear + MLP-2 contact predictors on per-residue embeddings.
    y_tr / y_te = (tertiary_array, valid_mask_array)  — ragged object arrays.
    Returns dict of probe_name → precision@L/5.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    tertiaries_tr, vmasks_tr = y_tr
    tertiaries_te, vmasks_te = y_te

    rng = np.random.RandomState(seed)
    X_pairs, y_pairs = _contact_pair_data(X_tr, tertiaries_tr, vmasks_tr, max_pairs, rng)
    if X_pairs is None:
        return {"linear": None, "mlp2": None}

    scaler   = StandardScaler()
    X_pairs  = scaler.fit_transform(X_pairs)
    D        = X_pairs.shape[1]
    hidden   = max(128, min(512, D // 2))

    scores = {}
    for probe_name, model in [
        ("linear", LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, random_state=seed)),
        ("mlp2",   MLPClassifier(hidden_layer_sizes=(hidden, hidden), max_iter=300,
                                  early_stopping=True, n_iter_no_change=15,
                                  validation_fraction=0.1, random_state=seed)),
    ]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_pairs, y_pairs)
            prec = _precision_at_L5_from_embs(X_te, tertiaries_te, vmasks_te, model, scaler)
            scores[probe_name] = round(float(prec), 6) if not np.isnan(prec) else None
        except Exception as e:
            scores[probe_name] = None
            print(f"    [warn] contact/{probe_name}: {e}")

    return scores


# ── Per-seed runner ───────────────────────────────────────────────────────────

def run_seed(encoder: str, task: str, seed: int) -> Optional[Dict]:
    out_dir  = RES_DIR / encoder / task
    out_file = out_dir / f"seed_{seed}.json"

    if out_file.exists():
        with open(out_file) as f:
            return json.load(f)

    cfg         = TASKS[task]
    residue_lvl = (cfg["level"] == "residue")

    X_tr, y_tr = load_split(task, encoder, "train")
    X_te, y_te = load_split(task, encoder, "test")

    if X_tr is None or X_te is None:
        return None

    # Contact prediction: handle before flatten (y is a tuple of ragged arrays)
    if task == "contact_prediction":
        # y_tr / y_te = (tertiary_object_array, valid_mask_object_array)
        scores = run_contact_probe(X_tr, y_tr, X_te, y_te, seed)
        result = {"seed": seed, "scores": scores,
                  "metric": cfg["metric"], "n_train": len(X_tr), "n_test": len(X_te)}
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        best = max((v for v in scores.values() if v is not None and not np.isnan(v)), default=None)
        print(f"    seed {seed}: {scores}  best={best:.4f}" if best else f"    seed {seed}: {scores}")
        return result

    MAX_RESIDUES = 80_000    # cap residue-level train samples (probe speed)
    MAX_PROTEINS = 20_000   # cap protein-level train samples (probe speed)

    # Residue-level: flatten
    if residue_lvl:
        X_tr, y_tr = flatten_residue_data(X_tr, y_tr)
        X_te, y_te = flatten_residue_data(X_te, y_te)
        if X_tr is None or X_te is None:
            return None

    # GO multi-label: stack from object array
    if cfg["type"] == "multilabel":
        X_tr = np.stack(X_tr) if X_tr.dtype == object else X_tr
        X_te = np.stack(X_te) if X_te.dtype == object else X_te
        y_tr = np.stack(y_tr) if y_tr.dtype == object else y_tr
        y_te = np.stack(y_te) if y_te.dtype == object else y_te

    # Seed-controlled shuffle (also used for subsampling)
    rng     = np.random.RandomState(seed)
    n_train = len(X_tr)
    idx     = rng.permutation(n_train)
    X_tr    = X_tr[idx]
    y_tr    = y_tr[idx] if not isinstance(y_tr, list) else [y_tr[i] for i in idx]

    # Cap train set size to keep probe training time bounded
    if residue_lvl and len(X_tr) > MAX_RESIDUES:
        X_tr = X_tr[:MAX_RESIDUES]
        y_tr = y_tr[:MAX_RESIDUES] if not isinstance(y_tr, list) else y_tr[:MAX_RESIDUES]
    elif not residue_lvl and len(X_tr) > MAX_PROTEINS:
        X_tr = X_tr[:MAX_PROTEINS]
        y_tr = y_tr[:MAX_PROTEINS] if not isinstance(y_tr, list) else y_tr[:MAX_PROTEINS]

    scores = run_probe(X_tr, y_tr, X_te, y_te, task, seed)

    result = {
        "seed":    seed,
        "scores":  scores,
        "metric":  cfg["metric"],
        "n_train": int(len(X_tr)),
        "n_test":  int(len(X_te)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    best = max((v for v in scores.values() if v is not None and not np.isnan(v)), default=None)
    print(f"    seed {seed}: {scores}  best={best:.4f}" if best else f"    seed {seed}: {scores}")
    return result


def aggregate(encoder: str, task: str, seeds: List[int]) -> Optional[Dict]:
    """Average seed results and save aggregated.json."""
    out_dir  = RES_DIR / encoder / task
    agg_file = out_dir / "aggregated.json"

    seed_results = []
    for s in seeds:
        f = out_dir / f"seed_{s}.json"
        if f.exists():
            with open(f) as fh:
                seed_results.append(json.load(fh))

    if not seed_results:
        return None

    probes   = list(seed_results[0]["scores"].keys())
    agg      = {"encoder": encoder, "task": task,
                "n_seeds": len(seed_results),
                "metric":  seed_results[0].get("metric"),
                "scores":  {}}

    for probe in probes:
        vals = [r["scores"].get(probe) for r in seed_results
                if r["scores"].get(probe) is not None
                and not (isinstance(r["scores"].get(probe), float) and np.isnan(r["scores"].get(probe)))]
        if not vals:
            agg["scores"][probe] = {"mean": None, "std": None, "n": 0}
        else:
            agg["scores"][probe] = {
                "mean": round(float(np.mean(vals)), 6),
                "std":  round(float(np.std(vals)), 6),
                "n":    len(vals),
                "raw":  [round(v, 6) for v in vals],
            }

    # Best probe
    best_mean = max(
        (v["mean"] for v in agg["scores"].values() if v.get("mean") is not None),
        default=None)
    agg["best_mean"] = best_mean
    agg["best_probe"] = max(
        (k for k, v in agg["scores"].items() if v.get("mean") == best_mean),
        default=None)

    with open(agg_file, "w") as f:
        json.dump(agg, f, indent=2)

    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="schnet",
                        choices=ALL_ENCODERS + ["all"])
    parser.add_argument("--task",    default="remote_homology",
                        choices=list(TASKS.keys()) + ["all"])
    parser.add_argument("--seeds",   nargs="+", type=int, default=SEEDS)
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    encoders = ALL_ENCODERS if args.encoder == "all" else [args.encoder]
    tasks    = list(TASKS.keys()) if args.task == "all" else [args.task]
    seeds    = args.seeds

    total = len(encoders) * len(tasks)
    done  = 0
    for enc in encoders:
        for task in tasks:
            done += 1
            emb_meta = EMB_DIR / task / enc / "meta.json"
            if not emb_meta.exists():
                print(f"\n[{done}/{total}] {enc}/{task} — embeddings missing, skip")
                continue

            print(f"\n{'='*60}\n[{done}/{total}] {enc} × {task}\n{'='*60}")

            if args.force:
                for s in seeds:
                    seed_f = RES_DIR / enc / task / f"seed_{s}.json"
                    if seed_f.exists():
                        seed_f.unlink()

            for seed in seeds:
                run_seed(enc, task, seed)

            agg = aggregate(enc, task, seeds)
            if agg:
                bm = agg.get("best_mean")
                print(f"  → AGG: best={bm:.4f} ± {agg['scores'].get(agg.get('best_probe',''),{}).get('std',0):.4f}"
                      if bm else "  → AGG: no valid scores")


if __name__ == "__main__":
    main()
