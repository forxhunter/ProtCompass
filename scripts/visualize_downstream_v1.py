#!/usr/bin/env python3
"""
Downstream task evaluation + paper-style figures — v1.

Usage:
    python scripts/visualize_downstream_v1.py            # use cache CSV if present
    python scripts/visualize_downstream_v1.py --force    # re-run all evaluations
    python scripts/visualize_downstream_v1.py --quick    # 5 encoders only (debug)

Outputs:
    outputs/downstream_v1/downstream_results.csv   — per-encoder×task scores (cache)
    outputs/downstream_v1/fig1_heatmap.{png,pdf}
    outputs/downstream_v1/fig2_sequence_radar.{png,pdf}
    outputs/downstream_v1/fig3_structure_radar.{png,pdf}
    outputs/downstream_v1/fig4_multimodal_radar.{png,pdf}
    outputs/downstream_v1/fig5_pca.{png,pdf}
    outputs/downstream_v1/encoder_summary.csv
"""

import argparse
import json
import sys
import warnings
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
EMB_DIR = ROOT / "downstream_tasks" / "embeddings"
RES_DIR = ROOT / "downstream_tasks" / "results"   # pre-computed probe results
OUT_DIR = ROOT / "outputs" / "downstream_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Task definitions ──────────────────────────────────────────────────────────
TASKS = {
    "contact_prediction":       {"T": "T1",  "metric": "precision_L5", "level": "pair",     "type": "contact"},
    "remote_homology":          {"T": "T2",  "metric": "accuracy",     "level": "protein",  "type": "clf"},
    "fluorescence":             {"T": "T3",  "metric": "spearman_rho", "level": "protein",  "type": "reg"},
    "stability":                {"T": "T4",  "metric": "spearman_rho", "level": "protein",  "type": "reg"},
    "mutation_effect":          {"T": "T5",  "metric": "spearman_rho", "level": "protein",  "type": "reg"},
    "ec_classification":        {"T": "T6",  "metric": "accuracy",     "level": "protein",  "type": "clf"},
    "go_mf":                    {"T": "T7",  "metric": "Fmax",         "level": "protein",  "type": "multilabel"},
    "go_bp":                    {"T": "T8",  "metric": "Fmax",         "level": "protein",  "type": "multilabel"},
    "solubility":               {"T": "T9",  "metric": "AUROC",        "level": "protein",  "type": "clf"},
    "ppi_site":                 {"T": "T10", "metric": "AUPRC",        "level": "residue",  "type": "clf"},
    "metal_binding":            {"T": "T11", "metric": "AUPRC",        "level": "residue",  "type": "clf"},
    "ppi_affinity":             {"T": "T12", "metric": "spearman_rho", "level": "protein",  "type": "reg"},
    "subcellular_localization": {"T": "T13", "metric": "accuracy",     "level": "protein",  "type": "clf"},
    "membrane_soluble":         {"T": "T14", "metric": "AUROC",        "level": "protein",  "type": "clf"},
}

TASK_ORDER = [f"T{i}" for i in range(1, 15)]

TASK_SHORT = {
    "T1":  "Contact\nPred.",  "T2":  "Homology",        "T3":  "Fluor.",
    "T4":  "Stab.",           "T5":  "Mut.\nEffect",    "T6":  "EC\nClass.",
    "T7":  "GO-MF",           "T8":  "GO-BP",           "T9":  "Solub.",
    "T10": "PPI\nSites",      "T11": "Metal\nBind.",    "T12": "PPI\nAffin.",
    "T13": "Subcell.\nLoc.",  "T14": "Mem./Sol.",
}

TASK_SHORT_1LINE = {
    "T1":  "Contact",     "T2":  "Homology",    "T3":  "Fluor.",
    "T4":  "Stability",   "T5":  "Mut.Effect",  "T6":  "EC Class.",
    "T7":  "GO-MF",       "T8":  "GO-BP",       "T9":  "Solubility",
    "T10": "PPI Sites",   "T11": "Metal Bind.", "T12": "PPI Affin.",
    "T13": "Subcell.Loc.", "T14": "Mem./Sol.",
}

# Task groupings by category (5 groups)
TASK_GROUPS = {
    'Structure Prediction': ['T1', 'T2'],
    'Fitness Landscape': ['T3', 'T4', 'T5'],
    'Functional Annotation': ['T6', 'T7', 'T8', 'T9'],
    'Protein-Protein Interactions': ['T10', 'T11', 'T12'],
    'Subcellular Localization': ['T13', 'T14'],
}

TASK_GROUP_SHORT = {
    'Structure Prediction': 'Struct.\nPred.',
    'Fitness Landscape': 'Fitness',
    'Functional Annotation': 'Function',
    'Protein-Protein Interactions': 'PPI',
    'Subcellular Localization': 'Subcell.',
}

SEEDS = [0, 1, 2, 3, 4]

# ── Encoder taxonomy — matches visualize_paper_v12.py exactly ─────────────────
ENCODER_MODALITY = {
    **{e: 'Sequence' for e in [
        'ankh', 'esm1b', 'esm2', 'esm2_35m', 'esm2_150m', 'esm2_650m',
        'prottrans', 'prottrans_t5', 'prot_t5', 'proteinbert_bfd',
        'prostt5', 'unirep',
    ]},
    **{e: 'Structure' for e in [
        'foldseek', 'geobpe', 'bpe',
        'proteinmpnn', 'proteinmpnn_pretrained', 'ligandmpnn',
        'pottsmpnn', 'pottsmpnn_pretrained',
        'pottsmpnn_soluble_pretrained', 'pottsmpnn_ft_pretrained',
        'pottsmpnn_ca_pretrained',
        'gearnet', 'gearnet_pretrained', 'gearnet_edge', 'gearnet_edge_ief',
        'schnet', 'dimenet', 'dimenet_plus_plus', 'mace',
        'dmasif', 'dmasif_pretrained',
        'egnn', 'egnn_pretrained',
        'gvp', 'gvp_pretrained',
        'tfn', 'tfn_pretrained',
        'ipa', 'ipa_pretrained',
        'gcpnet', 'gcpnet_pretrained',
        'esm_if', 'esm_if_inverse',
        'comenet', 'equiformer', 'equiformer_v2', 'gemnet', 'painn',
        'se3_transformer', 'spherenet', 'lefttnet',
        'density', 'mif', 'proteinae', 'starling', 'dual_stream',
    ]},
    **{e: 'Multimodal' for e in [
        'saprot', 'saprot_pretrained',
        'esm3',
        'foldvision',
        'cdconv', 'cdconv_pretrained',
    ]},
    **{e: 'Baseline' for e in [
        'random', 'random_token', 'length', 'torsion', 'onehot', 'blosum', 'test',
    ]},
}

ENCODER_SUBTYPE = {
    **{e: 'Sequence' for e in [
        'ankh', 'esm1b', 'esm2', 'esm2_35m', 'esm2_150m', 'esm2_650m',
        'prottrans', 'prottrans_t5', 'prot_t5', 'proteinbert_bfd',
        'prostt5', 'unirep',
    ]},

    'foldseek': 'Struct: Token',
    'geobpe':   'Struct: Token',
    'bpe':      'Struct: Token',

    **{e: 'Struct: Graph' for e in [
        'proteinmpnn', 'proteinmpnn_pretrained', 'ligandmpnn',
        'pottsmpnn', 'pottsmpnn_pretrained',
        'pottsmpnn_soluble_pretrained', 'pottsmpnn_ft_pretrained',
        'pottsmpnn_ca_pretrained',
        'gearnet', 'gearnet_pretrained', 'gearnet_edge', 'gearnet_edge_ief',
        'schnet', 'dimenet', 'dimenet_plus_plus', 'mace',
        'dmasif', 'dmasif_pretrained',
        'comenet', 'equiformer', 'equiformer_v2', 'gemnet', 'painn',
        'se3_transformer', 'spherenet', 'lefttnet',
        'density', 'mif', 'proteinae', 'starling', 'dual_stream',
    ]},

    **{e: 'Struct: Equiv' for e in [
        'egnn', 'egnn_pretrained',
        'gvp', 'gvp_pretrained',
        'tfn', 'tfn_pretrained',
        'ipa', 'ipa_pretrained',
        'gcpnet', 'gcpnet_pretrained',
        'esm_if', 'esm_if_inverse',
    ]},

    **{e: 'Multimodal' for e in [
        'saprot', 'saprot_pretrained',
        'esm3',
        'foldvision',
        'cdconv', 'cdconv_pretrained',
    ]},

    **{e: 'Baseline' for e in [
        'random', 'random_token', 'length', 'torsion', 'onehot', 'blosum', 'test',
    ]},
}

DISPLAY_NAMES = {
    'ankh': 'Ankh', 'esm1b': 'ESM-1b', 'esm2': 'ESM-2(650M)', 'esm2_35m': 'ESM-2(35M)',
    'esm2_150m': 'ESM-2(150M)', 'esm2_650m': 'ESM-2(650M)',
    'prottrans': 'ProtTrans', 'prottrans_t5': 'ProtTrans-T5', 'prot_t5': 'ProtT5',
    'proteinbert_bfd': 'ProteinBERT', 'prostt5': 'ProstT5', 'unirep': 'UniRep',
    'geobpe': 'GeoBPE', 'bpe': 'BPE',
    'foldseek': 'FoldSeek', 'proteinmpnn': 'ProteinMPNN', 'ligandmpnn': 'LigandMPNN',
    'pottsmpnn': 'PottsMPNN', 'pottsmpnn_soluble': 'PottsMPNN(Sol)',
    'pottsmpnn_ft': 'PottsMPNN(FT)', 'pottsmpnn_ca': 'PottsMPNN(CA)',
    'gearnet': 'GearNet', 'gearnet_edge': 'GearNet-Edge', 'gearnet_edge_ief': 'GearNet-Edge-IEF',
    'schnet': 'SchNet', 'dimenet': 'DimeNet++', 'dimenet_plus_plus': 'DimeNet++',
    'mace': 'MACE', 'dmasif': 'dMaSIF', 'egnn': 'EGNN', 'gvp': 'GVP', 'tfn': 'TFN',
    'ipa': 'IPA', 'gcpnet': 'GCPNet', 'esm_if': 'ESM-IF', 'esm_if_inverse': 'ESM-IF-Inv',
    'comenet': 'ComENet', 'equiformer': 'EquiformerV1', 'equiformer_v2': 'EquiformerV2',
    'gemnet': 'GemNet', 'painn': 'PaiNN', 'se3_transformer': 'SE3-Transformer',
    'spherenet': 'SphereNet', 'lefttnet': 'LEFTNet', 'density': 'Density',
    'mif': 'MIF', 'proteinae': 'ProteinAE', 'starling': 'Starling', 'dual_stream': 'DualStream',
    'saprot': 'SaProt', 'esm3': 'ESM-3', 'foldvision': 'FoldVision', 'cdconv': 'CDConv',
    'random': 'Random', 'random_token': 'Random-Token', 'length': 'Length',
    'torsion': 'Torsion', 'onehot': 'OneHot', 'blosum': 'BLOSUM', 'test': 'Test',
}

MODALITY_COLORS = {
    'Sequence':  '#1a4a8a',   # navy blue
    'Structure': '#D2691E',   # chocolate orange
    'Multimodal':'#7c4fa0',   # purple
    'Baseline':  '#666666',   # darker gray
}

SUBTYPE_COLORS = {
    'Sequence':     '#1a4a8a',   # navy blue
    'Struct: Token':'#FFA500',   # orange
    'Struct: Graph':'#D2691E',   # chocolate
    'Struct: Equiv':'#A0522D',   # sienna
    'Multimodal':   '#7c4fa0',
    'Baseline':     '#666666',   # darker gray
}

# heatmap colormap: light rose → deep berry (distinct from teal probing heatmap)
HMAP_CMAP = LinearSegmentedColormap.from_list(
    'paper_rose',
    ['#fdf0f0', '#f0c4c4', '#d4777a', '#a83a3e', '#6b1a1d'],
    N=256,
)

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.titlesize': 10, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained-only filter (same logic as v12)
# ─────────────────────────────────────────────────────────────────────────────

def is_shown(enc, all_encoders):
    if '_pretrained' in enc:
        return True
    return f"{enc}_pretrained" not in all_encoders


def disp(enc):
    """Strip _pretrained suffix for display and format."""
    clean = enc.replace('_pretrained', '')
    return DISPLAY_NAMES.get(clean, clean)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_flat(d: Path, split: str) -> np.ndarray:
    """Object array of mmap slices — only accessed proteins are paged in."""
    flat    = np.load(d / f"{split}_embeddings_flat.npy", mmap_mode='r')
    offsets = np.load(d / f"{split}_offsets.npy")
    N = len(offsets) - 1
    X = np.empty(N, dtype=object)
    for i in range(N):
        X[i] = flat[offsets[i]:offsets[i+1]]   # no .copy() — lazy mmap slice
    return X


def load_embeddings(task: str, encoder: str):
    d = EMB_DIR / task / encoder
    if not (d / "meta.json").exists():
        return None
    try:
        with open(d / "meta.json") as f:
            meta = json.load(f)

        def _load_emb(split):
            flat_f = d / f"{split}_embeddings_flat.npy"
            off_f  = d / f"{split}_offsets.npy"
            if flat_f.exists() and off_f.exists():
                return _load_flat(d, split)
            return np.load(d / f"{split}_embeddings.npy", allow_pickle=True)

        tr_emb = _load_emb("train")
        te_emb = _load_emb("test")

        # Contact prediction has split label files
        if task == "contact_prediction":
            tr_coords = np.load(d / "train_labels_0.npy", allow_pickle=True)
            tr_mask   = np.load(d / "train_labels_1.npy", allow_pickle=True)
            te_coords = np.load(d / "test_labels_0.npy",  allow_pickle=True)
            te_mask   = np.load(d / "test_labels_1.npy",  allow_pickle=True)
            return dict(tr_emb=tr_emb, te_emb=te_emb,
                        tr_coords=tr_coords, tr_mask=tr_mask,
                        te_coords=te_coords, te_mask=te_mask, meta=meta)

        tr_lbl = np.load(d / "train_labels.npy", allow_pickle=True)
        te_lbl = np.load(d / "test_labels.npy",  allow_pickle=True)
        return dict(tr_emb=tr_emb, te_emb=te_emb,
                    tr_lbl=tr_lbl, te_lbl=te_lbl, meta=meta)
    except Exception as e:
        print(f"  [WARN] Load {encoder}/{task}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmax(y_true, y_scores):
    thresholds = np.linspace(0, 1, 51)
    best = 0.0
    for t in thresholds:
        yp = (y_scores >= t).astype(int)
        tp = (y_true * yp).sum(1)
        fp = ((1 - y_true) * yp).sum(1)
        fn = (y_true * (1 - yp)).sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = np.where((tp + fp) > 0, tp / (tp + fp), 0.0).mean()
            r = np.where((tp + fn) > 0, tp / (tp + fn), 0.0).mean()
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        best = max(best, f)
    return float(best)


def _pool_ragged(emb_arr):
    """Mean-pool ragged per-residue embeddings → (N, D) protein embeddings."""
    out = []
    for e in emb_arr:
        e = np.array(e, dtype=np.float32)
        out.append(e.mean(0) if e.ndim == 2 else e)
    return np.stack(out)


def _flatten_residue(emb_arr, lbl_arr, max_residues=60_000, seed=0):
    """Flatten per-protein residue arrays to flat (N_residues, D) and (N_residues,)."""
    X, y = [], []
    for emb, lbl in zip(emb_arr, lbl_arr):
        emb = np.array(emb, dtype=np.float32)
        lbl = np.array(lbl)
        X.append(emb); y.append(lbl)
    X = np.concatenate(X, 0)
    y = np.concatenate(y, 0)
    if len(X) > max_residues:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_residues, replace=False)
        X, y = X[idx], y[idx]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Contact prediction — precision @ L/5
# ─────────────────────────────────────────────────────────────────────────────

def _eval_contact(data, seed, max_tr_proteins=80, pairs_per_protein=80):
    rng = np.random.default_rng(seed)
    tr_emb    = data['tr_emb']
    tr_coords = data['tr_coords']
    tr_mask   = data['tr_mask']
    te_emb    = data['te_emb']
    te_coords = data['te_coords']
    te_mask   = data['te_mask']

    # Build bilinear (element-wise product) training pairs
    X_tr, y_tr = [], []
    idx_order = rng.permutation(len(tr_emb))[:max_tr_proteins]
    for i in idx_order:
        emb    = np.array(tr_emb[i],    dtype=np.float32)   # (L, D)
        coords = np.array(tr_coords[i], dtype=np.float32)   # (L, 3)
        mask   = np.array(tr_mask[i],   dtype=bool)         # (L,)
        vi = np.where(mask)[0]
        if len(vi) < 6:
            continue
        emb_v = emb[vi]; coords_v = coords[vi]
        L = len(vi)
        sep  = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
        diff = coords_v[:, None] - coords_v[None, :]        # (L,L,3)
        dist = np.sqrt((diff ** 2).sum(-1))
        contact = ((dist < 8.0) & (sep >= 6)).astype(np.int8)
        rows, cols = np.where(sep >= 6)
        if len(rows) > pairs_per_protein:
            sel = rng.choice(len(rows), pairs_per_protein, replace=False)
            rows, cols = rows[sel], cols[sel]
        for r, c in zip(rows, cols):
            X_tr.append(emb_v[r] * emb_v[c])
            y_tr.append(contact[r, c])

    if len(X_tr) < 20:
        return np.nan

    X_tr = np.array(X_tr, dtype=np.float32)
    y_tr = np.array(y_tr,  dtype=np.int8)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200,
                        early_stopping=True, random_state=seed)
    clf.fit(X_tr, y_tr)

    # Evaluate precision @ L/5 per test protein
    precs = []
    for i in range(len(te_emb)):
        emb    = np.array(te_emb[i],    dtype=np.float32)
        coords = np.array(te_coords[i], dtype=np.float32)
        mask   = np.array(te_mask[i],   dtype=bool)
        vi = np.where(mask)[0]
        if len(vi) < 6:
            continue
        emb_v = emb[vi]; coords_v = coords[vi]
        L = len(vi)
        sep  = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
        rows, cols = np.where(sep >= 6)
        if len(rows) == 0:
            continue
        diff = coords_v[:, None] - coords_v[None, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        contact = (dist < 8.0) & (sep >= 6)

        feats = np.array([emb_v[r] * emb_v[c] for r, c in zip(rows, cols)], dtype=np.float32)
        feats = scaler.transform(feats)
        scores = clf.predict_proba(feats)[:, 1]

        k = max(1, L // 5)
        top_idx = np.argsort(scores)[::-1][:k]
        prec = float(np.mean([contact[rows[j], cols[j]] for j in top_idx]))
        precs.append(prec)

    return float(np.mean(precs)) if precs else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed evaluation for standard tasks
# ─────────────────────────────────────────────────────────────────────────────

def _eval_seed(task, cfg, data, seed):
    task_type = cfg['type']
    metric    = cfg['metric']
    residue   = cfg['level'] == 'residue'

    tr_emb = data['tr_emb']
    te_emb = data['te_emb']
    tr_lbl = data['tr_lbl']
    te_lbl = data['te_lbl']

    # Pool residue-level embeddings for MLP probe input
    if residue and tr_emb.dtype == object:
        tr_X = _pool_ragged(tr_emb)
        te_X = _pool_ragged(te_emb)
    else:
        tr_X = tr_emb.astype(np.float32)
        te_X = te_emb.astype(np.float32)

    # Handle residue-level labels (flatten for classification, pool otherwise)
    if residue and tr_lbl.dtype == object:
        if metric in ('AUPRC', 'Q3_accuracy'):
            # Need per-residue labels — flatten (subsample if large)
            tr_X_flat, tr_y_flat = _flatten_residue(tr_emb, tr_lbl, seed=seed)
            te_X_flat, te_y_flat = _flatten_residue(te_emb, te_lbl, seed=seed)
        else:
            tr_y_flat = tr_lbl
            te_y_flat = te_lbl

    # Multilabel: convert ragged → 2D
    if task_type == 'multilabel' and tr_lbl.dtype == object:
        n_cls = max(len(x) for x in tr_lbl)
        tr_lbl_2d = np.zeros((len(tr_lbl), n_cls), dtype=np.float32)
        for i, x in enumerate(tr_lbl): tr_lbl_2d[i, :len(x)] = x
        te_lbl_2d = np.zeros((len(te_lbl), n_cls), dtype=np.float32)
        for i, x in enumerate(te_lbl): te_lbl_2d[i, :len(x)] = x
        tr_lbl = tr_lbl_2d; te_lbl = te_lbl_2d

    scaler = StandardScaler()

    # ── Residue-level binary (AUPRC / Q3) ────────────────────────────────────
    if metric in ('AUPRC', 'Q3_accuracy'):
        X_tr = scaler.fit_transform(tr_X_flat.astype(np.float32))
        X_te = scaler.transform(te_X_flat.astype(np.float32))
        probe = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200,
                              early_stopping=True, random_state=seed)
        probe.fit(X_tr, tr_y_flat)
        y_pred = probe.predict(X_te)
        if metric == 'Q3_accuracy':
            return float(accuracy_score(te_y_flat, y_pred))
        # AUPRC
        y_proba = probe.predict_proba(X_te)[:, 1]
        return float(average_precision_score(te_y_flat, y_proba))

    # ── Protein-level clf / reg / multilabel ─────────────────────────────────
    X_tr = scaler.fit_transform(tr_X)
    X_te = scaler.transform(te_X)

    if task_type in ('clf', 'reg'):
        if task_type == 'clf':
            probe = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200,
                                  early_stopping=True, random_state=seed)
        else:
            probe = MLPRegressor(hidden_layer_sizes=(256,), max_iter=200,
                                 early_stopping=True, random_state=seed)
        probe.fit(X_tr, tr_lbl)
        y_pred = probe.predict(X_te)

        if metric == 'spearman_rho':
            rho, _ = spearmanr(te_lbl.ravel(), y_pred.ravel())
            return float(rho) if not np.isnan(rho) else 0.0
        if metric == 'accuracy':
            return float(accuracy_score(te_lbl, y_pred))
        if metric == 'AUROC':
            proba = probe.predict_proba(X_te)[:, 1]
            return float(roc_auc_score(te_lbl, proba))

    if task_type == 'multilabel':
        probe = MLPRegressor(hidden_layer_sizes=(256,), max_iter=200,
                             early_stopping=True, random_state=seed)
        probe.fit(X_tr, tr_lbl)
        y_scores = probe.predict(X_te)
        y_scores = np.clip(y_scores, 0, 1)
        return _fmax(te_lbl, y_scores)

    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Master evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(encoder, task):
    cfg  = TASKS[task]
    data = load_embeddings(task, encoder)
    if data is None:
        return None

    if cfg['type'] == 'contact':
        scores = []
        for seed in SEEDS:
            s = _eval_contact(data, seed)
            if not np.isnan(s):
                scores.append(s)
        if not scores:
            return None
        return {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'n': len(scores)}

    scores = []
    for seed in SEEDS:
        try:
            s = _eval_seed(task, cfg, data, seed)
            if s is not None and not np.isnan(s):
                scores.append(s)
        except Exception as e:
            pass
    if not scores:
        return None
    return {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'n': len(scores)}


# ─────────────────────────────────────────────────────────────────────────────
# Build / load results CSV
# ─────────────────────────────────────────────────────────────────────────────

# Tasks where no per-protein PDB structure exists — structure encoders used dummy coords
# Results for structure encoders on these tasks are meaningless (show as 0)
SEQ_ONLY_TASKS_NO_STRUCT = {
    "secondary_structure", "contact_prediction", "remote_homology",
    "subcellular_localization", "membrane_soluble",
}

STRUCTURE_ENCODERS = {e for e, m in ENCODER_MODALITY.items() if m == 'Structure'}


def build_results(force=False, quick_encoders=None, priority_encoders=None):
    csv_path = OUT_DIR / "downstream_results.csv"

    # Load existing results for resume (unless --force)
    if csv_path.exists() and not force:
        existing_df = pd.read_csv(csv_path)
        done = set(zip(existing_df['encoder'], existing_df['task']))
        print(f"Loaded {len(existing_df)} cached results from {csv_path}")
    else:
        existing_df = pd.DataFrame()
        done = set()
        if force and csv_path.exists():
            csv_path.unlink()

    # Discover all (encoder, task) combos from embeddings AND pre-computed results
    combos = set()

    # 1) Scan embeddings directory
    for task_dir in sorted(EMB_DIR.iterdir()):
        if not task_dir.is_dir() or task_dir.name not in TASKS:
            continue
        for enc_dir in sorted(task_dir.iterdir()):
            if enc_dir.is_dir() and (enc_dir / "meta.json").exists():
                combos.add((enc_dir.name, task_dir.name))

    # 2) Scan pre-computed results directory
    if RES_DIR.exists():
        for enc_dir in sorted(RES_DIR.iterdir()):
            if not enc_dir.is_dir():
                continue
            for task_dir in sorted(enc_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                agg_f = task_dir / "aggregated.json"
                if agg_f.exists() and task_dir.name in TASKS:
                    combos.add((enc_dir.name, task_dir.name))

    combos = sorted(combos)

    if quick_encoders:
        combos = [(e, t) for e, t in combos if e in quick_encoders]

    # Filter out already-done combos
    todo = [(e, t) for e, t in combos if (e, t) not in done]

    # Priority ordering: put priority encoders first
    if priority_encoders:
        pri  = [(e, t) for e, t in todo if e in priority_encoders]
        rest = [(e, t) for e, t in todo if e not in priority_encoders]
        todo = pri + rest
        print(f"  Priority encoders ({len(pri)} combos first): {priority_encoders}")

    print(f"  {len(done)} already done, {len(todo)} remaining")
    if not todo:
        return existing_df if not existing_df.empty else pd.DataFrame()

    # Evaluate with incremental saves
    rows = list(existing_df.to_dict('records')) if not existing_df.empty else []
    for encoder, task in tqdm(todo):
        mean_score = std_score = 0.0
        n_seeds = 0

        # Structure encoders on tasks with no per-protein PDB → force 0
        if encoder in STRUCTURE_ENCODERS and task in SEQ_ONLY_TASKS_NO_STRUCT:
            pass  # leave as 0

        else:
            # Try pre-computed results from evaluate_tasks.py (no live eval fallback)
            agg_f = RES_DIR / encoder / task / "aggregated.json"
            if agg_f.exists():
                try:
                    agg = json.load(open(agg_f))
                    mean_score = float(agg.get('best_mean', 0.0) or 0.0)
                    # std from best probe
                    bp = agg.get('best_probe', '')
                    sc = agg.get('scores', {}).get(bp, {})
                    std_score = float(sc.get('std', 0.0) or 0.0)
                    n_seeds   = int(sc.get('n', 5))
                except Exception:
                    pass
            # If no pre-computed result, leave as 0 (don't do slow live evaluation)

        rows.append({
            'encoder':    encoder,
            'task':       task,
            'task_num':   TASKS[task]['T'],
            'metric':     TASKS[task]['metric'],
            'mean_score': mean_score,
            'std_score':  std_score,
            'n_seeds':    n_seeds,
        })
        # Save after each evaluation for crash safety
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} results → {csv_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build encoder × task matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_matrix(df):
    all_enc = sorted(df['encoder'].unique())
    encoders = [e for e in all_enc if is_shown(e, all_enc)]

    def sort_key(e):
        order = {
            'Sequence': 0, 'Struct: Token': 1, 'Struct: Graph': 2,
            'Struct: Equiv': 3, 'Multimodal': 4, 'Baseline': 5
        }
        stype = ENCODER_SUBTYPE.get(e, 'Baseline')
        return (order.get(stype, 6), e)

    encoders = sorted(encoders, key=sort_key)
    n = len(encoders)
    m = len(TASK_ORDER)
    mean_mat = np.zeros((n, m))   # 0 for missing/not-applicable
    std_mat  = np.zeros((n, m))

    # Task-number → task-name reverse lookup
    tnum_to_name = {cfg['T']: tname for tname, cfg in TASKS.items()}

    for i, enc in enumerate(encoders):
        sub = df[df['encoder'] == enc]
        for j, tnum in enumerate(TASK_ORDER):
            row = sub[sub['task_num'] == tnum]
            if len(row):
                mean_mat[i, j] = row['mean_score'].values[0]
                std_mat[i, j]  = row['std_score'].values[0]
            # Structure encoders on sequence-only tasks → force 0
            tname = tnum_to_name.get(tnum, '')
            if enc in STRUCTURE_ENCODERS and tname in SEQ_ONLY_TASKS_NO_STRUCT:
                mean_mat[i, j] = 0.0
                std_mat[i, j]  = 0.0

    return encoders, mean_mat, std_mat


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig1_heatmap(encoders, mean_mat):
    n_enc, n_task = mean_mat.shape
    # Adjusted height for 15 tasks (vs 10 in v12) to prevent y-axis overlap
    # Use larger multiplier to give more vertical space
    fig_height = 4.5 * n_task / n_enc + 1.0
    fig, ax = plt.subplots(figsize=(4.5, fig_height))

    im = ax.imshow(mean_mat.T, cmap=HMAP_CMAP, aspect='auto',
                   vmin=0.0, vmax=1.0, interpolation='nearest')

    disp_names = [disp(e) for e in encoders]
    ax.set_xticks(np.arange(n_enc))
    ax.set_xticklabels(disp_names, rotation=45, fontsize=5, ha='right')
    for tick, enc in zip(ax.get_xticklabels(), encoders):
        stype = ENCODER_SUBTYPE.get(enc, 'Baseline')
        tick.set_color(SUBTYPE_COLORS.get(stype, '#333'))

    # white group separators
    prev_stype = None
    for i, enc in enumerate(encoders):
        stype = ENCODER_SUBTYPE.get(enc, 'Baseline')
        if stype != prev_stype and prev_stype is not None:
            ax.axvline(i - 0.5, color='white', linewidth=2.5)
        prev_stype = stype

    ax.set_yticks(np.arange(n_task))
    ax.set_yticklabels([TASK_SHORT_1LINE[t] for t in TASK_ORDER],
                       fontsize=5, rotation=45, ha='right', va='center')

    # Add NaN hatching
    for i in range(n_enc):
        for j in range(n_task):
            if np.isnan(mean_mat[i, j]):
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1,
                             fill=True, facecolor='#cccccc', alpha=0.6, zorder=2))

    # Colorbar - match v12 styling
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    # cbar.set_label('Performance Score', rotation=270, labelpad=14, fontsize=8.5)
    cbar.ax.tick_params(labelsize=6)

    # Title and labels removed to match v12
    # ax.set_title('Downstream Task Performance: Encoders × 15 Tasks', fontsize=11, pad=10)

    legend_elements = [
        mpatches.Patch(facecolor=MODALITY_COLORS['Sequence'],   label='Sequence'),
        mpatches.Patch(facecolor=MODALITY_COLORS['Structure'],  label='Structure'),
        mpatches.Patch(facecolor=MODALITY_COLORS['Multimodal'], label='Multimodal'),
        mpatches.Patch(facecolor=MODALITY_COLORS['Baseline'],   label='Baseline'),
        mpatches.Patch(facecolor='#cccccc',                      label='Missing'),
    ]
    # ax.legend(handles=legend_elements, loc='upper left',
    #           bbox_to_anchor=(1.06, 1.0), frameon=True, fontsize=8)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(OUT_DIR / f'fig1_heatmap.{ext}', dpi=300, bbox_inches='tight')
    print("  saved fig1")
    plt.close()


def radar_chart(ax, vals_list, labels, line_colors, mean_color, title, ylim=1.0):
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, ylim)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=4.5, color='grey')
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.spines['polar'].set_visible(False)

    # Store all encoder values for overlap detection
    all_vals = []

    # Individual encoder lines
    for i, v in enumerate(vals_list):
        all_vals.append(np.where(np.isnan(v), 0.0, v))
        filled = np.where(np.isnan(v), 0.0, v)
        data = np.concatenate([filled, filled[:1]])
        ax.plot(angles, data, linewidth=0.6, alpha=0.4, color=line_colors[i])
        ax.fill(angles, data, alpha=0.05, color=line_colors[i])

    # Mean line (skip NaN per dimension)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        means = np.nanmean(vals_list, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    data = np.concatenate([means, means[:1]])
    ax.plot(angles, data, linewidth=1.5, color=mean_color,
            marker='o', markersize=2.5,
            markeredgecolor='white', markeredgewidth=0.5, zorder=5)
    ax.fill(angles, data, alpha=0.20, color=mean_color)

    # Smart label positioning to avoid overlap with radar grid
    # Place labels well outside the plot area
    for angle, label_text in zip(angles[:-1], labels):
        x_pos = angle
        # Check if any encoder has high value (>0.85) at this task
        task_idx = angles[:-1].index(angle)
        max_val = max(vals[task_idx] for vals in all_vals)

        # Push labels further out to avoid overlap with grid lines
        if max_val > 0.85:
            label_radius = ylim * 1.35  # Push much further out for high values
        else:
            label_radius = ylim * 1.25  # Push further out for normal values

        # Place label with white background to improve readability
        ax.text(x_pos, label_radius, label_text,
                ha='center', va='center', fontsize=5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.8))

    # Clear default thetagrid labels to avoid duplication
    ax.set_thetagrids(np.degrees(angles[:-1]), [])
    ax.set_xticklabels([])

    # Title removed - will be added in paper caption
    # ax.set_title(title, fontweight='bold', pad=18, fontsize=10)


def radar_chart_compact(ax, vals_list, labels, line_colors, mean_color, ylim=1.0):
    """Compact radar chart with minimal details - just shape and large labels."""
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, ylim)

    # Remove all grid elements for minimal look
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.spines['polar'].set_visible(False)

    # Individual encoder lines - thinner and more transparent
    for i, v in enumerate(vals_list):
        filled = np.where(np.isnan(v), 0.0, v)
        data = np.concatenate([filled, filled[:1]])
        ax.plot(angles, data, linewidth=0.4, alpha=0.3, color=line_colors[i])
        ax.fill(angles, data, alpha=0.03, color=line_colors[i])

    # Mean line - emphasize the shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        means = np.nanmean(vals_list, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    data = np.concatenate([means, means[:1]])
    ax.plot(angles, data, linewidth=1.2, color=mean_color, zorder=5)
    ax.fill(angles, data, alpha=0.15, color=mean_color)

    # Labels positioned outside - smaller to emphasize radar shape
    for angle, label_text in zip(angles[:-1], labels):
        label_radius = ylim * 1.4
        ax.text(angle, label_radius, label_text,
                ha='center', va='center', fontsize=6,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.9))

    ax.set_thetagrids(np.degrees(angles[:-1]), [])
    ax.set_xticklabels([])


# Light-to-dark palettes matching visualize_paper_v12.py
SEQ_PALETTE = ['#B0E0E6', '#87CEEB', '#5bacd4', '#4682B4', '#4169E1', '#0000CD', '#00008B', '#1a4a8a']
STRUCT_PALETTE = [
    '#FFF4D6', '#FFEAB3', '#FFE0A3', '#FFD580', '#FFCC66', '#FFC04D',
    '#FFB347', '#FFA500', '#F4A460', '#F39C12', '#E67E22', '#D2691E',
    '#CD7F32', '#B8621B', '#A0522D', '#8B2500'
]
MULTI_PALETTE = ['#c9a0e0', '#b07fd4', '#9966bb', '#7c4fa0', '#4d2475']
BASELINE_PALETTE = ['#cccccc', '#888888', '#444444']

MODALITY_PALETTES = {
    'Sequence':  SEQ_PALETTE,
    'Structure': STRUCT_PALETTE,
    'Multimodal': MULTI_PALETTE,
    'Baseline':  BASELINE_PALETTE,
}

def make_palette(base_color, n):
    """Pick n colors from the modality palette, or generate shades."""
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(base_color))
    palette = []
    for i in range(n):
        t = 0.4 + 0.6 * (i / max(n - 1, 1))
        palette.append(tuple(np.clip(rgb * t + (1 - t) * np.array([1, 1, 1]), 0, 1)))
    return palette


def fig_radar(encoders, mean_mat, modality, fig_num, filename, title_prefix):
    enc_idx  = [i for i, e in enumerate(encoders) if ENCODER_MODALITY.get(e) == modality]
    encs     = [encoders[i] for i in enc_idx]
    vals     = [mean_mat[i] for i in enc_idx]
    if not encs:
        return

    # Hide dimensions where >50% of encoders lack data (score == 0)
    vals_arr = np.array(vals)                   # (n_encoders, n_tasks)
    n_enc = len(encs)
    keep_mask = np.array([(np.count_nonzero(vals_arr[:, j]) / n_enc) > 0.5
                          for j in range(vals_arr.shape[1])])
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) == 0:
        print(f"  [SKIP] {filename}: all dimensions lack data for >50% of encoders")
        return
    vals_filtered     = [v[keep_mask] for v in vals]
    labels_filtered   = [TASK_SHORT_1LINE[TASK_ORDER[j]] for j in keep_idx]
    n_hidden = int(np.sum(~keep_mask))
    if n_hidden > 0:
        hidden_tasks = [TASK_SHORT_1LINE[TASK_ORDER[j]] for j in np.where(~keep_mask)[0]]
        print(f"  {filename}: hiding {n_hidden} dimensions (>50% missing): {hidden_tasks}")

    base_col = MODALITY_COLORS[modality]
    mod_pal  = MODALITY_PALETTES.get(modality)
    if mod_pal and len(encs) <= len(mod_pal):
        palette = mod_pal[:len(encs)]
    else:
        palette = make_palette(base_col, len(encs))

    fig, ax = plt.subplots(figsize=(2.25, 2.25), subplot_kw=dict(projection='polar'))
    radar_chart(ax, vals_filtered,
                labels=labels_filtered,
                line_colors=palette,
                mean_color=base_col,
                title=f'{title_prefix}  (n={len(encs)})')

    # Legend removed - see fig_legend_all.png for standalone legend

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(OUT_DIR / f'{filename}.{ext}', dpi=300, bbox_inches='tight')
    print(f"  saved {filename}")
    plt.close()


def fig_radar_grouped(encoders, mean_mat, modality, filename, title_prefix):
    """Radar chart with tasks grouped by category (5 groups instead of 15 tasks)."""
    enc_idx  = [i for i, e in enumerate(encoders) if ENCODER_MODALITY.get(e) == modality]
    encs     = [encoders[i] for i in enc_idx]
    vals     = [mean_mat[i] for i in enc_idx]
    if not encs:
        return

    # Compute mean score per task group for each encoder
    group_names = list(TASK_GROUPS.keys())
    n_groups = len(group_names)
    n_enc = len(encs)

    grouped_vals = []
    for enc_vals in vals:
        group_scores = []
        for group_name in group_names:
            task_nums = TASK_GROUPS[group_name]
            # Get indices in TASK_ORDER
            task_indices = [TASK_ORDER.index(t) for t in task_nums]
            # Get scores for this group
            scores = [enc_vals[idx] for idx in task_indices]
            # Filter out NaN and zeros (missing data)
            valid_scores = [s for s in scores if not np.isnan(s) and s > 0]
            if valid_scores:
                group_scores.append(np.mean(valid_scores))
            else:
                group_scores.append(np.nan)
        grouped_vals.append(np.array(group_scores))

    # Check if we have enough data
    grouped_arr = np.array(grouped_vals)
    keep_mask = np.array([(np.count_nonzero(~np.isnan(grouped_arr[:, j])) / n_enc) > 0.5
                          for j in range(n_groups)])
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) == 0:
        print(f"  [SKIP] {filename}: all groups lack data for >50% of encoders")
        return

    grouped_vals_filtered = [v[keep_mask] for v in grouped_vals]
    labels_filtered = [TASK_GROUP_SHORT[group_names[j]] for j in keep_idx]

    n_hidden = int(np.sum(~keep_mask))
    if n_hidden > 0:
        hidden_groups = [group_names[j] for j in np.where(~keep_mask)[0]]
        print(f"  {filename}: hiding {n_hidden} groups (>50% missing): {hidden_groups}")

    base_col = MODALITY_COLORS[modality]
    mod_pal  = MODALITY_PALETTES.get(modality)
    if mod_pal and len(encs) <= len(mod_pal):
        palette = mod_pal[:len(encs)]
    else:
        palette = make_palette(base_col, len(encs))

    # Compact version: 2 inch main figure with smaller labels to emphasize radar shape
    fig, ax = plt.subplots(figsize=(2.0, 2.0), subplot_kw=dict(projection='polar'))
    radar_chart_compact(ax, grouped_vals_filtered,
                        labels=labels_filtered,
                        line_colors=palette,
                        mean_color=base_col)

    # Legend removed

    plt.tight_layout(pad=0.1)
    for ext in ('png', 'pdf'):
        plt.savefig(OUT_DIR / f'{filename}.{ext}', dpi=300, bbox_inches='tight')
    print(f"  saved {filename}")
    plt.close()


def fig5_pca(encoders, mean_mat):
    main_enc = [e for e in encoders
                if ENCODER_MODALITY.get(e) in ('Sequence', 'Structure', 'Multimodal')]
    # Only use encoders that have at least 5 tasks
    valid = [(e, mean_mat[encoders.index(e)]) for e in main_enc
             if np.sum(~np.isnan(mean_mat[encoders.index(e)])) >= 5]
    if len(valid) < 3:
        print("  [SKIP] Not enough encoders for PCA")
        return
    encs_pca, vecs = zip(*valid)
    X = np.array(vecs, dtype=np.float32)
    # Fill NaN with column mean
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]

    pca = PCA(n_components=2)
    X2  = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    # Match v12: 6.75" x 6.75" square figure
    fig, ax = plt.subplots(figsize=(6.75, 6.75))
    for enc, (x, y) in zip(encs_pca, X2):
        stype = ENCODER_SUBTYPE.get(enc, ENCODER_MODALITY.get(enc))
        color = SUBTYPE_COLORS.get(stype, MODALITY_COLORS.get(ENCODER_MODALITY.get(enc), '#333'))
        ax.scatter(x, y, color=color, s=55, edgecolors='white', linewidths=0.6, zorder=3)
        ax.annotate(disp(enc), xy=(x, y), xytext=(3, 2),
                    textcoords='offset points', fontsize=5.2,
                    color=color, alpha=0.85)

    # Convex hulls per modality
    for mod in ('Sequence', 'Structure', 'Multimodal'):
        idx = [i for i, e in enumerate(encs_pca) if ENCODER_MODALITY.get(e) == mod]
        if len(idx) < 3:
            continue
        pts = X2[idx]
        try:
            hull = ConvexHull(pts)
            for s in hull.simplices:
                ax.plot(pts[s, 0], pts[s, 1], color=MODALITY_COLORS[mod],
                        alpha=0.4, linewidth=1.0, linestyle='--')
        except Exception:
            pass

    ax.set_xlabel(f'PC1  ({var[0]*100:.1f}% variance)', fontweight='bold')
    ax.set_ylabel(f'PC2  ({var[1]*100:.1f}% variance)', fontweight='bold')
    # Title removed to match v12
    # ax.set_title('PCA of Downstream Performance Vectors', fontsize=10)
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    legend_elements = [
        mpatches.Patch(facecolor=MODALITY_COLORS[m], label=m, edgecolor='black', lw=0.5)
        for m in ('Sequence', 'Structure', 'Multimodal')
    ] + [
        mpatches.Patch(facecolor=SUBTYPE_COLORS[s], label=s, edgecolor='black', lw=0.5)
        for s in ('Struct: Token', 'Struct: Graph', 'Struct: Equiv')
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='best',
              frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(OUT_DIR / f'fig5_pca.{ext}', dpi=300, bbox_inches='tight')
    print("  saved fig5")
    plt.close()


def save_encoder_summary(encoders, mean_mat, std_mat):
    rows = []
    for i, enc in enumerate(encoders):
        valid = mean_mat[i][~np.isnan(mean_mat[i])]
        rows.append({
            'encoder':        disp(enc),
            'modality':       ENCODER_MODALITY.get(enc, 'Unknown'),
            'n_tasks':        int(np.sum(~np.isnan(mean_mat[i]))),
            'mean_score':     float(np.nanmean(mean_mat[i])) if len(valid) else np.nan,
            'std_score':      float(np.nanstd(mean_mat[i]))  if len(valid) else np.nan,
            'min_score':      float(valid.min())             if len(valid) else np.nan,
            'max_score':      float(valid.max())             if len(valid) else np.nan,
            **{TASK_ORDER[j]: (float(mean_mat[i, j]) if not np.isnan(mean_mat[i, j]) else '')
               for j in range(len(TASK_ORDER))},
        })
    summary = pd.DataFrame(rows).sort_values('mean_score', ascending=False)
    summary.to_csv(OUT_DIR / 'encoder_summary.csv', index=False)
    print(f"\nTop 10 by mean score:")
    print(summary[['encoder', 'modality', 'n_tasks', 'mean_score']].head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Re-run all evaluations')
    parser.add_argument('--quick', action='store_true', help='Debug: 5 encoders only')
    parser.add_argument('--priority', type=str, default=None,
                        help='Comma-separated encoder names to evaluate first')
    args = parser.parse_args()

    quick_encs = ['ankh', 'esm2', 'gvp', 'schnet', 'esm3'] if args.quick else None
    priority_encs = [e.strip() for e in args.priority.split(',')] if args.priority else None

    print("=" * 60)
    print("Downstream Task Benchmark — Evaluation + Figures (MLP probe)")
    print("=" * 60)

    df = build_results(force=args.force, quick_encoders=quick_encs,
                       priority_encoders=priority_encs)
    print(f"\nLoaded {len(df)} results | {df['encoder'].nunique()} encoders | {df['task'].nunique()} tasks")

    print("\nBuilding encoder × task matrix…")
    encoders, mean_mat, std_mat = build_matrix(df)
    print(f"  {len(encoders)} encoders (after pretrained-only filter) × {len(TASK_ORDER)} tasks")

    print("\nGenerating figures…")
    fig1_heatmap(encoders, mean_mat)
    fig_radar(encoders, mean_mat, 'Sequence',  2, 'fig2_sequence_radar',  'Sequence Encoders')
    fig_radar(encoders, mean_mat, 'Structure', 3, 'fig3_structure_radar', 'Structure Encoders')
    fig_radar(encoders, mean_mat, 'Multimodal',4, 'fig4_multimodal_radar','Multimodal Encoders')
    fig5_pca(encoders, mean_mat)

    print("\nGenerating grouped radar charts (by task category)…")
    fig_radar_grouped(encoders, mean_mat, 'Sequence',  'fig2_sequence_radar_grouped',  'Sequence Encoders')
    fig_radar_grouped(encoders, mean_mat, 'Structure', 'fig3_structure_radar_grouped', 'Structure Encoders')
    fig_radar_grouped(encoders, mean_mat, 'Multimodal','fig4_multimodal_radar_grouped','Multimodal Encoders')

    print("\nExporting summary CSV…")
    save_encoder_summary(encoders, mean_mat, std_mat)

    print(f"\nAll outputs → {OUT_DIR}/")


if __name__ == '__main__':
    main()
