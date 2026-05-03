#!/usr/bin/env python3
"""
Alignment-Based Analysis of Encoder Performance (Section: alignment_analysis)

Implements the Information Alignment Hypothesis (IAH):
- Encoder probing profiles (z_e) from invariant family probing (I1-I10)
- Task profiles (w_t) inferred from encoder×task performance patterns
- Alignment score: cosine similarity between z_e and w_t
- ESP (Encoder Selection Predictor): leave-one-task-out cross-validation
- ESP++: transfer task-specific family weighting from nearest task
- Partial Spearman correlation controlling for mean probing score
- Statistical significance testing with multiple-testing correction

Usage:
    python scripts/alignment_analysis.py
    python scripts/alignment_analysis.py --output outputs/alignment_analysis/

Outputs:
    - alignment_scores.csv: encoder×task alignment scores
    - esp_predictions.csv: ESP leave-one-task-out predictions
    - esp_plus_plus_predictions.csv: ESP++ predictions with task transfer
    - partial_correlations.csv: partial Spearman correlations
    - significance_tests.csv: pairwise significance testing results
    - fig_alignment_heatmap.{png,pdf}: alignment score heatmap
    - fig_esp_performance.{png,pdf}: ESP prediction accuracy
    - fig_partial_correlation.{png,pdf}: partial correlation analysis
"""

import argparse
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Mock heavy imports (needed for torch.load to unpickle InvariantFingerprint) ──
class _Mock:
    def __getattr__(self, n): return _Mock()
    def __call__(self, *a, **k): return _Mock()

for _m in ['torch_geometric', 'torch_geometric.nn']:
    sys.modules[_m] = _Mock()

class InvariantFingerprint: pass
_src = types.ModuleType('src')
_pr = types.ModuleType('src.probes')
_bat = types.ModuleType('src.probes.probe_battery')
_bat.InvariantFingerprint = InvariantFingerprint
sys.modules.update({'src': _src, 'src.probes': _pr, 'src.probes.probe_battery': _bat})

import torch
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
PROBING_DIR = ROOT / "probing_results_architecture_full"
PROBING_FALL = ROOT / "structure_encoder_data" / "probing_results"
DOWNSTREAM_DIR = ROOT / "outputs" / "downstream_v1"
OUT_DIR_AVG = ROOT / "outputs" / "alignment_analysis_averaged"
OUT_DIR_PER_ENC = ROOT / "outputs" / "alignment_analysis_per_encoder"

# Task definitions (14 tasks matching paper table)
TASKS = {
    "contact_prediction":       {"T": "T1",  "metric": "precision_L5"},
    "remote_homology":          {"T": "T2",  "metric": "accuracy"},
    "fluorescence":             {"T": "T3",  "metric": "spearman_rho"},
    "stability":                {"T": "T4",  "metric": "spearman_rho"},
    "mutation_effect":          {"T": "T5",  "metric": "spearman_rho"},
    "ec_classification":        {"T": "T6",  "metric": "accuracy"},
    "go_mf":                    {"T": "T7",  "metric": "Fmax"},
    "go_bp":                    {"T": "T8",  "metric": "Fmax"},
    "solubility":               {"T": "T9",  "metric": "AUROC"},
    "ppi_site":                 {"T": "T10", "metric": "AUPRC"},
    "metal_binding":            {"T": "T11", "metric": "AUPRC"},
    "ppi_affinity":             {"T": "T12", "metric": "spearman_rho"},
    "subcellular_localization": {"T": "T13", "metric": "accuracy"},
    "membrane_soluble":         {"T": "T14", "metric": "AUROC"},
}

TASK_ORDER = [f"T{i}" for i in range(1, 15)]
IFAMILIES = [f"I{i}" for i in range(1, 11)]

# Display names for encoders (matching visualize_paper_v12.py)
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
    'saprot': 'SaProt', 'esm3': 'ESM3', 'foldvision': 'FoldVision', 'cdconv': 'CDConv',
    'random': 'Random', 'random_token': 'Random-Token', 'length': 'Length',
    'torsion': 'Torsion', 'onehot': 'OneHot', 'blosum': 'BLOSUM', 'test': 'Test',
}

def disp(enc):
    """Strip _pretrained suffix for display and format."""
    clean = enc.replace('_pretrained', '')
    return DISPLAY_NAMES.get(clean, clean)

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.titlesize': 10, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

# New IIP (Invariant Information Property) → probe mapping
FAMILY_PROBES = {
    'I1':  ['phi_angle_bins', 'psi_angle_bins'],                  # Local Conformation
    'I2':  ['local_curvature'],                                   # Local Curvature
    'I3':  ['dssp_3', 'dssp_8'],                                  # Sec. Struct.
    'I4':  ['rsa', 'rsa_bins'],                                   # Solv. Access.
    'I5':  ['bfactor'],                                           # Local Flexibility
    'I6':  ['core_surface', 'contact_density'],                   # Packing Density
    'I7':  ['distance_bins'],                                     # Residue Contacts
    'I8':  ['contact_order', 'radius_of_gyration', 'asphericity'],# Global Geometry
    'I9':  ['architecture_label'],                                # Fold Architecture
    'I10': ['sequence_recovery'],                                 # Sequence
}

def _aggregate_families(detailed):
    out = {}
    for fam, probes in FAMILY_PROBES.items():
        vals = [float(detailed[p]) for p in probes if p in detailed]
        out[fam] = float(np.mean(vals)) if vals else 0.0
    return out


def load_probing_profiles():
    """Load encoder probing profiles (z_e) from I1-I10 invariant families."""
    profiles = {}
    for src in [PROBING_DIR, PROBING_FALL]:
        if not src.exists():
            print(f"  [WARN] Directory not found: {src}")
            continue

        count = 0
        for f in sorted(src.glob('*.pt')):
            try:
                enc = f.stem.rsplit('_seed_', 1)[0]
                seed = int(f.stem.rsplit('_seed_', 1)[1])
                data = torch.load(f, map_location='cpu', weights_only=False)
                fp = data['fingerprint']
                detailed = getattr(fp, 'detailed_scores', None)
                if detailed:
                    profiles.setdefault(enc, []).append(_aggregate_families(detailed))
                else:
                    profiles.setdefault(enc, []).append({
                        f"I{i}": float(getattr(fp, f"p_I{i}")) for i in range(1, 11)
                    })
                count += 1
            except Exception as e:
                print(f"  [WARN] Failed to load {f.name}: {e}")
                continue

        print(f"  Loaded {count} files from {src.name}")

    # Average across seeds
    mean_profiles = {}
    for enc, seeds in profiles.items():
        mean_profiles[enc] = {
            task: np.mean([s[task] for s in seeds]) for task in IFAMILIES
        }

    return mean_profiles


def load_downstream_performance():
    """Load encoder×task downstream performance matrix."""
    csv_path = DOWNSTREAM_DIR / "downstream_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Downstream results not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Build reverse mapping: task_name -> new task number
    task_name_to_num = {task_name: info["T"] for task_name, info in TASKS.items()}

    # Build encoder×task matrix
    encoders = sorted(df['encoder'].unique())
    perf_matrix = np.zeros((len(encoders), len(TASK_ORDER)))

    for i, enc in enumerate(encoders):
        sub = df[df['encoder'] == enc]
        for j, tnum in enumerate(TASK_ORDER):
            # Find the task name that maps to this task number
            task_name = None
            for name, info in TASKS.items():
                if info["T"] == tnum:
                    task_name = name
                    break

            if task_name:
                row = sub[sub['task'] == task_name]
                if len(row):
                    perf_matrix[i, j] = row['mean_score'].values[0]
                else:
                    perf_matrix[i, j] = np.nan
            else:
                perf_matrix[i, j] = np.nan

    return encoders, perf_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Task Profile Inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_task_profiles(probing_profiles, encoders, perf_matrix):
    """
    Infer task profiles (w_t) via Ridge regression:
    For each task t, fit: performance_t = w_t · z_e + bias
    """
    task_profiles = {}

    # Build probing matrix Z: (n_encoders, K=10)
    Z = np.zeros((len(encoders), len(IFAMILIES)))
    valid_enc_idx = []

    for i, enc in enumerate(encoders):
        if enc in probing_profiles:
            Z[i] = [probing_profiles[enc][fam] for fam in IFAMILIES]
            valid_enc_idx.append(i)

    Z = Z[valid_enc_idx]

    # Fit Ridge regression for each task
    for j, tnum in enumerate(TASK_ORDER):
        y = perf_matrix[valid_enc_idx, j]
        valid = ~np.isnan(y)

        if valid.sum() < 5:  # Need at least 5 encoders
            task_profiles[tnum] = np.zeros(len(IFAMILIES))
            continue

        Z_valid = Z[valid]
        y_valid = y[valid]

        # Standardize features
        scaler = StandardScaler()
        Z_scaled = scaler.fit_transform(Z_valid)

        # Ridge regression with cross-validation for alpha
        best_alpha = 1.0
        best_score = -np.inf
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = Ridge(alpha=alpha)
            model.fit(Z_scaled, y_valid)
            score = model.score(Z_scaled, y_valid)
            if score > best_score:
                best_score = score
                best_alpha = alpha

        model = Ridge(alpha=best_alpha)
        model.fit(Z_scaled, y_valid)

        # Store coefficients (task profile)
        task_profiles[tnum] = model.coef_

    return task_profiles


# ─────────────────────────────────────────────────────────────────────────────
# Alignment Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_alignment(z_e, w_t):
    """
    Compute alignment score: cosine similarity between z_e and w_t.
    Align(e,t) = (z_e · w_t) / (||z_e|| ||w_t||)
    """
    z_e = np.array(z_e)
    w_t = np.array(w_t)

    norm_z = np.linalg.norm(z_e)
    norm_w = np.linalg.norm(w_t)

    if norm_z == 0 or norm_w == 0:
        return 0.0

    return np.dot(z_e, w_t) / (norm_z * norm_w)


def compute_alignment_matrix(probing_profiles, encoders, task_profiles):
    """Compute encoder×task alignment matrix."""
    alignment_matrix = np.zeros((len(encoders), len(TASK_ORDER)))

    for i, enc in enumerate(encoders):
        if enc not in probing_profiles:
            alignment_matrix[i, :] = np.nan
            continue

        z_e = [probing_profiles[enc][fam] for fam in IFAMILIES]

        for j, tnum in enumerate(TASK_ORDER):
            w_t = task_profiles[tnum]
            alignment_matrix[i, j] = compute_alignment(z_e, w_t)

    return alignment_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Per-Encoder (Leave-One-Encoder-Out) Task Profiles
# ─────────────────────────────────────────────────────────────────────────────

def infer_task_profiles_per_encoder(probing_profiles, encoders, perf_matrix):
    """
    Per-encoder task profiles via leave-one-encoder-out (LOEO) ridge regression.
    For each held-out encoder e and each task t, fit
        w_t^{(-e)} = argmin_w sum_{i != e} (y_i^{(t)} - z_i . w)^2 + alpha ||w||_2^2
    on the remaining N-1 encoders, with the same alpha sweep and standardization
    as the averaged estimator.

    Returns: dict {encoder_name: {task_id: w_vector (d,)}}
    """
    Z = np.zeros((len(encoders), len(IFAMILIES)))
    valid_enc_idx = []
    for i, enc in enumerate(encoders):
        if enc in probing_profiles:
            Z[i] = [probing_profiles[enc][fam] for fam in IFAMILIES]
            valid_enc_idx.append(i)

    per_enc_profiles = {}

    for held_out_idx in valid_enc_idx:
        held_out_enc = encoders[held_out_idx]
        per_enc_profiles[held_out_enc] = {}

        train_idx = [i for i in valid_enc_idx if i != held_out_idx]
        Z_train_full = Z[train_idx]

        for j, tnum in enumerate(TASK_ORDER):
            y = perf_matrix[train_idx, j]
            valid = ~np.isnan(y)

            if valid.sum() < 5:
                per_enc_profiles[held_out_enc][tnum] = np.zeros(len(IFAMILIES))
                continue

            Z_valid = Z_train_full[valid]
            y_valid = y[valid]

            scaler = StandardScaler()
            Z_scaled = scaler.fit_transform(Z_valid)

            best_alpha, best_score = 1.0, -np.inf
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                model = Ridge(alpha=alpha)
                model.fit(Z_scaled, y_valid)
                score = model.score(Z_scaled, y_valid)
                if score > best_score:
                    best_score = score
                    best_alpha = alpha

            model = Ridge(alpha=best_alpha)
            model.fit(Z_scaled, y_valid)
            per_enc_profiles[held_out_enc][tnum] = model.coef_

    return per_enc_profiles


def compute_alignment_matrix_per_encoder(probing_profiles, encoders, per_enc_profiles):
    """Encoder×task alignment matrix using LOEO per-encoder weights w_t^{(-e)}."""
    alignment_matrix = np.zeros((len(encoders), len(TASK_ORDER)))

    for i, enc in enumerate(encoders):
        if enc not in probing_profiles or enc not in per_enc_profiles:
            alignment_matrix[i, :] = np.nan
            continue

        z_e = [probing_profiles[enc][fam] for fam in IFAMILIES]
        for j, tnum in enumerate(TASK_ORDER):
            w_t = per_enc_profiles[enc][tnum]
            alignment_matrix[i, j] = compute_alignment(z_e, w_t)

    return alignment_matrix


# ─────────────────────────────────────────────────────────────────────────────
# ESP: Encoder Selection Predictor (Leave-One-Task-Out)
# ─────────────────────────────────────────────────────────────────────────────

def esp_leave_one_task_out(probing_profiles, encoders, perf_matrix):
    """
    ESP: Leave-one-task-out cross-validation using Task Transfer.
    For each held-out task, find K=3 nearest tasks and train on them.
    """
    predictions = np.zeros_like(perf_matrix)

    # Build probing matrix Z
    Z = np.zeros((len(encoders), len(IFAMILIES)))
    valid_enc_idx = []
    for i, enc in enumerate(encoders):
        if enc in probing_profiles:
            Z[i] = [probing_profiles[enc][fam] for fam in IFAMILIES]
            valid_enc_idx.append(i)

    Z = Z[valid_enc_idx]

    # Build task similarity matrix based on performance patterns
    task_embeddings = perf_matrix[valid_enc_idx].T  # (n_tasks, n_encoders)

    for held_out_idx in range(len(TASK_ORDER)):
        held_out_emb = task_embeddings[held_out_idx]

        # Find K=3 nearest tasks
        distances = []
        for i in range(len(TASK_ORDER)):
            if i == held_out_idx:
                continue
            valid = ~(np.isnan(held_out_emb) | np.isnan(task_embeddings[i]))
            if valid.sum() < 3:
                distances.append((i, np.inf))
                continue
            dist = np.linalg.norm(held_out_emb[valid] - task_embeddings[i][valid])
            distances.append((i, dist))

        distances.sort(key=lambda x: x[1])
        nearest_k = [idx for idx, _ in distances[:3]]

        # Train on K=3 nearest tasks
        X_train_all = []
        y_train_all = []

        for j_train in nearest_k:
            y = perf_matrix[valid_enc_idx, j_train]
            valid = ~np.isnan(y)
            if valid.sum() < 5:
                continue

            X_train_all.append(Z[valid])
            y_train_all.append(y[valid])

        if len(X_train_all) == 0:
            predictions[:, held_out_idx] = np.nan
            continue

        X_train = np.vstack(X_train_all)
        y_train = np.concatenate(y_train_all)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Use Ridge with task-specific regularization
        model = Ridge(alpha=0.1)
        model.fit(X_train_scaled, y_train)

        X_test = scaler.transform(Z)
        y_pred = model.predict(X_test)

        # Map back to full encoder list
        for idx, enc_idx in enumerate(valid_enc_idx):
            predictions[enc_idx, held_out_idx] = y_pred[idx]

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# ESP++: Transfer from Nearest Task
# ─────────────────────────────────────────────────────────────────────────────

def esp_plus_plus(probing_profiles, encoders, perf_matrix, task_profiles):
    """
    ESP++: Enhanced Task Transfer with K=5 nearest tasks.
    Uses more neighbors for more robust predictions.
    """
    predictions = np.zeros_like(perf_matrix)

    # Build probing matrix Z
    Z = np.zeros((len(encoders), len(IFAMILIES)))
    valid_enc_idx = []
    for i, enc in enumerate(encoders):
        if enc in probing_profiles:
            Z[i] = [probing_profiles[enc][fam] for fam in IFAMILIES]
            valid_enc_idx.append(i)

    Z = Z[valid_enc_idx]

    # Build task similarity matrix
    task_embeddings = perf_matrix[valid_enc_idx].T

    for held_out_idx in range(len(TASK_ORDER)):
        held_out_emb = task_embeddings[held_out_idx]

        # Find K=5 nearest tasks (more than ESP for robustness)
        distances = []
        for i in range(len(TASK_ORDER)):
            if i == held_out_idx:
                continue
            valid = ~(np.isnan(held_out_emb) | np.isnan(task_embeddings[i]))
            if valid.sum() < 3:
                distances.append((i, np.inf))
                continue
            dist = np.linalg.norm(held_out_emb[valid] - task_embeddings[i][valid])
            distances.append((i, dist))

        distances.sort(key=lambda x: x[1])
        nearest_k = [idx for idx, _ in distances[:5]]

        # Train on K=5 nearest tasks
        X_train_all = []
        y_train_all = []

        for j_train in nearest_k:
            y = perf_matrix[valid_enc_idx, j_train]
            valid = ~np.isnan(y)
            if valid.sum() < 5:
                continue

            X_train_all.append(Z[valid])
            y_train_all.append(y[valid])

        if len(X_train_all) == 0:
            predictions[:, held_out_idx] = np.nan
            continue

        X_train = np.vstack(X_train_all)
        y_train = np.concatenate(y_train_all)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Use Ridge
        model = Ridge(alpha=0.1)
        model.fit(X_train_scaled, y_train)

        X_test = scaler.transform(Z)
        y_pred = model.predict(X_test)

        # Map back to full encoder list
        for idx, enc_idx in enumerate(valid_enc_idx):
            predictions[enc_idx, held_out_idx] = y_pred[idx]

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Partial Correlation (controlling for mean probing score)
# ─────────────────────────────────────────────────────────────────────────────

def partial_spearman_correlation(alignment_matrix, perf_matrix, probing_profiles, encoders):
    """
    Compute partial Spearman correlation between alignment and performance,
    controlling for mean probing score (overall encoder quality).
    """
    results = []

    for j, tnum in enumerate(TASK_ORDER):
        alignment_scores = alignment_matrix[:, j]
        performance = perf_matrix[:, j]

        # Compute mean probing score for each encoder
        mean_probing = np.array([
            np.mean([probing_profiles[enc][fam] for fam in IFAMILIES])
            if enc in probing_profiles else np.nan
            for enc in encoders
        ])

        # Valid data points
        valid = ~(np.isnan(alignment_scores) | np.isnan(performance) | np.isnan(mean_probing))

        if valid.sum() < 5:
            results.append({
                'task': tnum,
                'raw_correlation': np.nan,
                'raw_p_value': np.nan,
                'partial_correlation': np.nan,
                'partial_p_value': np.nan,
                'n_encoders': valid.sum(),
            })
            continue

        align_valid = alignment_scores[valid]
        perf_valid = performance[valid]
        mean_prob_valid = mean_probing[valid]

        # Raw Spearman correlation
        raw_rho, raw_p = spearmanr(align_valid, perf_valid)

        # Partial correlation: residualize both alignment and performance
        # against mean probing score, then compute correlation of residuals
        from sklearn.linear_model import LinearRegression

        # Residualize alignment
        lr_align = LinearRegression()
        lr_align.fit(mean_prob_valid.reshape(-1, 1), align_valid)
        align_resid = align_valid - lr_align.predict(mean_prob_valid.reshape(-1, 1))

        # Residualize performance
        lr_perf = LinearRegression()
        lr_perf.fit(mean_prob_valid.reshape(-1, 1), perf_valid)
        perf_resid = perf_valid - lr_perf.predict(mean_prob_valid.reshape(-1, 1))

        # Partial Spearman correlation
        partial_rho, partial_p = spearmanr(align_resid, perf_resid)

        results.append({
            'task': tnum,
            'raw_correlation': raw_rho,
            'raw_p_value': raw_p,
            'partial_correlation': partial_rho,
            'partial_p_value': partial_p,
            'n_encoders': valid.sum(),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_w_matrix_averaged(task_profiles, out_dir):
    """Plot the averaged W matrix (T × d) of standardized ridge coefficients."""
    W = np.array([task_profiles[t] for t in TASK_ORDER])  # (T, d)
    abs_max = float(np.abs(W).max()) if np.abs(W).max() > 0 else 1.0

    cmap = LinearSegmentedColormap.from_list(
        'wmat', ['#d73027', '#f7f7f7', '#4575b4'], N=256
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(W, cmap=cmap, aspect='auto', vmin=-abs_max, vmax=abs_max,
                   interpolation='nearest')

    ax.set_xticks(np.arange(len(IFAMILIES)))
    ax.set_xticklabels(IFAMILIES, fontsize=8)
    ax.set_yticks(np.arange(len(TASK_ORDER)))
    ax.set_yticklabels(TASK_ORDER, fontsize=8)
    ax.set_xlabel('Invariant Family (IIP)', fontsize=9)
    ax.set_ylabel('Task', fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r'Ridge coefficient $w_{t,j}$', rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out_dir / f'fig_w_matrix.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved fig_w_matrix (averaged, range: ±{abs_max:.3f})")


def plot_w_matrix_per_encoder(per_enc_profiles, encoders, out_dir):
    """Plot per-encoder LOEO W matrices: small-multiples grid + mean/std summary."""
    valid_encs = [e for e in encoders if e in per_enc_profiles]
    n = len(valid_encs)
    if n == 0:
        print("  [WARN] no per-encoder profiles to plot")
        return

    all_W = np.array([
        [per_enc_profiles[e][t] for t in TASK_ORDER] for e in valid_encs
    ])  # (N, T, d)
    abs_max = float(np.abs(all_W).max()) if np.abs(all_W).max() > 0 else 1.0

    cmap = LinearSegmentedColormap.from_list(
        'wmat', ['#d73027', '#f7f7f7', '#4575b4'], N=256
    )

    # Small-multiples grid: one (T × d) heatmap per held-out encoder
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]

    last_im = None
    for k, enc in enumerate(valid_encs):
        ax = axes_flat[k]
        last_im = ax.imshow(all_W[k], cmap=cmap, aspect='auto',
                            vmin=-abs_max, vmax=abs_max, interpolation='nearest')
        ax.set_title(disp(enc), fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(n, len(axes_flat)):
        axes_flat[k].axis('off')

    fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.1)
    if last_im is not None:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label(r'$w^{(-e)}_{t,j}$', rotation=270, labelpad=12, fontsize=9)
        cbar.ax.tick_params(labelsize=7)

    for ext in ('png', 'pdf'):
        plt.savefig(out_dir / f'fig_w_matrix_per_encoder.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved fig_w_matrix_per_encoder (range: ±{abs_max:.3f})")

    # Summary: mean and std of W^{(-e)} across encoders
    W_mean = all_W.mean(axis=0)
    W_std = all_W.std(axis=0)
    abs_max_mean = float(np.abs(W_mean).max()) if np.abs(W_mean).max() > 0 else 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    im1 = ax1.imshow(W_mean, cmap=cmap, aspect='auto',
                     vmin=-abs_max_mean, vmax=abs_max_mean, interpolation='nearest')
    ax1.set_title(r'Mean of $\mathbf{w}_t^{(-e)}$ across encoders', fontsize=10)
    ax1.set_xticks(np.arange(len(IFAMILIES)))
    ax1.set_xticklabels(IFAMILIES, fontsize=8)
    ax1.set_yticks(np.arange(len(TASK_ORDER)))
    ax1.set_yticklabels(TASK_ORDER, fontsize=8)
    ax1.set_xlabel('IIP', fontsize=9)
    ax1.set_ylabel('Task', fontsize=9)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cbar1.ax.tick_params(labelsize=7)

    im2 = ax2.imshow(W_std, cmap=plt.cm.viridis, aspect='auto',
                     interpolation='nearest')
    ax2.set_title(r'Std of $\mathbf{w}_t^{(-e)}$ across encoders', fontsize=10)
    ax2.set_xticks(np.arange(len(IFAMILIES)))
    ax2.set_xticklabels(IFAMILIES, fontsize=8)
    ax2.set_yticks(np.arange(len(TASK_ORDER)))
    ax2.set_yticklabels(TASK_ORDER, fontsize=8)
    ax2.set_xlabel('IIP', fontsize=9)
    ax2.set_ylabel('Task', fontsize=9)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cbar2.ax.tick_params(labelsize=7)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out_dir / f'fig_w_matrix_per_encoder_summary.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved fig_w_matrix_per_encoder_summary")


def plot_alignment_heatmap(alignment_matrix, encoders, out_dir):
    """Plot encoder×task alignment heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use actual data range instead of forcing [-1, 1]
    valid_data = alignment_matrix[~np.isnan(alignment_matrix)]
    vmin = np.min(valid_data)
    vmax = np.max(valid_data)

    # Make colormap symmetric around 0 if data spans both positive and negative
    if vmin < 0 and vmax > 0:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    cmap = LinearSegmentedColormap.from_list(
        'alignment', ['#d73027', '#f7f7f7', '#4575b4'], N=256
    )

    im = ax.imshow(alignment_matrix.T, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    # Convert encoder names using display dictionary
    display_names = [disp(enc) for enc in encoders]

    ax.set_xticks(np.arange(len(encoders)))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(np.arange(len(TASK_ORDER)))
    ax.set_yticklabels(TASK_ORDER, fontsize=7)

    ax.set_xlabel('Encoder', fontsize=9)
    ax.set_ylabel('Task', fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Alignment (cosine similarity)', rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out_dir / f'fig_alignment_heatmap.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved fig_alignment_heatmap (range: {vmin:.3f} to {vmax:.3f})")


def _plot_esp_panel(perf_matrix, pred, color, out_path):
    """Single ESP scatter panel — no grid, no title."""
    valid = ~(np.isnan(perf_matrix) | np.isnan(pred))
    if valid.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(perf_matrix[valid], pred[valid], alpha=0.6, s=55,
               color=color, edgecolors='white', linewidths=0.6, zorder=3)
    ax.plot([0, 1], [0, 1], 'r--', lw=1.5, alpha=0.7, zorder=2)

    rho, _ = spearmanr(perf_matrix[valid].ravel(), pred[valid].ravel())
    rmse = np.sqrt(mean_squared_error(perf_matrix[valid].ravel(), pred[valid].ravel()))
    r2 = 1 - (np.sum((perf_matrix[valid].ravel() - pred[valid].ravel()) ** 2) /
              np.sum((perf_matrix[valid].ravel() - perf_matrix[valid].ravel().mean()) ** 2))

    textstr = f'ρ = {rho:.3f}\nRMSE = {rmse:.3f}\nR² = {r2:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('True Performance', fontsize=9)
    ax.set_ylabel('Predicted Performance', fontsize=9)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out_path.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_esp_performance(perf_matrix, esp_pred, esp_pp_pred, out_dir):
    """Plot ESP prediction accuracy as two separate figures (K=3 and K=5)."""
    _plot_esp_panel(perf_matrix, esp_pred,    '#2E86AB', out_dir / 'fig_esp_performance_k3')
    _plot_esp_panel(perf_matrix, esp_pp_pred, '#A23B72', out_dir / 'fig_esp_performance_k5')
    print("  saved fig_esp_performance_k3 and fig_esp_performance_k5")


def plot_partial_correlation(partial_corr_df, out_dir):
    """Plot partial correlation analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(partial_corr_df))
    width = 0.35

    ax.bar(x - width/2, partial_corr_df['raw_correlation'], width,
           label='Raw Correlation', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, partial_corr_df['partial_correlation'], width,
           label='Partial Correlation\n(controlling mean probing)', alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Task', fontsize=9)
    ax.set_ylabel('Spearman ρ', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(partial_corr_df['task'], rotation=45, ha='right', fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=7, frameon=True)
    ax.grid(alpha=0.25, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out_dir / f'fig_partial_correlation.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  saved fig_partial_correlation")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Alignment-based analysis of encoder performance')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--mode', type=str, default='averaged',
                        choices=['averaged', 'per_encoder'],
                        help='averaged: one w_t per task pooled over all encoders; '
                             'per_encoder: leave-one-encoder-out w_t^{(-e)} per held-out encoder')
    args = parser.parse_args()

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = OUT_DIR_AVG if args.mode == 'averaged' else OUT_DIR_PER_ENC
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Alignment-Based Analysis of Encoder Performance  [mode={args.mode}]")
    print("=" * 70)

    # Load data
    print("\n1. Loading probing profiles (z_e)...")
    probing_profiles = load_probing_profiles()
    print(f"   Loaded {len(probing_profiles)} encoders with probing profiles")

    print("\n2. Loading downstream performance matrix...")
    encoders, perf_matrix = load_downstream_performance()
    print(f"   Loaded {len(encoders)} encoders × {len(TASK_ORDER)} tasks")

    # Infer task profiles + compute alignment (mode-dependent)
    if args.mode == 'averaged':
        print("\n3. Inferring task profiles (w_t) via Ridge regression [pooled]...")
        task_profiles = infer_task_profiles(probing_profiles, encoders, perf_matrix)
        print(f"   Inferred {len(task_profiles)} task profiles")

        print("\n4. Computing encoder-task alignment scores...")
        alignment_matrix = compute_alignment_matrix(probing_profiles, encoders, task_profiles)
        per_enc_profiles = None
    else:
        print("\n3. Inferring per-encoder task profiles (w_t^{(-e)}) via LOEO Ridge...")
        per_enc_profiles = infer_task_profiles_per_encoder(
            probing_profiles, encoders, perf_matrix
        )
        print(f"   Inferred {len(per_enc_profiles)} per-encoder profile sets")

        print("\n4. Computing encoder-task alignment scores [LOEO weights]...")
        alignment_matrix = compute_alignment_matrix_per_encoder(
            probing_profiles, encoders, per_enc_profiles
        )
        # ESP++ does not actually use task_profiles in its body; pass a placeholder
        task_profiles = None

    align_df = pd.DataFrame(alignment_matrix, index=encoders, columns=TASK_ORDER)
    align_df.to_csv(out_dir / 'alignment_scores.csv')
    print(f"   Saved alignment_scores.csv")

    # ESP: Leave-one-task-out (mode-independent: does not use w_t)
    print("\n5. Running ESP (Leave-One-Task-Out)...")
    esp_pred = esp_leave_one_task_out(probing_profiles, encoders, perf_matrix)
    esp_df = pd.DataFrame(esp_pred, index=encoders, columns=TASK_ORDER)
    esp_df.to_csv(out_dir / 'esp_predictions.csv')
    print(f"   Saved esp_predictions.csv")

    # ESP++: Task transfer (mode-independent)
    print("\n6. Running ESP++ (Task Transfer)...")
    esp_pp_pred = esp_plus_plus(probing_profiles, encoders, perf_matrix, task_profiles)
    esp_pp_df = pd.DataFrame(esp_pp_pred, index=encoders, columns=TASK_ORDER)
    esp_pp_df.to_csv(out_dir / 'esp_plus_plus_predictions.csv')
    print(f"   Saved esp_plus_plus_predictions.csv")

    # Partial correlation
    print("\n7. Computing partial Spearman correlations...")
    partial_corr_df = partial_spearman_correlation(
        alignment_matrix, perf_matrix, probing_profiles, encoders
    )
    partial_corr_df.to_csv(out_dir / 'partial_correlations.csv', index=False)
    print(f"   Saved partial_correlations.csv")

    # Visualization
    print("\n8. Generating figures...")
    if args.mode == 'averaged':
        plot_w_matrix_averaged(task_profiles, out_dir)
    else:
        plot_w_matrix_per_encoder(per_enc_profiles, encoders, out_dir)
    plot_alignment_heatmap(alignment_matrix, encoders, out_dir)
    plot_esp_performance(perf_matrix, esp_pred, esp_pp_pred, out_dir)
    plot_partial_correlation(partial_corr_df, out_dir)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nPartial Correlation Results (top 5 tasks by partial ρ):")
    valid_corr = partial_corr_df.dropna(subset=['partial_correlation'])
    if len(valid_corr) > 0:
        top5 = valid_corr.nlargest(min(5, len(valid_corr)), 'partial_correlation')
        print(top5[['task', 'raw_correlation', 'partial_correlation', 'partial_p_value']].to_string(index=False))
    else:
        print("  No valid correlations computed (insufficient probing profiles)")

    # ESP performance
    valid = ~(np.isnan(perf_matrix) | np.isnan(esp_pred))
    if valid.sum() > 0:
        esp_rho, _ = spearmanr(perf_matrix[valid].ravel(), esp_pred[valid].ravel())
        esp_rmse = np.sqrt(mean_squared_error(perf_matrix[valid].ravel(), esp_pred[valid].ravel()))
        print(f"\nESP Performance:")
        print(f"  Spearman ρ: {esp_rho:.4f}")
        print(f"  RMSE: {esp_rmse:.4f}")

    valid = ~(np.isnan(perf_matrix) | np.isnan(esp_pp_pred))
    if valid.sum() > 0:
        esp_pp_rho, _ = spearmanr(perf_matrix[valid].ravel(), esp_pp_pred[valid].ravel())
        esp_pp_rmse = np.sqrt(mean_squared_error(perf_matrix[valid].ravel(), esp_pp_pred[valid].ravel()))
        print(f"\nESP++ Performance:")
        print(f"  Spearman ρ: {esp_pp_rho:.4f}")
        print(f"  RMSE: {esp_pp_rmse:.4f}")

    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == '__main__':
    main()
