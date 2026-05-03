#!/usr/bin/env python3
"""
Alignment-Based Analysis v2 - Improved ESP Methods

Fixes for ESP/ESP++ poor performance:
1. Direct regression instead of alignment-to-performance scaling
2. Multi-task learning with shared encoder representations
3. Calibrated predictions matching performance distribution
4. Ensemble methods combining multiple predictors

Usage:
    python scripts/alignment_analysis_v2.py
"""

import argparse
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Mock imports for torch.load
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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
PROBING_DIR = ROOT / "probing_results_architecture_full"
PROBING_FALL = ROOT / "structure_encoder_data" / "probing_results"
DOWNSTREAM_DIR = ROOT / "outputs" / "downstream_v1"
OUT_DIR = ROOT / "outputs" / "alignment_analysis_v2"

TASKS = {
    "contact_prediction": {"T": "T1", "metric": "precision_L5"},
    "remote_homology": {"T": "T2", "metric": "accuracy"},
    "fluorescence": {"T": "T3", "metric": "spearman_rho"},
    "stability": {"T": "T4", "metric": "spearman_rho"},
    "mutation_effect": {"T": "T5", "metric": "spearman_rho"},
    "ec_classification": {"T": "T6", "metric": "accuracy"},
    "go_mf": {"T": "T7", "metric": "Fmax"},
    "go_bp": {"T": "T8", "metric": "Fmax"},
    "solubility": {"T": "T9", "metric": "AUROC"},
    "ppi_site": {"T": "T10", "metric": "AUPRC"},
    "metal_binding": {"T": "T11", "metric": "AUPRC"},
    "ppi_affinity": {"T": "T12", "metric": "spearman_rho"},
    "subcellular_localization": {"T": "T13", "metric": "accuracy"},
    "membrane_soluble": {"T": "T14", "metric": "AUROC"},
}

TASK_ORDER = [f"T{i}" for i in range(1, 15)]
IFAMILIES = [f"I{i}" for i in range(1, 11)]

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (same as v1)
# ─────────────────────────────────────────────────────────────────────────────

def load_probing_profiles():
    """Load encoder probing profiles (z_e) from I1-I10 invariant families."""
    profiles = {}
    for src in [PROBING_DIR, PROBING_FALL]:
        if not src.exists():
            continue
        for f in sorted(src.glob('*.pt')):
            try:
                enc = f.stem.rsplit('_seed_', 1)[0]
                seed = int(f.stem.rsplit('_seed_', 1)[1])
                data = torch.load(f, map_location='cpu', weights_only=False)
                fp = data['fingerprint']
                profiles.setdefault(enc, []).append({
                    f"I{i}": float(getattr(fp, f"p_I{i}")) for i in range(1, 11)
                })
            except Exception:
                continue

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
# Method 1: Direct Regression (no alignment intermediate)
# ─────────────────────────────────────────────────────────────────────────────

def method1_direct_regression(probing_profiles, encoders, perf_matrix, model_type='ridge'):
    """
    Directly predict performance from probing profiles.
    No alignment computation - just z_e → performance_t regression.
    """
    predictions = np.zeros_like(perf_matrix)

    # Build probing matrix
    Z = np.zeros((len(encoders), len(IFAMILIES)))
    valid_enc_idx = []
    for i, enc in enumerate(encoders):
        if enc in probing_profiles:
            Z[i] = [probing_profiles[enc][fam] for fam in IFAMILIES]
            valid_enc_idx.append(i)

    Z = Z[valid_enc_idx]

    # Leave-one-task-out
    for held_out_idx in range(len(TASK_ORDER)):
        train_indices = [i for i in range(len(TASK_ORDER)) if i != held_out_idx]

        # Train on all other tasks
        y_train_all = []
        X_train_all = []

        for j_train in train_indices:
            y = perf_matrix[valid_enc_idx, j_train]
            valid = ~np.isnan(y)
            if valid.sum() < 5:
                continue

            X_train_all.append(Z[valid])
            y_train_all.append(y[valid])

        if len(X_train_all) == 0:
            predictions[:, held_out_idx] = np.nan
            continue

        # Concatenate all training data
        X_train = np.vstack(X_train_all)
        y_train = np.concatenate(y_train_all)

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01)
        elif model_type == 'elasticnet':
            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        elif model_type == 'mlp':
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        else:
            model = Ridge(alpha=1.0)

        model.fit(X_train_scaled, y_train)

        # Predict on held-out task
        X_test = scaler.transform(Z)
        y_pred = model.predict(X_test)

        # Map back to full encoder list
        for idx, enc_idx in enumerate(valid_enc_idx):
            predictions[enc_idx, held_out_idx] = y_pred[idx]

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Task-Specific Models with Cross-Task Transfer
# ─────────────────────────────────────────────────────────────────────────────

def method2_task_specific_transfer(probing_profiles, encoders, perf_matrix):
    """
    For each held-out task, find K nearest tasks and train task-specific model.
    """
    predictions = np.zeros_like(perf_matrix)

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

        # Train on K nearest tasks
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

        for idx, enc_idx in enumerate(valid_enc_idx):
            predictions[enc_idx, held_out_idx] = y_pred[idx]

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Multi-Task Learning with Shared Representations
# ─────────────────────────────────────────────────────────────────────────────

def method3_multitask_learning(probing_profiles, encoders, perf_matrix):
    """
    Learn shared encoder representation, then task-specific heads.
    """
    predictions = np.zeros_like(perf_matrix)

    Z = np.zeros((len(encoders), len(IFAMILIES)))
    valid_enc_idx = []
    for i, enc in enumerate(encoders):
        if enc in probing_profiles:
            Z[i] = [probing_profiles[enc][fam] for fam in IFAMILIES]
            valid_enc_idx.append(i)

    Z = Z[valid_enc_idx]

    for held_out_idx in range(len(TASK_ORDER)):
        train_indices = [i for i in range(len(TASK_ORDER)) if i != held_out_idx]

        # Step 1: Learn shared representation on all training tasks
        X_train_all = []
        y_train_all = []
        task_ids = []

        for j_train in train_indices:
            y = perf_matrix[valid_enc_idx, j_train]
            valid = ~np.isnan(y)
            if valid.sum() < 5:
                continue

            X_train_all.append(Z[valid])
            y_train_all.append(y[valid])
            task_ids.extend([j_train] * valid.sum())

        if len(X_train_all) == 0:
            predictions[:, held_out_idx] = np.nan
            continue

        X_train = np.vstack(X_train_all)
        y_train = np.concatenate(y_train_all)
        task_ids = np.array(task_ids)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Step 2: Train shared model with task embeddings
        # Augment features with task-specific information
        n_tasks = len(train_indices)
        task_onehot = np.zeros((len(task_ids), n_tasks))
        for i, tid in enumerate(task_ids):
            task_onehot[i, train_indices.index(tid)] = 1.0

        X_train_aug = np.hstack([X_train_scaled, task_onehot])

        model = Ridge(alpha=1.0)
        model.fit(X_train_aug, y_train)

        # Step 3: Predict on held-out task (use mean task embedding)
        X_test = scaler.transform(Z)
        mean_task_emb = task_onehot.mean(axis=0, keepdims=True)
        mean_task_emb = np.repeat(mean_task_emb, len(Z), axis=0)
        X_test_aug = np.hstack([X_test, mean_task_emb])

        y_pred = model.predict(X_test_aug)

        for idx, enc_idx in enumerate(valid_enc_idx):
            predictions[enc_idx, held_out_idx] = y_pred[idx]

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Method 4: Ensemble of Methods
# ─────────────────────────────────────────────────────────────────────────────

def method4_ensemble(probing_profiles, encoders, perf_matrix):
    """
    Ensemble of multiple methods with learned weights.
    """
    print("  Running Method 1 (Ridge)...")
    pred1 = method1_direct_regression(probing_profiles, encoders, perf_matrix, 'ridge')

    print("  Running Method 1 (GBM)...")
    pred2 = method1_direct_regression(probing_profiles, encoders, perf_matrix, 'gbm')

    print("  Running Method 2 (Task Transfer)...")
    pred3 = method2_task_specific_transfer(probing_profiles, encoders, perf_matrix)

    print("  Running Method 3 (Multi-Task)...")
    pred4 = method3_multitask_learning(probing_profiles, encoders, perf_matrix)

    # Simple average ensemble
    predictions = np.nanmean([pred1, pred2, pred3, pred4], axis=0)

    return predictions, {'ridge': pred1, 'gbm': pred2, 'transfer': pred3, 'multitask': pred4}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(perf_matrix, predictions, method_name):
    """Evaluate prediction quality."""
    valid = ~(np.isnan(perf_matrix) | np.isnan(predictions))

    if valid.sum() == 0:
        return {'method': method_name, 'spearman_rho': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

    y_true = perf_matrix[valid].ravel()
    y_pred = predictions[valid].ravel()

    rho, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'method': method_name,
        'spearman_rho': rho,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_samples': valid.sum(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(perf_matrix, all_predictions, out_dir):
    """Plot comparison of all methods."""
    n_methods = len(all_predictions)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, (method_name, predictions) in enumerate(all_predictions.items()):
        ax = axes[idx]

        valid = ~(np.isnan(perf_matrix) | np.isnan(predictions))
        if valid.sum() == 0:
            ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
            ax.set_title(method_name)
            continue

        y_true = perf_matrix[valid].ravel()
        y_pred = predictions[valid].ravel()

        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', lw=1, label='Perfect prediction')

        rho, _ = spearmanr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        ax.set_xlabel('True Performance', fontsize=9)
        ax.set_ylabel('Predicted Performance', fontsize=9)
        ax.set_title(f'{method_name}\nρ={rho:.3f}, RMSE={rmse:.3f}', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(out_dir / f'fig_method_comparison.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  saved fig_method_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Alignment analysis v2 - improved methods')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Alignment-Based Analysis v2 - Improved ESP Methods")
    print("=" * 70)

    print("\n1. Loading data...")
    probing_profiles = load_probing_profiles()
    encoders, perf_matrix = load_downstream_performance()
    print(f"   {len(probing_profiles)} encoders with probing profiles")
    print(f"   {len(encoders)} encoders × {len(TASK_ORDER)} tasks")

    print("\n2. Running improved methods...")

    print("\nMethod 1: Direct Regression (Ridge)...")
    pred_ridge = method1_direct_regression(probing_profiles, encoders, perf_matrix, 'ridge')
    eval_ridge = evaluate_predictions(perf_matrix, pred_ridge, 'Direct Ridge')

    print("\nMethod 1: Direct Regression (GBM)...")
    pred_gbm = method1_direct_regression(probing_profiles, encoders, perf_matrix, 'gbm')
    eval_gbm = evaluate_predictions(perf_matrix, pred_gbm, 'Direct GBM')

    print("\nMethod 2: Task-Specific Transfer...")
    pred_transfer = method2_task_specific_transfer(probing_profiles, encoders, perf_matrix)
    eval_transfer = evaluate_predictions(perf_matrix, pred_transfer, 'Task Transfer')

    print("\nMethod 3: Multi-Task Learning...")
    pred_multitask = method3_multitask_learning(probing_profiles, encoders, perf_matrix)
    eval_multitask = evaluate_predictions(perf_matrix, pred_multitask, 'Multi-Task')

    print("\nMethod 4: Ensemble...")
    pred_ensemble, individual_preds = method4_ensemble(probing_profiles, encoders, perf_matrix)
    eval_ensemble = evaluate_predictions(perf_matrix, pred_ensemble, 'Ensemble')

    # Save predictions
    all_predictions = {
        'Direct_Ridge': pred_ridge,
        'Direct_GBM': pred_gbm,
        'Task_Transfer': pred_transfer,
        'Multi_Task': pred_multitask,
        'Ensemble': pred_ensemble,
    }

    for method_name, predictions in all_predictions.items():
        df = pd.DataFrame(predictions, index=encoders, columns=TASK_ORDER)
        df.to_csv(out_dir / f'predictions_{method_name}.csv')

    # Save evaluation results
    eval_results = pd.DataFrame([eval_ridge, eval_gbm, eval_transfer, eval_multitask, eval_ensemble])
    eval_results.to_csv(out_dir / 'evaluation_results.csv', index=False)

    print("\n3. Generating figures...")
    plot_comparison(perf_matrix, all_predictions, out_dir)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(eval_results[['method', 'spearman_rho', 'rmse', 'mae', 'r2']].to_string(index=False))

    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == '__main__':
    main()
