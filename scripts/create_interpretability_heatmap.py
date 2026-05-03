#!/usr/bin/env python3
"""
Interpretability Heatmap: Which Invariant Family Best Explains Each Task?

For each downstream task, shows which invariant family (I1-I10) has the
highest weight in the task profile (from Ridge regression).

X-axis: Encoders (sorted by modality)
Y-axis: Downstream tasks (T1-T15)
Color: Dominant invariant family for that task

Style: Consistent with visualize_paper_v12.py
"""

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# Mock imports
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
PROBING_DIR = ROOT / "probing_results_architecture_full"
PROBING_FALL = ROOT / "structure_encoder_data" / "probing_results"
DOWNSTREAM_DIR = ROOT / "outputs" / "downstream_v1"
OUT_DIR = ROOT / "outputs" / "alignment_analysis"

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

# Task and invariant definitions
TASK_ORDER = [f"T{i}" for i in range(1, 15)]
IFAMILIES = [f"I{i}" for i in range(1, 11)]

TASK_NAMES = {
    "T1": "Contact\nPred.", "T2": "Remote\nHomol.", "T3": "Fluor.",
    "T4": "Stab.", "T5": "Mut.\nEffect", "T6": "EC\nClass.",
    "T7": "GO-MF", "T8": "GO-BP", "T9": "Solub.",
    "T10": "PPI\nSites", "T11": "Metal\nBind.", "T12": "PPI\nAffin.",
    "T13": "Subcell.\nLoc.", "T14": "Mem./Sol.",
}

IFAMILY_NAMES = {
    'I1':  'Local Conformation',  'I2':  'Local Curvature',   'I3':  'Sec. Struct.',
    'I4':  'Solv. Access.',       'I5':  'Local Flexibility', 'I6':  'Packing Density',
    'I7':  'Residue Contacts',    'I8':  'Global Geometry',   'I9':  'Fold Architecture',
    'I10': 'Sequence',
}

# Paper palette: reordered to match new IIP arrangement
IFAMILY_COLORS = [
    '#4169E1',  # I1  Local Conformation  — royal blue    (local geometry)
    '#1a4a8a',  # I2  Local Curvature     — navy          (local geometry)
    '#B0E0E6',  # I3  Sec. Struct.        — powder blue   (local structure)
    '#c9a0e0',  # I4  Solv. Access.       — light purple  (biophysics)
    '#FFA500',  # I5  Local Flexibility   — orange        (flexibility)
    '#4d2475',  # I6  Packing Density     — deep purple   (biophysics)
    '#5bacd4',  # I7  Residue Contacts    — sky blue      (pairwise structure)
    '#888888',  # I8  Global Geometry     — gray          (global structure)
    '#D2691E',  # I9  Fold Architecture   — chocolate     (global structure)
    '#9966bb',  # I10 Sequence            — medium purple (sequence)
]

# Display names (from visualize_paper_v12.py)
DISPLAY_NAMES = {
    'ankh': 'Ankh', 'esm1b': 'ESM-1b', 'esm2': 'ESM-2(650M)', 'esm2_35m': 'ESM-2(35M)',
    'esm2_150m': 'ESM-2(150M)', 'prottrans': 'ProtTrans', 'proteinbert_bfd': 'ProteinBERT',
    'prostt5': 'ProstT5', 'foldseek': 'FoldSeek', 'proteinmpnn': 'ProteinMPNN',
    'pottsmpnn': 'PottsMPNN', 'pottsmpnn_soluble': 'PottsMPNN(Sol)',
    'pottsmpnn_ft': 'PottsMPNN(FT)', 'gearnet': 'GearNet', 'schnet': 'SchNet',
    'dimenet': 'DimeNet++', 'mace': 'MACE', 'dmasif': 'dMaSIF', 'egnn': 'EGNN',
    'gvp': 'GVP', 'tfn': 'TFN', 'ipa': 'IPA', 'gcpnet': 'GCPNet', 'esm_if': 'ESM-IF',
    'saprot': 'SaProt', 'esm3': 'ESM-3', 'foldvision': 'FoldVision', 'cdconv': 'CDConv',
    'random': 'Random', 'length': 'Length', 'torsion': 'Torsion',
}

def disp(enc):
    clean = enc.replace('_pretrained', '')
    return DISPLAY_NAMES.get(clean, clean)

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.titlesize': 10, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})


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
    """Aggregate probe scores into IIP families."""
    out = {}
    for fam, probes in FAMILY_PROBES.items():
        vals = [float(detailed[p]) for p in probes if p in detailed]
        out[fam] = float(np.mean(vals)) if vals else 0.0
    return out


def load_probing_profiles():
    """Load encoder probing profiles."""
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
                detailed = getattr(fp, 'detailed_scores', None)
                if detailed:
                    profiles.setdefault(enc, []).append(_aggregate_families(detailed))
                else:
                    # Fallback: use stored p_I* (legacy semantics)
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
    """Load downstream performance matrix."""
    csv_path = DOWNSTREAM_DIR / "downstream_results.csv"
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


def infer_task_profiles(probing_profiles, encoders, perf_matrix):
    """Infer task profiles via Ridge regression."""
    Z = np.zeros((len(encoders), len(IFAMILIES)))
    valid_enc_idx = []
    for i, enc in enumerate(encoders):
        if enc in probing_profiles:
            Z[i] = [probing_profiles[enc][fam] for fam in IFAMILIES]
            valid_enc_idx.append(i)

    Z = Z[valid_enc_idx]
    task_profiles = {}

    for j, tnum in enumerate(TASK_ORDER):
        y = perf_matrix[valid_enc_idx, j]
        valid = ~np.isnan(y)

        if valid.sum() < 5:
            task_profiles[tnum] = np.zeros(len(IFAMILIES))
            continue

        Z_valid = Z[valid]
        y_valid = y[valid]

        scaler = StandardScaler()
        Z_scaled = scaler.fit_transform(Z_valid)

        # Cross-validate alpha
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
        task_profiles[tnum] = model.coef_

    return task_profiles


def create_interpretability_heatmap(task_profiles, encoders):
    """Create heatmap showing dominant invariant for each encoder-task pair."""

    # For each task, find the dominant invariant (highest absolute weight)
    dominant_invariants = {}
    for task, profile in task_profiles.items():
        dominant_idx = np.argmax(np.abs(profile))
        dominant_invariants[task] = dominant_idx

    # Create matrix: each row is an encoder, each column is a task
    # Value is the dominant invariant index for that task
    n_enc = len(encoders)
    n_task = len(TASK_ORDER)
    matrix = np.zeros((n_enc, n_task), dtype=int)

    for j, task in enumerate(TASK_ORDER):
        if task in dominant_invariants:
            inv_idx = dominant_invariants[task]
            matrix[:, j] = inv_idx  # All encoders get same color for this task

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create colormap
    cmap = ListedColormap(IFAMILY_COLORS)

    # Plot
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=9, interpolation='nearest')

    # Set ticks
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_NAMES.get(t, t) for t in TASK_ORDER], rotation=45, ha='right', fontsize=7)

    ax.set_yticks(range(len(encoders)))
    ax.set_yticklabels([disp(e) for e in encoders], fontsize=7)

    # Labels
    ax.set_xlabel('Downstream Task', fontsize=9)
    ax.set_ylabel('Encoder', fontsize=9)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=IFAMILY_COLORS[i],
                                      label=f'{IFAMILIES[i]}: {IFAMILY_NAMES[IFAMILIES[i]]}',
                                      edgecolor='black', linewidth=0.5)
                      for i in range(10)]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
             title='Invariant Family', frameon=True, fontsize=7, title_fontsize=7)

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        out_path = OUT_DIR / f'fig_interpretability_heatmap.{ext}'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {out_path}")

    plt.close()


def infer_task_profiles_per_encoder(probing_profiles, encoders, perf_matrix):
    """
    Per-encoder task profiles via leave-one-encoder-out (LOEO) ridge regression.
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


def create_interpretability_heatmap_per_encoder(per_enc_profiles, encoders):
    """Create heatmap showing dominant invariant for each encoder-task pair (LOEO)."""

    # Create matrix: each cell gets its own dominant invariant
    n_enc = len(encoders)
    n_task = len(TASK_ORDER)
    matrix = np.zeros((n_enc, n_task), dtype=int)

    for i, enc in enumerate(encoders):
        if enc not in per_enc_profiles:
            continue
        for j, task in enumerate(TASK_ORDER):
            if task in per_enc_profiles[enc]:
                profile = per_enc_profiles[enc][task]
                dominant_idx = np.argmax(np.abs(profile))
                matrix[i, j] = dominant_idx

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create colormap
    cmap = ListedColormap(IFAMILY_COLORS)

    # Plot
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=9, interpolation='nearest')

    # Set ticks
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_NAMES.get(t, t) for t in TASK_ORDER], rotation=45, ha='right', fontsize=7)

    ax.set_yticks(range(len(encoders)))
    ax.set_yticklabels([disp(e) for e in encoders], fontsize=7)

    # Labels
    ax.set_xlabel('Downstream Task', fontsize=9)
    ax.set_ylabel('Encoder', fontsize=9)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=IFAMILY_COLORS[i],
                                      label=f'{IFAMILIES[i]}: {IFAMILY_NAMES[IFAMILIES[i]]}',
                                      edgecolor='black', linewidth=0.5)
                      for i in range(10)]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
             title='Invariant Family', frameon=True, fontsize=7, title_fontsize=7)

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        out_path = OUT_DIR / f'fig_interpretability_heatmap_per_encoder.{ext}'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {out_path}")

    plt.close()


def main():
    print("=" * 70)
    print("Interpretability Heatmap Generation (Averaged + Per-Encoder)")
    print("=" * 70)

    print("\n1. Loading probing profiles...")
    probing_profiles = load_probing_profiles()
    print(f"   Loaded {len(probing_profiles)} encoders")

    print("\n2. Loading downstream performance...")
    encoders, perf_matrix = load_downstream_performance()
    print(f"   {len(encoders)} encoders × {len(TASK_ORDER)} tasks")

    print("\n3. Inferring task profiles (averaged)...")
    task_profiles = infer_task_profiles(probing_profiles, encoders, perf_matrix)
    print(f"   Inferred {len(task_profiles)} task profiles")

    print("\n4. Dominant invariant for each task (averaged):")
    for task, profile in task_profiles.items():
        dominant_idx = np.argmax(np.abs(profile))
        dominant_inv = IFAMILIES[dominant_idx]
        weight = profile[dominant_idx]
        print(f"   {task}: {dominant_inv} ({IFAMILY_NAMES[dominant_inv]}) = {weight:.3f}")

    print("\n5. Creating interpretability heatmap (averaged)...")
    create_interpretability_heatmap(task_profiles, encoders)

    print("\n6. Inferring task profiles (per-encoder LOEO)...")
    per_enc_profiles = infer_task_profiles_per_encoder(probing_profiles, encoders, perf_matrix)
    print(f"   Inferred profiles for {len(per_enc_profiles)} encoders")

    print("\n7. Creating interpretability heatmap (per-encoder)...")
    create_interpretability_heatmap_per_encoder(per_enc_profiles, encoders)

    print("\n✓ Done! Generated both averaged and per-encoder heatmaps.")


if __name__ == '__main__':
    main()
