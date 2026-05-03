#!/usr/bin/env python3
"""
Interpretability Heatmaps v2 (per-encoder).

The original `create_interpretability_heatmap.py` produces a heatmap where
every encoder row is identical for a given task — by construction, because
the Ridge regression yields ONE task profile per task (encoders are the
rows of the design matrix, not separate models).

This script produces TWO genuinely encoder-specific heatmaps:

  (1) fig_interpretability_heatmap_encoder.png
      For each encoder, show the dominant invariant family of its OWN
      probing fingerprint (argmax over I1..I10 of z_enc). Rows differ;
      columns within a row are uniform — what the encoder is "best at"
      regardless of the task.

  (2) fig_interpretability_heatmap_interaction.png
      For each (encoder, task) cell, attribute the encoder's predicted
      task performance to each invariant family via beta_{T,k} * z_enc,k
      (where beta_T is the task's Ridge coefficient and z_enc is the
      encoder's standardized fingerprint). The cell's color is argmax_k.
      Rows AND columns vary — this is the true (encoder, task) view.

Re-uses the 14-task layout (Sec.Struct removed) and consistent styling
with create_interpretability_heatmap.py.
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
OUT_DIR_AVG = ROOT / "outputs" / "alignment_analysis_averaged"
OUT_DIR_PER_ENC = ROOT / "outputs" / "alignment_analysis_per_encoder"

# 14-task layout (Sec.Struct removed)
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

TASK_NAMES = {
    "T1": "Contact",      "T2": "Homology",    "T3": "Fluor.",
    "T4": "Stability",    "T5": "Mut.Effect",  "T6": "EC Class.",
    "T7": "GO-MF",        "T8": "GO-BP",       "T9": "Solubility",
    "T10": "PPI Sites",   "T11": "Metal Bind.","T12": "PPI Affin.",
    "T13": "Subcell.Loc.","T14": "Mem./Sol.",
}

IFAMILY_NAMES = {
    'I1':  'Local Conformation',  'I2':  'Local Curvature',   'I3':  'Sec. Struct.',
    'I4':  'Solv. Access.',       'I5':  'Local Flexibility', 'I6':  'Packing Density',
    'I7':  'Residue Contacts',    'I8':  'Global Geometry',   'I9':  'Fold Architecture',
    'I10': 'Sequence',
}

# Paper palette: blues for local structure, oranges for global structure,
# purples for biophysics/sequence, gray for flexibility — reordered to match new IIP arrangement.
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

DISPLAY_NAMES = {
    'ankh': 'Ankh', 'esm1b': 'ESM-1b', 'esm2': 'ESM-2(650M)', 'esm2_35m': 'ESM-2(35M)',
    'esm2_150m': 'ESM-2(150M)', 'esm2_650m': 'ESM-2(650M)',
    'prottrans': 'ProtTrans', 'prottrans_t5': 'ProtTrans-T5', 'prot_t5': 'ProtT5',
    'proteinbert_bfd': 'ProteinBERT', 'prostt5': 'ProstT5', 'unirep': 'UniRep',
    'geobpe': 'GeoBPE', 'bpe': 'BPE',
    'foldseek': 'FoldSeek', 'proteinmpnn': 'ProteinMPNN', 'ligandmpnn': 'LigandMPNN',
    'pottsmpnn': 'PottsMPNN', 'pottsmpnn_soluble': 'PottsMPNN(Sol)',
    'pottsmpnn_ft': 'PottsMPNN(FT)', 'pottsmpnn_ca': 'PottsMPNN(CA)',
    'gearnet': 'GearNet', 'schnet': 'SchNet', 'dimenet': 'DimeNet++',
    'dimenet_plus_plus': 'DimeNet++', 'mace': 'MACE', 'dmasif': 'dMaSIF',
    'egnn': 'EGNN', 'gvp': 'GVP', 'tfn': 'TFN', 'ipa': 'IPA', 'gcpnet': 'GCPNet',
    'esm_if': 'ESM-IF', 'proteinae': 'ProteinAE',
    'saprot': 'SaProt', 'esm3': 'ESM3', 'foldvision': 'FoldVision', 'cdconv': 'CDConv',
    'random': 'Random', 'length': 'Length', 'torsion': 'Torsion',
}

# ── Encoder subtype taxonomy + colors (mirrors visualize_downstream_v1.py) ──
ENCODER_SUBTYPE = {
    **{e: 'Sequence' for e in [
        'ankh', 'esm1b', 'esm2', 'esm2_35m', 'esm2_150m', 'esm2_650m',
        'prottrans', 'prottrans_t5', 'prot_t5', 'proteinbert_bfd',
        'prostt5', 'unirep',
    ]},
    'foldseek': 'Struct: Token', 'geobpe': 'Struct: Token', 'bpe': 'Struct: Token',
    **{e: 'Struct: Graph' for e in [
        'proteinmpnn', 'proteinmpnn_pretrained', 'ligandmpnn',
        'pottsmpnn', 'pottsmpnn_pretrained',
        'pottsmpnn_soluble_pretrained', 'pottsmpnn_ft_pretrained', 'pottsmpnn_ca_pretrained',
        'pottsmpnn_soluble', 'pottsmpnn_ft', 'pottsmpnn_ca',
        'gearnet', 'gearnet_pretrained', 'gearnet_edge', 'gearnet_edge_ief',
        'schnet', 'dimenet', 'dimenet_plus_plus', 'mace',
        'dmasif', 'dmasif_pretrained',
        'comenet', 'equiformer', 'equiformer_v2', 'gemnet', 'painn',
        'se3_transformer', 'spherenet', 'lefttnet',
        'density', 'mif', 'proteinae', 'starling', 'dual_stream',
    ]},
    **{e: 'Struct: Equiv' for e in [
        'egnn', 'egnn_pretrained', 'gvp', 'gvp_pretrained',
        'tfn', 'tfn_pretrained', 'ipa', 'ipa_pretrained',
        'gcpnet', 'gcpnet_pretrained', 'esm_if', 'esm_if_inverse',
    ]},
    **{e: 'Multimodal' for e in [
        'saprot', 'saprot_pretrained', 'esm3', 'foldvision',
        'cdconv', 'cdconv_pretrained',
    ]},
    **{e: 'Baseline' for e in [
        'random', 'random_token', 'length', 'torsion', 'onehot', 'blosum', 'test',
    ]},
}

SUBTYPE_COLORS = {
    'Sequence':      '#1a4a8a',  # navy blue
    'Struct: Token': '#FFA500',  # orange
    'Struct: Graph': '#D2691E',  # chocolate
    'Struct: Equiv': '#A0522D',  # sienna
    'Multimodal':    '#7c4fa0',  # purple
    'Baseline':      '#666666',  # gray
}

SUBTYPE_ORDER = {
    'Sequence': 0, 'Struct: Token': 1, 'Struct: Graph': 2,
    'Struct: Equiv': 3, 'Multimodal': 4, 'Baseline': 5,
}


def encoder_sort_key(enc):
    clean = enc.replace('_pretrained', '')
    stype = ENCODER_SUBTYPE.get(clean, ENCODER_SUBTYPE.get(enc, 'Baseline'))
    return (SUBTYPE_ORDER.get(stype, 6), disp(enc))


def encoder_subtype(enc):
    clean = enc.replace('_pretrained', '')
    return ENCODER_SUBTYPE.get(clean, ENCODER_SUBTYPE.get(enc, 'Baseline'))


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
    profiles = {}
    for src in [PROBING_DIR, PROBING_FALL]:
        if not src.exists():
            continue
        for f in sorted(src.glob('*.pt')):
            try:
                enc = f.stem.rsplit('_seed_', 1)[0]
                data = torch.load(f, map_location='cpu', weights_only=False)
                fp = data['fingerprint']
                detailed = getattr(fp, 'detailed_scores', None)
                if detailed:
                    profiles.setdefault(enc, []).append(_aggregate_families(detailed))
                else:
                    # Fallback: use stored p_I* (legacy semantics)
                    profiles.setdefault(enc, []).append(
                        {f"I{i}": float(getattr(fp, f"p_I{i}")) for i in range(1, 11)}
                    )
            except Exception:
                continue
    return {enc: {fam: np.mean([s[fam] for s in seeds]) for fam in IFAMILIES}
            for enc, seeds in profiles.items()}


def load_downstream_performance():
    df = pd.read_csv(DOWNSTREAM_DIR / "downstream_results.csv")
    encoders = sorted(df['encoder'].unique())
    perf = np.full((len(encoders), len(TASK_ORDER)), np.nan)
    name_for = {info["T"]: name for name, info in TASKS.items()}
    for i, enc in enumerate(encoders):
        sub = df[df['encoder'] == enc]
        for j, tnum in enumerate(TASK_ORDER):
            task_name = name_for.get(tnum)
            row = sub[sub['task'] == task_name]
            if len(row):
                perf[i, j] = row['mean_score'].values[0]
    return encoders, perf


def fit_task_profiles(probing_profiles, encoders, perf_matrix):
    """Return (task_betas, scaler, valid_enc_idx, Z) for downstream re-use."""
    Z = np.zeros((len(encoders), len(IFAMILIES)))
    valid_idx = []
    for i, enc in enumerate(encoders):
        if enc in probing_profiles:
            Z[i] = [probing_profiles[enc][fam] for fam in IFAMILIES]
            valid_idx.append(i)

    Z_valid = Z[valid_idx]
    scaler = StandardScaler().fit(Z_valid)
    Z_scaled = scaler.transform(Z_valid)

    task_betas = {}
    for j, tnum in enumerate(TASK_ORDER):
        y = perf_matrix[valid_idx, j]
        ok = ~np.isnan(y)
        if ok.sum() < 5:
            task_betas[tnum] = np.zeros(len(IFAMILIES))
            continue
        # CV alpha
        best_alpha, best_score = 1.0, -np.inf
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            m = Ridge(alpha=alpha).fit(Z_scaled[ok], y[ok])
            s = m.score(Z_scaled[ok], y[ok])
            if s > best_score:
                best_alpha, best_score = alpha, s
        task_betas[tnum] = Ridge(alpha=best_alpha).fit(Z_scaled[ok], y[ok]).coef_
    return task_betas, scaler, valid_idx, Z


def fit_task_profiles_per_encoder(probing_profiles, encoders, perf_matrix):
    """
    Per-encoder LOEO task profiles.
    Returns: dict {encoder_name: {task_id: beta_vector}}, scaler, valid_idx, Z
    """
    Z = np.zeros((len(encoders), len(IFAMILIES)))
    valid_idx = []
    for i, enc in enumerate(encoders):
        if enc in probing_profiles:
            Z[i] = [probing_profiles[enc][fam] for fam in IFAMILIES]
            valid_idx.append(i)

    Z_valid = Z[valid_idx]
    scaler = StandardScaler().fit(Z_valid)

    per_enc_betas = {}
    for held_out_idx in valid_idx:
        held_out_enc = encoders[held_out_idx]
        per_enc_betas[held_out_enc] = {}

        train_idx = [i for i in valid_idx if i != held_out_idx]
        Z_train = Z[train_idx]
        Z_train_scaled = scaler.transform(Z_train)

        for j, tnum in enumerate(TASK_ORDER):
            y = perf_matrix[train_idx, j]
            ok = ~np.isnan(y)
            if ok.sum() < 5:
                per_enc_betas[held_out_enc][tnum] = np.zeros(len(IFAMILIES))
                continue

            best_alpha, best_score = 1.0, -np.inf
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                m = Ridge(alpha=alpha).fit(Z_train_scaled[ok], y[ok])
                s = m.score(Z_train_scaled[ok], y[ok])
                if s > best_score:
                    best_alpha, best_score = alpha, s
            per_enc_betas[held_out_enc][tnum] = Ridge(alpha=best_alpha).fit(
                Z_train_scaled[ok], y[ok]
            ).coef_

    return per_enc_betas, scaler, valid_idx, Z


def _heatmap(ax, matrix, encoders, title, out_dir):
    """Render matrix shaped (n_enc, n_task) as a transposed heatmap:
    x-axis = encoders (sorted by subtype, tick-colored), y-axis = tasks."""
    cmap = ListedColormap(IFAMILY_COLORS)
    # Transpose so encoders are columns and tasks are rows.
    ax.imshow(matrix.T, cmap=cmap, aspect='auto', vmin=0, vmax=9, interpolation='nearest')

    # x-axis: encoders, tick label colored by subtype
    ax.set_xticks(range(len(encoders)))
    ax.set_xticklabels([disp(e) for e in encoders], rotation=45, ha='right', fontsize=6)
    for tick, enc in zip(ax.get_xticklabels(), encoders):
        tick.set_color(SUBTYPE_COLORS.get(encoder_subtype(enc), '#333'))

    # White vertical separators between subtypes
    prev = None
    for i, enc in enumerate(encoders):
        st = encoder_subtype(enc)
        if prev is not None and st != prev:
            ax.axvline(i - 0.5, color='white', linewidth=2.0)
        prev = st

    # y-axis: tasks (single-line labels; rotated 45° for readability)
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels([TASK_NAMES.get(t, t) for t in TASK_ORDER],
                       fontsize=7, rotation=45, ha='right', va='center')


def _legend(fig, ax):
    handles = [mpatches.Patch(facecolor=IFAMILY_COLORS[i],
                              label=f'{IFAMILIES[i]}: {IFAMILY_NAMES[IFAMILIES[i]]}',
                              edgecolor='black', linewidth=0.5)
               for i in range(10)]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
              title='Invariant Family', frameon=True, fontsize=7, title_fontsize=7)


def heatmap_encoder_dominant(probing_profiles, encoders, valid_idx, out_dir):
    """Option 1: each encoder colored by argmax of its own probing fingerprint."""
    enc_subset = sorted([encoders[i] for i in valid_idx], key=encoder_sort_key)
    matrix = np.zeros((len(enc_subset), len(TASK_ORDER)), dtype=int)
    for i, enc in enumerate(enc_subset):
        z = np.array([probing_profiles[enc][fam] for fam in IFAMILIES])
        matrix[i, :] = int(np.argmax(z))

    fig, ax = plt.subplots(figsize=(8, 4))
    _heatmap(ax, matrix, enc_subset, "", out_dir)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(out_dir / f'fig_interpretability_heatmap_encoder.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_interpretability_heatmap_encoder.[png|pdf]")


def heatmap_interaction(probing_profiles, encoders, valid_idx, task_betas, scaler, out_dir):
    """Option 2: per-cell argmax of beta_{T,k} * z_{enc,k} (standardized)."""
    enc_subset = sorted([encoders[i] for i in valid_idx], key=encoder_sort_key)
    Z = np.array([[probing_profiles[enc][fam] for fam in IFAMILIES] for enc in enc_subset])
    Z_scaled = scaler.transform(Z)

    matrix = np.zeros((len(enc_subset), len(TASK_ORDER)), dtype=int)
    for j, tnum in enumerate(TASK_ORDER):
        beta = task_betas[tnum]
        contrib = Z_scaled * beta[None, :]
        matrix[:, j] = np.argmax(np.abs(contrib), axis=1)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _heatmap(ax, matrix, enc_subset, "", out_dir)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(out_dir / f'fig_interpretability_heatmap_interaction.{ext}',
                    dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  Saved: fig_interpretability_heatmap_interaction.[png|pdf]")


def heatmap_interaction_per_encoder(probing_profiles, encoders, valid_idx, per_enc_betas, scaler, out_dir):
    """Per-encoder LOEO version: each encoder uses its own w_t^{(-e)}."""
    enc_subset = sorted([encoders[i] for i in valid_idx], key=encoder_sort_key)
    Z = np.array([[probing_profiles[enc][fam] for fam in IFAMILIES] for enc in enc_subset])
    Z_scaled = scaler.transform(Z)

    matrix = np.zeros((len(enc_subset), len(TASK_ORDER)), dtype=int)
    for i, enc in enumerate(enc_subset):
        if enc not in per_enc_betas:
            continue
        for j, tnum in enumerate(TASK_ORDER):
            beta = per_enc_betas[enc][tnum]
            contrib = Z_scaled[i] * beta
            matrix[i, j] = int(np.argmax(np.abs(contrib)))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _heatmap(ax, matrix, enc_subset, "", out_dir)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(out_dir / f'fig_interpretability_heatmap_interaction.{ext}',
                    dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  Saved: fig_interpretability_heatmap_interaction.[png|pdf]")


def standalone_legend(out_dir):
    """Standalone legend figure — shared across the three heatmaps."""
    handles = [mpatches.Patch(facecolor=IFAMILY_COLORS[i],
                              label=f'{IFAMILIES[i]}: {IFAMILY_NAMES[IFAMILIES[i]]}',
                              edgecolor='black', linewidth=0.5)
               for i in range(10)]

    # Horizontal layout (one row of 10 swatches) — fits a paper figure caption strip.
    fig_h, ax_h = plt.subplots(figsize=(6, 0.8))
    ax_h.axis('off')
    ax_h.legend(handles=handles, loc='center', ncol=10, frameon=False,
                fontsize=8, handlelength=1.2, handleheight=1.0,
                columnspacing=1.0, borderpad=0.2)
    for ext in ['png', 'pdf']:
        fig_h.savefig(out_dir / f'fig_interpretability_legend_h.{ext}',
                      dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_h)

    # Vertical layout (single column) — fits beside a side-by-side figure pair.
    fig_v, ax_v = plt.subplots(figsize=(2.0, 3.0))
    ax_v.axis('off')
    ax_v.legend(handles=handles, loc='center', ncol=1, frameon=True,
                title='Invariant Family', title_fontsize=8, fontsize=8,
                edgecolor='black', borderpad=0.6)
    for ext in ['png', 'pdf']:
        fig_v.savefig(out_dir / f'fig_interpretability_legend_v.{ext}',
                      dpi=300, bbox_inches='tight')
    plt.close(fig_v)

    print(f"  Saved: fig_interpretability_legend_h.[png|pdf]  (horizontal)")
    print(f"  Saved: fig_interpretability_legend_v.[png|pdf]  (vertical)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Interpretability heatmaps v2')
    parser.add_argument('--mode', type=str, default='averaged',
                        choices=['averaged', 'per_encoder'],
                        help='averaged: pooled task profiles; per_encoder: LOEO task profiles')
    args = parser.parse_args()

    out_dir = OUT_DIR_AVG if args.mode == 'averaged' else OUT_DIR_PER_ENC
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Interpretability Heatmaps v2 — per-encoder views  [mode={args.mode}]")
    print("=" * 70)

    print("\n1. Loading probing profiles...")
    probing_profiles = load_probing_profiles()
    print(f"   Loaded {len(probing_profiles)} encoders")

    print("\n2. Loading downstream performance...")
    encoders, perf_matrix = load_downstream_performance()
    print(f"   {len(encoders)} encoders × {len(TASK_ORDER)} tasks")

    if args.mode == 'averaged':
        print("\n3. Fitting per-task Ridge profiles [pooled]...")
        task_betas, scaler, valid_idx, Z = fit_task_profiles(probing_profiles, encoders, perf_matrix)
        print(f"   {len(task_betas)} task profiles, {len(valid_idx)} encoders with probing data")

        print("\n4. Heatmap (1): encoder fingerprint dominance...")
        # heatmap_encoder_dominant(probing_profiles, encoders, valid_idx, out_dir)

        print("\n5. Heatmap (2): encoder × task interaction...")
        heatmap_interaction(probing_profiles, encoders, valid_idx, task_betas, scaler, out_dir)
    else:
        print("\n3. Fitting per-encoder LOEO Ridge profiles...")
        per_enc_betas, scaler, valid_idx, Z = fit_task_profiles_per_encoder(
            probing_profiles, encoders, perf_matrix
        )
        print(f"   {len(per_enc_betas)} per-encoder profile sets, {len(valid_idx)} encoders with probing data")

        print("\n4. Heatmap (1): encoder fingerprint dominance...")
        # heatmap_encoder_dominant(probing_profiles, encoders, valid_idx, out_dir)

        print("\n5. Heatmap (2): encoder × task interaction [LOEO]...")
        heatmap_interaction_per_encoder(probing_profiles, encoders, valid_idx, per_enc_betas, scaler, out_dir)

    print("\n6. Standalone legend...")
    standalone_legend(out_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
