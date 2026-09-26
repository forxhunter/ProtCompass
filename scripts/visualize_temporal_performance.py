#!/usr/bin/env python3
"""
Visualize encoder performance over time (by invention date).
Shows trends in IIP probing accuracy and downstream task performance.
"""

import sys, types, warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── mock heavy imports ──────────────────────────────────────────────────────
class _Mock:
    def __getattr__(self, n): return _Mock()
    def __call__(self, *a, **k): return _Mock()

for _m in ['torch_geometric', 'torch_geometric.nn']:
    sys.modules[_m] = _Mock()

class InvariantFingerprint: pass
_src = types.ModuleType('src'); _pr = types.ModuleType('src.probes')
_bat = types.ModuleType('src.probes.probe_battery')
_bat.InvariantFingerprint = InvariantFingerprint
sys.modules.update({'src': _src, 'src.probes': _pr, 'src.probes.probe_battery': _bat})

import torch

# ── paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__import__("os").environ.get("PROTCOMPASS_ROOT") or Path(__file__).resolve().parents[1])  # repository root
ARCH_DIR = ROOT / "probing_results_architecture_full"
FALL_DIR = ROOT / "structure_encoder_data" / "probing_results"
DOWN_CSV = ROOT / "outputs" / "downstream_v1" / "downstream_results.csv"
OUT_DIR  = ROOT / "outputs" / "temporal_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── encoder invention dates (from EncoderSelection.tex Table 2) ────────────
ENCODER_DATES = {
    # Sequence
    'esm1b': '2019-04',
    'prottrans': '2020-07',
    'proteinbert_bfd': '2020-07',
    'esm2_35m': '2022-07',
    'esm2_150m': '2022-07',
    'esm2': '2022-07',  # ESM-2 650M
    'ankh': '2023-01',

    # Structure
    'schnet': '2017-06',
    'tfn': '2018-02',
    'tfn_pretrained': '2018-02',
    'gvp': '2020-09',
    'gvp_pretrained': '2020-09',
    'dimenet': '2020-11',
    'egnn': '2021-02',
    'egnn_pretrained': '2021-02',
    'ipa': '2021-07',
    'ipa_pretrained': '2021-07',
    'foldseek': '2022-02',
    'gearnet': '2022-03',
    'gearnet_pretrained': '2022-03',
    'esm_if': '2022-04',
    'mace': '2022-06',
    'proteinmpnn': '2022-06',
    'proteinmpnn_pretrained': '2022-06',
    'gcpnet': '2022-11',
    'gcpnet_pretrained': '2022-11',
    'pottsmpnn': '2022-06',  # Approximate, based on ProteinMPNN
    'pottsmpnn_pretrained': '2022-06',
    'pottsmpnn_soluble_pretrained': '2022-06',
    'pottsmpnn_ft_pretrained': '2022-06',
    'geobpe': '2025-11',
    'dmasif_pretrained': '2020-01',  # Approximate

    # Multimodal
    'cdconv': '2023-02',
    'cdconv_pretrained': '2023-02',
    'prostt5': '2023-07',
    'saprot': '2023-10',
    'saprot_pretrained': '2023-10',
    'esm3': '2024-07',
    'foldvision': '2026-01',
}

def date_to_float(date_str):
    """Convert YYYY-MM to float year (e.g., 2022-07 -> 2022.5)"""
    year, month = map(int, date_str.split('-'))
    return year + (month - 1) / 12.0

# ── IIP mapping ────────────────────────────────────────────────────────────
FAMILY_PROBES = {
    'I1':  ['phi_angle_bins', 'psi_angle_bins'],
    'I2':  ['local_curvature'],
    'I3':  ['dssp_3', 'dssp_8'],
    'I4':  ['rsa', 'rsa_bins'],
    'I5':  ['bfactor'],
    'I6':  ['core_surface', 'contact_density'],
    'I7':  ['distance_bins'],
    'I8':  ['contact_order', 'radius_of_gyration', 'asphericity'],
    'I9':  ['architecture_label'],
    'I10': ['sequence_recovery'],
}

TASK_LABEL = {
    'I1':  'Local Conformation', 'I2':  'Local Curvature',  'I3':  'Sec. Struct.',
    'I4':  'Solv. Access.',      'I5':  'Local Flexibility','I6':  'Packing Density',
    'I7':  'Residue Contacts',   'I8':  'Global Geometry',  'I9':  'Fold Architecture',
    'I10': 'Sequence',
}

DTASK_NAME_TO_NUM = {
    "contact_prediction": "T1", "remote_homology": "T2", "fluorescence": "T3",
    "stability": "T4", "mutation_effect": "T5", "ec_classification": "T6",
    "go_mf": "T7", "go_bp": "T8", "solubility": "T9", "ppi_site": "T10",
    "metal_binding": "T11", "ppi_affinity": "T12",
    "subcellular_localization": "T13", "membrane_soluble": "T14",
}

DTASK_LABEL = {
    "T1": "Contact",   "T2": "Homology",   "T3": "Fluor.",
    "T4": "Stability", "T5": "Mut.Effect", "T6": "EC Class.",
    "T7": "GO-MF",     "T8": "GO-BP",      "T9": "Solubility",
    "T10": "PPI Sites","T11": "Metal Bind.","T12": "PPI Affin.",
    "T13": "Subcell.Loc.","T14": "Mem./Sol.",
}

MODALITY_COLORS = {
    'Sequence':  '#1a4a8a',
    'Structure': '#D2691E',
    'Multimodal':'#7c4fa0',
}

ENCODER_MODALITY = {
    **{e: 'Sequence'  for e in ['ankh','esm1b','esm2','esm2_35m','esm2_150m',
                                 'prottrans','prostt5','proteinbert_bfd']},
    **{e: 'Structure' for e in [
        'foldseek','proteinmpnn','proteinmpnn_pretrained','pottsmpnn','pottsmpnn_pretrained',
        'pottsmpnn_soluble_pretrained','pottsmpnn_ft_pretrained',
        'gearnet','gearnet_pretrained','schnet','dimenet','mace',
        'cdconv','cdconv_pretrained','dmasif_pretrained',
        'egnn','egnn_pretrained','gvp','gvp_pretrained',
        'tfn','tfn_pretrained','ipa','ipa_pretrained','gcpnet','gcpnet_pretrained','geobpe']},
    **{e: 'Multimodal' for e in ['saprot','saprot_pretrained','esm_if','esm3','foldvision']},
}

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.titlesize': 10, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})

def _aggregate_families(detailed):
    out = {}
    for fam, probes in FAMILY_PROBES.items():
        vals = [float(detailed[p]) for p in probes if p in detailed]
        out[fam] = float(np.mean(vals)) if vals else 0.0
    return out

# ── Load IIP probing data ──────────────────────────────────────────────────
print("Loading IIP probing results…")
iip_data = {}
for src in [ARCH_DIR, FALL_DIR]:
    for f in sorted(src.glob('*.pt')):
        try:
            enc = f.stem.rsplit('_seed_', 1)[0]
            if enc not in ENCODER_DATES:
                continue
            data = torch.load(f, map_location='cpu', weights_only=False)
            fp = data['fingerprint']
            detailed = getattr(fp, 'detailed_scores', None)
            if detailed:
                scores = _aggregate_families(detailed)
            else:
                scores = {f"I{i}": float(getattr(fp, f"p_I{i}")) for i in range(1, 11)}

            if enc not in iip_data:
                iip_data[enc] = []
            iip_data[enc].append(scores)
        except Exception:
            continue

# Average across seeds
iip_mean = {}
for enc, seed_list in iip_data.items():
    iip_mean[enc] = {
        task: np.mean([s[task] for s in seed_list])
        for task in [f"I{i}" for i in range(1, 11)]
    }

print(f"  Loaded {len(iip_mean)} encoders with IIP scores")

# ── Load downstream data ───────────────────────────────────────────────────
print("Loading downstream results…")
df = pd.read_csv(DOWN_CSV)
df['T'] = df['task'].map(DTASK_NAME_TO_NUM)
df = df.dropna(subset=['T'])

down_mean = {}
for enc in df['encoder'].unique():
    if enc not in ENCODER_DATES:
        continue
    sub = df[df['encoder'] == enc]
    down_mean[enc] = {}
    for t in DTASK_NAME_TO_NUM.values():
        row = sub[sub['T'] == t]
        if len(row):
            down_mean[enc][t] = row['mean_score'].values[0]

print(f"  Loaded {len(down_mean)} encoders with downstream scores")

# ── Prepare temporal data ──────────────────────────────────────────────────
# Combine encoders that have both IIP and downstream data
common_encoders = sorted(set(iip_mean.keys()) & set(down_mean.keys()))
print(f"  {len(common_encoders)} encoders with both IIP and downstream data")

temporal_data = []
for enc in common_encoders:
    date_str = ENCODER_DATES[enc]
    year_float = date_to_float(date_str)
    modality = ENCODER_MODALITY.get(enc, 'Structure')

    # IIP scores
    for task in [f"I{i}" for i in range(1, 11)]:
        temporal_data.append({
            'encoder': enc,
            'year': year_float,
            'modality': modality,
            'task_type': 'IIP',
            'task': task,
            'score': iip_mean[enc].get(task, np.nan)
        })

    # Downstream scores
    for task in DTASK_NAME_TO_NUM.values():
        temporal_data.append({
            'encoder': enc,
            'year': year_float,
            'modality': modality,
            'task_type': 'Downstream',
            'task': task,
            'score': down_mean[enc].get(task, np.nan)
        })

df_temporal = pd.DataFrame(temporal_data)

# ── Figure 1: Average performance over time (by modality) ──────────────────
print("\nGenerating temporal performance plots…")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# IIP average
ax = axes[0]
for modality in ['Sequence', 'Structure', 'Multimodal']:
    sub = df_temporal[(df_temporal['task_type'] == 'IIP') &
                      (df_temporal['modality'] == modality)]
    if len(sub) == 0:
        continue

    # Group by encoder and year, compute mean IIP score
    grouped = sub.groupby(['encoder', 'year'])['score'].mean().reset_index()
    grouped = grouped.sort_values('year')

    ax.scatter(grouped['year'], grouped['score'],
               color=MODALITY_COLORS[modality],
               label=modality, s=80, alpha=0.7, edgecolors='white', linewidth=1)

ax.set_xlabel('Year', fontsize=9, fontweight='bold')
ax.set_ylabel('Mean IIP Probing Accuracy', fontsize=9, fontweight='bold')
ax.set_title('IIP Performance Over Time', fontsize=10, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9, fontsize=7)
ax.set_ylim(0, 1.0)

# Downstream average
ax = axes[1]
for modality in ['Sequence', 'Structure', 'Multimodal']:
    sub = df_temporal[(df_temporal['task_type'] == 'Downstream') &
                      (df_temporal['modality'] == modality)]
    if len(sub) == 0:
        continue

    # Group by encoder and year, compute mean downstream score
    grouped = sub.groupby(['encoder', 'year'])['score'].mean().reset_index()
    grouped = grouped.sort_values('year')

    ax.scatter(grouped['year'], grouped['score'],
               color=MODALITY_COLORS[modality],
               label=modality, s=80, alpha=0.7, edgecolors='white', linewidth=1)

ax.set_xlabel('Year', fontsize=9, fontweight='bold')
ax.set_ylabel('Mean Downstream Score', fontsize=9, fontweight='bold')
ax.set_title('Downstream Performance Over Time', fontsize=10, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9, fontsize=7)
ax.set_ylim(0, 1.0)

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'temporal_performance_overview.{ext}', dpi=300, bbox_inches='tight')
print(f"  Saved temporal_performance_overview.png/pdf")
plt.close()

# ── Figure 2: Per-task temporal trends (heatmap style) ─────────────────────
# Show all tasks (IIP + Downstream) on y-axis, time on x-axis, color = score

# Bin years into discrete time points
years = sorted(df_temporal['year'].unique())
year_bins = np.arange(2017, 2027, 1)  # 2017, 2018, ..., 2026
year_labels = [str(int(y)) for y in year_bins]

# All tasks
all_tasks = [f"I{i}" for i in range(1, 11)] + [f"T{i}" for i in range(1, 15)]
task_labels = [TASK_LABEL.get(t, t) if t.startswith('I') else DTASK_LABEL.get(t, t)
               for t in all_tasks]

# Create matrix: tasks × years, value = max score achieved by that year
perf_matrix = np.full((len(all_tasks), len(year_bins)), np.nan)

for i, task in enumerate(all_tasks):
    task_type = 'IIP' if task.startswith('I') else 'Downstream'
    sub = df_temporal[(df_temporal['task'] == task) & (df_temporal['task_type'] == task_type)]

    for j, year_bin in enumerate(year_bins):
        # Get all encoders up to this year
        up_to_year = sub[sub['year'] <= year_bin + 0.99]
        if len(up_to_year) > 0:
            perf_matrix[i, j] = up_to_year['score'].max()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(perf_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1.0)

ax.set_xticks(range(len(year_bins)))
ax.set_xticklabels(year_labels, rotation=0)
ax.set_yticks(range(len(all_tasks)))
ax.set_yticklabels(task_labels, fontsize=9)

# Add separator line between IIP and Downstream
ax.axhline(y=9.5, color='black', linewidth=2, linestyle='--')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Best Score Achieved by Year', rotation=270, labelpad=15, fontsize=9)
cbar.ax.tick_params(labelsize=7)

ax.set_xlabel('Year', fontsize=9, fontweight='bold')
ax.set_ylabel('Task', fontsize=9, fontweight='bold')
ax.set_title('Temporal Evolution of Best Performance per Task',
             fontsize=10, fontweight='bold', pad=15)

# Add text annotations for IIP and Downstream sections
ax.text(-1.5, 4.5, 'IIP\nTasks', rotation=90, va='center', ha='center',
        fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(-1.5, 16.5, 'Downstream\nTasks', rotation=90, va='center', ha='center',
        fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'temporal_heatmap_all_tasks.{ext}', dpi=300, bbox_inches='tight')
print(f"  Saved temporal_heatmap_all_tasks.png/pdf")
plt.close()

# ── Figure 3: Trend lines with best-so-far envelope ───────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# IIP tasks
ax = axes[0]
for task in [f"I{i}" for i in range(1, 11)]:
    sub = df_temporal[(df_temporal['task'] == task) & (df_temporal['task_type'] == 'IIP')]
    sub = sub.sort_values('year')

    # Compute cumulative max (best-so-far)
    years_sorted = sorted(sub['year'].unique())
    best_so_far = []
    for y in years_sorted:
        best_so_far.append(sub[sub['year'] <= y]['score'].max())

    ax.plot(years_sorted, best_so_far, marker='o', label=TASK_LABEL[task],
            linewidth=2, markersize=5, alpha=0.8)

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Best IIP Accuracy Achieved', fontsize=12, fontweight='bold')
ax.set_title('IIP Tasks: Best Performance Over Time (Cumulative)',
             fontsize=13, fontweight='bold')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=9)
ax.set_ylim(0, 1.05)

# Downstream tasks
ax = axes[1]
for task in [f"T{i}" for i in range(1, 15)]:
    sub = df_temporal[(df_temporal['task'] == task) & (df_temporal['task_type'] == 'Downstream')]
    sub = sub.sort_values('year')

    years_sorted = sorted(sub['year'].unique())
    best_so_far = []
    for y in years_sorted:
        best_so_far.append(sub[sub['year'] <= y]['score'].max())

    ax.plot(years_sorted, best_so_far, marker='o', label=DTASK_LABEL[task],
            linewidth=2, markersize=5, alpha=0.8)

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Best Downstream Score Achieved', fontsize=12, fontweight='bold')
ax.set_title('Downstream Tasks: Best Performance Over Time (Cumulative)',
             fontsize=13, fontweight='bold')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=9)
ax.set_ylim(0, 1.05)

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'temporal_trends_cumulative.{ext}', dpi=300, bbox_inches='tight')
print(f"  Saved temporal_trends_cumulative.png/pdf")
plt.close()

print(f"\nAll temporal analysis figures saved to {OUT_DIR}")
