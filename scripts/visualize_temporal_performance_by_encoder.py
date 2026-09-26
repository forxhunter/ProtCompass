#!/usr/bin/env python3
"""
Visualize encoder performance with encoders on x-axis (ordered by invention date).
Shows IIP and downstream task performance for each encoder chronologically.
"""

import sys, types, warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

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

# ── encoder invention dates ────────────────────────────────────────────────
ENCODER_DATES = {
    'schnet': '2017-06',
    'tfn': '2018-02', 'tfn_pretrained': '2018-02',
    'esm1b': '2019-04',
    'dmasif_pretrained': '2020-01',
    'prottrans': '2020-07', 'proteinbert_bfd': '2020-07',
    'gvp': '2020-09', 'gvp_pretrained': '2020-09',
    'dimenet': '2020-11',
    'egnn': '2021-02', 'egnn_pretrained': '2021-02',
    'ipa': '2021-07', 'ipa_pretrained': '2021-07',
    'foldseek': '2022-02',
    'gearnet': '2022-03', 'gearnet_pretrained': '2022-03',
    'esm_if': '2022-04',
    'mace': '2022-06',
    'proteinmpnn': '2022-06', 'proteinmpnn_pretrained': '2022-06',
    'pottsmpnn': '2022-06', 'pottsmpnn_pretrained': '2022-06',
    'pottsmpnn_soluble_pretrained': '2022-06',
    'pottsmpnn_ft_pretrained': '2022-06',
    'esm2_35m': '2022-07', 'esm2_150m': '2022-07', 'esm2': '2022-07',
    'gcpnet': '2022-11', 'gcpnet_pretrained': '2022-11',
    'ankh': '2023-01',
    'cdconv': '2023-02', 'cdconv_pretrained': '2023-02',
    'prostt5': '2023-07',
    'saprot': '2023-10', 'saprot_pretrained': '2023-10',
    'esm3': '2024-07',
    'geobpe': '2025-11',
    'foldvision': '2026-01',
}

DISPLAY_NAMES = {
    'ankh': 'Ankh', 'esm1b': 'ESM-1b', 'esm2': 'ESM-2(650M)',
    'esm2_35m': 'ESM-2(35M)', 'esm2_150m': 'ESM-2(150M)',
    'prottrans': 'ProtTrans', 'proteinbert_bfd': 'ProteinBERT',
    'prostt5': 'ProstT5', 'foldseek': 'FoldSeek',
    'proteinmpnn': 'ProteinMPNN', 'proteinmpnn_pretrained': 'ProteinMPNN',
    'pottsmpnn': 'PottsMPNN', 'pottsmpnn_pretrained': 'PottsMPNN',
    'pottsmpnn_soluble_pretrained': 'PottsMPNN(Sol)',
    'pottsmpnn_ft_pretrained': 'PottsMPNN(FT)',
    'gearnet': 'GearNet', 'gearnet_pretrained': 'GearNet',
    'schnet': 'SchNet', 'dimenet': 'DimeNet++',
    'mace': 'MACE', 'dmasif_pretrained': 'dMaSIF',
    'egnn': 'EGNN', 'egnn_pretrained': 'EGNN',
    'gvp': 'GVP', 'gvp_pretrained': 'GVP',
    'tfn': 'TFN', 'tfn_pretrained': 'TFN',
    'ipa': 'IPA', 'ipa_pretrained': 'IPA',
    'gcpnet': 'GCPNet', 'gcpnet_pretrained': 'GCPNet',
    'esm_if': 'ESM-IF', 'saprot': 'SaProt', 'saprot_pretrained': 'SaProt',
    'esm3': 'ESM3', 'foldvision': 'FoldVision',
    'cdconv': 'CDConv', 'cdconv_pretrained': 'CDConv',
    'geobpe': 'GeoBPE',
}

def date_to_float(date_str):
    year, month = map(int, date_str.split('-'))
    return year + (month - 1) / 12.0

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

MODALITY_COLORS = {
    'Sequence':  '#1a4a8a',
    'Structure': '#D2691E',
    'Multimodal':'#7c4fa0',
}

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
    'I1':  'I1', 'I2':  'I2',  'I3':  'I3',
    'I4':  'I4', 'I5':  'I5',  'I6':  'I6',
    'I7':  'I7', 'I8':  'I8',  'I9':  'I9',
    'I10': 'I10',
}

DTASK_NAME_TO_NUM = {
    "contact_prediction": "T1", "remote_homology": "T2", "fluorescence": "T3",
    "stability": "T4", "mutation_effect": "T5", "ec_classification": "T6",
    "go_mf": "T7", "go_bp": "T8", "solubility": "T9", "ppi_site": "T10",
    "metal_binding": "T11", "ppi_affinity": "T12",
    "subcellular_localization": "T13", "membrane_soluble": "T14",
}

DTASK_LABEL = {
    "T1": "T1", "T2": "T2", "T3": "T3",
    "T4": "T4", "T5": "T5", "T6": "T6",
    "T7": "T7", "T8": "T8", "T9": "T9",
    "T10": "T10", "T11": "T11", "T12": "T12",
    "T13": "T13", "T14": "T14",
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

# ── Sort encoders by date ──────────────────────────────────────────────────
common_encoders = sorted(set(iip_mean.keys()) & set(down_mean.keys()))
print(f"  {len(common_encoders)} encoders with both IIP and downstream data")

# Sort by date, then by name for ties
encoder_order = sorted(common_encoders,
                       key=lambda e: (date_to_float(ENCODER_DATES[e]), e))

# ── Build matrices ─────────────────────────────────────────────────────────
n_enc = len(encoder_order)
iip_tasks = [f"I{i}" for i in range(1, 11)]
down_tasks = [f"T{i}" for i in range(1, 15)]

iip_matrix = np.full((len(iip_tasks), n_enc), np.nan)
down_matrix = np.full((len(down_tasks), n_enc), np.nan)

for i, enc in enumerate(encoder_order):
    for j, task in enumerate(iip_tasks):
        iip_matrix[j, i] = iip_mean[enc].get(task, np.nan)
    for j, task in enumerate(down_tasks):
        down_matrix[j, i] = down_mean[enc].get(task, np.nan)

# ── Figure 1: IIP heatmap (tasks × encoders) ───────────────────────────────
print("\nGenerating encoder-ordered heatmaps…")

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(iip_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1.0)

ax.set_xticks(range(n_enc))
ax.set_xticklabels([DISPLAY_NAMES.get(e, e) for e in encoder_order],
                    rotation=90, ha='right', fontsize=7)
ax.set_yticks(range(len(iip_tasks)))
ax.set_yticklabels([TASK_LABEL[t] for t in iip_tasks], fontsize=7)

# Add modality color bars on top
for i, enc in enumerate(encoder_order):
    mod = ENCODER_MODALITY.get(enc, 'Structure')
    ax.add_patch(plt.Rectangle((i - 0.4, -0.6), 0.8, 0.3,
                               facecolor=MODALITY_COLORS[mod],
                               edgecolor='none', clip_on=False,
                               transform=ax.get_xaxis_transform()))

# Add year labels
year_changes = []
prev_year = None
for i, enc in enumerate(encoder_order):
    year = ENCODER_DATES[enc][:4]
    if year != prev_year:
        year_changes.append((i, year))
        prev_year = year

for i, year in year_changes:
    ax.axvline(x=i - 0.5, color='white', linewidth=2, linestyle='--', alpha=0.5)
    ax.text(i, len(iip_tasks) + 0.3, year, ha='left', va='bottom',
            fontsize=7, fontweight='bold', rotation=0)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('IIP Probing Accuracy', rotation=270, labelpad=15, fontsize=7)

ax.set_xlabel('Encoder (ordered by invention date)', fontsize=9, fontweight='bold')
ax.set_ylabel('IIP Task', fontsize=9, fontweight='bold')
ax.set_title('IIP Performance Across Encoders (Chronological Order)',
             fontsize=10, fontweight='bold', pad=20)

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'temporal_iip_by_encoder.{ext}', dpi=300, bbox_inches='tight')
print(f"  Saved temporal_iip_by_encoder.png/pdf")
plt.close()

# ── Figure 2: Downstream heatmap (tasks × encoders) ────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(down_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1.0)

ax.set_xticks(range(n_enc))
ax.set_xticklabels([DISPLAY_NAMES.get(e, e) for e in encoder_order],
                    rotation=90, ha='right', fontsize=7)
ax.set_yticks(range(len(down_tasks)))
ax.set_yticklabels([DTASK_LABEL[t] for t in down_tasks], fontsize=7)

# Add modality color bars on top
for i, enc in enumerate(encoder_order):
    mod = ENCODER_MODALITY.get(enc, 'Structure')
    ax.add_patch(plt.Rectangle((i - 0.4, -0.6), 0.8, 0.3,
                               facecolor=MODALITY_COLORS[mod],
                               edgecolor='none', clip_on=False,
                               transform=ax.get_xaxis_transform()))

# Add year labels
for i, year in year_changes:
    ax.axvline(x=i - 0.5, color='white', linewidth=2, linestyle='--', alpha=0.5)
    ax.text(i, len(down_tasks) + 0.3, year, ha='left', va='bottom',
            fontsize=7, fontweight='bold', rotation=0)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Downstream Score', rotation=270, labelpad=15, fontsize=7)

ax.set_xlabel('Encoder (ordered by invention date)', fontsize=9, fontweight='bold')
ax.set_ylabel('Downstream Task', fontsize=9, fontweight='bold')
ax.set_title('Downstream Performance Across Encoders (Chronological Order)',
             fontsize=10, fontweight='bold', pad=20)

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'temporal_downstream_by_encoder.{ext}', dpi=300, bbox_inches='tight')
print(f"  Saved temporal_downstream_by_encoder.png/pdf")
plt.close()

# ── Figure 3: Combined heatmap (all tasks × encoders) ──────────────────────
all_matrix = np.vstack([iip_matrix, down_matrix])
all_task_labels = [TASK_LABEL[t] for t in iip_tasks] + [DTASK_LABEL[t] for t in down_tasks]

fig, ax = plt.subplots(figsize=(16, 12))
im = ax.imshow(all_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1.0)

ax.set_xticks(range(n_enc))
ax.set_xticklabels([DISPLAY_NAMES.get(e, e) for e in encoder_order],
                    rotation=90, ha='right', fontsize=7)
ax.set_yticks(range(len(all_task_labels)))
ax.set_yticklabels(all_task_labels, fontsize=7)

# Add separator line between IIP and Downstream
ax.axhline(y=len(iip_tasks) - 0.5, color='black', linewidth=3, linestyle='-')

# Add modality color bars on top
for i, enc in enumerate(encoder_order):
    mod = ENCODER_MODALITY.get(enc, 'Structure')
    ax.add_patch(plt.Rectangle((i - 0.4, -0.6), 0.8, 0.3,
                               facecolor=MODALITY_COLORS[mod],
                               edgecolor='none', clip_on=False,
                               transform=ax.get_xaxis_transform()))

# Add year labels
for i, year in year_changes:
    ax.axvline(x=i - 0.5, color='white', linewidth=2, linestyle='--', alpha=0.5)
    ax.text(i, len(all_task_labels) + 0.5, year, ha='left', va='bottom',
            fontsize=7, fontweight='bold', rotation=0)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Score', rotation=270, labelpad=15, fontsize=7)

ax.set_xlabel('Encoder (ordered by invention date)', fontsize=9, fontweight='bold')
ax.set_ylabel('Task', fontsize=9, fontweight='bold')
ax.set_title('All Tasks Performance Across Encoders (Chronological Order)',
             fontsize=10, fontweight='bold', pad=20)

# Add section labels
ax.text(-2, len(iip_tasks) / 2, 'IIP', rotation=90, va='center', ha='center',
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(-2, len(iip_tasks) + len(down_tasks) / 2, 'Downstream',
        rotation=90, va='center', ha='center',
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'temporal_all_tasks_by_encoder.{ext}', dpi=300, bbox_inches='tight')
print(f"  Saved temporal_all_tasks_by_encoder.png/pdf")
plt.close()

# ── Figure 4: Average performance line plot ────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# IIP average
ax = axes[0]
iip_avg = np.nanmean(iip_matrix, axis=0)
x = np.arange(n_enc)

colors = [MODALITY_COLORS[ENCODER_MODALITY.get(e, 'Structure')] for e in encoder_order]
ax.bar(x, iip_avg, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)

# Add year separators
for i, year in year_changes:
    ax.axvline(x=i - 0.5, color='gray', linewidth=1.5, linestyle='--', alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels([DISPLAY_NAMES.get(e, e) for e in encoder_order],
                    rotation=90, ha='right', fontsize=7)
ax.set_ylabel('Mean IIP Accuracy', fontsize=9, fontweight='bold')
ax.set_title('Mean IIP Performance by Encoder (Chronological Order)',
             fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

# Downstream average
ax = axes[1]
down_avg = np.nanmean(down_matrix, axis=0)

ax.bar(x, down_avg, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)

for i, year in year_changes:
    ax.axvline(x=i - 0.5, color='gray', linewidth=1.5, linestyle='--', alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels([DISPLAY_NAMES.get(e, e) for e in encoder_order],
                    rotation=90, ha='right', fontsize=7)
ax.set_ylabel('Mean Downstream Score', fontsize=9, fontweight='bold')
ax.set_xlabel('Encoder (ordered by invention date)', fontsize=9, fontweight='bold')
ax.set_title('Mean Downstream Performance by Encoder (Chronological Order)',
             fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MODALITY_COLORS[m], label=m)
                   for m in ['Sequence', 'Structure', 'Multimodal']]
axes[0].legend(handles=legend_elements, loc='upper left', fontsize=7)

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'temporal_average_by_encoder.{ext}', dpi=300, bbox_inches='tight')
print(f"  Saved temporal_average_by_encoder.png/pdf")
plt.close()

print(f"\nAll encoder-ordered temporal figures saved to {OUT_DIR}")
