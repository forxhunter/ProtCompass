#!/usr/bin/env python3
"""
Generate uncertainty figure for NeurIPS appendix.
Shows error bars (std dev across 5 seeds) for all encoders across all 10 invariant tasks.
"""

import sys, types, warnings
import numpy as np
import matplotlib.pyplot as plt
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
ROOT     = Path("/home/tianyu16/esm3_separate_machine")
ARCH_DIR = ROOT / "probing_results_architecture_full"
FALL_DIR = ROOT / "structure_encoder_data" / "probing_results"
DOWN_CSV = ROOT / "outputs" / "downstream_v1" / "downstream_results.csv"
OUT_DIR  = ROOT / "outputs" / "uncertainty_appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IIP_OUT = OUT_DIR / "IIP"
IIP_OUT.mkdir(parents=True, exist_ok=True)
DOWN_OUT = OUT_DIR / "downstream"
DOWN_OUT.mkdir(parents=True, exist_ok=True)
ALIGN_OUT = OUT_DIR / "alignment"
ALIGN_OUT.mkdir(parents=True, exist_ok=True)

# ── taxonomy ───────────────────────────────────────────────────────────────
ENCODER_MODALITY = {
    **{e: 'Sequence'  for e in ['ankh','esm1b','esm2','esm2_35m','esm2_150m',
                                 'prottrans','prostt5','proteinbert_bfd']},
    **{e: 'Structure' for e in [
        'foldseek',
        'proteinmpnn','proteinmpnn_pretrained','pottsmpnn','pottsmpnn_pretrained',
        'pottsmpnn_soluble_pretrained','pottsmpnn_ft_pretrained',
        'gearnet','gearnet_pretrained','schnet','dimenet','mace',
        'cdconv','cdconv_pretrained','dmasif_pretrained',
        'egnn','egnn_pretrained','gvp','gvp_pretrained',
        'tfn','tfn_pretrained','ipa','ipa_pretrained','gcpnet','gcpnet_pretrained']},
    **{e: 'Multimodal' for e in ['saprot','saprot_pretrained','esm_if','esm3','foldvision']},
    **{e: 'Baseline'   for e in ['random','length','torsion']},
}

def is_shown(enc, all_encoders):
    """Show only pretrained version if both encoder and encoder_pretrained exist."""
    if '_pretrained' in enc:
        return True
    pretrained_version = f"{enc}_pretrained"
    return pretrained_version not in all_encoders

def disp(enc):
    """Strip _pretrained suffix for display."""
    return enc.replace('_pretrained', '')

MODALITY_COLORS = {
    'Sequence':  '#1a4a8a',
    'Structure': '#D2691E',
    'Multimodal':'#7c4fa0',
    'Baseline':  '#666666',
}

TASKS = [f"I{i}" for i in range(1, 11)]
TASK_LABEL = {
    'I1':  'Local Conformation', 'I2':  'Local Curvature',  'I3':  'Sec. Struct.',
    'I4':  'Solv. Access.',      'I5':  'Local Flexibility','I6':  'Packing Density',
    'I7':  'Residue Contacts',   'I8':  'Global Geometry',  'I9':  'Fold Architecture',
    'I10': 'Sequence',
}

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.titlesize': 10, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})

# ── new IIP (Invariant Information Property) → probe mapping ─────────────
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

# ── load data ──────────────────────────────────────────────────────────────
def load_results():
    all_data = {}
    for src in [ARCH_DIR, FALL_DIR]:
        for f in sorted(src.glob('*.pt')):
            try:
                enc  = f.stem.rsplit('_seed_', 1)[0]
                seed = int(f.stem.rsplit('_seed_', 1)[1])
                if enc in all_data and seed in all_data[enc]:
                    continue
                data = torch.load(f, map_location='cpu', weights_only=False)
                fp   = data['fingerprint']
                detailed = getattr(fp, 'detailed_scores', None)
                if detailed:
                    all_data.setdefault(enc, {})[seed] = _aggregate_families(detailed)
                else:
                    all_data.setdefault(enc, {})[seed] = {
                        f"I{i}": float(getattr(fp, f"p_I{i}")) for i in range(1, 11)
                    }
            except Exception:
                continue
    return all_data

print("Loading results for uncertainty analysis…")
raw_data = load_results()

def sort_key(enc):
    order = {'Sequence': 0, 'Structure': 1, 'Multimodal': 2, 'Baseline': 3}
    return (order.get(ENCODER_MODALITY.get(enc, 'Baseline'), 4), enc)

all_encoders = sorted(raw_data.keys(), key=sort_key)
encoders = [e for e in all_encoders if is_shown(e, all_encoders)]
n_enc = len(encoders)
n_task = len(TASKS)

print(f"  {n_enc} encoders after pretrained-only filter")

# ── compute mean and std ───────────────────────────────────────────────────
mean_mat = np.zeros((n_enc, n_task))
std_mat  = np.zeros((n_enc, n_task))

for i, enc in enumerate(encoders):
    for j, task in enumerate(TASKS):
        vals = [raw_data[enc][s][task] for s in raw_data[enc]]
        mean_mat[i, j] = np.mean(vals)
        std_mat[i, j]  = np.std(vals)

# ── Figure: Uncertainty plots (one per modality) ──────────────────────────
print("\nGenerating uncertainty figures for appendix…")

# Create separate figure for each modality
for modality in ['Sequence', 'Structure', 'Multimodal']:
    # Filter encoders by modality
    mod_encs = [e for e in encoders if ENCODER_MODALITY.get(e) == modality]
    mod_indices = [encoders.index(e) for e in mod_encs]

    if not mod_encs:
        continue

    # Extract data for this modality
    mod_means = mean_mat[mod_indices, :]
    mod_stds = std_mat[mod_indices, :]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot with error bars for each task
    x = np.arange(len(TASKS))
    width = 0.8 / len(mod_encs)

    for i, enc in enumerate(mod_encs):
        offset = (i - len(mod_encs)/2) * width + width/2
        ax.bar(x + offset, mod_means[i], width,
               yerr=mod_stds[i],
               label=disp(enc),
               color=MODALITY_COLORS[modality],
               alpha=0.3 + 0.6 * i / max(1, len(mod_encs)-1),
               capsize=3,
               error_kw={'linewidth': 1, 'elinewidth': 0.8})

    ax.set_xlabel('Invariant Task', fontsize=11)
    ax.set_ylabel('Probing Accuracy', fontsize=11)
    ax.set_title(f'{modality} Encoders: Mean ± Std Dev (5 seeds)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, loc='upper left', ncol=1)

    plt.tight_layout()
    fname = f'appendix_uncertainty_{modality.lower()}'
    for ext in ('png', 'pdf'):
        plt.savefig(IIP_OUT / f'{fname}.{ext}', dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}.png/pdf")
    plt.close()

# ── Additional figure: std dev heatmap ─────────────────────────────────────
print("\nGenerating std dev heatmap…")

fig, ax = plt.subplots(figsize=(10, 0.35 * n_enc))

im = ax.imshow(std_mat, aspect='auto', cmap='Reds', vmin=0, vmax=0.15)

ax.set_xticks(range(n_task))
ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], rotation=45, ha='right')
ax.set_yticks(range(n_enc))
ax.set_yticklabels([disp(e) for e in encoders], fontsize=8)

# Add modality color bars
for i, enc in enumerate(encoders):
    mod = ENCODER_MODALITY.get(enc, 'Baseline')
    ax.add_patch(plt.Rectangle((-0.6, i-0.4), 0.3, 0.8,
                               facecolor=MODALITY_COLORS[mod],
                               edgecolor='none', clip_on=False,
                               transform=ax.get_yaxis_transform()))

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Standard Deviation', rotation=270, labelpad=15)

ax.set_xlabel('Invariant Task', fontsize=10)
ax.set_ylabel('Encoder', fontsize=10)
ax.set_title('Uncertainty Across Seeds (Standard Deviation)', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(IIP_OUT / f'appendix_std_heatmap.{ext}', dpi=300, bbox_inches='tight')
print(f"  Saved std dev heatmap to {IIP_OUT}/appendix_std_heatmap.png/pdf")
plt.close()

print("\nUncertainty analysis (probing) complete!")
print(f"  Mean std dev across all: {np.mean(std_mat):.4f}")
print(f"  Max std dev: {np.max(std_mat):.4f}")
print(f"  Min std dev: {np.min(std_mat):.4f}")


# ───────────────────────────────────────────────────────────────────────────
# Section 2: Downstream-task uncertainty
# ───────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 2: Downstream-task uncertainty")
print("=" * 70)

import pandas as pd

# 14-task ordering matching the paper
DTASKS = [f"T{i}" for i in range(1, 15)]
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

DISPLAY_NAMES = {
    'ankh': 'Ankh', 'esm1b': 'ESM-1b', 'esm2': 'ESM-2(650M)',
    'esm2_35m': 'ESM-2(35M)', 'esm2_150m': 'ESM-2(150M)',
    'prottrans': 'ProtTrans', 'proteinbert_bfd': 'ProteinBERT',
    'prostt5': 'ProstT5', 'foldseek': 'FoldSeek',
    'proteinmpnn': 'ProteinMPNN', 'pottsmpnn': 'PottsMPNN',
    'pottsmpnn_soluble': 'PottsMPNN(Sol)', 'pottsmpnn_ft': 'PottsMPNN(FT)',
    'pottsmpnn_ca': 'PottsMPNN(CA)',
    'gearnet': 'GearNet', 'schnet': 'SchNet', 'dimenet': 'DimeNet++',
    'mace': 'MACE', 'dmasif': 'dMaSIF', 'egnn': 'EGNN', 'gvp': 'GVP',
    'tfn': 'TFN', 'ipa': 'IPA', 'gcpnet': 'GCPNet', 'esm_if': 'ESM-IF',
    'saprot': 'SaProt', 'esm3': 'ESM-3', 'foldvision': 'FoldVision',
    'cdconv': 'CDConv', 'geobpe': 'GeoBPE',
    'random': 'Random', 'length': 'Length', 'torsion': 'Torsion',
    'proteinae': 'ProteinAE',
}

def disp_d(enc):
    clean = enc.replace('_pretrained', '')
    return DISPLAY_NAMES.get(clean, clean)

DOWN_MODALITY = {
    **{e: 'Sequence' for e in [
        'ankh','esm1b','esm2','esm2_35m','esm2_150m','prottrans',
        'proteinbert_bfd','prostt5']},
    **{e: 'Multimodal' for e in [
        'saprot','saprot_pretrained','esm3','foldvision',
        'cdconv','cdconv_pretrained']},
}
def down_modality(enc):
    clean = enc.replace('_pretrained', '')
    if clean in DOWN_MODALITY: return DOWN_MODALITY[clean]
    if enc in DOWN_MODALITY:    return DOWN_MODALITY[enc]
    if clean in ('random','length','torsion'): return 'Baseline'
    return 'Structure'

DOWN_MOD_COLORS = {
    'Sequence': '#1a4a8a', 'Structure': '#D2691E',
    'Multimodal': '#7c4fa0', 'Baseline': '#666666',
}

def down_sort_key(enc):
    order = {'Sequence': 0, 'Structure': 1, 'Multimodal': 2, 'Baseline': 3}
    return (order.get(down_modality(enc), 4), enc)

print(f"Loading downstream results from {DOWN_CSV.name}…")
df = pd.read_csv(DOWN_CSV)
df = df[df['n_seeds'] >= 2].copy()  # need ≥2 seeds for std
df['T'] = df['task'].map(DTASK_NAME_TO_NUM)
df = df.dropna(subset=['T'])

d_encoders_all = sorted(df['encoder'].unique(), key=down_sort_key)
def is_shown_d(enc, all_encs):
    if '_pretrained' in enc: return True
    return f"{enc}_pretrained" not in all_encs
d_encoders = [e for e in d_encoders_all if is_shown_d(e, d_encoders_all)]
n_d = len(d_encoders); n_dt = len(DTASKS)
print(f"  {n_d} encoders × {n_dt} tasks (after pretrained-only filter)")

d_mean = np.full((n_d, n_dt), np.nan)
d_std  = np.full((n_d, n_dt), np.nan)
for i, enc in enumerate(d_encoders):
    sub = df[df['encoder'] == enc]
    for j, t in enumerate(DTASKS):
        row = sub[sub['T'] == t]
        if len(row):
            d_mean[i, j] = row['mean_score'].values[0]
            d_std[i, j]  = row['std_score'].values[0]

# ── Per-modality bar plots ─────────────────────────────────────────────────
for modality in ['Sequence', 'Structure', 'Multimodal']:
    mod_idx = [i for i, e in enumerate(d_encoders) if down_modality(e) == modality]
    if not mod_idx:
        continue
    m_mean = d_mean[mod_idx, :]
    m_std  = d_std[mod_idx, :]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(n_dt)
    width = 0.8 / len(mod_idx)
    base_color = DOWN_MOD_COLORS[modality]
    for k, i in enumerate(mod_idx):
        offset = (k - len(mod_idx) / 2) * width + width / 2
        alpha  = 0.35 + 0.6 * k / max(1, len(mod_idx) - 1)
        valid  = ~np.isnan(m_mean[k])
        ax.bar(x[valid] + offset, m_mean[k][valid], width,
               yerr=m_std[k][valid],
               label=disp_d(d_encoders[i]),
               color=base_color, alpha=alpha,
               capsize=3, error_kw={'linewidth': 0.9, 'elinewidth': 0.7})

    ax.set_xlabel('Downstream Task', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{modality} Encoders: Downstream Mean ± Std Dev (5 seeds)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DTASK_LABEL[t] for t in DTASKS], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, loc='upper left', ncol=2)

    plt.tight_layout()
    fname = f'appendix_uncertainty_downstream_{modality.lower()}'
    for ext in ('png', 'pdf'):
        plt.savefig(DOWN_OUT / f'{fname}.{ext}', dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}.png/pdf")
    plt.close()

# ── Std heatmap (encoders × tasks) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 0.32 * n_d + 1))
im = ax.imshow(d_std, aspect='auto', cmap='Reds', vmin=0, vmax=0.10)
ax.set_xticks(range(n_dt))
ax.set_xticklabels([DTASK_LABEL[t] for t in DTASKS], rotation=45, ha='right')
ax.set_yticks(range(n_d))
ax.set_yticklabels([disp_d(e) for e in d_encoders], fontsize=8)
for i, enc in enumerate(d_encoders):
    mod = down_modality(enc)
    ax.add_patch(plt.Rectangle((-0.6, i - 0.4), 0.3, 0.8,
                               facecolor=DOWN_MOD_COLORS[mod],
                               edgecolor='none', clip_on=False,
                               transform=ax.get_yaxis_transform()))
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Standard Deviation', rotation=270, labelpad=15)
ax.set_xlabel('Downstream Task', fontsize=10)
ax.set_ylabel('Encoder', fontsize=10)
ax.set_title('Downstream Uncertainty Across Seeds (Standard Deviation)',
             fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(DOWN_OUT / f'appendix_std_heatmap_downstream.{ext}',
                dpi=300, bbox_inches='tight')
plt.close()

valid_d = ~np.isnan(d_std)
print(f"  Mean downstream std: {np.nanmean(d_std):.4f}")
print(f"  Max downstream std:  {np.nanmax(d_std):.4f}")
print(f"  Min downstream std:  {np.nanmin(d_std):.4f}")


# ───────────────────────────────────────────────────────────────────────────
# Section 3: IAH (alignment) uncertainty
# ───────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Section 3: IAH (alignment / ESP) uncertainty")
print("=" * 70)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# Build per-seed probing matrices using the raw_data already loaded.
# raw_data: enc -> seed -> {Ii: value}
seeds_by_enc = {e: sorted(raw_data[e].keys()) for e in raw_data}
common_seeds = sorted(set.intersection(*[set(s) for s in seeds_by_enc.values()])) \
                if seeds_by_enc else []
if not common_seeds:
    # Fallback: union of seeds; for missing seeds we'll skip the encoder for that seed
    all_seeds = sorted({s for ss in seeds_by_enc.values() for s in ss})
    common_seeds = all_seeds
print(f"  Seeds available for alignment recomputation: {common_seeds}")

# Match probing encoders against downstream encoders.
align_encoders = sorted([e for e in d_encoders if e in raw_data], key=down_sort_key)
n_a = len(align_encoders)
print(f"  {n_a} encoders shared between probing and downstream")

# Mean downstream performance matrix for these encoders
y_mat = np.full((n_a, n_dt), np.nan)
for i, enc in enumerate(align_encoders):
    sub = df[df['encoder'] == enc]
    for j, t in enumerate(DTASKS):
        row = sub[sub['T'] == t]
        if len(row):
            y_mat[i, j] = row['mean_score'].values[0]

def build_Z_seed(seed):
    Z = np.full((n_a, n_task), np.nan)
    for i, enc in enumerate(align_encoders):
        if seed in raw_data[enc]:
            Z[i] = [raw_data[enc][seed][t] for t in TASKS]
    return Z

def fit_task_profiles(Z, y_mat):
    profiles = np.zeros((n_dt, n_task))
    for j in range(n_dt):
        y = y_mat[:, j]
        valid = ~(np.isnan(y) | np.isnan(Z).any(axis=1))
        if valid.sum() < 5:
            continue
        scaler = StandardScaler()
        Zs = scaler.fit_transform(Z[valid])
        best, best_score = 1.0, -np.inf
        for a in [0.01, 0.1, 1.0, 10.0, 100.0]:
            m = Ridge(alpha=a).fit(Zs, y[valid])
            sc = m.score(Zs, y[valid])
            if sc > best_score:
                best_score, best = sc, a
        m = Ridge(alpha=best).fit(Zs, y[valid])
        profiles[j] = m.coef_
    return profiles

def cosine_align(z, w):
    nz, nw = np.linalg.norm(z), np.linalg.norm(w)
    if nz == 0 or nw == 0: return 0.0
    return float(np.dot(z, w) / (nz * nw))

def alignment_matrix(Z, profiles):
    A = np.full((n_a, n_dt), np.nan)
    for i in range(n_a):
        if np.isnan(Z[i]).any(): continue
        for j in range(n_dt):
            A[i, j] = cosine_align(Z[i], profiles[j])
    return A

def esp_predict(Z, y_mat, K):
    pred = np.full_like(y_mat, np.nan, dtype=float)
    valid_e = ~np.isnan(Z).any(axis=1)
    Zv = Z[valid_e]
    yv = y_mat[valid_e]
    task_emb = yv.T  # n_t × n_e
    for ho in range(n_dt):
        # nearest-K tasks by L2 over encoder-performance vectors
        d_list = []
        for j in range(n_dt):
            if j == ho: continue
            mask = ~(np.isnan(task_emb[ho]) | np.isnan(task_emb[j]))
            if mask.sum() < 3:
                d_list.append((j, np.inf)); continue
            d_list.append((j, np.linalg.norm(task_emb[ho][mask] - task_emb[j][mask])))
        d_list.sort(key=lambda x: x[1])
        nearest = [j for j, _ in d_list[:K]]

        Xs, ys = [], []
        for j in nearest:
            yj = yv[:, j]
            mk = ~np.isnan(yj)
            if mk.sum() < 5: continue
            Xs.append(Zv[mk]); ys.append(yj[mk])
        if not Xs: continue
        X = np.vstack(Xs); yy = np.concatenate(ys)
        sc = StandardScaler(); Xs_ = sc.fit_transform(X)
        m = Ridge(alpha=0.1).fit(Xs_, yy)
        pred[valid_e, ho] = m.predict(sc.transform(Zv))
    return pred

# Compute alignment + ESP per seed
print("  Computing alignment + ESP across seeds…")
align_per_seed = []
esp3_per_seed = []
esp5_per_seed = []
for s in common_seeds:
    Z_s = build_Z_seed(s)
    if np.isnan(Z_s).any(axis=1).all():
        continue
    Z_filled = np.nan_to_num(Z_s, nan=np.nanmean(Z_s))
    profiles = fit_task_profiles(Z_filled, y_mat)
    A_s = alignment_matrix(Z_filled, profiles)
    align_per_seed.append(A_s)
    esp3_per_seed.append(esp_predict(Z_filled, y_mat, K=3))
    esp5_per_seed.append(esp_predict(Z_filled, y_mat, K=5))
    print(f"    seed {s}: done")

align_stack = np.stack(align_per_seed)  # (n_seeds, n_a, n_dt)
esp3_stack  = np.stack(esp3_per_seed)
esp5_stack  = np.stack(esp5_per_seed)

align_mean = np.nanmean(align_stack, axis=0)
align_std  = np.nanstd(align_stack,  axis=0)
esp3_mean  = np.nanmean(esp3_stack,  axis=0)
esp3_std   = np.nanstd(esp3_stack,   axis=0)
esp5_mean  = np.nanmean(esp5_stack,  axis=0)
esp5_std   = np.nanstd(esp5_stack,   axis=0)

# ── Alignment std heatmap ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 0.32 * n_a + 1))
vmax = float(np.nanpercentile(align_std, 95)) if np.isfinite(np.nanmax(align_std)) else 0.1
im = ax.imshow(align_std, aspect='auto', cmap='Reds', vmin=0, vmax=max(vmax, 0.05))
ax.set_xticks(range(n_dt))
ax.set_xticklabels([DTASK_LABEL[t] for t in DTASKS], rotation=45, ha='right')
ax.set_yticks(range(n_a))
ax.set_yticklabels([disp_d(e) for e in align_encoders], fontsize=8)
for i, enc in enumerate(align_encoders):
    mod = down_modality(enc)
    ax.add_patch(plt.Rectangle((-0.6, i - 0.4), 0.3, 0.8,
                               facecolor=DOWN_MOD_COLORS[mod],
                               edgecolor='none', clip_on=False,
                               transform=ax.get_yaxis_transform()))
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'Std of $\mathrm{Align}(e,t)$', rotation=270, labelpad=15)
ax.set_xlabel('Downstream Task', fontsize=10)
ax.set_ylabel('Encoder', fontsize=10)
ax.set_title('Alignment Cosine: Std Across Probing Seeds',
             fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(ALIGN_OUT / f'appendix_align_std_heatmap.{ext}',
                dpi=300, bbox_inches='tight')
plt.close()
print(f"  Mean alignment std: {np.nanmean(align_std):.4f}")
print(f"  Max  alignment std: {np.nanmax(align_std):.4f}")

# ── ESP / ESP++ scatter with seed std error bars ───────────────────────────
def _esp_scatter_with_err(pred_mean, pred_std, color, title, fname):
    obs = y_mat
    valid = ~(np.isnan(pred_mean) | np.isnan(obs))
    if valid.sum() == 0: return
    o = obs[valid]; p = pred_mean[valid]; e = pred_std[valid]
    rho, _ = spearmanr(o, p)
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(o, p, yerr=e, fmt='o', ms=4, color=color,
                ecolor=color, alpha=0.55, elinewidth=0.8, capsize=2)
    ax.plot([0, 1], [0, 1], 'r--', lw=1)
    ax.text(0.05, 0.95,
            f'$\\rho$ = {rho:.3f}\nRMSE = {rmse:.3f}\nmean seed std = {np.mean(e):.3f}',
            transform=ax.transAxes, va='top',
            bbox={'facecolor': 'white', 'edgecolor': 'gray', 'alpha': 0.8})
    ax.set_xlabel('Observed performance')
    ax.set_ylabel('Predicted performance (mean ± std over 5 probing seeds)')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(ALIGN_OUT / f'{fname}.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

_esp_scatter_with_err(esp3_mean, esp3_std, '#2E86AB',
                      'ESP (K=3): seed-resampled predictions',
                      'appendix_esp_uncertainty_k3')
_esp_scatter_with_err(esp5_mean, esp5_std, '#A23B72',
                      'ESP++ (K=5): seed-resampled predictions',
                      'appendix_esp_uncertainty_k5')

# Per-seed Spearman ρ (one number per seed, summarized as mean ± std)
def per_seed_rho(stack):
    rhos = []
    for s in range(stack.shape[0]):
        v = ~(np.isnan(stack[s]) | np.isnan(y_mat))
        if v.sum() < 10: continue
        rho, _ = spearmanr(y_mat[v], stack[s][v])
        rhos.append(rho)
    return rhos

r3 = per_seed_rho(esp3_stack)
r5 = per_seed_rho(esp5_stack)
print(f"  ESP   (K=3) per-seed ρ: mean={np.mean(r3):.3f}, std={np.std(r3):.3f}, "
      f"values={[f'{x:.3f}' for x in r3]}")
print(f"  ESP++ (K=5) per-seed ρ: mean={np.mean(r5):.3f}, std={np.std(r5):.3f}, "
      f"values={[f'{x:.3f}' for x in r5]}")

print("\nAll uncertainty figures written to:", OUT_DIR)
