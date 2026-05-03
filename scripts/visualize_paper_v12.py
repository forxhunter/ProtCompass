#!/usr/bin/env python3
"""
Paper figures v12 — encoder performance across invariant families.

Key change from v9: Apply pretrained-only filter to ALL modalities, not just structure.
If both encoder and encoder_pretrained exist, show only encoder_pretrained (with suffix stripped).

Figure 1: Main heatmap — encoders (grouped Seq→Struct→Multi) × 10 tasks
Figures 2–4: Radar fingerprint charts per modality
Figure 5: PCA of all encoder probing vectors in 2D performance-space

Color palette: purple/blue throughout.
Output: outputs/paper_figures_v12/
"""

import sys, types, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from math import pi
from pathlib import Path
from sklearn.decomposition import PCA

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
OUT_DIR  = ROOT / "outputs" / "paper_figures_v12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── taxonomy ───────────────────────────────────────────────────────────────
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

# ── pretrained-only filter for ALL encoders ───────────────────────────────
def is_shown(enc, all_encoders):
    """
    Show only pretrained version if both encoder and encoder_pretrained exist.
    For encoders without a pretrained variant, show the base version.
    """
    # If this is a pretrained version, always show it
    if '_pretrained' in enc:
        return True

    # If this is a base version, only show if pretrained doesn't exist
    pretrained_version = f"{enc}_pretrained"
    return pretrained_version not in all_encoders

DISPLAY_NAMES = {
    'ankh': 'Ankh', 'esm1b': 'ESM-1b', 'esm2': 'ESM-2(650M)', 'esm2_35m': 'ESM-2(35M)', 'esm2_150m': 'ESM-2(150M)',
    'prottrans': 'ProtTrans', 'proteinbert_bfd': 'ProteinBERT',
    'foldseek': 'FoldSeek', 'proteinmpnn': 'ProteinMPNN', 'pottsmpnn': 'PottsMPNN', 
    'pottsmpnn_soluble': 'PottsMPNN(Sol)', 'pottsmpnn_ft': 'PottsMPNN(FT)', 
    'gearnet': 'GearNet', 'schnet': 'SchNet', 'dimenet': 'DimeNet++', 'mace': 'MACE', 
    'dmasif': 'dMaSIF', 'egnn': 'EGNN', 'gvp': 'GVP', 'tfn': 'TFN', 'ipa': 'IPA', 
    'gcpnet': 'GCPNet', 'esm_if': 'ESM-IF',
    'saprot': 'SaProt', 'prostt5': 'ProstT5', 'esm3': 'ESM3', 'foldvision': 'FoldVision', 'cdconv': 'CDConv',
    'random': 'Random', 'length': 'Length', 'torsion': 'Torsion'
}

def disp(enc):
    """Strip _pretrained suffix for display and format."""
    clean = enc.replace('_pretrained', '')
    return DISPLAY_NAMES.get(clean, clean)

# ── purple/blue/orange color palette ──────────────────────────────────────
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

# Light-to-dark palettes for individual encoders
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

# heatmap colormap: light teal → deep teal
HMAP_CMAP = LinearSegmentedColormap.from_list(
    'paper_teal',
    ['#f2f9f9', '#aed9d9', '#5eb5b5', '#2b8c8c', '#144d4d'],
    N=256
)

TASKS = [f"I{i}" for i in range(1, 11)]
TASK_SHORT = {
    'I1':  'Local\nConf.',     'I2':  'Local\nCurv.',     'I3':  'Sec.\nStruct.',
    'I4':  'Solv.\nAccess.',   'I5':  'Local\nFlex.',     'I6':  'Packing\nDensity',
    'I7':  'Residue\nContacts','I8':  'Global\nGeom.',    'I9':  'Fold\nArch.',
    'I10': 'Sequence',
}
TASK_LABEL = {
    'I1':  'Local Conformation', 'I2':  'Local Curvature',  'I3':  'Sec. Struct.',
    'I4':  'Solv. Access.',      'I5':  'Local Flexibility','I6':  'Packing Density',
    'I7':  'Residue Contacts',   'I8':  'Global Geometry',  'I9':  'Fold Architecture',
    'I10': 'Sequence',
}

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.titlesize': 10, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})

# ── new IIP (Invariant Information Property) → probe mapping ─────────────
# Recomputes IIP scores from `detailed_scores` so the figure labels match
# the actual probes aggregated under each property.
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
    """Mean of available per-probe scores within each family."""
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
                    # Fallback: use stored p_I* (legacy semantics)
                    all_data.setdefault(enc, {})[seed] = {
                        f"I{i}": float(getattr(fp, f"p_I{i}")) for i in range(1, 11)
                    }
            except Exception:
                continue
    return all_data

print("Loading results…")
raw_data = load_results()

def sort_key(enc):
    order = {
        'Sequence': 0, 
        'Struct: Token': 1, 
        'Struct: Graph': 2, 
        'Struct: Equiv': 3, 
        'Multimodal': 4, 
        'Baseline': 5
    }
    stype = ENCODER_SUBTYPE.get(enc, 'Baseline')
    return (order.get(stype, 6), enc)

# full encoder list (for reference)
all_encoders = sorted(raw_data.keys(), key=sort_key)
print(f"  Total encoders loaded: {len(all_encoders)}")

# filtered list: show only pretrained if both versions exist
# Drop geobpe — no downstream results yet, so excluded from final results figure.
EXCLUDE = {'geobpe'}
encoders = [e for e in all_encoders
            if is_shown(e, all_encoders)
            and e.replace('_pretrained', '') not in EXCLUDE]
n_enc    = len(encoders)
n_task   = len(TASKS)
print(f"  {n_enc} encoders after pretrained-only filter (geobpe excluded)")
print(f"  Filtered encoders: {[disp(e) for e in encoders]}")

mean_mat = np.zeros((n_enc, n_task))
std_mat  = np.zeros((n_enc, n_task))
for i, enc in enumerate(encoders):
    for j, task in enumerate(TASKS):
        vals = [raw_data[enc][s][task] for s in raw_data[enc]]
        mean_mat[i, j] = np.mean(vals)
        std_mat[i, j]  = np.std(vals)

# ── Figure 1: main heatmap ─────────────────────────────────────────────────
print("\nFigure 1: main heatmap…")

# Heatmap takes 4.5" width, leaving 2.25" for legend (total 6.75")
# Height adjusted to make cells square: 31 encoders / 10 tasks = 3.1 ratio
# So height = 4.5 / 3.1 = ~1.45"
fig_height = 4.5 * n_task / n_enc + 0.5
fig, ax = plt.subplots(figsize=(4.5, fig_height))

im = ax.imshow(mean_mat.T, cmap=HMAP_CMAP, aspect='auto', vmin=0.05, vmax=0.95,
               interpolation='nearest')

# x-axis: cleaned encoder names, colored by subtype
disp_names = [disp(e) for e in encoders]
ax.set_xticks(np.arange(n_enc))
ax.set_xticklabels(disp_names, rotation=45, fontsize=5, ha='right')
for tick, enc in zip(ax.get_xticklabels(), encoders):
    stype = ENCODER_SUBTYPE.get(enc, 'Baseline')
    tick.set_color(SUBTYPE_COLORS.get(stype, '#333'))

# y-axis
ax.set_yticks(np.arange(n_task))
ax.set_yticklabels([TASK_LABEL[t] for t in TASKS], fontsize=5.5)

# white group separators
prev_stype = None
for i, enc in enumerate(encoders):
    stype = ENCODER_SUBTYPE.get(enc, 'Baseline')
    if stype != prev_stype and prev_stype is not None:
        ax.axvline(i - 0.5, color='white', linewidth=2.5)
    prev_stype = stype

# Modality labels removed - will be explained in caption

cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
# cbar.set_label('Probe Accuracy', rotation=270, labelpad=14, fontsize=8.5)
cbar.ax.tick_params(labelsize=6)

# Legend removed - modality labels now appear below x-axis

# ax.set_xlabel('Encoder  (grouped by input modality)', fontweight='bold', labelpad=6)
# ax.set_ylabel('Invariant Family', fontweight='bold')
# ax.set_title('Encoder Performance Across Invariant Families', fontweight='bold', pad=14)

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'fig1_all_encoders_heatmap.{ext}', dpi=300, bbox_inches='tight')
print("  saved fig1")
plt.close()


# ── Radar chart helper ─────────────────────────────────────────────────────
def radar_chart(ax, values_list, labels, line_colors, mean_color, title, ylim=1.0):
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=5)
    ax.set_ylim(0, ylim)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=4.5, color='grey')
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.spines['polar'].set_visible(False)

    all_vals = []
    for i, vals in enumerate(values_list):
        v = list(vals) + [vals[0]]
        ax.plot(angles, v, linewidth=0.6, alpha=0.4, color=line_colors[i])
        ax.fill(angles, v, alpha=0.05, color=line_colors[i])
        all_vals.append(vals)

    mu = np.mean(all_vals, axis=0)
    mv = list(mu) + [mu[0]]
    ax.plot(angles, mv, linewidth=1.5, color=mean_color,
            marker='o', markersize=2.5,
            markeredgecolor='white', markeredgewidth=0.5, zorder=5)
    ax.fill(angles, mv, alpha=0.20, color=mean_color)

    # Smart label positioning to avoid overlap when values are close to 1
    # Place labels outside the plot area with intelligent offset
    for angle, label_text in zip(angles[:-1], labels):
        # Calculate position for label
        x_pos = angle
        # Check if any encoder has high value (>0.85) at this task
        task_idx = angles[:-1].index(angle)
        max_val = max(vals[task_idx] for vals in all_vals)

        # If values are high, push label further out
        if max_val > 0.85:
            label_radius = ylim * 1.25  # Push further out
        else:
            label_radius = ylim * 1.15  # Normal distance

        # Get the label and reposition it
        ax.text(x_pos, label_radius, label_text,
                ha='center', va='center', fontsize=5)

    # Clear default labels to avoid duplication
    ax.set_xticklabels([])

    # Title removed - will be added in paper caption
    # ax.set_title(title, fontweight='bold', pad=18, fontsize=10)


# ── Figure 2: Sequence radar ───────────────────────────────────────────────
print("Figure 2: Sequence radar…")
seq_encs = [e for e in encoders if ENCODER_MODALITY.get(e) == 'Sequence']
seq_vals = [mean_mat[encoders.index(e)] for e in seq_encs]

fig, ax = plt.subplots(figsize=(2.25, 2.25), subplot_kw=dict(projection='polar'))
radar_chart(ax, seq_vals,
            labels=[TASK_SHORT[t] for t in TASKS],
            line_colors=SEQ_PALETTE[:len(seq_encs)],
            mean_color=MODALITY_COLORS['Sequence'],
            title=f'Sequence Encoders  (n={len(seq_encs)})')

# Legend removed - see fig_legend_all.png for standalone legend
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'fig2_sequence_radar.{ext}', dpi=300, bbox_inches='tight')
print("  saved fig2")
plt.close()


# ── Figure 3: Structure radar (distinct orange/red shades by subtype) ──────
print("Figure 3: Structure radar…")
struct_encs = [e for e in encoders if ENCODER_MODALITY.get(e) == 'Structure']
struct_vals = [mean_mat[encoders.index(e)] for e in struct_encs]

# Assign colors by subtype
struct_colors = []
for enc in struct_encs:
    stype = ENCODER_SUBTYPE.get(enc, 'Structure')
    struct_colors.append(SUBTYPE_COLORS.get(stype, MODALITY_COLORS['Structure']))

fig, ax = plt.subplots(figsize=(2.25, 2.25), subplot_kw=dict(projection='polar'))
radar_chart(ax, struct_vals,
            labels=[TASK_SHORT[t] for t in TASKS],
            line_colors=struct_colors,
            mean_color='#D2691E',  # chocolate orange for mean
            title=f'Structure Encoders  (n={len(struct_encs)})')

# Legend removed - see fig_legend_all.png for standalone legend

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'fig3_structure_radar.{ext}', dpi=300, bbox_inches='tight')
print("  saved fig3")
plt.close()


# ── Figure 4: Multimodal radar ─────────────────────────────────────────────
print("Figure 4: Multimodal radar…")
mm_encs = [e for e in encoders if ENCODER_MODALITY.get(e) == 'Multimodal']
mm_vals = [mean_mat[encoders.index(e)] for e in mm_encs]

fig, ax = plt.subplots(figsize=(2.25, 2.25), subplot_kw=dict(projection='polar'))
radar_chart(ax, mm_vals,
            labels=[TASK_SHORT[t] for t in TASKS],
            line_colors=MULTI_PALETTE[:len(mm_encs)],
            mean_color=MODALITY_COLORS['Multimodal'],
            title=f'Multimodal Encoders  (n={len(mm_encs)})')

# Legend removed - see fig_legend_all.png for standalone legend

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'fig4_multimodal_radar.{ext}', dpi=300, bbox_inches='tight')
print("  saved fig4")
plt.close()


# ── Figure 5: PCA performance space ───────────────────────────────────────
print("Figure 5: PCA performance space…")
from scipy.spatial import ConvexHull

main_encs = [e for e in encoders if ENCODER_MODALITY.get(e) in ('Sequence','Structure','Multimodal')]
X  = np.stack([mean_mat[encoders.index(e)] for e in main_encs])
pca = PCA(n_components=2)
X2  = pca.fit_transform(X)
var = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(6.75, 6.75))

for enc, (x, y) in zip(main_encs, X2):
    stype = ENCODER_SUBTYPE.get(enc, ENCODER_MODALITY.get(enc))
    color = SUBTYPE_COLORS.get(stype, MODALITY_COLORS.get(ENCODER_MODALITY.get(enc), '#333'))
    ax.scatter(x, y, color=color, s=55, edgecolors='white', linewidths=0.6, zorder=3)
    ax.annotate(disp(enc), xy=(x, y), xytext=(3, 2),
                textcoords='offset points', fontsize=5.2, color=color, alpha=0.85)

# convex hull per modality
for mod in ['Sequence', 'Structure', 'Multimodal']:
    pts = np.array([X2[i] for i, e in enumerate(main_encs)
                    if ENCODER_MODALITY.get(e) == mod])
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            for simplex in hull.simplices:
                ax.plot(pts[simplex, 0], pts[simplex, 1],
                        color=MODALITY_COLORS[mod], linewidth=1.0,
                        alpha=0.4, linestyle='--')
        except Exception:
            pass

legend_handles = [
    mpatches.Patch(facecolor=MODALITY_COLORS[m], label=m, edgecolor='black', lw=0.5)
    for m in ['Sequence', 'Structure', 'Multimodal']
] + [
    mpatches.Patch(facecolor=SUBTYPE_COLORS[s], label=s, edgecolor='black', lw=0.5)
    for s in ['Struct: Token', 'Struct: Graph', 'Struct: Equiv']
]
ax.legend(handles=legend_handles, loc='best', fontsize=7,
          frameon=True, fancybox=False, edgecolor='black')

ax.set_xlabel(f'PC1  ({var[0]*100:.1f}% variance)', fontweight='bold')
ax.set_ylabel(f'PC2  ({var[1]*100:.1f}% variance)', fontweight='bold')
ax.set_title('Encoder Performance Space\n(PCA of I1–I10 probing scores)',
             fontweight='bold', pad=10)
ax.grid(alpha=0.25, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(OUT_DIR / f'fig5_pca_performance_space.{ext}', dpi=300, bbox_inches='tight')
print("  saved fig5")
plt.close()

print(f"\nAll figures saved to {OUT_DIR}")
