#!/usr/bin/env python3
"""
Generate standalone legend figures for paper figures 2-4.
Separate legends by modality: Sequence (blue), Structure (orange), Multimodal (purple).
"""

import sys, types, warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

def is_shown(enc, all_encoders):
    """Show only pretrained version if both encoder and encoder_pretrained exist."""
    if '_pretrained' in enc:
        return True
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
                all_data.setdefault(enc, {})[seed] = {
                    f"I{i}": float(getattr(fp, f"p_I{i}")) for i in range(1, 11)
                }
            except Exception:
                continue
    return all_data

print("Loading results for legend generation…")
raw_data = load_results()

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

def sort_key(enc):
    # Match downstream_v1.py: order by subtype, then alphabetically within subtype
    order = {
        'Sequence': 0, 'Struct: Token': 1, 'Struct: Graph': 2,
        'Struct: Equiv': 3, 'Multimodal': 4, 'Baseline': 5
    }
    stype = ENCODER_SUBTYPE.get(enc, 'Baseline')
    return (order.get(stype, 6), enc)

all_encoders = sorted(raw_data.keys(), key=sort_key)
encoders = [e for e in all_encoders if is_shown(e, all_encoders)]

# ── color palettes (matching main figures) ─────────────────────────────────
SEQ_PALETTE = ['#B0E0E6', '#87CEEB', '#5bacd4', '#4682B4', '#4169E1', '#0000CD', '#00008B', '#1a4a8a']
STRUCT_PALETTE = [
    '#FFF4D6', '#FFEAB3', '#FFE0A3', '#FFD580', '#FFCC66', '#FFC04D',
    '#FFB347', '#FFA500', '#F4A460', '#F39C12', '#E67E22', '#D2691E',
    '#CD7F32', '#B8621B', '#A0522D', '#8B2500'
]
MULTI_PALETTE = ['#c9a0e0', '#b07fd4', '#9966bb', '#7c4fa0', '#4d2475']
BASELINE_PALETTE = ['#cccccc', '#888888', '#444444']

MODALITY_COLORS = {
    'Sequence':  '#1a4a8a',
    'Structure': '#D2691E',
    'Multimodal':'#7c4fa0',
    'Baseline':  '#666666',
}

SUBTYPE_COLORS = {
    'Sequence':     '#1a4a8a',   # navy blue
    'Struct: Token':'#FFA500',   # orange
    'Struct: Graph':'#D2691E',   # chocolate
    'Struct: Equiv':'#A0522D',   # sienna
    'Multimodal':   '#7c4fa0',
    'Baseline':     '#666666',
}

MODALITY_PALETTES = {
    'Sequence':  SEQ_PALETTE,
    'Structure': STRUCT_PALETTE,
    'Multimodal': MULTI_PALETTE,
    'Baseline':  BASELINE_PALETTE,
}

# ── get encoders by modality ───────────────────────────────────────────────
seq_encs   = [e for e in encoders if ENCODER_MODALITY.get(e) == 'Sequence']
struct_encs = [e for e in encoders if ENCODER_MODALITY.get(e) == 'Structure']
multi_encs  = [e for e in encoders if ENCODER_MODALITY.get(e) == 'Multimodal']
base_encs   = [e for e in encoders if ENCODER_MODALITY.get(e) == 'Baseline']

# Structure subtypes
struct_token_encs = [e for e in struct_encs if ENCODER_SUBTYPE.get(e) == 'Struct: Token']
struct_graph_encs = [e for e in struct_encs if ENCODER_SUBTYPE.get(e) == 'Struct: Graph']
struct_equiv_encs = [e for e in struct_encs if ENCODER_SUBTYPE.get(e) == 'Struct: Equiv']

# ── create combined legend figure ──────────────────────────────────────────
# print("\nGenerating standalone legend figure…")

# # Legend width: 2.25", height adjusted to fit new Baseline elements
# fig, ax = plt.subplots(figsize=(2.25, 4.0))
# ax.axis('off')

# y_pos = 0.98
# line_spacing = 0.024

# # Sequence section header
# ax.text(0.1, y_pos, 'Sequence', fontsize=7, fontweight='bold',
#         color=MODALITY_COLORS['Sequence'], transform=ax.transAxes)
# y_pos -= 0.04

# # Sequence encoders
# for i, enc in enumerate(seq_encs):
#     ax.plot([0.05, 0.15], [y_pos, y_pos], color=seq_palette[i], linewidth=2,
#             transform=ax.transAxes, solid_capstyle='butt')
#     ax.text(0.18, y_pos, disp(enc), fontsize=5.5, va='center',
#             transform=ax.transAxes)
#     y_pos -= line_spacing

# y_pos -= 0.02

# # Structure section header
# ax.text(0.1, y_pos, 'Structure', fontsize=7, fontweight='bold',
#         color=MODALITY_COLORS['Structure'], transform=ax.transAxes)
# y_pos -= 0.04

# # Structure encoders
# for i, enc in enumerate(struct_encs):
#     ax.plot([0.05, 0.15], [y_pos, y_pos], color=struct_palette[i], linewidth=2,
#             transform=ax.transAxes, solid_capstyle='butt')
#     ax.text(0.18, y_pos, disp(enc), fontsize=5.5, va='center',
#             transform=ax.transAxes)
#     y_pos -= line_spacing

# y_pos -= 0.02

# # Multimodal section header
# ax.text(0.1, y_pos, 'Multimodal', fontsize=7, fontweight='bold',
#         color=MODALITY_COLORS['Multimodal'], transform=ax.transAxes)
# y_pos -= 0.04

# # Multimodal encoders
# for i, enc in enumerate(multi_encs):
#     ax.plot([0.05, 0.15], [y_pos, y_pos], color=multi_palette[i], linewidth=2,
#             transform=ax.transAxes, solid_capstyle='butt')
#     ax.text(0.18, y_pos, disp(enc), fontsize=5.5, va='center',
#             transform=ax.transAxes)
#     y_pos -= line_spacing

# y_pos -= 0.02

# # Baseline section header
# ax.text(0.1, y_pos, 'Baseline', fontsize=7, fontweight='bold',
#         color=MODALITY_COLORS['Baseline'], transform=ax.transAxes)
# y_pos -= 0.04

# # Baseline encoders
# for i, enc in enumerate(base_encs):
#     ax.plot([0.05, 0.15], [y_pos, y_pos], color=baseline_palette[i], linewidth=2,
#             transform=ax.transAxes, solid_capstyle='butt')
#     ax.text(0.18, y_pos, disp(enc), fontsize=5.5, va='center',
#             transform=ax.transAxes)
#     y_pos -= line_spacing

# plt.tight_layout()
# for ext in ('png', 'pdf'):
#     plt.savefig(OUT_DIR / f'fig_legend_all.{ext}', dpi=300, bbox_inches='tight')
# print(f"  Saved standalone legend to {OUT_DIR}/fig_legend_all.png/pdf")
# plt.close()

# print("\nLegend generation complete!")

def save_modality_legend(
    title,
    encoders,
    palette,
    title_color,
    filename_stem,
    width=2.25,
    row_step=0.40,
    header_gap=0.6,
):
    """
    Save one standalone legend per modality with adjustable spacing.
    """

    n = len(encoders)

    # Height scales with spacing
    height = max(0.6, 0.22 + row_step * n * 0.28)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")

    y_top = header_gap + row_step * n

    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_top + 0.15)

    # Header
    ax.text(
        0.10,
        y_top,
        title,
        fontsize=7,
        fontweight="bold",
        color=title_color,
        va="center",
    )

    # Encoder entries
    for i, enc in enumerate(encoders):
        y = y_top - header_gap - i * row_step

        ax.plot(
            [0.05, 0.15],
            [y, y],
            color=palette[i],
            linewidth=2,
            solid_capstyle="butt",
        )

        ax.text(
            0.18,
            y,
            disp(enc),
            fontsize=5.5,
            va="center",
        )

    for ext in ("png", "pdf"):
        plt.savefig(
            OUT_DIR / f"{filename_stem}.{ext}",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.02,
        )

    print(f"  Saved {title} legend to {OUT_DIR}/{filename_stem}.png/pdf")
    plt.close()


save_modality_legend(
    title="Sequence",
    encoders=seq_encs,
    palette=SEQ_PALETTE,
    title_color=MODALITY_COLORS["Sequence"],
    filename_stem="fig_legend_sequence",
)

# Structure legend with subtypes - assign colors by subtype
struct_colors = []
for enc in struct_encs:
    stype = ENCODER_SUBTYPE.get(enc, 'Structure')
    struct_colors.append(SUBTYPE_COLORS.get(stype, MODALITY_COLORS['Structure']))

save_modality_legend(
    title="Structure",
    encoders=struct_encs,
    palette=struct_colors,
    title_color=MODALITY_COLORS["Structure"],
    filename_stem="fig_legend_structure",
)

save_modality_legend(
    title="Multimodal",
    encoders=multi_encs,
    palette=MULTI_PALETTE,
    title_color=MODALITY_COLORS["Multimodal"],
    filename_stem="fig_legend_multimodal",
)

save_modality_legend(
    title="Baseline",
    encoders=base_encs,
    palette=BASELINE_PALETTE,
    title_color=MODALITY_COLORS["Baseline"],
    filename_stem="fig_legend_baseline",
)

print("\nSeparate legend generation complete!")