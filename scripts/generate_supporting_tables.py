#!/usr/bin/env python3
"""
Generate LaTeX tables with detailed numerical data for the three main results sections.
Creates tables for: (1) IIP probing results, (2) Downstream task results, (3) IAH alignment data
"""

import sys, types, warnings
import numpy as np
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
ROOT     = Path("/home/tianyu16/esm3_separate_machine")
ARCH_DIR = ROOT / "probing_results_architecture_full"
FALL_DIR = ROOT / "structure_encoder_data" / "probing_results"
DOWN_CSV = ROOT / "outputs" / "downstream_v1" / "downstream_results.csv"
OUT_DIR  = ROOT / "papers" / "appendices"

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

TASK_NAMES = {
    'I1':  'Local Conformation', 'I2':  'Local Curvature',  'I3':  'Secondary Structure',
    'I4':  'Solvent Accessibility', 'I5':  'Local Flexibility','I6':  'Packing Density',
    'I7':  'Residue Contacts',   'I8':  'Global Geometry',  'I9':  'Fold Architecture',
    'I10': 'Sequence Recovery',
}

DTASK_NAME_TO_NUM = {
    "contact_prediction": "T1", "remote_homology": "T2", "fluorescence": "T3",
    "stability": "T4", "mutation_effect": "T5", "ec_classification": "T6",
    "go_mf": "T7", "go_bp": "T8", "solubility": "T9", "ppi_site": "T10",
    "metal_binding": "T11", "ppi_affinity": "T12",
    "subcellular_localization": "T13", "membrane_soluble": "T14",
}

DTASK_NAMES = {
    "T1": "Contact Prediction", "T2": "Remote Homology", "T3": "Fluorescence",
    "T4": "Stability", "T5": "Mutational Effect", "T6": "EC Classification",
    "T7": "GO Molecular Function", "T8": "GO Biological Process", "T9": "Solubility",
    "T10": "PPI Sites", "T11": "Metal Binding", "T12": "PPI Affinity",
    "T13": "Subcellular Localization", "T14": "Membrane/Soluble",
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
    'esm3': 'ESM-3', 'foldvision': 'FoldVision',
    'cdconv': 'CDConv', 'cdconv_pretrained': 'CDConv',
    'geobpe': 'GeoBPE',
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

def _aggregate_families(detailed):
    out = {}
    for fam, probes in FAMILY_PROBES.items():
        vals = [float(detailed[p]) for p in probes if p in detailed]
        out[fam] = float(np.mean(vals)) if vals else 0.0
    return out

def is_shown(enc, all_encoders):
    """Show only pretrained version if both exist."""
    if '_pretrained' in enc:
        return True
    pretrained_version = f"{enc}_pretrained"
    return pretrained_version not in all_encoders

# ── Load IIP probing data ──────────────────────────────────────────────────
print("Loading IIP probing results…")
iip_data = {}
for src in [ARCH_DIR, FALL_DIR]:
    for f in sorted(src.glob('*.pt')):
        try:
            enc = f.stem.rsplit('_seed_', 1)[0]
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

all_encoders = sorted(iip_mean.keys())
encoders = [e for e in all_encoders if is_shown(e, all_encoders)]
print(f"  Loaded {len(encoders)} encoders (after pretrained-only filter)")

# ── Load downstream data ───────────────────────────────────────────────────
print("Loading downstream results…")
df = pd.read_csv(DOWN_CSV)
df['T'] = df['task'].map(DTASK_NAME_TO_NUM)
df = df.dropna(subset=['T'])

# ── Generate Table 1: IIP Probing Results ─────────────────────────────────
print("\nGenerating Table 1: IIP Probing Results…")

iip_tasks = [f"I{i}" for i in range(1, 11)]
n_enc = len(encoders)
n_task = len(iip_tasks)

# Build matrix
iip_matrix = np.zeros((n_enc, n_task))
for i, enc in enumerate(encoders):
    for j, task in enumerate(iip_tasks):
        iip_matrix[i, j] = iip_mean[enc].get(task, 0.0)

# Compute mean per encoder
iip_encoder_mean = iip_matrix.mean(axis=1)

# Sort by modality then by mean score descending
def sort_key(idx):
    enc = encoders[idx]
    mod = ENCODER_MODALITY.get(enc, 'Structure')
    mod_order = {'Sequence': 0, 'Structure': 1, 'Multimodal': 2}
    return (mod_order.get(mod, 3), -iip_encoder_mean[idx])

sorted_indices = sorted(range(n_enc), key=sort_key)

# Generate LaTeX table
latex_lines = []
latex_lines.append(r"\begin{table}[h]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{IIP probing accuracy for all encoders. Values are mean accuracy across 5 random seeds. Encoders are grouped by modality and sorted by mean performance within each group.}")
latex_lines.append(r"\label{tab:iip_probing_full}")
latex_lines.append(r"")
latex_lines.append(r"\footnotesize")
latex_lines.append(r"\setlength{\tabcolsep}{3pt}")
latex_lines.append(r"\renewcommand{\arraystretch}{1.05}")
latex_lines.append(r"")

# Header
header = r"\begin{tabular}{l" + "c" * (n_task + 1) + "}"
latex_lines.append(header)
latex_lines.append(r"\toprule")
col_names = r"\textbf{Encoder} & " + " & ".join([r"\textbf{" + task + "}" for task in iip_tasks]) + r" & \textbf{Mean} \\"
latex_lines.append(col_names)
latex_lines.append(r"\midrule")

# Data rows
current_modality = None
for idx in sorted_indices:
    enc = encoders[idx]
    mod = ENCODER_MODALITY.get(enc, 'Structure')

    # Add modality header
    if mod != current_modality:
        if current_modality is not None:
            latex_lines.append(r"\midrule")
        latex_lines.append(r"\rowcolor{gray!10}")
        latex_lines.append(r"\multicolumn{" + str(n_task + 2) + r"}{l}{\textbf{" + mod + r" Encoders}} \\")
        current_modality = mod

    # Encoder row
    disp_name = DISPLAY_NAMES.get(enc, enc)
    row_vals = [f"{iip_matrix[idx, j]:.3f}" for j in range(n_task)]
    mean_val = f"{iip_encoder_mean[idx]:.3f}"
    row = disp_name + " & " + " & ".join(row_vals) + " & " + mean_val + r" \\"
    latex_lines.append(row)

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

table1_content = "\n".join(latex_lines)

# ── Generate Table 2: Downstream Task Results ─────────────────────────────
print("Generating Table 2: Downstream Task Results…")

down_tasks = [f"T{i}" for i in range(1, 15)]
d_encoders = [e for e in encoders if e in df['encoder'].unique()]
n_d = len(d_encoders)
n_dt = len(down_tasks)

# Build matrix
down_matrix = np.full((n_d, n_dt), np.nan)
for i, enc in enumerate(d_encoders):
    sub = df[df['encoder'] == enc]
    for j, task in enumerate(down_tasks):
        row = sub[sub['T'] == task]
        if len(row):
            down_matrix[i, j] = row['mean_score'].values[0]

# Compute mean per encoder (ignoring NaN)
down_encoder_mean = np.nanmean(down_matrix, axis=1)

# Sort by modality then by mean score descending
def sort_key_down(idx):
    enc = d_encoders[idx]
    mod = ENCODER_MODALITY.get(enc, 'Structure')
    mod_order = {'Sequence': 0, 'Structure': 1, 'Multimodal': 2}
    return (mod_order.get(mod, 3), -down_encoder_mean[idx])

sorted_indices_down = sorted(range(n_d), key=sort_key_down)

# Generate LaTeX table
latex_lines = []
latex_lines.append(r"\begin{table}[h]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Downstream task performance for all encoders. Values are mean scores across 5 random seeds. Encoders are grouped by modality and sorted by mean performance within each group. Empty cells indicate tasks where the encoder was not evaluated.}")
latex_lines.append(r"\label{tab:downstream_full}")
latex_lines.append(r"")
latex_lines.append(r"\footnotesize")
latex_lines.append(r"\setlength{\tabcolsep}{2.5pt}")
latex_lines.append(r"\renewcommand{\arraystretch}{1.05}")
latex_lines.append(r"")

# Header
header = r"\begin{tabular}{l" + "c" * (n_dt + 1) + "}"
latex_lines.append(header)
latex_lines.append(r"\toprule")
col_names = r"\textbf{Encoder} & " + " & ".join([r"\textbf{" + task + "}" for task in down_tasks]) + r" & \textbf{Mean} \\"
latex_lines.append(col_names)
latex_lines.append(r"\midrule")

# Data rows
current_modality = None
for idx in sorted_indices_down:
    enc = d_encoders[idx]
    mod = ENCODER_MODALITY.get(enc, 'Structure')

    # Add modality header
    if mod != current_modality:
        if current_modality is not None:
            latex_lines.append(r"\midrule")
        latex_lines.append(r"\rowcolor{gray!10}")
        latex_lines.append(r"\multicolumn{" + str(n_dt + 2) + r"}{l}{\textbf{" + mod + r" Encoders}} \\")
        current_modality = mod

    # Encoder row
    disp_name = DISPLAY_NAMES.get(enc, enc)
    row_vals = []
    for j in range(n_dt):
        val = down_matrix[idx, j]
        if np.isnan(val):
            row_vals.append("--")
        else:
            row_vals.append(f"{val:.3f}")
    mean_val = f"{down_encoder_mean[idx]:.3f}"
    row = disp_name + " & " + " & ".join(row_vals) + " & " + mean_val + r" \\"
    latex_lines.append(row)

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

table2_content = "\n".join(latex_lines)

# ── Generate Table 3: Task Profile Weights (IAH) ──────────────────────────
print("Generating Table 3: Task Profile Weights…")

# Load alignment analysis results
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Match encoders between probing and downstream
align_encoders = sorted([e for e in d_encoders if e in iip_mean])
n_a = len(align_encoders)

# Build probing matrix Z
Z = np.zeros((n_a, n_task))
for i, enc in enumerate(align_encoders):
    for j, task in enumerate(iip_tasks):
        Z[i, j] = iip_mean[enc].get(task, 0.0)

# Build downstream matrix y
y_mat = np.full((n_a, n_dt), np.nan)
for i, enc in enumerate(align_encoders):
    sub = df[df['encoder'] == enc]
    for j, task in enumerate(down_tasks):
        row = sub[sub['T'] == task]
        if len(row):
            y_mat[i, j] = row['mean_score'].values[0]

# Fit task profiles using Ridge regression
task_profiles = np.zeros((n_dt, n_task))
for j in range(n_dt):
    y = y_mat[:, j]
    valid = ~np.isnan(y)
    if valid.sum() < 5:
        continue

    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z[valid])

    # Select best alpha
    best_alpha, best_score = 1.0, -np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha).fit(Zs, y[valid])
        score = model.score(Zs, y[valid])
        if score > best_score:
            best_score, best_alpha = score, alpha

    model = Ridge(alpha=best_alpha).fit(Zs, y[valid])
    task_profiles[j] = model.coef_

# Generate LaTeX table
latex_lines = []
latex_lines.append(r"\begin{table}[h]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Task profile weights $\mathbf{w}_t$ estimated via Ridge regression. Each row shows the IIP weights for one downstream task. Positive weights indicate that higher IIP accuracy predicts better task performance; negative weights indicate inverse relationships.}")
latex_lines.append(r"\label{tab:task_profiles}")
latex_lines.append(r"")
latex_lines.append(r"\footnotesize")
latex_lines.append(r"\setlength{\tabcolsep}{3pt}")
latex_lines.append(r"\renewcommand{\arraystretch}{1.05}")
latex_lines.append(r"")

# Header
header = r"\begin{tabular}{l" + "c" * n_task + "}"
latex_lines.append(header)
latex_lines.append(r"\toprule")
col_names = r"\textbf{Task} & " + " & ".join([r"\textbf{" + task + "}" for task in iip_tasks]) + r" \\"
latex_lines.append(col_names)
latex_lines.append(r"\midrule")

# Data rows
for j, task in enumerate(down_tasks):
    task_name = DTASK_NAMES[task]
    row_vals = [f"{task_profiles[j, k]:.3f}" for k in range(n_task)]
    row = task_name + " & " + " & ".join(row_vals) + r" \\"
    latex_lines.append(row)

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

table3_content = "\n".join(latex_lines)

# ── Write all tables to appendix file ─────────────────────────────────────
print("\nWriting tables to appendix file…")

appendix_content = r"""\section{Supporting Data Tables}
\label{sec:appendix_supporting_tables}

This appendix provides detailed numerical data for the three main results sections: IIP probing (Section~\ref{sec:probing}), downstream task benchmarking (Section~\ref{sec:downstream}), and the Information Alignment Hypothesis (Section~\ref{sec:alignment}). These tables complement the visualizations in the main text and enable precise comparison of encoder performance.

\subsection{IIP Probing Results}

Table~\ref{tab:iip_probing_full} reports the complete probing accuracy matrix for all 26 encoders across the 10 Invariant Information Properties (IIPs). Each value is the mean accuracy across 5 random seeds. The rightmost column shows the mean accuracy across all IIPs for each encoder.

""" + table1_content + r"""

\clearpage

\subsection{Downstream Task Results}

Table~\ref{tab:downstream_full} reports the complete downstream performance matrix for all encoders across the 14 downstream tasks. Each value is the mean score across 5 random seeds. Empty cells (--) indicate tasks where the encoder was not evaluated, typically because structure-only encoders cannot be applied to sequence-only tasks. The rightmost column shows the mean score across all evaluated tasks for each encoder.

""" + table2_content + r"""

\clearpage

\subsection{Task Profile Weights}

Table~\ref{tab:task_profiles} reports the Ridge-estimated task profile weights $\mathbf{w}_t$ for each of the 14 downstream tasks. These weights quantify how much each IIP contributes to predicting performance on each downstream task under the Information Alignment Hypothesis framework (Section~\ref{sec:alignment}). Positive weights indicate that higher IIP accuracy predicts better task performance; negative weights indicate inverse relationships. These weights are used to compute the alignment scores $\mathrm{Align}(e,t)$ shown in Figure~\ref{fig:figure4-alignment}.

""" + table3_content + r"""

"""

output_file = OUT_DIR / "SupportingTables.tex"
with open(output_file, 'w') as f:
    f.write(appendix_content)

print(f"\nAll tables written to {output_file}")
print("Done!")
