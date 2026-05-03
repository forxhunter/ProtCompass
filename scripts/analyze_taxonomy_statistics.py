#!/usr/bin/env python3
"""
Generate summary statistics and key findings for taxonomy-based probing analysis.
Creates tables and statistics for paper writing.
"""
import sys
import types
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Mock modules
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

sys.modules['torch_geometric'] = MockModule()
sys.modules['torch_geometric.nn'] = MockModule()

class InvariantFingerprint:
    pass

src_module = types.ModuleType('src')
probes_module = types.ModuleType('probes')
battery_module = types.ModuleType('probe_battery')
battery_module.InvariantFingerprint = InvariantFingerprint
probes_module.probe_battery = battery_module
src_module.probes = probes_module
sys.modules['src'] = src_module
sys.modules['src.probes'] = probes_module
sys.modules['src.probes.probe_battery'] = battery_module

import torch

# Configuration
results_dir = Path("outputs/probing_results")
output_dir = Path("outputs/paper_figures_v8")

tasks = [f"I{i}" for i in range(1, 11)]
task_names = {
    "I1": "Secondary Structure",
    "I2": "Solvent Accessibility",
    "I3": "Contact Prediction",
    "I4": "Backbone Angles",
    "I5": "Global Geometry",
    "I6": "Sequence Recovery",
    "I7": "Fold Classification",
    "I8": "Local Curvature",
    "I9": "B-factor",
    "I10": "Core/Surface"
}

encoder_modality = {
    'esm2': 'Sequence', 'esm2_150m': 'Sequence', 'esm2_35m': 'Sequence',
    'esm1b': 'Sequence', 'prottrans': 'Sequence', 'prostt5': 'Sequence',
    'proteinbert_bfd': 'Sequence', 'ankh': 'Sequence',
    'foldseek': 'Structure', 'proteinmpnn_pretrained': 'Structure',
    'pottsmpnn': 'Structure', 'pottsmpnn_pretrained': 'Structure',
    'pottsmpnn_soluble_pretrained': 'Structure', 'pottsmpnn_ft_pretrained': 'Structure',
    'gearnet': 'Structure', 'gearnet_pretrained': 'Structure',
    'schnet': 'Structure', 'dimenet': 'Structure', 'mace': 'Structure',
    'cdconv': 'Structure', 'cdconv_pretrained': 'Structure',
    'dmasif_pretrained': 'Structure',
    'egnn': 'Structure', 'egnn_pretrained': 'Structure',
    'gvp': 'Structure', 'gvp_pretrained': 'Structure',
    'tfn': 'Structure', 'tfn_pretrained': 'Structure',
    'ipa_pretrained': 'Structure',
    'gcpnet': 'Structure', 'gcpnet_pretrained': 'Structure',
    'saprot': 'Multimodal', 'saprot_pretrained': 'Multimodal',
    'esm_if': 'Multimodal', 'esm3': 'Multimodal',
    'foldvision': 'Multimodal',
    'random': 'Baseline', 'length': 'Baseline', 'torsion': 'Baseline'
}

print("Loading probing results...")

# Load all data
all_data = {}
for result_file in sorted(results_dir.glob('*.pt')):
    try:
        filename = result_file.stem
        encoder_name = filename.rsplit('_seed_', 1)[0]
        if encoder_name not in all_data:
            all_data[encoder_name] = {}
        data = torch.load(result_file, map_location='cpu', weights_only=False)
        fingerprint = data['fingerprint']
        seed = int(filename.rsplit('_seed_', 1)[1])
        all_data[encoder_name][seed] = {
            f"I{i}": float(getattr(fingerprint, f"p_I{i}"))
            for i in range(1, 11)
        }
    except:
        continue

encoders = sorted(all_data.keys())
print(f"Loaded {len(encoders)} encoders")

# Compute statistics
results_matrix = np.zeros((len(encoders), len(tasks)))
for i, encoder in enumerate(encoders):
    for j, task in enumerate(tasks):
        accs = [all_data[encoder][seed][task] for seed in all_data[encoder].keys()]
        if accs:
            results_matrix[i, j] = np.mean(accs)

print("\n" + "="*80)
print("TAXONOMY-BASED PROBING ANALYSIS SUMMARY")
print("="*80)

# 1. Overall modality performance
print("\n1. OVERALL PERFORMANCE BY MODALITY")
print("-" * 80)

modalities = ['Sequence', 'Structure', 'Multimodal']
modality_stats = []

for mod in modalities:
    mod_encoders = [enc for enc in encoders if encoder_modality.get(enc) == mod]
    if mod_encoders:
        mod_scores = [results_matrix[encoders.index(enc), :].mean() for enc in mod_encoders]
        modality_stats.append({
            'Modality': mod,
            'N': len(mod_encoders),
            'Mean': np.mean(mod_scores),
            'Std': np.std(mod_scores),
            'Min': np.min(mod_scores),
            'Max': np.max(mod_scores)
        })

df_modality = pd.DataFrame(modality_stats)
print(df_modality.to_string(index=False))

# 2. Task-specific modality dominance
print("\n\n2. BEST MODALITY PER TASK")
print("-" * 80)

task_winners = []
for j, task in enumerate(tasks):
    mod_means = {}
    for mod in modalities:
        mod_encoders = [enc for enc in encoders if encoder_modality.get(enc) == mod]
        if mod_encoders:
            scores = [results_matrix[encoders.index(enc), j] for enc in mod_encoders]
            mod_means[mod] = np.mean(scores)

    best_mod = max(mod_means, key=mod_means.get)
    task_winners.append({
        'Task': task,
        'Name': task_names[task],
        'Winner': best_mod,
        'Score': mod_means[best_mod],
        'Gap': mod_means[best_mod] - min(mod_means.values())
    })

df_winners = pd.DataFrame(task_winners)
print(df_winners.to_string(index=False))

# 3. Best encoder per modality per task
print("\n\n3. BEST ENCODER PER MODALITY (REPRESENTATIVES)")
print("-" * 80)

for mod in modalities:
    print(f"\n{mod}:")
    mod_encoders = [enc for enc in encoders if encoder_modality.get(enc) == mod]
    if not mod_encoders:
        continue

    # Overall best
    overall_scores = [results_matrix[encoders.index(enc), :].mean() for enc in mod_encoders]
    best_idx = np.argmax(overall_scores)
    print(f"  Overall best: {mod_encoders[best_idx]} (mean={overall_scores[best_idx]:.3f})")

    # Best per task
    task_bests = []
    for j, task in enumerate(tasks):
        scores = [results_matrix[encoders.index(enc), j] for enc in mod_encoders]
        best_idx = np.argmax(scores)
        task_bests.append(f"{task}:{mod_encoders[best_idx]}")
    print(f"  Task specialists: {', '.join(task_bests[:5])}...")

# 4. Within-modality variance
print("\n\n4. WITHIN-MODALITY VARIANCE")
print("-" * 80)

variance_stats = []
for mod in modalities:
    mod_encoders = [enc for enc in encoders if encoder_modality.get(enc) == mod]
    if len(mod_encoders) > 1:
        for j, task in enumerate(tasks):
            scores = [results_matrix[encoders.index(enc), j] for enc in mod_encoders]
            variance_stats.append({
                'Modality': mod,
                'Task': task,
                'Mean': np.mean(scores),
                'Std': np.std(scores),
                'Range': np.max(scores) - np.min(scores)
            })

df_variance = pd.DataFrame(variance_stats)

# Show tasks with highest variance per modality
print("\nTasks with highest within-modality variance:")
for mod in modalities:
    mod_data = df_variance[df_variance['Modality'] == mod].nlargest(3, 'Std')
    print(f"\n{mod}:")
    for _, row in mod_data.iterrows():
        print(f"  {row['Task']} ({task_names[row['Task']]}): std={row['Std']:.3f}, range={row['Range']:.3f}")

# 5. Statistical significance tests
print("\n\n5. STATISTICAL SIGNIFICANCE (Sequence vs Structure)")
print("-" * 80)

seq_encoders = [enc for enc in encoders if encoder_modality.get(enc) == 'Sequence']
struct_encoders = [enc for enc in encoders if encoder_modality.get(enc) == 'Structure']

sig_tests = []
for j, task in enumerate(tasks):
    seq_scores = [results_matrix[encoders.index(enc), j] for enc in seq_encoders]
    struct_scores = [results_matrix[encoders.index(enc), j] for enc in struct_encoders]

    t_stat, p_val = stats.ttest_ind(seq_scores, struct_scores)

    sig_tests.append({
        'Task': task,
        'Name': task_names[task],
        'Seq_mean': np.mean(seq_scores),
        'Struct_mean': np.mean(struct_scores),
        'Difference': np.mean(seq_scores) - np.mean(struct_scores),
        'p_value': p_val,
        'Significant': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    })

df_sig = pd.DataFrame(sig_tests)
print(df_sig.to_string(index=False))

# 6. Key findings summary
print("\n\n6. KEY FINDINGS FOR PAPER")
print("="*80)

# Find tasks where sequence dominates
seq_dominant = df_winners[df_winners['Winner'] == 'Sequence']['Task'].tolist()
struct_dominant = df_winners[df_winners['Winner'] == 'Structure']['Task'].tolist()
multi_dominant = df_winners[df_winners['Winner'] == 'Multimodal']['Task'].tolist()

print(f"\n✓ Sequence-only encoders dominate {len(seq_dominant)}/10 tasks: {', '.join(seq_dominant)}")
print(f"✓ Structure-only encoders dominate {len(struct_dominant)}/10 tasks: {', '.join(struct_dominant)}")
print(f"✓ Multimodal encoders dominate {len(multi_dominant)}/10 tasks: {', '.join(multi_dominant)}")

# Largest gaps
largest_gap = df_winners.nlargest(3, 'Gap')
print(f"\n✓ Largest modality gaps:")
for _, row in largest_gap.iterrows():
    print(f"  {row['Task']} ({row['Name']}): {row['Winner']} leads by {row['Gap']:.3f}")

# Most significant differences
most_sig = df_sig.nsmallest(3, 'p_value')
print(f"\n✓ Most significant Sequence vs Structure differences:")
for _, row in most_sig.iterrows():
    winner = 'Sequence' if row['Difference'] > 0 else 'Structure'
    print(f"  {row['Task']} ({row['Name']}): {winner} (p={row['p_value']:.2e})")

# Save summary tables
df_modality.to_csv(output_dir / 'table_modality_overall.csv', index=False)
df_winners.to_csv(output_dir / 'table_task_winners.csv', index=False)
df_sig.to_csv(output_dir / 'table_significance_tests.csv', index=False)

print(f"\n\n✓ Summary tables saved to {output_dir}/")
print("  - table_modality_overall.csv")
print("  - table_task_winners.csv")
print("  - table_significance_tests.csv")
