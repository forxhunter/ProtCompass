#!/usr/bin/env python3
"""
Visualize probing results for ALL 39 encoders across 10 invariant families.
Uses the same style as existing probing visualizations.
Custom color scheme: #ece5df, #b7cad2, #ae9fc2, #8d86b6
"""
import sys
import types
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# Custom color palette (consistent with previous probing visualizations)
COLORS = ['#ece5df', '#b7cad2', '#ae9fc2', '#8d86b6']

# Mock problematic modules
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
output_dir = Path("outputs/probing_visualizations")
output_dir.mkdir(exist_ok=True, parents=True)

tasks = [f"I{i}" for i in range(1, 11)]
task_labels = {
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

# Encoder categories for coloring
encoder_categories = {
    # PLM
    'prottrans': 'PLM', 'esm3': 'PLM', 'esm2': 'PLM', 'esm2_150m': 'PLM', 'esm2_35m': 'PLM',
    'esm1b': 'PLM', 'prostt5': 'PLM', 'proteinbert_bfd': 'PLM', 'ankh': 'PLM',
    # Structure
    'gearnet': 'Structure', 'gearnet_pretrained': 'Structure', 'gcpnet': 'Structure',
    'gcpnet_pretrained': 'Structure', 'gvp': 'Structure', 'gvp_pretrained': 'Structure',
    'egnn': 'Structure', 'egnn_pretrained': 'Structure', 'dimenet': 'Structure',
    'schnet': 'Structure', 'mace': 'Structure', 'tfn': 'Structure', 'tfn_pretrained': 'Structure',
    'cdconv': 'Structure', 'cdconv_pretrained': 'Structure',
    # Hybrid
    'saprot': 'Hybrid', 'saprot_pretrained': 'Hybrid', 'esm_if': 'Hybrid', 'foldseek': 'Hybrid',
    'foldvision': 'Hybrid',
    # Design
    'proteinmpnn_pretrained': 'Design', 'pottsmpnn': 'Design', 'pottsmpnn_pretrained': 'Design',
    'pottsmpnn_soluble_pretrained': 'Design', 'pottsmpnn_ft_pretrained': 'Design',
    'ipa_pretrained': 'Design', 'dmasif_pretrained': 'Design',
    # Baseline
    'random': 'Baseline', 'length': 'Baseline', 'torsion': 'Baseline'
}

category_colors = {
    'PLM': '#e74c3c',
    'Structure': '#3498db',
    'Hybrid': '#9b59b6',
    'Design': '#f39c12',
    'Baseline': '#95a5a6'
}

print("Loading probing results for all 39 encoders...")

# Collect all data
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
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        continue

encoders = sorted(all_data.keys())
print(f"Loaded {len(encoders)} encoders")

# Compute mean and std for each encoder-task pair
results_matrix = np.zeros((len(encoders), len(tasks)))
std_matrix = np.zeros((len(encoders), len(tasks)))

for i, encoder in enumerate(encoders):
    for j, task in enumerate(tasks):
        accs = [all_data[encoder][seed][task] for seed in all_data[encoder].keys()]
        if accs:
            results_matrix[i, j] = np.mean(accs)
            std_matrix[i, j] = np.std(accs)

print("Creating visualizations...")

# 1. Heatmap of mean scores (39 encoders × 10 tasks)
plt.figure(figsize=(14, 20))
cmap = LinearSegmentedColormap.from_list('custom', COLORS, N=256)
sns.heatmap(results_matrix,
            xticklabels=[task_labels[t] for t in tasks],
            yticklabels=encoders,
            annot=True, fmt='.3f', cmap=cmap, center=results_matrix.mean(),
            cbar_kws={'label': 'Probe Accuracy'}, linewidths=0.5, linecolor='white',
            vmin=0, vmax=1.0)
plt.title('Probing Results: Mean Accuracy Across Seeds\n39 Encoders × 10 Invariant Families',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Invariant Family', fontsize=13, fontweight='bold')
plt.ylabel('Encoder', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(output_dir / 'all_encoders_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'all_encoders_heatmap.png'}")
plt.close()

# 2. Best encoder per task (bar chart)
best_encoders = []
best_scores = []
for j, task in enumerate(tasks):
    best_idx = np.argmax(results_matrix[:, j])
    best_encoders.append(encoders[best_idx])
    best_scores.append(results_matrix[best_idx, j])

fig, ax = plt.subplots(figsize=(14, 7))

bar_colors = [category_colors[encoder_categories.get(enc, 'Baseline')] for enc in best_encoders]
bars = ax.bar(range(len(tasks)), best_scores, color=bar_colors, edgecolor='black', linewidth=1.0, alpha=0.8)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels([task_labels[t] for t in tasks], rotation=45, ha='right', fontsize=12)
ax.set_ylabel('Best Probe Accuracy', fontsize=13, fontweight='bold')
ax.set_xlabel('Invariant Family', fontsize=13, fontweight='bold')
ax.set_title('Best Encoder Performance per Task\n39 Encoders × 10 Invariant Families',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, encoder, score) in enumerate(zip(bars, best_encoders, best_scores)):
    ax.text(i, score + 0.02, encoder, ha='center', va='bottom', fontsize=9, rotation=90)

# Add legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat)
                   for cat, color in category_colors.items()]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, title='Category')

plt.tight_layout()
plt.savefig(output_dir / 'all_encoders_best_per_task.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'all_encoders_best_per_task.png'}")
plt.close()

# 3. Overall encoder ranking (top 20)
overall_scores = results_matrix.mean(axis=1)
overall_ranks = np.argsort(overall_scores)[::-1]  # Descending order
top_n = 20

fig, ax = plt.subplots(figsize=(12, 10))
top_encoders = [encoders[i] for i in overall_ranks[:top_n]]
top_scores = [overall_scores[i] for i in overall_ranks[:top_n]]
top_colors = [category_colors[encoder_categories.get(enc, 'Baseline')] for enc in top_encoders]

bars = ax.barh(range(top_n), top_scores, color=top_colors, edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_encoders, fontsize=11)
ax.set_xlabel('Mean Probe Accuracy (averaged across 10 tasks)', fontsize=13, fontweight='bold')
ax.set_title(f'Top {top_n} Encoder Rankings\n39 Encoders × 10 Invariant Families',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()

for i, score in enumerate(top_scores):
    ax.text(score + 0.005, i, f'{score:.3f}', va='center', fontsize=10)

# Add legend
legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat)
                   for cat, color in category_colors.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, title='Category')

plt.tight_layout()
plt.savefig(output_dir / 'all_encoders_top20_ranking.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'all_encoders_top20_ranking.png'}")
plt.close()

# 4. Category comparison (boxplots per task)
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()

for j, task in enumerate(tasks):
    ax = axes[j]

    # Group by category
    category_data = {cat: [] for cat in category_colors.keys()}

    for i, encoder in enumerate(encoders):
        cat = encoder_categories.get(encoder, 'Baseline')
        category_data[cat].append(results_matrix[i, j])

    # Plot boxplots
    data_for_plot = [category_data[cat] for cat in category_colors.keys() if category_data[cat]]
    labels_for_plot = [cat for cat in category_colors.keys() if category_data[cat]]
    colors_for_plot = [category_colors[cat] for cat in labels_for_plot]

    bp = ax.boxplot(data_for_plot, tick_labels=labels_for_plot, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_for_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

    ax.set_title(task_labels[task], fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy' if j % 5 == 0 else '', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.set_ylim(0, 1.0)

plt.suptitle('Category Comparison Across Invariant Families\n39 Encoders × 10 Tasks',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'all_encoders_category_boxplots.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'all_encoders_category_boxplots.png'}")
plt.close()

# 5. Task difficulty analysis
task_means = results_matrix.mean(axis=0)
task_stds = results_matrix.std(axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Task difficulty (mean)
colors_difficulty = [COLORS[3] if task == 'I7' else COLORS[1] for task in tasks]
bars1 = ax1.bar(range(len(tasks)), task_means, color=colors_difficulty,
                edgecolor='black', linewidth=0.5, alpha=0.8)
ax1.set_xticks(range(len(tasks)))
ax1.set_xticklabels([task_labels[t] for t in tasks], rotation=45, ha='right', fontsize=11)
ax1.set_ylabel('Mean Accuracy (across 39 encoders)', fontsize=12, fontweight='bold')
ax1.set_title('Task Difficulty\n(Higher = Easier)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 1.0)

for i, (bar, mean) in enumerate(zip(bars1, task_means)):
    ax1.text(i, mean + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# Task discriminability (std)
bars2 = ax2.bar(range(len(tasks)), task_stds, color=COLORS[2],
                edgecolor='black', linewidth=0.5, alpha=0.8)
ax2.set_xticks(range(len(tasks)))
ax2.set_xticklabels([task_labels[t] for t in tasks], rotation=45, ha='right', fontsize=11)
ax2.set_ylabel('Std Dev (across 39 encoders)', fontsize=12, fontweight='bold')
ax2.set_title('Task Discriminability\n(Higher = Better at separating encoders)',
              fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, std) in enumerate(zip(bars2, task_stds)):
    ax2.text(i, std + 0.005, f'{std:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Task Analysis: Difficulty and Discriminability\n39 Encoders × 10 Invariant Families',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'all_encoders_task_analysis.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'all_encoders_task_analysis.png'}")
plt.close()

print("\n✓ All visualizations complete!")
print(f"\nGenerated files in {output_dir}:")
print(f"  1. all_encoders_heatmap.png - Full 39×10 heatmap")
print(f"  2. all_encoders_best_per_task.png - Best encoder per task")
print(f"  3. all_encoders_top20_ranking.png - Top 20 overall rankings")
print(f"  4. all_encoders_category_boxplots.png - Category comparison")
print(f"  5. all_encoders_task_analysis.png - Task difficulty & discriminability")
