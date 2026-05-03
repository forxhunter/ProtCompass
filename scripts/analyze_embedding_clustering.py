#!/usr/bin/env python3
"""
Analyze clustering quality in encoder embedding spaces and correlate with probing performance.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_embeddings(encoder_name, seed=0):
    """Load embeddings and labels."""
    emb_path = Path(f"outputs/encoder_embeddings/{encoder_name}_seed_{seed}.npz")
    data = np.load(emb_path, allow_pickle=True)

    embeddings = data['embeddings']
    labels = {}
    for key in data.keys():
        if key.startswith('labels_'):
            inv_name = key.replace('labels_', '')
            labels[inv_name] = data[key]

    return embeddings, labels

def load_probing_results(encoder_name, seed=0):
    """Load probing performance for this encoder."""
    probe_path = Path(f"outputs/probing_results_architecture_full/{encoder_name}_seed_{seed}.pt")
    if not probe_path.exists():
        return None

    data = torch.load(probe_path, weights_only=False)
    fingerprint = data['fingerprint']

    # Extract probing scores
    scores = {}
    for i in range(1, 11):
        key = f'p_I{i}'
        if hasattr(fingerprint, key):
            scores[f'I{i}'] = getattr(fingerprint, key)

    return scores

def compute_clustering_metrics(embeddings, labels, min_samples=5):
    """Compute clustering quality metrics for each invariant."""
    metrics = {}

    for inv_name, inv_labels in labels.items():
        # Skip if too few samples or invalid labels
        if len(inv_labels) < min_samples:
            continue

        # Convert to numeric if needed
        if inv_labels.dtype == object:
            # Skip if labels are complex objects (lists, arrays, etc.)
            try:
                # Try to convert first element to check if it's hashable
                _ = hash(inv_labels[0])
                unique_labels = list(set(inv_labels))
                label_map = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_map.get(l, -1) for l in inv_labels])
            except (TypeError, AttributeError):
                # Skip non-hashable types (lists, arrays, etc.)
                continue
        else:
            numeric_labels = inv_labels

        # Skip if only one class or too many classes (continuous variables)
        n_classes = len(np.unique(numeric_labels))
        if n_classes < 2 or n_classes >= len(numeric_labels):
            continue

        try:
            # Silhouette score (higher is better, range [-1, 1])
            sil = silhouette_score(embeddings, numeric_labels)

            # Davies-Bouldin index (lower is better, range [0, inf])
            db = davies_bouldin_score(embeddings, numeric_labels)

            # Calinski-Harabasz index (higher is better, range [0, inf])
            ch = calinski_harabasz_score(embeddings, numeric_labels)

            metrics[inv_name] = {
                'silhouette': sil,
                'davies_bouldin': db,
                'calinski_harabasz': ch,
                'n_classes': n_classes,
                'n_samples': len(numeric_labels)
            }
        except Exception as e:
            print(f"  Warning: Could not compute metrics for {inv_name}: {e}")
            continue

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoders', nargs='+', default=[
        'esm2', 'esm2_150m', 'proteinbert_bfd',
        'gearnet_pretrained', 'gcpnet_pretrained'
    ])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default='outputs/embedding_space_analysis')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect results
    all_results = []

    for encoder_name in args.encoders:
        print(f"\n{'='*60}")
        print(f"Analyzing: {encoder_name}")
        print(f"{'='*60}")

        # Load embeddings
        embeddings, labels = load_embeddings(encoder_name, args.seed)
        print(f"Embeddings shape: {embeddings.shape}")

        # Compute clustering metrics
        print("Computing clustering metrics...")
        clustering_metrics = compute_clustering_metrics(embeddings, labels)

        # Load probing results
        probing_scores = load_probing_results(encoder_name, args.seed)

        # Store results
        result = {
            'encoder': encoder_name,
            'embedding_dim': embeddings.shape[1],
            'n_proteins': embeddings.shape[0],
            'clustering': clustering_metrics,
            'probing': probing_scores
        }
        all_results.append(result)

        # Print summary
        print("\nClustering Quality (Silhouette Score):")
        for inv_name, metrics in sorted(clustering_metrics.items(),
                                       key=lambda x: x[1]['silhouette'],
                                       reverse=True):
            print(f"  {inv_name:25s}: {metrics['silhouette']:6.3f} "
                  f"(n_classes={metrics['n_classes']}, DB={metrics['davies_bouldin']:.2f})")

    # Create comparative visualizations
    print(f"\n{'='*60}")
    print("Creating comparative visualizations...")
    print(f"{'='*60}")

    # Map invariant names to probing task IDs
    inv_to_task = {
        'phi_bins': 'I1',
        'psi_bins': 'I2',
        'curvature': 'I3',
        'contact_density': 'I4',
        'core_surface_labels': 'I5',
        'contact_order': 'I6',
        'architecture_label': 'I7',
        'dssp_8': 'I8',
        'rsa_bins': 'I9',
        'distance_bins': 'I10'
    }

    # 1. Heatmap of clustering quality across encoders
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics_to_plot = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    metric_labels = ['Silhouette Score\n(higher better)',
                     'Davies-Bouldin Index\n(lower better)',
                     'Calinski-Harabasz Index\n(higher better)']

    for ax_idx, (metric_name, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        # Build matrix
        invariants = sorted(inv_to_task.keys())
        matrix = []

        for result in all_results:
            row = []
            for inv in invariants:
                if inv in result['clustering']:
                    row.append(result['clustering'][inv][metric_name])
                else:
                    row.append(np.nan)
            matrix.append(row)

        matrix = np.array(matrix)

        # Plot
        ax = axes[ax_idx]

        # Normalize for better visualization
        if metric_name == 'davies_bouldin':
            # Lower is better, so invert for color scale
            vmin, vmax = 0, np.nanmax(matrix)
            cmap = 'RdYlGn_r'
        else:
            vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
            cmap = 'RdYlGn'

        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        # Labels
        ax.set_xticks(range(len(invariants)))
        ax.set_xticklabels([inv_to_task.get(inv, inv) for inv in invariants], rotation=45, ha='right')
        ax.set_yticks(range(len(all_results)))
        ax.set_yticklabels([r['encoder'] for r in all_results])
        ax.set_title(metric_label, fontsize=12, pad=10)

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add values
        for i in range(len(all_results)):
            for j in range(len(invariants)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_quality_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'clustering_quality_comparison.png'}")

    # 2. Correlation between clustering quality and probing performance
    fig, ax = plt.subplots(figsize=(10, 8))

    correlations = []
    for result in all_results:
        if result['probing'] is None:
            continue

        for inv_name, task_id in inv_to_task.items():
            if inv_name in result['clustering'] and task_id in result['probing']:
                sil = result['clustering'][inv_name]['silhouette']
                probe = result['probing'][task_id]
                correlations.append({
                    'encoder': result['encoder'],
                    'task': task_id,
                    'silhouette': sil,
                    'probing': probe
                })

    if correlations:
        # Scatter plot
        x = [c['silhouette'] for c in correlations]
        y = [c['probing'] for c in correlations]
        colors = [hash(c['encoder']) % 20 for c in correlations]

        scatter = ax.scatter(x, y, c=colors, cmap='tab20', alpha=0.6, s=100)

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Compute correlation
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Silhouette Score (Clustering Quality)', fontsize=12)
        ax.set_ylabel('Probing Performance', fontsize=12)
        ax.set_title('Correlation: Embedding Space Clustering vs Probing Performance',
                    fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'clustering_vs_probing_correlation.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'clustering_vs_probing_correlation.png'}")
        print(f"\nOverall correlation (Silhouette vs Probing): r = {corr:.3f}")

    # Save detailed results
    import json
    with open(output_dir / 'clustering_analysis_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json_results = []
        for result in all_results:
            json_result = {
                'encoder': result['encoder'],
                'embedding_dim': convert(result['embedding_dim']),
                'n_proteins': convert(result['n_proteins']),
                'clustering': {k: {kk: convert(vv) for kk, vv in v.items()}
                             for k, v in result['clustering'].items()},
                'probing': {k: convert(v) for k, v in result['probing'].items()} if result['probing'] else None
            }
            json_results.append(json_result)

        json.dump(json_results, f, indent=2)

    print(f"Saved: {output_dir / 'clustering_analysis_results.json'}")
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
