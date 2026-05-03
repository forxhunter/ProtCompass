#!/usr/bin/env python3
"""
Visualize encoder embedding spaces colored by invariant families.
Shows how different structural properties cluster in each encoder's representation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from typing import Dict, List
import argparse


def load_embeddings(encoder_name: str, seed: int = 0):
    """Load saved embeddings and labels."""
    emb_dir = Path("outputs/encoder_embeddings")
    emb_file = emb_dir / f"{encoder_name}_seed_{seed}.npz"

    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_file}")

    data = np.load(emb_file, allow_pickle=True)
    return data


def plot_embedding_space_by_invariant(
    embeddings: np.ndarray,
    labels: np.ndarray,
    invariant_name: str,
    encoder_name: str,
    output_path: Path,
    method: str = "tsne"
):
    """
    Reduce embeddings to 2D and color by invariant family labels.

    Args:
        embeddings: (N, D) array
        labels: (N,) array of invariant labels
        invariant_name: Name of the invariant (e.g., 'architecture_label')
        encoder_name: Name of the encoder
        method: 'tsne' or 'pca'
    """

    print(f"\nVisualizing {encoder_name} embedding space by {invariant_name}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Unique labels: {len(np.unique(labels))}")

    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        coords_2d = reducer.fit_transform(embeddings)
        method_label = "t-SNE"
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
        var_explained = reducer.explained_variance_ratio_
        method_label = f"PCA (PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%})"

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Use a colormap
    if n_labels <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
    elif n_labels <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_labels))

    # Plot each label group
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[colors[i]],
            label=f"Class {label}",
            alpha=0.6,
            s=20,
            edgecolors='none'
        )

    # Formatting
    ax.set_xlabel(f"{method_label} Component 1", fontsize=11)
    ax.set_ylabel(f"{method_label} Component 2", fontsize=11)
    ax.set_title(f"{encoder_name}\nEmbedding Space by {invariant_name}",
                 fontsize=12, fontweight='bold')

    # Legend
    if n_labels <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                 fontsize=8, frameon=True, ncol=1)
    else:
        # Too many labels, skip legend
        pass

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_multi_invariant_grid(
    embeddings: np.ndarray,
    all_labels: Dict[str, np.ndarray],
    encoder_name: str,
    output_dir: Path,
    method: str = "tsne"
):
    """
    Create a grid of subplots showing the same embedding space
    colored by different invariant families.
    """

    # Select key invariants to visualize
    key_invariants = [
        'architecture_label',  # I7
        'dssp_8',              # I8
        'core_surface',        # I5
        'contact_order',       # I6
    ]

    available_invariants = [inv for inv in key_invariants
                           if f"labels_{inv}" in all_labels]

    if not available_invariants:
        print(f"No key invariants found for {encoder_name}")
        return

    n_plots = len(available_invariants)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Compute 2D coordinates once
    print(f"\nComputing {method.upper()} for {encoder_name}...")
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        coords_2d = reducer.fit_transform(embeddings)
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)

    # Plot each invariant
    for idx, inv_name in enumerate(available_invariants):
        ax = axes[idx]
        labels = all_labels[f"labels_{inv_name}"]

        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        # Color map
        if n_labels <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, n_labels))

        # Scatter plot
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[colors[i]],
                label=f"{label}",
                alpha=0.6,
                s=15,
                edgecolors='none'
            )

        ax.set_title(f"{inv_name}", fontsize=11, fontweight='bold')
        ax.set_xlabel(f"{method.upper()} 1", fontsize=9)
        ax.set_ylabel(f"{method.upper()} 2", fontsize=9)

        if n_labels <= 10:
            ax.legend(fontsize=7, loc='best', frameon=True, ncol=2)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f"{encoder_name} Embedding Space\nColored by Different Invariant Families",
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    output_path = output_dir / f"{encoder_name}_multi_invariant_grid.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Saved grid: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True,
                       help="Encoder name (must have embeddings extracted)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="tsne",
                       choices=["tsne", "pca"])
    parser.add_argument("--invariant", type=str, default="all",
                       help="Specific invariant to visualize, or 'all' for grid")
    args = parser.parse_args()

    # Load embeddings
    data = load_embeddings(args.encoder, args.seed)
    embeddings = data['embeddings']

    # Output directory
    output_dir = Path("outputs/embedding_space_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all label arrays
    all_labels = {k: data[k] for k in data.files if k.startswith('labels_')}

    if args.invariant == "all":
        # Create multi-invariant grid
        plot_multi_invariant_grid(
            embeddings, all_labels, args.encoder, output_dir, args.method
        )
    else:
        # Single invariant plot
        label_key = f"labels_{args.invariant}"
        if label_key not in all_labels:
            print(f"Error: {args.invariant} not found in saved labels")
            print(f"Available: {[k.replace('labels_', '') for k in all_labels.keys()]}")
            return

        labels = all_labels[label_key]
        output_path = output_dir / f"{args.encoder}_{args.invariant}_{args.method}.png"

        plot_embedding_space_by_invariant(
            embeddings, labels, args.invariant, args.encoder,
            output_path, args.method
        )


if __name__ == "__main__":
    main()
