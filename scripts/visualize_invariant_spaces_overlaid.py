#!/usr/bin/env python3
"""
Visualize overlaid UMAP projections of encoder embeddings colored by invariant families.

For each modality (sequence/structure/multimodal), creates a figure showing:
- All encoders in that modality overlaid in the same UMAP space
- Different marker shapes per encoder
- Colored by invariant family labels
- Can generate figures for all 10 invariants or a subset

Usage:
    python visualize_invariant_spaces_overlaid.py --modality sequence --invariants all
    python visualize_invariant_spaces_overlaid.py --modality structure --invariants I1,I7,I5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from umap import UMAP

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Encoder taxonomy
ENCODER_TAXONOMY = {
    'sequence': [
        'esm2', 'esm2_150m', 'esm2_35m', 'esm1b',
        'prottrans', 'prostt5', 'proteinbert_bfd', 'ankh'
    ],
    'structure': [
        'foldseek', 'gearnet_pretrained', 'gcpnet_pretrained',
        'egnn_pretrained', 'gvp', 'tfn', 'ipa',
        'dmasif_pretrained', 'cdconv_pretrained', 'schnet',
        'dimenet', 'mace', 'proteinmpnn'
    ],
    'multimodal': [
        'saprot', 'saprot_pretrained', 'esm_if', 'esm3', 'foldvision'
    ]
}

# Marker shapes for encoders (cycle through if more than available)
MARKER_SHAPES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']

# Invariant family metadata - mapped to actual keys in saved files
INVARIANT_METADATA = {
    'I1': {'name': 'Secondary Structure', 'label_key': 'labels_dssp_8', 'n_classes': 8},
    'I2': {'name': 'Solvent Accessibility', 'label_key': 'labels_rsa_bins', 'n_classes': 3},
    'I3': {'name': 'Contact Density', 'label_key': 'labels_contact_density', 'n_classes': 3},
    'I4': {'name': 'Backbone Angles', 'label_key': 'labels_phi_bins', 'n_classes': 4},
    'I5': {'name': 'Global Geometry', 'label_key': 'labels_dssp_3', 'n_classes': 3},
    'I6': {'name': 'Sequence Recovery', 'label_key': 'labels_sequence', 'n_classes': 20},
    'I7': {'name': 'Fold Classification', 'label_key': 'labels_architecture_label', 'n_classes': 31},
    'I8': {'name': 'Local Curvature', 'label_key': 'labels_curvature', 'n_classes': 3, 'is_continuous': True},
    'I9': {'name': 'B-factor', 'label_key': 'labels_bfactor', 'n_classes': 3, 'is_continuous': True},
    'I10': {'name': 'Core/Surface', 'label_key': 'labels_core_surface_labels', 'n_classes': 2},
}

def load_embeddings_and_labels(encoder_name, embedding_dir, seed=0):
    """Load embeddings and all invariant labels for an encoder."""
    emb_path = Path(embedding_dir) / f"{encoder_name}_seed_{seed}.npz"

    if not emb_path.exists():
        print(f"Warning: {emb_path} not found, skipping {encoder_name}")
        return None, None

    data = np.load(emb_path, allow_pickle=True)
    embeddings = data['embeddings']

    # Load all invariant labels
    labels = {}
    for inv_id, inv_meta in INVARIANT_METADATA.items():
        label_key = inv_meta['label_key']
        if label_key in data:
            label_data = data[label_key]

            # Handle continuous variables - bin them
            if inv_meta.get('is_continuous', False):
                # Bin continuous values into n_classes bins
                label_data = np.digitize(label_data,
                                        bins=np.linspace(label_data.min(), label_data.max(),
                                                        inv_meta['n_classes'] + 1)[1:-1])

            labels[inv_id] = label_data
        else:
            print(f"Warning: {label_key} not found in {encoder_name}")
            labels[inv_id] = None

    return embeddings, labels


def align_dimensions(embeddings_dict, target_dim=128):
    """Align all embeddings to same dimension using PCA."""
    # Determine actual target dimension based on n_samples
    n_samples = min(emb.shape[0] for emb in embeddings_dict.values())
    actual_target_dim = min(target_dim, n_samples - 1)

    if actual_target_dim < target_dim:
        print(f"  Note: Reducing target dimension to {actual_target_dim} (limited by n_samples={n_samples})")

    aligned = {}

    for encoder, emb in embeddings_dict.items():
        if emb.shape[1] > actual_target_dim:
            print(f"  {encoder}: {emb.shape[1]}D -> {actual_target_dim}D via PCA")
            pca = PCA(n_components=actual_target_dim, random_state=42)
            aligned[encoder] = pca.fit_transform(emb)
        elif emb.shape[1] < actual_target_dim:
            # Pad with zeros if smaller
            print(f"  {encoder}: {emb.shape[1]}D -> {actual_target_dim}D via zero-padding")
            padding = np.zeros((emb.shape[0], actual_target_dim - emb.shape[1]))
            aligned[encoder] = np.hstack([emb, padding])
        else:
            aligned[encoder] = emb

    return aligned


def compute_overlaid_umap(embeddings_dict, labels_dict, invariant_id):
    """
    Compute single UMAP projection for all encoders concatenated.

    Returns:
        umap_coords: (N_total, 2) array
        encoder_ids: (N_total,) array indicating which encoder each point belongs to
        invariant_labels: (N_total,) array of invariant labels
    """
    # Concatenate all embeddings
    all_embeddings = []
    encoder_ids = []
    invariant_labels = []

    for encoder, emb in embeddings_dict.items():
        all_embeddings.append(emb)
        encoder_ids.extend([encoder] * len(emb))

        # Get invariant labels for this encoder
        inv_labels = labels_dict[encoder][invariant_id]
        if inv_labels is None:
            inv_labels = np.full(len(emb), -1)  # Missing label
        invariant_labels.extend(inv_labels)

    all_embeddings = np.vstack(all_embeddings)
    encoder_ids = np.array(encoder_ids)
    invariant_labels = np.array(invariant_labels)

    print(f"  Computing UMAP on {len(all_embeddings)} total points from {len(embeddings_dict)} encoders...")

    # Fit UMAP
    umap_model = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        metric='cosine'
    )
    umap_coords = umap_model.fit_transform(all_embeddings)

    return umap_coords, encoder_ids, invariant_labels


def compute_silhouette_per_encoder(umap_coords, encoder_ids, invariant_labels):
    """Compute silhouette score for each encoder separately."""
    scores = {}

    for encoder in np.unique(encoder_ids):
        mask = encoder_ids == encoder
        coords = umap_coords[mask]
        labels = invariant_labels[mask]

        # Remove missing labels
        valid_mask = labels >= 0
        if valid_mask.sum() < 2:
            scores[encoder] = np.nan
            continue

        coords_valid = coords[valid_mask]
        labels_valid = labels[valid_mask]

        # Need at least 2 classes
        if len(np.unique(labels_valid)) < 2:
            scores[encoder] = np.nan
            continue

        try:
            score = silhouette_score(coords_valid, labels_valid)
            scores[encoder] = score
        except:
            scores[encoder] = np.nan

    return scores


def plot_overlaid_umap(umap_coords, encoder_ids, invariant_labels,
                       invariant_id, modality, output_path, silhouette_scores):
    """Plot overlaid UMAP with different markers per encoder, colored by invariant."""

    inv_meta = INVARIANT_METADATA[invariant_id]
    inv_name = inv_meta['name']
    n_classes = inv_meta['n_classes']

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get unique encoders and assign markers
    unique_encoders = np.unique(encoder_ids)
    encoder_to_marker = {enc: MARKER_SHAPES[i % len(MARKER_SHAPES)]
                         for i, enc in enumerate(unique_encoders)}

    # Color palette for invariant classes
    if n_classes <= 10:
        colors = sns.color_palette("tab10", n_classes)
    elif n_classes <= 20:
        colors = sns.color_palette("tab20", n_classes)
    else:
        colors = sns.color_palette("husl", n_classes)

    # Plot each encoder with its marker
    for encoder in unique_encoders:
        mask = encoder_ids == encoder
        coords = umap_coords[mask]
        labels = invariant_labels[mask]

        # Plot each class with its color
        for class_id in range(n_classes):
            class_mask = labels == class_id
            if class_mask.sum() == 0:
                continue

            ax.scatter(
                coords[class_mask, 0],
                coords[class_mask, 1],
                c=[colors[class_id]],
                marker=encoder_to_marker[encoder],
                s=30,
                alpha=0.6,
                edgecolors='white',
                linewidths=0.3,
                label=f"{encoder}_{class_id}" if class_id == 0 else ""
            )

    # Create custom legends
    # Legend 1: Encoders (marker shapes)
    encoder_handles = [
        Line2D([0], [0], marker=encoder_to_marker[enc], color='gray',
               linestyle='', markersize=8, label=enc, markeredgecolor='white', markeredgewidth=0.5)
        for enc in unique_encoders
    ]

    # Legend 2: Invariant classes (colors)
    class_handles = [
        Patch(facecolor=colors[i], label=f"Class {i}", edgecolor='white', linewidth=0.5)
        for i in range(min(n_classes, 10))  # Show first 10 classes only
    ]
    if n_classes > 10:
        class_handles.append(Patch(facecolor='gray', label=f"... ({n_classes} total)", edgecolor='white'))

    # Position legends
    legend1 = ax.legend(handles=encoder_handles, loc='upper left',
                       title='Encoders', framealpha=0.9, fontsize=9)
    ax.add_artist(legend1)

    legend2 = ax.legend(handles=class_handles, loc='upper right',
                       title=f'{inv_name} Classes', framealpha=0.9, fontsize=9)

    # Compute mean silhouette
    valid_scores = [s for s in silhouette_scores.values() if not np.isnan(s)]
    if valid_scores:
        mean_sil = np.mean(valid_scores)
        sem_sil = np.std(valid_scores) / np.sqrt(len(valid_scores))
        sil_text = f"Mean Silhouette: {mean_sil:.3f} ± {sem_sil:.3f}"
    else:
        sil_text = "Mean Silhouette: N/A"

    # Title and labels
    ax.set_title(f"{modality.capitalize()} Encoders - {invariant_id}: {inv_name}\n{sil_text}",
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize overlaid encoder embeddings by invariant families")
    parser.add_argument('--modality', type=str, required=True,
                       choices=['sequence', 'structure', 'multimodal'],
                       help='Which encoder modality to visualize')
    parser.add_argument('--invariants', type=str, default='all',
                       help='Comma-separated invariant IDs (e.g., I1,I7,I5) or "all"')
    parser.add_argument('--embedding-dir', type=str,
                       default='outputs/encoder_embeddings',
                       help='Directory containing encoder embeddings')
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed number for embedding files')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/paper_figures_v10',
                       help='Output directory for figures')
    parser.add_argument('--layout', type=str, default='vertical',
                       choices=['vertical', 'grid'],
                       help='Layout: vertical stack or grid')

    args = parser.parse_args()

    # Parse invariants
    if args.invariants.lower() == 'all':
        invariants_to_plot = list(INVARIANT_METADATA.keys())
    else:
        invariants_to_plot = [inv.strip() for inv in args.invariants.split(',')]
        # Validate
        for inv in invariants_to_plot:
            if inv not in INVARIANT_METADATA:
                raise ValueError(f"Unknown invariant: {inv}")

    print(f"\n{'='*60}")
    print(f"Overlaid UMAP Visualization")
    print(f"{'='*60}")
    print(f"Modality: {args.modality}")
    print(f"Invariants: {', '.join(invariants_to_plot)}")
    print(f"Layout: {args.layout}")

    # Get encoders for this modality
    encoders = ENCODER_TAXONOMY[args.modality]
    print(f"\nEncoders ({len(encoders)}): {', '.join(encoders)}")

    # Load embeddings and labels
    print("\nLoading embeddings and labels...")
    embeddings_dict = {}
    labels_dict = {}

    for encoder in encoders:
        emb, labels = load_embeddings_and_labels(encoder, args.embedding_dir, args.seed)
        if emb is not None:
            embeddings_dict[encoder] = emb
            labels_dict[encoder] = labels

    if not embeddings_dict:
        print("Error: No embeddings loaded!")
        return

    print(f"\nLoaded {len(embeddings_dict)} encoders successfully")

    # Align dimensions
    print("\nAligning embedding dimensions...")
    embeddings_dict = align_dimensions(embeddings_dict, target_dim=128)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures for each invariant
    print(f"\nGenerating figures for {len(invariants_to_plot)} invariants...")

    for inv_id in invariants_to_plot:
        print(f"\n{inv_id}: {INVARIANT_METADATA[inv_id]['name']}")

        # Compute overlaid UMAP
        umap_coords, encoder_ids, invariant_labels = compute_overlaid_umap(
            embeddings_dict, labels_dict, inv_id
        )

        # Compute silhouette scores per encoder
        silhouette_scores = compute_silhouette_per_encoder(
            umap_coords, encoder_ids, invariant_labels
        )

        print(f"  Silhouette scores:")
        for enc, score in silhouette_scores.items():
            if not np.isnan(score):
                print(f"    {enc}: {score:.3f}")

        # Plot
        output_path = output_dir / f"{args.modality}_invariant_{inv_id}_{INVARIANT_METADATA[inv_id]['name'].lower().replace(' ', '_').replace('/', '_')}.png"
        plot_overlaid_umap(
            umap_coords, encoder_ids, invariant_labels,
            inv_id, args.modality, output_path, silhouette_scores
        )

    print(f"\n{'='*60}")
    print(f"Complete! Figures saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
