"""
Visualization utilities for invariant fingerprints.

Provides plots for:
- Radar/spider plots comparing encoder fingerprints
- Heatmaps of probe results
- Embedding space visualizations
- Intervention effect plots
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from src.probes import InvariantFingerprint


# Color scheme for bottleneck classes
BOTTLENECK_COLORS = {
    'A': '#E64B35',  # Red for Token
    'B': '#4DBBD5',  # Blue for Graph
    'C': '#00A087',  # Green for Equivariant
    'hybrid': '#7E6148',  # Brown for Hybrid
}

# Labels for invariant families
INVARIANT_LABELS = ['I1\nLocal', 'I2\nMetric', 'I3\nPacking', 'I4\nTopology']
INVARIANT_LABELS_SHORT = ['I1', 'I2', 'I3', 'I4']


def check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def radar_plot(
    fingerprints: List[InvariantFingerprint],
    labels: Optional[List[str]] = None,
    title: str = "Invariant Fingerprints",
    figsize: Tuple[int, int] = (8, 8),
    output_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Create radar/spider plot comparing encoder fingerprints.

    Args:
        fingerprints: List of invariant fingerprints
        labels: Custom axis labels (default: I1-I4)
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure
        show: Whether to display plot

    Returns:
        matplotlib Figure
    """
    check_matplotlib()

    labels = labels or INVARIANT_LABELS
    num_vars = len(labels)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for fp in fingerprints:
        values = fp.vector.tolist()
        values += values[:1]  # Complete the circle

        # Determine color based on bottleneck class if available
        color = BOTTLENECK_COLORS.get('A', '#333333')  # Default gray

        ax.plot(angles, values, 'o-', linewidth=2, label=fp.encoder_name)
        ax.fill(angles, values, alpha=0.25)

    # Configure axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title(title, size=14, fontweight='bold', y=1.08)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def probe_heatmap(
    probe_results: Dict[str, Dict[str, float]],
    title: str = "Probe Results",
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "RdYlGn",
    output_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Create heatmap of probe results across encoders.

    Args:
        probe_results: Dict mapping encoder_name -> {probe_name -> score}
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        output_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    check_matplotlib()

    # Build data matrix
    encoders = list(probe_results.keys())
    probes = list(probe_results[encoders[0]].keys())

    data = np.zeros((len(probes), len(encoders)))
    for j, encoder in enumerate(encoders):
        for i, probe in enumerate(probes):
            data[i, j] = probe_results[encoder].get(probe, 0)

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            data,
            xticklabels=encoders,
            yticklabels=probes,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax,
        )
    else:
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(encoders)))
        ax.set_xticklabels(encoders, rotation=45, ha='right')
        ax.set_yticks(range(len(probes)))
        ax.set_yticklabels(probes)
        plt.colorbar(im, ax=ax)

        # Add annotations
        for i in range(len(probes)):
            for j in range(len(encoders)):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center')

    ax.set_title(title, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def fingerprint_bar_chart(
    fingerprints: List[InvariantFingerprint],
    title: str = "Invariant Preservation by Encoder",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Create grouped bar chart comparing fingerprints.

    Args:
        fingerprints: List of fingerprints
        title: Plot title
        figsize: Figure size
        output_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    check_matplotlib()

    encoders = [fp.encoder_name for fp in fingerprints]
    n_encoders = len(encoders)
    n_invariants = 4

    x = np.arange(n_invariants)
    width = 0.8 / n_encoders

    fig, ax = plt.subplots(figsize=figsize)

    for i, fp in enumerate(fingerprints):
        offset = (i - n_encoders / 2 + 0.5) * width
        bars = ax.bar(x + offset, fp.vector, width, label=fp.encoder_name)

    ax.set_xlabel('Invariant Family')
    ax.set_ylabel('Preservation Score')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(INVARIANT_LABELS_SHORT)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def intervention_effect_plot(
    intervention_results: List[Dict[str, Any]],
    metric: str = 'l2_change',
    title: str = "Intervention Effects",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot effects of interventions on embeddings.

    Args:
        intervention_results: List of intervention result dicts
        metric: Which metric to plot ('l2_change' or 'cosine_sim')
        title: Plot title
        figsize: Figure size
        output_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    check_matplotlib()

    names = []
    values = []

    for result in intervention_results:
        name = result.get('intervention_name', 'unknown')
        value = result.get(metric, result.get('embedding_change_l2', 0))
        names.append(name)
        values.append(value)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(names, values, color=colors)

    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.3f}',
            va='center',
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def embedding_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
    title: str = "Embedding Space (t-SNE)",
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
    output_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings: Embedding matrix (N, D)
        labels: Label array (N,)
        label_names: Names for each label class
        title: Plot title
        figsize: Figure size
        perplexity: t-SNE perplexity
        output_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    check_matplotlib()

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("sklearn required for t-SNE. Install with: pip install scikit-learn")

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[int(label)] if label_names else str(label)
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=name,
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def decision_table_plot(
    decision_table: Dict[str, Dict[str, Any]],
    title: str = "Task-Encoder Decision Table",
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Visualize the task-encoder decision table.

    Args:
        decision_table: Dict mapping task -> {encoder -> score}
        title: Plot title
        figsize: Figure size
        output_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    check_matplotlib()

    tasks = list(decision_table.keys())
    encoders = list(decision_table[tasks[0]].keys())

    data = np.zeros((len(tasks), len(encoders)))
    for i, task in enumerate(tasks):
        for j, encoder in enumerate(encoders):
            data[i, j] = decision_table[task].get(encoder, 0)

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            data,
            xticklabels=encoders,
            yticklabels=tasks,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            ax=ax,
        )
    else:
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(encoders)))
        ax.set_xticklabels(encoders, rotation=45, ha='right')
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks)
        plt.colorbar(im, ax=ax)

    ax.set_title(title, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def save_all_plots(
    fingerprints: List[InvariantFingerprint],
    probe_results: Optional[Dict[str, Dict[str, float]]] = None,
    intervention_results: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = "outputs/plots",
) -> List[str]:
    """
    Generate and save all standard plots.

    Args:
        fingerprints: List of fingerprints
        probe_results: Optional probe results
        intervention_results: Optional intervention results
        output_dir: Directory to save plots

    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    # Radar plot
    path = os.path.join(output_dir, "fingerprint_radar.png")
    radar_plot(fingerprints, output_path=path, show=False)
    saved_paths.append(path)

    # Bar chart
    path = os.path.join(output_dir, "fingerprint_bars.png")
    fingerprint_bar_chart(fingerprints, output_path=path, show=False)
    saved_paths.append(path)

    # Probe heatmap
    if probe_results:
        path = os.path.join(output_dir, "probe_heatmap.png")
        probe_heatmap(probe_results, output_path=path, show=False)
        saved_paths.append(path)

    # Intervention effects
    if intervention_results:
        path = os.path.join(output_dir, "intervention_effects.png")
        intervention_effect_plot(intervention_results, output_path=path, show=False)
        saved_paths.append(path)

    return saved_paths
