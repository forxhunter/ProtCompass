"""
Protein-Protein Interaction (PPI) prediction evaluation.

Evaluates binary classification of protein pairs using PFMBench datasets:
- human_ppi: Human protein interactions (30k train, 196 test) - binary classification
- yeast_ppi: Yeast protein interactions (4k train, 336 test) - binary classification
- ppi_affinity: PPI affinity prediction (2.4k train, 327 test) - regression

Usage:
  python downstream_tasks/downstream_ppi.py --encoder random --dataset human_ppi
  python downstream_tasks/downstream_ppi.py --encoder esm2 --dataset all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Data paths
DATA_DIR = Path("/mnt/b/structuralenembedded_data/ppi/PFMBench/dataset")
RESULTS_DIR = ROOT / "downstream_tasks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "human_ppi": {
        "train": DATA_DIR / "human_ppi" / "human_ppi_with_pdb_train.csv",
        "val": DATA_DIR / "human_ppi" / "human_ppi_with_pdb_valid.csv",
        "test": DATA_DIR / "human_ppi" / "human_ppi_with_pdb_test.csv",
    },
    "yeast_ppi": {
        "train": DATA_DIR / "yeast_ppi" / "yeast_ppi_with_pdb_train.csv",
        "val": DATA_DIR / "yeast_ppi" / "yeast_ppi_with_pdb_valid.csv",
        "test": DATA_DIR / "yeast_ppi" / "yeast_ppi_with_pdb_test.csv",
    },
    "ppi_affinity": {
        "train": DATA_DIR / "ppi_affinity" / "ppi_affinity_train.csv",
        "val": DATA_DIR / "ppi_affinity" / "ppi_affinity_valid.csv",
        "test": DATA_DIR / "ppi_affinity" / "ppi_affinity_test.csv",
    },
}


def load_ppi_data(dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load PPI data and return (pair_sequences, labels)."""
    csv_path = DATASETS[dataset][split]
    df = pd.read_csv(csv_path)

    # Each row has: unique_id, aa_seq (seq1|seq2), label, pdb_hash
    sequences = []
    for seq_pair in df['aa_seq']:
        seqs = seq_pair.split('|')
        if len(seqs) == 2:
            sequences.append(seqs)
        else:
            # Handle edge case
            sequences.append([seq_pair, seq_pair])

    labels = df['label'].values
    return sequences, labels


def embed_sequence(seq: str, encoder: str, dim: int = 512) -> np.ndarray:
    """
    Generate embedding for a single sequence.

    For now, all encoders use random embeddings as placeholder.
    Random embeddings are deterministic based on sequence hash.
    """
    # Random baseline: fixed seed based on sequence hash for reproducibility
    rng = np.random.RandomState(hash(seq) % (2**32))
    return rng.randn(dim).astype(np.float32)


def embed_pairs(sequences: list, encoder: str, dim: int = 512) -> np.ndarray:
    """
    Embed protein pairs.

    For each pair (seq1, seq2), we concatenate their embeddings: [emb1 || emb2]
    This gives a 2*dim dimensional feature vector.
    """
    embeddings = []
    for seq1, seq2 in sequences:
        emb1 = embed_sequence(seq1, encoder, dim)
        emb2 = embed_sequence(seq2, encoder, dim)
        # Concatenate pair embeddings
        pair_emb = np.concatenate([emb1, emb2])
        embeddings.append(pair_emb)
    return np.array(embeddings)


def train_and_evaluate_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42
) -> Dict[str, float]:
    """Train logistic regression probe and evaluate for binary classification."""

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        class_weight='balanced'  # Handle class imbalance
    )
    clf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # Compute metrics
    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_proba)),
        "auprc": float(average_precision_score(y_test, y_proba)),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pos_ratio_train": float(y_train.mean()),
        "pos_ratio_test": float(y_test.mean()),
    }

    return results


def train_and_evaluate_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42
) -> Dict[str, float]:
    """Train ridge regression probe and evaluate for continuous values."""

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ridge regression
    reg = Ridge(alpha=1.0, random_state=seed)
    reg.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = reg.predict(X_test_scaled)

    # Compute metrics
    results = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "mean_train": float(y_train.mean()),
        "std_train": float(y_train.std()),
        "mean_test": float(y_test.mean()),
        "std_test": float(y_test.std()),
    }

    return results


def evaluate_ppi(encoder: str, dataset: str, dim: int = 512, seed: int = 42) -> Dict[str, float]:
    """Evaluate PPI prediction on a single dataset."""

    print(f"\n{'='*60}")
    print(f"Evaluating {dataset} with {encoder} encoder")
    print(f"{'='*60}")

    # Load data
    print("Loading data...")
    train_seqs, y_train = load_ppi_data(dataset, "train")
    test_seqs, y_test = load_ppi_data(dataset, "test")

    # Determine if this is classification or regression
    is_regression = dataset == "ppi_affinity"

    if is_regression:
        print(f"Train: {len(y_train)} pairs (mean: {y_train.mean():.2f}, std: {y_train.std():.2f})")
        print(f"Test: {len(y_test)} pairs (mean: {y_test.mean():.2f}, std: {y_test.std():.2f})")
    else:
        print(f"Train: {len(y_train)} pairs ({int(y_train.sum())} positive, {int((~y_train.astype(bool)).sum())} negative)")
        print(f"Test: {len(y_test)} pairs ({int(y_test.sum())} positive, {int((~y_test.astype(bool)).sum())} negative)")

    # Generate embeddings
    print(f"Generating {encoder} embeddings (dim={dim})...")
    X_train = embed_pairs(train_seqs, encoder, dim)
    X_test = embed_pairs(test_seqs, encoder, dim)

    print(f"Train embeddings: {X_train.shape}")
    print(f"Test embeddings: {X_test.shape}")

    # Train and evaluate
    print("Training probe...")
    if is_regression:
        results = train_and_evaluate_regression(X_train, y_train, X_test, y_test, seed)
        # Print results
        print(f"\nResults:")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  R²: {results['r2']:.4f}")
    else:
        results = train_and_evaluate_classification(X_train, y_train, X_test, y_test, seed)
        # Print results
        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  AUROC: {results['auroc']:.4f}")
        print(f"  AUPRC: {results['auprc']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPI prediction")
    parser.add_argument("--encoder", type=str, default="random",
                        help="Encoder to use (random, esm2, etc.)")
    parser.add_argument("--dataset", type=str, default="human_ppi",
                        choices=list(DATASETS.keys()) + ["all"],
                        help="Dataset to evaluate")
    parser.add_argument("--dim", type=int, default=512,
                        help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Determine which datasets to evaluate
    datasets_to_eval = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    # Evaluate each dataset
    all_results = {}
    for dataset in datasets_to_eval:
        results = evaluate_ppi(args.encoder, dataset, args.dim, args.seed)
        all_results[dataset] = results

        # Save results
        output_file = RESULTS_DIR / f"ppi_{dataset}_{args.encoder}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    # Print summary if multiple datasets
    if len(datasets_to_eval) > 1:
        print(f"\n{'='*60}")
        print("Summary across all datasets:")
        print(f"{'='*60}")
        for dataset, results in all_results.items():
            print(f"{dataset:15s} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | AUROC: {results['auroc']:.4f}")


if __name__ == "__main__":
    main()
