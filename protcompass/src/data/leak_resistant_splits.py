"""
Leak-resistant split generation utilities for downstream tasks.

Implements the core principles:
1. Split by biological unit (not by example)
2. Sequence clustering at 30% identity before splitting
3. Keep near-duplicate measurements together
4. Respect official published splits when available
5. Use curated label sources only
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import subprocess
import tempfile
from pathlib import Path


def cluster_sequences_mmseqs2(
    sequences: Dict[str, str],
    identity_threshold: float = 0.3,
    coverage: float = 0.8,
    mmseqs_path: str = "mmseqs",
) -> Dict[str, int]:
    """
    Cluster sequences using MMseqs2 at specified identity threshold.

    Args:
        sequences: Dict mapping sequence IDs to sequences
        identity_threshold: Sequence identity threshold (0.0-1.0)
        coverage: Coverage threshold (0.0-1.0)
        mmseqs_path: Path to mmseqs binary

    Returns:
        Dict mapping sequence ID to cluster ID
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write sequences to FASTA
        fasta_path = tmpdir / "sequences.fasta"
        with open(fasta_path, 'w') as f:
            for seq_id, seq in sequences.items():
                f.write(f">{seq_id}\n{seq}\n")

        # Run MMseqs2 clustering
        db_path = tmpdir / "seqDB"
        cluster_path = tmpdir / "clusterDB"
        tmp_path = tmpdir / "tmp"

        try:
            # Create sequence database
            subprocess.run(
                [mmseqs_path, "createdb", str(fasta_path), str(db_path)],
                check=True, capture_output=True
            )

            # Cluster
            subprocess.run(
                [
                    mmseqs_path, "cluster",
                    str(db_path), str(cluster_path), str(tmp_path),
                    "--min-seq-id", str(identity_threshold),
                    "-c", str(coverage),
                    "--cov-mode", "0",  # bidirectional coverage
                ],
                check=True, capture_output=True
            )

            # Create TSV output
            tsv_path = tmpdir / "clusters.tsv"
            subprocess.run(
                [
                    mmseqs_path, "createtsv",
                    str(db_path), str(db_path), str(cluster_path), str(tsv_path)
                ],
                check=True, capture_output=True
            )

            # Parse clusters
            seq_to_cluster = {}
            cluster_id = 0
            cluster_map = {}

            with open(tsv_path) as f:
                for line in f:
                    rep, member = line.strip().split('\t')
                    if rep not in cluster_map:
                        cluster_map[rep] = cluster_id
                        cluster_id += 1
                    seq_to_cluster[member] = cluster_map[rep]

            return seq_to_cluster

        except subprocess.CalledProcessError as e:
            print(f"MMseqs2 clustering failed: {e}")
            print(f"stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
            # Fallback: each sequence is its own cluster
            return {seq_id: i for i, seq_id in enumerate(sequences.keys())}


def split_by_clusters(
    cluster_assignments: Dict[str, int],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Split sequence IDs by cluster membership.

    All sequences in the same cluster go to the same split.

    Args:
        cluster_assignments: Dict mapping sequence ID to cluster ID
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for test
        seed: Random seed

    Returns:
        (train_ids, val_ids, test_ids)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    # Group sequences by cluster
    clusters = defaultdict(list)
    for seq_id, cluster_id in cluster_assignments.items():
        clusters[cluster_id].append(seq_id)

    # Shuffle clusters
    rng = np.random.RandomState(seed)
    cluster_ids = list(clusters.keys())
    rng.shuffle(cluster_ids)

    # Assign clusters to splits
    n_clusters = len(cluster_ids)
    n_train = int(n_clusters * train_frac)
    n_val = int(n_clusters * val_frac)

    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train:n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val:]

    # Collect sequence IDs
    train_ids = set()
    val_ids = set()
    test_ids = set()

    for cid in train_clusters:
        train_ids.update(clusters[cid])
    for cid in val_clusters:
        val_ids.update(clusters[cid])
    for cid in test_clusters:
        test_ids.update(clusters[cid])

    return train_ids, val_ids, test_ids


def split_by_scaffold(
    data: List[Dict],
    scaffold_key: str = "scaffold_id",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data by scaffold/complex/assay ID.

    All examples from the same scaffold go to the same split.

    Args:
        data: List of data dictionaries
        scaffold_key: Key in dict that identifies the scaffold
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for test
        seed: Random seed

    Returns:
        (train_data, val_data, test_data)
    """
    # Group by scaffold
    scaffolds = defaultdict(list)
    for item in data:
        scaffold_id = item[scaffold_key]
        scaffolds[scaffold_id].append(item)

    # Shuffle scaffolds
    rng = np.random.RandomState(seed)
    scaffold_ids = list(scaffolds.keys())
    rng.shuffle(scaffold_ids)

    # Assign scaffolds to splits
    n_scaffolds = len(scaffold_ids)
    n_train = int(n_scaffolds * train_frac)
    n_val = int(n_scaffolds * val_frac)

    train_scaffolds = scaffold_ids[:n_train]
    val_scaffolds = scaffold_ids[n_train:n_train + n_val]
    test_scaffolds = scaffold_ids[n_train + n_val:]

    # Collect data
    train_data = []
    val_data = []
    test_data = []

    for sid in train_scaffolds:
        train_data.extend(scaffolds[sid])
    for sid in val_scaffolds:
        val_data.extend(scaffolds[sid])
    for sid in test_scaffolds:
        test_data.extend(scaffolds[sid])

    return train_data, val_data, test_data


def split_by_protein_for_ppi(
    interactions: List[Tuple[str, str, int]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    Split PPI data by protein, not by edge.

    Ensures no protein appears in multiple splits.

    Args:
        interactions: List of (protein_a, protein_b, label) tuples
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for test
        seed: Random seed

    Returns:
        (train_interactions, val_interactions, test_interactions)
    """
    # Collect all unique proteins
    proteins = set()
    for prot_a, prot_b, _ in interactions:
        proteins.add(prot_a)
        proteins.add(prot_b)

    # Shuffle proteins
    rng = np.random.RandomState(seed)
    protein_list = list(proteins)
    rng.shuffle(protein_list)

    # Assign proteins to splits
    n_proteins = len(protein_list)
    n_train = int(n_proteins * train_frac)
    n_val = int(n_proteins * val_frac)

    train_proteins = set(protein_list[:n_train])
    val_proteins = set(protein_list[n_train:n_train + n_val])
    test_proteins = set(protein_list[n_train + n_val:])

    # Assign interactions
    train_interactions = []
    val_interactions = []
    test_interactions = []

    for prot_a, prot_b, label in interactions:
        # Interaction goes to split if BOTH proteins are in that split
        if prot_a in train_proteins and prot_b in train_proteins:
            train_interactions.append((prot_a, prot_b, label))
        elif prot_a in val_proteins and prot_b in val_proteins:
            val_interactions.append((prot_a, prot_b, label))
        elif prot_a in test_proteins and prot_b in test_proteins:
            test_interactions.append((prot_a, prot_b, label))
        # Else: interaction spans splits, discard

    return train_interactions, val_interactions, test_interactions


def split_by_superfamily(
    domains: List[Dict],
    superfamily_key: str = "superfamily",
    family_key: str = "family",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split domain data by superfamily, then by family.

    Ensures no superfamily overlap between splits.

    Args:
        domains: List of domain dictionaries
        superfamily_key: Key for superfamily ID
        family_key: Key for family ID
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for test
        seed: Random seed

    Returns:
        (train_domains, val_domains, test_domains)
    """
    # Group by superfamily
    superfamilies = defaultdict(list)
    for domain in domains:
        sf_id = domain[superfamily_key]
        superfamilies[sf_id].append(domain)

    # Shuffle superfamilies
    rng = np.random.RandomState(seed)
    sf_ids = list(superfamilies.keys())
    rng.shuffle(sf_ids)

    # Assign superfamilies to splits
    n_sf = len(sf_ids)
    n_train = int(n_sf * train_frac)
    n_val = int(n_sf * val_frac)

    train_sf = sf_ids[:n_train]
    val_sf = sf_ids[n_train:n_train + n_val]
    test_sf = sf_ids[n_train + n_val:]

    # Collect domains
    train_domains = []
    val_domains = []
    test_domains = []

    for sf_id in train_sf:
        train_domains.extend(superfamilies[sf_id])
    for sf_id in val_sf:
        val_domains.extend(superfamilies[sf_id])
    for sf_id in test_sf:
        test_domains.extend(superfamilies[sf_id])

    return train_domains, val_domains, test_domains


def verify_no_overlap(
    train_ids: Set[str],
    val_ids: Set[str],
    test_ids: Set[str],
) -> Dict[str, int]:
    """
    Verify no overlap between splits and return statistics.

    Args:
        train_ids: Training set IDs
        val_ids: Validation set IDs
        test_ids: Test set IDs

    Returns:
        Dict with overlap statistics
    """
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    stats = {
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "total_overlap": len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap),
    }

    if stats["total_overlap"] > 0:
        print(f"WARNING: Found {stats['total_overlap']} overlapping IDs between splits!")
        print(f"  Train-Val: {stats['train_val_overlap']}")
        print(f"  Train-Test: {stats['train_test_overlap']}")
        print(f"  Val-Test: {stats['val_test_overlap']}")
    else:
        print("✓ No overlap between splits")

    return stats


def verify_sequence_identity(
    train_sequences: Dict[str, str],
    test_sequences: Dict[str, str],
    identity_threshold: float = 0.3,
    sample_size: int = 1000,
) -> Dict[str, float]:
    """
    Verify no high-identity sequences between train and test.

    Samples pairs and checks identity using simple alignment.
    For production use, run full MMseqs2 search.

    Args:
        train_sequences: Training sequences
        test_sequences: Test sequences
        identity_threshold: Maximum allowed identity
        sample_size: Number of pairs to check

    Returns:
        Dict with identity statistics
    """
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment

    rng = np.random.RandomState(42)

    train_ids = list(train_sequences.keys())
    test_ids = list(test_sequences.keys())

    n_samples = min(sample_size, len(train_ids) * len(test_ids))

    identities = []
    violations = 0

    for _ in range(n_samples):
        train_id = rng.choice(train_ids)
        test_id = rng.choice(test_ids)

        seq1 = train_sequences[train_id]
        seq2 = test_sequences[test_id]

        # Simple identity calculation
        alignments = pairwise2.align.globalxx(seq1, seq2)
        if alignments:
            aln = alignments[0]
            matches = sum(a == b for a, b in zip(aln.seqA, aln.seqB))
            identity = matches / max(len(seq1), len(seq2))
            identities.append(identity)

            if identity > identity_threshold:
                violations += 1

    stats = {
        "n_samples": n_samples,
        "mean_identity": np.mean(identities) if identities else 0.0,
        "max_identity": np.max(identities) if identities else 0.0,
        "violations": violations,
        "violation_rate": violations / n_samples if n_samples > 0 else 0.0,
    }

    if stats["violation_rate"] > 0.01:
        print(f"WARNING: {stats['violation_rate']:.2%} of sampled pairs exceed {identity_threshold:.0%} identity!")
    else:
        print(f"✓ Identity check passed (max={stats['max_identity']:.2%})")

    return stats
