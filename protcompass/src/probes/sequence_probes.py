"""
I5 Sequence-Based Probing - Label Computation Functions

These functions compute labels for I5 probes directly from protein structures.
All labels can be computed without external databases.

I5 Probes:
1. amino_acid_identity - 20-class classification (already in dataset)
2. hydrophobicity - Kyte-Doolittle scale regression
3. molecular_weight - protein-level MW in kDa
4. net_charge - protein-level net charge at pH 7
5. aromatic_content - fraction of aromatic residues (F, W, Y)
6. aliphatic_index - Ikai's aliphatic index
7. accessibility_class - 3-class buried/intermediate/exposed (from DSSP RSA)
"""

import torch
import numpy as np
from typing import Dict, Optional, List


# Amino acid properties
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

MOLECULAR_WEIGHT = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}

# Net charge at pH 7 (approximate)
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,  # His ~10% protonated at pH 7
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

AROMATIC = {'F', 'W', 'Y'}
ALIPHATIC = {'A', 'V', 'I', 'L'}

# Aliphatic index coefficients (Ikai 1980)
ALIPHATIC_COEFF = {
    'A': 1.0, 'V': 2.9, 'I': 2.9, 'L': 3.9,
}

# Standard amino acid alphabet
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}


def compute_hydrophobicity(sequence: str) -> torch.Tensor:
    """
    Compute Kyte-Doolittle hydrophobicity for each residue.

    Args:
        sequence: Amino acid sequence string

    Returns:
        Tensor of shape (L,) with hydrophobicity values
    """
    values = [HYDROPHOBICITY.get(aa, 0.0) for aa in sequence]
    return torch.tensor(values, dtype=torch.float32)


def compute_molecular_weight(sequence: str) -> float:
    """
    Compute total molecular weight in kDa.

    Args:
        sequence: Amino acid sequence string

    Returns:
        Molecular weight in kDa
    """
    mw = sum(MOLECULAR_WEIGHT.get(aa, 0.0) for aa in sequence)
    # Subtract water molecules lost in peptide bonds
    mw -= 18.015 * (len(sequence) - 1)
    return mw / 1000.0  # Convert to kDa


def compute_net_charge(sequence: str) -> float:
    """
    Compute net charge at pH 7.

    Args:
        sequence: Amino acid sequence string

    Returns:
        Net charge (positive or negative)
    """
    charge = sum(CHARGE.get(aa, 0.0) for aa in sequence)
    # Add terminal charges
    charge += 1.0  # N-terminus (NH3+)
    charge -= 1.0  # C-terminus (COO-)
    return charge


def compute_aromatic_content(sequence: str) -> float:
    """
    Compute fraction of aromatic residues (F, W, Y).

    Args:
        sequence: Amino acid sequence string

    Returns:
        Fraction of aromatic residues [0, 1]
    """
    if len(sequence) == 0:
        return 0.0
    aromatic_count = sum(1 for aa in sequence if aa in AROMATIC)
    return aromatic_count / len(sequence)


def compute_aliphatic_index(sequence: str) -> float:
    """
    Compute Ikai's aliphatic index.

    AI = X_A + a*X_V + b*(X_I + X_L)
    where X_i is mole fraction and a=2.9, b=3.9

    Args:
        sequence: Amino acid sequence string

    Returns:
        Aliphatic index (typically 0-200)
    """
    if len(sequence) == 0:
        return 0.0

    counts = {aa: 0 for aa in ALIPHATIC}
    for aa in sequence:
        if aa in counts:
            counts[aa] += 1

    L = len(sequence)
    ai = (counts['A'] / L * 100.0 +
          counts['V'] / L * 100.0 * 2.9 +
          (counts['I'] + counts['L']) / L * 100.0 * 3.9)

    return ai


def compute_accessibility_class(rsa: torch.Tensor) -> torch.Tensor:
    """
    Convert relative solvent accessibility to 3-class labels.

    Classes:
        0 = buried (RSA < 0.25)
        1 = intermediate (0.25 <= RSA < 0.5)
        2 = exposed (RSA >= 0.5)

    Args:
        rsa: Relative solvent accessibility tensor of shape (L,)

    Returns:
        Class labels tensor of shape (L,) with values in {0, 1, 2}
    """
    classes = torch.zeros_like(rsa, dtype=torch.long)
    classes[rsa >= 0.25] = 1
    classes[rsa >= 0.5] = 2
    return classes


def compute_i5_labels(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compute all I5 labels for a batch of proteins.

    Args:
        batch: Dictionary containing:
            - sequence: (B, L) tensor of amino acid indices
            - mask: (B, L) boolean mask
            - rsa: (B, L) relative solvent accessibility (optional, for accessibility_class)

    Returns:
        Dictionary with computed labels:
            - hydrophobicity: (B, L) float tensor
            - molecular_weight: (B,) float tensor
            - net_charge: (B,) float tensor
            - aromatic_content: (B,) float tensor
            - aliphatic_index: (B,) float tensor
            - accessibility_class: (B, L) long tensor (if RSA available)
    """
    sequence_indices = batch['sequence']  # (B, L)
    mask = batch['mask']  # (B, L)
    B, L = sequence_indices.shape

    # Convert indices to sequences
    sequences = []
    for b in range(B):
        seq_len = mask[b].sum().item()
        seq_indices = sequence_indices[b, :seq_len].cpu().numpy()
        seq = ''.join(AA_ALPHABET[idx] if idx < 20 else 'A' for idx in seq_indices)
        sequences.append(seq)

    # Compute residue-level labels
    hydrophobicity_batch = []
    for seq in sequences:
        hydro = compute_hydrophobicity(seq)
        # Pad to max length
        padded = torch.zeros(L, dtype=torch.float32)
        padded[:len(hydro)] = hydro
        hydrophobicity_batch.append(padded)

    # Compute protein-level labels
    molecular_weight_batch = torch.tensor(
        [compute_molecular_weight(seq) for seq in sequences],
        dtype=torch.float32
    )

    net_charge_batch = torch.tensor(
        [compute_net_charge(seq) for seq in sequences],
        dtype=torch.float32
    )

    aromatic_content_batch = torch.tensor(
        [compute_aromatic_content(seq) for seq in sequences],
        dtype=torch.float32
    )

    aliphatic_index_batch = torch.tensor(
        [compute_aliphatic_index(seq) for seq in sequences],
        dtype=torch.float32
    )

    labels = {
        'hydrophobicity': torch.stack(hydrophobicity_batch),
        'molecular_weight': molecular_weight_batch,
        'net_charge': net_charge_batch,
        'aromatic_content': aromatic_content_batch,
        'aliphatic_index': aliphatic_index_batch,
    }

    # Add accessibility class if RSA is available
    if 'rsa' in batch:
        rsa = batch['rsa']  # (B, L)
        accessibility_class_batch = []
        for b in range(B):
            acc_class = compute_accessibility_class(rsa[b])
            accessibility_class_batch.append(acc_class)
        labels['accessibility_class'] = torch.stack(accessibility_class_batch)

    return labels
