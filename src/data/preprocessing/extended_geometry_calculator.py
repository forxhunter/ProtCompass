"""
Extended geometric feature computation for protein structures.

Computes ground truth labels for invariant families I5-I8:
- I5: Sequence-Structure Relationship (amino acid identity, hydrophobicity)
- I6: Global Shape & Composition (radius of gyration, SS fractions, asphericity)
- I7: Flexibility & Dynamics Proxy (B-factor proxy, backbone entropy proxy)
- I8: Electrostatics & Chemistry (charge distribution, hydrogen bond count)
"""

import math
from typing import Dict, Tuple

import torch

from src.data.preprocessing.geometry_calculator import compute_torsion_angles


# ============================================================================
# I5: Sequence-Structure Relationship
# ============================================================================

# Kyte-Doolittle hydrophobicity scale (standard 20 AA, indexed 0-19)
# Order: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
KYTE_DOOLITTLE = torch.tensor([
    1.8,   # 0  A (Ala)
    2.5,   # 1  C (Cys)
   -3.5,   # 2  D (Asp)
   -3.5,   # 3  E (Glu)
    2.8,   # 4  F (Phe)
   -0.4,   # 5  G (Gly)
   -3.2,   # 6  H (His)
    4.5,   # 7  I (Ile)
   -3.9,   # 8  K (Lys)
    3.8,   # 9  L (Leu)
    1.9,   # 10 M (Met)
   -3.5,   # 11 N (Asn)
   -1.6,   # 12 P (Pro)
   -3.5,   # 13 Q (Gln)
   -4.5,   # 14 R (Arg)
   -0.8,   # 15 S (Ser)
   -0.7,   # 16 T (Thr)
    4.2,   # 17 V (Val)
   -0.9,   # 18 W (Trp)
   -1.3,   # 19 Y (Tyr)
], dtype=torch.float32)


def compute_amino_acid_identity(
    sequence: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Return amino acid identity labels for inverse folding probing.

    Args:
        sequence: (B, L) amino acid indices 0-19
        mask: (B, L) valid residue mask

    Returns:
        (B, L) amino acid class labels (0-19), masked invalid to -1
    """
    labels = sequence.clone().long()
    labels = labels.clamp(0, 19)
    labels[~mask.bool()] = -1
    return labels


def compute_hydrophobicity(
    sequence: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-residue hydrophobicity from Kyte-Doolittle scale.

    Args:
        sequence: (B, L) amino acid indices 0-19
        mask: (B, L)

    Returns:
        (B, L) hydrophobicity values (continuous, range ~[-4.5, 4.5])
    """
    lookup = KYTE_DOOLITTLE.to(sequence.device)
    seq_clamped = sequence.clamp(0, 19).long()
    hydro = lookup[seq_clamped]  # (B, L)
    hydro = hydro * mask.float()
    return hydro


# Molecular weight lookup (Da) - indexed 0-19
MOLECULAR_WEIGHT = torch.tensor([
    89.1,   # 0  A
    121.2,  # 1  C
    133.1,  # 2  D
    147.1,  # 3  E
    165.2,  # 4  F
    75.1,   # 5  G
    155.2,  # 6  H
    131.2,  # 7  I
    146.2,  # 8  K
    131.2,  # 9  L
    149.2,  # 10 M
    132.1,  # 11 N
    115.1,  # 12 P
    146.2,  # 13 Q
    174.2,  # 14 R
    105.1,  # 15 S
    119.1,  # 16 T
    117.1,  # 17 V
    204.2,  # 18 W
    181.2,  # 19 Y
], dtype=torch.float32)

# Net charge at pH 7 - indexed 0-19
NET_CHARGE = torch.tensor([
    0.0,   # 0  A
    0.0,   # 1  C
    -1.0,  # 2  D (acidic)
    -1.0,  # 3  E (acidic)
    0.0,   # 4  F
    0.0,   # 5  G
    0.1,   # 6  H (partially protonated at pH 7)
    0.0,   # 7  I
    1.0,   # 8  K (basic)
    0.0,   # 9  L
    0.0,   # 10 M
    0.0,   # 11 N
    0.0,   # 12 P
    0.0,   # 13 Q
    1.0,   # 14 R (basic)
    0.0,   # 15 S
    0.0,   # 16 T
    0.0,   # 17 V
    0.0,   # 18 W
    0.0,   # 19 Y
], dtype=torch.float32)


def compute_molecular_weight(
    sequence: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute total molecular weight in kDa for each protein.

    Args:
        sequence: (B, L) amino acid indices 0-19
        mask: (B, L)

    Returns:
        (B,) molecular weight in kDa
    """
    lookup = MOLECULAR_WEIGHT.to(sequence.device)
    seq_clamped = sequence.clamp(0, 19).long()
    weights = lookup[seq_clamped]  # (B, L)

    # Sum over valid residues
    total_weight = (weights * mask.float()).sum(dim=1)  # (B,)

    # Subtract water molecules lost in peptide bonds
    n_residues = mask.float().sum(dim=1)  # (B,)
    n_bonds = (n_residues - 1).clamp(min=0)
    total_weight = total_weight - 18.015 * n_bonds

    # Convert to kDa
    return total_weight / 1000.0


def compute_net_charge(
    sequence: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute net charge at pH 7 for each protein.

    Args:
        sequence: (B, L) amino acid indices 0-19
        mask: (B, L)

    Returns:
        (B,) net charge
    """
    lookup = NET_CHARGE.to(sequence.device)
    seq_clamped = sequence.clamp(0, 19).long()
    charges = lookup[seq_clamped]  # (B, L)

    # Sum over valid residues
    total_charge = (charges * mask.float()).sum(dim=1)  # (B,)

    # Add terminal charges (N-terminus +1, C-terminus -1)
    # Net effect: 0 (they cancel out)

    return total_charge


def compute_aromatic_content(
    sequence: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute fraction of aromatic residues (F, W, Y).

    Args:
        sequence: (B, L) amino acid indices 0-19
        mask: (B, L)

    Returns:
        (B,) aromatic content fraction [0, 1]
    """
    seq_clamped = sequence.clamp(0, 19).long()

    # Aromatic: F=4, W=18, Y=19
    is_aromatic = ((seq_clamped == 4) | (seq_clamped == 18) | (seq_clamped == 19)).float()

    aromatic_count = (is_aromatic * mask.float()).sum(dim=1)  # (B,)
    total_count = mask.float().sum(dim=1).clamp(min=1)  # (B,)

    return aromatic_count / total_count


def compute_aliphatic_index(
    sequence: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute Ikai's aliphatic index.

    AI = X_A + 2.9*X_V + 3.9*(X_I + X_L)
    where X_i is mole fraction (percentage)

    Args:
        sequence: (B, L) amino acid indices 0-19
        mask: (B, L)

    Returns:
        (B,) aliphatic index (typically 0-200)
    """
    seq_clamped = sequence.clamp(0, 19).long()
    mask_f = mask.float()
    total_count = mask_f.sum(dim=1).clamp(min=1)  # (B,)

    # Aliphatic: A=0, V=17, I=7, L=9
    count_A = ((seq_clamped == 0).float() * mask_f).sum(dim=1)
    count_V = ((seq_clamped == 17).float() * mask_f).sum(dim=1)
    count_I = ((seq_clamped == 7).float() * mask_f).sum(dim=1)
    count_L = ((seq_clamped == 9).float() * mask_f).sum(dim=1)

    # Compute index
    ai = (count_A / total_count * 100.0 +
          count_V / total_count * 100.0 * 2.9 +
          (count_I + count_L) / total_count * 100.0 * 3.9)

    return ai


def compute_accessibility_class(
    coords: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute 3-class accessibility labels from approximate RSA.

    Classes:
        0 = buried (RSA < 0.25)
        1 = intermediate (0.25 <= RSA < 0.5)
        2 = exposed (RSA >= 0.5)

    Uses a simple distance-based approximation:
    - Count neighbors within 10Å of CA
    - Normalize by max possible neighbors
    - Convert to RSA-like score

    Args:
        coords: (B, L, 5, 3) backbone coords
        mask: (B, L)

    Returns:
        (B, L) accessibility class labels {0, 1, 2}
    """
    ca = coords[:, :, 1, :]  # (B, L, 3)
    B, L, _ = ca.shape

    # Compute pairwise distances
    ca_i = ca.unsqueeze(2)  # (B, L, 1, 3)
    ca_j = ca.unsqueeze(1)  # (B, 1, L, 3)
    dist = torch.norm(ca_i - ca_j, dim=-1)  # (B, L, L)

    # Count neighbors within 10Å (excluding self)
    mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, L, L)
    neighbor_mask = (dist < 10.0) & (dist > 0.1) & mask_2d
    neighbor_count = neighbor_mask.float().sum(dim=2)  # (B, L)

    # Normalize by max possible (use 30 as typical max for buried residues)
    rsa_approx = 1.0 - (neighbor_count / 30.0).clamp(0, 1)

    # Convert to 3 classes
    classes = torch.zeros_like(rsa_approx, dtype=torch.long)
    classes[rsa_approx >= 0.25] = 1
    classes[rsa_approx >= 0.5] = 2
    classes[~mask] = -1  # Mask invalid positions

    return classes


# ============================================================================
# I6: Global Shape & Composition
# ============================================================================

def compute_radius_of_gyration(
    coords: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute radius of gyration for each protein.

    R_g = sqrt(1/N * sum_i ||r_i - r_center||^2)

    Args:
        coords: (B, L, 5, 3) backbone coords [N, CA, C, O, CB]
        mask: (B, L) valid residue mask

    Returns:
        (B,) radius of gyration values
    """
    ca = coords[:, :, 1, :]  # (B, L, 3) - CA coordinates
    mask_f = mask.float()
    n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

    # Center of mass (masked)
    center = (ca * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)  # (B, 1, 3)

    # Squared distances from center
    diff = ca - center  # (B, L, 3)
    sq_dist = (diff ** 2).sum(dim=-1)  # (B, L)
    sq_dist = sq_dist * mask_f

    # RMS
    rg = torch.sqrt(sq_dist.sum(dim=1) / n_valid.squeeze(-1))  # (B,)
    return rg


def compute_ss_content_fractions(
    coords: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute secondary structure content fractions via Ramachandran region proxy.

    Helix: phi in [-80, -48] deg, psi in [-59, -27] deg
    Sheet: phi in [-150, -100] deg, psi in [100, 170] deg
    Coil: everything else

    Args:
        coords: (B, L, 5, 3) backbone coords
        mask: (B, L) valid residue mask

    Returns:
        (B, 3) fractions [helix_frac, sheet_frac, coil_frac]
    """
    torsions = compute_torsion_angles(coords, mask)
    phi = torsions["phi"]  # (B, L) radians
    psi = torsions["psi"]  # (B, L) radians
    phi_valid = torsions["phi_valid"]  # (B, L)
    psi_valid = torsions["psi_valid"]  # (B, L)

    # Convert boundaries to radians
    phi_deg = phi * (180.0 / math.pi)
    psi_deg = psi * (180.0 / math.pi)

    valid = phi_valid & psi_valid & mask.bool()
    valid_f = valid.float()
    n_valid = valid_f.sum(dim=1).clamp(min=1)  # (B,)

    # Helix region
    helix = (
        (phi_deg >= -80) & (phi_deg <= -48) &
        (psi_deg >= -59) & (psi_deg <= -27) &
        valid
    ).float()

    # Sheet region
    sheet = (
        (phi_deg >= -150) & (phi_deg <= -100) &
        (psi_deg >= 100) & (psi_deg <= 170) &
        valid
    ).float()

    helix_frac = helix.sum(dim=1) / n_valid  # (B,)
    sheet_frac = sheet.sum(dim=1) / n_valid
    coil_frac = 1.0 - helix_frac - sheet_frac

    return torch.stack([helix_frac, sheet_frac, coil_frac], dim=-1)  # (B, 3)


def compute_asphericity(
    coords: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute asphericity from gyration tensor eigenvalues.

    Asphericity = (l1 - l3)^2 / (l1 + l2 + l3)^2
    where l1 >= l2 >= l3 are eigenvalues of the gyration tensor.

    Range: [0, 1] where 0 = perfect sphere, 1 = rod-like.

    Args:
        coords: (B, L, 5, 3)
        mask: (B, L)

    Returns:
        (B,) asphericity values
    """
    ca = coords[:, :, 1, :]  # (B, L, 3)
    mask_f = mask.float()
    n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

    # Center of mass
    center = (ca * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)
    diff = (ca - center) * mask_f.unsqueeze(-1)  # (B, L, 3)

    # Gyration tensor: S_ij = 1/N * sum_k diff_ki * diff_kj
    # (B, 3, L) @ (B, L, 3) = (B, 3, 3)
    S = torch.bmm(diff.transpose(1, 2), diff) / n_valid.unsqueeze(-1)

    # Eigenvalues (symmetric matrix)
    eigenvalues = torch.linalg.eigvalsh(S)  # (B, 3), sorted ascending
    l3, l2, l1 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]

    trace = (l1 + l2 + l3).clamp(min=1e-8)
    asphericity = (l1 - l3) ** 2 / (trace ** 2)

    return asphericity  # (B,)


# ============================================================================
# I7: Flexibility & Dynamics Proxy
# ============================================================================

def compute_bfactor_proxy(
    coords: torch.Tensor, mask: torch.Tensor, radius: float = 10.0
) -> torch.Tensor:
    """
    Compute B-factor proxy from local packing distance variance.

    For each residue, compute the variance of CA-CA distances to neighbors
    within a radius. High variance indicates a flexible region.

    Args:
        coords: (B, L, 5, 3) backbone coords
        mask: (B, L)
        radius: Distance cutoff in Angstroms

    Returns:
        (B, L) B-factor proxy values (normalized per protein)
    """
    ca = coords[:, :, 1, :]  # (B, L, 3)
    B, L, _ = ca.shape
    mask_f = mask.float()

    # Pairwise CA distances: (B, L, L)
    diff = ca.unsqueeze(2) - ca.unsqueeze(1)  # (B, L, L, 3)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (B, L, L)

    # Mask: valid pairs within radius, exclude self and seq neighbors
    seq_idx = torch.arange(L, device=ca.device)
    seq_sep = (seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1)).abs()  # (L, L)
    pair_mask = (
        (dist < radius) &
        (seq_sep.unsqueeze(0) > 3) &
        mask.unsqueeze(2).bool() &
        mask.unsqueeze(1).bool()
    ).float()  # (B, L, L)

    # Mean and variance of distances per residue
    n_neighbors = pair_mask.sum(dim=2).clamp(min=1)  # (B, L)
    mean_dist = (dist * pair_mask).sum(dim=2) / n_neighbors
    sq_diff = ((dist - mean_dist.unsqueeze(2)) ** 2) * pair_mask
    var_dist = sq_diff.sum(dim=2) / n_neighbors  # (B, L)

    # Normalize per protein (zero-mean, unit-var)
    var_masked = var_dist * mask_f
    protein_mean = var_masked.sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1)
    centered = (var_masked - protein_mean) * mask_f
    protein_std = torch.sqrt(
        (centered ** 2).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1) + 1e-8
    )
    normalized = centered / protein_std.clamp(min=1e-8) * mask_f

    return normalized  # (B, L)


# Ramachandran basin centers (in degrees) for entropy proxy
RAMA_BASINS = torch.tensor([
    [-63.0, -43.0],   # alpha-helix
    [-120.0, 135.0],  # beta-sheet
    [-60.0, -30.0],   # left-handed helix (approx)
    [60.0, 40.0],     # left-handed alpha
], dtype=torch.float32)


def compute_backbone_entropy_proxy(
    coords: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute backbone conformational entropy proxy.

    For each residue, compute the minimum angular distance from its (phi, psi)
    to the nearest Ramachandran basin center. Residues far from any basin
    center are in flexible/disordered regions (higher entropy).

    Args:
        coords: (B, L, 5, 3)
        mask: (B, L)

    Returns:
        (B, L) entropy proxy values (higher = more flexible)
    """
    torsions = compute_torsion_angles(coords, mask)
    phi = torsions["phi"] * (180.0 / math.pi)  # (B, L) degrees
    psi = torsions["psi"] * (180.0 / math.pi)
    phi_valid = torsions["phi_valid"]
    psi_valid = torsions["psi_valid"]

    valid = phi_valid & psi_valid & mask.bool()
    mask_f = valid.float()

    basins = RAMA_BASINS.to(coords.device)  # (4, 2)
    n_basins = basins.shape[0]

    # Angular distance (circular) to each basin
    # phi: (B, L, 1), basins: (1, 1, 4)
    phi_diff = phi.unsqueeze(-1) - basins[:, 0].view(1, 1, n_basins)
    psi_diff = psi.unsqueeze(-1) - basins[:, 1].view(1, 1, n_basins)

    # Wrap to [-180, 180]
    phi_diff = (phi_diff + 180.0) % 360.0 - 180.0
    psi_diff = (psi_diff + 180.0) % 360.0 - 180.0

    # Euclidean distance in angle space
    angular_dist = torch.sqrt(phi_diff ** 2 + psi_diff ** 2 + 1e-8)  # (B, L, 4)

    # Minimum distance to any basin
    min_dist, _ = angular_dist.min(dim=-1)  # (B, L)
    min_dist = min_dist * mask_f

    # Normalize per protein
    protein_mean = min_dist.sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1)
    protein_std = torch.sqrt(
        ((min_dist - protein_mean) ** 2 * mask_f).sum(dim=1, keepdim=True)
        / mask_f.sum(dim=1, keepdim=True).clamp(min=1) + 1e-8
    )
    normalized = (min_dist - protein_mean) / protein_std.clamp(min=1e-8) * mask_f

    return normalized  # (B, L)


# ============================================================================
# I8: Electrostatics & Chemistry
# ============================================================================

# Formal charges at pH 7.0 (indexed 0-19)
AA_CHARGES = torch.tensor([
    0.0,   # 0  A
    0.0,   # 1  C
   -1.0,   # 2  D (Asp)
   -1.0,   # 3  E (Glu)
    0.0,   # 4  F
    0.0,   # 5  G
    0.0,   # 6  H (pKa ~6.0, partially protonated)
    0.0,   # 7  I
    1.0,   # 8  K (Lys)
    0.0,   # 9  L
    0.0,   # 10 M
    0.0,   # 11 N
    0.0,   # 12 P
    0.0,   # 13 Q
    1.0,   # 14 R (Arg)
    0.0,   # 15 S
    0.0,   # 16 T
    0.0,   # 17 V
    0.0,   # 18 W
    0.0,   # 19 Y
], dtype=torch.float32)


def compute_charge_distribution(
    sequence: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-residue formal charge at pH 7.0.

    Args:
        sequence: (B, L) amino acid indices 0-19
        mask: (B, L)

    Returns:
        (B, L) charge values (-1, 0, or +1)
    """
    lookup = AA_CHARGES.to(sequence.device)
    seq_clamped = sequence.clamp(0, 19).long()
    charges = lookup[seq_clamped]
    charges = charges * mask.float()
    return charges


def compute_hydrogen_bond_count(
    coords: torch.Tensor, mask: torch.Tensor,
    dist_cutoff: float = 3.5, angle_cutoff: float = 120.0, min_seq_sep: int = 2
) -> torch.Tensor:
    """
    Compute approximate backbone hydrogen bond count per residue.

    An H-bond exists between residue i's N and residue j's O if:
    - d(N_i, O_j) < dist_cutoff Angstroms
    - angle(C_j, O_j, N_i) > angle_cutoff degrees
    - |i - j| >= min_seq_sep

    Args:
        coords: (B, L, 5, 3) atom order: N(0), CA(1), C(2), O(3), CB(4)
        mask: (B, L)
        dist_cutoff: Maximum N-O distance
        angle_cutoff: Minimum C-O-N angle (degrees)
        min_seq_sep: Minimum sequence separation

    Returns:
        (B, L) hydrogen bond count per residue (as donor + acceptor)
    """
    B, L = mask.shape
    N = coords[:, :, 0, :]   # (B, L, 3) - backbone N
    C = coords[:, :, 2, :]   # (B, L, 3) - backbone C
    O = coords[:, :, 3, :]   # (B, L, 3) - backbone O

    # Pairwise N_i - O_j distances: (B, L_i, L_j)
    diff_no = N.unsqueeze(2) - O.unsqueeze(1)  # (B, L, L, 3)
    dist_no = torch.sqrt((diff_no ** 2).sum(dim=-1) + 1e-8)  # (B, L, L)

    # Sequence separation mask
    seq_idx = torch.arange(L, device=coords.device)
    seq_sep = (seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1)).abs()  # (L, L)

    # Distance + separation criterion
    dist_mask = (dist_no < dist_cutoff) & (seq_sep.unsqueeze(0) >= min_seq_sep)

    # Angle criterion: angle(C_j, O_j, N_i) > cutoff
    # Vectors: O_j -> C_j and O_j -> N_i
    vec_oc = C.unsqueeze(1) - O.unsqueeze(1)  # Need (B, L_i, L_j, 3) for j dimension
    # Reshape properly: C_j is indexed by j
    vec_oc = (C - O).unsqueeze(1).expand(-1, L, -1, -1)  # (B, L_i, L_j, 3) where j indexes C,O
    vec_on = diff_no  # N_i - O_j = vector from O_j to N_i: (B, L_i, L_j, 3)

    # Compute angle using dot product
    cos_angle = (
        (vec_oc * vec_on).sum(dim=-1) /
        (torch.norm(vec_oc, dim=-1) * torch.norm(vec_on, dim=-1) + 1e-8)
    )
    angle_deg = torch.acos(cos_angle.clamp(-1 + 1e-7, 1 - 1e-7)) * (180.0 / math.pi)

    # Combined mask
    pair_valid = mask.unsqueeze(2).bool() & mask.unsqueeze(1).bool()
    hbond_mask = dist_mask & (angle_deg > angle_cutoff) & pair_valid  # (B, L, L)

    # Count per residue: sum as donor (i) and acceptor (j)
    donor_count = hbond_mask.float().sum(dim=2)     # (B, L) - i donates N to j's O
    acceptor_count = hbond_mask.float().sum(dim=1)  # (B, L) - i accepts at O from j's N

    total = (donor_count + acceptor_count) * mask.float()
    return total  # (B, L)


# ============================================================================
# Aggregator
# ============================================================================

def compute_extended_residue_invariants(
    coords: torch.Tensor, mask: torch.Tensor, sequence: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute all I5-I8 invariants.

    Args:
        coords: (B, L, 5, 3)
        mask: (B, L)
        sequence: (B, L) amino acid indices

    Returns:
        Dictionary with all invariant labels.
    """
    result = {}

    # I5: Sequence-Structure (2 probes)
    result["amino_acid_identity"] = compute_amino_acid_identity(sequence, mask)
    result["hydrophobicity"] = compute_hydrophobicity(sequence, mask)

    # I6: Global Shape (2 probes)
    result["radius_of_gyration"] = compute_radius_of_gyration(coords, mask)
    result["asphericity"] = compute_asphericity(coords, mask)

    # I7: Flexibility (2 probes)
    result["bfactor_proxy"] = compute_bfactor_proxy(coords, mask)
    result["backbone_entropy_proxy"] = compute_backbone_entropy_proxy(coords, mask)

    # I8: Chemistry (2 probes)
    result["charge_distribution"] = compute_charge_distribution(sequence, mask)
    result["hydrogen_bond_count"] = compute_hydrogen_bond_count(coords, mask)

    return result
