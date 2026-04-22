"""
Geometric feature computation for protein structures.

Computes ground truth labels for invariant probing:
- I1: Torsion angles (phi, psi, omega), local curvature
- I2: Pairwise distances, relative orientations
- I3: Contact density, solvent accessibility proxies
- I4: Contact maps, contact order
"""

import math
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import torch


def compute_dihedral(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> torch.Tensor:
    """
    Compute dihedral angle between four points.

    Args:
        p0, p1, p2, p3: Coordinates of shape (..., 3)

    Returns:
        Dihedral angle in radians, shape (...)
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    # Normal vectors to planes
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    # Normalize
    n1 = n1 / (torch.norm(n1, dim=-1, keepdim=True) + 1e-8)
    n2 = n2 / (torch.norm(n2, dim=-1, keepdim=True) + 1e-8)
    b2_norm = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)

    # Compute angle
    m1 = torch.cross(n1, b2_norm, dim=-1)
    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)

    return torch.atan2(y, x)


def compute_torsion_angles(coords: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute backbone torsion angles (phi, psi, omega).

    Args:
        coords: Backbone coordinates (B, L, 5, 3) - N, CA, C, O, CB
        mask: Valid residue mask (B, L)

    Returns:
        Dictionary with:
        - phi: (B, L) phi angles in radians (C[i-1]-N[i]-CA[i]-C[i])
        - psi: (B, L) psi angles in radians (N[i]-CA[i]-C[i]-N[i+1])
        - omega: (B, L) omega angles in radians (CA[i-1]-C[i-1]-N[i]-CA[i])
        - phi_valid: (B, L) mask for valid phi angles
        - psi_valid: (B, L) mask for valid psi angles
    """
    B, L = coords.shape[:2]
    device = coords.device
    mask = mask.bool() # Ensure boolean for bitwise ops

    # Extract backbone atoms
    N = coords[:, :, 0, :]  # (B, L, 3)
    CA = coords[:, :, 1, :]
    C = coords[:, :, 2, :]

    # Initialize outputs
    phi = torch.zeros(B, L, device=device)
    psi = torch.zeros(B, L, device=device)
    omega = torch.zeros(B, L, device=device)
    phi_valid = torch.zeros(B, L, dtype=torch.bool, device=device)
    psi_valid = torch.zeros(B, L, dtype=torch.bool, device=device)

    # Phi: C[i-1] - N[i] - CA[i] - C[i]
    # Valid for residues 1 to L-1 (need previous C)
    if L > 1:
        phi[:, 1:] = compute_dihedral(C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:])
        phi_valid[:, 1:] = mask[:, :-1] & mask[:, 1:]

    # Psi: N[i] - CA[i] - C[i] - N[i+1]
    # Valid for residues 0 to L-2 (need next N)
    if L > 1:
        psi[:, :-1] = compute_dihedral(N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:])
        psi_valid[:, :-1] = mask[:, :-1] & mask[:, 1:]

    # Omega: CA[i-1] - C[i-1] - N[i] - CA[i]
    if L > 1:
        omega[:, 1:] = compute_dihedral(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:])

    return {
        "phi": phi,
        "psi": psi,
        "omega": omega,
        "phi_valid": phi_valid,
        "psi_valid": psi_valid,
    }


def bin_angles(angles: torch.Tensor, num_bins: int = 36) -> torch.Tensor:
    """
    Bin angles into discrete classes.

    Args:
        angles: Angles in radians (any shape)
        num_bins: Number of bins (default 36 = 10 degree bins)

    Returns:
        Bin indices 0 to num_bins-1
    """
    # Convert to degrees and shift to [0, 360)
    degrees = (angles * 180 / math.pi) % 360
    bin_size = 360 / num_bins
    return (degrees / bin_size).long().clamp(0, num_bins - 1)


def compute_pairwise_distances(
    coords: torch.Tensor, mask: torch.Tensor, atom_idx: int = 1
) -> torch.Tensor:
    """
    Compute pairwise distances between specified atoms.

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        atom_idx: Atom index (0=N, 1=CA, 2=C, 3=O, 4=CB)

    Returns:
        Distance matrix (B, L, L) in Angstroms
    """
    atoms = coords[:, :, atom_idx, :]  # (B, L, 3)
    # Compute pairwise distances
    diff = atoms.unsqueeze(2) - atoms.unsqueeze(1)  # (B, L, L, 3)
    distances = torch.norm(diff, dim=-1)  # (B, L, L)

    # Mask invalid pairs
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, L, L)
    distances = distances * pair_mask.float()

    return distances


def compute_contact_map(
    coords: torch.Tensor, mask: torch.Tensor, threshold: float = 8.0, atom_idx: int = 4
) -> torch.Tensor:
    """
    Compute binary contact map.

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        threshold: Distance threshold in Angstroms (default 8.0 for CB-CB)
        atom_idx: Atom index for distance calculation (default 4=CB)

    Returns:
        Binary contact map (B, L, L)
    """
    distances = compute_pairwise_distances(coords, mask, atom_idx)
    contacts = (distances < threshold) & (distances > 0)  # Exclude self-contacts

    # Mask invalid pairs
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    contacts = contacts & pair_mask

    return contacts


def compute_contact_density(
    coords: torch.Tensor, mask: torch.Tensor, threshold: Union[float, List[float]] = 8.0
) -> torch.Tensor:
    """
    Compute local contact density per residue.
    Can compute for multiple thresholds.

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        threshold: Contact threshold(s) in Angstroms

    Returns:
        Contact count per residue (B, L) or (B, L, num_thresholds)
    """
    device = coords.device
    
    if isinstance(threshold, float) or isinstance(threshold, int):
        thresholds = [float(threshold)]
        return_single = True
    else:
        thresholds = threshold
        return_single = False
        
    B, L = coords.shape[:2]
    CA = coords[:, :, 1, :]
    
    # Pairwise distances
    dists = torch.cdist(CA, CA)
    
    # Mask invalid pairs
    mask_bool = mask.bool()
    valid = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
    dists.masked_fill_(~valid, float('inf'))
    
    # Exclude sequence neighbors (|i-j| <= 3)
    idx = torch.arange(L, device=device)
    seq_sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    seq_mask = seq_sep <= 3
    dists.masked_fill_(seq_mask.unsqueeze(0), float('inf'))
    
    counts = []
    for t in thresholds:
        count = (dists < t).sum(dim=-1).float()
        counts.append(count)
        
    if return_single:
        return counts[0]
    
    return torch.stack(counts, dim=-1)


def compute_local_curvature(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute local backbone curvature using CA trace.

    Uses angle between consecutive CA-CA vectors.

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)

    Returns:
        Curvature values (B, L) in radians
    """
    B, L = coords.shape[:2]
    device = coords.device

    CA = coords[:, :, 1, :]  # (B, L, 3)

    # Compute CA-CA vectors
    if L < 3:
        return torch.zeros(B, L, device=device)

    v1 = CA[:, 1:-1] - CA[:, :-2]  # (B, L-2, 3)
    v2 = CA[:, 2:] - CA[:, 1:-1]  # (B, L-2, 3)

    # Normalize
    v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
    v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)

    # Angle between consecutive vectors
    cos_angle = (v1_norm * v2_norm).sum(dim=-1).clamp(-1, 1)
    angles = torch.acos(cos_angle)  # (B, L-2)

    # Pad to full length
    curvature = torch.zeros(B, L, device=device)
    curvature[:, 1:-1] = angles

    return curvature


def compute_relative_orientations(
    coords: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute relative orientations between residue local frames.

    Uses backbone atoms to define local coordinate frames.

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)

    Returns:
        Tuple of:
        - rotation_features: (B, L, L, 9) flattened rotation matrices
        - translation_features: (B, L, L, 3) relative translations
    """
    B, L = coords.shape[:2]
    device = coords.device

    N = coords[:, :, 0, :]
    CA = coords[:, :, 1, :]
    C = coords[:, :, 2, :]

    # Build local frames (x: CA->C, y: perpendicular in plane, z: cross product)
    x = C - CA
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

    # y is in the N-CA-C plane, perpendicular to x
    n_ca = N - CA
    y = n_ca - (n_ca * x).sum(dim=-1, keepdim=True) * x
    y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)

    # z is cross product
    z = torch.cross(x, y, dim=-1)

    # Stack to form rotation matrices (B, L, 3, 3)
    frames = torch.stack([x, y, z], dim=-1)  # (B, L, 3, 3)

    # Compute relative rotations: R_ij = R_j^T @ R_i
    frames_i = frames.unsqueeze(2)  # (B, L, 1, 3, 3)
    frames_j = frames.unsqueeze(1)  # (B, 1, L, 3, 3)
    rel_rot = torch.matmul(frames_j.transpose(-2, -1), frames_i)  # (B, L, L, 3, 3)
    rotation_features = rel_rot.reshape(B, L, L, 9)

    # Compute relative translations in local frame j
    ca_i = CA.unsqueeze(2)  # (B, L, 1, 3)
    ca_j = CA.unsqueeze(1)  # (B, 1, L, 3)
    diff = ca_i - ca_j  # (B, L, L, 3)
    translation_features = torch.matmul(frames_j.transpose(-2, -1), diff.unsqueeze(-1)).squeeze(-1)

    return rotation_features, translation_features


def compute_contact_order(coords: torch.Tensor, mask: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    """
    Compute contact order (average sequence separation of contacts).

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        threshold: Contact threshold in Angstroms

    Returns:
        Contact order values (B,)
    """
    B, L = coords.shape[:2]
    device = coords.device

    contacts = compute_contact_map(coords, mask, threshold)

    # Compute sequence separation
    seq_idx = torch.arange(L, device=device)
    seq_sep = torch.abs(seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1)).float()  # (L, L)
    seq_sep = seq_sep.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)

    # Only count contacts with |i-j| > 3
    valid_contacts = contacts & (seq_sep > 3)

    # Compute mean sequence separation
    total_sep = (seq_sep * valid_contacts.float()).sum(dim=(1, 2))
    num_contacts = valid_contacts.sum(dim=(1, 2)).float().clamp(min=1)
    lengths = mask.sum(dim=1).float()

    contact_order = total_sep / (num_contacts * lengths)

    return contact_order


def compute_long_range_contacts(
    coords: torch.Tensor, mask: torch.Tensor, min_seq_sep: int = 24, threshold: float = 8.0
) -> torch.Tensor:
    """
    Compute long-range contact map (|i-j| >= min_seq_sep).

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        min_seq_sep: Minimum sequence separation (default 24)
        threshold: Distance threshold in Angstroms

    Returns:
        Binary long-range contact map (B, L, L)
    """
    B, L = coords.shape[:2]
    device = coords.device

    contacts = compute_contact_map(coords, mask, threshold)

    # Compute sequence separation mask
    seq_idx = torch.arange(L, device=device)
    seq_sep = torch.abs(seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1))
    seq_sep = seq_sep.unsqueeze(0).expand(B, -1, -1)

    long_range = contacts & (seq_sep >= min_seq_sep)

    return long_range


def bin_distances(distances: torch.Tensor, num_bins: int = 16, max_dist: float = 40.0) -> torch.Tensor:
    """
    Bin distances into discrete classes.

    Args:
        distances: Distance values in Angstroms
        num_bins: Number of bins
        max_dist: Maximum distance (distances above this go to last bin)

    Returns:
        Bin indices 0 to num_bins-1
    """
    bin_size = max_dist / num_bins
    bins = (distances / bin_size).long().clamp(0, num_bins - 1)
    return bins


def compute_solvent_accessibility_proxy(
    coords: torch.Tensor, mask: torch.Tensor, threshold: float = 10.0
) -> torch.Tensor:
    """
    Compute proxy for solvent accessibility based on contact count.

    High contact count = buried (core), low = exposed (surface).

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        threshold: Contact threshold in Angstroms

    Returns:
        Inverse contact density (B, L) - higher = more exposed
    """
    contact_density = compute_contact_density(coords, mask, threshold)
    # Normalize by max contacts to get burial fraction
    max_contacts = contact_density.max(dim=1, keepdim=True)[0].clamp(min=1)
    burial = contact_density / max_contacts

    # Invert: high value = exposed
    accessibility_proxy = 1.0 - burial

    return accessibility_proxy


def compute_topology_features(
    coords: torch.Tensor, mask: torch.Tensor, thresholds: List[float] = [8.0, 10.0]
) -> torch.Tensor:
    """
    Compute topology features (long-range degree).
    
    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        thresholds: List of distance thresholds
        
    Returns:
        Tensor (B, L, num_feats) where num_feats = 2 * len(thresholds)
        (deg_med, deg_long for each threshold)
    """
    device = coords.device
    B, L = coords.shape[:2]
    CA = coords[:, :, 1, :]
    
    # Pairwise distances
    dists = torch.cdist(CA, CA)
    
    # Mask invalid pairs
    mask_bool = mask.bool()
    valid = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
    dists.masked_fill_(~valid, float('inf'))
    
    # Sequence separation
    idx = torch.arange(L, device=device)
    seq_sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() # (L, L)
    seq_sep = seq_sep.unsqueeze(0).expand(B, -1, -1)
    
    # Masks for regimes
    mask_med = (seq_sep >= 12) & (seq_sep < 24)
    mask_long = (seq_sep >= 24)
    
    feats = []
    
    for t in thresholds:
        is_neighbor = dists < t
        
        # Deg Med
        deg_med = (is_neighbor & mask_med).sum(dim=-1).float()
        feats.append(deg_med)
        
        # Deg Long
        deg_long = (is_neighbor & mask_long).sum(dim=-1).float()
        feats.append(deg_long)
        
    return torch.stack(feats, dim=-1)


def classify_core_surface(
    coords: torch.Tensor, mask: torch.Tensor, core_threshold: float = 0.3, surface_threshold: float = 0.7
) -> torch.Tensor:
    """
    Classify residues as core (0), boundary (1), or surface (2).

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        core_threshold: Below this accessibility = core
        surface_threshold: Above this accessibility = surface

    Returns:
        Class labels (B, L) with 0=core, 1=boundary, 2=surface
    """
    accessibility = compute_solvent_accessibility_proxy(coords, mask)

    labels = torch.ones_like(accessibility, dtype=torch.long)  # Default: boundary
    labels[accessibility < core_threshold] = 0  # Core
    labels[accessibility > surface_threshold] = 2  # Surface

    return labels


def compute_residue_level_invariants(
    coords: torch.Tensor, mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute residue-level geometric invariants (fast, suitable for preprocessing).

    Does NOT compute pairwise features (O(L²)) - those should be computed
    on-the-fly during probing to save storage and preprocessing time.

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)

    Returns:
        Dictionary containing residue-level invariants:
        - I1: phi, psi, omega, phi_bins, psi_bins, curvature
        - I3: contact_density, core_surface_labels, accessibility_proxy
        - I4: contact_order (scalar per protein)
    """
    results = {}

    # I1: Local conformational (all O(L))
    torsion = compute_torsion_angles(coords, mask)
    results["phi"] = torsion["phi"]
    results["psi"] = torsion["psi"]
    results["omega"] = torsion["omega"]
    results["phi_valid"] = torsion["phi_valid"]
    results["psi_valid"] = torsion["psi_valid"]
    results["phi_bins"] = bin_angles(torsion["phi"])
    results["psi_bins"] = bin_angles(torsion["psi"])
    results["curvature"] = compute_local_curvature(coords, mask)

    # I3: Packing & neighborhood (uses pairwise internally but returns O(L))
    results["contact_density"] = compute_contact_density(coords, mask)
    results["core_surface_labels"] = classify_core_surface(coords, mask)
    results["accessibility_proxy"] = compute_solvent_accessibility_proxy(coords, mask)

    # I4: Contact order (scalar per protein)
    results["contact_order"] = compute_contact_order(coords, mask)

    return results


def compute_pairwise_invariants(
    coords: torch.Tensor, mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute pairwise geometric invariants (O(L²), compute on-the-fly during probing).

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)

    Returns:
        Dictionary containing pairwise invariants:
        - I2: ca_distances, cb_distances, contacts, distance_bins,
              relative_rotations, relative_translations
    """
    results = {}

    # I2: Metric geometry
    results["ca_distances"] = compute_pairwise_distances(coords, mask, atom_idx=1)
    results["cb_distances"] = compute_pairwise_distances(coords, mask, atom_idx=4)
    results["contacts"] = compute_contact_map(coords, mask, threshold=8.0)
    results["distance_bins"] = bin_distances(results["cb_distances"])

    rot_feat, trans_feat = compute_relative_orientations(coords, mask)
    results["relative_rotations"] = rot_feat
    results["relative_translations"] = trans_feat

    return results


def compute_all_invariants(
    coords: torch.Tensor, mask: torch.Tensor, include_pairwise: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute all geometric invariant labels for a batch of proteins.

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        include_pairwise: Whether to include O(L²) pairwise features

    Returns:
        Dictionary containing all invariant labels:
        - I1: phi, psi, omega, phi_bins, psi_bins, curvature
        - I2: (if include_pairwise) ca_distances, cb_distances, contacts, distance_bins
        - I3: contact_density, core_surface_labels, accessibility_proxy
        - I4: contact_order
    """
    results = compute_residue_level_invariants(coords, mask)

    if include_pairwise:
        pairwise = compute_pairwise_invariants(coords, mask)
        results.update(pairwise)

    return results
