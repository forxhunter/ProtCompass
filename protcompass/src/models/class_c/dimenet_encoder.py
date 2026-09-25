"""
DimeNet++ Encoder for Protein Structures.

Directional message passing with Bessel basis functions and angular
information. Captures both distance and angular relationships between atoms.

Reference:
    Klicpera et al. (2020). "Fast and Uncertainty-Aware Directional Message
    Passing for Non-Equilibrium Molecules." (DimeNet++)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)


class BesselBasis(nn.Module):
    """Bessel radial basis functions with envelope."""

    def __init__(self, num_rbf: int = 16, cutoff: float = 30.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        freq = torch.arange(1, num_rbf + 1).float() * math.pi / cutoff
        self.register_buffer("freq", freq)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (...) -> (..., num_rbf)"""
        d = dist.unsqueeze(-1)
        # Envelope: smooth cutoff
        envelope = 1.0 - 6 * (d / self.cutoff) ** 5 + 15 * (d / self.cutoff) ** 4 - 10 * (d / self.cutoff) ** 3
        envelope = envelope * (d < self.cutoff).float()
        # Bessel basis
        rbf = torch.sin(self.freq * d) / (d + 1e-8)
        return rbf * envelope


class SphericalBasis(nn.Module):
    """Simplified spherical basis using cosine of angles."""

    def __init__(self, num_sph: int = 7):
        super().__init__()
        self.num_sph = num_sph

    def forward(self, cos_angle: torch.Tensor) -> torch.Tensor:
        """cos_angle: (...) -> (..., num_sph) using Chebyshev polynomials."""
        basis = [torch.ones_like(cos_angle), cos_angle]
        for n in range(2, self.num_sph):
            basis.append(2 * cos_angle * basis[-1] - basis[-2])
        return torch.stack(basis, dim=-1)


class InteractionBlock(nn.Module):
    """DimeNet++ interaction block with angular messages."""

    def __init__(self, hidden_dim: int, num_rbf: int, num_sph: int):
        super().__init__()
        self.rbf_proj = nn.Linear(num_rbf, hidden_dim)
        self.sph_proj = nn.Linear(num_sph, hidden_dim)

        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,              # (B, N, D)
        rbf: torch.Tensor,            # (B, N, K, num_rbf)
        sph: torch.Tensor,            # (B, N, K, num_sph)
        edge_index: torch.Tensor,     # (B, N, K)
    ) -> torch.Tensor:
        B, N, D = h.shape
        K = edge_index.shape[2]

        idx = edge_index.unsqueeze(-1).expand(-1, -1, -1, D)
        h_j = h.unsqueeze(2).expand(-1, -1, K, -1).gather(1, idx)

        # Combine distance + angular info
        rbf_feat = self.rbf_proj(rbf)   # (B, N, K, D)
        sph_feat = self.sph_proj(sph)   # (B, N, K, D)

        msg = h_j * rbf_feat * sph_feat
        msg = self.msg_net(msg).sum(dim=2)  # (B, N, D)

        h_new = self.update_net(torch.cat([h, msg], dim=-1))
        return self.ln(h + h_new)


@register_encoder("dimenet")
class DimeNetEncoder(BaseProteinEncoder):
    """
    DimeNet++ encoder for protein structures.

    Uses directional message passing with Bessel RBF and angular
    information to capture both distance and angle relationships.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of interaction blocks
        num_rbf: Number of Bessel radial basis functions
        num_sph: Number of spherical basis functions
        cutoff: Distance cutoff
        k_neighbors: kNN neighbors
        seed: Random seed for reproducibility (optional)
    """

    bottleneck_class = BottleneckClass.C

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 16,
        num_sph: int = 7,
        cutoff: float = 30.0,
        k_neighbors: int = 30,
        seed: int = None,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.cutoff = cutoff
        self.seed = seed

        # Set seed BEFORE initializing weights
        if seed is not None:
            self._seed_everything(seed)

        self.aa_embed = nn.Embedding(21, hidden_dim)
        self.bessel = BesselBasis(num_rbf, cutoff)
        self.spherical = SphericalBasis(num_sph)

        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, num_rbf, num_sph) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def _seed_everything(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def embedding_dim(self) -> int:
        return self._hidden_dim

    def _build_knn(self, ca, mask):
        B, N, _ = ca.shape
        K = min(self.k_neighbors, N - 1)
        dist = torch.cdist(ca, ca)
        mask_bool = mask.bool()
        dist.masked_fill_(~(mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)), float("inf"))
        dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        _, idx = dist.topk(K, dim=-1, largest=False)
        return idx

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        B, L = protein.sequence.shape
        ca = protein.coords[:, :, 1, :]

        edge_index = self._build_knn(ca, protein.mask)
        K = edge_index.shape[2]

        # Neighbor coords
        idx_3 = edge_index.unsqueeze(-1).expand(-1, -1, -1, 3)
        ca_j = ca.unsqueeze(2).expand(-1, -1, K, -1).gather(1, idx_3)

        # Distances and RBF
        rel = ca.unsqueeze(2) - ca_j  # (B, N, K, 3)
        dist = rel.norm(dim=-1)       # (B, N, K)
        rbf = self.bessel(dist)

        # Angles: cos(angle) between center→neighbor vectors
        # Use angle between each pair of neighbor directions from the same center
        rel_normed = F.normalize(rel, dim=-1)  # (B, N, K, 3)
        # Compute mean direction as reference for angular encoding
        mean_dir = F.normalize(rel_normed.mean(dim=2, keepdim=True), dim=-1)  # (B, N, 1, 3)
        cos_angle = (rel_normed * mean_dir).sum(dim=-1)  # (B, N, K)
        sph = self.spherical(cos_angle)  # (B, N, K, num_sph)

        h = self.aa_embed(protein.sequence)

        for interaction in self.interactions:
            h = interaction(h, rbf, sph, edge_index)

        h = self.output_proj(h)
        h = h * protein.mask.unsqueeze(-1).float()

        global_emb = h.sum(dim=1) / protein.mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return EncoderOutput(residue_embeddings=h, global_embedding=global_emb)

    def load_pretrained(self, path: str):
        if path and path != "none":
            self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def get_state_dict(self) -> dict:
        """Return state dict for serialization and reproducibility."""
        return self.state_dict()

    def set_state_dict(self, state_dict: dict) -> None:
        """Load a serialized encoder state."""
        self.load_state_dict(state_dict)
