"""
SchNet Encoder for Protein Structures.

Continuous-filter convolutional neural network using radial basis function
expansions and interaction blocks. Invariant to rotations and translations.

Reference:
    Schutt et al. (2017). "SchNet: A continuous-filter convolutional neural
    network for modeling quantum interactions."
"""

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


class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion."""

    def __init__(self, num_rbf: int = 50, cutoff: float = 30.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        offset = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer("offset", offset)
        self.width = (offset[1] - offset[0]).item()

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (...) -> (..., num_rbf)"""
        return torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.offset) / self.width) ** 2)


class CFConv(nn.Module):
    """Continuous-filter convolution layer (SchNet interaction block)."""

    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # (B, N, D)
        rbf: torch.Tensor,         # (B, N, K, num_rbf)
        edge_index: torch.Tensor,  # (B, N, K)
    ) -> torch.Tensor:
        B, N, D = h.shape
        K = edge_index.shape[2]

        # Gather neighbor features
        idx = edge_index.unsqueeze(-1).expand(-1, -1, -1, D)
        h_j = h.unsqueeze(2).expand(-1, -1, K, -1).gather(1, idx)  # (B, N, K, D)

        # Continuous filter
        W = self.filter_net(rbf)  # (B, N, K, D)

        # Filter and aggregate
        msg = (h_j * W).sum(dim=2)  # (B, N, D)

        # Update
        return self.ln(h + self.node_net(msg))


@register_encoder("schnet")
class SchNetEncoder(BaseProteinEncoder):
    """
    SchNet encoder for protein structures.

    Uses Gaussian RBF distance expansions and continuous-filter convolution
    to learn invariant representations.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of interaction blocks
        num_rbf: Number of radial basis functions
        cutoff: Distance cutoff in Angstroms
        k_neighbors: Number of neighbors in kNN graph
    """

    bottleneck_class = BottleneckClass.C

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 50,
        cutoff: float = 30.0,
        k_neighbors: int = 30,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.cutoff = cutoff

        self.aa_embed = nn.Embedding(21, hidden_dim)
        self.rbf = GaussianRBF(num_rbf, cutoff)

        self.interactions = nn.ModuleList([
            CFConv(hidden_dim, num_rbf) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    @property
    def embedding_dim(self) -> int:
        return self._hidden_dim

    def _build_knn(self, ca: torch.Tensor, mask: torch.Tensor):
        B, N, _ = ca.shape
        K = min(self.k_neighbors, N - 1)
        dist = torch.cdist(ca, ca)
        dist.masked_fill_(~(mask.unsqueeze(2) & mask.unsqueeze(1)).bool(), float("inf"))
        dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        _, idx = dist.topk(K, dim=-1, largest=False)
        return idx

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        B, L = protein.sequence.shape
        ca = protein.coords[:, :, 1, :]  # CA atoms

        edge_index = self._build_knn(ca, protein.mask)
        K = edge_index.shape[2]

        # Compute pairwise distances for RBF
        idx_3 = edge_index.unsqueeze(-1).expand(-1, -1, -1, 3)
        ca_j = ca.unsqueeze(2).expand(-1, -1, K, -1).gather(1, idx_3)
        dist = (ca.unsqueeze(2) - ca_j).norm(dim=-1)  # (B, N, K)
        rbf = self.rbf(dist)  # (B, N, K, num_rbf)

        h = self.aa_embed(protein.sequence)

        for interaction in self.interactions:
            h = interaction(h, rbf, edge_index)

        h = self.output_proj(h)
        h = h * protein.mask.unsqueeze(-1).float()

        global_emb = h.sum(dim=1) / protein.mask.sum(dim=1, keepdim=True).float().clamp(min=1)

        return EncoderOutput(residue_embeddings=h, global_embedding=global_emb)

    def load_pretrained(self, path: str):
        if path and path != "none":
            self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
