"""
GearNet encoder wrapper.

GearNet is a Geometry-Aware Relational Graph Neural Network that uses
multiple edge types (sequential, radius, k-NN) for protein structure encoding.

Reference:
    Zhang et al. (2022). "Protein Representation Learning by Geometric Structure Pretraining"
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)


class RelationalConv(nn.Module):
    """
    Relational graph convolution layer.
    
    Different edge types have different learnable transformations.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # One weight matrix per relation type
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
        self.root_weight = nn.Linear(in_dim, out_dim, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.root_weight.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,           # (B, N, D)
        edge_index: torch.Tensor,  # (B, N, K)
        edge_type: torch.Tensor,   # (B, N, K) - values in [0, num_relations)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (B, N, D_in)
            edge_index: Neighbor indices (B, N, K)
            edge_type: Edge type indices (B, N, K)
            mask: Valid node mask (B, N)
            
        Returns:
            Updated node features (B, N, D_out)
        """
        B, N, D = x.shape
        K = edge_index.shape[2]

        # Gather neighbor features using proper indexing
        # x shape: (B, N, D), edge_index shape: (B, N, K)
        # x_j[b,i,k] = x[b, edge_index[b,i,k]]
        idx = edge_index.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, N, K, D)
        x_expanded = x.unsqueeze(2).expand(-1, -1, K, -1)  # (B, N, K, D)
        x_j = x_expanded.gather(1, idx)  # (B, N, K, D)

        # Apply relation-specific weights
        # For each edge, select the appropriate weight matrix
        out = torch.zeros(B, N, K, self.out_dim, device=x.device)
        
        for r in range(self.num_relations):
            r_mask = (edge_type == r).float().unsqueeze(-1)  # (B, N, K, 1)
            msg = torch.matmul(x_j, self.weight[r])  # (B, N, K, out_dim)
            out = out + r_mask * msg

        # Aggregate (sum over neighbors)
        out = out.sum(dim=2)  # (B, N, out_dim)

        # Add root (self-loop)
        out = out + self.root_weight(x)

        if self.bias is not None:
            out = out + self.bias

        return out


@register_encoder("gearnet")
class GearNetEncoder(BaseProteinEncoder):
    """
    GearNet encoder for protein structure embeddings.

    Uses relational graph neural networks with multiple edge types:
    - Sequential edges (i, i+1)
    - Radius edges (within distance threshold)
    - K-NN edges (k nearest neighbors)

    Args:
        hidden_dim: Hidden dimension for node features
        num_layers: Number of relational GNN layers
        num_relations: Number of edge types
        k_neighbors: Number of k-NN neighbors
        radius: Distance threshold for radius edges
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_relations: int = 3,
        k_neighbors: int = 10,
        radius: float = 10.0,
        dropout: float = 0.2,
        seed: int = None,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.seed = seed

        # Set seed BEFORE initializing weights
        if seed is not None:
            self._seed_everything(seed)

        # Input embeddings
        self.node_embedding = nn.Sequential(
            nn.Embedding(21, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Edge feature encoder (distance)
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # Relational GNN layers
        self.layers = nn.ModuleList([
            RelationalConv(hidden_dim, hidden_dim, num_relations)
            for _ in range(num_layers)
        ])

        # Layer norms and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Output projection
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
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B

    @property
    def embedding_dim(self) -> int:
        return self._hidden_dim

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    def get_state_dict(self) -> dict:
        """Return state dict for serialization and reproducibility."""
        return self.state_dict()

    def set_state_dict(self, state_dict: dict) -> None:
        """Load a serialized encoder state."""
        self.load_state_dict(state_dict)

    def _build_multi_relational_graph(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build multi-relational graph with 3 edge types.
        
        Returns:
            edge_index: (B, N, K_total)
            edge_type: (B, N, K_total)
        """
        B, L = coords.shape[:2]
        device = coords.device
        
        # Use CA coordinates
        ca_coords = coords[:, :, 1, :]  # (B, L, 3)

        # Compute pairwise distances
        diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)  # (B, L, L, 3)
        dist = (diff ** 2).sum(dim=-1).sqrt()  # (B, L, L)

        # Initialize lists for edges
        all_indices = []
        all_types = []

        # Type 0: Sequential edges (i, i+1, i-1)
        for offset in [-1, 1]:
            indices = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            neighbor_idx = (indices + offset).clamp(0, L - 1)
            all_indices.append(neighbor_idx.unsqueeze(-1))
            all_types.append(torch.zeros(B, L, 1, device=device, dtype=torch.long))

        # Type 1: Radius edges
        K_radius = min(self.k_neighbors, L - 1)
        radius_mask = (dist < self.radius) & (dist > 0)
        radius_mask = radius_mask & mask.unsqueeze(2).expand(-1, -1, L)
        
        # Get top-k within radius
        dist_masked = dist.masked_fill(~radius_mask, float('inf'))
        _, radius_idx = dist_masked.topk(K_radius, dim=-1, largest=False)
        all_indices.append(radius_idx)
        all_types.append(torch.ones(B, L, K_radius, device=device, dtype=torch.long))

        # Type 2: K-NN edges (regardless of radius)
        K_knn = min(self.k_neighbors, L - 1)
        dist_knn = dist.clone()
        dist_knn = dist_knn.masked_fill(~mask.unsqueeze(2).expand(-1, -1, L), float('inf'))
        eye = torch.eye(L, device=device).unsqueeze(0).expand(B, -1, -1)
        dist_knn = dist_knn.masked_fill(eye.bool(), float('inf'))
        _, knn_idx = dist_knn.topk(K_knn, dim=-1, largest=False)
        all_indices.append(knn_idx)
        all_types.append(torch.full((B, L, K_knn), 2, device=device, dtype=torch.long))

        # Concatenate
        edge_index = torch.cat(all_indices, dim=-1)  # (B, L, K_total)
        edge_type = torch.cat(all_types, dim=-1)    # (B, L, K_total)

        return edge_index, edge_type

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        """
        Encode protein structure with GearNet.

        Args:
            protein: ProteinInput with coords, sequence, mask

        Returns:
            EncoderOutput with residue embeddings
        """
        B, L = protein.sequence.shape
        device = protein.coords.device

        # Initial node features
        h = self.node_embedding[0](protein.sequence)  # (B, L, D)
        h = self.node_embedding[1](h)

        # Build multi-relational graph
        edge_index, edge_type = self._build_multi_relational_graph(
            protein.coords, protein.mask
        )

        # Apply relational GNN layers
        for i, (layer, ln) in enumerate(zip(self.layers, self.layer_norms)):
            h_new = layer(h, edge_index, edge_type, protein.mask)
            h_new = F.relu(h_new)
            h_new = ln(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # Residual connection

        # Output projection
        h = self.output_proj(h)

        # Mask invalid positions
        h = h * protein.mask.unsqueeze(-1).float()

        return EncoderOutput(
            residue_embeddings=h,
            global_embedding=h.sum(dim=1) / protein.mask.sum(dim=1, keepdim=True).float(),
        )
