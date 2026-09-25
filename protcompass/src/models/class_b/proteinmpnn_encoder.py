"""
ProteinMPNN encoder wrapper.

ProteinMPNN uses message passing on k-nearest neighbor graphs of residues.
The encoder portion extracts structural embeddings before sequence decoding.

Reference:
    Dauparas et al. "Robust deep learning based protein sequence design using ProteinMPNN"
    https://github.com/dauparas/ProteinMPNN

Bottleneck Class: B (Neighborhood Graph)
    - Continuous geometry preserved within local k-NN residue graphs
    - Preserves I3 (packing/neighborhood) strongly
    - Moderate I2 (metric geometry) locally
    - Weak I4 (long-range topology)
"""

import math
from typing import Any, Dict, Optional, Tuple

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


def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """
    Gather edge features for k-nearest neighbors.

    Args:
        edges: Full edge features (B, L, L, D)
        neighbor_idx: Neighbor indices (B, L, K)

    Returns:
        Gathered edge features (B, L, K, D)
    """
    B, L, K = neighbor_idx.shape
    D = edges.shape[-1]

    # Expand neighbor indices for gathering
    neighbor_idx_expanded = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, D)

    # Gather edges
    edges_gathered = torch.gather(
        edges.unsqueeze(2).expand(-1, -1, K, -1, -1),
        dim=3,
        index=neighbor_idx_expanded.unsqueeze(3).expand(-1, -1, -1, L, -1),
    )

    # This is wrong, let's fix:
    # We want edges[b, i, neighbor_idx[b,i,k], :]
    neighbor_idx_flat = neighbor_idx.reshape(B, L * K)
    edges_flat = edges.reshape(B, L, L, D)

    result = torch.zeros(B, L, K, D, device=edges.device, dtype=edges.dtype)
    for b in range(B):
        for i in range(L):
            for k in range(K):
                j = neighbor_idx[b, i, k]
                if j >= 0 and j < L:
                    result[b, i, k] = edges[b, i, j]

    return result


def gather_edges_fast(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """
    Fast vectorized edge gathering.

    Args:
        edges: Full edge features (B, L, L, D)
        neighbor_idx: Neighbor indices (B, L, K)

    Returns:
        Gathered edge features (B, L, K, D)
    """
    B, L, _, D = edges.shape
    K = neighbor_idx.shape[-1]

    # Clamp indices to valid range
    neighbor_idx = neighbor_idx.clamp(0, L - 1)

    # Expand for gather: need (B, L, K, D)
    idx = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, L, K, D)

    # Expand edges to (B, L, L, D) and gather along dim 2
    gathered = edges.gather(
        dim=2,
        index=idx.view(B, L, K, D).transpose(1, 2).reshape(B, K, L, D).transpose(1, 2),
    )

    # Simpler approach: index_select style
    gathered = torch.zeros(B, L, K, D, device=edges.device, dtype=edges.dtype)
    for k in range(K):
        idx_k = neighbor_idx[:, :, k]  # (B, L)
        # Gather from edges along dim 2
        for b in range(B):
            gathered[b, :, k, :] = edges[b, torch.arange(L, device=edges.device), idx_k[b], :]

    return gathered


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for distances and angles."""

    def __init__(self, num_embeddings: int, period_range: Tuple[float, float] = (2, 1000)):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode scalar values with sinusoidal features.

        Args:
            x: Input values (any shape)

        Returns:
            Positional encodings (..., num_embeddings)
        """
        device = x.device
        dtype = x.dtype

        # Compute frequencies
        freq_bands = torch.linspace(
            math.log(self.period_range[0]),
            math.log(self.period_range[1]),
            self.num_embeddings // 2,
            device=device,
            dtype=dtype,
        )
        freq_bands = torch.exp(freq_bands)

        # Expand x for broadcasting
        x_expanded = x.unsqueeze(-1)  # (..., 1)

        # Compute sin and cos
        arg = x_expanded * 2 * math.pi / freq_bands
        encodings = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

        return encodings


class EdgeFeatures(nn.Module):
    """
    Compute edge features between residue pairs.

    Features include:
    - Pairwise distances (CA-CA, CB-CB, N-O)
    - Relative orientations
    - Sequence separation
    """

    def __init__(self, edge_dim: int = 128, num_positional: int = 16):
        super().__init__()
        self.edge_dim = edge_dim
        self.num_positional = num_positional

        # Positional encodings for distances
        self.dist_encoding = PositionalEncoding(num_positional)

        # Number of distance features: 3 distances * num_positional
        dist_features = 3 * num_positional

        # Sequence separation encoding
        self.seq_sep_encoding = nn.Embedding(65, num_positional)  # 0-64+ separation

        # Direction features (unit vectors)
        direction_features = 3  # CA->CA unit vector

        # Project to edge dimension
        total_features = dist_features + num_positional + direction_features
        self.edge_proj = nn.Linear(total_features, edge_dim)

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge features.

        Args:
            coords: Backbone coordinates (B, L, 5, 3)
            mask: Valid residue mask (B, L)

        Returns:
            Edge features (B, L, L, edge_dim)
        """
        B, L = coords.shape[:2]
        device = coords.device

        # Extract atom positions
        N = coords[:, :, 0, :]  # (B, L, 3)
        CA = coords[:, :, 1, :]
        C = coords[:, :, 2, :]
        O = coords[:, :, 3, :]
        CB = coords[:, :, 4, :]

        # Pairwise distances
        def pairwise_dist(x):
            diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B, L, L, 3)
            return torch.norm(diff, dim=-1)  # (B, L, L)

        dist_ca = pairwise_dist(CA)
        dist_cb = pairwise_dist(CB)
        dist_no = torch.norm(
            N.unsqueeze(2) - O.unsqueeze(1), dim=-1
        )  # N[i] to O[j]

        # Encode distances
        dist_features = torch.cat(
            [
                self.dist_encoding(dist_ca),
                self.dist_encoding(dist_cb),
                self.dist_encoding(dist_no),
            ],
            dim=-1,
        )  # (B, L, L, 3*num_pos)

        # Sequence separation
        seq_idx = torch.arange(L, device=device)
        seq_sep = torch.abs(seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1))  # (L, L)
        seq_sep = seq_sep.clamp(max=64).unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
        seq_sep_features = self.seq_sep_encoding(seq_sep)  # (B, L, L, num_pos)

        # Direction features (CA->CA unit vectors)
        ca_diff = CA.unsqueeze(2) - CA.unsqueeze(1)  # (B, L, L, 3)
        ca_dist = torch.norm(ca_diff, dim=-1, keepdim=True).clamp(min=1e-6)
        ca_direction = ca_diff / ca_dist  # (B, L, L, 3)

        # Concatenate all features
        edge_features = torch.cat(
            [dist_features, seq_sep_features, ca_direction], dim=-1
        )

        # Project to edge dimension
        edge_features = self.edge_proj(edge_features)

        # Mask invalid pairs
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, L, L)
        edge_features = edge_features * pair_mask.unsqueeze(-1).float()

        return edge_features


class NodeFeatures(nn.Module):
    """
    Compute node (residue) features from coordinates.

    Features include:
    - Amino acid embedding
    - Local frame features
    - Sinusoidal position encoding
    """

    def __init__(self, node_dim: int = 128, num_positional: int = 16):
        super().__init__()
        self.node_dim = node_dim

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(21, node_dim)  # 20 AA + 1 unknown

        # Position encoding
        self.pos_encoding = nn.Embedding(2048, node_dim)

    def forward(
        self,
        coords: torch.Tensor,
        sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute node features.

        Args:
            coords: Backbone coordinates (B, L, 5, 3)
            sequence: Amino acid indices (B, L)
            mask: Valid residue mask (B, L)

        Returns:
            Node features (B, L, node_dim)
        """
        B, L = sequence.shape
        device = sequence.device

        # Amino acid embedding
        aa_emb = self.aa_embedding(sequence.clamp(0, 20))

        # Position encoding
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_encoding(positions)

        # Combine
        node_features = aa_emb + pos_emb

        # Mask
        node_features = node_features * mask.unsqueeze(-1).float()

        return node_features


class MPNNLayer(nn.Module):
    """
    Message Passing Neural Network layer.

    Updates node features by aggregating messages from neighbors.
    """

    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Message passing update.

        Args:
            node_features: Node features (B, L, D)
            edge_features: Edge features (B, L, K, D_e) - already gathered
            neighbor_idx: Neighbor indices (B, L, K)
            mask: Valid residue mask (B, L)

        Returns:
            Updated node features (B, L, D)
        """
        B, L, D = node_features.shape
        K = neighbor_idx.shape[-1]

        # Gather neighbor node features
        # node_features: (B, L, D)
        # neighbor_idx: (B, L, K)
        # Output: (B, L, K, D)

        # Flatten indices: (B, L, K) -> (B, L*K)
        # Expand for D output dims: (B, L*K, D)
        neighbor_idx_flat = neighbor_idx.reshape(B, L * K).unsqueeze(-1).expand(-1, -1, D)
        
        # Gather from node_features
        # node_features needs to accept indices up to L-1
        # gather dim 1: indices must match node_features dims except dim 1
        # node_features: (B, L, D) -> needs to be gathered into (B, L*K, D)
        # So we use torch.gather on dim 1
        
        neighbor_features_flat = torch.gather(node_features, 1, neighbor_idx_flat)
        neighbor_features = neighbor_features_flat.reshape(B, L, K, D)

        # Center node features expanded
        center_features = node_features.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, D)

        # Compute messages
        message_input = torch.cat(
            [center_features, neighbor_features, edge_features], dim=-1
        )  # (B, L, K, 2D + D_e)
        messages = self.message_mlp(message_input)  # (B, L, K, D)

        # Aggregate messages (mean pooling over neighbors)
        aggregated = messages.mean(dim=2)  # (B, L, D)

        # Update
        update_input = torch.cat([node_features, aggregated], dim=-1)
        updated = self.update_mlp(update_input)

        # Residual connection
        node_features = node_features + self.dropout(updated)
        node_features = self.layer_norm(node_features)

        # Mask
        node_features = node_features * mask.unsqueeze(-1).float()

        return node_features


def build_knn_graph(
    coords: torch.Tensor,
    mask: torch.Tensor,
    k: int = 30,
) -> torch.Tensor:
    """
    Build k-nearest neighbor graph based on CA distances.

    Args:
        coords: Backbone coordinates (B, L, 5, 3)
        mask: Valid residue mask (B, L)
        k: Number of neighbors

    Returns:
        Neighbor indices (B, L, K) where K = min(k, L-1)
    """
    B, L = coords.shape[:2]
    device = coords.device

    # CA coordinates
    CA = coords[:, :, 1, :]  # (B, L, 3)

    # Pairwise distances
    dist = torch.cdist(CA, CA)  # (B, L, L)

    # Set self-distance and invalid distances to large value
    dist = dist + torch.eye(L, device=device).unsqueeze(0) * 1e6

    # Mask invalid residues
    valid_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, L, L)
    dist = dist + (~valid_mask).float() * 1e6

    # Get k nearest neighbors
    K = min(k, L - 1)
    _, neighbor_idx = dist.topk(K, dim=-1, largest=False)  # (B, L, K)

    return neighbor_idx


@register_encoder("proteinmpnn")
class ProteinMPNNEncoder(BaseProteinEncoder):
    """
    ProteinMPNN encoder for protein structure embeddings.

    Uses message passing on k-NN graphs to encode protein structures.

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of message passing layers
        k_neighbors: Number of neighbors in k-NN graph
        dropout: Dropout probability
    """

    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 3,
        k_neighbors: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embedding_dim = node_dim
        self.k_neighbors = k_neighbors

        # Feature computation
        self.node_features = NodeFeatures(node_dim)
        self.edge_features = EdgeFeatures(edge_dim)

        # Message passing layers
        self.layers = nn.ModuleList(
            [
                MPNNLayer(node_dim, edge_dim, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(node_dim, node_dim)

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.load_state_dict(state_dict, strict=False)

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        """
        Encode protein structure.

        Args:
            protein: ProteinInput with coords, sequence, mask

        Returns:
            EncoderOutput with residue embeddings
        """
        device = protein.coords.device
        self.to(device)

        # Compute features
        node_feat = self.node_features(protein.coords, protein.sequence, protein.mask)
        edge_feat_full = self.edge_features(protein.coords, protein.mask)

        # Build k-NN graph
        neighbor_idx = build_knn_graph(protein.coords, protein.mask, self.k_neighbors)

        # Gather edge features for neighbors
        B, L = protein.mask.shape
        K = neighbor_idx.shape[-1]
        D_e = edge_feat_full.shape[-1]

        edge_feat = torch.zeros(B, L, K, D_e, device=device, dtype=edge_feat_full.dtype)
        for k in range(K):
            idx_k = neighbor_idx[:, :, k].clamp(0, L - 1)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)
            seq_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            edge_feat[:, :, k, :] = edge_feat_full[batch_idx, seq_idx, idx_k, :]

        # Message passing
        h = node_feat
        for layer in self.layers:
            h = layer(h, edge_feat, neighbor_idx, protein.mask)

        # Output projection
        h = self.output_proj(h)

        # Global pooling
        mask_expanded = protein.mask.unsqueeze(-1).float()
        global_emb = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return EncoderOutput(
            residue_embeddings=h,
            global_embedding=global_emb,
            metadata={"neighbor_idx": neighbor_idx},
        )
