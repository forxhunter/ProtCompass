"""
GVP (Geometric Vector Perceptron) encoder wrapper.

Uses the official gvp-pytorch library (https://github.com/drorlab/gvp-pytorch),
cloned at /mnt/f/2026_structuerembed/external_repos/gvp-pytorch/.

The official library provides:
- GVP: Base Geometric Vector Perceptron layer
- GVPConv: Torch-geometric message passing with GVPs
- GVPConvLayer: Full convolution layer with residual connections + feedforward

Architecture: Class C (Equivariant Field)
    Preserves SE(3) equivariance for vector channels.
    Random weights only — no protein-pretrained checkpoint shipped.
    Pretraining on Swiss-Prot AlphaFold structures recommended (see pretrain_structure_encoders.py).
"""

import sys
from pathlib import Path

# Allow importing from the local gvp-pytorch clone
# gvp_encoder.py is at src/models/class_c/gvp_encoder.py
# project root = src/models/class_c/../../.. = 3 parents up
_GVP_REPO = Path(__file__).parent.parent.parent.parent / "external_repos" / "gvp-pytorch"
if str(_GVP_REPO) not in sys.path:
    sys.path.insert(0, str(_GVP_REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)

# Import official gvp-pytorch components
from gvp import GVPConvLayer, tuple_sum
from gvp import LayerNorm as _OfficialLayerNorm


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """Radial basis function encoding of distances."""
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    RBF = torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)
    return RBF


def build_knn_graph(ca_pos: torch.Tensor, k: int = 30
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build k-NN graph over CA positions using torch.

    Returns:
        edge_index: (2, n_edges) tensor with source, target indices
        edge_distance: (n_edges,) distance per edge
    """
    B, L = ca_pos.shape[:2]
    device = ca_pos.device

    # Compute pairwise distances
    dist = torch.cdist(ca_pos, ca_pos)  # (B, L, L)
    dist = dist + torch.eye(L, device=device).unsqueeze(0) * 1e6

    # k-NN per node (exclude self)
    k = min(k, L - 1)
    dist_sorted, idx_sorted = dist.sort(dim=-1)
    neighbor_idx = idx_sorted[:, :, 1:k+1]  # (B, L, K), exclude self

    # Build edge list per batch
    all_src, all_dst, all_dist = [], [], []
    for b in range(B):
        src = torch.arange(L, device=device).unsqueeze(1).expand(-1, k).flatten()
        dst = neighbor_idx[b].flatten()
        # Get distances for the k nearest neighbors
        d = dist_sorted[b, :, 1:k+1].flatten()
        all_src.append(src + b * L)
        all_dst.append(dst + b * L)
        all_dist.append(d)

    edge_index = torch.stack([
        torch.cat(all_src),
        torch.cat(all_dst),
    ], dim=0)  # (2, n_edges)

    edge_distance = torch.cat(all_dist)
    return edge_index, edge_distance


def compute_edge_features(
    ca_pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute edge scalar and vector features for GVP.

    Args:
        ca_pos: (B, L, 3) CA positions
        edge_index: (2, n_edges) edge indices
        node_offset: offset for batch-indexed nodes

    Returns:
        edge_s: (n_edges, n_scalar) scalar features (distance RBF)
        edge_v: (n_edges, n_vector, 3) vector features (direction)
    """
    B, L = ca_pos.shape[:2]
    device = ca_pos.device
    n_edges = edge_index.shape[1]

    # Compute per-edge distance and direction
    src = edge_index[0] - node_offset
    dst = edge_index[1] - node_offset

    batch_src = src // L
    local_src = src % L
    batch_dst = dst // L
    local_dst = dst % L

    # Clamp indices
    local_src = local_src.clamp(0, L - 1)
    local_dst = local_dst.clamp(0, L - 1)

    # Distance
    diff = ca_pos[batch_dst, local_dst] - ca_pos[batch_src, local_src]
    dist = torch.norm(diff, dim=-1)  # (n_edges,)

    # RBF encoding
    edge_s = _rbf(dist, D_count=16, device=device)  # (n_edges, 16)

    # Direction (unit vector)
    edge_v = diff / dist.unsqueeze(-1).clamp(min=1e-8)  # (n_edges, 3)

    # Stack as single vector channel
    edge_v = edge_v.unsqueeze(1)  # (n_edges, 1, 3)

    return edge_s, edge_v


@register_encoder("gvp")
class GVPEncoder(BaseProteinEncoder):
    """
    GVP encoder using the official gvp-pytorch library.

    Uses GVPConvLayer (SE(3)-equivariant graph convolutions) with:
    - node_s_dim: scalar embedding dimension (100 default)
    - node_v_dim: number of vector channels (16 default)
    - edge_s_dim: edge scalar dimension (32 default)
    - edge_v_dim: edge vector dimension (1 default)
    - num_layers: number of GVPConvLayer blocks (3 default)
    - k_neighbors: k for k-NN graph construction (30 default)

    Input: ProteinInput with backbone coords (N, CA, C) and sequence.
    Output: per-residue embeddings (B, L, node_s_dim + node_v_dim * 3).
    """

    bottleneck_class = BottleneckClass.C

    def __init__(
        self,
        node_s_dim: int = 100,
        node_v_dim: int = 16,
        edge_s_dim: int = 32,
        edge_v_dim: int = 1,
        num_layers: int = 3,
        k_neighbors: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_s_dim = node_s_dim
        self.node_v_dim = node_v_dim
        self.edge_s_dim = edge_s_dim
        self.edge_v_dim = edge_v_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors

        # Embedding dimension: scalars + flattened vectors
        self._embedding_dim = node_s_dim + node_v_dim * 3

        # Node embedding: one-hot AA → (s_dim, v_dim)
        self.embed_s = nn.Linear(20, node_s_dim)
        self.embed_v = nn.Linear(3, node_v_dim, bias=False)  # initial vector from backbone dirs

        # Edge embedding: (s, V) from distance RBF + direction
        self.edge_embed_s = nn.Linear(16, edge_s_dim)
        # For edge vectors, we just replicate/pad to edge_v_dim channels
        # No learnable transformation needed for the 3D coordinates

        # GVPConvLayer blocks
        node_dims = (node_s_dim, node_v_dim)
        edge_dims = (edge_s_dim, edge_v_dim)

        self.layers = nn.ModuleList([
            GVPConvLayer(
                node_dims=node_dims,
                edge_dims=edge_dims,
                n_message=2,
                n_feedforward=2,
                drop_rate=dropout,
                activations=(F.relu, torch.sigmoid),
                vector_gate=True,
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        B, L = protein.coords.shape[:2]

        # Extract CA coordinates
        ca_pos = protein.coords[:, :, 1, :].float()  # (B, L, 3)
        mask = protein.mask.bool()

        # Node scalar features: one-hot amino acid
        seq = protein.sequence.clamp(0, 19)
        node_s = F.one_hot(seq, num_classes=20).float()  # (B, L, 20)
        node_s = self.embed_s(node_s)  # (B, L, s_dim)

        # Node vector features: backbone directions from N→CA, CA→C
        N = protein.coords[:, :, 0, :].float()
        C = protein.coords[:, :, 2, :].float()
        forward = F.normalize(C - ca_pos, dim=-1)  # CA→C
        backward = F.normalize(N - ca_pos, dim=-1)  # CA→N
        node_v = torch.stack([forward, backward], dim=-2)  # (B, L, 2, 3)

        # Reduce to node_v_dim vectors
        if node_v.shape[-2] >= self.node_v_dim:
            node_v = node_v[:, :, :self.node_v_dim, :]
        else:
            pad = torch.zeros(B, L, self.node_v_dim - node_v.shape[-2], 3,
                             device=device)
            node_v = torch.cat([node_v, pad], dim=-2)

        # Build k-NN graph
        edge_index, edge_distance = build_knn_graph(ca_pos, self.k_neighbors)

        # Compute edge features
        edge_s, edge_v = compute_edge_features(ca_pos, edge_index)
        edge_s = self.edge_embed_s(edge_s)
        # Expand edge_v from (n_edges, 1, 3) to (n_edges, edge_v_dim, 3)
        if self.edge_v_dim > 1:
            edge_v = edge_v.expand(-1, self.edge_v_dim, -1)
        # edge_v is now (n_edges, edge_v_dim, 3)

        # Convert to tuple format for official GVP layers
        node_v_flat = node_v.reshape(B * L, self.node_v_dim, 3)  # (B*L, v_dim, 3)
        node_s_flat = node_s.reshape(B * L, self.node_s_dim)  # (B*L, s_dim)

        # edge_index is already global (build_knn_graph adds batch offsets)
        edge_index_batch = edge_index

        # Edge features (flattened)
        edge_s_flat = edge_s  # (n_edges, s_dim)
        edge_v_flat = edge_v  # (n_edges, edge_v_dim, 3)
        edge_attr = (edge_s_flat, edge_v_flat)

        # Pad nodes to max batch size
        max_L = L
        node_s_pad = torch.zeros(B * max_L, self.node_s_dim, device=device)
        node_v_pad = torch.zeros(B * max_L, self.node_v_dim, 3, device=device)
        mask_flat = mask.reshape(-1).float()

        node_s_pad[:B * L] = node_s_flat
        node_v_pad[:B * L] = node_v_flat

        node_features = (node_s_pad, node_v_pad)

        # Apply GVPConvLayer blocks
        for layer in self.layers:
            node_features = layer(node_features, edge_index_batch.long(), edge_attr)
            node_s_pad, node_v_pad = node_features
            node_s_pad = self.dropout(node_s_pad)
            node_features = (node_s_pad, node_v_pad)

        node_s_out = node_features[0][:B * L]  # (B*L, s_dim)
        node_v_out = node_features[1][:B * L]  # (B*L, v_dim, 3)

        # Mask invalid nodes
        node_s_out = node_s_out * mask_flat.unsqueeze(-1)
        node_v_out = node_v_out * mask_flat.unsqueeze(-1).unsqueeze(-1)

        # Flatten vectors and concatenate
        node_v_flat_out = node_v_out.reshape(B, L, -1)  # (B, L, v_dim * 3)
        embeddings = torch.cat([node_s_out.reshape(B, L, -1), node_v_flat_out], dim=-1)

        # Global pooling
        mask_f = mask.float().unsqueeze(-1)
        global_emb = (embeddings * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        return EncoderOutput(
            residue_embeddings=embeddings,
            global_embedding=global_emb,
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights (e.g., from pretrain_structure_encoders.py)."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "encoder_state" in ckpt:
            self.load_state_dict(ckpt["encoder_state"], strict=False)
        else:
            self.load_state_dict(ckpt, strict=False)
