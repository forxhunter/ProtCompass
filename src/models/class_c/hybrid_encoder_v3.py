"""
Hybrid Encoder V3 - EGNN-Dominant architecture.

Key changes from V2:
1. 5 EGNN layers (dominant) + 1 GearNet layer at END only
2. Simple concatenation (no gated fusion)
3. EGNN output is the primary signal

Goal: Preserve EGNN's I3 (0.815) while adding minimal I1 boost from GearNet.
"""

import math
from typing import Tuple

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


class EGNNLayerV3(nn.Module):
    """EGNN layer with residual and LayerNorm."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h, x, edge_index):
        """h: (B*N, D), x: (B*N, 3), edge_index: (2, E)"""
        src, dst = edge_index
        
        # Compute distances
        rel_pos = x[src] - x[dst]
        dist = rel_pos.norm(dim=-1, keepdim=True)
        
        # Edge messages
        edge_input = torch.cat([h[src], h[dst], dist], dim=-1)
        m = self.edge_mlp(edge_input)
        
        # Aggregate
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        
        # Node update with residual
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = self.norm(h_new)
        
        # Coord update
        coord_weight = self.coord_mlp(m)
        coord_update = torch.zeros_like(x)
        coord_update.index_add_(0, dst, rel_pos * coord_weight)
        x_new = x + coord_update * 0.1
        
        return h_new, x_new


class GearNetLayerSimple(nn.Module):
    """Simplified GearNet layer for relational refinement."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h, edge_index):
        src, dst = edge_index
        
        # Simple message passing
        messages = self.mlp(torch.cat([h[src], h[dst]], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        
        return self.norm(h + agg)


def build_knn_graph(coords, mask, k=30):
    """Build k-NN graph from CA coordinates."""
    B, N, A, _ = coords.shape
    device = coords.device
    
    ca = coords[:, :, 1, :]  # (B, N, 3)
    
    # Pairwise distances
    dists = torch.cdist(ca, ca)  # (B, N, N)
    
    # Mask invalid
    mask_bool = mask.bool()
    valid = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
    dists.masked_fill_(~valid, float('inf'))
    dists.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    
    # k-NN
    k = min(k, N - 1)
    if k <= 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    _, knn_idx = dists.topk(k, dim=-1, largest=False)
    
    # Build edge index
    batch_offset = torch.arange(B, device=device).view(B, 1, 1) * N
    src_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
    
    src_flat = (batch_offset + src_idx).reshape(-1)
    dst_flat = (batch_offset + knn_idx).reshape(-1)
    
    # Filter by mask
    src_valid = mask.reshape(-1)[src_flat].bool()
    dst_valid = mask.reshape(-1)[dst_flat].bool()
    valid_edges = src_valid & dst_valid
    
    edge_index = torch.stack([src_flat[valid_edges], dst_flat[valid_edges]])
    return edge_index


@register_encoder("hybrid_v3")
class HybridEncoderV3(BaseProteinEncoder):
    """
    EGNN-Dominant Hybrid: 5 EGNN + 1 GearNet at end.
    
    Architecture:
    - 5 EGNN layers (main feature extraction)
    - 1 GearNet layer (relational refinement at end only)
    - No gated fusion - EGNN output is primary
    """
    
    bottleneck_class = BottleneckClass.C
    
    def __init__(self, hidden_dim: int = 256, k_neighbors: int = 30, freeze: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        self.aa_embed = nn.Embedding(21, hidden_dim)
        
        # 5 EGNN layers (dominant)
        self.egnn_layers = nn.ModuleList([
            EGNNLayerV3(hidden_dim) for _ in range(5)
        ])
        
        # 1 GearNet layer at end
        self.gearnet_layer = GearNetLayerSimple(hidden_dim)
        
        # Output
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        if freeze:
            self.freeze()
    
    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim
    
    def encode(self, protein_input: ProteinInput) -> EncoderOutput:
        coords = protein_input.coords
        seq = protein_input.sequence
        mask = protein_input.mask
        
        B, N, A, _ = coords.shape
        
        # Center
        center = (coords[:, :, 1, :] * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
        coords = coords - center.unsqueeze(2)
        
        # Embed
        h = self.aa_embed(seq).view(B * N, -1)
        x = coords[:, :, 1, :].reshape(B * N, 3)
        
        # Build graph
        edge_index = build_knn_graph(coords, mask, self.k_neighbors)
        
        # 5 EGNN layers
        for layer in self.egnn_layers:
            h, x = layer(h, x, edge_index)
        
        # 1 GearNet layer at end
        h = self.gearnet_layer(h, edge_index)
        
        # Output
        h = self.output_proj(h)
        h = h.view(B, N, -1)
        
        # Global pooling (sum / count)
        global_emb = (h * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        return EncoderOutput(
            residue_embeddings=h,
            global_embedding=global_emb,
        )
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        if checkpoint_path and checkpoint_path != "none":
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"HybridEncoderV3(dim={self.hidden_dim}, params={n_params:.1f}M)"
