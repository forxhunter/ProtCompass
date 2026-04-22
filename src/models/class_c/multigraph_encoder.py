"""
Multi-Graph Encoder - Different graphs capture different invariants.

Strategy:
- Graph 1: Spatial k-NN (for I3 - packing)
- Graph 2: Sequential + H-bonds (for I1 - local)
- Separate EGNN tracks, late fusion via attention

This avoids corrupting one invariant's features with another.
"""

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


class EGNNLayerBatched(nn.Module):
    """EGNN layer using batched format (B, N, K) like original."""
    
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
            nn.Linear(hidden_dim, 1, bias=False),
        )
        
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, h, x, edge_index, mask):
        """
        h: (B, N, D)
        x: (B, N, 3)
        edge_index: (B, N, K) - neighbor indices per node
        mask: (B, N)
        """
        B, N, D = h.shape
        K = edge_index.shape[2]
        
        # Gather neighbors
        idx = edge_index.unsqueeze(-1).expand(-1, -1, -1, D)
        h_j = h.unsqueeze(2).expand(-1, -1, K, -1).gather(1, idx)
        
        idx_coord = edge_index.unsqueeze(-1).expand(-1, -1, -1, 3)
        x_j = x.unsqueeze(2).expand(-1, -1, K, -1).gather(1, idx_coord)
        
        # Distances (squared, like original)
        rel_pos = x.unsqueeze(2) - x_j
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        
        # Messages
        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)
        edge_feat = torch.cat([h_i, h_j, dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_feat)
        
        # Attention aggregation
        att = self.att_mlp(m_ij)
        m_i = (att * m_ij).sum(dim=2)
        
        # Node update
        h_out = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        
        # Coord update
        coord_diff = F.normalize(rel_pos, dim=-1)
        coord_weights = self.coord_mlp(m_ij)
        coord_update = (coord_weights * coord_diff).mean(dim=2)
        x_out = x + coord_update
        
        return h_out, x_out


def build_knn_edge_index(coords, mask, k=30):
    """Build k-NN edge_index in batched format (B, N, K)."""
    B, N, _ = coords.shape
    device = coords.device
    
    dists = torch.cdist(coords, coords)
    
    mask_bool = mask.bool()
    valid = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
    dists.masked_fill_(~valid, float('inf'))
    dists.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    
    k = min(k, N - 1)
    _, knn_idx = dists.topk(k, dim=-1, largest=False)
    
    return knn_idx  # (B, N, K)


def build_seq_edge_index(N, device, k=4):
    """Build sequential edge_index (offset ±1, ±2)."""
    # For each node i, neighbors are i-2, i-1, i+1, i+2 (clamped)
    base = torch.arange(N, device=device)
    offsets = torch.tensor([-2, -1, 1, 2], device=device)
    
    neighbors = base.unsqueeze(1) + offsets.unsqueeze(0)  # (N, 4)
    neighbors = neighbors.clamp(0, N-1)
    
    return neighbors.unsqueeze(0)  # (1, N, 4) - broadcast to batch


@register_encoder("multigraph")
class MultiGraphEncoder(BaseProteinEncoder):
    """
    Multi-Graph Encoder with separate graphs for different invariants.
    
    - Spatial Track: k-NN graph for packing (I3)
    - Sequential Track: ±1,±2 for local structure (I1)
    - Late fusion via learned combination
    """
    
    bottleneck_class = BottleneckClass.C
    
    def __init__(
        self, 
        hidden_dim: int = 128,  # Match original EGNN
        num_layers: int = 4,
        k_neighbors: int = 30,
        freeze: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        self.aa_embed = nn.Embedding(21, hidden_dim)
        
        # Spatial track (for I3)
        self.spatial_layers = nn.ModuleList([
            EGNNLayerBatched(hidden_dim) for _ in range(num_layers)
        ])
        self.spatial_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Sequential track (for I1)
        self.seq_layers = nn.ModuleList([
            EGNNLayerBatched(hidden_dim) for _ in range(num_layers)
        ])
        self.seq_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Fusion
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
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
        device = coords.device
        
        # CA coords
        ca = coords[:, :, 1, :]
        
        # Build graphs
        spatial_edge_index = build_knn_edge_index(ca, mask, self.k_neighbors)  # (B, N, K)
        seq_edge_index = build_seq_edge_index(N, device).expand(B, -1, -1)  # (B, N, 4)
        
        # Initial embedding
        h = self.aa_embed(seq)  # (B, N, D)
        x_spatial = ca.clone()
        x_seq = ca.clone()
        
        # Spatial track
        h_spatial = h
        for layer, norm in zip(self.spatial_layers, self.spatial_norms):
            h_spatial, x_spatial = layer(h_spatial, x_spatial, spatial_edge_index, mask)
            h_spatial = norm(h_spatial)
        
        # Sequential track
        h_seq = h
        for layer, norm in zip(self.seq_layers, self.seq_norms):
            h_seq, x_seq = layer(h_seq, x_seq, seq_edge_index, mask)
            h_seq = norm(h_seq)
        
        # Fusion
        h_fused = self.fusion(torch.cat([h_spatial, h_seq], dim=-1))
        
        # Mask invalid
        h_fused = h_fused * mask.unsqueeze(-1)
        
        # Global pooling
        global_emb = h_fused.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        return EncoderOutput(
            residue_embeddings=h_fused,
            global_embedding=global_emb,
        )
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        if checkpoint_path and checkpoint_path != "none":
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"MultiGraphEncoder(dim={self.hidden_dim}, layers={len(self.spatial_layers)}, params={n_params:.1f}M)"
