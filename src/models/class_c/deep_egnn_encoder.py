"""
Deep EGNN Encoder - Pure EGNN with more depth.

Hypothesis: EGNN alone with more depth can achieve both high I1 and I3.
No GearNet layers to avoid feature dilution.
"""

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


class DeepEGNNLayer(nn.Module):
    """EGNN layer with residual, LayerNorm, and attention aggregation."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Edge MLP (larger capacity)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        
        # Attention for aggregation
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h, x, edge_index):
        src, dst = edge_index
        
        # Distances
        rel_pos = x[src] - x[dst]
        dist = rel_pos.norm(dim=-1, keepdim=True)
        
        # Edge messages
        edge_input = torch.cat([h[src], h[dst], dist], dim=-1)
        m = self.edge_mlp(edge_input)
        
        # Attention-weighted aggregation
        attn_weights = self.attn(m)
        m_weighted = m * attn_weights
        
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m_weighted)
        
        # Node update with residual
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = self.norm(h_new)
        
        # Coord update
        coord_weight = self.coord_mlp(m)
        coord_update = torch.zeros_like(x)
        coord_update.index_add_(0, dst, rel_pos * coord_weight)
        x_new = x + coord_update * 0.1
        
        return h_new, x_new


def build_knn_graph(coords, mask, k=40):
    """Build k-NN graph."""
    B, N, A, _ = coords.shape
    device = coords.device
    
    ca = coords[:, :, 1, :]
    dists = torch.cdist(ca, ca)
    
    mask_bool = mask.bool()
    valid = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
    dists.masked_fill_(~valid, float('inf'))
    dists.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    
    k = min(k, N - 1)
    if k <= 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    _, knn_idx = dists.topk(k, dim=-1, largest=False)
    
    batch_offset = torch.arange(B, device=device).view(B, 1, 1) * N
    src_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
    
    src_flat = (batch_offset + src_idx).reshape(-1)
    dst_flat = (batch_offset + knn_idx).reshape(-1)
    
    src_valid = mask.reshape(-1)[src_flat].bool()
    dst_valid = mask.reshape(-1)[dst_flat].bool()
    valid_edges = src_valid & dst_valid
    
    return torch.stack([src_flat[valid_edges], dst_flat[valid_edges]])


@register_encoder("deep_egnn")
class DeepEGNNEncoder(BaseProteinEncoder):
    """
    Deep EGNN: 6-8 layers of pure EGNN.
    
    No GearNet, no hybridization - relies entirely on EGNN's
    equivariant message passing with increased depth.
    """
    
    bottleneck_class = BottleneckClass.C
    
    def __init__(
        self, 
        hidden_dim: int = 256, 
        num_layers: int = 6,
        k_neighbors: int = 40,
        dropout: float = 0.1,
        freeze: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        self.aa_embed = nn.Embedding(21, hidden_dim)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Deep EGNN stack
        self.layers = nn.ModuleList([
            DeepEGNNLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
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
        h = self.input_proj(self.aa_embed(seq)).view(B * N, -1)
        x = coords[:, :, 1, :].reshape(B * N, 3)
        
        # Build graph
        edge_index = build_knn_graph(coords, mask, self.k_neighbors)
        
        # Deep EGNN layers
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        
        # Output
        h = self.output_proj(h)
        h = h.view(B, N, -1)
        
        # Global pooling
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
        return f"DeepEGNNEncoder(layers={len(self.layers)}, dim={self.hidden_dim}, k={self.k_neighbors}, params={n_params:.1f}M)"
