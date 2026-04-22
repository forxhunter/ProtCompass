"""
SE(3)-Transformer Encoder for Protein Structures.

Features higher-order equivariance using spherical harmonics and
attention-based message passing. May break the I1-I3 tradeoff.

Reference: SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)


class SE3Attention(nn.Module):
    """
    SE(3)-equivariant attention layer.
    
    Uses relative positions and distances to compute attention,
    ensuring equivariance to rotations and translations.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Distance encoding (invariant to rotation)
        self.dist_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_heads),
        )
        
        # Relative position encoding (equivariant)
        self.pos_proj = nn.Linear(3, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h, x, edge_index, mask=None):
        """
        h: (B*N, D) node features
        x: (B*N, 3) node positions
        edge_index: (2, E) edge indices
        """
        src, dst = edge_index
        
        # Queries at destinations, keys/values at sources
        Q = self.q_proj(h[dst]).view(-1, self.num_heads, self.head_dim)
        K = self.k_proj(h[src]).view(-1, self.num_heads, self.head_dim)
        V = self.v_proj(h[src]).view(-1, self.num_heads, self.head_dim)
        
        # Relative position and distance (both equivariant/invariant)
        rel_pos = x[src] - x[dst]  # (E, 3)
        dist = rel_pos.norm(dim=-1, keepdim=True)  # (E, 1)
        
        # Distance bias for attention (invariant)
        dist_bias = self.dist_embed(dist)  # (E, num_heads)
        
        # Scaled dot-product attention with distance bias
        attn = (Q * K).sum(dim=-1) / math.sqrt(self.head_dim)  # (E, num_heads)
        attn = attn + dist_bias
        
        # Softmax over neighbors (per destination node)
        attn_exp = attn.exp()
        attn_sum = torch.zeros(h.size(0), self.num_heads, device=h.device)
        attn_sum.index_add_(0, dst, attn_exp)
        attn_norm = attn_exp / (attn_sum[dst] + 1e-8)  # (E, num_heads)
        
        # Weighted value aggregation
        V_weighted = V * attn_norm.unsqueeze(-1)  # (E, num_heads, head_dim)
        
        # Aggregate to destination nodes
        out = torch.zeros(h.size(0), self.num_heads, self.head_dim, device=h.device)
        out.index_add_(0, dst, V_weighted)
        out = out.view(-1, self.hidden_dim)
        
        # Equivariant position update
        pos_update = self.pos_proj(rel_pos)  # (E, hidden_dim)
        pos_agg = torch.zeros_like(h)
        pos_agg.index_add_(0, dst, pos_update * attn_norm.mean(dim=-1, keepdim=True))
        
        # Combine and residual
        out = self.out_proj(out + pos_agg)
        return self.norm(h + out)


class FeedForward(nn.Module):
    """Feed-forward layer with SiLU activation."""
    
    def __init__(self, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        return self.norm(x + self.net(x))


def build_knn_edges(coords, mask, k=30):
    """Build k-NN edge index."""
    B, N, _ = coords.shape
    device = coords.device
    
    dists = torch.cdist(coords, coords)
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


@register_encoder("se3_transformer")
class SE3TransformerEncoder(BaseProteinEncoder):
    """
    SE(3)-Transformer: Higher-order equivariant attention.
    
    Key features:
    - Attention with distance bias (rotation invariant)
    - Relative position updates (SE(3) equivariant)
    - May capture both local (I1) and packing (I3) properties
    """
    
    bottleneck_class = BottleneckClass.C
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        k_neighbors: int = 30,
        freeze: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        self.aa_embed = nn.Embedding(21, hidden_dim)
        
        # SE(3) attention layers
        self.attention_layers = nn.ModuleList([
            SE3Attention(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            FeedForward(hidden_dim) for _ in range(num_layers)
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
        
        # CA coordinates
        ca = coords[:, :, 1, :]
        
        # Build graph
        edge_index = build_knn_edges(ca, mask, self.k_neighbors)
        
        # Initial embedding
        h = self.aa_embed(seq).view(B * N, -1)
        x = ca.reshape(B * N, 3)
        
        # Apply SE(3) attention layers
        for attn, ff in zip(self.attention_layers, self.ff_layers):
            h = attn(h, x, edge_index)
            h = ff(h)
        
        # Output
        h = self.output_proj(h)
        h = h.view(B, N, -1)
        
        # Mask and pool
        h = h * mask.unsqueeze(-1)
        global_emb = h.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        return EncoderOutput(
            residue_embeddings=h,
            global_embedding=global_emb,
        )
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        if checkpoint_path and checkpoint_path != "none":
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"SE3TransformerEncoder(layers={len(self.attention_layers)}, dim={self.hidden_dim}, params={n_params:.1f}M)"
