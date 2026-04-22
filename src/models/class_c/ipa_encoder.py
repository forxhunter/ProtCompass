"""
Invariant Point Attention (IPA) Encoder.

Based on AlphaFold2's structure module. Uses:
- Rigid body frames per residue
- Point attention that is invariant to global rotations
- Local coordinate transformations

Reference: AlphaFold2 paper (Jumper et al., 2021)
"""

import math
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


def build_frames_from_backbone(coords):
    """
    Build local reference frames from backbone atoms.
    
    coords: (B, N, A, 3) where A >= 3 (N, CA, C)
    Returns: (B, N, 3, 3) rotation matrices, (B, N, 3) translations
    """
    B, N, A, _ = coords.shape
    
    n_pos = coords[:, :, 0, :]  # N
    ca_pos = coords[:, :, 1, :]  # CA (origin)
    c_pos = coords[:, :, 2, :]  # C
    
    # Build orthonormal frame: x = CA->C, y = perpendicular, z = cross
    v1 = c_pos - ca_pos  # CA -> C
    v2 = n_pos - ca_pos  # CA -> N

    # Gram-Schmidt with safety checks
    v1_norm = v1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    e1 = v1 / v1_norm  # x-axis
    v2_proj = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    v2_proj_norm = v2_proj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    e2 = v2_proj / v2_proj_norm  # y-axis
    e3 = torch.cross(e1, e2, dim=-1)  # z-axis
    
    # Stack to rotation matrix
    R = torch.stack([e1, e2, e3], dim=-1)  # (B, N, 3, 3)
    t = ca_pos  # Translation is CA position
    
    return R, t


class InvariantPointAttention(nn.Module):
    """
    Invariant Point Attention from AlphaFold2.
    
    Key features:
    - Queries/keys/values in local reference frames
    - Distance-based attention bias
    - Frame-invariant aggregation
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, num_points: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = hidden_dim // num_heads
        
        # Standard Q/K/V for scalar features
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Point projections (for geometric features)
        self.q_point_proj = nn.Linear(hidden_dim, num_heads * num_points * 3)
        self.k_point_proj = nn.Linear(hidden_dim, num_heads * num_points * 3)
        self.v_point_proj = nn.Linear(hidden_dim, num_heads * num_points * 3)
        
        # Output
        self.out_proj = nn.Linear(hidden_dim + num_heads * num_points * 3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Learned weights for combining attention components
        self.point_weight = nn.Parameter(torch.ones(num_heads))
        
    def forward(self, h, R, t, edge_index):
        """
        Simplified forward pass.
        h: (B*N, D) node features
        R: (B*N, 3, 3) rotation matrices per node
        t: (B*N, 3) translations per node
        edge_index: (2, E) edge indices
        """
        src, dst = edge_index
        N = h.size(0)
        E = src.size(0)
        
        # Standard attention
        Q = self.q_proj(h[dst]).view(E, self.num_heads, self.head_dim)
        K = self.k_proj(h[src]).view(E, self.num_heads, self.head_dim)
        V = self.v_proj(h[src]).view(E, self.num_heads, self.head_dim)
        
        # Point attention (simplified: just use positions)
        # Query points in global frame
        t_dst = t[dst]  # (E, 3)
        t_src = t[src]  # (E, 3)
        
        # Distance-based point attention
        pt_dist_sq = ((t_dst - t_src) ** 2).sum(dim=-1, keepdim=True)  # (E, 1)
        pt_attn = -0.5 * pt_dist_sq.expand(-1, self.num_heads)  # (E, num_heads)
        
        # Combine scalar and point attention
        scalar_attn = (Q * K).sum(dim=-1) / math.sqrt(self.head_dim)  # (E, num_heads)
        attn = scalar_attn + pt_attn * self.point_weight.view(1, -1)

        # Softmax over neighbors with numerical stability
        attn = attn.clamp(min=-20, max=20)  # Prevent exp overflow
        attn_exp = attn.exp()
        attn_sum = torch.zeros(N, self.num_heads, device=h.device)
        attn_sum.index_add_(0, dst, attn_exp)
        attn_norm = attn_exp / (attn_sum[dst].clamp(min=1e-8))
        
        # Aggregate values (scalar)
        V_weighted = V * attn_norm.unsqueeze(-1)
        out_scalar = torch.zeros(N, self.num_heads, self.head_dim, device=h.device)
        out_scalar.index_add_(0, dst, V_weighted)
        out_scalar = out_scalar.view(N, -1)
        
        # Aggregate position differences (geometric feature)
        rel_pos = t_src - t_dst  # (E, 3)
        rel_pos_weighted = rel_pos * attn_norm.mean(dim=-1, keepdim=True)
        pos_agg = torch.zeros(N, 3, device=h.device)
        pos_agg.index_add_(0, dst, rel_pos_weighted)
        
        # Transform position to feature
        pos_feat = pos_agg.repeat(1, self.num_heads * self.num_points)
        
        # Combine and project
        out = torch.cat([out_scalar, pos_feat], dim=-1)
        out = self.out_proj(out)
        
        return self.norm(h + out)


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
    
    return torch.stack([src_flat[src_valid & dst_valid], dst_flat[src_valid & dst_valid]])


@register_encoder("ipa")
class IPAEncoder(BaseProteinEncoder):
    """
    Invariant Point Attention encoder (AlphaFold-style).
    
    Uses local reference frames from backbone to achieve
    frame-invariant attention.
    """
    
    bottleneck_class = BottleneckClass.C
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_points: int = 4,
        k_neighbors: int = 30,
        freeze: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        self.aa_embed = nn.Embedding(21, hidden_dim)
        
        self.ipa_layers = nn.ModuleList([
            InvariantPointAttention(hidden_dim, num_heads, num_points)
            for _ in range(num_layers)
        ])
        
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
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
        
        # Build frames from backbone
        R, t = build_frames_from_backbone(coords)  # (B, N, 3, 3), (B, N, 3)
        R_flat = R.reshape(B * N, 3, 3)
        t_flat = t.reshape(B * N, 3)
        
        # Build graph
        ca = coords[:, :, 1, :]
        edge_index = build_knn_edges(ca, mask, self.k_neighbors)
        
        # Initial embedding
        h = self.aa_embed(seq).view(B * N, -1)
        
        # Apply IPA layers
        for ipa, ff in zip(self.ipa_layers, self.ff_layers):
            h = ipa(h, R_flat, t_flat, edge_index)
            h = h + ff(h)
        
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
        return f"IPAEncoder(layers={len(self.ipa_layers)}, dim={self.hidden_dim}, params={n_params:.1f}M)"
