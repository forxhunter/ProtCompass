"""
Multi-k EGNN Encoder with Scale-Gated Residual Adds.

Instead of mixing architectures, keep everything EGNN but with multiple neighborhood scales.

Key design:
1. Three neighbor sets: k = {12, 30, 60}
2. Separate message passes per scale (no concatenation)
3. Scale-gated addition: h <- h + g1*Δh_k12 + g2*Δh_k30 + g3*Δh_k60
4. Initialize gates near 0 (start from baseline behavior)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)


class EGNNLayer(nn.Module):
    """Single EGNN message passing layer."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Attention for aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h, x, edge_index):
        """
        h: (B*N, D) node features
        x: (B*N, 3) coordinates
        edge_index: (2, E)
        """
        src, dst = edge_index
        
        # Edge features
        rel_pos = x[src] - x[dst]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        
        # Messages
        msg_input = torch.cat([h[src], h[dst], dist_sq], dim=-1)
        msg = self.message_mlp(msg_input)
        
        # Attention weighting
        attn = self.attention(msg)
        msg = msg * attn
        
        # Aggregate
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        
        # Update
        update_input = torch.cat([h, agg], dim=-1)
        h_new = self.update_mlp(update_input)
        
        return self.norm(h + h_new)


def build_knn_edges(coords, mask, k):
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


@register_encoder("multik_egnn")
class MultiKEGNN(BaseProteinEncoder):
    """
    Multi-k EGNN: Multiple neighborhood scales with gated fusion.
    
    Key insight: Use multiple k values (12, 30, 60) and fuse with
    learned gates initialized near 0 (starts like baseline EGNN).
    """
    
    bottleneck_class = BottleneckClass.C
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        k_values: List[int] = [12, 30, 60],
        freeze: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_values = k_values
        
        self.aa_embed = nn.Embedding(21, hidden_dim)
        
        # Separate EGNN layers per scale
        self.scale_layers = nn.ModuleDict({
            f"k{k}": nn.ModuleList([
                EGNNLayer(hidden_dim) for _ in range(num_layers)
            ])
            for k in k_values
        })
        
        # Scale gates - initialized to small values (near 0 output)
        # Use sigmoid(-4) ≈ 0.018 as starting weight
        self.scale_gates = nn.ParameterDict({
            f"k{k}": nn.Parameter(torch.full((hidden_dim,), -4.0))
            for k in k_values
        })
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
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
        
        # CA coordinates
        ca = coords[:, :, 1, :]
        
        # Build graphs for each k
        edge_indices = {}
        for k in self.k_values:
            edge_indices[k] = build_knn_edges(ca, mask, k)
        
        # Initial embedding
        h = self.aa_embed(seq).view(B * N, -1)
        x = ca.reshape(B * N, 3)
        
        # Base representation (starting point)
        h_base = h.clone()
        
        # Process each scale separately
        scale_outputs = {}
        for k in self.k_values:
            h_k = h.clone()
            layers = self.scale_layers[f"k{k}"]
            edges = edge_indices[k]
            
            for layer in layers:
                h_k = layer(h_k, x, edges)
            
            # Store the delta (change from base)
            scale_outputs[k] = h_k - h_base
        
        # Gated fusion
        h_out = h_base.clone()
        for k in self.k_values:
            gate = torch.sigmoid(self.scale_gates[f"k{k}"])  # (D,)
            h_out = h_out + gate * scale_outputs[k]
        
        h_out = self.norm(h_out)
        h_out = self.output_proj(h_out)
        h_out = h_out.view(B, N, -1)
        
        # Mask and pool
        h_out = h_out * mask.unsqueeze(-1)
        global_emb = h_out.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        return EncoderOutput(
            residue_embeddings=h_out,
            global_embedding=global_emb,
        )
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        if checkpoint_path and checkpoint_path != "none":
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"MultiKEGNN(k={self.k_values}, dim={self.hidden_dim}, params={n_params:.1f}M)"
