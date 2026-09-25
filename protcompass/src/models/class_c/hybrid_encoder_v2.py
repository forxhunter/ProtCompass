"""
Hybrid Encoder V2 - Improved architecture for balanced invariant performance.

Key improvements over V1:
1. Parallel EGNN/GearNet branches (no dilution)
2. Residual skip from EGNN to output (preserves I3)
3. Attention pooling (learns contact density weighting)
4. Increased spatial context (k=30)
"""

import math
from typing import Optional, Tuple

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


class EGNNLayerV2(nn.Module):
    """Improved EGNN layer with LayerNorm."""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
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
        
        self.edge_embed = nn.Linear(4, edge_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        
        edge_type_onehot = F.one_hot(edge_type.clamp(0, 2), num_classes=3).float()
        edge_feat = torch.cat([edge_type_onehot, edge_dist.unsqueeze(-1)], dim=-1)
        edge_feat = self.edge_embed(edge_feat)
        
        x_src = x[src]
        x_dst = x[dst]
        edge_input = torch.cat([x_src, x_dst, edge_dist.unsqueeze(-1), edge_feat], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m_ij)
        
        x_new = x + self.node_mlp(torch.cat([x, agg], dim=-1))
        x_new = self.norm(x_new)
        
        coord_diff = coords[src] - coords[dst]
        coord_weight = self.coord_mlp(m_ij)
        coord_update = torch.zeros_like(coords)
        coord_update.index_add_(0, dst, coord_diff * coord_weight)
        coords_new = coords + coord_update * 0.1
        
        return x_new, coords_new


class GearNetLayerV2(nn.Module):
    """Improved GearNet layer with LayerNorm."""
    
    def __init__(self, hidden_dim: int, num_edge_types: int = 3, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_type_embed = nn.Embedding(num_edge_types, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        edge_emb = self.edge_type_embed(edge_type.clamp(0, 2))
        
        q = self.q_proj(x[dst]).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x[src] + edge_emb).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x[src]).view(-1, self.num_heads, self.head_dim)
        
        attn = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)
        attn_exp = attn.exp()
        attn_sum = torch.zeros(x.size(0), self.num_heads, device=x.device)
        attn_sum.index_add_(0, dst, attn_exp)
        attn_norm = attn_exp / (attn_sum[dst] + 1e-8)
        
        v_weighted = v * attn_norm.unsqueeze(-1)
        agg = torch.zeros(x.size(0), self.num_heads, self.head_dim, device=x.device)
        agg.index_add_(0, dst, v_weighted)
        agg = agg.view(-1, self.hidden_dim)
        
        out = self.out_proj(agg)
        return self.norm(x + out)


class AttentionPooling(nn.Module):
    """Attention-based pooling that learns to weight residues."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), mask: (B, N)
        attn_scores = self.attn(x).squeeze(-1)  # (B, N)
        attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, N, 1)
        return (attn_weights * x).sum(dim=1)  # (B, D)


def build_graph_v2(
    coords: torch.Tensor,
    mask: torch.Tensor,
    k_spatial: int = 30,
    spatial_radius: float = 12.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build multi-edge graph with increased spatial context."""
    B, N, A, _ = coords.shape
    device = coords.device
    
    all_src, all_dst, all_types, all_vecs, all_dists = [], [], [], [], []
    
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    node_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    flat_indices = batch_idx * N + node_idx
    valid_mask = mask.bool()
    
    # Sequential edges
    for offset in [1, 2, -1, -2]:
        src_nodes = node_idx
        dst_nodes = node_idx + offset
        valid = (dst_nodes >= 0) & (dst_nodes < N) & valid_mask
        
        abs_offset = abs(offset)
        if offset > 0:
            # dst_nodes = i + offset, so dst at position i corresponds to src at i+offset
            # The last `offset` positions don't have valid destinations
            dst_mask = torch.cat([torch.zeros(B, abs_offset, device=device), mask[:, :-abs_offset]], dim=1).bool()
        else:
            # dst_nodes = i - |offset|, so first |offset| positions don't have valid destinations
            dst_mask = torch.cat([mask[:, abs_offset:], torch.zeros(B, abs_offset, device=device)], dim=1).bool()
        valid = valid & dst_mask
        
        if valid.any():
            src_flat = flat_indices[valid]
            dst_flat = (batch_idx * N + dst_nodes)[valid]
            ca_flat = coords[:, :, 1, :].reshape(B * N, 3)
            vecs = ca_flat[dst_flat] - ca_flat[src_flat]
            dists = vecs.norm(dim=-1)
            all_src.append(src_flat)
            all_dst.append(dst_flat)
            all_types.append(torch.zeros(src_flat.size(0), device=device, dtype=torch.long))
            all_vecs.append(vecs)
            all_dists.append(dists)
    
    # Spatial edges (k-NN)
    ca_coords = coords[:, :, 1, :]
    dists_matrix = torch.cdist(ca_coords, ca_coords)
    mask_bool = mask.bool()
    valid_pair = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
    dists_matrix.masked_fill_(~valid_pair, float('inf'))
    dists_matrix.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    dists_matrix.masked_fill_(dists_matrix > spatial_radius, float('inf'))
    
    k = min(k_spatial, N - 1)
    if k > 0:
        knn_vals, knn_idx = dists_matrix.topk(k, dim=-1, largest=False)
        valid_knns = knn_vals < float('inf')
        
        if valid_knns.any():
            src_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
            flat_src = (batch_idx.unsqueeze(2) * N + src_idx)[valid_knns]
            flat_dst = (batch_idx.unsqueeze(2) * N + knn_idx)[valid_knns]
            ca_flat = coords[:, :, 1, :].reshape(B * N, 3)
            vecs = ca_flat[flat_dst] - ca_flat[flat_src]
            dists = vecs.norm(dim=-1)
            all_src.append(flat_src)
            all_dst.append(flat_dst)
            all_types.append(torch.ones(flat_src.size(0), device=device, dtype=torch.long))
            all_vecs.append(vecs)
            all_dists.append(dists)
    
    # H-bond edges
    if A >= 4:
        n_coords = coords[:, :, 0, :]
        o_coords = coords[:, :, 3, :]
        hb_dists = torch.cdist(n_coords, o_coords)
        valid_hb = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
        hb_dists.masked_fill_(~valid_hb, float('inf'))
        idx = torch.arange(N, device=device)
        seq_sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= 3
        hb_dists.masked_fill_(~seq_sep.unsqueeze(0).expand(B, N, N), float('inf'))
        hbond_hits = hb_dists < 3.5
        
        if hbond_hits.any():
            b_idx, src_idx, dst_idx = torch.where(hbond_hits)
            src_flat = b_idx * N + src_idx
            dst_flat = b_idx * N + dst_idx
            ca_flat = coords[:, :, 1, :].reshape(B * N, 3)
            vecs = ca_flat[dst_flat] - ca_flat[src_flat]
            dists = vecs.norm(dim=-1)
            all_src.append(src_flat)
            all_dst.append(dst_flat)
            all_types.append(torch.full((src_flat.size(0),), 2, device=device, dtype=torch.long))
            all_vecs.append(vecs)
            all_dists.append(dists)
    
    if all_src:
        edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
        edge_type = torch.cat(all_types)
        edge_vec = torch.cat(all_vecs, dim=0)
        edge_dist = torch.cat(all_dists)
    else:
        edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
        edge_type = torch.zeros((0,), device=device, dtype=torch.long)
        edge_vec = torch.zeros((0, 3), device=device)
        edge_dist = torch.zeros((0,), device=device)
        
    return edge_index, edge_type, edge_vec, edge_dist


@register_encoder("hybrid_v2")
class HybridEncoderV2(BaseProteinEncoder):
    """
    Improved Hybrid Encoder with parallel branches and residual skip.
    
    Key improvements:
    - Parallel EGNN + GearNet branches (not sequential)
    - Residual skip from EGNN to preserve I3
    - Attention pooling for global embedding
    - Increased spatial context (k=30)
    """
    
    bottleneck_class = BottleneckClass.C
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_egnn_layers: int = 4,
        num_gearnet_layers: int = 3,
        k_spatial: int = 30,
        spatial_radius: float = 12.0,
        freeze: bool = False,
        seed: int = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_spatial = k_spatial
        self.spatial_radius = spatial_radius
        self.seed = seed

        # Set seed BEFORE initializing weights
        if seed is not None:
            self._seed_everything(seed)

        # Input embedding
        self.aa_embed = nn.Embedding(21, hidden_dim)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Parallel branches
        self.egnn_layers = nn.ModuleList([
            EGNNLayerV2(hidden_dim) for _ in range(num_egnn_layers)
        ])
        self.gearnet_layers = nn.ModuleList([
            GearNetLayerV2(hidden_dim) for _ in range(num_gearnet_layers)
        ])
        
        # Fusion gate (learned)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Residual skip projection
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention pooling
        self.attn_pool = AttentionPooling(hidden_dim)
        
        # Output projection
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
        device = coords.device
        
        # Center coordinates
        center = torch.sum(coords[:, :, 1, :] * mask.unsqueeze(-1), dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
        coords = coords - center.unsqueeze(2)
        
        # Initial embedding
        x = self.input_proj(self.aa_embed(seq))
        
        # Build graph
        edge_index, edge_type, edge_vec, edge_dist = build_graph_v2(
            coords, mask, self.k_spatial, self.spatial_radius
        )
        
        # Flatten
        x_flat = x.view(B * N, -1)
        ca_coords = coords[:, :, 1, :].reshape(B * N, 3)
        
        # === PARALLEL BRANCHES ===
        
        # EGNN branch
        x_egnn = x_flat.clone()
        coords_egnn = ca_coords.clone()
        for layer in self.egnn_layers:
            x_egnn, coords_egnn = layer(x_egnn, coords_egnn, edge_index, edge_type, edge_dist)
        
        # GearNet branch
        x_gearnet = x_flat.clone()
        for layer in self.gearnet_layers:
            x_gearnet = layer(x_gearnet, edge_index, edge_type)
        
        # === FUSION ===
        gate = self.fusion_gate(torch.cat([x_egnn, x_gearnet], dim=-1))
        x_fused = gate * x_egnn + (1 - gate) * x_gearnet
        
        # === RESIDUAL SKIP from EGNN (preserves I3) ===
        x_out = self.output_proj(x_fused) + self.skip_proj(x_egnn)
        x_out = x_out.view(B, N, -1)
        
        # === ATTENTION POOLING ===
        global_emb = self.attn_pool(x_out, mask)
        
        return EncoderOutput(
            residue_embeddings=x_out,
            global_embedding=global_emb,
        )
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        if checkpoint_path and checkpoint_path != "none":
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            print("HybridEncoderV2: Initialized with random weights")
    
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"HybridEncoderV2(dim={self.hidden_dim}, params={n_params:.1f}M)"

    def _seed_everything(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_state_dict(self) -> dict:
        """Return state dict for serialization and reproducibility."""
        return self.state_dict()

    def set_state_dict(self, state_dict: dict) -> None:
        """Load a serialized encoder state."""
        self.load_state_dict(state_dict)
