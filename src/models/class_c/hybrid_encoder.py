"""
Hybrid EGNN-GearNet Encoder.

Combines:
- EGNN's equivariant message passing for I2/I3 (packing, metric geometry)
- GearNet's relational structure for I1/I4 (local conformational, topology)
- Spectral topology head for global awareness

Architecture:
1. Multi-edge graph construction (sequential + spatial + H-bond)
2. EGNN layers (3x) - equivariant, coordinate-aware
3. GearNet layers (3x) - edge-type attention
4. Spectral topology head - graph Laplacian eigenvectors
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_encoder import BaseProteinEncoder, ProteinInput, EncoderOutput, register_encoder, BottleneckClass


class MultiEdgeGraph:
    """Container for multi-edge graph structure."""
    
    def __init__(
        self,
        edge_index: torch.Tensor,      # (2, E) all edges
        edge_type: torch.Tensor,       # (E,) edge type: 0=seq, 1=spatial, 2=hbond
        edge_attr: torch.Tensor,       # (E, D) edge features
        batch: Optional[torch.Tensor] = None,
    ):
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_attr = edge_attr
        self.batch = batch


def build_multi_edge_graph(
    coords: torch.Tensor,  # (B, N, A, 3) - A=5 atoms: N, CA, C, O, CB
    mask: torch.Tensor,    # (B, N)
    k_spatial: int = 16,
    spatial_radius: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build multi-edge graph with three edge types (Vectorized).
    
    Returns:
        edge_index: (2, E) all edges
        edge_type: (E,) 0=sequential, 1=spatial, 2=hydrogen bond
        edge_vec: (E, 3) edge vectors
        edge_dist: (E,) edge distances
    """
    B, N, A, _ = coords.shape
    device = coords.device
    
    # Pre-allocate lists for batch processing
    all_edges_src = []
    all_edges_dst = []
    all_types = []
    all_vecs = []
    all_dists = []
    
    # 1. Sequential edges (vectorized)
    # Define offsets: i -> i+1, i -> i+2, i -> i-1, i -> i-2
    offsets = torch.tensor([1, 2, -1, -2], device=device)
    
    # Create grid of indices (B, N)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    node_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    
    # Flat indices: b * N + i
    flat_indices = batch_idx * N + node_idx
    
    # Mask out invalid nodes
    valid_mask = mask.bool()
    
    for offset in offsets:
        # Source and destination indices
        src_nodes = node_idx
        dst_nodes = node_idx + offset
        
        # Valid edges: within bounds [0, N-1] AND both source/dest are valid in mask
        valid_edges = (dst_nodes >= 0) & (dst_nodes < N) & valid_mask
        # Check if dst is also valid
        if offset > 0:
            dst_mask = torch.cat([torch.zeros_like(mask[:, :offset]), mask[:, :-offset]], dim=1).bool()
        else:
            dst_mask = torch.cat([mask[:, -offset:], torch.zeros_like(mask[:, :-offset])], dim=1).bool()
            
        valid_edges = valid_edges & dst_mask
        
        if valid_edges.any():
            # Get flat indices
            src_flat = flat_indices[valid_edges]
            dst_flat = (batch_idx * N + dst_nodes)[valid_edges]
            
            # Compute vectors/dists
            ca_coords_flat = coords[:, :, 1, :].reshape(B * N, 3)
            vecs = ca_coords_flat[dst_flat] - ca_coords_flat[src_flat]
            dists = vecs.norm(dim=-1)
            
            all_edges_src.append(src_flat)
            all_edges_dst.append(dst_flat)
            all_types.append(torch.zeros(src_flat.size(0), device=device, dtype=torch.long))
            all_vecs.append(vecs)
            all_dists.append(dists)

    # 2. Spatial edges (k-NN) - Batched
    # CA coords: (B, N, 3)
    ca_coords = coords[:, :, 1, :]
    
    # Compute pairwise distances (B, N, N)
    dists_matrix = torch.cdist(ca_coords, ca_coords)  # (B, N, N)
    
    # Apply mask: set distances involving masked nodes to inf
    mask_bool = mask.bool()
    valid_pair_mask = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)  # (B, N, N)
    dists_matrix.masked_fill_(~valid_pair_mask, float('inf'))
    
    # Mask diagonal (self-loops)
    dists_matrix.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    
    # Mask distances > radius
    dists_matrix.masked_fill_(dists_matrix > spatial_radius, float('inf'))
    
    # Get k-NN
    k = min(k_spatial, N - 1)
    if k > 0:
        knn_vals, knn_idx = dists_matrix.topk(k, dim=-1, largest=False)
        
        # Valid neighbors are those with finite distance
        valid_knns = knn_vals < float('inf')  # (B, N, k)
        
        if valid_knns.any():
            # Source: indices repeated k times
            src_indices = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
            flat_src_base = batch_idx.unsqueeze(2) * N + src_indices
            
            # Dest: from knn_idx
            flat_dst_base = batch_idx.unsqueeze(2) * N + knn_idx
            
            # Filter
            src_flat = flat_src_base[valid_knns]
            dst_flat = flat_dst_base[valid_knns]
            
            # Vectors
            ca_coords_flat = coords[:, :, 1, :].reshape(B * N, 3)
            vecs = ca_coords_flat[dst_flat] - ca_coords_flat[src_flat]
            dists = vecs.norm(dim=-1)
            
            all_edges_src.append(src_flat)
            all_edges_dst.append(dst_flat)
            all_types.append(torch.ones(src_flat.size(0), device=device, dtype=torch.long))
            all_vecs.append(vecs)
            all_dists.append(dists)

    # 3. Hydrogen bond edges (vectorized)
    # N-O distance < 3.5Å
    if A >= 4:
        n_coords = coords[:, :, 0, :]  # (B, N, 3)
        o_coords = coords[:, :, 3, :]  # (B, N, 3)
        
        # Batched pairwise distance between N and O
        hb_dists = torch.cdist(n_coords, o_coords)
        
        # Masking
        # 1. Node validity
        valid_hb_mask = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
        hb_dists.masked_fill_(~valid_hb_mask, float('inf'))
        
        # 2. Seq separation
        idx = torch.arange(N, device=device)
        idx_diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (N, N)
        seq_mask = idx_diff >= 3
        seq_mask = seq_mask.unsqueeze(0).expand(B, N, N)
        hb_dists.masked_fill_(~seq_mask, float('inf'))
        
        # 3. Distance threshold
        hbond_hits = hb_dists < 3.5
        
        if hbond_hits.any():
            # Get indices
            b_idx, src_idx, dst_idx = torch.where(hbond_hits)
            
            src_flat = b_idx * N + src_idx
            dst_flat = b_idx * N + dst_idx
            
            # Vectors (CA-CA for consistency)
            ca_coords_flat = coords[:, :, 1, :].reshape(B * N, 3)
            vecs = ca_coords_flat[dst_flat] - ca_coords_flat[src_flat]
            dists = vecs.norm(dim=-1)
            
            all_edges_src.append(src_flat)
            all_edges_dst.append(dst_flat)
            # Edge type 2
            all_types.append(torch.full((src_flat.size(0),), 2, device=device, dtype=torch.long))
            all_vecs.append(vecs)
            all_dists.append(dists)
            
    if all_edges_src:
        src = torch.cat(all_edges_src)
        dst = torch.cat(all_edges_dst)
        edge_index = torch.stack([src, dst], dim=0)
        edge_type = torch.cat(all_types)
        edge_vec = torch.cat(all_vecs, dim=0)
        edge_dist = torch.cat(all_dists)
    else:
        edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
        edge_type = torch.zeros((0,), device=device, dtype=torch.long)
        edge_vec = torch.zeros((0, 3), device=device)
        edge_dist = torch.zeros((0,), device=device)
        
    return edge_index, edge_type, edge_vec, edge_dist


class EGNNLayer(nn.Module):
    """E(n)-equivariant graph neural network layer."""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Coordinate update (attention-weighted)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.edge_embed = nn.Linear(4, edge_dim)  # edge_type one-hot + distance
        
    def forward(
        self,
        x: torch.Tensor,           # (B*N, D)
        coords: torch.Tensor,      # (B*N, 3)
        edge_index: torch.Tensor,  # (2, E)
        edge_type: torch.Tensor,   # (E,)
        edge_dist: torch.Tensor,   # (E,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with equivariant updates."""
        src, dst = edge_index
        
        # Edge type one-hot (3 types)
        edge_type_onehot = F.one_hot(edge_type, num_classes=3).float()
        edge_feat = torch.cat([edge_type_onehot, edge_dist.unsqueeze(-1)], dim=-1)
        edge_feat = self.edge_embed(edge_feat)
        
        # Edge messages
        x_src = x[src]
        x_dst = x[dst]
        edge_input = torch.cat([x_src, x_dst, edge_dist.unsqueeze(-1), edge_feat], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        
        # Aggregate messages
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m_ij)
        
        # Node update
        x_new = x + self.node_mlp(torch.cat([x, agg], dim=-1))
        
        # Coordinate update (equivariant)
        coord_diff = coords[src] - coords[dst]
        coord_weight = self.coord_mlp(m_ij)
        coord_update = torch.zeros_like(coords)
        coord_update.index_add_(0, dst, coord_diff * coord_weight)
        coords_new = coords + coord_update * 0.1  # Small update
        
        return x_new, coords_new


class GearNetLayer(nn.Module):
    """Relational graph convolution with edge-type attention."""
    
    def __init__(self, hidden_dim: int, num_edge_types: int = 3, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Per-edge-type projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge type embeddings
        self.edge_type_embed = nn.Embedding(num_edge_types, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,           # (B*N, D)
        edge_index: torch.Tensor,  # (2, E)
        edge_type: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """Forward with edge-type attention."""
        src, dst = edge_index
        
        # Get edge type embeddings
        edge_emb = self.edge_type_embed(edge_type)  # (E, D)
        
        # Multi-head attention
        q = self.q_proj(x[dst]).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x[src] + edge_emb).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x[src]).view(-1, self.num_heads, self.head_dim)
        
        # Attention scores
        attn = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)  # (E, H)
        
        # Softmax per destination node
        attn_exp = attn.exp()
        attn_sum = torch.zeros(x.size(0), self.num_heads, device=x.device)
        attn_sum.index_add_(0, dst, attn_exp)
        attn_norm = attn_exp / (attn_sum[dst] + 1e-8)
        
        # Weighted aggregation
        v_weighted = v * attn_norm.unsqueeze(-1)  # (E, H, D/H)
        agg = torch.zeros(x.size(0), self.num_heads, self.head_dim, device=x.device)
        agg.index_add_(0, dst, v_weighted)
        agg = agg.view(-1, self.hidden_dim)
        
        # Output
        out = self.out_proj(agg)
        return self.norm(x + out)


class SpectralTopologyHead(nn.Module):
    """Spectral encoding for global topology awareness."""
    
    def __init__(self, hidden_dim: int, num_eigenvecs: int = 8):
        super().__init__()
        self.num_eigenvecs = num_eigenvecs
        self.proj = nn.Linear(num_eigenvecs, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Add spectral positional encoding."""
        device = x.device
        
        # Reshape to (B, N, D)
        x_batched = x.view(batch_size, seq_len, -1)
        
        # Compute approximate spectral encoding (simplified)
        # Using sinusoidal positional encoding as proxy
        pos = torch.arange(seq_len, device=device).float().unsqueeze(-1)
        freq = torch.exp(torch.arange(0, self.num_eigenvecs, device=device).float() * 
                        (-math.log(10000.0) / self.num_eigenvecs))
        spectral_enc = torch.sin(pos * freq)  # (N, num_eigenvecs)
        spectral_enc = spectral_enc.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, K)
        
        spectral_feat = self.proj(spectral_enc)  # (B, N, D)
        
        # Self-attention for global context
        x_with_spectral = x_batched + spectral_feat
        attn_out, _ = self.attn(x_with_spectral, x_with_spectral, x_with_spectral)
        
        out = self.norm(x_batched + attn_out)
        return out.view(-1, out.size(-1))


@register_encoder("hybrid")
class HybridEncoder(BaseProteinEncoder):
    """
    Hybrid EGNN-GearNet encoder combining geometric equivariance with relational structure.
    
    Bottleneck Class: C (equivariant field bottleneck)
    """
    
    bottleneck_class = BottleneckClass.C
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_egnn_layers: int = 3,
        num_gearnet_layers: int = 3,
        k_spatial: int = 16,
        spatial_radius: float = 10.0,
        freeze: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_spatial = k_spatial
        self.spatial_radius = spatial_radius
        
        # Initial embedding
        self.aa_embed = nn.Embedding(21, hidden_dim)  # 20 AA + padding
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # EGNN layers (equivariant)
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_egnn_layers)
        ])
        
        # GearNet layers (relational)
        self.gearnet_layers = nn.ModuleList([
            GearNetLayer(hidden_dim) for _ in range(num_gearnet_layers)
        ])
        
        # Spectral topology head
        self.spectral_head = SpectralTopologyHead(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        if freeze:
            self.freeze()
    
    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim
    
    def encode(self, protein_input: ProteinInput) -> EncoderOutput:
        """Encode protein structure."""
        coords = protein_input.coords  # (B, N, A, 3)
        seq = protein_input.sequence   # (B, N)
        mask = protein_input.mask      # (B, N)
        
        B, N, A, _ = coords.shape
        device = coords.device
        
        # Center coordinates for numerical stability
        center = torch.sum(coords[:, :, 1, :] * mask.unsqueeze(-1), dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
        coords = coords - center.unsqueeze(2)  # (B, N, A, 3)
        
        # Initial embeddings (Sequence only, geometric info comes from GNN layers)
        x = self.aa_embed(seq)  # (B, N, D)
        x = self.input_proj(x)  # (B, N, D)
        
        # Build multi-edge graph
        edge_index, edge_type, edge_vec, edge_dist = build_multi_edge_graph(
            coords, mask, self.k_spatial, self.spatial_radius
        )
        
        # Flatten for graph processing
        x_flat = x.view(B * N, -1)  # (B*N, D)
        ca_coords = coords[:, :, 1, :].reshape(B * N, 3)  # (B*N, 3)
        
        # EGNN layers
        for egnn in self.egnn_layers:
            x_flat, ca_coords = egnn(x_flat, ca_coords, edge_index, edge_type, edge_dist)
        
        # GearNet layers
        for gearnet in self.gearnet_layers:
            x_flat = gearnet(x_flat, edge_index, edge_type)
        
        # Spectral topology head
        x_flat = self.spectral_head(x_flat, edge_index, B, N)
        
        # Output
        x_out = self.output_proj(x_flat)
        x_out = x_out.view(B, N, -1)
        
        return EncoderOutput(
            residue_embeddings=x_out,
            global_embedding=x_out.mean(dim=1),
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights (not applicable for scratch model, but required by interface)."""
        if checkpoint_path and checkpoint_path != "none":
            print(f"Loading weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            print("HybridEncoder: Initialized with random weights (no checkpoint provided)")
    
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"HybridEncoder(bottleneck={self.bottleneck_class}, dim={self.hidden_dim}, params={n_params:.1f}M, frozen={self.is_frozen})"
