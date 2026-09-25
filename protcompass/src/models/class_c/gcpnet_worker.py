#!/usr/bin/env python
"""
GCPNet worker script — runs in `gcpnet` conda env.

Receives protein data via JSON, runs GCPNet encoding, saves result to .pt file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add GCPNet paths for imports.
# We run from GCPNet root (cwd set by encoder), so we need both:
# - GCPNet root for `from src import utils`
# - GCPNet/src for `from models.components import ...`
_GCPNET_ROOT = Path(__file__).parent.parent.parent.parent / "external_repos" / "GCPNet"
_GCPNET_SRC = _GCPNET_ROOT / "src"
sys.path.insert(0, str(_GCPNET_ROOT))   # makes `from src import utils` work
sys.path.insert(0, str(_GCPNET_SRC))    # makes `from models.components import ...` work

from models.components import (
    ScalarVector,
    GCPLayerNorm,
    GCPDropout,
    localize,
)
from models.components.gcpnet import GCP


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """Radial basis function encoding of distances."""
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    RBF = torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)
    return RBF


def build_knn_graph(ca_pos: torch.Tensor, k: int = 30) -> torch.Tensor:
    """Build k-NN graph over CA positions. Returns edge_index (2, n_edges)."""
    B, L = ca_pos.shape[:2]
    device = ca_pos.device

    dist = torch.cdist(ca_pos, ca_pos)
    dist = dist + torch.eye(L, device=device).unsqueeze(0) * 1e6

    k = min(k, L - 1)
    _, idx_sorted = dist.sort(dim=-1)
    neighbor_idx = idx_sorted[:, :, 1:k+1]

    all_src, all_dst = [], []
    for b in range(B):
        src = torch.arange(L, device=device).unsqueeze(1).expand(-1, k).flatten()
        dst = neighbor_idx[b].flatten()
        all_src.append(src + b * L)
        all_dst.append(dst + b * L)

    edge_index = torch.stack([
        torch.cat(all_src),
        torch.cat(all_dst),
    ], dim=0)

    return edge_index


class _GCPMessagePassing(nn.Module):
    """Single GCP message passing layer."""

    def __init__(self, node_dims: ScalarVector, edge_dims: ScalarVector, dropout: float = 0.1):
        super().__init__()
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.gcp = GCP(
            input_dims=node_dims,
            output_dims=node_dims,
            nonlinearities=("relu", "sigmoid"),
            scalar_gate=1,
            vector_gate=True,
            frame_gate=False,
            enable_e3_equivariance=False,
        )
        self.norm = GCPLayerNorm(node_dims)
        self.dropout = GCPDropout(dropout)

    def forward(self, node_feats: ScalarVector, edge_index: torch.Tensor,
                edge_s: torch.Tensor, frames: torch.Tensor) -> ScalarVector:
        updated = self.gcp(node_feats, edge_index, frames, node_inputs=True)
        updated = self.dropout(updated)
        return updated


class _GCPNetModel(nn.Module):
    """GCPNet encoder model (used in subprocess)."""

    def __init__(
        self,
        node_s_dim: int = 128,
        node_v_dim: int = 16,
        edge_s_dim: int = 32,
        edge_v_dim: int = 1,
        num_layers: int = 3,
        k_neighbors: int = 30,
        dropout: float = 0.1,
        pretrained_path: str = None,
        seed: int = None,
    ):
        super().__init__()
        self.node_s_dim = node_s_dim
        self.node_v_dim = node_v_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors

        # Set seed BEFORE initializing weights
        if seed is not None:
            self._seed_everything(seed)

        node_dims = ScalarVector(node_s_dim, node_v_dim)
        edge_dims = ScalarVector(edge_s_dim, edge_v_dim)

        self.node_embedding_s = nn.Linear(20 + 1, node_s_dim)
        # Project 2 input vectors (forward, backward) to node_v_dim vectors.
        # Linear acts on the vector-count dim (2), not the spatial dim (3),
        # so equivariance is preserved and output shape is [B,L,node_v_dim,3].
        self.node_embedding_v = nn.Linear(2, node_v_dim, bias=False)
        self.edge_embedding_s = nn.Linear(16, edge_s_dim)

        self.layers = nn.ModuleList([
            _GCPMessagePassing(node_dims, edge_dims, dropout)
            for _ in range(num_layers)
        ])

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            if "encoder_state" in ckpt:
                self.load_state_dict(ckpt["encoder_state"], strict=False)
            else:
                self.load_state_dict(ckpt, strict=False)

    def _seed_everything(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def forward(self, coords: torch.Tensor, sequence: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            coords: (B, L, 5, 3) CA at index 1
            sequence: (B, L)
            mask: (B, L)
        """
        device = coords.device
        B, L = coords.shape[:2]

        ca_pos = coords[:, :, 1, :].float()
        seq = sequence.clamp(0, 19)
        mask_b = mask.bool()

        # Node scalar features
        aa_one_hot = F.one_hot(seq, num_classes=20).float()
        atom_type = torch.ones(B, L, 1, device=device)
        node_input_s = torch.cat([aa_one_hot, atom_type], dim=-1)
        node_s = self.node_embedding_s(node_input_s)

        # Node vector features: [B, L, 2, 3] (forward and backward unit vectors)
        N = coords[:, :, 0, :].float()
        C = coords[:, :, 2, :].float()
        forward = F.normalize(C - ca_pos, dim=-1)
        backward = F.normalize(N - ca_pos, dim=-1)
        node_v = torch.stack([forward, backward], dim=-2)  # [B, L, 2, 3]

        # Project 2 vectors -> node_v_dim vectors via Linear(2, node_v_dim).
        # Transpose so Linear acts on vector-count dim, then transpose back.
        # [B,L,2,3] -> [B,L,3,2] -> Linear -> [B,L,3,node_v_dim] -> [B,L,node_v_dim,3]
        node_v = self.node_embedding_v(node_v.transpose(-1, -2)).transpose(-1, -2)

        node_s_flat = node_s.reshape(B * L, node_s.shape[-1])
        node_v_flat = node_v.reshape(B * L, self.node_v_dim, 3)
        node_feats = ScalarVector(node_s_flat, node_v_flat)

        # k-NN graph
        edge_index = build_knn_graph(ca_pos, self.k_neighbors)

        # Batch-aware edge index
        batch_offset = torch.arange(B, device=device).unsqueeze(1) * L
        edge_index_batch = edge_index.clone()
        for b in range(B):
            mask_batch = (edge_index[0] >= b * L) & (edge_index[0] < (b + 1) * L)
            edge_index_batch[0][mask_batch] = edge_index[0][mask_batch] - b * L + b * L
            edge_index_batch[1][mask_batch] = edge_index[1][mask_batch] - b * L + b * L

        # Edge scalar features
        src = edge_index[0]
        dst = edge_index[1]
        batch_src = src // L
        batch_dst = dst // L
        local_src = src % L
        local_dst = dst % L

        dist = torch.norm(
            ca_pos[batch_dst, local_dst] - ca_pos[batch_src, local_src],
            dim=-1
        )
        edge_s = self.edge_embedding_s(_rbf(dist, device=device))

        # Local coordinate frames
        frames = localize(
            ca_pos.reshape(B * L, 3),
            edge_index_batch.long(),
            norm_x_diff=True,
            node_mask=None,
        )

        # GCP message passing
        for layer in self.layers:
            node_feats = layer(node_feats, edge_index_batch.long(), edge_s, frames)

        node_s_out = node_feats.scalar
        node_v_out = node_feats.vector

        # Mask
        mask_flat = mask.reshape(-1).float()
        node_s_out = node_s_out * mask_flat.unsqueeze(-1)
        node_v_out = node_v_out * mask_flat.unsqueeze(-1).unsqueeze(-1)

        # Per-residue embeddings
        node_v_flat_out = node_v_out.reshape(B, L, -1)
        embeddings = torch.cat([
            node_s_out.reshape(B, L, -1),
            node_v_flat_out,
        ], dim=-1)

        # Global pooling
        mask_f = mask.float().unsqueeze(-1)
        global_emb = (embeddings * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        return embeddings, global_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON input file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    # Deserialize
    coords = torch.tensor(data["coords"], dtype=torch.float32)
    sequence = torch.tensor(data["sequence"], dtype=torch.long)
    mask = torch.tensor(data["mask"], dtype=torch.bool)

    # Build model
    model = _GCPNetModel(
        node_s_dim=data["node_s_dim"],
        node_v_dim=data["node_v_dim"],
        edge_s_dim=data["edge_s_dim"],
        edge_v_dim=data["edge_v_dim"],
        num_layers=data["num_layers"],
        k_neighbors=data["k_neighbors"],
        dropout=data["dropout"],
        pretrained_path=data.get("pretrained_path"),
        seed=data.get("seed"),
    )
    model.eval()

    with torch.no_grad():
        embeddings, global_emb = model(coords, sequence, mask)

    # Save as numpy (compatible across envs)
    result = {
        "embeddings": embeddings.cpu().numpy(),
        "global_embedding": global_emb.cpu().numpy(),
    }
    torch.save(result, data["output_path"])


if __name__ == "__main__":
    main()
