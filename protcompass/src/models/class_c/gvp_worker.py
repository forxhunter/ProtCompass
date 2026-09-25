#!/usr/bin/env python
"""
GVP worker script — runs in `gvp` conda env (Python 3.6, PyTorch 1.8).

Receives protein data via JSON, runs GVP encoding, saves result to .pt file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add gvp-pytorch to path
_GVP_REPO = Path(__file__).parent.parent.parent.parent / "external_repos" / "gvp-pytorch"
sys.path.insert(0, str(_GVP_REPO))

from gvp import GVPConvLayer, tuple_sum


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)


def build_knn_graph(ca_pos, k=30):
    B, L = ca_pos.shape[:2]
    device = ca_pos.device
    dist = torch.cdist(ca_pos, ca_pos)
    dist = dist + torch.eye(L, device=device).unsqueeze(0) * 1e6
    k = min(k, L - 1)
    dist_sorted, idx_sorted = dist.sort(dim=-1)
    neighbor_idx = idx_sorted[:, :, 1:k+1]
    all_src, all_dst, all_dist = [], [], []
    for b in range(B):
        src = torch.arange(L, device=device).unsqueeze(1).expand(-1, k).flatten()
        dst = neighbor_idx[b].flatten()
        d = dist_sorted[b, :, 1:k+1].flatten()
        all_src.append(src + b * L)
        all_dst.append(dst + b * L)
        all_dist.append(d)
    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
    return edge_index, torch.cat(all_dist)


def compute_edge_features(ca_pos, edge_index):
    B, L = ca_pos.shape[:2]
    device = ca_pos.device
    src = edge_index[0]
    dst = edge_index[1]
    batch_src = src // L
    local_src = (src % L).clamp(0, L - 1)
    batch_dst = dst // L
    local_dst = (dst % L).clamp(0, L - 1)
    diff = ca_pos[batch_dst, local_dst] - ca_pos[batch_src, local_src]
    dist = torch.norm(diff, dim=-1)
    edge_s = _rbf(dist, D_count=16, device=device)
    edge_v = (diff / dist.unsqueeze(-1).clamp(min=1e-8)).unsqueeze(1)
    return edge_s, edge_v


class GVPModel(nn.Module):
    def __init__(self, node_s_dim=100, node_v_dim=16, edge_s_dim=32,
                 edge_v_dim=1, num_layers=3, k_neighbors=30, dropout=0.1,
                 seed=None, state_dict_path=None):
        super(GVPModel, self).__init__()
        self.node_s_dim = node_s_dim
        self.node_v_dim = node_v_dim
        self.edge_v_dim = edge_v_dim
        self.k_neighbors = k_neighbors
        self._embedding_dim = node_s_dim + node_v_dim * 3

        # Set seed BEFORE initializing weights
        if seed is not None:
            self._seed_everything(seed)

        self.embed_s = nn.Linear(20, node_s_dim)
        self.embed_v = nn.Linear(3, node_v_dim, bias=False)
        self.edge_embed_s = nn.Linear(16, edge_s_dim)

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

        # Load serialized state dict if provided (for reproducibility)
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
            self.load_state_dict(state_dict)

    def _seed_everything(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def forward(self, coords, sequence, mask):
        device = coords.device
        B, L = coords.shape[:2]

        ca_pos = coords[:, :, 1, :].float()
        seq = sequence.clamp(0, 19)
        node_s = F.one_hot(seq, num_classes=20).float()
        node_s = self.embed_s(node_s)

        N = coords[:, :, 0, :].float()
        C = coords[:, :, 2, :].float()
        forward = F.normalize(C - ca_pos, dim=-1)
        backward = F.normalize(N - ca_pos, dim=-1)
        node_v = torch.stack([forward, backward], dim=-2)  # (B, L, 2, 3)
        if node_v.shape[-2] >= self.node_v_dim:
            node_v = node_v[:, :, :self.node_v_dim, :]
        else:
            pad = torch.zeros(B, L, self.node_v_dim - node_v.shape[-2], 3, device=device)
            node_v = torch.cat([node_v, pad], dim=-2)

        edge_index, _ = build_knn_graph(ca_pos, self.k_neighbors)
        edge_s, edge_v = compute_edge_features(ca_pos, edge_index)
        edge_s = self.edge_embed_s(edge_s)
        if self.edge_v_dim > 1:
            edge_v = edge_v.expand(-1, self.edge_v_dim, -1)

        node_s_flat = node_s.reshape(B * L, self.node_s_dim)
        node_v_flat = node_v.reshape(B * L, self.node_v_dim, 3)
        mask_flat = mask.reshape(-1).float()

        node_features = (node_s_flat, node_v_flat)
        edge_attr = (edge_s, edge_v)

        for layer in self.layers:
            node_features = layer(node_features, edge_index.long(), edge_attr)
            node_s_out, node_v_out = node_features
            node_s_out = self.dropout(node_s_out)
            node_features = (node_s_out, node_v_out)

        node_s_out = node_features[0]
        node_v_out = node_features[1]

        node_s_out = node_s_out * mask_flat.unsqueeze(-1)
        node_v_out = node_v_out * mask_flat.unsqueeze(-1).unsqueeze(-1)

        node_v_flat_out = node_v_out.reshape(B, L, -1)
        embeddings = torch.cat([node_s_out.reshape(B, L, -1), node_v_flat_out], dim=-1)

        mask_f = mask.float().unsqueeze(-1)
        global_emb = (embeddings * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        return embeddings, global_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--state-dict", type=str, default=None,
                        help="Path to serialized model state dict for reproducibility")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for initialization")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    coords   = torch.tensor(data["coords"], dtype=torch.float32)
    sequence = torch.tensor(data["sequence"], dtype=torch.long)
    mask     = torch.tensor(data["mask"], dtype=torch.bool)

    model = GVPModel(
        node_s_dim=data["node_s_dim"],
        node_v_dim=data["node_v_dim"],
        edge_s_dim=data["edge_s_dim"],
        edge_v_dim=data["edge_v_dim"],
        num_layers=data["num_layers"],
        k_neighbors=data["k_neighbors"],
        dropout=data["dropout"],
        seed=args.seed,
        state_dict_path=args.state_dict,
    )
    model.eval()

    with torch.no_grad():
        embeddings, global_emb = model(coords, sequence, mask)

    result = {
        "embeddings": embeddings.cpu().numpy(),
        "global_embedding": global_emb.cpu().numpy(),
    }
    torch.save(result, data["output_path"])


if __name__ == "__main__":
    main()
