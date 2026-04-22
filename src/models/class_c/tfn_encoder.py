"""
Tensor Field Network (TFN) Encoder using e3nn.
TFN (Thomas et al., 2018) is the foundational 3D equivariant architecture.
e3nn (Geiger & Smidt, 2022) is its official PyTorch successor.
Reference:
- N. Thomas et al., arXiv:1802.08219
- M. Geiger and T. Smidt, arXiv:2207.09453
"""

import torch
import torch.nn as nn
from src.models.base_encoder import (
    BaseProteinEncoder, BottleneckClass, EncoderOutput, ProteinInput, register_encoder,
)
from e3nn import o3


class EdgeFeatures(nn.Module):
    def __init__(self, irreps_sh: str = "0e + 1o + 2e"):
        super().__init__()
        self.sh = o3.SphericalHarmonics(
            irreps_sh, normalize=True, normalization="norm"
        )
        self.irreps_out = o3.Irreps(irreps_sh)

    def forward(self, edge_vec: torch.Tensor) -> torch.Tensor:
        return self.sh(edge_vec)


def build_knn_edges_packed(coords: torch.Tensor, mask: torch.Tensor, k: int = 16) -> torch.Tensor:
    B, N, _ = coords.shape
    device = coords.device
    dists = torch.cdist(coords, coords)
    mask_bool = mask.bool()
    valid = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
    dists.masked_fill_(~valid, float("inf"))
    dists.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    k_actual = min(k, N - 1)
    if k_actual <= 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    _, knn_idx = dists.topk(k_actual, dim=-1, largest=False)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1)
    n_idx = torch.arange(N, device=device).view(1, N, 1)
    dst_packed = (batch_idx * N + n_idx).expand(B, N, k_actual)
    src_packed = (batch_idx * N + knn_idx)
    dst_flat = dst_packed.reshape(-1)
    src_flat = src_packed.reshape(-1)
    # Filter edges where source or destination is masked
    mask_flat = mask_bool.view(-1)
    valid_edges = mask_flat[dst_flat] & mask_flat[src_flat]
    return torch.stack([dst_flat[valid_edges], src_flat[valid_edges]], dim=0)


class TFNConvLayer(nn.Module):
    def __init__(self, node_irreps, edge_irreps, hidden_irreps, num_neighbors=16.0):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.tp = o3.FullyConnectedTensorProduct(
            node_irreps, edge_irreps, hidden_irreps, internal_weights=True
        )
        num_scalars = sum(m for m, ir in hidden_irreps if ir.is_scalar())
        self.gate_net = nn.Linear(num_scalars, num_scalars) if num_scalars > 0 else None
        self.update = nn.Linear(hidden_irreps.dim, node_irreps.dim)

    def forward(self, node_features, edge_index, edge_sh):
        dst, src = edge_index
        messages = self.tp(node_features[src], edge_sh)
        N = node_features.shape[0]
        aggregated = torch.zeros(N, messages.shape[1], device=node_features.device)
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        num_incoming = torch.zeros(N, device=node_features.device)
        num_incoming.scatter_add_(0, dst, torch.ones(dst.shape[0], device=node_features.device))
        num_incoming = torch.clamp(num_incoming, min=1.0)
        aggregated = aggregated / num_incoming.unsqueeze(1)
        aggregated = aggregated * (self.num_neighbors ** 0.5)
        if self.gate_net is not None:
            scalars = aggregated[:, :self.gate_net.in_features]
            gates = torch.sigmoid(self.gate_net(scalars))
            aggregated = aggregated * gates.mean(dim=1, keepdim=True)
        return self.update(aggregated)


class TFNEncoderCore(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=128, num_layers=4,
                 edge_irreps="0e + 1o + 2e", num_neighbors=16.0):
        super().__init__()
        self.node_irreps = o3.Irreps(f"{hidden_dim}x0e")
        self.edge_irreps_obj = o3.Irreps(edge_irreps)
        self.aa_embed = nn.Embedding(input_dim, hidden_dim)
        self.sh_layer = EdgeFeatures(irreps_sh=edge_irreps)
        self.tfn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        for i in range(num_layers):
            self.tfn_layers.append(TFNConvLayer(
                self.node_irreps, self.edge_irreps_obj, self.node_irreps, num_neighbors))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.skip_proj.append(nn.Linear(hidden_dim, hidden_dim) if i > 0 else nn.Identity())

    def forward(self, node_features, coords, edge_index):
        dst, src = edge_index
        pos_flat = coords.reshape(-1, 3)
        edge_vec = pos_flat[dst] - pos_flat[src]
        edge_sh = self.sh_layer(edge_vec)
        h = node_features
        for layer, norm, skip in zip(self.tfn_layers, self.norms, self.skip_proj):
            h = norm(skip(h) + layer(h, edge_index, edge_sh))
        return h


@register_encoder("tfn")
class TFNEncoder(BaseProteinEncoder):
    bottleneck_class = BottleneckClass.C

    def __init__(self, hidden_dim=128, depth=4, k_neighbors=16,
                 edge_irreps="0e + 1o + 2e", freeze=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.k_neighbors = k_neighbors
        self.core = TFNEncoderCore(
            input_dim=21, hidden_dim=hidden_dim, num_layers=depth,
            edge_irreps=edge_irreps, num_neighbors=float(k_neighbors))
        if freeze:
            self.freeze()

    @property
    def embedding_dim(self):
        return self.hidden_dim

    def encode(self, protein_input: ProteinInput) -> EncoderOutput:
        coords = protein_input.coords
        seq = protein_input.sequence
        mask = protein_input.mask
        B, N, A, _ = coords.shape
        ca = coords[:, :, 1, :]
        edge_index = build_knn_edges_packed(ca, mask, self.k_neighbors)
        h = self.core.aa_embed(seq.reshape(-1).clamp(0, 20))
        h = self.core(h, ca, edge_index)
        h = h.view(B, N, -1) * mask.unsqueeze(-1)
        global_emb = h.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        return EncoderOutput(residue_embeddings=h, global_embedding=global_emb)

    def load_pretrained(self, checkpoint_path: str) -> None:
        if checkpoint_path and checkpoint_path != "none":
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters()) / 1e6
        return f"TFNEncoder(layers={self.depth}, dim={self.hidden_dim}, params={n:.1f}M)"
