"""
GearNet-Edge pretrained encoder wrapper.

Loads the official Multiview Contrast pretrained GearNet-Edge checkpoint
from TorchDrug (Zhang et al. 2022).

Architecture details (derived from checkpoint inspection):
- 6 node message-passing layers with 7 relation types
- 6 edge message-passing layers with 8 relation types
- hidden_dim=512, BatchNorm, ReLU
- Node input: 21-dim one-hot AA
- Edge input: 59-dim geometric features (distance, angle, dihedral + types)

Reference:
    Zhang et al. (2022). "Protein Representation Learning by Geometric Structure Pretraining"
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


class GearNetEdgeLayer(nn.Module):
    """Single GearNet-Edge message passing layer (node update).

    Matches TorchDrug's GeometricRelationalGraphConv:
    - linear: maps concatenated relation messages -> hidden
    - self_loop: skip connection on node features
    - batch_norm: batch normalization
    """

    def __init__(self, input_dim: int, output_dim: int, num_relation: int = 7):
        super().__init__()
        # linear takes concat of (num_relation * input_dim) -> output_dim
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.num_relation = num_relation
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,          # (N, input_dim) - all node features flat
        edge_index: torch.Tensor,  # (2, E) - source, target
        edge_type: torch.Tensor,   # (E,) - relation type per edge
        num_nodes: int,
    ) -> torch.Tensor:
        """Message passing: aggregate neighbor features by relation type."""
        N = num_nodes
        D = self.input_dim
        R = self.num_relation

        # Gather source node features for each edge
        src, dst = edge_index[0], edge_index[1]
        msg = x[src]  # (E, D)

        # Aggregate by relation type into (N, R, D)
        agg = torch.zeros(N, R, D, device=x.device)
        for r in range(R):
            mask = (edge_type == r)
            if mask.any():
                # Scatter-add messages of this relation type to destination nodes
                idx = dst[mask].unsqueeze(-1).expand(-1, D)
                agg[:, r, :].scatter_add_(0, idx, msg[mask])

        # Flatten relations: (N, R*D)
        agg_flat = agg.reshape(N, R * D)

        # Linear transform + self-loop + batch norm + relu
        out = self.linear(agg_flat) + self.self_loop(x)
        out = self.batch_norm(out)
        out = F.relu(out)
        return out


class GearNetEdgeEdgeLayer(nn.Module):
    """Edge message passing layer for GearNet-Edge.

    Updates edge features by aggregating from neighboring edges.
    """

    def __init__(self, input_dim: int, output_dim: int, num_relation: int = 8):
        super().__init__()
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.num_relation = num_relation
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(
        self,
        edge_feat: torch.Tensor,       # (E, input_dim)
        edge_edge_index: torch.Tensor,  # (2, EE) - source edge, target edge
        edge_edge_type: torch.Tensor,   # (EE,) - edge-edge relation type
        num_edges: int,
    ) -> torch.Tensor:
        E = num_edges
        D = self.input_dim
        R = self.num_relation

        src, dst = edge_edge_index[0], edge_edge_index[1]
        msg = edge_feat[src]  # (EE, D)

        agg = torch.zeros(E, R, D, device=edge_feat.device)
        for r in range(R):
            mask = (edge_edge_type == r)
            if mask.any():
                idx = dst[mask].unsqueeze(-1).expand(-1, D)
                agg[:, r, :].scatter_add_(0, idx, msg[mask])

        agg_flat = agg.reshape(E, R * D)
        out = self.linear(agg_flat) + self.self_loop(edge_feat)
        out = self.batch_norm(out)
        out = F.relu(out)
        return out


def _compute_edge_features(
    ca_coords: torch.Tensor,  # (N, 3)
    edge_index: torch.Tensor,  # (2, E)
    edge_type: torch.Tensor,   # (E,)
    sequence: torch.Tensor,    # (N,) AA indices
    num_relation: int = 7,
) -> torch.Tensor:
    """Compute 59-dim edge features matching TorchDrug GearNet-Edge.

    Features: sequential_distance(1) + spatial_distance(1) +
              direction_unit(3) + edge_type_onehot(7) +
              residue_type_src_onehot(21) + residue_type_dst_onehot(21) +
              sin_features(3) + cos_features(2) = 59
    """
    src, dst = edge_index[0], edge_index[1]

    # Spatial features
    diff = ca_coords[dst] - ca_coords[src]  # (E, 3)
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)
    direction = diff / dist  # (E, 3)

    # Sequential distance
    seq_dist = (dst - src).float().unsqueeze(-1)  # (E, 1)

    # Edge type one-hot
    edge_type_oh = F.one_hot(edge_type, num_classes=num_relation).float()  # (E, 7)

    # Residue type one-hot (21 classes)
    src_aa = F.one_hot(sequence[src].clamp(0, 20), num_classes=21).float()  # (E, 21)
    dst_aa = F.one_hot(sequence[dst].clamp(0, 20), num_classes=21).float()  # (E, 21)

    # Angular features (sin/cos of angles between direction and axes)
    # 3 sin features + 2 cos features = 5
    sin_feats = torch.sin(direction)  # (E, 3)
    cos_feats = torch.cos(direction[:, :2])  # (E, 2)

    # Concatenate: 1 + 1 + 3 + 7 + 21 + 21 + 3 + 2 = 59
    edge_feat = torch.cat([
        seq_dist,
        dist,
        direction,
        edge_type_oh,
        src_aa,
        dst_aa,
        sin_feats,
        cos_feats,
    ], dim=-1)

    return edge_feat


def _build_edge_edge_graph(
    edge_index: torch.Tensor,  # (2, E)
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build edge-edge adjacency: two edges sharing a node are connected.

    For each node v, collects all incident edges (both incoming and outgoing),
    then connects all pairs. Relation type (0-7) encodes which ends touch v:

    For unordered pair {e1, e2} sharing node v:
      base_type = e1_end * 2 + e2_end  (end: 0=src, 1=dst)
        Type 0: both have v as src  (src-src)
        Type 1: e1 src, e2 dst      (src-dst)
        Type 2: e1 dst, e2 src      (dst-src)
        Type 3: both have v as dst  (dst-dst)
      Forward edge (e1→e2): base_type
      Backward edge (e2→e1): base_type + 4
    """
    src, dst = edge_index[0], edge_index[1]
    E = edge_index.shape[1]
    device = edge_index.device

    if E == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    MAX_INCIDENT = 16  # max incident edges per node to consider

    # Create expanded index: each edge appears twice (once per endpoint)
    # (node, edge_id, end_type) where end_type: 0=src, 1=dst
    edge_ids = torch.arange(E, device=device)
    all_nodes = torch.cat([src, dst])                     # (2*E,)
    all_edge_ids = torch.cat([edge_ids, edge_ids])        # (2*E,)
    all_end_types = torch.cat([
        torch.zeros(E, dtype=torch.long, device=device),  # src end
        torch.ones(E, dtype=torch.long, device=device),   # dst end
    ])                                                     # (2*E,)

    # Sort by node to group incident edges
    sorted_idx = torch.argsort(all_nodes)
    sorted_nodes = all_nodes[sorted_idx]
    sorted_edges = all_edge_ids[sorted_idx]
    sorted_ends = all_end_types[sorted_idx]

    # Find group boundaries
    change = torch.cat([
        torch.tensor([True], device=device),
        sorted_nodes[1:] != sorted_nodes[:-1],
        torch.tensor([True], device=device),
    ])
    boundaries = change.nonzero(as_tuple=True)[0]

    ee_src_all = []
    ee_dst_all = []
    ee_type_all = []

    for g in range(len(boundaries) - 1):
        start, end = boundaries[g].item(), boundaries[g + 1].item()
        group_size = end - start
        if group_size < 2:
            continue

        g_edges = sorted_edges[start:end]
        g_ends = sorted_ends[start:end]

        # Limit group size for memory
        if group_size > MAX_INCIDENT:
            g_edges = g_edges[:MAX_INCIDENT]
            g_ends = g_ends[:MAX_INCIDENT]
            group_size = MAX_INCIDENT

        # Form unordered pairs (i < j) to avoid duplicates
        ii = torch.arange(group_size, device=device)
        e1_local = ii.unsqueeze(1).expand(-1, group_size).reshape(-1)
        e2_local = ii.unsqueeze(0).expand(group_size, -1).reshape(-1)
        valid = e1_local < e2_local
        e1_local = e1_local[valid]
        e2_local = e2_local[valid]

        e1_edges = g_edges[e1_local]
        e2_edges = g_edges[e2_local]
        e1_end = g_ends[e1_local]
        e2_end = g_ends[e2_local]

        # Skip pairs that are the same edge (self-loops in edge graph)
        diff_edges = e1_edges != e2_edges
        e1_edges = e1_edges[diff_edges]
        e2_edges = e2_edges[diff_edges]
        e1_end = e1_end[diff_edges]
        e2_end = e2_end[diff_edges]

        if len(e1_edges) == 0:
            continue

        # base_type = e1_end * 2 + e2_end → {0,1,2,3}
        base_type = e1_end * 2 + e2_end

        # Forward: e1 → e2 with base_type
        ee_src_all.append(e1_edges)
        ee_dst_all.append(e2_edges)
        ee_type_all.append(base_type)

        # Backward: e2 → e1 with base_type + 4
        ee_src_all.append(e2_edges)
        ee_dst_all.append(e1_edges)
        ee_type_all.append(base_type + 4)

    if not ee_src_all:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    ee_index = torch.stack([torch.cat(ee_src_all), torch.cat(ee_dst_all)])
    ee_type = torch.cat(ee_type_all)

    return ee_index, ee_type


@register_encoder("gearnet_pretrained")
class GearNetPretrainedEncoder(BaseProteinEncoder):
    """
    GearNet-Edge pretrained encoder using official Multiview Contrast weights.

    Architecture matches TorchDrug's GearNet-Edge with:
    - 6 layers, 512 hidden dim, 7 node relations, 8 edge relations
    - BatchNorm, ReLU activations
    - Edge message passing between adjacent edges

    Output: 512-dim per-residue embeddings (sum of all layer outputs).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_node_relation: int = 7,
        num_edge_relation: int = 8,
        edge_input_dim: int = 59,
        k_neighbors: int = 10,
        radius: float = 10.0,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_node_relation = num_node_relation
        self.num_edge_relation = num_edge_relation
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.edge_input_dim = edge_input_dim

        # Node message passing layers
        # Layer 0: input_dim=21 (one-hot AA)
        self.layers = nn.ModuleList()
        self.layers.append(GearNetEdgeLayer(21, hidden_dim, num_node_relation))
        for _ in range(1, num_layers):
            self.layers.append(GearNetEdgeLayer(hidden_dim, hidden_dim, num_node_relation))

        # Edge message passing layers
        # Layer 0: edge_input_dim -> 21
        # Layer 1: 21 -> hidden_dim
        # Layers 2-5: hidden_dim -> hidden_dim
        self.edge_layers = nn.ModuleList()
        self.edge_layers.append(GearNetEdgeEdgeLayer(edge_input_dim, 21, num_edge_relation))
        self.edge_layers.append(GearNetEdgeEdgeLayer(21, hidden_dim, num_edge_relation))
        for _ in range(2, num_layers):
            self.edge_layers.append(GearNetEdgeEdgeLayer(hidden_dim, hidden_dim, num_edge_relation))

        # Additional batch norms (on node outputs before residual)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B

    @property
    def embedding_dim(self) -> int:
        return self._hidden_dim

    def load_pretrained(self, checkpoint_path: str):
        """Load official GearNet-Edge Multiview Contrast checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Keys should match directly since we replicated the TorchDrug naming
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        loaded = len(state_dict) - len(unexpected)
        total = len(state_dict)
        print(f"GearNet-Edge pretrained: loaded {loaded}/{total} keys")
        if missing:
            print(f"  Missing: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected: {unexpected[:5]}...")
        self._needs_bn_recalibration = True

    def recalibrate_bn(self, data_loader, device="cpu", max_batches=200):
        """Recalibrate BatchNorm running stats on our data distribution.

        TorchDrug computes edge features differently, so the checkpoint's
        running_mean/running_var don't match our input distribution. Running
        a forward pass in training mode updates these statistics.
        """
        self.to(device)
        self.train()
        print("  Recalibrating BatchNorm statistics...")
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= max_batches:
                    break
                protein = ProteinInput(
                    coords=batch["coords"].to(device),
                    sequence=batch["sequence"].to(device),
                    mask=batch["mask"].to(device),
                )
                self.encode(protein)
        self.eval()
        self._needs_bn_recalibration = False
        print("  BatchNorm recalibration complete.")

    def _build_graph(
        self,
        ca_coords: torch.Tensor,  # (N, 3) - flat CA coords
        seq_ids: torch.Tensor,    # (N,) - which protein each residue belongs to
        sequence: torch.Tensor,   # (N,) - AA indices
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build multi-relational graph with 7 edge types (vectorized).

        Relations: sequential_fwd(0), sequential_bwd(1),
                   radius_fwd(2), radius_bwd(3),
                   knn_fwd(4), knn_bwd(5), long_range(6)
        """
        N = ca_coords.shape[0]
        device = ca_coords.device

        all_src = []
        all_dst = []
        all_type = []

        unique_ids = seq_ids.unique()

        for pid in unique_ids:
            mask = (seq_ids == pid)
            indices = mask.nonzero(as_tuple=True)[0]
            n = len(indices)
            if n < 2:
                continue

            local_coords = ca_coords[indices]  # (n, 3)

            # Type 0/1: Sequential edges (vectorized)
            seq_src = indices[:-1]
            seq_dst = indices[1:]
            all_src.append(seq_src)
            all_dst.append(seq_dst)
            all_type.append(torch.zeros(n - 1, dtype=torch.long, device=device))
            all_src.append(seq_dst)
            all_dst.append(seq_src)
            all_type.append(torch.ones(n - 1, dtype=torch.long, device=device))

            # Pairwise distances
            dist = torch.cdist(local_coords, local_coords)  # (n, n)
            dist.fill_diagonal_(float('inf'))

            # Type 2/3: Radius edges (vectorized)
            ri, rj = (dist < self.radius).nonzero(as_tuple=True)
            if len(ri) > 0:
                all_src.append(indices[ri])
                all_dst.append(indices[rj])
                all_type.append(torch.full((len(ri),), 2, dtype=torch.long, device=device))
                all_src.append(indices[rj])
                all_dst.append(indices[ri])
                all_type.append(torch.full((len(ri),), 3, dtype=torch.long, device=device))

            # Type 4/5: KNN edges (vectorized)
            k = min(self.k_neighbors, n - 1)
            _, knn_idx = dist.topk(k, dim=-1, largest=False)  # (n, k)
            knn_src = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
            knn_dst = knn_idx.reshape(-1)
            all_src.append(indices[knn_src])
            all_dst.append(indices[knn_dst])
            all_type.append(torch.full((n * k,), 4, dtype=torch.long, device=device))
            all_src.append(indices[knn_dst])
            all_dst.append(indices[knn_src])
            all_type.append(torch.full((n * k,), 5, dtype=torch.long, device=device))

            # Type 6: Long-range edges (>20 residues apart, within radius)
            row_idx = torch.arange(n, device=device)
            seq_dist = (row_idx.unsqueeze(1) - row_idx.unsqueeze(0)).abs()
            lr_mask = (seq_dist > 20) & (dist < self.radius)
            lr_i, lr_j = lr_mask.nonzero(as_tuple=True)
            if len(lr_i) > 0:
                all_src.append(indices[lr_i])
                all_dst.append(indices[lr_j])
                all_type.append(torch.full((len(lr_i),), 6, dtype=torch.long, device=device))

        if not all_src:
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)])
        edge_type = torch.cat(all_type)

        return edge_index, edge_type

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        B, L = protein.sequence.shape
        device = protein.coords.device

        # Flatten batch into single graph
        # Use CA coordinates (atom index 1)
        ca_coords_list = []
        seq_list = []
        seq_id_list = []
        mask_flat = []

        for b in range(B):
            n = protein.mask[b].sum().item()
            ca_coords_list.append(protein.coords[b, :n, 1, :])  # CA coords
            seq_list.append(protein.sequence[b, :n])
            seq_id_list.append(torch.full((n,), b, device=device))
            mask_flat.append(torch.ones(n, dtype=torch.bool, device=device))

        ca_coords = torch.cat(ca_coords_list, dim=0)  # (N_total, 3)
        sequence = torch.cat(seq_list, dim=0)          # (N_total,)
        seq_ids = torch.cat(seq_id_list, dim=0)        # (N_total,)
        N = ca_coords.shape[0]

        # Build graph
        edge_index, edge_type = self._build_graph(ca_coords, seq_ids, sequence)
        E = edge_index.shape[1]

        # Node input: one-hot AA (21-dim)
        x = F.one_hot(sequence.clamp(0, 20), num_classes=21).float()  # (N, 21)

        # Edge features
        if E > 0:
            edge_feat = _compute_edge_features(
                ca_coords, edge_index, edge_type, sequence,
                num_relation=self.num_node_relation,
            )
        else:
            edge_feat = torch.zeros(0, self.edge_input_dim, device=device)

        # Build edge-edge graph for edge message passing
        if E > 0:
            ee_index, ee_type = _build_edge_edge_graph(edge_index, N)
        else:
            ee_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            ee_type = torch.zeros(0, dtype=torch.long, device=device)

        # Message passing (matches TorchDrug GearNet forward)
        layer_outputs = []
        for i in range(self.num_layers):
            x_prev = x
            # Node update (internal: linear + self_loop + BN + ReLU)
            x = self.layers[i](x, edge_index, edge_type, N)

            # Residual connection (short_cut) — skip layer 0 due to dim change
            if x.shape == x_prev.shape:
                x = x + x_prev

            # External batch norm + activation
            x = self.batch_norms[i](x)
            x = F.relu(x)

            # Edge update
            if E > 0:
                edge_feat = self.edge_layers[i](edge_feat, ee_index, ee_type, E)

            layer_outputs.append(x)

        # Output: mean of all layer outputs (sum/num_layers for stable 512D)
        h = torch.stack(layer_outputs, dim=0).mean(dim=0)  # (N, hidden_dim)

        # Unflatten back to (B, L, D)
        residue_embeddings = torch.zeros(B, L, self._hidden_dim, device=device)
        offset = 0
        for b in range(B):
            n = protein.mask[b].sum().item()
            residue_embeddings[b, :n] = h[offset:offset + n]
            offset += n

        # Global: masked mean
        mask_f = protein.mask.unsqueeze(-1).float()
        global_embedding = (residue_embeddings * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return EncoderOutput(
            residue_embeddings=residue_embeddings,
            global_embedding=global_embedding,
        )
