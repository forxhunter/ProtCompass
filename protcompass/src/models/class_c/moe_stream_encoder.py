"""
MoE Stream Encoder: Mixture-of-Experts Fusion for Multi-Stream Protein Encoding.

Extends the DualStream/TriStream architecture with sparse MoE routing
for stream fusion. Instead of fixed concatenation (DualStream) or
task-conditioned cross-attention (TriStream), each residue gets a
learned routing decision over which stream(s) to use.

Architecture:
    Stream L (Local): 1D Conv ResNet on geometric features -> 64D
    Stream P (Packing): MLP on density/topology -> 64D
    Stream S (Sequence): AA embeddings + Conv1D -> 64D (optional)
    MoE Router: Per-residue gating over streams -> Top-k selection
    Output: Weighted combination of selected stream embeddings

Key design choices:
    - Router input: concatenation of all stream outputs (not raw features)
    - Top-k routing: k=2 by default (sparse, interpretable)
    - Load balancing loss: encourages even expert utilization
    - Auxiliary output: routing weights for analysis

Usage:
    PYTHONPATH=. python scripts/benchmark_invariants.py \\
        --encoder moe_stream --checkpoint none \\
        --data-path data/processed/cath.h5 \\
        --output-dir outputs/sweep_v3/moe_stream \\
        --device cuda --batch-size 32 --num-workers 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)
from src.models.class_c.dual_stream_encoder import (
    StreamL,
    StreamP,
)
from src.models.class_c.task_conditioned_dual_stream import StreamS
from src.data.preprocessing.geometry_calculator import (
    compute_contact_density,
    compute_topology_features,
)
from src.models.components.handcrafted_features import compute_i1_features


class StreamRouter(nn.Module):
    """
    Per-residue MoE router that gates over stream outputs.

    Given concatenated stream representations, produces gating weights
    for each stream. Uses top-k selection for sparsity.

    Args:
        input_dim: Total dim of concatenated streams (e.g., 128 for 2 streams, 192 for 3)
        num_experts: Number of streams/experts
        top_k: Number of experts to activate per residue (default 2)
        noise_std: Gaussian noise added to logits during training for load balancing
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std

        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, num_experts),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D_total) concatenated stream outputs
            mask: (B, L) boolean mask

        Returns:
            weights: (B, L, num_experts) — normalized gating weights (sparse)
            indices: (B, L, top_k) — indices of selected experts
            load_balance_loss: scalar — auxiliary loss for even utilization
        """
        logits = self.gate(x)  # (B, L, num_experts)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)  # (B, L, k)

        # Softmax over selected experts only
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, L, k)

        # Scatter into full weight tensor
        weights = torch.zeros_like(logits)  # (B, L, num_experts)
        weights.scatter_(-1, top_k_indices, top_k_weights)

        # Mask out padded positions
        weights = weights * mask.unsqueeze(-1).float()

        # Load balancing loss (encourages even expert utilization)
        # fraction of tokens routed to each expert
        if mask.any():
            mask_f = mask.float()
            tokens_per_expert = (weights * mask_f.unsqueeze(-1)).sum(dim=(0, 1))  # (num_experts,)
            total_tokens = mask_f.sum()
            fractions = tokens_per_expert / total_tokens.clamp(min=1)
            # Ideal: 1/num_experts for each
            target = 1.0 / self.num_experts
            load_balance_loss = self.num_experts * (fractions * fractions).sum()
        else:
            load_balance_loss = torch.tensor(0.0, device=x.device)

        return weights, top_k_indices, load_balance_loss


@register_encoder("moe_stream")
class MoEStreamEncoder(BaseProteinEncoder):
    """
    MoE Stream Encoder: Sparse routing over specialized streams.

    Each stream processes a different aspect of protein structure:
        - Stream L: Local geometry (torsion angles, curvature)
        - Stream P: Packing environment (contact density, topology)
        - Stream S: Sequence identity (optional, can be disabled)

    The MoE router learns per-residue which streams are most informative,
    producing sparse, interpretable routing decisions.

    Args:
        hidden_dim: Total output embedding dimension
        stream_dim: Dimension per stream (hidden_dim // num_streams)
        use_sequence: Whether to include Stream S (sequence)
        top_k: Number of streams activated per residue
        radii: Distance thresholds for contact density
    """

    bottleneck_class = BottleneckClass.C

    def __init__(
        self,
        hidden_dim: int = 128,
        stream_dim: int = 64,
        use_sequence: bool = False,
        top_k: int = 2,
        radii: List[float] = [6.0, 8.0, 10.0, 12.0],
    ):
        super().__init__()

        self.use_sequence = use_sequence
        self.num_streams = 3 if use_sequence else 2
        self.stream_dim = stream_dim
        self.top_k = top_k
        self.radii = radii

        # Streams
        self.stream_p = StreamP(
            input_dim=len(radii) + 4,
            output_dim=stream_dim,
        )
        self.stream_l = StreamL(
            input_dim=21,
            output_dim=stream_dim,
        )
        if use_sequence:
            self.stream_s = StreamS(
                hidden_dim=stream_dim,
            )

        # Global context for Stream P (same as DualStream)
        self.global_mlp = nn.Sequential(
            nn.Linear(stream_dim, stream_dim),
            nn.ReLU(),
            nn.Linear(stream_dim, stream_dim),
        )

        # MoE Router
        router_input_dim = stream_dim * self.num_streams
        self.router = StreamRouter(
            input_dim=router_input_dim,
            num_experts=self.num_streams,
            top_k=top_k,
        )

        # Output projection: project weighted stream to final dim
        self.output_proj = nn.Linear(stream_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        self._last_routing_weights = None  # For analysis

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim

    def encode(self, protein_input: ProteinInput) -> EncoderOutput:
        coords = protein_input.coords
        sequence = protein_input.sequence
        mask = protein_input.mask

        # 1. Compute stream inputs
        density = compute_contact_density(coords, mask, threshold=self.radii)
        topology = compute_topology_features(coords, mask, thresholds=[8.0, 10.0])
        p_in = torch.cat([density, topology], dim=-1)
        i1_feats = compute_i1_features(coords, mask)

        # 2. Run streams
        hp = self.stream_p(p_in)  # (B, L, stream_dim)

        # Global context for P
        g = (hp * mask.unsqueeze(-1).float()).sum(dim=1) / (
            mask.float().sum(dim=1, keepdim=True) + 1e-8
        )
        g_out = self.global_mlp(g)
        hp = hp + g_out.unsqueeze(1)

        hl = self.stream_l(i1_feats, mask)  # (B, L, stream_dim)

        streams = [hl, hp]  # Order: L, P, [S]

        if self.use_sequence:
            hs = self.stream_s(sequence, mask)  # (B, L, stream_dim)
            streams.append(hs)

        # 3. Stack streams: (B, L, num_streams, stream_dim)
        stream_stack = torch.stack(streams, dim=2)

        # 4. Route: concatenate for router input
        router_input = torch.cat(streams, dim=-1)  # (B, L, num_streams * stream_dim)
        weights, indices, lb_loss = self.router(router_input, mask)  # weights: (B, L, num_streams)

        # Save for analysis
        self._last_routing_weights = weights.detach()
        self._last_load_balance_loss = lb_loss

        # 5. Weighted combination
        # weights: (B, L, num_streams) -> (B, L, num_streams, 1)
        # stream_stack: (B, L, num_streams, stream_dim)
        h = (stream_stack * weights.unsqueeze(-1)).sum(dim=2)  # (B, L, stream_dim)

        # 6. Project to output dim
        h = self.output_proj(h)  # (B, L, hidden_dim)

        # Mask
        h = h * mask.unsqueeze(-1).float()

        # Global pooling
        global_emb = h.sum(dim=1) / (mask.float().sum(dim=1, keepdim=True) + 1e-8)

        return EncoderOutput(
            residue_embeddings=h,
            global_embedding=global_emb,
        )

    def get_routing_weights(self) -> Optional[torch.Tensor]:
        """Return last routing weights for analysis. Shape: (B, L, num_streams)."""
        return self._last_routing_weights

    def get_stream_names(self) -> List[str]:
        """Return ordered stream names matching routing weight indices."""
        names = ["Local (L)", "Packing (P)"]
        if self.use_sequence:
            names.append("Sequence (S)")
        return names

    def load_pretrained(self, checkpoint_path: str) -> None:
        if checkpoint_path and checkpoint_path != "none":
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
