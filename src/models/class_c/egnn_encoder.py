"""
EGNN (E(n)-Equivariant Graph Neural Network) encoder wrapper.

Uses the official egnn-pytorch library (lucidrains/egnn-pytorch) which implements
Satorras et al. (2021) "E(n) Equivariant Graph Neural Networks".

EGNN updates both node features and coordinates while maintaining E(n) equivariance.
Simpler than SE(3)-Transformer while matching or beating it on most tasks.

Reference:
    Satorras et al. (2021). https://arxiv.org/abs/2102.09844
    Library: https://github.com/lucidrains/egnn-pytorch

Architecture: Class C (Equivariant Field)
    Random weights only — no protein-pretrained checkpoint available.
    Serves as a geometric inductive bias baseline.
"""

import torch
import torch.nn as nn
from typing import Optional

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)

_HIDDEN_DIM = 128
_DEPTH = 4
_K_NEIGHBORS = 16  # num_nearest_neighbors; must be < min(L) in batch


class _InputProjection(nn.Module):
    """Project one-hot amino acid features to EGNN hidden dim."""
    def __init__(self, in_dim: int = 20, hidden_dim: int = _HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        return self.proj(x)


@register_encoder("egnn")
class EGNNEncoder(BaseProteinEncoder):
    """
    EGNN encoder using official egnn-pytorch library (random weights).

    Input features: one-hot amino acid identity (20D) → projected to 128D.
    Coordinates: CA atom positions (Å).
    k-NN graph with k=16 neighbors per residue.
    Output: per-residue node features after 4 EGNN layers (128D).
    """

    def __init__(
        self,
        hidden_dim: int = _HIDDEN_DIM,
        depth: int = _DEPTH,
        k_neighbors: int = _K_NEIGHBORS,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.k_neighbors = k_neighbors

        from egnn_pytorch import EGNN_Network
        self._current_k = k_neighbors
        self.input_proj = _InputProjection(in_dim=20, hidden_dim=hidden_dim)
        self.egnn = EGNN_Network(
            depth=depth,
            dim=hidden_dim,
            num_nearest_neighbors=k_neighbors,
            only_sparse_neighbors=False,
            update_feats=True,
            update_coors=True,
        )

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.C

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        B, L, _, _ = protein.coords.shape

        # CA coordinates: (B, L, 3)
        ca_coords = protein.coords[:, :, 1, :].float()

        # One-hot amino acid features: (B, L, 20)
        seq = protein.sequence.long().clamp(0, 19)
        feats_onehot = torch.zeros(B, L, 20, device=device)
        feats_onehot.scatter_(2, seq.unsqueeze(-1), 1.0)

        # Project to hidden dim: (B, L, hidden_dim)
        feats = self.input_proj(feats_onehot)

        mask = protein.mask.bool()  # (B, L)

        # Clamp k to avoid out-of-range errors on short sequences
        min_valid = int(mask.sum(dim=1).min().item())
        k = min(self.k_neighbors, max(1, min_valid - 1))
        if k != self._current_k:
            from egnn_pytorch import EGNN_Network
            self.egnn = EGNN_Network(
                depth=self.depth,
                dim=self.hidden_dim,
                num_nearest_neighbors=k,
                only_sparse_neighbors=False,
                update_feats=True,
                update_coors=True,
            ).to(device)
            self._current_k = k

        with torch.set_grad_enabled(not self._is_frozen):
            out_feats, _ = self.egnn(feats, ca_coords, mask=mask)
        # out_feats: (B, L, hidden_dim)

        # Zero out padding positions
        out_feats = out_feats * mask.unsqueeze(-1).float()

        mask_f = mask.float().unsqueeze(-1)
        global_emb = (out_feats * mask_f).sum(1) / (mask_f.sum(1) + 1e-8)

        return EncoderOutput(
            residue_embeddings=out_feats,
            global_embedding=global_emb,
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass  # random-only baseline — no pretrained weights
