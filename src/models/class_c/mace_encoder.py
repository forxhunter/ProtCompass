"""
MACE (Multi-ACE) Encoder for Protein Structures.

Higher-order equivariant message passing via multi-body interactions.
Uses products of two-body messages to capture three-body correlations
without explicitly enumerating triplets.

Simplified to L=0,1 (scalar + vector) without e3nn dependency.

Reference:
    Batatia et al. (2022). "MACE: Higher Order Equivariant Message
    Passing Neural Networks for Fast and Accurate Force Fields."
"""

import math
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


class BesselRBF(nn.Module):
    """Bessel radial basis functions with polynomial envelope."""

    def __init__(self, num_rbf: int = 8, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff
        freq = torch.arange(1, num_rbf + 1).float() * math.pi / cutoff
        self.register_buffer("freq", freq)

    def forward(self, dist):
        d = dist.unsqueeze(-1).clamp(min=1e-8)
        # Bessel basis
        rbf = math.sqrt(2.0 / self.cutoff) * torch.sin(self.freq * d) / d
        # Polynomial envelope p=5
        u = dist.unsqueeze(-1) / self.cutoff
        envelope = 1.0 - 6 * u ** 5 + 15 * u ** 4 - 10 * u ** 3
        envelope = envelope * (dist.unsqueeze(-1) < self.cutoff).float()
        return rbf * envelope


class TwoBodyMessage(nn.Module):
    """Two-body message: standard equivariant message passing.

    Produces scalar and vector messages from neighbor features
    modulated by radial basis and edge directions.
    """

    def __init__(self, s_dim: int, v_dim: int, num_rbf: int):
        super().__init__()
        self.v_dim = v_dim

        # Radial filter for scalars
        self.radial_s = nn.Sequential(
            nn.Linear(num_rbf, s_dim), nn.SiLU(), nn.Linear(s_dim, s_dim),
        )
        # Radial filter for vectors
        self.radial_v = nn.Sequential(
            nn.Linear(num_rbf, s_dim), nn.SiLU(), nn.Linear(s_dim, v_dim),
        )
        # Node transform
        self.node_proj = nn.Linear(s_dim, s_dim)

    def forward(self, s, rbf, edge_dir, edge_index):
        """
        s: (B, N, S) node scalar features
        rbf: (B, N, K, num_rbf)
        edge_dir: (B, N, K, 3)
        edge_index: (B, N, K)
        Returns: msg_s (B, N, S), msg_v (B, N, V, 3)
        """
        B, N, S = s.shape
        K = edge_index.shape[2]

        # Gather neighbor scalars
        idx = edge_index.unsqueeze(-1).expand(-1, -1, -1, S)
        s_j = s.unsqueeze(2).expand(-1, -1, K, -1).gather(1, idx)  # (B,N,K,S)

        # Scalar message: weighted sum of neighbor features
        w_s = self.radial_s(rbf)  # (B, N, K, S)
        msg_s = (self.node_proj(s_j) * w_s).sum(dim=2)  # (B, N, S)

        # Vector message: edge directions weighted by radial + scalar info
        w_v = self.radial_v(rbf)  # (B, N, K, V)
        msg_v = (w_v.unsqueeze(-1) * edge_dir.unsqueeze(-2)).sum(dim=2)  # (B, N, V, 3)

        return msg_s, msg_v


class SymmetricContraction(nn.Module):
    """Symmetric contraction for multi-body interactions.

    Takes products of two-body messages to create three-body features.
    For L=0,1: contracts pairs of vector messages into scalars (dot products)
    and scalars into vectors (outer products with message vectors).
    """

    def __init__(self, s_dim: int, v_dim: int):
        super().__init__()
        # Scalar from vector self-products: (v, v) -> s via dot
        self.contract_vv_s = nn.Linear(v_dim, s_dim)

        # Vector from scalar × vector: (s, v) -> v
        self.contract_sv_v = nn.Sequential(
            nn.Linear(s_dim, v_dim),
            nn.Sigmoid(),
        )

        # Scalar self-product: s * s -> s
        self.contract_ss_s = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            nn.SiLU(),
            nn.Linear(s_dim, s_dim),
        )

    def forward(self, msg_s, msg_v):
        """
        msg_s: (B, N, S) two-body scalar messages
        msg_v: (B, N, V, 3) two-body vector messages
        Returns: three-body (s, v) features
        """
        # Vector self-dot → scalar (three-body angular info)
        v_dot = (msg_v * msg_v).sum(dim=-1)  # (B, N, V)
        s_from_vv = self.contract_vv_s(v_dot)  # (B, N, S)

        # Scalar self-product → scalar
        s_from_ss = self.contract_ss_s(msg_s * msg_s)  # (B, N, S)

        # Scalar gates vector → vector (three-body directional info)
        gates = self.contract_sv_v(msg_s)  # (B, N, V)
        v_from_sv = msg_v * gates.unsqueeze(-1)  # (B, N, V, 3)

        s_out = s_from_vv + s_from_ss
        v_out = v_from_sv

        return s_out, v_out


class MACEBlock(nn.Module):
    """MACE interaction block: two-body messages + symmetric contraction."""

    def __init__(self, s_dim: int, v_dim: int, num_rbf: int):
        super().__init__()
        self.two_body = TwoBodyMessage(s_dim, v_dim, num_rbf)
        self.contraction = SymmetricContraction(s_dim, v_dim)

        # Combine two-body and three-body
        self.update_s = nn.Sequential(
            nn.Linear(s_dim * 3, s_dim),
            nn.SiLU(),
            nn.Linear(s_dim, s_dim),
        )
        self.update_v = nn.Linear(v_dim * 2, v_dim, bias=False)

        self.ln_s = nn.LayerNorm(s_dim)

    def forward(self, s, v, rbf, edge_dir, edge_index, mask):
        """
        s: (B, N, S), v: (B, N, V, 3)
        Returns updated (s, v)
        """
        # Two-body messages
        msg_s, msg_v = self.two_body(s, rbf, edge_dir, edge_index)

        # Three-body via symmetric contraction
        three_s, three_v = self.contraction(msg_s, msg_v)

        # Combine and update
        s_cat = torch.cat([s, msg_s, three_s], dim=-1)  # (B, N, 3S)
        s_new = self.update_s(s_cat)

        v_cat = torch.cat([msg_v, three_v], dim=-2)  # (B, N, 2V, 3)
        v_new = self.update_v(v_cat.transpose(-1, -2)).transpose(-1, -2)  # (B, N, V, 3)

        # Residual
        s_out = self.ln_s(s + s_new) * mask.unsqueeze(-1).float()
        v_out = (v + v_new) * mask.unsqueeze(-1).unsqueeze(-1).float()

        return s_out, v_out


@register_encoder("mace")
class MACEEncoder(BaseProteinEncoder):
    """
    MACE encoder for protein structures.

    Uses multi-body equivariant message passing: standard two-body
    messages are combined via symmetric contractions to capture
    three-body (angular) interactions without explicit triplet enumeration.

    Simplified L=0,1 implementation without e3nn dependency.

    Args:
        hidden_dim: Scalar feature dimension
        num_vectors: Number of L=1 vector channels
        num_layers: Number of MACE blocks
        num_rbf: Number of Bessel radial basis functions
        cutoff: Distance cutoff
        k_neighbors: kNN neighbors
    """

    bottleneck_class = BottleneckClass.C

    def __init__(
        self,
        hidden_dim: int = 128,
        num_vectors: int = 16,
        num_layers: int = 4,
        num_rbf: int = 8,
        cutoff: float = 10.0,
        k_neighbors: int = 30,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.num_vectors = num_vectors
        self.k_neighbors = k_neighbors

        self.aa_embed = nn.Embedding(21, hidden_dim)
        self.radial_basis = BesselRBF(num_rbf, cutoff)

        self.blocks = nn.ModuleList([
            MACEBlock(hidden_dim, num_vectors, num_rbf)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    @property
    def embedding_dim(self) -> int:
        return self._hidden_dim

    def _build_knn(self, ca, mask):
        B, N, _ = ca.shape
        K = min(self.k_neighbors, N - 1)
        dist = torch.cdist(ca, ca)
        dist.masked_fill_(~(mask.unsqueeze(2) & mask.unsqueeze(1)).bool(), float("inf"))
        dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        _, idx = dist.topk(K, dim=-1, largest=False)
        return idx

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        B, L = protein.sequence.shape
        ca = protein.coords[:, :, 1, :]  # CA atoms

        edge_index = self._build_knn(ca, protein.mask)
        K = edge_index.shape[2]

        # Edge geometric features
        idx_3 = edge_index.unsqueeze(-1).expand(-1, -1, -1, 3)
        ca_j = ca.unsqueeze(2).expand(-1, -1, K, -1).gather(1, idx_3)
        rel = ca_j - ca.unsqueeze(2)
        dist = rel.norm(dim=-1)
        edge_dir = F.normalize(rel, dim=-1)

        rbf = self.radial_basis(dist)

        # Initialize features
        s = self.aa_embed(protein.sequence)  # (B, L, D)
        v = torch.zeros(B, L, self.num_vectors, 3, device=s.device, dtype=s.dtype)

        for block in self.blocks:
            s, v = block(s, v, rbf, edge_dir, edge_index, protein.mask)

        h = self.output_proj(s)
        h = h * protein.mask.unsqueeze(-1).float()

        global_emb = h.sum(dim=1) / protein.mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return EncoderOutput(residue_embeddings=h, global_embedding=global_emb)

    def load_pretrained(self, path: str):
        if path and path != "none":
            self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
