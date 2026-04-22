"""
Official ProteinMPNN encoder wrapper for pretrained weight loading.

Reimplements the encoder portion of ProteinMPNN (Dauparas et al. 2022) with
architecture matching the official checkpoint (v_48_020.pt) exactly.

Only the encoder layers are used (decoder layers are ignored for probing).
This ensures pretrained weights load correctly via state_dict matching.

Reference:
    Dauparas et al. "Robust deep learning based protein sequence design using ProteinMPNN"
    https://github.com/dauparas/ProteinMPNN
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


# ── Building blocks matching official ProteinMPNN ──


class PositionalEncodings(nn.Module):
    """Positional encoding for relative residue offsets (matches official)."""

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32):
        super().__init__()
        # Official: 2*max_relative_feature + 1 + 1 = 66 → linear → num_embeddings
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            offset: Relative residue offsets (B, L, K) — integers
            mask: Same-chain mask (B, L, K) — 1 if same chain, else 0
        Returns:
            Positional features (B, L, K, num_embeddings)
        """
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        )
        # One-hot encode offset
        d_onehot = F.one_hot(d.long(), 2 * self.max_relative_feature + 1).float()
        # Concatenate chain mask
        d_onehot = torch.cat([d_onehot, mask.unsqueeze(-1).float()], dim=-1)
        return self.linear(d_onehot)


def rbf(D: torch.Tensor, D_min: float = 2.0, D_max: float = 22.0, D_count: int = 16) -> torch.Tensor:
    """Radial basis function expansion of distances."""
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_sigma = (D_max - D_min) / D_count
    D_expand = D.unsqueeze(-1)
    return torch.exp(-((D_expand - D_mu) ** 2) / (2 * D_sigma ** 2))


class ProteinFeatures(nn.Module):
    """Feature computation matching official ProteinMPNN (encoder side)."""

    def __init__(
        self,
        edge_features: int = 128,
        node_features: int = 128,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 48,
    ):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.num_rbf = num_rbf

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        # 25 atom-pair distances × num_rbf + positional embeddings
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _get_backbone_atoms(self, coords: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Extract N, CA, C, O, CB from backbone coords (B, L, 5, 3) or (B, L, 4, 3)."""
        N = coords[:, :, 0, :]
        CA = coords[:, :, 1, :]
        C = coords[:, :, 2, :]
        if coords.shape[2] >= 4:
            O = coords[:, :, 3, :]
        else:
            O = CA  # fallback
        # Virtual CB from N, CA, C
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        return N, CA, C, O, CB

    def _compute_distances(
        self, atoms: Tuple[torch.Tensor, ...], E_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute 25 inter-atom distances for k-nearest neighbor pairs."""
        N, CA, C, O, CB = atoms
        B, L, K = E_idx.shape

        # Gather neighbor atoms
        def gather(x):
            idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            return torch.gather(x.unsqueeze(2).expand(-1, -1, K, -1), 1, idx)

        N_j = gather(N)
        CA_j = gather(CA)
        C_j = gather(C)
        O_j = gather(O)
        CB_j = gather(CB)

        # 5 source atoms × 5 target atoms = 25 distance pairs
        sources = [N, CA, C, O, CB]
        targets = [N_j, CA_j, C_j, O_j, CB_j]

        all_rbf = []
        for src in sources:
            src_exp = src.unsqueeze(2).expand(-1, -1, K, -1)
            for tgt in targets:
                dist = torch.sqrt(((src_exp - tgt) ** 2).sum(-1) + 1e-6)
                all_rbf.append(rbf(dist))

        return torch.cat(all_rbf, dim=-1)  # (B, L, K, 25*num_rbf)

    def forward(
        self,
        coords: torch.Tensor,  # (B, L, 5, 3) backbone coords
        mask: torch.Tensor,     # (B, L) valid residue mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge features and k-NN indices.

        Returns:
            E: Edge features (B, L, K, edge_features)
            E_idx: Neighbor indices (B, L, K)
        """
        atoms = self._get_backbone_atoms(coords)
        N, CA, C, O, CB = atoms
        B, L = CA.shape[:2]

        # k-NN based on CA distances
        ca_dist = torch.cdist(CA, CA)  # (B, L, L)
        ca_dist = ca_dist + torch.eye(L, device=CA.device).unsqueeze(0) * 1e6
        invalid = ~(mask.unsqueeze(1) & mask.unsqueeze(2))
        ca_dist = ca_dist + invalid.float() * 1e6
        K = min(self.top_k, L - 1)
        _, E_idx = ca_dist.topk(K, dim=-1, largest=False)  # (B, L, K)

        # Compute RBF features for 25 atom pairs
        rbf_feats = self._compute_distances(atoms, E_idx)  # (B, L, K, 400)

        # Positional encoding from residue offset
        residue_idx = torch.arange(L, device=CA.device).unsqueeze(0).expand(B, -1)
        offset = (
            residue_idx.unsqueeze(2).expand(-1, -1, K)
            - torch.gather(
                residue_idx.unsqueeze(2).expand(-1, -1, K), 1, E_idx
            )
        )
        # All residues assumed same chain for single-domain proteins
        chain_mask = torch.ones(B, L, K, device=CA.device)
        pos_feats = self.embeddings(offset, chain_mask)  # (B, L, K, 16)

        # Concatenate and embed
        E = torch.cat([pos_feats, rbf_feats], dim=-1)  # (B, L, K, 416)
        E = self.edge_embedding(E)  # (B, L, K, 128)
        E = self.norm_edges(E)

        return E, E_idx


class PositionWiseFeedForward(nn.Module):
    """FFN matching official ProteinMPNN naming: W_in, W_out."""

    def __init__(self, num_hidden: int, num_ff: int):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h):
        return self.W_out(F.gelu(self.W_in(h)))


class EncLayer(nn.Module):
    """Encoder layer matching official ProteinMPNN architecture."""

    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: float = 30.0):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        # Node message attention
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        # Edge message attention
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,     # (B, L, D) node features
        h_E: torch.Tensor,     # (B, L, K, D) edge features
        E_idx: torch.Tensor,   # (B, L, K) neighbor indices
        mask_V: Optional[torch.Tensor] = None,  # (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass matching official ProteinMPNN EncLayer exactly.

        Concatenation order: [h_V_i, h_V_j, h_E_ij] = 3D for encoder (D + num_in).
        Aggregation: simple sum / scale (not attention).
        """
        B, L, K, D = h_E.shape

        # cat_neighbors_nodes: gather h_V neighbors and concatenate with h_E
        idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        h_V_neighbors = torch.gather(
            h_V.unsqueeze(2).expand(-1, -1, K, -1), 1, idx
        )  # (B, L, K, D)
        h_EV = torch.cat([h_V_neighbors, h_E], dim=-1)  # (B, L, K, 2D)

        # Expand source node and concatenate: [h_V_i, h_V_j, h_E_ij]
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, D)
        h_EV = torch.cat([h_V_expand, h_EV], dim=-1)  # (B, L, K, 3D)

        # Node message: W1 → GELU → W2 → GELU → W3
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        # Aggregate: simple sum over K neighbors, scaled
        dh = h_message.sum(dim=2) / self.scale  # (B, L, D)

        h_V = self.norm1(h_V + self.dropout1(dh))

        # FFN
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            h_V = h_V * mask_V.unsqueeze(-1).float()

        # Edge update: same concatenation [h_V_i, h_V_j, h_E_ij]
        h_V_neighbors = torch.gather(
            h_V.unsqueeze(2).expand(-1, -1, K, -1), 1, idx
        )
        h_EV = torch.cat([h_V_neighbors, h_E], dim=-1)
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
        h_EV = torch.cat([h_V_expand, h_EV], dim=-1)

        h_E = self.norm3(h_E + self.dropout3(
            self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        ))

        return h_V, h_E


class DecLayer(nn.Module):
    """Decoder layer matching official ProteinMPNN (loaded but not used for encoding)."""

    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: float = 30.0):
        super().__init__()
        self.num_hidden = num_hidden
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None):
        # Not used for encoding — stub for weight loading
        return h_V


@register_encoder("proteinmpnn_pretrained")
class ProteinMPNNPretrainedEncoder(BaseProteinEncoder):
    """
    Official ProteinMPNN encoder with pretrained weight loading.

    Architecture matches the official checkpoint (v_48_020.pt) exactly.
    Only uses encoder layers (decoder is discarded for probing).

    Args:
        hidden_dim: Hidden dimension (must be 128 to match pretrained)
        num_encoder_layers: Number of encoder layers (3 in official)
        num_decoder_layers: Number of decoder layers (3, loaded but unused)
        k_neighbors: Number of k-NN neighbors (48 in v_48_020)
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        k_neighbors: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim

        # Feature computation (matches official)
        self.features = ProteinFeatures(
            edge_features=hidden_dim,
            node_features=hidden_dim,
            top_k=k_neighbors,
        )

        # Edge projection (official: W_e)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Sequence embedding (loaded but not used for structure-only encoding)
        self.W_s = nn.Embedding(21, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers (loaded for checkpoint compatibility, not used in encode())
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output (loaded for checkpoint compatibility, not used)
        self.W_out = nn.Linear(hidden_dim, 21, bias=True)

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B

    @property
    def embedding_dim(self) -> int:
        return self._hidden_dim

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load official ProteinMPNN pretrained weights."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        n_loaded = len(state_dict) - len(unexpected)
        print(f"  [ProteinMPNN Pretrained] Loaded {n_loaded}/{len(state_dict)} keys")
        if missing:
            print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        """
        Encode protein using pretrained ProteinMPNN encoder layers.

        Only uses the encoder portion (no autoregressive decoding).
        """
        device = protein.coords.device
        self.to(device)

        # Compute edge features
        h_E, E_idx = self.features(protein.coords, protein.mask)

        # Project edges
        h_E = self.W_e(h_E)

        # Initialize node features to zeros (official behavior)
        B, L = protein.mask.shape
        h_V = torch.zeros(B, L, self._hidden_dim, device=device)

        # Run encoder layers
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask_V=protein.mask)

        return EncoderOutput(
            residue_embeddings=h_V,
        )
