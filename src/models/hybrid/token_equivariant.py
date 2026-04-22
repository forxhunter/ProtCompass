"""
Token + Equivariant Hybrid Architecture.

Combines the efficiency of token-based encoders (Class A) with
equivariant refinement (Class C) to improve metric geometry preservation.

Architecture:
1. Base: Token encoder (e.g., SaProt) for fast, I1-rich embeddings
2. Refinement: Small GVP head to add geometric information (I2)

Expected: Improves I2 (metric geometry) without losing I1 efficiency.
"""

from typing import Any, Dict, Optional, Tuple

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
from src.models.class_c.gvp_encoder import GVP, compute_gvp_features


class EquivariantRefinementHead(nn.Module):
    """
    Small equivariant refinement head using GVP layers.

    Takes token embeddings and coordinates, produces refined embeddings
    with added geometric information.

    Args:
        input_dim: Dimension of input embeddings
        output_dim: Dimension of output embeddings
        hidden_dim: Hidden dimension
        num_layers: Number of GVP layers
        node_v_dim: Vector feature dimension
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        node_v_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.node_v_dim = node_v_dim

        # Project input to GVP scalar dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GVP layers for refinement
        self.gvp_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dims = (hidden_dim, node_v_dim if i > 0 else 3)  # 3 initial vectors from coords
            out_dims = (hidden_dim, node_v_dim)
            self.gvp_layers.append(
                GVP(in_dims, out_dims, activations=(F.relu, torch.sigmoid), vector_gate=True)
            )

        # Final projection
        final_dim = hidden_dim + node_v_dim * 3  # Scalars + flattened vectors
        self.output_proj = nn.Linear(final_dim, output_dim)

        # Residual gate
        self.gate = nn.Sequential(
            nn.Linear(input_dim + output_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine embeddings with geometric information.

        Args:
            embeddings: Token embeddings (B, L, input_dim)
            coords: Backbone coordinates (B, L, 5, 3)
            mask: Valid residue mask (B, L)

        Returns:
            Refined embeddings (B, L, output_dim)
        """
        B, L = embeddings.shape[:2]
        device = embeddings.device

        # Project to hidden dimension
        h_s = self.input_proj(embeddings)  # (B, L, hidden_dim)

        # Initialize vector features from local frames
        N = coords[:, :, 0, :]
        CA = coords[:, :, 1, :]
        C = coords[:, :, 2, :]

        # Compute local frame vectors
        forward = C - CA
        forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-8)
        backward = N - CA
        backward = backward / (torch.norm(backward, dim=-1, keepdim=True) + 1e-8)
        side = torch.cross(forward, backward, dim=-1)
        side = side / (torch.norm(side, dim=-1, keepdim=True) + 1e-8)

        h_v = torch.stack([forward, backward, side], dim=-2)  # (B, L, 3, 3)

        # GVP layers
        for layer in self.gvp_layers:
            h_s, h_v = layer((h_s, h_v))

        # Flatten vectors
        h_v_flat = h_v.reshape(B, L, -1)  # (B, L, node_v_dim * 3)

        # Concatenate and project
        h_combined = torch.cat([h_s, h_v_flat], dim=-1)
        h_out = self.output_proj(h_combined)  # (B, L, output_dim)

        # Gated residual connection
        # Gate based on both original and refined features
        gate_input = torch.cat([embeddings, h_out], dim=-1)
        gate = self.gate(gate_input)

        # If output_dim matches input_dim, use residual
        if self.output_dim == self.input_dim:
            output = embeddings + gate * h_out
        else:
            output = gate * h_out

        # Mask
        output = output * mask.unsqueeze(-1).float()

        return output


@register_encoder("token_equivariant_hybrid")
class TokenEquivariantHybrid(BaseProteinEncoder):
    """
    Hybrid encoder combining token and equivariant components.

    Uses a token encoder (e.g., SaProt) as the base, then applies
    a small equivariant refinement head to add geometric information.

    Args:
        base_encoder: Token-based encoder (Class A)
        refinement_layers: Number of GVP refinement layers
        refinement_hidden_dim: Hidden dimension for refinement
        refinement_vector_dim: Vector feature dimension
        freeze_base: Whether to freeze base encoder
    """

    def __init__(
        self,
        base_encoder: Optional[BaseProteinEncoder] = None,
        refinement_layers: int = 2,
        refinement_hidden_dim: int = 128,
        refinement_vector_dim: int = 8,
        freeze_base: bool = True,
    ):
        super().__init__()

        # Initialize base encoder if not provided
        if base_encoder is None:
            from src.models.class_a.saprot_encoder import SaProtEncoder
            base_encoder = SaProtEncoder(model_size="650M", use_pretrained=False)

        self.base_encoder = base_encoder

        if freeze_base:
            self.base_encoder.freeze()

        # Equivariant refinement head
        self.refinement = EquivariantRefinementHead(
            input_dim=base_encoder.embedding_dim,
            output_dim=base_encoder.embedding_dim,
            hidden_dim=refinement_hidden_dim,
            num_layers=refinement_layers,
            node_v_dim=refinement_vector_dim,
        )

        self._embedding_dim = base_encoder.embedding_dim

    @property
    def bottleneck_class(self) -> BottleneckClass:
        # Hybrid - but primarily token-based
        return BottleneckClass.A

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights for base encoder."""
        self.base_encoder.load_pretrained(checkpoint_path)

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        """
        Encode protein with hybrid architecture.

        Args:
            protein: Protein input

        Returns:
            EncoderOutput with refined embeddings
        """
        # Get base embeddings
        base_output = self.base_encoder.encode(protein)
        base_emb = base_output.residue_embeddings

        # Refine with geometric information
        refined_emb = self.refinement(
            base_emb,
            protein.coords,
            protein.mask,
        )

        # Global pooling
        mask_expanded = protein.mask.unsqueeze(-1).float()
        global_emb = (refined_emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return EncoderOutput(
            residue_embeddings=refined_emb,
            global_embedding=global_emb,
            metadata={
                "base_embeddings": base_emb,
                "base_output": base_output,
            },
        )


@register_encoder("graph_global_hybrid")
class GraphGlobalHybrid(BaseProteinEncoder):
    """
    Graph + Global Topology Hybrid.

    Adds global attention/spectral features to a graph encoder
    to improve long-range topology preservation (I4).

    Args:
        base_encoder: Graph-based encoder (Class B)
        global_layers: Number of global attention layers
        global_heads: Number of attention heads
        freeze_base: Whether to freeze base encoder
    """

    def __init__(
        self,
        base_encoder: Optional[BaseProteinEncoder] = None,
        global_layers: int = 2,
        global_heads: int = 4,
        freeze_base: bool = True,
    ):
        super().__init__()

        # Initialize base encoder if not provided
        if base_encoder is None:
            from src.models.class_b.proteinmpnn_encoder import ProteinMPNNEncoder
            base_encoder = ProteinMPNNEncoder()

        self.base_encoder = base_encoder

        if freeze_base:
            self.base_encoder.freeze()

        # Global attention layers
        self.global_attention = nn.ModuleList()
        for _ in range(global_layers):
            self.global_attention.append(
                nn.MultiheadAttention(
                    embed_dim=base_encoder.embedding_dim,
                    num_heads=global_heads,
                    dropout=0.1,
                    batch_first=True,
                )
            )
            self.global_attention.append(
                nn.LayerNorm(base_encoder.embedding_dim)
            )

        self._embedding_dim = base_encoder.embedding_dim

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights."""
        self.base_encoder.load_pretrained(checkpoint_path)

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        """
        Encode with graph + global attention.

        Args:
            protein: Protein input

        Returns:
            EncoderOutput with globally-aware embeddings
        """
        # Get base embeddings
        base_output = self.base_encoder.encode(protein)
        h = base_output.residue_embeddings

        # Apply global attention layers
        key_padding_mask = ~protein.mask  # True = ignore

        for i in range(0, len(self.global_attention), 2):
            attn_layer = self.global_attention[i]
            norm_layer = self.global_attention[i + 1]

            # Self-attention
            attn_out, _ = attn_layer(
                h, h, h,
                key_padding_mask=key_padding_mask,
            )

            # Residual + norm
            h = norm_layer(h + attn_out)

        # Mask
        h = h * protein.mask.unsqueeze(-1).float()

        # Global pooling
        mask_expanded = protein.mask.unsqueeze(-1).float()
        global_emb = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return EncoderOutput(
            residue_embeddings=h,
            global_embedding=global_emb,
            metadata={
                "base_embeddings": base_output.residue_embeddings,
                "base_output": base_output,
            },
        )
