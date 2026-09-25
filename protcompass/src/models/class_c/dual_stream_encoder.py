"""
Dual-Steam Encoder (Deep Researcher Spec).

Resolves I1 vs I3 corruption by structural separation.
Stream P (Packing): MLP on multi-radius neighbor counts (I3).
Stream L (Local): 1D Conv ResNet on handcrafted geometric features (I1).
Fusion: Late concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)
from src.data.preprocessing.geometry_calculator import compute_contact_density, compute_topology_features
from src.models.components.handcrafted_features import compute_i1_features


class StreamP(nn.Module):
    """
    Packing Stream: Simple MLP on density statistics.
    Input: Neighbor counts at [6, 8, 10, 12] Angstroms.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, density_counts):
        """
        density_counts: (B, L, num_radii)
        Returns: (B, L, output_dim)
        """
        return self.net(density_counts)


class ResNetBlock1D(nn.Module):
    """1D Convolutional Residual Block."""
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x: (B, C, L)
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class StreamL(nn.Module):
    """
    Local Stream: Temporal Conv ResNet on geometric features.
    Input: Handcrafted I1 features (21 dims).
    """
    def __init__(self, input_dim: int = 21, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        
        self.expand = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        # ResNet blocks with increasing dilation (receptive field growth)
        self.blocks = nn.ModuleList([
            ResNetBlock1D(hidden_dim, kernel_size=3, dilation=1),
            ResNetBlock1D(hidden_dim, kernel_size=3, dilation=2),
            ResNetBlock1D(hidden_dim, kernel_size=5, dilation=4),
            ResNetBlock1D(hidden_dim, kernel_size=7, dilation=8),
        ])
        
        self.proj = nn.Conv1d(hidden_dim, output_dim, 1)
        self.output_dim = output_dim

    def forward(self, x, mask):
        """
        x: (B, L, D) features
        mask: (B, L)
        Returns: (B, L, output_dim)
        """
        # Transpose to (B, C, L) for Conv1d
        x = x.transpose(1, 2)
        
        # Mask input (zero out padded)
        mask_expanded = mask.unsqueeze(1)  # (B, 1, L)
        x = x * mask_expanded
        
        x = self.expand(x)
        
        for block in self.blocks:
            x = block(x) * mask_expanded
            
        x = self.proj(x)
        
        # Transpose back to (B, L, C)
        return x.transpose(1, 2)


@register_encoder("dual_stream")
class DualStreamEncoder(BaseProteinEncoder):
    """
    Dual-Stream Encoder for balanced I1/I3 representation.
    """
    bottleneck_class = BottleneckClass.C  # It encodes structure
    
    def __init__(
        self,
        hidden_dim: int = 128,  # Total embedding dim
        radii: List[float] = [6.0, 8.0, 10.0, 12.0],
        stream_p_dim: int = 64,
        stream_l_dim: int = 64,
        seed: int = None,
    ):
        super().__init__()
        assert stream_p_dim + stream_l_dim == hidden_dim, "Stream dims must sum to hidden_dim"

        self.hidden_dim = hidden_dim
        self.radii = radii
        self.output_mode = "fused"  # fused, p, l
        self.seed = seed

        # Set seed BEFORE initializing weights
        if seed is not None:
            self._seed_everything(seed)

        # Stream P: Packing (I3) + Topology (I4)
        # Input: 4 density radii + 4 topology features (med/long for 8A/10A)
        self.stream_p = StreamP(
            input_dim=len(radii) + 4,
            output_dim=stream_p_dim
        )
        
        # Global Token Interaction (Minimal)
        self.global_mlp = nn.Sequential(
            nn.Linear(stream_p_dim, stream_p_dim),
            nn.ReLU(),
            nn.Linear(stream_p_dim, stream_p_dim)
        )
        
        # Stream L: Local (I1)
        # Input dim 21 from handcrafted features
        self.stream_l = StreamL(
            input_dim=21,
            output_dim=stream_l_dim
        )
        
        self.stream_p_dim = stream_p_dim
        self.stream_l_dim = stream_l_dim
        
    @property
    def embedding_dim(self) -> int:
        if self.output_mode == "p":
            return self.stream_p_dim
        elif self.output_mode == "l":
            return self.stream_l_dim
        return self.hidden_dim
        
    def encode(self, protein_input: ProteinInput) -> EncoderOutput:
        coords = protein_input.coords
        mask = protein_input.mask
        
        # 1. Compute Features
        # Stream P input: Multi-radius density + Topology
        density = compute_contact_density(coords, mask, threshold=self.radii) # (B, L, 4)
        topology = compute_topology_features(coords, mask, thresholds=[8.0, 10.0]) # (B, L, 4)
        
        p_in = torch.cat([density, topology], dim=-1) # (B, L, 8)
        
        # Stream L input: Handcrafted I1 features
        i1_feats = compute_i1_features(coords, mask) # (B, L, 21)
        
        # 2. Encode Streams
        # Stream P (MLP)
        hp = self.stream_p(p_in) # (B, L, 64)
        
        # Global Token (Step B)
        # 1. Pool P features
        g = (hp * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8) # (B, 64)
        # 2. Update Global Token
        g_out = self.global_mlp(g) # (B, 64)
        # 3. Broadcast back
        hp = hp + g_out.unsqueeze(1) # (B, L, 64)
        
        # Stream L (ResNet)
        hl = self.stream_l(i1_feats, mask) # (B, L, 64)
        
        # 3. Fusion (Concatenation)
        h = torch.cat([hl, hp], dim=-1) # (B, L, 128)
        
        # Output Selection based on mode
        if self.output_mode == "p":
            h_out = hp
        elif self.output_mode == "l":
            h_out = hl
        else:
            h_out = h
        
        # Mask output
        h_out = h_out * mask.unsqueeze(-1)
        
        # Global pooling
        global_emb = h_out.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        return EncoderOutput(
            residue_embeddings=h_out,
            global_embedding=global_emb,
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        if checkpoint_path and checkpoint_path != "none":
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    def _seed_everything(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_state_dict(self) -> dict:
        """Return state dict for serialization and reproducibility."""
        return self.state_dict()

    def set_state_dict(self, state_dict: dict) -> None:
        """Load a serialized encoder state."""
        self.load_state_dict(state_dict)
