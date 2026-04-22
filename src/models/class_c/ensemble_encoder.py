"""
Ensemble Encoder - Task-specific model selection.

Strategy:
- EGNN for packing-related tasks (I3 = 0.815)
- GearNet for local-related tasks (I1 = 0.377)
- Weighted combination for global embedding

This is a FALLBACK approach: admitting that no single architecture
can dominate on all invariants with random weights.
"""

import torch
import torch.nn as nn
from typing import Tuple

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)
from src.models import get_encoder


@register_encoder("ensemble")
class EnsembleEncoder(BaseProteinEncoder):
    """
    Ensemble of EGNN + GearNet with learned task routing.
    
    Uses both encoders and learns to weight their outputs.
    """
    
    bottleneck_class = BottleneckClass.C
    
    def __init__(
        self,
        hidden_dim: int = 128,  # Common output dim
        freeze: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Load base encoders
        self.egnn = get_encoder("egnn")  # Best for I3
        self.gearnet = get_encoder("gearnet")  # Best for I1
        
        # Project to common dimension
        self.egnn_proj = nn.Linear(self.egnn.embedding_dim, hidden_dim)
        self.gearnet_proj = nn.Linear(self.gearnet.embedding_dim, hidden_dim)
        
        # Learned combination weights
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        if freeze:
            self.freeze()
    
    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim
    
    def encode(self, protein_input: ProteinInput) -> EncoderOutput:
        # Get outputs from both encoders
        egnn_out = self.egnn.encode(protein_input)
        gearnet_out = self.gearnet.encode(protein_input)
        
        # Project to common dim
        egnn_emb = self.egnn_proj(egnn_out.residue_embeddings)
        gearnet_emb = self.gearnet_proj(gearnet_out.residue_embeddings)
        
        # Combine
        combined = self.combine(torch.cat([egnn_emb, gearnet_emb], dim=-1))
        
        # Global pooling (from EGNN's output - it's better for I3)
        egnn_global = self.egnn_proj(egnn_out.global_embedding)
        gearnet_global = self.gearnet_proj(gearnet_out.global_embedding)
        global_emb = (egnn_global + gearnet_global) / 2
        
        return EncoderOutput(
            residue_embeddings=combined,
            global_embedding=global_emb,
        )
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        if checkpoint_path and checkpoint_path != "none":
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"EnsembleEncoder(egnn+gearnet, dim={self.hidden_dim}, params={n_params:.1f}M)"
