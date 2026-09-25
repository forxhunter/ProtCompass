"""
Sanity Baselines for InvariantsBench.

These are deterministic feature extractors that serve as "floors" for performance.
Any learned encoder MUST beat these to be considered non-trivial.

1. DensityBaseline: Multi-scale neighbor counts (should solve I3).
2. TorsionBaseline: Sin/Cos of Phi/Psi angles (should solve I1).
3. FrameBaseline: Relative frame quaternions (should solve I1).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from src.models.base_encoder import BaseProteinEncoder, EncoderOutput, ProteinInput, BottleneckClass
from src.data.preprocessing.geometry_calculator import compute_contact_density, compute_torsion_angles

class SanityBaseline(BaseProteinEncoder):
    def __init__(self, name: str, feature_dim: int):
        super().__init__()
        self._name = name
        self._dim = feature_dim
        
    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.A # Treated as discrete/fixed features
        
    @property
    def embedding_dim(self) -> int:
        return self._dim
        
    @property
    def name(self) -> str:
        return self._name
        
    def load_pretrained(self, path: str):
        pass # No weights to load

class DensityBaseline(SanityBaseline):
    """
    Baseline that just outputs neighbor counts at multiple radii.
    Should achieve high I3 scores.
    """
    def __init__(self, radii: List[float] = [6.0, 8.0, 10.0, 12.0]):
        # Output dim = num_radii
        super().__init__("DensityBaseline", len(radii))
        self.radii = radii
        
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        # compute_contact_density returns (B, L, num_radii)
        density = compute_contact_density(
            protein.coords, 
            protein.mask, 
            threshold=self.radii
        )
        return EncoderOutput(residue_embeddings=density)

class TorsionBaseline(SanityBaseline):
    """
    Baseline that outputs sin/cos of Phi/Psi angles.
    Should achieve high I1 scores (local conformational).
    """
    def __init__(self):
        # sin/cos phi, sin/cos psi = 4 dims
        super().__init__("TorsionBaseline", 4)
        
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        # compute_torsion_angles returns dict of phi/psi/omega
        torsion = compute_torsion_angles(protein.coords, protein.mask)
        
        phi = torsion["phi"]
        psi = torsion["psi"]
        
        # Convert to sin/cos
        features = torch.stack([
            torch.sin(phi), torch.cos(phi),
            torch.sin(psi), torch.cos(psi)
        ], dim=-1) # (B, L, 4)
        
        # Zero out mask (NaNs are handled by sin/cos but masking is cleaner)
        # compute_torsion_angles returns 0 for invalid, so sin(0)=0, cos(0)=1
        # We want masked positions to be 0 for the encoder output usually
        mask_expanded = protein.mask.unsqueeze(-1)
        features = features * mask_expanded
        
        return EncoderOutput(residue_embeddings=features)

