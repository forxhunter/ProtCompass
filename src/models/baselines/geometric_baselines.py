"""
Geometric Baselines for InvariantsBench control experiments.

TorsionBaseline: "Perfect Local Information" — sin/cos(phi, psi)
DensityBaseline: "Perfect Packing Information" — multi-radius contact counts
"""

import torch
import torch.nn as nn
from ..base_encoder import BaseProteinEncoder, BottleneckClass, EncoderOutput, ProteinInput


class TorsionBaseline(BaseProteinEncoder):
    """
    Baseline that treats Torsion Angles (sin/cos) as the embedding.
    Tests 'Perfect Local Information'.
    Expected: High I1, Low I3.
    """
    bottleneck_class = BottleneckClass.C

    def __init__(self, **kwargs):
        super().__init__()
        self._embedding_dim = 4  # sin/cos phi, sin/cos psi

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        from src.data.preprocessing.geometry_calculator import compute_torsion_angles

        # coords: (B, L, 5, 3)
        torsions = compute_torsion_angles(protein.coords, protein.mask)
        phi = torsions["phi"]  # (B, L)
        psi = torsions["psi"]  # (B, L)

        # Handle NaNs (boundary residues)
        phi = torch.nan_to_num(phi, 0.0)
        psi = torch.nan_to_num(psi, 0.0)

        feat = torch.stack([
            torch.sin(phi), torch.cos(phi),
            torch.sin(psi), torch.cos(psi)
        ], dim=-1)  # (B, L, 4)

        # Global pooling (Mean over valid residues)
        mask = protein.mask.unsqueeze(-1).float()
        global_emb = (feat * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return EncoderOutput(residue_embeddings=feat, global_embedding=global_emb)

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass  # No pretrained weights


class DensityBaseline(BaseProteinEncoder):
    """
    Baseline that uses Contact Density as the embedding.
    Tests 'Perfect Packing Information'.
    Expected: Low I1, High I3.
    """
    bottleneck_class = BottleneckClass.C

    def __init__(self, **kwargs):
        super().__init__()
        self.radii = [6.0, 8.0, 10.0, 12.0]
        self._embedding_dim = len(self.radii)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        # CA coordinates: (B, L, 3)
        ca = protein.coords[:, :, 1, :]

        # Pairwise distances: (B, L, L)
        diff = ca.unsqueeze(2) - ca.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)

        feats = []
        target_mask = protein.mask.unsqueeze(1) * protein.mask.unsqueeze(2)

        for r in self.radii:
            contacts = (dist < r).float() * target_mask.float()
            # Exclude self-contacts
            diag = torch.eye(dist.shape[1], device=dist.device).unsqueeze(0)
            contacts = contacts - diag
            # Count neighbors per residue: (B, L)
            count = contacts.sum(dim=2)
            feats.append(count)

        emb = torch.stack(feats, dim=-1)  # (B, L, 4)

        # Global pooling
        mask = protein.mask.unsqueeze(-1).float()
        global_emb = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return EncoderOutput(residue_embeddings=emb, global_embedding=global_emb)

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass  # No pretrained weights


class RandomTokenBaseline(BaseProteinEncoder):
    """
    Random-token baseline: assigns each residue the same fixed random embedding.
    Vocab size = 1 means all residues get token 0 regardless of structure.

    Null model for GeoBPE: shows what score a single uniform random token achieves.
    Expected: ~0.5 on classification tasks (chance), ~0 correlation on regression.
    """
    bottleneck_class = BottleneckClass.C

    def __init__(self, embedding_dim: int = 128, seed: int = 42, **kwargs):
        super().__init__()
        self._embedding_dim = embedding_dim
        torch.manual_seed(seed)
        self.register_buffer('token_emb', torch.randn(1, embedding_dim))

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        B, L = protein.mask.shape
        emb = self.token_emb.unsqueeze(0).expand(B, L, -1)
        mask = protein.mask.unsqueeze(-1).float()
        global_emb = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return EncoderOutput(residue_embeddings=emb, global_embedding=global_emb)

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass