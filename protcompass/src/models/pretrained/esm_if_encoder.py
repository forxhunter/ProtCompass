"""
ESM-IF1 (Inverse Folding) encoder wrapper for InvariantsBench.

Uses the official ESM-IF1 pretrained model (142M parameters, 512D embeddings).
The model takes backbone coordinates (N, CA, C) and produces per-residue
structure-conditioned embeddings via a GVP-Transformer encoder.

Reference:
    Hsu et al. "Learning inverse folding from millions of predicted structures"
    https://github.com/facebookresearch/esm
"""

import sys
import types
from typing import Optional

import torch
import torch.nn as nn

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)


def _patch_esm_if_deps():
    """Monkey-patch biotite and torch_scatter to avoid dependency issues."""
    if "biotite" not in sys.modules:
        biotite = types.ModuleType("biotite")
        bs = types.ModuleType("biotite.structure")
        bs.AtomArray = type("AtomArray", (), {})
        bs.filter_backbone = bs.get_chains = lambda *a, **k: None
        biotite.structure = bs
        for name, mod in [
            ("biotite", biotite),
            ("biotite.structure", bs),
            ("biotite.structure.io", types.ModuleType("biotite.structure.io")),
            ("biotite.structure.io.pdbx", types.ModuleType("biotite.structure.io.pdbx")),
            ("biotite.structure.io.pdb", types.ModuleType("biotite.structure.io.pdb")),
            ("biotite.structure.residues", types.ModuleType("biotite.structure.residues")),
            ("biotite.sequence", types.ModuleType("biotite.sequence")),
        ]:
            sys.modules[name] = mod
        sys.modules["biotite.structure.residues"].get_residues = lambda *a, **k: None
        sys.modules["biotite.sequence"].ProteinSequence = type("ProteinSequence", (), {})
        sys.modules["biotite.structure.io"].pdbx = sys.modules["biotite.structure.io.pdbx"]
        sys.modules["biotite.structure.io"].pdb = sys.modules["biotite.structure.io.pdb"]

    if "torch_scatter" not in sys.modules:

        def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
            if dim_size is None:
                dim_size = int(index.max()) + 1
            size = list(src.shape)
            size[dim] = dim_size
            if out is None:
                out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index.expand_as(src), src)

        def scatter(src, index, dim=-1, out=None, dim_size=None, fill_value=0, reduce="sum"):
            if reduce in ("sum", "add"):
                return scatter_add(src, index, dim, out, dim_size, fill_value)
            elif reduce == "mean":
                s = scatter_add(src, index, dim, None, dim_size, fill_value)
                c = scatter_add(torch.ones_like(src), index, dim, None, dim_size, 0)
                return s / c.clamp(min=1)
            raise ValueError(f"Unknown reduce: {reduce}")

        ts = types.ModuleType("torch_scatter")
        ts.scatter_add = scatter_add
        ts.scatter = scatter
        sys.modules["torch_scatter"] = ts


@register_encoder("esm_if")
class ESMIFEncoder(BaseProteinEncoder):
    """ESM-IF1 GVP-Transformer encoder (142M params, 512D)."""

    def __init__(self, **kwargs):
        super().__init__()
        _patch_esm_if_deps()
        import torch

        # Load via torch.hub (fair-esm package doesn't include IF models)
        self._model, self._alphabet = torch.hub.load(
            'facebookresearch/esm:main', 'esm_if1_gvp4_t16_142M_UR50'
        )
        self._model.eval()
        self._encoder = self._model.encoder
        self._embed_dim = self._model.args.encoder_embed_dim  # 512

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.C

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        B, L = protein.coords.shape[0], protein.coords.shape[1]
        device = protein.coords.device

        # ESM-IF expects coords as (B, L, 3, 3) for N, CA, C atoms
        # Our ProteinInput has coords (B, L, 5, 3) for N, CA, C, O, CB
        coords_ncc = protein.coords[:, :, :3, :]  # (B, L, 3, 3)

        # Padding mask: True = padding (inverse of our mask where True = valid)
        padding_mask = ~protein.mask  # (B, L)

        # Confidence: 1.0 for valid residues, 0.0 for padding
        confidence = protein.mask.float()  # (B, L)

        with torch.no_grad():
            result = self._encoder(
                coords_ncc, padding_mask, confidence, return_all_hiddens=False
            )
            # encoder_out is (L, B, D), transpose to (B, L, D)
            enc_out = result["encoder_out"][0].transpose(0, 1)

        # Zero out padded positions
        enc_out = enc_out * protein.mask.unsqueeze(-1).float()

        return EncoderOutput(residue_embeddings=enc_out)

    def load_pretrained(self, checkpoint_path: str):
        """No-op: weights are loaded automatically from ESM hub."""
        pass
