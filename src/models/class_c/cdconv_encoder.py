"""
CDConv (Continuous-Discrete Convolution) encoder wrapper.

Uses the official CDConv implementation from
/mnt/f/2026_structuerembed/external_repos/Continuous-Discrete-Convolution/.

CDConv operates on point clouds with a continuous orientation kernel,
combining discrete graph convolutions with continuous angular filters.

Reference:
    Liu et al. "Continuous-Discrete Convolution for Geometry-Aware Protein Representation"
    https://github.com/AI4ScienceOpen/Continuous-Discrete-Convolution

Architecture: Class C (Equivariant Field)
    Random weights only — no protein-pretrained checkpoint shipped.
    Pretraining on Swiss-Prot AlphaFold structures recommended.
"""

import sys
from pathlib import Path

_CDCV_REPO = Path(__file__).parent.parent.parent.parent / "external_repos" / "Continuous-Discrete-Convolution"
if str(_CDCV_REPO) not in sys.path:
    sys.path.insert(0, str(_CDCV_REPO))

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

# Import CDConv's Model class from the external repo
# Must come after sys.path manipulation above
# Use importlib to avoid conflict with src.models package
import importlib.util
_cdconv_models_path = _CDCV_REPO / "models.py"
_spec = importlib.util.spec_from_file_location("cdconv_models", _cdconv_models_path)
_cdconv_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cdconv_models)
_CDConvModel = _cdconv_models.Model


def _orientation_from_ca(pos: torch.Tensor) -> torch.Tensor:
    """
    Compute local orientation frame from CA positions.

    Uses the same algorithm as CDConv's orientation() function in utils.py:
    1. Tangent vectors between consecutive CA atoms
    2. Binormal: difference of consecutive tangents
    3. Normal: cross product of consecutive tangents
    4. Orientation: cross of binormal and normal

    Args:
        pos: (N, 3) CA positions

    Returns:
        ori: (N, 3, 3) orientation matrices
    """
    N = pos.shape[0]
    if N < 3:
        return torch.eye(3, device=pos.device, dtype=pos.dtype).unsqueeze(0).expand(N, -1, -1)

    diff = pos[1:] - pos[:-1]
    norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
    u = diff / norm  # tangents

    u1 = u[1:]
    u2 = u[:-1]

    b = u2 - u1
    b_norm = torch.norm(b, dim=-1, keepdim=True).clamp(min=1e-8)
    b = b / b_norm

    n = torch.cross(u2, u1)
    n_norm = torch.norm(n, dim=-1, keepdim=True).clamp(min=1e-8)
    n = n / n_norm

    o = torch.cross(b, n)
    o_norm = torch.norm(o, dim=-1, keepdim=True).clamp(min=1e-8)
    o = o / o_norm

    ori_center = torch.stack([b, n, o], dim=1)

    ori = torch.cat([
        ori_center[0:1],
        ori_center,
        ori_center[-1:],
    ], dim=0)
    return ori


class _CDConvEncoderBackbone(_CDConvModel):
    """
    CDConv model that outputs per-residue embeddings.

    Mirrors the official CDConv Model but intercepts the forward pass
    to return per-residue features instead of classification logits.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, data):
        """
        Encode a protein with CDConv, returning per-residue embeddings.

        Args:
            data: Object with x, pos, seq, ori, batch attributes

        Returns:
            (n_nodes, channels[-1]) per-residue embeddings
        """
        x = self.embedding(data.x)
        pos = data.pos
        seq = data.seq
        ori = data.ori
        batch = data.batch

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)
            # Skip pooling to preserve per-residue resolution and avoid batch size mismatch
            # (pooling causes assertion failures in radius() when batch tensor size != node count)

        return x


@register_encoder("cdconv")
class CDConvEncoder(BaseProteinEncoder):
    """
    CDConv encoder using the official Continuous-Discrete-Convolution library.

    CDConv combines continuous angular filters with discrete graph convolutions,
    operating on per-residue point clouds with local orientation frames.

    The orientation is computed from CA atom positions using a local coordinate
    frame (binormal, normal, orientation axes).

    Args:
        geometric_radii: List of radii for each CDConv layer
        sequential_kernel_size: Size of the sequential kernel
        kernel_channels: Number of channels in the angular kernel
        channels: Output channels for each layer
        embedding_dim: Dimension of amino acid embedding
        dropout: Dropout probability
    """

    bottleneck_class = BottleneckClass.C

    def __init__(
        self,
        geometric_radii: list = None,
        sequential_kernel_size: int = 5,
        kernel_channels: list = None,
        channels: list = None,
        embedding_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()

        if geometric_radii is None:
            geometric_radii = [10.0, 12.0, 15.0]
        if kernel_channels is None:
            kernel_channels = [8, 16, 32]
        if channels is None:
            channels = [64, 128, 256]

        self.geometric_radii = geometric_radii
        self.sequential_kernel_size = sequential_kernel_size
        self.kernel_channels = kernel_channels
        self.channels = channels
        self._embedding_dim = channels[-1]

        self.model = _CDConvEncoderBackbone(
            geometric_radii=geometric_radii,
            sequential_kernel_size=sequential_kernel_size,
            kernel_channels=kernel_channels,
            channels=channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_classes=channels[-1],
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        B, L = protein.coords.shape[:2]

        ca_pos = protein.coords[:, :, 1, :].float()
        seq = protein.sequence.clamp(0, 20)
        mask = protein.mask.bool()

        embeddings_list = []
        global_list = []

        for b in range(B):
            L_b = mask[b].sum().item()
            if L_b == 0:
                embeddings_list.append(torch.zeros(0, self._embedding_dim, device=device))
                global_list.append(torch.zeros(self._embedding_dim, device=device))
                continue

            ca_valid = ca_pos[b, :L_b]
            center = ca_valid.mean(dim=0, keepdim=True)
            ca_centered = ca_valid - center
            ori = _orientation_from_ca(ca_centered)
            seq_valid = seq[b, :L_b].clamp(0, 20)

            class _Data:
                pass

            data = _Data()
            data.x = seq_valid.long()
            data.pos = ca_centered
            # seq must be residue position indices [0, 1, ..., N-1] shaped [N, 1],
            # NOT amino acid types. CDConv uses seq_j - seq_i as sequential distance
            # to index into the WeightNet sequential kernel.
            data.seq = torch.arange(L_b, dtype=torch.long, device=device).unsqueeze(-1)
            data.ori = ori
            data.batch = torch.zeros(L_b, dtype=torch.long, device=device)

            embeddings = self.model.encode(data)
            global_emb = embeddings.mean(dim=0)

            embeddings_list.append(embeddings)
            global_list.append(global_emb)

        # Always pad to the original input length L so that mask shapes are
        # consistent across batches (probe_battery cats masks along dim=0).
        # CDConv's local_mean_pool shrinks residue count, so pooled_len <= L_b <= L.
        embeddings_batched = torch.zeros(B, L, self._embedding_dim, device=device)
        mask_batched = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            L_b_out = embeddings_list[b].shape[0]
            if L_b_out > 0:
                fill = min(L_b_out, L)
                embeddings_batched[b, :fill] = embeddings_list[b][:fill]
                mask_batched[b, :fill] = True

        global_cat = torch.stack(global_list, dim=0)

        return EncoderOutput(
            residue_embeddings=embeddings_batched,
            global_embedding=global_cat,
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "encoder_state" in ckpt:
            self.model.load_state_dict(ckpt["encoder_state"], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
