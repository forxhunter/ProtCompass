"""
ProteinAE encoder wrapper.

ProteinAE is a diffusion-based protein autoencoder (NVIDIA) that maps backbone
coordinates into a continuous latent space via an AF3-style transformer encoder.

Reference:
    arXiv:2510.10634 — "ProteinAE: Diffusion Protein AutoEncoder for Structure Encoding"
    Checkpoint: external_repos/ProteinAE_v1/checkpoints/ae_r1_d8_v1.ckpt
    Encoder: AF3-style transformer, token_dim=256, 5 layers, dim_latent=8

Output: per-residue latent embeddings (B, L, 8).
    dim_latent=8 is the compressed structural bottleneck — small but information-dense.

Conda env: proteinae (requires openfold, lightning, einops, etc.)
    Inference runs via subprocess calling proteinae_worker.py.

Architecture: Class B (structure-based, no sequence pretraining)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)

_PROTEINAE_ROOT  = Path(__file__).parents[3] / "external_repos" / "ProteinAE_v1"
_DEFAULT_CKPT    = _PROTEINAE_ROOT / "checkpoints" / "ae_r1_d8_v1.ckpt"
_WORKER_SCRIPT   = Path(__file__).parent / "proteinae_worker.py"
_CONDA_ENV       = "proteinae"
_EMBEDDING_DIM   = 8   # dim_latent from hparams


@register_encoder("proteinae")
class ProteinAEEncoder(BaseProteinEncoder):
    """
    ProteinAE diffusion autoencoder (pretrained).

    Runs backbone coords → AF3-style transformer encoder → 8D latent per residue.
    Inference is delegated to proteinae_worker.py in the proteinae conda env.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__()
        self._ckpt = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CKPT
        self._embedding_dim = _EMBEDDING_DIM

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _run_worker(self, coords: torch.Tensor, masks: torch.Tensor,
                    seqs: torch.Tensor) -> torch.Tensor:
        """
        Serialize batch, call proteinae_worker.py in proteinae env, return embeddings.

        Args:
            coords: (N, L, 5, 3) float cpu
            masks:  (N, L) bool cpu
            seqs:   (N, L) long cpu

        Returns:
            embeddings: (N, L, 8) float cpu
        """
        if not self._ckpt.exists():
            raise FileNotFoundError(
                f"ProteinAE checkpoint not found: {self._ckpt}\n"
                f"Download: wget https://hf-mirror.com/lisn519010/ProteinAE_v1/"
                f"resolve/main/ae_r1_d8_v1.ckpt -O {self._ckpt}"
            )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            input_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            output_file = f.name

        try:
            torch.save({"coords": coords.cpu(), "masks": masks.cpu(),
                        "seqs": seqs.cpu()}, input_file)

            # Use bash activation instead of conda run (avoids cross-compiler issues)
            bash_script = f"""
source ~/miniconda3/bin/activate {_CONDA_ENV}
cd {_PROTEINAE_ROOT}
python {_WORKER_SCRIPT} --input {input_file} --output {output_file} --ckpt {self._ckpt} --root {_PROTEINAE_ROOT}
"""
            result = subprocess.run(
                ["bash", "-c", bash_script],
                capture_output=True,
                text=True,
                timeout=3600
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ProteinAE worker failed (exit {result.returncode}):\n"
                    f"{result.stderr[-2000:]}"
                )

            out = torch.load(output_file, map_location="cpu", weights_only=False)
            return out["embeddings"]  # (N, L, 8)
        finally:
            for p in [input_file, output_file]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        B, L, _, _ = protein.coords.shape

        embeddings = self._run_worker(
            coords=protein.coords.cpu(),
            masks=protein.mask.cpu(),
            seqs=protein.sequence.cpu(),
        )  # (B, L_worker, 8)

        # Align output length to input L (worker pads to max in batch)
        L_out = embeddings.shape[1]
        if L_out < L:
            pad = torch.zeros(B, L - L_out, self._embedding_dim)
            embeddings = torch.cat([embeddings, pad], dim=1)
        elif L_out > L:
            embeddings = embeddings[:, :L, :]

        embeddings = embeddings.to(device=device, dtype=torch.float32)

        mask_f = protein.mask.float().unsqueeze(-1)
        global_emb = (embeddings * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)

        return EncoderOutput(
            residue_embeddings=embeddings,
            global_embedding=global_emb,
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        self._ckpt = Path(checkpoint_path)
