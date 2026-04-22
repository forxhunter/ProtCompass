"""
Starling encoder wrapper.

Starling (Novak, Lotthammer et al. 2025) is a latent-space diffusion model for
predicting IDP conformational ensembles. The VAE component encodes pairwise
Cα-Cα distance matrices into a structural latent space.

We adapt it as a structure encoder by:
  1. Computing Cα-Cα distance matrix from backbone coords
  2. Normalising and encoding with the pretrained VAE encoder
  3. Global-average-pooling the spatial latent → global embedding
  4. Broadcasting global embedding to per-residue level

Architecture notes:
  - Input: 2D Cα distance matrix (NOT sequence)
  - Backbone: ResNet VAE encoder
  - Max sequence length: 380 residues (STARLING MAX_SEQUENCE_LENGTH)
  - Pretrained on coarse-grained IDP MD ensembles — transfers to folded proteins
    but the latent captures distance-map geometry, not IDP dynamics per se.
  - Latent dim: determined by checkpoint (typically 4–16 channels)
  - Embedding dim: latent_dim (determined at runtime from loaded checkpoint)

Reference:
  Novak B., Lotthammer J. M., Emenecker R. J. & Holehouse A. S.
  "Accurate predictions of conformational ensembles of disordered proteins
  with STARLING." bioRxiv 2025.02.14.638373

Conda env: starling
  Inference is delegated to starling_worker.py via subprocess.

Source: /mnt/f/2026_structuerembed/external_repos/starling/
"""

import json
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

_WORKER_SCRIPT = Path(__file__).parent / "starling_worker.py"
_CONDA_ENV     = "starling"
_UNKNOWN_DIM   = 16    # fallback if worker hasn't run yet


@register_encoder("starling")
class StarlingEncoder(BaseProteinEncoder):
    """
    Starling VAE encoder — encodes Cα distance matrices into structural latents.

    Requires backbone Cα coordinates; sequences alone are not sufficient.
    Runs in the 'starling' conda env via subprocess.
    Output: (latent_dim,) global embedding, broadcast to per-residue.
    """

    def __init__(self, embedding_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self._embedding_dim = embedding_dim or _UNKNOWN_DIM
        self._cache: dict = {}

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.CONTINUOUS_FIELD

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, batch: ProteinInput) -> EncoderOutput:
        """
        Encode a batch of proteins by calling the starling_worker subprocess.
        Caches results by domain name to avoid re-computation.
        """
        names     = batch.names if hasattr(batch, "names") else [f"prot_{i}" for i in range(len(batch.coords))]
        coords    = batch.coords           # list of (L, 3) tensors (Cα only or full backbone)
        sequences = batch.sequences if hasattr(batch, "sequences") else [""] * len(coords)

        # Build list of proteins to encode (skip cached)
        to_encode = []
        for name, coord, seq in zip(names, coords, sequences):
            if name not in self._cache:
                if coord is not None and coord.shape[0] >= 2:
                    # Extract Cα: if shape is (L, 3) assume already Cα;
                    # if (L, 4, 3) take index 1 (CA)
                    if coord.ndim == 3:
                        ca = coord[:, 1, :]
                    else:
                        ca = coord
                    to_encode.append({
                        "name": name,
                        "sequence": seq if isinstance(seq, str) else "",
                        "ca_coords": ca.cpu().float().tolist(),
                    })

        if to_encode:
            self._run_worker(to_encode)

        # Assemble outputs
        residue_embs = []
        global_embs  = []
        for name, coord in zip(names, coords):
            L = coord.shape[0] if coord.ndim == 2 else coord.shape[0]
            if name in self._cache:
                g = torch.tensor(self._cache[name], dtype=torch.float32)
                self._embedding_dim = g.shape[0]
                r = g.unsqueeze(0).expand(L, -1)
            else:
                # Fallback: zeros
                g = torch.zeros(self._embedding_dim, dtype=torch.float32)
                r = g.unsqueeze(0).expand(L, -1)
            residue_embs.append(r)
            global_embs.append(g)

        return EncoderOutput(
            residue_embeddings=torch.stack(residue_embs),
            global_embedding=torch.stack(global_embs),
        )

    def _run_worker(self, proteins: list):
        """Serialize proteins → subprocess → deserialise results into cache."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
            json.dump({"proteins": proteins}, f_in)
            input_path = f_in.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_out:
            output_path = f_out.name

        try:
            conda_run = [
                "conda", "run", "--no-capture-output", "-n", _CONDA_ENV,
                "python", str(_WORKER_SCRIPT), input_path, output_path,
            ]
            result = subprocess.run(
                conda_run,
                capture_output=False,
                timeout=3600,
            )
            if result.returncode != 0:
                print(f"[StarlingEncoder] Worker exited with code {result.returncode}",
                      file=sys.stderr)
                return

            with open(output_path) as f:
                out = json.load(f)

            for name, v in out.items():
                self._cache[name] = v["global_emb"]
                self._embedding_dim = v["emb_dim"]

        finally:
            os.unlink(input_path)
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass

    def load_pretrained(self, checkpoint_path: str) -> None:
        # Weights are managed by Starling's own ModelManager (downloads to ~/.starling_weights)
        pass

    def __repr__(self) -> str:
        return f"StarlingEncoder(embedding_dim={self._embedding_dim}, env={_CONDA_ENV})"
