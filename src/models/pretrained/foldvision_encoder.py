"""
FoldVision encoder wrapper.

FoldVision is a 3D-CNN protein encoder that voxelizes the atomic point cloud
of a protein structure into a sparse 3D grid and produces a 1024D global
embedding via hierarchical 3D convolutions + adaptive average pooling.

Reference:
    AlexanderKroll/foldvision-encoder (HuggingFace)
    Repo: external_repos/FoldVision/
    Input: PDB file → preprocess_Structures (numba) → 3D voxel grid → 1024D

Conda env: foldvision (uses numba — incompatible with protein_embeddings)
    Inference is run via subprocess calling foldvision_worker.py

Architecture: protein-level only (global 1024D embedding, no per-residue).
    residue_embeddings: global embedding broadcast across valid residues.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)

_FOLDVISION_ROOT = Path(__file__).parents[3] / "external_repos" / "FoldVision"
_PDB_DIR         = Path(__file__).parents[3] / "data" / "raw" / "cath" / "structures"
_WORKER_SCRIPT   = Path(__file__).parent / "foldvision_worker.py"
_HF_MODEL_ID     = "AlexanderKroll/foldvision-encoder"
_CONDA_ENV       = "foldvision"


@register_encoder("foldvision")
class FoldVisionEncoder(BaseProteinEncoder):
    """
    FoldVision 3D-CNN encoder (AlexanderKroll/foldvision-encoder).

    Requires PDB files at data/raw/cath/structures/<pdbid>.pdb.
    Runs preprocessing and inference in the foldvision conda env via subprocess.
    Output: 1024D global embedding; residue_embeddings = broadcast of global.
    """

    def __init__(self, pdb_dir: Optional[str] = None, **kwargs):
        super().__init__()
        self._pdb_dir = Path(pdb_dir) if pdb_dir else _PDB_DIR
        self._embedding_dim = 1024
        self._cache: dict = {}  # pdb_id → embedding tensor

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B   # structure-based

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _pdb_id_from_protein(self, protein: ProteinInput, idx: int) -> Optional[str]:
        """Extract PDB ID from protein metadata if available."""
        if hasattr(protein, "pdb_ids") and protein.pdb_ids is not None:
            return protein.pdb_ids[idx].lower()
        return None

    def _find_pdb(self, pdb_id: str) -> Optional[Path]:
        """Find PDB file for given ID."""
        for name in [pdb_id, pdb_id.upper(), pdb_id.lower()]:
            p = self._pdb_dir / f"{name}.pdb"
            if p.exists():
                return p
        return None

    def _embed_pdbs_via_subprocess(self, pdb_paths: list[str]) -> dict[str, np.ndarray]:
        """
        Run FoldVision inference in foldvision conda env via subprocess.
        Returns dict mapping pdb_path → (1024,) numpy array.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            input_file = f.name
            json.dump(pdb_paths, f)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            output_file = f.name

        try:
            cmd = [
                "conda", "run", "-n", _CONDA_ENV, "--no-capture-output",
                "python", str(_WORKER_SCRIPT),
                "--input", input_file,
                "--output", output_file,
                "--model_id", _HF_MODEL_ID,
                "--foldvision_root", str(_FOLDVISION_ROOT),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                raise RuntimeError(
                    f"FoldVision worker failed:\n{result.stderr[-2000:]}"
                )
            embeddings = torch.load(output_file, map_location="cpu",
                                    weights_only=False)
            return embeddings
        finally:
            os.unlink(input_file)
            try:
                os.unlink(output_file)
            except FileNotFoundError:
                pass

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        B, L, _, _ = protein.coords.shape

        global_embs = torch.zeros(B, self._embedding_dim)
        found = [False] * B

        # Collect PDB paths for proteins not yet cached
        to_embed = {}  # idx → pdb_path
        for i in range(B):
            pdb_id = self._pdb_id_from_protein(protein, i)
            if pdb_id is None:
                continue
            if pdb_id in self._cache:
                global_embs[i] = self._cache[pdb_id]
                found[i] = True
                continue
            pdb_path = self._find_pdb(pdb_id)
            if pdb_path is not None:
                to_embed[i] = str(pdb_path)

        # Batch subprocess call
        if to_embed:
            pdb_paths = list(to_embed.values())
            results = self._embed_pdbs_via_subprocess(pdb_paths)
            path_to_idx = {v: k for k, v in to_embed.items()}
            for path, emb in results.items():
                idx = path_to_idx.get(path)
                if idx is not None:
                    emb_t = torch.tensor(emb, dtype=torch.float32)
                    global_embs[idx] = emb_t
                    pdb_id = self._pdb_id_from_protein(protein, idx)
                    if pdb_id:
                        self._cache[pdb_id] = emb_t
                    found[idx] = True

        global_embs = global_embs.to(device)

        # Broadcast global embedding to all valid residue positions
        residue_embs = global_embs.unsqueeze(1).expand(B, L, -1).clone()
        mask = protein.mask.unsqueeze(-1).float()
        residue_embs = residue_embs * mask

        return EncoderOutput(
            residue_embeddings=residue_embs,
            global_embedding=global_embs,
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass  # weights loaded from HF in worker subprocess
