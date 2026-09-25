"""
GVP (Geometric Vector Perceptron) encoder wrapper.

Runs in a separate conda env (`gvp`) via subprocess to avoid
dependency conflicts with the main `protein_embeddings` env.

Architecture: Class C (Equivariant Field)
    SE(3)-equivariant graph convolutions via GVPConvLayer.
    Random weights only — no protein-pretrained checkpoint shipped.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
_HOME = _os.path.expanduser("~")

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)

_GVP_WORKER = Path(__file__).parent / "gvp_worker.py"


@register_encoder("gvp")
class GVPEncoder(BaseProteinEncoder):
    """GVP encoder — runs gvp_worker.py in `gvp` conda env via subprocess.

    Supports reproducible random initialization via seed or serialized state dict.
    """

    def __init__(
        self,
        node_s_dim: int = 100,
        node_v_dim: int = 16,
        edge_s_dim: int = 32,
        edge_v_dim: int = 1,
        num_layers: int = 3,
        k_neighbors: int = 30,
        dropout: float = 0.1,
        seed: int = None,
        state_dict_path: str = None,
    ):
        super().__init__()
        self.node_s_dim = node_s_dim
        self.node_v_dim = node_v_dim
        self.edge_s_dim = edge_s_dim
        self.edge_v_dim = edge_v_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.seed = seed
        self.state_dict_path = state_dict_path
        self._embedding_dim = node_s_dim + node_v_dim * 3

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.C

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            output_path = f.name

        try:
            input_data = {
                "coords":       protein.coords.cpu().numpy().tolist(),
                "sequence":     protein.sequence.cpu().numpy().tolist(),
                "mask":         protein.mask.cpu().numpy().tolist(),
                "node_s_dim":   self.node_s_dim,
                "node_v_dim":   self.node_v_dim,
                "edge_s_dim":   self.edge_s_dim,
                "edge_v_dim":   self.edge_v_dim,
                "num_layers":   self.num_layers,
                "k_neighbors":  self.k_neighbors,
                "dropout":      self.dropout,
                "output_path":  output_path,
            }
            with open(input_path, "w") as f:
                json.dump(input_data, f)

            # Build command with optional seed and state_dict_path
            cmd_parts = [
                f"python {_GVP_WORKER} --input {input_path}"
            ]
            if self.seed is not None:
                cmd_parts.append(f"--seed {self.seed}")
            if self.state_dict_path is not None:
                cmd_parts.append(f"--state-dict {self.state_dict_path}")

            # Find the conda env containing 'gvp'
            conda_bin = subprocess.run(
                ["conda", "info", "--base"], capture_output=True, text=True
            ).stdout.strip()
            # Try miniforge3 first, then miniconda3
            for base in [conda_bin, (_HOME + "/miniconda3")]:
                check = subprocess.run(
                    [f"{base}/bin/conda", "run", "-n", "gvp", "echo", "ok"],
                    capture_output=True, text=True
                )
                if check.returncode == 0:
                    conda_activate = f"{base}/bin/activate"
                    break

            bash_script = (
                f"source {conda_activate} gvp\n"
                + " && ".join(cmd_parts) + "\n"
            )
            result = subprocess.run(
                ["bash", "-c", bash_script],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"GVP encoding failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
                )

            result_data = torch.load(output_path, map_location=protein.coords.device, weights_only=False)
            embeddings = torch.from_numpy(result_data["embeddings"]).to(protein.coords.device)
            global_emb = torch.from_numpy(result_data["global_embedding"]).to(protein.coords.device)

            return EncoderOutput(
                residue_embeddings=embeddings,
                global_embedding=global_emb,
            )

        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def load_pretrained(self, checkpoint_path: str) -> None:
        self.pretrained_path = checkpoint_path
        self.state_dict_path = checkpoint_path

    def get_state_dict(self) -> dict:
        """Return state dict for serialization and reproducibility."""
        # For GVP, we need to serialize via the worker
        # This requires the worker to support state dict export
        raise NotImplementedError(
            "GVP state dict must be serialized via the subprocess worker. "
            "Use the --state-dict flag in gvp_worker.py."
        )
