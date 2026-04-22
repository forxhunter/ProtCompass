"""
GCPNet (Geometry-Complete Perceptron Network) encoder wrapper.

Uses the official GCPNet implementation from
/mnt/f/2026_structuerembed/external_repos/GCPNet/src/.

Runs in a separate conda env (`gcpnet`) via subprocess to avoid
dependency conflicts with the main `protein_embeddings` env.

Architecture: Class C (Equivariant Field)
    Random weights only — no protein-pretrained checkpoint shipped.
    Pretraining on Swiss-Prot AlphaFold structures recommended.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)

# Worker script path
_GCPNET_WORKER = Path(__file__).parent / "gcpnet_worker.py"


@register_encoder("gcpnet")
class GCPNetEncoder(BaseProteinEncoder):
    """
    GCPNet encoder using the official GCPNet library.

    GCPNet uses Geometry-Complete Perceptrons (GCP) that achieve geometric
    completeness by processing features in local coordinate frames derived
    from edge geometry.

    Runs in `gcpnet` conda env via subprocess.
    """

    bottleneck_class = BottleneckClass.C

    def __init__(
        self,
        node_s_dim: int = 128,
        node_v_dim: int = 16,
        edge_s_dim: int = 32,
        edge_v_dim: int = 1,
        num_layers: int = 3,
        k_neighbors: int = 30,
        dropout: float = 0.1,
        pretrained_path: str = None,
    ):
        super().__init__()
        self.node_s_dim = node_s_dim
        self.node_v_dim = node_v_dim
        self.edge_s_dim = edge_s_dim
        self.edge_v_dim = edge_v_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.pretrained_path = pretrained_path
        self._embedding_dim = node_s_dim + node_v_dim * 3

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            output_path = f.name

        try:
            # Serialize protein data
            coords = protein.coords.cpu().numpy().tolist()
            sequence = protein.sequence.cpu().numpy().tolist()
            mask = protein.mask.cpu().numpy().tolist()

            input_data = {
                "coords": coords,
                "sequence": sequence,
                "mask": mask,
                "node_s_dim": self.node_s_dim,
                "node_v_dim": self.node_v_dim,
                "edge_s_dim": self.edge_s_dim,
                "edge_v_dim": self.edge_v_dim,
                "num_layers": self.num_layers,
                "k_neighbors": self.k_neighbors,
                "dropout": self.dropout,
                "pretrained_path": self.pretrained_path,
                "output_path": output_path,
            }

            with open(input_path, "w") as f:
                json.dump(input_data, f)

            # Run in gcpnet conda env from GCPNet directory to avoid path collision
            gcpnet_root = Path(__file__).parent.parent.parent.parent / "external_repos" / "GCPNet"

            # Use bash script to activate conda env (conda run has cross-compiler issues)
            bash_script = f"""
source ~/miniconda3/bin/activate gcpnet
cd {gcpnet_root}
python {_GCPNET_WORKER} --input {input_path}
"""
            result = subprocess.run(
                ["bash", "-c", bash_script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"GCPNet encoding failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
                )

            # Load output
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
