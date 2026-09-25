"""
ESM3 (EvolutionaryScale) encoder wrapper.

Runs in a separate conda env (`esm3`) via subprocess to avoid
dependency conflicts with the main `protein_embeddings` env.

Architecture: Class A (Discrete Token + Equivariant)
    Trimodal: sequence tokens + structure coordinates + function tokens.
    Random weights only — uses local weights file.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
_HOME = _os.path.expanduser("~")

import json
import subprocess
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

_ESM3_WORKER = Path(__file__).parent / "esm3_worker.py"


@register_encoder("esm3")
class ESM3Encoder(BaseProteinEncoder):
    """
    ESM3-sm-open (1.4B params, 1536D) — sequence + structure trimodal encoder.

    Runs in `esm3` conda env via subprocess.
    """

    def __init__(
        self,
        local_dir: str = None,
        use_structure: bool = True,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self._local_dir = Path(local_dir) if local_dir else Path(__file__).parents[3] / "esm3-sm-open-v1"
        self._use_structure = use_structure
        self._device = device
        self._embedding_dim = 1536

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.A

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
                "coords": protein.coords.cpu().numpy().tolist(),
                "sequence": protein.sequence.cpu().numpy().tolist(),
                "mask": protein.mask.cpu().numpy().tolist(),
                "output_path": output_path,
                "use_structure": self._use_structure,
            }

            with open(input_path, "w") as f:
                json.dump(input_data, f)

            # Find conda env for ESM3 - use the python directly from esm3 env
            conda_bin = subprocess.run(
                ["conda", "info", "--base"], capture_output=True, text=True
            ).stdout.strip()

            # Try miniforge3 first, then miniconda3
            esm3_python = None
            for base in [conda_bin, (_HOME + "/miniconda3"), (_HOME + "/miniforge3")]:
                python_path = Path(base) / "envs" / "esm3" / "bin" / "python"
                if python_path.exists():
                    esm3_python = str(python_path)
                    break

            if esm3_python is None:
                raise RuntimeError("Could not find esm3 conda environment")

            bash_script = f"{esm3_python} {_ESM3_WORKER} --input {input_path}\n"

            result = subprocess.run(
                ["bash", "-c", bash_script],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"ESM3 encoding failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
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
        """No-op: weights loaded automatically from local_dir."""
        pass

    def get_state_dict(self) -> dict:
        raise NotImplementedError(
            "ESM3 state dict must be serialized via the subprocess worker."
        )

    def set_state_dict(self, state_dict: dict) -> None:
        raise NotImplementedError(
            "ESM3 state dict must be loaded via the subprocess worker."
        )
