from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pickle
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor

from src.models.base_encoder import (
    AA_VOCAB_INV,
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)


_DEFAULT_GEO_BPE_CKPT = Path(__file__).parents[3] / "checkpoints" / "bpe_iter=0.pkl"
_GEO_BPE_ROOT = Path(__file__).parents[3] / "external_repos" / "Geo_BPE" / "GeoBPE"
_AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


class _GeoBPEWorker(nn.Module):
    def __init__(self, checkpoint_path: Path, embedding_dim: int, random_seed: int):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.embedding_dim = embedding_dim
        self.random_seed = random_seed
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        if str(_GEO_BPE_ROOT) not in sys.path:
            sys.path.insert(0, str(_GEO_BPE_ROOT))

        from foldingdiff.angles_and_coords import (
            canonical_distances_and_dihedrals,
            extract_aa_seq,
            extract_backbone_coords,
            extract_c_beta_coords,
            extract_side_chain_coords,
        )
        from foldingdiff.tokenizer import Tokenizer

        self._bpe = pickle.load(open(self.checkpoint_path, "rb"))
        self._tokenizer_cls = Tokenizer
        self._canonical_distances_and_dihedrals = canonical_distances_and_dihedrals
        self._extract_aa_seq = extract_aa_seq
        self._extract_backbone_coords = extract_backbone_coords
        self._extract_c_beta_coords = extract_c_beta_coords
        self._extract_side_chain_coords = extract_side_chain_coords

        vocab_size_attr = getattr(self._bpe, "vocab_size", None)
        vocab_size = vocab_size_attr() if callable(vocab_size_attr) else vocab_size_attr
        if vocab_size is None:
            raise ValueError("Loaded GeoBPE checkpoint does not expose vocab_size")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.random_seed)
        emb = torch.randn(vocab_size, self.embedding_dim, generator=gen) / (self.embedding_dim ** 0.5)
        self._token_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        with torch.no_grad():
            self._token_embedding.weight.copy_(emb)
        self._token_embedding.weight.requires_grad = False
        self._initialized = True

    @staticmethod
    def _write_temp_pdb(coords_nca_c: torch.Tensor, sequence: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            atom_serial = 1
            for i, aa in enumerate(sequence, start=1):
                resname = _AA3.get(aa, "ALA")
                for atom_name, xyz in zip(("N", "CA", "C"), coords_nca_c[i - 1]):
                    x, y, z = [float(v) for v in xyz.tolist()]
                    f.write(
                        f"ATOM  {atom_serial:5d} {atom_name:^4s} {resname:>3s} A{i:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                    )
                    atom_serial += 1
            f.write("END\n")
            return f.name

    def _build_structure_from_pdb(self, pdb_path: str) -> Dict[str, Any]:
        aa = self._extract_aa_seq(pdb_path)
        return {
            "angles": self._canonical_distances_and_dihedrals(
                pdb_path,
                distances=["0C:1N", "N:CA", "CA:C"],
                angles=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
            ),
            "coords": self._extract_backbone_coords(pdb_path, atoms=["CA"]),
            "c_beta": self._extract_c_beta_coords(pdb_path),
            "full_coords": self._extract_backbone_coords(pdb_path, atoms=["N", "CA", "C"]),
            "full_idxes": sum([[i, i, i] for i in range(1, len(aa) + 1)], []),
            "side_chain": self._extract_side_chain_coords(pdb_path),
            "aa": aa,
            "fname": pdb_path,
        }

    @staticmethod
    def _residue_token_ids(tokenizer, quantized: List[int], n_res: int) -> List[int]:
        residue_ids = [0] * n_res
        motif_idx = 0
        for start, _, size in getattr(tokenizer, "tokens", []):
            residue_start = start // 3
            residue_end = min(n_res, residue_start + ((size + 1) // 3))
            token_id = int(quantized[motif_idx]) if motif_idx < len(quantized) else 0
            for r in range(residue_start, residue_end):
                residue_ids[r] = token_id
            motif_idx += 3 if size == 3 else 1
        return residue_ids

    @torch.no_grad()
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        self._lazy_init()
        device = protein.coords.device
        self._token_embedding = self._token_embedding.to(device)

        B, L = protein.sequence.shape
        residue_embeddings = torch.zeros(B, L, self.embedding_dim, device=device)

        for b in range(B):
            seq_len = int(protein.mask[b].sum().item())
            if seq_len < 2:
                continue

            seq = "".join(AA_VOCAB_INV.get(int(protein.sequence[b, i].item()), "A") for i in range(seq_len))
            coords_nca_c = protein.coords[b, :seq_len, :3, :].detach().cpu()
            pdb_path = self._write_temp_pdb(coords_nca_c, seq)
            try:
                structure = self._build_structure_from_pdb(pdb_path)
                if structure["angles"] is None:
                    continue
                tokenizer = self._tokenizer_cls(structure)
                self._bpe.tokenize(tokenizer)
                quantized = self._bpe.quantize(tokenizer.tokenize())
                residue_token_ids = self._residue_token_ids(tokenizer, quantized, seq_len)
                token_tensor = torch.tensor(residue_token_ids, dtype=torch.long, device=device)
                residue_embeddings[b, :seq_len] = self._token_embedding(token_tensor)
            except ValueError:
                continue
            finally:
                Path(pdb_path).unlink(missing_ok=True)

        mask_f = protein.mask.unsqueeze(-1).float()
        residue_embeddings = residue_embeddings * mask_f
        global_emb = (residue_embeddings * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        return EncoderOutput(
            residue_embeddings=residue_embeddings,
            global_embedding=global_emb,
            metadata={"checkpoint_path": str(self.checkpoint_path), "embedding_dim": self.embedding_dim},
        )


def _encode_single_sample(
    checkpoint_path: str,
    embedding_dim: int,
    random_seed: int,
    sequence: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    # Cache the worker per process to avoid reloading the 12MB BPE pickle for every protein.
    # ProcessPoolExecutor reuses worker processes across submitted jobs, so this module-level
    # cache survives across many _encode_single_sample calls within the same worker.
    global _CACHED_WORKER
    try:
        cached = _CACHED_WORKER
    except NameError:
        cached = None
    key = (checkpoint_path, embedding_dim, random_seed)
    if cached is None or cached[0] != key:
        worker = _GeoBPEWorker(Path(checkpoint_path), embedding_dim, random_seed)
        worker._lazy_init()
        _CACHED_WORKER = (key, worker)
    else:
        worker = cached[1]
    protein = ProteinInput(
        coords=coords.unsqueeze(0),
        sequence=sequence.unsqueeze(0),
        mask=mask.unsqueeze(0),
    )
    out = worker.encode(protein)
    success = bool(out.residue_embeddings.abs().sum().item() > 0)
    return out.residue_embeddings[0].cpu(), out.global_embedding[0].cpu(), success


# Module-level cache for worker reuse across jobs in same process
_CACHED_WORKER = None


@register_encoder("geobpe")
class GeoBPEEncoder(BaseProteinEncoder):
    """GeoBPE discrete-token encoder wrapper for InvariantsBench."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        embedding_dim: int = 128,
        random_seed: int = 0,
        env_name: str = "GeoBPE-6",
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_GEO_BPE_CKPT
        self._embedding_dim = embedding_dim
        self.random_seed = random_seed
        self.env_name = env_name
        self.num_workers = max(1, num_workers)
        self._worker = None
        self._pool = None

    def __del__(self):
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False)
            except Exception:
                pass

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.A

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _lazy_init(self) -> None:
        if self._worker is not None:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"GeoBPE checkpoint not found at {self.checkpoint_path}. "
                "Pass checkpoint_path to GeoBPEEncoder or benchmark_invariants.py."
            )
        try:
            self._worker = _GeoBPEWorker(self.checkpoint_path, self._embedding_dim, self.random_seed)
            self._worker._lazy_init()
        except Exception:
            self._worker = None

    def load_pretrained(self, checkpoint_path: str) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._worker = None

    def _run_in_geobpe_env(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        payload_path = Path("/tmp/geobpe_payload.pt")
        out_path = Path("/tmp/geobpe_out.pt")
        script_path = Path("/tmp/geobpe_bridge.py")

        torch.save(
            {
                "checkpoint_path": str(self.checkpoint_path),
                "embedding_dim": self._embedding_dim,
                "random_seed": self.random_seed,
                "coords": protein.coords.detach().cpu(),
                "sequence": protein.sequence.detach().cpu(),
                "mask": protein.mask.detach().cpu(),
            },
            payload_path,
        )

        project_root = Path(__file__).parents[3]
        script_path.write_text(
            "import sys\n"
            f"sys.path.insert(0, r'{project_root}')\n"
            "import torch\n"
            "from src.models.base_encoder import ProteinInput\n"
            "from src.models.pretrained.geobpe_encoder import _GeoBPEWorker\n"
            f"payload = torch.load(r'{payload_path}', map_location='cpu', weights_only=False)\n"
            "worker = _GeoBPEWorker(payload['checkpoint_path'], payload['embedding_dim'], payload['random_seed'])\n"
            "protein = ProteinInput(coords=payload['coords'], sequence=payload['sequence'], mask=payload['mask'])\n"
            "out = worker.encode(protein)\n"
            f"torch.save({{'residue_embeddings': out.residue_embeddings, 'global_embedding': out.global_embedding, 'metadata': out.metadata}}, r'{out_path}')\n"
        )

        result = subprocess.run(
            ["conda", "run", "-n", self.env_name, "python", str(script_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr[-4000:] if result.stderr else ""
            stdout = result.stdout[-2000:] if result.stdout else ""
            raise RuntimeError(
                "GeoBPE encoding failed in GeoBPE env.\n"
                f"STDOUT:\n{stdout}\nSTDERR(tail):\n{stderr}"
            )

        saved = torch.load(out_path, map_location=device, weights_only=False)
        return EncoderOutput(
            residue_embeddings=saved["residue_embeddings"].to(device),
            global_embedding=saved["global_embedding"].to(device),
            metadata=saved.get("metadata"),
        )

    @torch.no_grad()
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        self._lazy_init()
        if self._worker is not None and self.num_workers == 1:
            try:
                return self._worker.encode(protein)
            except Exception:
                self._worker = None
        if self.num_workers > 1:
            try:
                B = protein.sequence.shape[0]
                jobs = []
                for b in range(B):
                    jobs.append((
                        str(self.checkpoint_path),
                        self._embedding_dim,
                        self.random_seed,
                        protein.sequence[b].detach().cpu(),
                        protein.coords[b].detach().cpu(),
                        protein.mask[b].detach().cpu(),
                    ))
                # Reuse persistent pool across batches to avoid re-loading BPE model 16x per batch
                if self._pool is None:
                    self._pool = ProcessPoolExecutor(max_workers=self.num_workers)
                futures = [self._pool.submit(_encode_single_sample, *job) for job in jobs]
                results = [f.result() for f in futures]
                residue_embeddings = torch.stack([r[0] for r in results], dim=0).to(protein.coords.device)
                global_embedding = torch.stack([r[1] for r in results], dim=0).to(protein.coords.device)
                return EncoderOutput(
                    residue_embeddings=residue_embeddings,
                    global_embedding=global_embedding,
                    metadata={"checkpoint_path": str(self.checkpoint_path), "embedding_dim": self._embedding_dim},
                )
            except Exception:
                # Pool may have died; reset and fall through to subprocess fallback
                if self._pool is not None:
                    try:
                        self._pool.shutdown(wait=False)
                    except Exception:
                        pass
                    self._pool = None
        return self._run_in_geobpe_env(protein)
