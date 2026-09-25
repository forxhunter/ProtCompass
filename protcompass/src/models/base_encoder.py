"""
Base encoder interface for protein structure embedding models.

This module defines the abstract interface that all encoder wrappers must implement,
ensuring uniform embedding extraction across different model classes (A, B, C).

Bottleneck Classes:
    - Class A: Discrete token bottleneck (Foldseek 3Di, SaProt, ProstT5)
    - Class B: Neighborhood graph bottleneck (ProteinMPNN, GearNet)
    - Class C: Equivariant field bottleneck (GVP, EGNN, SE3-Transformer)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BottleneckClass(Enum):
    """Taxonomy of representation bottleneck types."""

    A = "discrete_token"  # Discrete structural tokens (3Di, SA tokens)
    B = "graph"  # Neighborhood graph (k-NN message passing)
    C = "equivariant"  # Equivariant field (SE(3)/E(n) equivariant)


@dataclass
class EncoderOutput:
    """
    Standardized output from protein structure encoders.

    Attributes:
        residue_embeddings: Per-residue embeddings of shape (B, L, D)
        global_embedding: Optional pooled protein-level embedding (B, D)
        pair_embeddings: Optional pairwise embeddings (B, L, L, D_pair)
        intermediate: Optional dict of intermediate layer outputs
        attention: Optional attention weights if available
        metadata: Optional dict for encoder-specific outputs
    """

    residue_embeddings: torch.Tensor  # (B, L, D) - required
    global_embedding: Optional[torch.Tensor] = None  # (B, D)
    pair_embeddings: Optional[torch.Tensor] = None  # (B, L, L, D_pair)
    intermediate: Optional[Dict[str, torch.Tensor]] = None
    attention: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

    def to(self, device: Union[str, torch.device]) -> "EncoderOutput":
        """Move all tensors to specified device."""
        return EncoderOutput(
            residue_embeddings=self.residue_embeddings.to(device),
            global_embedding=(
                self.global_embedding.to(device) if self.global_embedding is not None else None
            ),
            pair_embeddings=(
                self.pair_embeddings.to(device) if self.pair_embeddings is not None else None
            ),
            intermediate=(
                {k: v.to(device) for k, v in self.intermediate.items()}
                if self.intermediate is not None
                else None
            ),
            attention=self.attention.to(device) if self.attention is not None else None,
            metadata=self.metadata,
        )


@dataclass
class ProteinInput:
    """
    Standardized input for protein structure encoders.

    Attributes:
        coords: Backbone atom coordinates (B, L, num_atoms, 3)
                Atom order: N, CA, C, O, CB (CB is virtual for Glycine)
        sequence: Amino acid indices 0-19 (B, L)
        mask: Valid residue positions (B, L), True = valid
        chain_ids: Optional chain identifiers (B, L)
        residue_ids: Optional residue numbers (B, L)
        structure_paths: Optional source PDB paths, one per batch item. This is
                         required by encoders whose native preprocessing starts
                         from a PDB file rather than an in-memory coordinate tensor.
    """

    coords: torch.Tensor  # (B, L, 5, 3) for N, CA, C, O, CB
    sequence: torch.Tensor  # (B, L) amino acid indices
    mask: torch.Tensor  # (B, L) boolean mask
    chain_ids: Optional[torch.Tensor] = None
    residue_ids: Optional[torch.Tensor] = None
    structure_paths: Optional[List[Optional[str]]] = None

    def to(self, device: Union[str, torch.device]) -> "ProteinInput":
        """Move all tensors to specified device."""
        return ProteinInput(
            coords=self.coords.to(device),
            sequence=self.sequence.to(device),
            mask=self.mask.to(device),
            chain_ids=self.chain_ids.to(device) if self.chain_ids is not None else None,
            residue_ids=self.residue_ids.to(device) if self.residue_ids is not None else None,
            structure_paths=self.structure_paths,
        )

    @property
    def batch_size(self) -> int:
        return self.coords.shape[0]

    @property
    def max_length(self) -> int:
        return self.coords.shape[1]

    def get_ca_coords(self) -> torch.Tensor:
        """Extract CA coordinates (B, L, 3)."""
        return self.coords[:, :, 1, :]  # CA is index 1

    def get_cb_coords(self) -> torch.Tensor:
        """Extract CB coordinates (B, L, 3)."""
        return self.coords[:, :, 4, :]  # CB is index 4


class BaseProteinEncoder(ABC, nn.Module):
    """
    Abstract base class for all protein structure encoders.

    All encoder wrappers must inherit from this class and implement:
    - bottleneck_class: Return the bottleneck taxonomy class (A, B, or C)
    - embedding_dim: Return the dimension of per-residue embeddings
    - encode(): Encode protein structure to embeddings
    - load_pretrained(): Load pretrained weights

    Example:
        >>> encoder = SaProtEncoder()
        >>> encoder.load_pretrained("checkpoints/saprot.pt")
        >>> protein = ProteinInput(coords=..., sequence=..., mask=...)
        >>> output = encoder.encode(protein)
        >>> embeddings = output.residue_embeddings  # (B, L, 1280)
    """

    def __init__(self):
        super().__init__()
        self._is_frozen = False

    @property
    @abstractmethod
    def bottleneck_class(self) -> BottleneckClass:
        """Return the bottleneck taxonomy class (A, B, or C)."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of per-residue embeddings."""
        pass

    @property
    def name(self) -> str:
        """Return the encoder name (defaults to class name)."""
        return self.__class__.__name__

    @abstractmethod
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        """
        Encode protein structure to embeddings.

        Args:
            protein: ProteinInput containing coords, sequence, and mask

        Returns:
            EncoderOutput with at minimum residue_embeddings (B, L, D)
        """
        pass

    @abstractmethod
    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pt, .pth, .ckpt)
        """
        pass

    def freeze(self) -> None:
        """Freeze all encoder parameters for probing."""
        for param in self.parameters():
            param.requires_grad = False
        self._is_frozen = True
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        self._is_frozen = False

    @property
    def is_frozen(self) -> bool:
        """Check if encoder is frozen."""
        return self._is_frozen

    def forward(self, protein: ProteinInput) -> EncoderOutput:
        """Forward pass (alias for encode)."""
        return self.encode(protein)

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"{self.name}("
            f"bottleneck={self.bottleneck_class.name}, "
            f"dim={self.embedding_dim}, "
            f"params={self.get_num_parameters() / 1e6:.1f}M, "
            f"frozen={self.is_frozen})"
        )


# Encoder registry for dynamic loading
ENCODER_REGISTRY: Dict[str, type] = {}


def register_encoder(name: str):
    """
    Decorator to register an encoder class.

    Usage:
        @register_encoder("saprot")
        class SaProtEncoder(BaseProteinEncoder):
            ...
    """

    def decorator(cls):
        ENCODER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_encoder(name: str, **kwargs) -> BaseProteinEncoder:
    """
    Get encoder by name from registry.

    Args:
        name: Registered encoder name (case-insensitive)
        **kwargs: Arguments to pass to encoder constructor

    Returns:
        Instantiated encoder

    Raises:
        KeyError: If encoder name not found in registry
    """
    name_lower = name.lower()
    if name_lower not in ENCODER_REGISTRY:
        available = list(ENCODER_REGISTRY.keys())
        raise KeyError(f"Encoder '{name}' not found. Available: {available}")

    return ENCODER_REGISTRY[name_lower](**kwargs)


def list_encoders() -> List[str]:
    """List all registered encoder names."""
    return list(ENCODER_REGISTRY.keys())


# Amino acid vocabulary
AA_VOCAB = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}
AA_VOCAB_INV = {v: k for k, v in AA_VOCAB.items()}


def sequence_to_tensor(sequence: str) -> torch.Tensor:
    """Convert amino acid string to tensor of indices."""
    return torch.tensor([AA_VOCAB.get(aa, 0) for aa in sequence.upper()])



def tensor_to_sequence(tensor: torch.Tensor) -> str:
    """Convert tensor of indices to amino acid string."""
    return "".join(AA_VOCAB_INV.get(idx.item(), "X") for idx in tensor)

class RandomEncoder(BaseProteinEncoder):
    """
    Random Gaussian Embeddings (Sanity Check).
    """
    bottleneck_class = BottleneckClass.C

    def __init__(self, hidden_dim: int = 128, seed: int = None, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seed = seed
        self._generator = torch.Generator()
        if seed is not None:
            self._generator = self._generator.manual_seed(seed)
        self.dummy_param = nn.Parameter(torch.zeros(1)) # Device tracking

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim

    def encode(self, protein_input: ProteinInput) -> EncoderOutput:
        B, L = protein_input.coords.shape[:2]
        device = self.dummy_param.device

        # Random with reproducible generator if seeded
        h = torch.randn(B, L, self.hidden_dim, device=device, generator=self._generator)
        g = torch.randn(B, self.hidden_dim, device=device, generator=self._generator)

        return EncoderOutput(residue_embeddings=h, global_embedding=g)

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass

    def get_state_dict(self) -> dict:
        """Return state dict for serialization and reproducibility."""
        return self.state_dict()

    def set_state_dict(self, state_dict: dict) -> None:
        """Load a serialized encoder state."""
        self.load_state_dict(state_dict)

class LengthEncoder(BaseProteinEncoder):
    """
    Length-based Embeddings (Sanity Check).
    """
    bottleneck_class = BottleneckClass.C

    def __init__(self, hidden_dim: int = 128, seed: int = None, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.proj = nn.Linear(1, hidden_dim) # Project length to dim
        if seed is not None:
            self._seed_everything(seed)

    def _seed_everything(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim

    def encode(self, protein_input: ProteinInput) -> EncoderOutput:
        B, L = protein_input.coords.shape[:2]
        mask = protein_input.mask
        device = mask.device

        lengths = mask.sum(dim=1, keepdim=True).float() # (B, 1)
        g = self.proj(lengths) # (B, D)

        # Broadcast to residues
        h = g.unsqueeze(1).expand(B, L, -1)

        return EncoderOutput(residue_embeddings=h, global_embedding=g)

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass

    def get_state_dict(self) -> dict:
        """Return state dict for serialization and reproducibility."""
        return self.state_dict()

    def set_state_dict(self, state_dict: dict) -> None:
        """Load a serialized encoder state."""
        self.load_state_dict(state_dict)

register_encoder("random")(RandomEncoder)
register_encoder("length")(LengthEncoder)
