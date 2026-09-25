"""
ProstT5 encoder wrapper.

ProstT5 is a T5-based protein language model that can process both
amino acid sequences and structural tokens.

Reference:
    Heinzinger et al. (2023). "ProstT5: Bilingual Language Model for Protein Sequence and Structure"
"""

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

# Import 3Di tokenization from saprot_encoder
from src.models.class_a.saprot_encoder import get_3di_tokens


@register_encoder("prostt5")
class ProstT5Encoder(BaseProteinEncoder):
    """
    ProstT5 encoder for protein structure embeddings.

    Uses T5 architecture with structure-aware 3Di tokens.
    Supports loading pretrained weights from HuggingFace.

    Args:
        model_name: HuggingFace model name (default: Rostlab/ProstT5)
        use_3di: Whether to use 3Di structural tokens
        use_half: Use FP16 for reduced memory
    """

    def __init__(
        self,
        model_name: str = "Rostlab/ProstT5",
        use_3di: bool = True,
        use_half: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_3di = use_3di
        self.use_half = use_half
        self._embedding_dim = 1024  # T5-XL hidden size

        # Lazy initialization
        self.model = None
        self.tokenizer = None
        self._initialized = False

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.A

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _lazy_init(self):
        """Lazily initialize model on first use."""
        if self._initialized:
            return

        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for ProstT5. "
                "Install with: pip install transformers sentencepiece"
            )

        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name, do_lower_case=False
        )
        self.model = T5EncoderModel.from_pretrained(self.model_name)

        if self.use_half:
            self.model = self.model.half()

        # Update embedding dim from actual model
        self._embedding_dim = self.model.config.d_model

        self._initialized = True

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        self._lazy_init()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)

    def _sequence_to_str(self, sequence: torch.Tensor, mask: torch.Tensor) -> list:
        """Convert sequence tensor to list of AA strings."""
        AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"
        batch_seqs = []
        for b in range(sequence.shape[0]):
            seq_len = mask[b].sum().item()
            seq_indices = sequence[b, :seq_len].tolist()
            seq_str = " ".join(AA_CHARS[i] if i < 20 else "X" for i in seq_indices)
            batch_seqs.append(seq_str)
        return batch_seqs

    def _3di_to_str(self, tokens_3di: torch.Tensor, mask: torch.Tensor) -> list:
        """Convert 3Di tokens to list of strings."""
        DI_CHARS = "ACDEFGHIKLMNPQRSTVWY"  # 3Di alphabet
        batch_seqs = []
        for b in range(tokens_3di.shape[0]):
            seq_len = mask[b].sum().item()
            di_indices = tokens_3di[b, :seq_len].tolist()
            di_str = " ".join(DI_CHARS[i] if i < 20 else "X" for i in di_indices)
            batch_seqs.append(di_str)
        return batch_seqs

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        """
        Encode protein structure to embeddings.

        Args:
            protein: ProteinInput with coords, sequence, mask

        Returns:
            EncoderOutput with residue embeddings
        """
        self._lazy_init()

        device = protein.coords.device
        self.model = self.model.to(device)
        self.model.eval()

        B, L = protein.sequence.shape

        # Convert sequence to strings
        if self.use_3di:
            # Get 3Di tokens and create combined input
            tokens_3di = get_3di_tokens(protein.coords, protein.mask)
            aa_seqs = self._sequence_to_str(protein.sequence, protein.mask)
            di_seqs = self._3di_to_str(tokens_3di, protein.mask)
            # ProstT5 format: "<AA> <3Di>" separated
            input_seqs = [f"{aa} <sep> {di}" for aa, di in zip(aa_seqs, di_seqs)]
        else:
            input_seqs = self._sequence_to_str(protein.sequence, protein.mask)

        # Tokenize
        encoding = self.tokenizer(
            input_seqs,
            padding=True,
            truncation=True,
            max_length=L * 2 + 10,  # Account for 3Di tokens and sep
            return_tensors="pt",
        ).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )

        # Get hidden states
        hidden_states = outputs.last_hidden_state  # (B, seq_len, D)

        # Extract only the AA positions (first L positions, skip 3Di part)
        # For simplicity, we take the mean over all tokens per residue
        # In production, you'd align tokens to residues properly
        residue_embeddings = torch.zeros(B, L, self._embedding_dim, device=device)

        for b in range(B):
            seq_len = protein.mask[b].sum().item()
            # Take first seq_len hidden states (corresponding to AA tokens)
            available = min(seq_len, hidden_states.shape[1])
            residue_embeddings[b, :available] = hidden_states[b, :available]

        mask_f = protein.mask.unsqueeze(-1).float()
        global_embedding = (
            (residue_embeddings * mask_f).sum(dim=1)
            / mask_f.sum(dim=1).clamp(min=1)
        )

        return EncoderOutput(
            residue_embeddings=residue_embeddings,
            global_embedding=global_embedding,
            metadata={"model_name": self.model_name, "use_3di": self.use_3di},
        )
