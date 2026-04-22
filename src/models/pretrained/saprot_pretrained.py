"""
SaProt pretrained encoder wrapper.

Loads the official SaProt-650M-AF2 model from HuggingFace and uses
structure-aware (AA + 3Di) tokens for encoding.

SaProt uses ESM-2 architecture with a combined AA+3Di vocabulary where
each residue is represented as a bigram token (e.g., 'Md' = Met + 3Di state 'd').

Reference:
    Su et al. (2024). "SaProt: Protein Language Modeling with Structure-aware Vocabulary"
"""

import torch
import torch.nn as nn

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)
from src.models.class_a.saprot_encoder import get_3di_tokens


AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"
DI_CHARS = "acdefghiklmnpqrstvwy"  # 3Di alphabet (lowercase)


@register_encoder("saprot_pretrained")
class SaProtPretrainedEncoder(BaseProteinEncoder):
    """
    SaProt pretrained encoder using HuggingFace weights.

    Takes protein coordinates + sequence, computes 3Di tokens from coords,
    creates SA bigram tokens, and encodes via pretrained ESM-2 backbone.
    """

    def __init__(self, model_name: str = "westlake-repl/SaProt_650M_AF2"):
        super().__init__()
        self.model_name = model_name
        self._embedding_dim = 1280
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
        if self._initialized:
            return

        from transformers import EsmTokenizer, EsmModel

        self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
        self.model = EsmModel.from_pretrained(self.model_name)
        self._embedding_dim = self.model.config.hidden_size
        self._initialized = True

    def load_pretrained(self, checkpoint_path: str):
        self._lazy_init()

    def _make_sa_strings(self, sequence, coords, mask):
        """Create SaProt bigram strings from sequence + coordinates."""
        tokens_3di = get_3di_tokens(coords, mask)  # (B, L) int tensor 0-19
        B, L = sequence.shape
        sa_strings = []
        for b in range(B):
            seq_len = mask[b].sum().item()
            parts = []
            for i in range(seq_len):
                aa_idx = sequence[b, i].item()
                di_idx = tokens_3di[b, i].item()
                aa_char = AA_CHARS[aa_idx] if aa_idx < 20 else "X"
                di_char = DI_CHARS[di_idx] if di_idx < 20 else "d"  # default 'd'
                parts.append(aa_char + di_char)
            sa_strings.append("".join(parts))
        return sa_strings

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        self._lazy_init()

        device = protein.coords.device
        self.model = self.model.to(device)
        self.model.eval()

        B, L = protein.sequence.shape

        # Create SA bigram strings
        sa_strings = self._make_sa_strings(
            protein.sequence, protein.coords, protein.mask
        )

        # Tokenize
        encoding = self.tokenizer(
            sa_strings,
            padding=True,
            truncation=True,
            max_length=L + 2,  # +2 for BOS/EOS
            return_tensors="pt",
        ).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )

        hidden_states = outputs.last_hidden_state  # (B, seq_len+2, D)

        # Extract residue embeddings (skip BOS token at position 0)
        residue_embeddings = torch.zeros(B, L, self._embedding_dim, device=device)
        for b in range(B):
            seq_len = protein.mask[b].sum().item()
            # Tokens: [BOS, tok1, tok2, ..., tokN, EOS]
            # Skip BOS (idx 0), take seq_len tokens
            available = min(seq_len, hidden_states.shape[1] - 2)
            residue_embeddings[b, :available] = hidden_states[b, 1 : 1 + available]

        return EncoderOutput(
            residue_embeddings=residue_embeddings,
            global_embedding=residue_embeddings.mean(dim=1),
            metadata={"model_name": self.model_name},
        )
