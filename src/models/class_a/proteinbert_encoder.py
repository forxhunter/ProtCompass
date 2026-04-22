"""
ProteinBERT-BFD encoder wrapper.

ProteinBERT (prot_bert_bfd) is a BERT-based protein language model pretrained
on BFD (Big Fantastic Database, ~2.1 billion sequences).

Reference:
    Elnaggar et al. (2021). "ProtTrans: Towards Cracking the Language of Life's Code
    Through Self-Supervised Deep Learning and High Performance Computing."
    IEEE TPAMI. doi:10.1109/TPAMI.2021.3095381

HuggingFace: Rostlab/prot_bert_bfd
Architecture: BERT-Large, 420M parameters, 1024D hidden, 16 heads, 30 layers
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
    tensor_to_sequence,
)


# Standard amino acid vocabulary (single-letter, same as InvariantsBench)
_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"


def _seq_to_protbert_str(seq_tensor: torch.Tensor) -> str:
    """Convert AA index tensor to space-separated string for ProtBERT tokenizer.

    ProtBERT tokenizer expects sequences where each AA is separated by a space,
    e.g., "M A K V L ...". Unknown/pad indices are mapped to 'X'.
    """
    chars = []
    for idx in seq_tensor.tolist():
        if 0 <= idx < len(_AA_VOCAB):
            chars.append(_AA_VOCAB[idx])
        else:
            chars.append("X")
    return " ".join(chars)


@register_encoder("proteinbert_bfd")
class ProteinBertBFDEncoder(BaseProteinEncoder):
    """
    ProteinBERT-BFD encoder (Rostlab/prot_bert_bfd).

    BERT-Large architecture pretrained on BFD (~2.1B sequences).
    420M parameters, 1024D per-residue embeddings.

    Args:
        model_name: HuggingFace model ID (default: Rostlab/prot_bert_bfd)
        freeze: Whether to freeze backbone weights (default: True)
        use_half: Use FP16 for reduced memory (default: False)
    """

    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert_bfd",
        freeze: bool = True,
        use_half: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self._freeze_backbone = freeze
        self.use_half = use_half
        self._embedding_dim = 1024  # BERT-Large hidden size

        # Lazy initialization — avoids 420M param load at import time
        self.model = None
        self.tokenizer = None
        self._initialized = False

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.A  # Sequence-based PLM

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _lazy_init(self):
        if self._initialized:
            return
        try:
            from transformers import BertModel, BertTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for ProteinBERT-BFD. "
                "Install with: pip install transformers sentencepiece"
            )
        print(f"Loading ProteinBERT-BFD: {self.model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name,
            do_lower_case=False,
        )
        self.model = BertModel.from_pretrained(self.model_name)
        if self.use_half:
            self.model = self.model.half()
        if self._freeze_backbone:
            self.freeze()
        self._initialized = True

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        self._lazy_init()
        device = protein.coords.device

        if not next(self.model.parameters()).device == device:
            self.model = self.model.to(device)

        B, L = protein.sequence.shape

        # Build space-separated sequences for ProtBERT tokenizer
        seqs = [_seq_to_protbert_str(protein.sequence[i]) for i in range(B)]

        # Tokenize (adds [CLS] and [SEP], pads to longest in batch)
        inputs = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(L + 2, 1024),  # +2 for CLS/SEP
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.set_grad_enabled(not self._is_frozen):
            if self.use_half:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        # Strip [CLS] token (index 0) and [SEP] token (last real token)
        # Token layout: [CLS] AA_0 AA_1 ... AA_{L-1} [SEP] [PAD] ...
        hidden = outputs.last_hidden_state  # (B, L_tok, 1024)

        # Extract only the AA token positions (skip CLS at 0, stop before SEP)
        # The number of AA tokens per sequence is min(len(seq), max_length-2)
        aa_embeddings = hidden[:, 1:-1, :]  # (B, L_tok-2, 1024)

        # Align to protein.coords length L via padding/truncation
        T = aa_embeddings.shape[1]
        if T < L:
            pad = torch.zeros(B, L - T, self._embedding_dim, device=device,
                              dtype=aa_embeddings.dtype)
            aa_embeddings = torch.cat([aa_embeddings, pad], dim=1)
        elif T > L:
            aa_embeddings = aa_embeddings[:, :L, :]

        # Apply sequence mask
        mask_3d = protein.mask.unsqueeze(-1).float()  # (B, L, 1)
        aa_embeddings = aa_embeddings * mask_3d

        # Global embedding: masked mean-pooling
        global_emb = (aa_embeddings * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp(min=1)

        return EncoderOutput(
            residue_embeddings=aa_embeddings,
            global_embedding=global_emb,
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        """No-op: weights are loaded from HuggingFace at init time."""
        pass
