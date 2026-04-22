"""
ESM3 (EvolutionaryScale) encoder wrapper for InvariantsBench.

ESM3 small open (1.4B parameters, 1536D) — trimodal foundation model:
  sequence + structure + function.

This wrapper uses sequence + backbone structure simultaneously, which is
ESM3's key advantage over ESM-2 (sequence-only) and ESM-IF (structure-only).

Weights loaded from local path (bypasses HuggingFace gated-repo download).
Local path: esm3-sm-open-v1/data/weights/esm3_sm_open_v1.pth

Reference:
    Hayes et al. "Simulating 500 million years of evolution with a language model"
    Science 2025. https://github.com/evolutionaryScale/esm
"""

from pathlib import Path
from typing import Optional

import torch

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
    AA_VOCAB_INV,
)

# Atom37 indices for our 5 backbone atoms: N=0, CA=1, C=2, O=3, CB=4
_BACKBONE_ATOM37_IDX = [0, 1, 2, 3, 4]
_ESM3_LOCAL_DIR = Path(__file__).parents[3] / "esm3-sm-open-v1"


def _patch_data_root(local_dir: Path) -> None:
    """
    Permanently patch esm.utils.constants.esm3.data_root to return
    local_dir instead of calling snapshot_download.

    The ESM3 library accesses data files via `C.data_root(model)` where
    C is the constants module object, so patching the module attribute is
    sufficient — all callsites will see the patched version.
    """
    import esm.utils.constants.esm3 as C

    def _local_data_root(model: str) -> Path:
        return local_dir

    C.data_root = _local_data_root

    # Also patch any submodule that imported data_root by name
    try:
        import esm.tokenization.residue_tokenizer as rt
        rt.C = C  # residue_tokenizer accesses data_root via C
    except ImportError:
        pass


def _build_esm3_local(local_dir: Path, device: torch.device):
    """
    Instantiate ESM3-sm-open and load weights from a local directory.
    Patches data_root globally before any ESM module calls it.
    """
    weights_path = local_dir / "data" / "weights" / "esm3_sm_open_v1.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"ESM3 weights not found at {weights_path}. "
            "Expected: esm3-sm-open-v1/data/weights/esm3_sm_open_v1.pth"
        )

    # Patch BEFORE importing anything that calls data_root
    _patch_data_root(local_dir)

    from esm.models.esm3 import ESM3
    from esm.pretrained import (
        ESM3_structure_encoder_v0,
        ESM3_structure_decoder_v0,
        ESM3_function_decoder_v0,
    )
    from esm.tokenization import get_esm3_model_tokenizers as get_model_tokenizers

    with torch.device(device):
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_model_tokenizers("esm3_sm_open_v1"),
        ).eval()

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    if device.type != "cpu":
        model = model.to(torch.bfloat16)

    return model, get_model_tokenizers("esm3_sm_open_v1")


@register_encoder("esm3")
class ESM3Encoder(BaseProteinEncoder):
    """
    ESM3-sm-open (1.4B params, 1536D) — sequence + structure trimodal encoder.

    Uses both amino acid sequence tokens AND backbone coordinates as input,
    which is the key distinction from ESM-2 (sequence-only) and ESM-IF
    (structure-only). Embeddings are extracted via the trunk's final hidden
    state (LogitsConfig(return_embeddings=True)).

    Args:
        local_dir: Path to the downloaded esm3-sm-open-v1 directory.
                   Defaults to <project_root>/esm3-sm-open-v1/
        use_structure: If True (default), pass backbone coords alongside
                       sequence tokens. If False, sequence-only mode.
    """

    def __init__(
        self,
        local_dir: Optional[str] = None,
        use_structure: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._local_dir = Path(local_dir) if local_dir else _ESM3_LOCAL_DIR
        self._use_structure = use_structure
        self._embedding_dim = 1536

        self._model = None
        self._tokenizers = None
        self._initialized = False

    @property
    def bottleneck_class(self) -> BottleneckClass:
        # Trimodal: discrete token sequence + equivariant structure path
        return BottleneckClass.A

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _lazy_init(self, device: torch.device):
        if self._initialized:
            return
        print(f"Loading ESM3-sm-open from {self._local_dir} ...")
        self._model, self._tokenizers = _build_esm3_local(self._local_dir, device)
        self._model.eval()
        self._initialized = True

    def load_pretrained(self, checkpoint_path: str) -> None:
        """No-op: weights loaded automatically from local_dir in __init__."""
        pass

    def _coords_to_atom37(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert (B, L, 5, 3) backbone coords [N, CA, C, O, CB]
        to (B, L, 37, 3) atom37 format expected by ESM3.
        Unresolved atoms are set to NaN; padding positions are NaN.
        """
        B, L = coords.shape[:2]
        atom37 = torch.full(
            (B, L, 37, 3), float("nan"),
            dtype=coords.dtype, device=coords.device
        )
        for i, idx in enumerate(_BACKBONE_ATOM37_IDX):
            atom37[:, :, idx, :] = coords[:, :, i, :]
        # Zero-out padding positions entirely
        pad = ~mask  # (B, L)
        atom37[pad] = float("nan")
        return atom37

    def _tokenize_sequences(
        self, sequences: torch.Tensor, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Convert (B, L) amino acid index tensors to ESM3 sequence token tensors.
        Returns (B, L+2) token ids (with BOS/EOS prepended/appended).
        """
        seq_tok = self._tokenizers.sequence
        batch_tokens = []
        for b in range(sequences.shape[0]):
            seq_len = int(mask[b].sum().item())
            aa_str = "".join(
                AA_VOCAB_INV.get(int(sequences[b, i].item()), "X")
                for i in range(seq_len)
            )
            ids = seq_tok.encode(aa_str)  # includes BOS/EOS
            batch_tokens.append(torch.tensor(ids, dtype=torch.long))

        # Pad to same length
        max_len = max(t.shape[0] for t in batch_tokens)
        pad_id = seq_tok.pad_token_id if hasattr(seq_tok, 'pad_token_id') and seq_tok.pad_token_id is not None else 1
        padded = torch.full((len(batch_tokens), max_len), pad_id, dtype=torch.long, device=device)
        for b, t in enumerate(batch_tokens):
            padded[b, : t.shape[0]] = t.to(device)
        return padded  # (B, L+2)

    @torch.no_grad()
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        self._lazy_init(device)

        # Move model to device if needed
        if next(self._model.parameters()).device != device:
            self._model = self._model.to(device)

        from esm.sdk.api import ESMProteinTensor, LogitsConfig

        B, L = protein.sequence.shape

        # --- Sequence tokens (B, L+2) ---
        seq_tokens = self._tokenize_sequences(protein.sequence, protein.mask, device)

        # --- Structure coordinates (B, L, 37, 3) or None ---
        structure_coords = None
        if self._use_structure:
            structure_coords = self._coords_to_atom37(protein.coords, protein.mask)
            # ESM3 expects float32 coords (bfloat16 causes NaN issues in structure encoder)
            structure_coords = structure_coords.float()

        # --- Run per-sample to avoid ESM3 batching quirks with variable-length ---
        # ESM3 logits() accepts ESMProteinTensor (single sample) only.
        # We loop over the batch, which is acceptable (batch_size ≤ 32 during probing).
        all_embeddings = []
        cfg = LogitsConfig(return_embeddings=True)

        for b in range(B):
            seq_len = int(protein.mask[b].sum().item())
            # ESMProteinTensor expects 1D token tensor (no batch dim).
            # logits() adds the batch dim internally.
            tok_len = seq_len + 2
            s_tokens = seq_tokens[b, :tok_len]  # (tok_len,)

            if self._use_structure:
                # Sequence tokens are (1, tok_len) where tok_len = seq_len+2 (BOS+EOS).
                # Structure coords must match tok_len: prepend/append NaN rows for BOS/EOS.
                res_coords = structure_coords[b, :seq_len]  # (seq_len, 37, 3)
                nan_row = torch.full((1, 37, 3), float("nan"),
                                     dtype=res_coords.dtype, device=res_coords.device)
                s_coords = torch.cat([nan_row, res_coords, nan_row], dim=0)  # (tok_len, 37, 3)
            else:
                s_coords = None

            # Build ESMProteinTensor for this sample (single-protein API, no batch dim)
            pt = ESMProteinTensor(
                sequence=s_tokens,
                coordinates=s_coords,
            )

            logits_out = self._model.logits(pt, cfg)
            # embeddings: (1, tok_len, 1536)
            emb = logits_out.embeddings  # (1, tok_len, D)

            # Strip BOS (index 0) and EOS (index -1), keeping only residue positions
            residue_emb = emb[0, 1 : 1 + seq_len, :]  # (seq_len, D)

            # Cast back to float32 for probe training
            residue_emb = residue_emb.float()

            # Pad to L
            full_emb = torch.zeros(L, self._embedding_dim, device=device)
            full_emb[:seq_len] = residue_emb
            all_embeddings.append(full_emb)

        residue_embeddings = torch.stack(all_embeddings, dim=0)  # (B, L, D)

        # Global: mean-pool over valid residues
        mask_f = protein.mask.unsqueeze(-1).float()
        global_emb = (residue_embeddings * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        return EncoderOutput(
            residue_embeddings=residue_embeddings,
            global_embedding=global_emb,
            metadata={"use_structure": self._use_structure},
        )
