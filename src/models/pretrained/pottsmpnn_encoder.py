"""
PottsMPNN encoder wrapper.

PottsMPNN extends ProteinMPNN with a Potts model energy term. The encoder
is a graph-based message-passing network over protein backbone coordinates
(N, CA, C, O atoms) producing per-residue embeddings.

Reference:
    PottsMPNN: https://github.com/PrettJacob/PottsMPNN

Available checkpoints in external_repos/PottsMPNN/:
    proteinmpnn_compatible_model_weights/
        pottsmpnn_20.pt        — standard, N+CA+C+O, 3 encoder layers
        pottsmpnn_30.pt        — standard, 30 encoder layers
        pottsmpnn_msa_20.pt    — MSA-enhanced, 20 layers
        sol_pottsmpnn_*.pt     — soluble-protein optimized variants
    ca_model_weights/
        v_48_002.pt            — CA-only, 48 layers
        v_48_010.pt
        v_48_020.pt
    vanilla_model_weights/
        pottsmpnn_20.pt        — standard, 20 layers
        pottsmpnn_30.pt
        pottsmpnn_msa_20.pt
    ft_model_weights/
        potts_ft.pt            — fine-tuned
    soluble_model_weights/
        sol_pottsmpnn_20.pt    — soluble-protein, 20 layers
        sol_pottsmpnn_30.pt
        sol_pottsmpnn_msa_20.pt

Architecture: Class B (Graph)
    - Processes backbone coordinates as a kNN graph
    - Encoder output h_V: (B, L, 128) per-residue embeddings
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)

_POTTSMPNN_ROOT = Path(__file__).parents[3] / "external_repos" / "PottsMPNN"


def _infer_ca_only(checkpoint_path: Path) -> bool:
    """Infer whether a checkpoint is CA-only from its path."""
    path_str = str(checkpoint_path)
    return "ca_model_weights" in path_str or path_str.endswith(("_002.pt", "_010.pt", "_020.pt"))


def _get_model(checkpoint_path: Path, device: torch.device, ca_only: Optional[bool] = None):
    """Load PottsMPNN and return encoder-ready model."""
    if str(_POTTSMPNN_ROOT) not in sys.path:
        sys.path.insert(0, str(_POTTSMPNN_ROOT))
    from potts_mpnn_utils import PottsMPNN, ProteinFeatures

    if ca_only is None:
        ca_only = _infer_ca_only(checkpoint_path)

    # CA-only models use same hidden_dim=128 but different k_neighbors
    if ca_only:
        hidden_dim = 128
        node_features = 128
        edge_features = 128
        k_neighbors = 30
        num_encoder_layers = 3
    else:
        hidden_dim = 128
        node_features = 128
        edge_features = 128
        k_neighbors = 64
        num_encoder_layers = 3

    model = PottsMPNN(
        num_letters=21,
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        potts_dim=400,          # only used by decoder/etab — not loaded, not used
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=k_neighbors,
        augment_eps=0.0,        # no augmentation at inference
        dropout=0.0,
        ca_only=ca_only,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # strict=False: etab_out not in compatible weights — that's fine, encoder unused
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        encoder_missing = [k for k in missing if "encoder" in k or "features" in k or "W_e" in k]
        if encoder_missing:
            raise RuntimeError(f"Missing encoder keys: {encoder_missing}")
    return model.to(device).eval(), hidden_dim


@register_encoder("pottsmpnn_pretrained")
class PottsMPNNPretrainedEncoder(BaseProteinEncoder):
    """
    PottsMPNN encoder with pretrained weights.

    Per-residue embeddings from the graph encoder stack (h_V, 128D or 48D).
    Supports all PottsMPNN checkpoint variants including:
    - proteinmpnn_compatible (N+CA+C+O)
    - ca_model_weights (CA-only, hidden_dim=48)
    - soluble_model_weights (soluble-protein optimized)
    - ft_model_weights (fine-tuned)
    """

    def __init__(self, checkpoint_path: Optional[str] = None,
                 ca_only: Optional[bool] = None, **kwargs):
        super().__init__()
        self._ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        self._ca_only = ca_only
        self._model = None
        self._embedding_dim = 128

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _default_ckpt(self):
        return _POTTSMPNN_ROOT / "proteinmpnn_compatible_model_weights" / "pottsmpnn_20.pt"

    def _ensure_model(self, device: torch.device):
        if self._model is None:
            ckpt_path = self._ckpt_path or self._default_ckpt()
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"PottsMPNN checkpoint not found: {ckpt_path}"
                )
            ca_only = self._ca_only if self._ca_only is not None else _infer_ca_only(ckpt_path)
            self._model, self._embedding_dim = _get_model(ckpt_path, device, ca_only)
        if next(self._model.parameters()).device != device:
            self._model = self._model.to(device)

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        self._ensure_model(device)

        B, L, n_atoms, _ = protein.coords.shape   # (B, L, 5, 3) — N,CA,C,O,CB
        mask = protein.mask.float()                # (B, L)

        ca_only = self._model.ca_only if hasattr(self._model, 'ca_only') else self._ca_only

        if ca_only:
            # CA-only model: use CA atoms only, shape (B, L, 3)
            X = protein.coords[:, :, 1, :].float()   # (B, L, 3)
        else:
            # Standard model: use N, CA, C, O, shape (B, L, 4, 3)
            X = protein.coords[:, :, :4, :].float()   # (B, L, 4, 3)

        residue_idx     = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).float()
        chain_encoding  = torch.ones(B, L, device=device).float()

        if str(_POTTSMPNN_ROOT) not in sys.path:
            sys.path.insert(0, str(_POTTSMPNN_ROOT))
        from potts_mpnn_utils import gather_nodes

        with torch.no_grad():
            E, E_idx = self._model.features(X, mask, residue_idx, chain_encoding)
            h_V = torch.zeros(B, L, E.shape[-1], device=device)
            h_E = self._model.W_e(E)

            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend

            for layer in self._model.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        mask_f = mask.unsqueeze(-1)
        global_emb = (h_V * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)

        return EncoderOutput(
            residue_embeddings=h_V.float(),
            global_embedding=global_emb.float(),
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        self._ckpt_path = Path(checkpoint_path)
        self._model = None


# ── Additional PottsMPNN pretrained variants ──────────────────────────────────

@register_encoder("pottsmpnn_ca_pretrained")
class PottsMPNNCAPretrainedEncoder(PottsMPNNPretrainedEncoder):
    """PottsMPNN CA-only variant (v_48 checkpoint family)."""
    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        default = _POTTSMPNN_ROOT / "ca_model_weights" / "v_48_002.pt"
        super().__init__(checkpoint_path=checkpoint_path, ca_only=True)
        if self._ckpt_path is None:
            self._ckpt_path = default


@register_encoder("pottsmpnn_soluble_pretrained")
class PottsMPPNSolublePretrainedEncoder(PottsMPNNPretrainedEncoder):
    """PottsMPNN soluble-protein optimized variant."""
    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        default = _POTTSMPNN_ROOT / "soluble_model_weights" / "sol_pottsmpnn_20.pt"
        super().__init__(checkpoint_path=checkpoint_path, ca_only=False)
        if self._ckpt_path is None:
            self._ckpt_path = default


@register_encoder("pottsmpnn_ft_pretrained")
class PottsMPNNFTPretrainedEncoder(PottsMPNNPretrainedEncoder):
    """PottsMPNN fine-tuned variant."""
    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        default = _POTTSMPNN_ROOT / "ft_model_weights" / "potts_ft.pt"
        super().__init__(checkpoint_path=checkpoint_path, ca_only=False)
        if self._ckpt_path is None:
            self._ckpt_path = default


# ── Random-weight baselines ───────────────────────────────────────────────────

@register_encoder("pottsmpnn")
class PottsMPNNRandomEncoder(BaseProteinEncoder):
    """
    PottsMPNN architecture with random weights (no pretraining).
    Architectural baseline measuring geometric inductive bias only.
    """

    def __init__(self, ca_only: bool = False, **kwargs):
        super().__init__()
        self._ca_only = ca_only
        self._model = None
        self._embedding_dim = 128

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _ensure_model(self, device: torch.device):
        if self._model is None:
            if str(_POTTSMPNN_ROOT) not in sys.path:
                sys.path.insert(0, str(_POTTSMPNN_ROOT))
            from potts_mpnn_utils import PottsMPNN
            self._model = PottsMPNN(
                num_letters=21, node_features=128, edge_features=128,
                hidden_dim=128, potts_dim=400,
                num_encoder_layers=3, num_decoder_layers=3,
                vocab=21, k_neighbors=64, augment_eps=0.0, dropout=0.0,
                ca_only=False,
            ).to(device).eval()
        if next(self._model.parameters()).device != device:
            self._model = self._model.to(device)

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        self._ensure_model(device)

        B, L, n_atoms, _ = protein.coords.shape
        mask = protein.mask.float()
        X = protein.coords[:, :, :4, :].float()   # N, CA, C, O

        residue_idx    = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).float()
        chain_encoding = torch.ones(B, L, device=device).float()

        if str(_POTTSMPNN_ROOT) not in sys.path:
            sys.path.insert(0, str(_POTTSMPNN_ROOT))
        from potts_mpnn_utils import gather_nodes

        with torch.no_grad():
            E, E_idx = self._model.features(X, mask, residue_idx, chain_encoding)
            h_V = torch.zeros(B, L, E.shape[-1], device=device)
            h_E = self._model.W_e(E)

            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend

            for layer in self._model.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        mask_f = mask.unsqueeze(-1)
        global_emb = (h_V * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)

        return EncoderOutput(
            residue_embeddings=h_V.float(),
            global_embedding=global_emb.float(),
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass
