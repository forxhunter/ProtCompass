"""
Random-weight versions of pretrained protein language models.

Each encoder uses the exact same architecture as its pretrained counterpart
but with randomly initialized weights (no checkpoint loaded). This provides
a clean architectural baseline to measure the contribution of pretraining.

Usage: same as pretrained versions; just use the _random-suffixed name.
"""

import torch

from src.models.base_encoder import register_encoder
from src.models.external import (
    ESM2Adapter,
    ESM1bAdapter,
    ESM2_150MAdapter,
    ESM2_35MAdapter,
    ProtTransAdapter,
    AnkhAdapter,
)
from src.models.class_a.proteinbert_encoder import ProteinBertBFDEncoder
from src.models.class_a.prostt5_encoder import ProstT5Encoder
from src.models.pretrained.esm_if_encoder import ESMIFEncoder
from src.models.class_a.saprot_encoder import SaProtEncoder


# ── ESM-2 family (EsmModel) ────────────────────────────────────────────────────

@register_encoder("esm2_random")
class ESM2RandomEncoder(ESM2Adapter):
    """ESM-2 650M architecture with random weights (no pretraining)."""

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D",
                 freeze: bool = True, **kwargs):
        # Bypass parent __init__ entirely
        torch.nn.Module.__init__(self)
        self.model_name = model_name
        self._freeze_backbone = freeze
        self.use_half = False
        self._cpu_offload = False

        from transformers import EsmConfig, EsmModel, EsmTokenizer
        print(f"[random] ESM-2 architecture (random weights): {model_name}")
        config = EsmConfig.from_pretrained(model_name)
        self.model = EsmModel(config)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        if freeze:
            self.freeze()


@register_encoder("esm2_150m_random")
class ESM2_150MRandomEncoder(ESM2RandomEncoder):
    """ESM-2 150M architecture with random weights."""

    def __init__(self, **kwargs):
        super().__init__(model_name="facebook/esm2_t30_150M_UR50D", **kwargs)


@register_encoder("esm2_35m_random")
class ESM2_35MRandomEncoder(ESM2RandomEncoder):
    """ESM-2 35M architecture with random weights."""

    def __init__(self, **kwargs):
        super().__init__(model_name="facebook/esm2_t12_35M_UR50D", **kwargs)


@register_encoder("esm1b_random")
class ESM1bRandomEncoder(ESM2RandomEncoder):
    """ESM-1b 650M architecture with random weights."""

    def __init__(self, **kwargs):
        super().__init__(model_name="facebook/esm1b_t33_650M_UR50S", **kwargs)


# ── ProtBERT-BFD (BertModel) ───────────────────────────────────────────────────

@register_encoder("proteinbert_bfd_random")
class ProteinBertBFDRandomEncoder(ProteinBertBFDEncoder):
    """ProtBERT-BFD architecture with random weights (no pretraining)."""

    def __init__(self, model_name: str = "Rostlab/prot_bert_bfd",
                 freeze: bool = True, **kwargs):
        # Bypass parent __init__; replicate structure with random weights
        torch.nn.Module.__init__(self)
        self.model_name = model_name
        self._freeze_backbone = freeze
        self.use_half = False
        self._embedding_dim = 1024
        self._initialized = True  # skip _lazy_init path

        from transformers import BertConfig, BertModel, BertTokenizer
        print(f"[random] ProtBERT-BFD architecture (random weights): {model_name}")
        config = BertConfig.from_pretrained(model_name)
        self.model = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        if freeze:
            self.freeze()


# ── ProtTrans T5 (T5EncoderModel) ─────────────────────────────────────────────

@register_encoder("prottrans_random")
class ProtTransRandomEncoder(ProtTransAdapter):
    """ProtTrans T5-XL architecture with random weights (no pretraining)."""

    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
                 freeze: bool = True, **kwargs):
        torch.nn.Module.__init__(self)
        self.model_name = model_name
        self._freeze_backbone = freeze

        from transformers import T5Config, T5EncoderModel, T5Tokenizer
        print(f"[random] ProtTrans architecture (random weights): {model_name}")
        config = T5Config.from_pretrained(model_name)
        self.model = T5EncoderModel(config)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        if freeze:
            self.freeze()


# ── Ankh (T5EncoderModel) ─────────────────────────────────────────────────────

@register_encoder("ankh_random")
class AnkhRandomEncoder(AnkhAdapter):
    """Ankh architecture with random weights (no pretraining)."""

    def __init__(self, model_name: str = "ElnaggarLab/ankh-base",
                 freeze: bool = True, **kwargs):
        torch.nn.Module.__init__(self)
        self.model_name = model_name
        self._freeze_backbone = freeze

        from transformers import T5Config, T5EncoderModel, AutoTokenizer
        print(f"[random] Ankh architecture (random weights): {model_name}")
        config = T5Config.from_pretrained(model_name)
        self.model = T5EncoderModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if freeze:
            self.freeze()


# ── ProstT5 (T5EncoderModel) ──────────────────────────────────────────────────

@register_encoder("prostt5_random")
class ProstT5RandomEncoder(ProstT5Encoder):
    """ProstT5 architecture with random weights (no pretraining)."""

    def __init__(self, model_name: str = "Rostlab/ProstT5",
                 freeze: bool = True, **kwargs):
        torch.nn.Module.__init__(self)
        self.model_name = model_name
        self._freeze_backbone = freeze
        self.use_3di = False

        from transformers import T5Config, T5EncoderModel, T5Tokenizer
        print(f"[random] ProstT5 architecture (random weights): {model_name}")
        config = T5Config.from_pretrained(model_name)
        self.model = T5EncoderModel(config)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        if freeze:
            self.freeze()


# ── ESM-IF (GVP-Transformer, structure-based) ─────────────────────────────────

@register_encoder("esm_if_random")
class ESMIFRandomEncoder(ESMIFEncoder):
    """ESM-IF1 GVP-Transformer architecture with random weights (no pretraining)."""

    def __init__(self, **kwargs):
        # Bypass parent __init__ to avoid loading pretrained weights
        torch.nn.Module.__init__(self)

        from src.models.pretrained.esm_if_encoder import _patch_esm_if_deps
        _patch_esm_if_deps()

        import esm
        print("[random] ESM-IF1 architecture (random weights)")

        # Load model architecture but reinitialize weights
        self._model, self._alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        # Reinitialize all parameters with random weights
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self._model.apply(init_weights)
        self._model.eval()
        self._encoder = self._model.encoder
        self._embed_dim = self._model.args.encoder_embed_dim  # 512


# ── SaProt (ESM-2 + 3Di tokens) ───────────────────────────────────────────────

@register_encoder("saprot_random")
class SaProtRandomEncoder(SaProtEncoder):
    """SaProt architecture with random weights (no pretraining)."""

    def __init__(self, model_size: str = "650M", use_3di: bool = True, **kwargs):
        # Initialize with random weights (use_pretrained=False, no checkpoint)
        super().__init__(
            model_size=model_size,
            use_pretrained=False,
            checkpoint_path=None,
            use_3di=use_3di
        )
        print(f"[random] SaProt {model_size} architecture (random weights)")
