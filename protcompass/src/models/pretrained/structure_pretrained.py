"""
Pretrained structure encoder wrappers.

These wrappers load Swiss-Prot pretrained checkpoints for structure encoders,
and official task checkpoints where available.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import torch
from src.models.base_encoder import register_encoder

# Import encoders with optional dependencies
try:
    from src.models.class_c.gvp_encoder import GVPEncoder
    GVP_AVAILABLE = True
except ImportError:
    GVP_AVAILABLE = False

from src.models.class_c.egnn_encoder import EGNNEncoder

try:
    from src.models.class_c.tfn_encoder import TFNEncoder
    TFN_AVAILABLE = True
except ImportError:
    TFN_AVAILABLE = False

from src.models.class_c.ipa_encoder import IPAEncoder
from src.models.class_c.gcpnet_encoder import GCPNetEncoder

try:
    from src.models.class_c.cdconv_encoder import CDConvEncoder
    CDCONV_AVAILABLE = True
except (ImportError, FileNotFoundError):
    CDCONV_AVAILABLE = False

from src.models.class_b.dmasif_encoder import dMaSIFEncoder



if GVP_AVAILABLE:
    @register_encoder("gvp_pretrained")
    class GVPPretrainedEncoder(GVPEncoder):
        """GVP encoder with Swiss-Prot pretraining."""

        def __init__(self, **kwargs):
            super().__init__(
                node_s_dim=kwargs.get("node_s_dim", 128),
                node_v_dim=kwargs.get("node_v_dim", 16),
                edge_s_dim=kwargs.get("edge_s_dim", 32),
                edge_v_dim=kwargs.get("edge_v_dim", 1),
                num_layers=kwargs.get("num_layers", 3),
                k_neighbors=kwargs.get("k_neighbors", 30),
            )
            self._pretrained_path = "/mnt/f/checkpoints/gvp_pretrained/best.pt"

        def load_pretrained(self, path=None):
            if path is None:
                path = self._pretrained_path
            checkpoint = torch.load(path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "encoder" in checkpoint:
                state_dict = checkpoint["encoder"]
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, strict=False)


@register_encoder("egnn_pretrained")
class EGNNPretrainedEncoder(EGNNEncoder):
    """EGNN encoder with Swiss-Prot pretraining."""

    def __init__(self, **kwargs):
        super().__init__(
            hidden_dim=kwargs.get("hidden_dim", 128),
            num_layers=kwargs.get("num_layers", 4),
            k_neighbors=kwargs.get("k_neighbors", 30),
        )
        self._pretrained_path = "/mnt/f/checkpoints/egnn_pretrained/best.pt"

    def load_pretrained(self, path=None):
        if path is None:
            path = self._pretrained_path
        checkpoint = torch.load(path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "encoder" in checkpoint:
            state_dict = checkpoint["encoder"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=False)


if TFN_AVAILABLE:
    @register_encoder("tfn_pretrained")
    class TFNPretrainedEncoder(TFNEncoder):
        """TFN/e3nn encoder with Swiss-Prot pretraining."""

        def __init__(self, **kwargs):
            super().__init__(
                hidden_dim=kwargs.get("hidden_dim", 128),
                depth=kwargs.get("depth", 4),
                k_neighbors=kwargs.get("k_neighbors", 16),
            )
            self._pretrained_path = "/mnt/f/checkpoints/tfn_pretrained/best.pt"

        def load_pretrained(self, path=None):
            if path is None:
                path = self._pretrained_path
            checkpoint = torch.load(path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "encoder" in checkpoint:
                state_dict = checkpoint["encoder"]
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, strict=False)


@register_encoder("ipa_pretrained")
class IPAPretrainedEncoder(IPAEncoder):
    """IPA encoder with Swiss-Prot pretraining."""

    def __init__(self, **kwargs):
        super().__init__(
            hidden_dim=kwargs.get("hidden_dim", 128),
            num_layers=kwargs.get("num_layers", 4),
            num_heads=kwargs.get("num_heads", 4),
            num_points=kwargs.get("num_points", 4),
            k_neighbors=kwargs.get("k_neighbors", 30),
        )
        self._pretrained_path = "/mnt/f/checkpoints/ipa_pretrained/best.pt"

    def load_pretrained(self, path=None):
        if path is None:
            path = self._pretrained_path
        checkpoint = torch.load(path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "encoder" in checkpoint:
            state_dict = checkpoint["encoder"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=False)


@register_encoder("gcpnet_pretrained")
class GCPNetPretrainedEncoder(GCPNetEncoder):
    """GCPNet encoder with official PSR pretrained weights."""

    def __init__(self, **kwargs):
        psr_checkpoint = (PROTCOMPASS_ROOT + "/protcompass/external_repos/GCPNet/checkpoints/PSR/model_epoch_115_localpearson_0_616_localspearman_0_532_localkendall_0_385_globalpearson_0_871_globalspearman_0_869_globalkendall_0_676.ckpt")
        super().__init__(pretrained_path=psr_checkpoint, **kwargs)


if CDCONV_AVAILABLE:
    @register_encoder("cdconv_pretrained")
    class CDConvPretrainedEncoder(CDConvEncoder):
        """CDConv encoder using official fold training package artifacts."""

        def __init__(self, **kwargs):
            # Official package present at weights/fold.tar.gz; wrapper stays architecture-compatible.
            # If direct model checkpoint extraction is added later, wire it through load_pretrained.
            super().__init__(**kwargs)
            self._pretrained_bundle = "/mnt/f/2026_structuerembed/external_repos/Continuous-Discrete-Convolution/weights/fold.tar.gz"


@register_encoder("dmasif_pretrained")
class DMaSIFPretrainedEncoder(dMaSIFEncoder):
    """dMaSIF encoder with official search-task checkpoint."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pretrained_path = "/mnt/f/2026_structuerembed/external_repos/dMaSIF/models/dMaSIF_search_3layer_12A_16dim"
        self.load_pretrained(self._pretrained_path)
