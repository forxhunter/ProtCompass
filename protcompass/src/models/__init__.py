"""
Protein structure encoder models.

This module provides encoder wrappers for different bottleneck classes:
- Class A: Discrete token (SaProt, Foldseek)
- Class B: Graph (ProteinMPNN)
- Class C: Equivariant (GVP, EGNN)
"""

from src.models.base_encoder import (
    AA_VOCAB,
    AA_VOCAB_INV,
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    ENCODER_REGISTRY,
    get_encoder,
    list_encoders,
    register_encoder,
    sequence_to_tensor,
    tensor_to_sequence,
)
from .external import (
    ESM2Adapter,
    CNN3DEncoder,
    ProtTransAdapter,
    AnkhAdapter,
    MIFAdapter
)

# Import encoder classes to register them
from src.models.class_a.saprot_encoder import SaProtEncoder, FoldseekEncoder
from src.models.class_a.prostt5_encoder import ProstT5Encoder
from src.models.class_a.proteinbert_encoder import ProteinBertBFDEncoder
from src.models.class_b.proteinmpnn_encoder import ProteinMPNNEncoder
from src.models.class_b.gearnet_encoder import GearNetEncoder
try:
    from src.models.class_c.gvp_encoder import GVPEncoder
except ImportError:
    pass  # torch_geometric not available
from src.models.class_c.egnn_encoder import EGNNEncoder
from src.models.class_c.se3_transformer_encoder import SE3TransformerEncoder
from src.models.class_c.ipa_encoder import IPAEncoder
from src.models.class_c.schnet_encoder import SchNetEncoder
from src.models.class_c.dimenet_encoder import DimeNetEncoder
try:
    from src.models.class_c.cdconv_encoder import CDConvEncoder
except (ImportError, FileNotFoundError):
    pass  # torch_geometric not available or external_repos missing
from src.models.class_c.gcpnet_encoder import GCPNetEncoder
try:
    from src.models.class_c.tfn_encoder import TFNEncoder
except ImportError:
    pass  # e3nn not available
try:
    from src.models.class_c.mace_encoder import MACEEncoder
except ImportError:
    pass  # e3nn not available
from src.models.baselines.geometric_baselines import RandomTokenBaseline
register_encoder("random_token")(RandomTokenBaseline)
from src.models.class_b.dmasif_encoder import dMaSIFEncoder

# Note: DIY/experimental encoders removed from the registry:
# - dual_stream, moe_stream, task_conditioned_tri_stream
# - hybrid, hybrid_v2, hybrid_v3
# - deep_egnn, multigraph, ensemble, multik_egnn
# These were exploratory architectures without published baselines

__all__ = [
    # Base classes
    "BaseProteinEncoder",
    "BottleneckClass",
    "EncoderOutput",
    "ProteinInput",
    # Registry
    "ENCODER_REGISTRY",
    "get_encoder",
    "list_encoders",
    "register_encoder",
    # Vocabulary
    "AA_VOCAB",
    "AA_VOCAB_INV",
    "sequence_to_tensor",
    "tensor_to_sequence",
    # Encoders
    "SaProtEncoder",
    "FoldseekEncoder",
    "ProstT5Encoder",
    "ProteinMPNNEncoder",
    "GearNetEncoder",
    "GVPEncoder",
    "EGNNEncoder",
    "ProtTransAdapter",
    "AnkhAdapter",
    "MIFAdapter",
]

# Pretrained encoder wrappers (official architectures)
from src.models.pretrained.proteinmpnn_official import ProteinMPNNPretrainedEncoder
from src.models.pretrained.esm_if_encoder import ESMIFEncoder
from src.models.pretrained.esm3_encoder import ESM3Encoder
from src.models.pretrained.geobpe_encoder import GeoBPEEncoder
from src.models.pretrained.saprot_pretrained import SaProtPretrainedEncoder
from src.models.pretrained.gearnet_pretrained import GearNetPretrainedEncoder
from src.models.pretrained.proteinae_encoder import ProteinAEEncoder
from src.models.pretrained.pottsmpnn_encoder import (
    PottsMPNNPretrainedEncoder,
    PottsMPNNRandomEncoder,
    PottsMPNNCAPretrainedEncoder,
    PottsMPPNSolublePretrainedEncoder,
    PottsMPNNFTPretrainedEncoder,
)
from src.models.pretrained.foldvision_encoder import FoldVisionEncoder
from src.models.pretrained.starling_encoder import StarlingEncoder
from src.models.pretrained.random_plm_encoders import (
    ESM2RandomEncoder,
    ESM2_150MRandomEncoder,
    ESM2_35MRandomEncoder,
    ESM1bRandomEncoder,
    ProteinBertBFDRandomEncoder,
    ProtTransRandomEncoder,
    AnkhRandomEncoder,
    ProstT5RandomEncoder,
)
try:
    from src.models.pretrained.structure_pretrained import (
        GVPPretrainedEncoder,
        EGNNPretrainedEncoder,
        TFNPretrainedEncoder,
        IPAPretrainedEncoder,
        GCPNetPretrainedEncoder,
        CDConvPretrainedEncoder,
        DMaSIFPretrainedEncoder,
    )
except ImportError:
    pass  # Some pretrained encoders unavailable due to missing dependencies

# Register baselines manually to avoid decorator dependency loops
from src.models.baselines.geometric_baselines import TorsionBaseline, DensityBaseline
ENCODER_REGISTRY["torsion"] = TorsionBaseline
ENCODER_REGISTRY["density"] = DensityBaseline
