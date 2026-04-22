"""
Centralized config for the downstream task benchmark.
Edit this file to control which encoders/tasks/seeds run.
"""
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "structure_encoder_data"
EMB_DIR   = Path(__file__).parent / "embeddings"
RES_DIR   = Path(__file__).parent / "results"

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]

# ── Encoder registry names (must match ENCODER_REGISTRY keys) ─────────────────
# Updated to match all 39 encoders from probing results for consistency
ENCODERS_STRUCTURE_RANDOM = [
    # Official library-based (random weights only, no protein-pretrained checkpoints)
    "egnn",           # egnn-pytorch
    "tfn",            # e3nn (official TFN successor)
    "gvp",            # gvp-pytorch
    "cdconv",         # Continuous-Discrete-Convolution
    "gcpnet",         # GCPNet
    "dimenet",        # DimeNet++
    "schnet",         # SchNet
    "mace",           # MACE
    # Local custom implementations (may have pretrained from training)
    "gearnet",        # GearNet-IEConv (local pretrained available)
    "pottsmpnn",      # PottsMPNN encoder-only (no pretrained checkpoint shipped)
]
ENCODERS_PRETRAINED = [
    # Language model encoders
    "esm2_35m", "esm2_150m", "esm2",     # ESM-2 family
    "esm1b", "esm_if", "esm3",           # other ESM variants
    "proteinbert_bfd", "prottrans", "ankh",
    "saprot_pretrained",                  # SaProt
    "prostt5", "saprot",
    # Structure-aware pretrained
    "egnn_pretrained",                    # pretrained EGNN
    "gearnet_pretrained",                 # mc_gearnet_edge.pth
    "proteinmpnn_pretrained",             # proteinmpnn_v48_020.pt
    "pottsmpnn_pretrained",               # proteinmpnn_v48_020.pt (same weights, encoder-only)
    "pottsmpnn_soluble_pretrained",       # sol_pottsmpnn_20.pt (soluble-protein optimized)
    "pottsmpnn_ft_pretrained",            # potts_ft.pt (fine-tuned)
    "cdconv_pretrained",                  # pretrained CDConv
    "gcpnet_pretrained",                  # pretrained GCPNet
    "gvp_pretrained",                     # pretrained GVP
    "dmasif_pretrained",                  # pretrained dMaSIF
    "ipa_pretrained",                     # pretrained IPA
    "tfn_pretrained",                     # pretrained TFN
    "foldvision",                         # AlexanderKroll/foldvision-encoder
]
# Baselines for comparison
ENCODERS_BASELINES = [
    "random",         # random embeddings
    "length",         # protein length baseline
    "torsion",        # torsion angles baseline
    "foldseek",       # Foldseek structural alphabet
]
ALL_ENCODERS = ENCODERS_STRUCTURE_RANDOM + ENCODERS_PRETRAINED + ENCODERS_BASELINES

# Encoders that accept only sequences (no 3D coords needed)
SEQUENCE_ONLY_ENCODERS = {
    "esm2_35m", "esm2_150m", "esm2", "esm1b", "esm3",
    "proteinbert_bfd", "prottrans", "ankh", "prostt5",
}

# Tasks where structure encoders with dummy coords are MEANINGLESS
# (sequence variants, no real PDB structure available per protein)
# Note: mutation_effect now has AF2 structures (216/217 assays), so structure encoders can run
SEQUENCE_ONLY_TASKS = {"fluorescence", "ppi_affinity", "solubility"}

# ── Task definitions ───────────────────────────────────────────────────────────
# metric: primary evaluation metric
# level: "protein" (mean-pool) or "residue" (per-residue)
# type:  "clf" | "reg" | "multilabel" | "contact"
TASKS = {
    "secondary_structure":      {"metric": "Q3_accuracy",     "level": "residue",  "type": "clf"},
    "contact_prediction":       {"metric": "precision_L5",    "level": "residue",  "type": "contact"},
    "remote_homology":          {"metric": "accuracy",        "level": "protein",  "type": "clf"},
    "fluorescence":             {"metric": "spearman_rho",    "level": "protein",  "type": "reg"},
    "stability":                {"metric": "spearman_rho",    "level": "protein",  "type": "reg"},
    "ec_classification":        {"metric": "accuracy",        "level": "protein",  "type": "clf"},
    "go_mf":                    {"metric": "Fmax",            "level": "protein",  "type": "multilabel"},
    "go_bp":                    {"metric": "Fmax",            "level": "protein",  "type": "multilabel"},
    "ppi_site":                 {"metric": "AUPRC",           "level": "residue",  "type": "clf"},
    "metal_binding":            {"metric": "AUPRC",           "level": "residue",  "type": "clf"},
    "subcellular_localization": {"metric": "accuracy",        "level": "protein",  "type": "clf"},
    "membrane_soluble":         {"metric": "AUROC",           "level": "protein",  "type": "clf"},
    # New tasks (T13-T15)
    "mutation_effect":           {"metric": "spearman_rho",    "level": "protein",  "type": "reg"},
    "ppi_affinity":             {"metric": "spearman_rho",    "level": "protein",  "type": "reg"},
    "solubility":               {"metric": "AUROC",           "level": "protein",  "type": "clf"},
}

# ── I-family → probing task mapping (from InvariantsBench sweep) ───────────────
IFAMILY_PROBE_TASKS = {
    "I1": ["phi_angle_bins", "psi_angle_bins", "local_curvature"],
    "I2": ["distance_bins", "contact_prediction"],
    "I3": ["contact_density", "core_surface"],
    "I4": ["contact_order"],
    "I5": ["amino_acid_identity", "hydrophobicity"],
    "I6": ["radius_of_gyration", "helix_fraction", "sheet_fraction",
           "coil_fraction", "asphericity"],
    "I7": ["bfactor_proxy", "backbone_entropy_proxy"],
    "I8": ["charge_distribution", "hydrogen_bond_count"],
}

PRETRAINED_SET = set(ENCODERS_PRETRAINED)
RANDOM_SET     = set(ENCODERS_STRUCTURE_RANDOM)
