"""
Probe battery for invariant fingerprinting.

Runs all probe families used in the invariant benchmark and computes the
invariant fingerprint for a frozen encoder.
"""


from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base_encoder import BaseProteinEncoder, EncoderOutput, ProteinInput
from src.probes.linear_probe import (
    ProbeConfig,
    ProbeLevel,
    ProbeTrainer,
    ProbeType,
    create_probe,
)


# ====================
# Probe Definitions
# ====================

# I1: Local Conformational Probes
I1_PROBES = [
    ProbeConfig(
        name="phi_angle_bins",
        probe_type=ProbeType.CLASSIFICATION,
        probe_level=ProbeLevel.RESIDUE,
        num_classes=36,  # 10-degree bins
        metric="accuracy",
    ),
    ProbeConfig(
        name="psi_angle_bins",
        probe_type=ProbeType.CLASSIFICATION,
        probe_level=ProbeLevel.RESIDUE,
        num_classes=36,
        metric="accuracy",
    ),
    ProbeConfig(
        name="local_curvature",
        probe_type=ProbeType.REGRESSION,
        probe_level=ProbeLevel.RESIDUE,
        metric="pearson",
    ),
]

# I2: Metric Geometry Probes
I2_PROBES = [
    ProbeConfig(
        name="distance_bins",
        probe_type=ProbeType.CLASSIFICATION,
        probe_level=ProbeLevel.PAIR,
        num_classes=16,  # Distance bins 0-40A
        metric="accuracy",
    ),
    ProbeConfig(
        name="contact_prediction",
        probe_type=ProbeType.BINARY,
        probe_level=ProbeLevel.PAIR,
        metric="f1",
    ),
]

# I3: Packing Probes
I3_PROBES = [
    ProbeConfig(
        name="contact_density",
        probe_type=ProbeType.CLASSIFICATION,
        probe_level=ProbeLevel.RESIDUE,
        num_classes=4,  # Contact density bins
        metric="accuracy",
    ),
    ProbeConfig(
        name="core_surface",
        probe_type=ProbeType.BINARY,
        probe_level=ProbeLevel.RESIDUE,
        metric="accuracy",
    ),
]

# I4: Topology Probes
I4_PROBES = [
    ProbeConfig(
        name="contact_order",
        probe_type=ProbeType.REGRESSION,
        probe_level=ProbeLevel.PROTEIN,
        metric="pearson",
    ),
]

# I5: Secondary Structure Probes
I5_PROBES = [
    ProbeConfig(
        name="dssp_3",
        probe_type=ProbeType.CLASSIFICATION,
        probe_level=ProbeLevel.RESIDUE,
        num_classes=3,  # H, E, C
        metric="accuracy",
    ),
    ProbeConfig(
        name="dssp_8",
        probe_type=ProbeType.CLASSIFICATION,
        probe_level=ProbeLevel.RESIDUE,
        num_classes=8,  # G, H, I, B, E, T, S, C
        metric="accuracy",
    ),
]

# I6: Solvent Exposure Probes
I6_PROBES = [
    ProbeConfig(
        name="rsa",
        probe_type=ProbeType.REGRESSION,
        probe_level=ProbeLevel.RESIDUE,
        metric="pearson",
    ),
    ProbeConfig(
        name="rsa_bins",
        probe_type=ProbeType.CLASSIFICATION,
        probe_level=ProbeLevel.RESIDUE,
        num_classes=3,  # Buried, intermediate, exposed
        metric="accuracy",
    ),
]

# I7: Global Fold / Architecture Probes
I7_PROBES = [
    ProbeConfig(
        name="architecture_label",
        probe_type=ProbeType.CLASSIFICATION,
        probe_level=ProbeLevel.PROTEIN,
        num_classes=31,  # CATH Architecture - 31 unique classes remapped to 0-30
        metric="accuracy",
    ),
]

# I8: Dynamics / Flexibility Probes
I8_PROBES = [
    ProbeConfig(
        name="bfactor",
        probe_type=ProbeType.REGRESSION,
        probe_level=ProbeLevel.RESIDUE,
        metric="pearson",
    ),
]

# I9: Tertiary Compactness Probes
I9_PROBES = [
    ProbeConfig(
        name="radius_of_gyration",
        probe_type=ProbeType.REGRESSION,
        probe_level=ProbeLevel.PROTEIN,
        metric="pearson",
    ),
    ProbeConfig(
        name="asphericity",
        probe_type=ProbeType.REGRESSION,
        probe_level=ProbeLevel.PROTEIN,
        metric="pearson",
    ),
]

# I10: Sequence-Structure Consistency Probes
I10_PROBES = [
    ProbeConfig(
        name="sequence_recovery",
        probe_type=ProbeType.CLASSIFICATION,
        probe_level=ProbeLevel.RESIDUE,
        num_classes=20,  # 20 amino acids
        metric="accuracy",
    ),
]

# All probe families I1-I10
ALL_PROBES = {
    "I1": I1_PROBES,
    "I2": I2_PROBES,
    "I3": I3_PROBES,
    "I4": I4_PROBES,
    "I5": I5_PROBES,
    "I6": I6_PROBES,
    "I7": I7_PROBES,
    "I8": I8_PROBES,
    "I9": I9_PROBES,
    "I10": I10_PROBES,
}

# Probes that have labels available in current dataset
# Updated 2026-04-11: Added I5, I6, I7, I8, I9 after label generation
AVAILABLE_PROBES = {
    "I1": I1_PROBES,   # phi/psi angles, curvature
    "I2": I2_PROBES,   # distances, contacts
    "I3": I3_PROBES,   # contact density, core/surface
    "I4": I4_PROBES,   # contact order
    "I5": I5_PROBES,   # DSSP secondary structure
    "I6": I6_PROBES,   # RSA solvent exposure
    "I7": I7_PROBES,   # CATH fold classification
    "I8": I8_PROBES,   # B-factor dynamics/flexibility
    "I9": I9_PROBES,   # Compactness (radius of gyration, asphericity)
    "I10": I10_PROBES, # sequence recovery
}



@dataclass
class InvariantFingerprint:
    """
    Invariant fingerprint for an encoder.

    Summarizes how well the encoder preserves each invariant family.
    """

    encoder_name: str
    p_I1: float  # Local conformational
    p_I2: float  # Metric geometry
    p_I3: float  # Packing
    p_I4: float  # Topology
    p_I5: float = 0.0  # Secondary structure
    p_I6: float = 0.0  # Solvent exposure
    p_I7: float = 0.0  # Global fold
    p_I8: float = 0.0  # Dynamics
    p_I9: float = 0.0  # Tertiary compactness
    p_I10: float = 0.0  # Sequence-structure consistency
    detailed_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def vector(self) -> np.ndarray:
        """Return fingerprint as numpy array."""
        return np.array([
            self.p_I1, self.p_I2, self.p_I3, self.p_I4, self.p_I5,
            self.p_I6, self.p_I7, self.p_I8, self.p_I9, self.p_I10,
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "encoder_name": self.encoder_name,
            "p_I1": self.p_I1,
            "p_I2": self.p_I2,
            "p_I3": self.p_I3,
            "p_I4": self.p_I4,
            "p_I5": self.p_I5,
            "p_I6": self.p_I6,
            "p_I7": self.p_I7,
            "p_I8": self.p_I8,
            "p_I9": self.p_I9,
            "p_I10": self.p_I10,
            "detailed_scores": self.detailed_scores,
        }


class ProbeBattery:
    """
    Full probing battery for invariant fingerprinting.

    Extracts embeddings from a frozen encoder and trains linear probes
    to measure decodability of I1-I10 invariant families.

    Pair-level probes (I2) use subsampled pairs to avoid O(N*L^2) memory.

    Args:
        encoder: The protein structure encoder to probe
        device: Device for computation
        probe_families: Which invariant families to probe (default: all)
    """

    def __init__(
        self,
        encoder: BaseProteinEncoder,
        device: str = "cuda",
        probe_families: Optional[List[str]] = None,
        probe_configs: Optional[List[ProbeConfig]] = None,
    ):
        self.encoder = encoder
        self.encoder.freeze()

        # Try to move to device, but skip if model uses device_map or is already placed
        try:
            if not hasattr(encoder, 'hf_device_map'):
                self.encoder.to(device)
        except (NotImplementedError, RuntimeError) as e:
            print(f"  Warning: Could not move encoder to {device} in ProbeBattery: {e}")
            print(f"  Model may already be on correct device or use device_map")

        self.device = device
        self.probe_configs = {}

        if probe_configs is not None:
            # use provided configs
            self.probe_families = list(ALL_PROBES.keys())
            for config in probe_configs:
                config.input_dim = encoder.embedding_dim
                self.probe_configs[config.name] = config
        else:
            # Select probe families from default
            self.probe_families = probe_families or list(ALL_PROBES.keys())
            for family in self.probe_families:
                if family in ALL_PROBES:
                    for config in ALL_PROBES[family]:
                        # Set input dimension from encoder
                        config.input_dim = encoder.embedding_dim
                        self.probe_configs[config.name] = config

        # Separate pair-level from non-pair probes
        self.pair_configs = {
            name: cfg for name, cfg in self.probe_configs.items()
            if cfg.probe_level == ProbeLevel.PAIR
        }
        self.non_pair_configs = {
            name: cfg for name, cfg in self.probe_configs.items()
            if cfg.probe_level != ProbeLevel.PAIR
        }

    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Extract embeddings from encoder for all samples.

        Args:
            dataloader: DataLoader for protein dataset
            max_samples: Maximum number of samples to process

        Returns:
            Tuple of (embeddings, invariant_labels, masks)
            - embeddings: (N, max_L, D) tensor
            - invariant_labels: dict of label tensors
            - masks: (N, max_L) boolean tensor
        """
        self.encoder.eval()

        all_embeddings = []
        all_labels = {name: [] for name in self.probe_configs.keys()}
        all_masks = []

        num_samples = 0
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if max_samples and num_samples >= max_samples:
                break

            # Get invariant labels from batch
            invariants = batch.get("invariants", {})
            
            # Check for missing labels
            missing_label = False
            for name in self.probe_configs:
                label_key = self._get_label_key(name)
                if label_key not in invariants:
                    # Debug only: print first few times
                    if num_samples < 5:
                        print(f"DEBUG: Missing label '{label_key}' for probe '{name}' in sample")
                    missing_label = True
                    break
            
            if missing_label:
                continue

            # Move to device
            coords = batch["coords"].to(self.device)
            sequence = batch["sequence"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Create ProteinInput
            protein_input = ProteinInput(coords=coords, sequence=sequence, mask=mask)

            # Get embeddings
            output = self.encoder.encode(protein_input)
            embeddings = output.residue_embeddings.cpu()

            all_embeddings.append(embeddings)
            all_masks.append(mask.cpu())

            # Append labels (guaranteed to exist now)
            for name, config in self.probe_configs.items():
                label_key = self._get_label_key(name)
                all_labels[name].append(invariants[label_key].cpu())

            num_samples += coords.shape[0]

        # Determine global max length
        max_len_found = 0
        for emb in all_embeddings:
            max_len_found = max(max_len_found, emb.shape[1])

        # Pad and concatenate
        final_embeddings = []
        final_masks = []
        
        for emb, msk in zip(all_embeddings, all_masks):
            B_emb, L_emb, D = emb.shape
            B_msk, L_msk = msk.shape

            if L_emb < max_len_found:
                pad_len_emb = max_len_found - L_emb
                emb = torch.cat(
                    [emb, torch.zeros(B_emb, pad_len_emb, D, dtype=emb.dtype)], dim=1
                )

            if L_msk < max_len_found:
                pad_len_msk = max_len_found - L_msk
                msk = torch.cat(
                    [msk, torch.zeros(B_msk, pad_len_msk, dtype=msk.dtype)], dim=1
                )

            final_embeddings.append(emb)
            final_masks.append(msk)
                
        # Concatenate
        embeddings = torch.cat(final_embeddings, dim=0)
        masks = torch.cat(final_masks, dim=0)

        # Pad and concatenate labels
        labels = {}
        for name, label_list in all_labels.items():
            if not label_list:
                continue
                
            final_lbls = []
            for lbl in label_list:
                if lbl.ndim > 1:
                    pad_len = max_len_found - lbl.shape[1]
                    if pad_len > 0:
                        if lbl.ndim == 2: # (B, L)
                            # Pad with 0 (ignored by mask)
                            lbl_padded = torch.cat([lbl, torch.zeros(lbl.shape[0], pad_len, dtype=lbl.dtype)], dim=1)
                            final_lbls.append(lbl_padded)
                        elif lbl.ndim == 3: # (B, L, L)
                            B, L, L2 = lbl.shape # L should equal L2
                            # Pad dim 1
                            lbl = torch.cat([lbl, torch.zeros(B, pad_len, L, dtype=lbl.dtype)], dim=1)
                            # Pad dim 2 (now shape is B, L+pad, L)
                            lbl = torch.cat([lbl, torch.zeros(B, L+pad_len, pad_len, dtype=lbl.dtype)], dim=2)
                            final_lbls.append(lbl)
                        else:
                            # Should not happen for currently supported probes
                            final_lbls.append(lbl)
                    else:
                        final_lbls.append(lbl)
                else:
                    final_lbls.append(lbl)
            
            labels[name] = torch.cat(final_lbls, dim=0)

        return embeddings, labels, masks

    def _get_label_key(self, probe_name: str) -> str:
        """Map probe name to label key in dataset."""
        mapping = {
            # I1: Local conformational
            "phi_angle_bins": "phi_bins",
            "psi_angle_bins": "psi_bins",
            "local_curvature": "curvature",
            # I2: Metric geometry
            "distance_bins": "distance_bins",
            "contact_prediction": "contacts",
            # I3: Packing
            "contact_density": "contact_density",
            "core_surface": "core_surface_labels",
            # I4: Topology
            "contact_order": "contact_order",
            # I5: Secondary structure
            "dssp_3": "dssp_3",
            "dssp_8": "dssp_8",
            # I6: Solvent exposure
            "rsa_regression": "rsa",
            "rsa_classification": "rsa_bins",
            # I7: Global fold (CATH Architecture)
            "architecture_label": "architecture_label",
            "fold_classification": "fold_label",  # Legacy, kept for compatibility
            # I8: Dynamics
            "bfactor_regression": "bfactor",
            # I9: Tertiary compactness
            "radius_of_gyration": "radius_of_gyration",
            "asphericity": "asphericity",
            # I10: Sequence-structure consistency
            "sequence_recovery": "sequence",
        }
        return mapping.get(probe_name, probe_name)

    @torch.no_grad()
    def _extract_pair_data(
        self,
        dataloader: DataLoader,
        probe_name: str,
        max_samples: Optional[int] = None,
        max_pairs_per_protein: int = 500,
        min_seq_sep: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract subsampled pair features and labels for a pairwise probe.

        Instead of storing full (N, L, L) label matrices, samples K pairs
        per protein and returns flat (N_pairs, 2*D) features + (N_pairs,) labels.

        Args:
            dataloader: DataLoader for protein dataset
            probe_name: Name of the pair probe
            max_samples: Max proteins to process
            max_pairs_per_protein: Max pairs sampled per protein
            min_seq_sep: Minimum |i-j| separation for valid pairs

        Returns:
            (pair_features, pair_labels) flat tensors
        """
        self.encoder.eval()

        all_pair_features = []
        all_pair_labels = []
        label_key = self._get_label_key(probe_name)
        num_samples = 0

        for batch in tqdm(dataloader, desc=f"Extracting pairs ({probe_name})"):
            if max_samples and num_samples >= max_samples:
                break

            invariants = batch.get("invariants", {})
            if label_key not in invariants:
                if num_samples < 3:
                    print(f"DEBUG: Missing pair label '{label_key}' for probe '{probe_name}'")
                continue

            coords = batch["coords"].to(self.device)
            sequence = batch["sequence"].to(self.device)
            mask = batch["mask"].to(self.device)

            protein_input = ProteinInput(coords=coords, sequence=sequence, mask=mask)
            output = self.encoder.encode(protein_input)
            embeddings = output.residue_embeddings.cpu()  # (B, L, D)

            pair_labels_batch = invariants[label_key].cpu()  # (B, L, L)
            mask_cpu = mask.cpu()
            B, L, D = embeddings.shape

            for b in range(B):
                m = mask_cpu[b]  # (L,)
                pair_mask = m.unsqueeze(1) & m.unsqueeze(0)  # (L, L)

                # Exclude self and close sequence neighbors
                seq_idx = torch.arange(L)
                seq_sep = (seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1)).abs()
                pair_mask = pair_mask & (seq_sep >= min_seq_sep)

                valid_pairs = pair_mask.nonzero(as_tuple=False)  # (N_valid, 2)
                if len(valid_pairs) == 0:
                    continue

                # Get labels for all valid pairs to enable balanced sampling
                all_labels_b = pair_labels_batch[b, valid_pairs[:, 0], valid_pairs[:, 1]]

                # Balanced sampling for binary probes (contact_prediction)
                # to avoid class-imbalance collapse (contacts ~1% of pairs)
                is_binary = probe_name in self.pair_configs and \
                    self.pair_configs[probe_name].probe_type == ProbeType.BINARY
                if is_binary:
                    pos_mask = all_labels_b.bool()
                    neg_mask = ~pos_mask
                    pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)
                    neg_idx = neg_mask.nonzero(as_tuple=False).squeeze(-1)

                    n_pos = len(pos_idx)
                    n_neg = len(neg_idx)
                    if n_pos == 0 or n_neg == 0:
                        continue

                    # Sample up to K/2 from each class
                    half_k = max_pairs_per_protein // 2
                    if n_pos > half_k:
                        pos_idx = pos_idx[torch.randperm(n_pos)[:half_k]]
                    if n_neg > half_k:
                        neg_idx = neg_idx[torch.randperm(n_neg)[:half_k]]

                    selected_idx = torch.cat([pos_idx, neg_idx])
                    valid_pairs = valid_pairs[selected_idx]
                elif len(valid_pairs) > max_pairs_per_protein:
                    perm = torch.randperm(len(valid_pairs))[:max_pairs_per_protein]
                    valid_pairs = valid_pairs[perm]

                i_idx = valid_pairs[:, 0]
                j_idx = valid_pairs[:, 1]

                # Construct pair features: concat(h_i, h_j)
                h_i = embeddings[b, i_idx]  # (K, D)
                h_j = embeddings[b, j_idx]  # (K, D)
                pair_feat = torch.cat([h_i, h_j], dim=-1)  # (K, 2*D)

                # Get corresponding labels
                labels = pair_labels_batch[b, i_idx, j_idx]  # (K,)

                all_pair_features.append(pair_feat)
                all_pair_labels.append(labels)

            num_samples += B

        if not all_pair_features:
            return torch.empty(0), torch.empty(0)

        # Memory-efficient concatenation: compute total size, pre-allocate, copy in-place
        total_pairs = sum(f.shape[0] for f in all_pair_features)
        feat_dim = all_pair_features[0].shape[1]
        label_dtype = all_pair_labels[0].dtype
        out_features = torch.empty(total_pairs, feat_dim)
        out_labels = torch.empty(total_pairs, dtype=label_dtype)
        offset = 0
        for feat, lbl in zip(all_pair_features, all_pair_labels):
            n = feat.shape[0]
            out_features[offset:offset + n] = feat
            out_labels[offset:offset + n] = lbl
            offset += n
        del all_pair_features, all_pair_labels

        return out_features, out_labels

    def _train_flat_pair_probe(
        self,
        config: ProbeConfig,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: Optional[torch.Tensor],
        val_labels: Optional[torch.Tensor],
        num_epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train a probe on pre-extracted flat pair data.

        Creates a LinearProbe with input_dim=2*D and trains on flat (N_pairs,)
        data, avoiding O(L^2) memory during training.
        """
        if len(train_features) == 0:
            return {
                "config": config,
                "best_metric": 0.0,
                "final_metrics": {},
                "probe": None,
            }

        # Create a flat probe (treat pairs as individual samples)
        flat_config = ProbeConfig(
            name=config.name,
            probe_type=config.probe_type,
            probe_level=ProbeLevel.RESIDUE,  # Flat data, no pair expansion needed
            num_classes=config.num_classes,
            input_dim=config.input_dim * 2,  # Concatenated pair dim
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            metric=config.metric,
        )

        probe = create_probe(flat_config)
        trainer = ProbeTrainer(probe=probe, config=flat_config, device=self.device, lr=lr)

        # Reshape: (N_pairs,) -> (N_pairs, 1, 2*D) with trivial mask
        train_emb = train_features.unsqueeze(1)  # (N, 1, 2D)
        train_lbl = train_labels.unsqueeze(1) if train_labels.ndim == 1 else train_labels
        train_mask = torch.ones(len(train_features), 1, dtype=torch.bool)

        val_emb, val_lbl, val_mask = None, None, None
        if val_features is not None and len(val_features) > 0:
            val_emb = val_features.unsqueeze(1)
            val_lbl = val_labels.unsqueeze(1) if val_labels.ndim == 1 else val_labels
            val_mask = torch.ones(len(val_features), 1, dtype=torch.bool)

        result = trainer.train(
            train_embeddings=train_emb,
            train_labels=train_lbl,
            train_mask=train_mask,
            val_embeddings=val_emb,
            val_labels=val_lbl,
            val_mask=val_mask,
            num_epochs=num_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        return {
            "config": config,
            "best_metric": result["best_metric"],
            "final_metrics": result["final_metrics"],
            "probe": probe,
        }

    def train_probes(
        self,
        train_embeddings: torch.Tensor,
        train_labels: Dict[str, torch.Tensor],
        train_masks: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_labels: Optional[Dict[str, torch.Tensor]] = None,
        val_masks: Optional[torch.Tensor] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all probes on cached embeddings.

        Args:
            train_embeddings: Training embeddings (N, L, D)
            train_labels: Training labels dict
            train_masks: Training masks (N, L)
            val_embeddings: Validation embeddings
            val_labels: Validation labels dict
            val_masks: Validation masks
            num_epochs: Training epochs per probe
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress

        Returns:
            Dict of probe results
        """
        results = {}

        for name, config in tqdm(
            self.probe_configs.items(), desc="Training probes", disable=not verbose
        ):
            if name not in train_labels:
                if verbose:
                    print(f"Skipping {name}: no labels found")
                continue

            # Create probe
            probe = create_probe(config)

            # Create trainer
            trainer = ProbeTrainer(
                probe=probe,
                config=config,
                device=self.device,
                lr=lr,
            )

            # Get labels for this probe
            train_lbl = train_labels[name]
            val_lbl = val_labels.get(name) if val_labels else None

            # Train
            result = trainer.train(
                train_embeddings=train_embeddings,
                train_labels=train_lbl,
                train_mask=train_masks,
                val_embeddings=val_embeddings,
                val_labels=val_lbl,
                val_mask=val_masks,
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=verbose,
            )

            results[name] = {
                "config": config,
                "best_metric": result["best_metric"],
                "final_metrics": result["final_metrics"],
                "probe": probe,
            }

        return results

    def compute_fingerprint(
        self,
        probe_results: Dict[str, Dict[str, Any]],
    ) -> InvariantFingerprint:
        """
        Compute invariant fingerprint from probe results.

        Aggregates probe scores by invariant family.

        Args:
            probe_results: Results from train_probes()

        Returns:
            InvariantFingerprint
        """
        # Map probes to families
        probe_to_family = {}
        for family, configs in ALL_PROBES.items():
            for config in configs:
                probe_to_family[config.name] = family

        # Aggregate scores by family
        family_scores = {f: [] for f in self.probe_families}
        detailed_scores = {}

        for name, result in probe_results.items():
            family = probe_to_family.get(name)
            if family and family in family_scores:
                # Use primary metric from config
                config = result["config"]
                metrics = result["final_metrics"]
                score = metrics.get(config.metric, result["best_metric"])

                # Normalize score to [0, 1] range
                if config.probe_type == ProbeType.REGRESSION:
                    # Pearson correlation is already in [-1, 1], map to [0, 1]
                    score = (score + 1) / 2
                # Classification metrics are already in [0, 1]

                family_scores[family].append(score)
                detailed_scores[name] = score

        # Compute family averages
        def mean_or_zero(scores):
            return np.mean(scores) if scores else 0.0

        return InvariantFingerprint(
            encoder_name=self.encoder.name,
            p_I1=mean_or_zero(family_scores.get("I1", [])),
            p_I2=mean_or_zero(family_scores.get("I2", [])),
            p_I3=mean_or_zero(family_scores.get("I3", [])),
            p_I4=mean_or_zero(family_scores.get("I4", [])),
            p_I5=mean_or_zero(family_scores.get("I5", [])),
            p_I6=mean_or_zero(family_scores.get("I6", [])),
            p_I7=mean_or_zero(family_scores.get("I7", [])),
            p_I8=mean_or_zero(family_scores.get("I8", [])),
            p_I9=mean_or_zero(family_scores.get("I9", [])),
            p_I10=mean_or_zero(family_scores.get("I10", [])),
            detailed_scores=detailed_scores,
        )

    def run(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> InvariantFingerprint:
        """
        Run full probing battery.

        Residue/protein-level probes use cached embeddings.
        Pair-level probes (I2) use subsampled pairs for memory efficiency.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            max_train_samples: Max training samples
            max_val_samples: Max validation samples
            num_epochs: Epochs per probe
            batch_size: Batch size for probe training
            lr: Learning rate
            verbose: Print progress

        Returns:
            InvariantFingerprint for the encoder
        """
        if verbose:
            print(f"Running probe battery on {self.encoder.name}")
            print(f"Encoder type: {self.encoder.bottleneck_class.name}")
            print(f"Embedding dim: {self.encoder.embedding_dim}")

        probe_results = {}

        # Phase 1: Residue and protein-level probes (cached extraction)
        if self.non_pair_configs:
            # Temporarily swap probe_configs to only non-pair
            saved_configs = self.probe_configs
            self.probe_configs = self.non_pair_configs

            train_emb, train_labels, train_masks = self.extract_embeddings(
                train_dataloader, max_train_samples
            )

            val_emb, val_labels, val_masks = None, None, None
            if val_dataloader:
                val_emb, val_labels, val_masks = self.extract_embeddings(
                    val_dataloader, max_val_samples
                )

            non_pair_results = self.train_probes(
                train_emb, train_labels, train_masks,
                val_emb, val_labels, val_masks,
                num_epochs, batch_size, lr, verbose,
            )
            probe_results.update(non_pair_results)

            # Restore configs and free memory
            self.probe_configs = saved_configs
            del train_emb, train_labels, train_masks
            del val_emb, val_labels, val_masks

        # Phase 2: Pair-level probes (subsampled extraction)
        if self.pair_configs:
            if verbose:
                print(f"\n--- Pair-level probes ({len(self.pair_configs)} tasks) ---")

            for name, config in self.pair_configs.items():
                if verbose:
                    print(f"  Pair probe: {name}")

                train_feat, train_lbl = self._extract_pair_data(
                    train_dataloader, name,
                    max_samples=max_train_samples,
                    max_pairs_per_protein=500,
                )

                val_feat, val_lbl = None, None
                if val_dataloader:
                    val_feat, val_lbl = self._extract_pair_data(
                        val_dataloader, name,
                        max_samples=max_val_samples,
                        max_pairs_per_protein=200,
                    )

                if verbose:
                    print(f"    Train pairs: {len(train_feat)}, Val pairs: {len(val_feat) if val_feat is not None else 0}")

                result = self._train_flat_pair_probe(
                    config, train_feat, train_lbl,
                    val_feat, val_lbl,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    verbose=verbose,
                )
                probe_results[name] = result

                del train_feat, train_lbl, val_feat, val_lbl

        # Compute fingerprint
        fingerprint = self.compute_fingerprint(probe_results)

        if verbose:
            print(f"\nInvariant Fingerprint for {self.encoder.name}:")
            print(f"  I1 (Local Conformational): {fingerprint.p_I1:.3f}")
            print(f"  I2 (Metric Geometry):      {fingerprint.p_I2:.3f}")
            print(f"  I3 (Packing):              {fingerprint.p_I3:.3f}")
            print(f"  I4 (Long-range Topology):  {fingerprint.p_I4:.3f}")
            print(f"  I5 (Sequence-Structure):   {fingerprint.p_I5:.3f}")
            print(f"  I6 (Global Shape):         {fingerprint.p_I6:.3f}")
            print(f"  I7 (Flexibility):          {fingerprint.p_I7:.3f}")
            print(f"  I8 (Chemistry):            {fingerprint.p_I8:.3f}")

        return fingerprint


def compare_fingerprints(
    fingerprints: List[InvariantFingerprint],
) -> Dict[str, Any]:
    """
    Compare fingerprints across multiple encoders.

    Args:
        fingerprints: List of fingerprints to compare

    Returns:
        Comparison statistics
    """
    names = [f.encoder_name for f in fingerprints]
    vectors = np.array([f.vector for f in fingerprints])

    # Best encoder per invariant family
    family_names = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9"]
    best_per_family = {
        family: names[np.argmax(vectors[:, i])]
        for i, family in enumerate(family_names)
        if i < vectors.shape[1]
    }

    # Mean and std per encoder
    encoder_stats = {}
    for i, name in enumerate(names):
        encoder_stats[name] = {
            "mean": float(vectors[i].mean()),
            "std": float(vectors[i].std()),
            "vector": vectors[i].tolist(),
        }

    return {
        "best_per_family": best_per_family,
        "encoder_stats": encoder_stats,
        "fingerprint_matrix": vectors.tolist(),
        "encoder_names": names,
    }
