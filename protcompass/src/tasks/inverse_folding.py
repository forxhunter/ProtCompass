"""
Inverse folding (sequence design) task.

Given a protein backbone structure, predict the amino acid sequence.
This is a key benchmark for evaluating structure encoders.

Metrics:
- Sequence recovery rate
- Perplexity
- Per-position accuracy
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base_encoder import BaseProteinEncoder, EncoderOutput, ProteinInput


@dataclass
class InverseFoldingConfig:
    """Configuration for inverse folding task."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    label_smoothing: float = 0.1
    num_epochs: int = 50
    batch_size: int = 8


class InverseFoldingHead(nn.Module):
    """
    Sequence prediction head for inverse folding.

    Takes encoder embeddings and predicts amino acid probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict amino acid logits.

        Args:
            embeddings: Per-residue embeddings (B, L, D)

        Returns:
            Logits (B, L, 20)
        """
        return self.classifier(embeddings)


class InverseFoldingTask(nn.Module):
    """
    Inverse folding task wrapper.

    Combines encoder with prediction head for sequence design.

    Args:
        encoder: Protein structure encoder
        config: Task configuration
        freeze_encoder: Whether to freeze encoder weights
    """

    def __init__(
        self,
        encoder: BaseProteinEncoder,
        config: Optional[InverseFoldingConfig] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config or InverseFoldingConfig()
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            self.encoder.freeze()

        self.head = InverseFoldingHead(
            input_dim=encoder.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

    def forward(self, protein: ProteinInput) -> Tuple[torch.Tensor, EncoderOutput]:
        """
        Forward pass.

        Args:
            protein: Protein input

        Returns:
            Tuple of (logits, encoder_output)
        """
        # Get embeddings
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_output = self.encoder.encode(protein)
        else:
            encoder_output = self.encoder.encode(protein)

        # Predict sequence
        logits = self.head(encoder_output.residue_embeddings)

        return logits, encoder_output

    def predict(self, protein: ProteinInput) -> torch.Tensor:
        """
        Predict amino acid sequence.

        Args:
            protein: Protein input

        Returns:
            Predicted sequence (B, L)
        """
        logits, _ = self.forward(protein)
        return logits.argmax(dim=-1)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with label smoothing.

        Args:
            logits: Predicted logits (B, L, 20)
            targets: Target sequence (B, L)
            mask: Valid residue mask (B, L)

        Returns:
            Scalar loss
        """
        B, L, C = logits.shape

        # Flatten
        logits_flat = logits.view(-1, C)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Cross entropy with label smoothing
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            label_smoothing=self.config.label_smoothing,
            reduction='none',
        )

        # Apply mask
        loss = (loss * mask_flat.float()).sum() / mask_flat.float().sum()

        return loss


def compute_recovery_rate(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute sequence recovery rate.

    Args:
        predictions: Predicted sequence (B, L)
        targets: Target sequence (B, L)
        mask: Valid residue mask (B, L)

    Returns:
        Recovery rate (0-1)
    """
    correct = (predictions == targets) & mask
    recovery = correct.float().sum() / mask.float().sum()
    return recovery.item()


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute perplexity.

    Args:
        logits: Predicted logits (B, L, 20)
        targets: Target sequence (B, L)
        mask: Valid residue mask (B, L)

    Returns:
        Perplexity
    """
    B, L, C = logits.shape

    # Cross entropy
    logits_flat = logits.view(-1, C)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)

    ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    mean_ce = (ce * mask_flat.float()).sum() / mask_flat.float().sum()

    return torch.exp(mean_ce).item()


def compute_per_aa_recovery(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute recovery rate per amino acid type.

    Args:
        predictions: Predicted sequence (B, L)
        targets: Target sequence (B, L)
        mask: Valid residue mask (B, L)

    Returns:
        Dict mapping AA index to recovery rate
    """
    aa_names = list('ACDEFGHIKLMNPQRSTVWY')
    results = {}

    for aa_idx in range(20):
        aa_mask = (targets == aa_idx) & mask
        if aa_mask.sum() > 0:
            correct = (predictions == targets) & aa_mask
            recovery = correct.float().sum() / aa_mask.float().sum()
            results[aa_names[aa_idx]] = recovery.item()

    return results


class InverseFoldingTrainer:
    """
    Trainer for inverse folding task.

    Args:
        model: InverseFoldingTask model
        device: Device for training
    """

    def __init__(
        self,
        model: InverseFoldingTask,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.device = device
        self.config = model.config

        # Optimizer (only for unfrozen parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recovery = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            coords = batch['coords'].to(self.device)
            sequence = batch['sequence'].to(self.device)
            mask = batch['mask'].to(self.device)

            protein = ProteinInput(coords=coords, sequence=sequence, mask=mask)

            self.optimizer.zero_grad()

            logits, _ = self.model(protein)
            loss = self.model.compute_loss(logits, sequence, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                recovery = compute_recovery_rate(predictions, sequence, mask)

            total_loss += loss.item()
            total_recovery += recovery
            num_batches += 1

        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'recovery': total_recovery / num_batches,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0.0
        total_recovery = 0.0
        total_perplexity = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []
        all_masks = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            coords = batch['coords'].to(self.device)
            sequence = batch['sequence'].to(self.device)
            mask = batch['mask'].to(self.device)

            protein = ProteinInput(coords=coords, sequence=sequence, mask=mask)

            logits, _ = self.model(protein)
            loss = self.model.compute_loss(logits, sequence, mask)

            predictions = logits.argmax(dim=-1)
            recovery = compute_recovery_rate(predictions, sequence, mask)
            perplexity = compute_perplexity(logits, sequence, mask)

            total_loss += loss.item()
            total_recovery += recovery
            total_perplexity += perplexity
            num_batches += 1

            all_predictions.append(predictions.cpu())
            all_targets.append(sequence.cpu())
            all_masks.append(mask.cpu())

        # Compute per-AA recovery
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        per_aa = compute_per_aa_recovery(all_predictions, all_targets, all_masks)

        return {
            'loss': total_loss / num_batches,
            'recovery': total_recovery / num_batches,
            'perplexity': total_perplexity / num_batches,
            'per_aa_recovery': per_aa,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (defaults to config)
            verbose: Print progress

        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.num_epochs

        history = {
            'train_loss': [],
            'train_recovery': [],
            'val_loss': [],
            'val_recovery': [],
            'val_perplexity': [],
        }

        best_recovery = 0.0
        best_state = None

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_recovery'].append(train_metrics['recovery'])

            # Validate
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_recovery'].append(val_metrics['recovery'])
                history['val_perplexity'].append(val_metrics['perplexity'])

                # Save best
                if val_metrics['recovery'] > best_recovery:
                    best_recovery = val_metrics['recovery']
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}: "
                        f"train_loss={train_metrics['loss']:.4f}, "
                        f"train_rec={train_metrics['recovery']:.4f}, "
                        f"val_rec={val_metrics['recovery']:.4f}, "
                        f"val_ppl={val_metrics['perplexity']:.2f}"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}: "
                        f"train_loss={train_metrics['loss']:.4f}, "
                        f"train_rec={train_metrics['recovery']:.4f}"
                    )

        # Restore best
        if best_state:
            self.model.load_state_dict(best_state)

        return {
            'history': history,
            'best_recovery': best_recovery,
        }
