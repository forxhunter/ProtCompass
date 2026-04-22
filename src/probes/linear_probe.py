"""
Linear probing for protein structure embeddings.

Linear probes measure the linear decodability of geometric invariants
from frozen encoder representations. Single linear layer ensures we
measure what is explicitly encoded, not what can be computed.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class ProbeType(Enum):
    """Type of probing task."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    BINARY = "binary"


class ProbeLevel(Enum):
    """Level of prediction."""

    RESIDUE = "residue"  # Per-residue prediction (B, L, ...)
    PAIR = "pair"  # Pairwise prediction (B, L, L, ...)
    PROTEIN = "protein"  # Protein-level prediction (B, ...)


@dataclass
class ProbeConfig:
    """Configuration for a linear probe."""

    name: str
    probe_type: ProbeType
    probe_level: ProbeLevel
    num_classes: Optional[int] = None  # For classification
    input_dim: int = 0  # Set by probe battery
    hidden_dim: Optional[int] = None # For MLP probes
    num_layers: int = 1 # Number of layers (1=Linear, 2=MLP-1, etc.)
    metric: str = "accuracy"  # Primary evaluation metric

    def __post_init__(self):
        if self.probe_type == ProbeType.CLASSIFICATION and self.num_classes is None:
            raise ValueError("num_classes required for classification probes")


class LinearProbe(nn.Module):
    """
    Linear probe for invariant prediction.

    A single linear layer that maps embeddings to predictions.
    Used with frozen encoder to measure linear decodability.

    Args:
        input_dim: Dimension of input embeddings
        output_dim: Dimension of output (num_classes or 1 for regression)
        probe_type: Type of prediction task
        probe_level: Level of prediction (residue, pair, protein)
        use_bias: Whether to include bias term
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        probe_type: ProbeType = ProbeType.CLASSIFICATION,
        probe_level: ProbeLevel = ProbeLevel.RESIDUE,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.probe_type = probe_type
        self.probe_level = probe_level

        if kwargs.get("hidden_dim") and kwargs.get("num_layers", 1) > 1:
            hidden = kwargs["hidden_dim"]
            num_layers = kwargs.get("num_layers", 2)
            
            layers = []
            # Input -> Hidden 1
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            
            # Hidden -> Hidden (if num_layers > 2)
            # nomenclature: "MLP-1" usually means 1 hidden layer (2 linear layers)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.ReLU())
                
            # Hidden -> Output
            layers.append(nn.Linear(hidden, output_dim))
            
            self.linear = nn.Sequential(*layers)
        else:
            self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)

        # Initialize weights
        for m in self.linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Input embeddings
                - Residue-level: (B, L, D)
                - Pair-level: (B, L, L, D) or (B, L, L, 2*D) for concatenated
                - Protein-level: (B, D)

        Returns:
            Predictions:
                - Classification: logits (B, L, C) or (B, L, L, C) or (B, C)
                - Regression: values (B, L, 1) or (B, L, L, 1) or (B, 1)
        """
        return self.linear(embeddings)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get predictions (class labels or values).

        Args:
            embeddings: Input embeddings

        Returns:
            Predicted classes (classification) or values (regression)
        """
        logits = self.forward(embeddings)

        if self.probe_type == ProbeType.CLASSIFICATION:
            return logits.argmax(dim=-1)
        elif self.probe_type == ProbeType.BINARY:
            return (torch.sigmoid(logits) > 0.5).squeeze(-1)
        else:  # REGRESSION
            return logits.squeeze(-1)


class PairwiseProbe(nn.Module):
    """
    Linear probe for pairwise predictions.

    Concatenates pairs of residue embeddings and applies linear layer.

    Args:
        input_dim: Dimension of residue embeddings
        output_dim: Dimension of output
        probe_type: Type of prediction task
        combination: How to combine residue pairs ('concat', 'diff', 'product')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        probe_type: ProbeType = ProbeType.CLASSIFICATION,
        combination: str = "concat",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.probe_type = probe_type
        self.combination = combination

        # Compute combined dimension
        if combination == "concat":
            combined_dim = input_dim * 2
        elif combination == "diff":
            combined_dim = input_dim
        elif combination == "product":
            combined_dim = input_dim
        else:
            raise ValueError(f"Unknown combination: {combination}")

        self.linear = nn.Linear(combined_dim, output_dim)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for pairwise prediction.

        Args:
            embeddings: Residue embeddings (B, L, D)
            mask: Optional residue mask (B, L)

        Returns:
            Pairwise logits/values (B, L, L, output_dim)
        """
        if self.combination == "concat":
            # Optimized forward for concat: W[u; v] + b = W1 u + W2 v + b
            # avoids creating (B, L, L, 2D) tensor
            weight = self.linear.weight  # (out, 2D)
            bias = self.linear.bias      # (out)
            
            w1 = weight[:, :self.input_dim]
            w2 = weight[:, self.input_dim:]
            
            # (B, L, D) @ (D, out) -> (B, L, out)
            # Note: Linear uses x @ W.T
            h1 = F.linear(embeddings, w1)
            h2 = F.linear(embeddings, w2)
            
            # Broadcast sum: (B, L, 1, out) + (B, 1, L, out)
            return h1.unsqueeze(2) + h2.unsqueeze(1) + (bias if bias is not None else 0)

        elif self.combination == "diff":
            # Optimized forward for diff: W(u - v) + b = Wu - Wv + b
            weight = self.linear.weight
            bias = self.linear.bias
            
            h = F.linear(embeddings, weight)  # (B, L, out)
            
            return h.unsqueeze(2) - h.unsqueeze(1) + (bias if bias is not None else 0)

        else:
            # Fallback for product (or massive memory usage)
            B, L, D = embeddings.shape
            emb_i = embeddings.unsqueeze(2).expand(-1, -1, L, -1)
            emb_j = embeddings.unsqueeze(1).expand(-1, L, -1, -1)
            
            if self.combination == "product":
                pair_features = emb_i * emb_j
            else:
                # Fallback to concat if somehow logic fails (should not happen)
                pair_features = torch.cat([emb_i, emb_j], dim=-1)

            return self.linear(pair_features)

    def predict(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get predictions."""
        logits = self.forward(embeddings, mask)

        if self.probe_type == ProbeType.CLASSIFICATION:
            return logits.argmax(dim=-1)
        elif self.probe_type == ProbeType.BINARY:
            return (torch.sigmoid(logits) > 0.5).squeeze(-1)
        else:
            return logits.squeeze(-1)


class ProteinLevelProbe(nn.Module):
    """
    Linear probe for protein-level predictions.

    Pools residue embeddings then applies linear layer.

    Args:
        input_dim: Dimension of residue embeddings
        output_dim: Dimension of output
        probe_type: Type of prediction task
        pooling: Pooling method ('mean', 'max', 'cls')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        probe_type: ProbeType = ProbeType.CLASSIFICATION,
        pooling: str = "mean",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.probe_type = probe_type
        self.pooling = pooling

        self.linear = nn.Linear(input_dim, output_dim)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for protein-level prediction.

        Args:
            embeddings: Residue embeddings (B, L, D)
            mask: Optional residue mask (B, L)

        Returns:
            Protein-level logits/values (B, output_dim)
        """
        if mask is None:
            mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)

        if self.pooling == "mean":
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            # Masked max pooling
            embeddings_masked = embeddings.clone()
            embeddings_masked[~mask] = float("-inf")
            pooled = embeddings_masked.max(dim=1)[0]
        elif self.pooling == "cls":
            # Use first token
            pooled = embeddings[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.linear(pooled)

    def predict(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get predictions."""
        logits = self.forward(embeddings, mask)

        if self.probe_type == ProbeType.CLASSIFICATION:
            return logits.argmax(dim=-1)
        elif self.probe_type == ProbeType.BINARY:
            return (torch.sigmoid(logits) > 0.5).squeeze(-1)
        else:
            return logits.squeeze(-1)


def create_probe(config: ProbeConfig) -> nn.Module:
    """
    Create a probe from configuration.

    Args:
        config: ProbeConfig specifying the probe

    Returns:
        Probe module (LinearProbe, PairwiseProbe, or ProteinLevelProbe)
    """
    if config.probe_type == ProbeType.REGRESSION:
        output_dim = 1
    elif config.probe_type == ProbeType.BINARY:
        output_dim = 1
    else:
        output_dim = config.num_classes

    if config.probe_level == ProbeLevel.RESIDUE:
        return LinearProbe(
            input_dim=config.input_dim,
            output_dim=output_dim,
            probe_type=config.probe_type,
            probe_level=config.probe_level,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
    elif config.probe_level == ProbeLevel.PAIR:
        return PairwiseProbe(
            input_dim=config.input_dim,
            output_dim=output_dim,
            probe_type=config.probe_type,
        )
    elif config.probe_level == ProbeLevel.PROTEIN:
        return ProteinLevelProbe(
            input_dim=config.input_dim,
            output_dim=output_dim,
            probe_type=config.probe_type,
        )
    else:
        raise ValueError(f"Unknown probe level: {config.probe_level}")


class ProbeTrainer:
    """
    Trainer for linear probes.

    Handles training loop, evaluation, and logging for probing experiments.

    Args:
        probe: The probe module to train
        config: Probe configuration
        device: Device to train on
        lr: Learning rate
        weight_decay: Weight decay for regularization
    """

    def __init__(
        self,
        probe: nn.Module,
        config: ProbeConfig,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.probe = probe.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            probe.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Loss function
        if config.probe_type == ProbeType.CLASSIFICATION:
            self.criterion = nn.CrossEntropyLoss(reduction="none")
        elif config.probe_type == ProbeType.BINARY:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.criterion = nn.MSELoss(reduction="none")

    def train_epoch(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> float:
        """
        Train for one epoch.

        Args:
            embeddings: Cached embeddings (N, L, D) or (N, D)
            labels: Ground truth labels
            mask: Optional mask for valid positions
            batch_size: Batch size

        Returns:
            Average training loss
        """
        self.probe.train()
        total_loss = 0.0
        num_batches = 0

        # Simple batching
        num_samples = embeddings.shape[0]
        indices = torch.randperm(num_samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_emb = embeddings[batch_indices].to(self.device)
            batch_labels = labels[batch_indices].to(self.device)
            batch_mask = mask[batch_indices].to(self.device) if mask is not None else None

            self.optimizer.zero_grad()

            # Forward pass
            if self.config.probe_level == ProbeLevel.PROTEIN:
                logits = self.probe(batch_emb, batch_mask)
            elif self.config.probe_level == ProbeLevel.PAIR:
                logits = self.probe(batch_emb, batch_mask)
            else:
                logits = self.probe(batch_emb)

            # Compute loss
            loss = self._compute_loss(logits, batch_labels, batch_mask)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute masked loss."""
        if self.config.probe_type == ProbeType.CLASSIFICATION:
            # Reshape for cross entropy
            if self.config.probe_level == ProbeLevel.RESIDUE:
                # (B, L, C) -> (B*L, C), (B, L) -> (B*L,)
                B, L, C = logits.shape
                logits_flat = logits.reshape(-1, C)
                labels_flat = labels.reshape(-1).long()  # Convert to int64 for CUDA
                
                if mask is not None:
                    mask_flat = mask.reshape(-1)
                    # SAFEGUARD: Masked positions might have invalid labels (e.g. -1).
                    labels_flat = torch.where(mask_flat, labels_flat, torch.zeros_like(labels_flat))
                    
                    # Double safeguard: clamp to be sure
                    labels_flat = labels_flat.clamp(0, C-1)
                    
                    loss = self.criterion(logits_flat, labels_flat)
                    loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
                else:
                    # SAFEGUARD: Check for invalids even without mask
                    if (labels_flat < 0).any() or (labels_flat >= C).any():
                        print(f"WARNING: Invalid labels in unmasked batch! Min: {labels_flat.min()}, Max: {labels_flat.max()}, C: {C}")
                        labels_flat = labels_flat.clamp(0, C-1)
                        
                    loss = self.criterion(logits_flat, labels_flat).mean()

            elif self.config.probe_level == ProbeLevel.PAIR:
                # (B, L, L, C) -> (B*L*L, C)
                B, L1, L2, C = logits.shape
                logits_flat = logits.reshape(-1, C)
                labels_flat = labels.reshape(-1).long()  # Convert to int64 for CUDA
                
                if mask is not None:
                    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                    mask_flat = pair_mask.reshape(-1)
                    
                    # SAFEGUARD: Masked positions might have invalid labels (e.g. -1).
                    labels_flat = torch.where(mask_flat, labels_flat, torch.zeros_like(labels_flat))
                    labels_flat = labels_flat.clamp(0, C-1)
                    
                    loss = self.criterion(logits_flat, labels_flat)
                    loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
                else:
                    if (labels_flat < 0).any() or (labels_flat >= C).any():
                        labels_flat = labels_flat.clamp(0, C-1)
                    loss = self.criterion(logits_flat, labels_flat).mean()

            else:  # PROTEIN
                loss = self.criterion(logits, labels.long()).mean()  # Convert to int64 for CUDA

        elif self.config.probe_type == ProbeType.BINARY:
            logits_squeezed = logits.squeeze(-1)
            loss = self.criterion(logits_squeezed, labels.float())

            if mask is not None:
                if self.config.probe_level == ProbeLevel.PAIR:
                    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                    loss = (loss * pair_mask).sum() / pair_mask.sum().clamp(min=1)
                else:
                    loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            else:
                loss = loss.mean()

        else:  # REGRESSION
            logits_squeezed = logits.squeeze(-1)
            loss = self.criterion(logits_squeezed, labels)

            if mask is not None:
                if self.config.probe_level == ProbeLevel.PROTEIN:
                    loss = loss.mean()
                elif self.config.probe_level == ProbeLevel.PAIR:
                    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                    loss = (loss * pair_mask).sum() / pair_mask.sum().clamp(min=1)
                else:
                    loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            else:
                loss = loss.mean()

        return loss

    @torch.no_grad()
    def evaluate(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Evaluate probe on data.

        Args:
            embeddings: Cached embeddings
            labels: Ground truth labels
            mask: Optional mask
            batch_size: Batch size

        Returns:
            Dictionary of metrics
        """
        self.probe.eval()
        all_predictions = []
        all_labels = []
        all_masks = []

        num_samples = embeddings.shape[0]

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            batch_emb = embeddings[start_idx:end_idx].to(self.device)
            batch_labels = labels[start_idx:end_idx]
            batch_mask = mask[start_idx:end_idx].to(self.device) if mask is not None else None

            # Get predictions
            if self.config.probe_level == ProbeLevel.PROTEIN:
                predictions = self.probe.predict(batch_emb, batch_mask)
            elif self.config.probe_level == ProbeLevel.PAIR:
                predictions = self.probe.predict(batch_emb, batch_mask)
            else:
                predictions = self.probe.predict(batch_emb)

            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels)
            if batch_mask is not None:
                all_masks.append(batch_mask.cpu())

        # Concatenate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0) if all_masks else None

        # Compute metrics
        return self._compute_metrics(all_predictions, all_labels, all_masks)

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}

        if mask is not None:
            if self.config.probe_level == ProbeLevel.PROTEIN:
                valid_preds = predictions
                valid_labels = labels
            elif self.config.probe_level == ProbeLevel.PAIR:
                # Expand mask for pairs
                pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                valid_preds = predictions[pair_mask]
                valid_labels = labels[pair_mask]
            else:
                valid_preds = predictions[mask]
                valid_labels = labels[mask]
        else:
            valid_preds = predictions.flatten()
            valid_labels = labels.flatten()

        if self.config.probe_type == ProbeType.CLASSIFICATION:
            # Accuracy
            correct = (valid_preds == valid_labels).float()
            metrics["accuracy"] = correct.mean().item()

            # Per-class accuracy
            num_classes = self.config.num_classes
            for c in range(num_classes):
                class_mask = valid_labels == c
                if class_mask.sum() > 0:
                    class_acc = correct[class_mask].mean().item()
                    metrics[f"accuracy_class_{c}"] = class_acc

        elif self.config.probe_type == ProbeType.BINARY:
            # Binary metrics
            correct = (valid_preds == valid_labels).float()
            metrics["accuracy"] = correct.mean().item()

            # Precision, recall
            tp = ((valid_preds == 1) & (valid_labels == 1)).sum().float()
            fp = ((valid_preds == 1) & (valid_labels == 0)).sum().float()
            fn = ((valid_preds == 0) & (valid_labels == 1)).sum().float()

            precision = tp / (tp + fp).clamp(min=1)
            recall = tp / (tp + fn).clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

            metrics["precision"] = precision.item()
            metrics["recall"] = recall.item()
            metrics["f1"] = f1.item()

        else:  # REGRESSION
            # MSE, MAE, Pearson correlation
            mse = F.mse_loss(valid_preds.float(), valid_labels.float())
            mae = F.l1_loss(valid_preds.float(), valid_labels.float())

            metrics["mse"] = mse.item()
            metrics["mae"] = mae.item()

            # Pearson correlation
            if len(valid_preds) > 1:
                pred_centered = valid_preds - valid_preds.mean()
                label_centered = valid_labels - valid_labels.mean()
                corr = (pred_centered * label_centered).sum() / (
                    pred_centered.norm() * label_centered.norm() + 1e-8
                )
                metrics["pearson"] = corr.item()

        return metrics

    def train(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        train_mask: Optional[torch.Tensor] = None,
        val_embeddings: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        num_epochs: int = 10,
        batch_size: int = 32,
        patience: int = 3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.

        Args:
            train_embeddings: Training embeddings
            train_labels: Training labels
            train_mask: Training mask
            val_embeddings: Validation embeddings
            val_labels: Validation labels
            val_mask: Validation mask
            num_epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
            verbose: Whether to print progress

        Returns:
            Dictionary with training history and best metrics
        """
        history = {"train_loss": [], "val_metrics": []}
        best_metric = -float("inf") if self.config.probe_type != ProbeType.REGRESSION else float("inf")
        best_state = None
        patience_counter = 0

        primary_metric = self.config.metric

        iterator = range(num_epochs)
        if verbose:
            iterator = tqdm(iterator, desc=f"Training {self.config.name}")

        for epoch in iterator:
            # Train
            train_loss = self.train_epoch(
                train_embeddings, train_labels, train_mask, batch_size
            )
            history["train_loss"].append(train_loss)

            # Validate
            if val_embeddings is not None:
                val_metrics = self.evaluate(
                    val_embeddings, val_labels, val_mask, batch_size
                )
                history["val_metrics"].append(val_metrics)

                # Early stopping
                current_metric = val_metrics.get(primary_metric, val_metrics.get("accuracy", 0))

                if self.config.probe_type == ProbeType.REGRESSION:
                    # Lower is better for regression
                    improved = current_metric < best_metric
                else:
                    # Higher is better for classification
                    improved = current_metric > best_metric

                if improved:
                    best_metric = current_metric
                    best_state = {k: v.cpu().clone() for k, v in self.probe.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

                if verbose:
                    tqdm.write(
                        f"Epoch {epoch + 1}: loss={train_loss:.4f}, {primary_metric}={current_metric:.4f}"
                    )

        # Restore best state
        if best_state is not None:
            self.probe.load_state_dict(best_state)

        return {
            "history": history,
            "best_metric": best_metric,
            "final_metrics": history["val_metrics"][-1] if history["val_metrics"] else {},
        }
