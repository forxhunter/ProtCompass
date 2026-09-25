"""
Downstream benchmark tasks for protein structure encoders.

Provides:
- Inverse folding (sequence recovery)
- Function prediction (GO/EC)
- Interface prediction
- Stability prediction
"""

from src.tasks.inverse_folding import (
    InverseFoldingConfig,
    InverseFoldingHead,
    InverseFoldingTask,
    InverseFoldingTrainer,
    compute_recovery_rate,
    compute_perplexity,
    compute_per_aa_recovery,
)

__all__ = [
    "InverseFoldingConfig",
    "InverseFoldingHead",
    "InverseFoldingTask",
    "InverseFoldingTrainer",
    "compute_recovery_rate",
    "compute_perplexity",
    "compute_per_aa_recovery",
]
