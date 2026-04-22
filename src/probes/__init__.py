"""
Linear probing framework for invariant analysis.

This module provides:
- LinearProbe: Single linear layer for probing
- PairwiseProbe: For pairwise predictions
- ProteinLevelProbe: For protein-level predictions
- ProbeBattery: Full probing suite for fingerprinting
- InvariantFingerprint: Summary of encoder capabilities
"""

from src.probes.linear_probe import (
    LinearProbe,
    PairwiseProbe,
    ProteinLevelProbe,
    ProbeConfig,
    ProbeLevel,
    ProbeTrainer,
    ProbeType,
    create_probe,
)

from src.probes.probe_battery import (
    ALL_PROBES,
    I1_PROBES,
    I2_PROBES,
    I3_PROBES,
    I4_PROBES,
    InvariantFingerprint,
    ProbeBattery,
    compare_fingerprints,
)

__all__ = [
    # Probe types
    "LinearProbe",
    "PairwiseProbe",
    "ProteinLevelProbe",
    "ProbeConfig",
    "ProbeLevel",
    "ProbeTrainer",
    "ProbeType",
    "create_probe",
    # Probe battery
    "ALL_PROBES",
    "I1_PROBES",
    "I2_PROBES",
    "I3_PROBES",
    "I4_PROBES",
    "InvariantFingerprint",
    "ProbeBattery",
    "compare_fingerprints",
]
