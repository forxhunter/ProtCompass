"""
Analysis module for model selection, decision tables, and statistical testing.

Provides:
- Predictive model selection based on invariant fingerprints
- Task requirement learning
- Decision table generation
- Publication-grade statistical comparisons (Wilcoxon, bootstrap CI, BH correction)
"""

from src.analysis.model_selection import (
    ModelSelector,
    ModelRecommendation,
    TaskRequirements,
    create_default_selector,
    print_decision_table,
    DEFAULT_TASK_REQUIREMENTS,
)
from src.analysis.statistical_tests import (
    BenchmarkStatistics,
    ComparisonResult,
    FamilySummary,
    TASK_TO_FAMILY,
)

__all__ = [
    "ModelSelector",
    "ModelRecommendation",
    "TaskRequirements",
    "create_default_selector",
    "print_decision_table",
    "DEFAULT_TASK_REQUIREMENTS",
    "BenchmarkStatistics",
    "ComparisonResult",
    "FamilySummary",
    "TASK_TO_FAMILY",
]
