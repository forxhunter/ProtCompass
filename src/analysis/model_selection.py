"""
Predictive model selection framework.

Uses invariant fingerprints and task requirement vectors to predict
which encoder will perform best for a given task, without expensive training.

Key concepts:
- Invariant Fingerprint: p(E) = [p_I1, p_I2, p_I3, p_I4] per encoder
- Task Requirement: w_T = [w_I1, w_I2, w_I3, w_I4] per task
- Match Score: Match(E, T) = sum(w_Tk * p_Ik) - lambda * Cost(E)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.probes import InvariantFingerprint


# Default task requirement vectors based on domain knowledge
# Higher values indicate stronger requirement for that invariant
DEFAULT_TASK_REQUIREMENTS = {
    "inverse_folding": {
        "I1": 0.8,  # Local conformational (secondary structure)
        "I2": 0.4,  # Metric geometry
        "I3": 0.9,  # Packing (critical for design)
        "I4": 0.3,  # Long-range (less critical)
    },
    "function_prediction": {
        "I1": 0.7,  # Local motifs matter
        "I2": 0.3,  # Distances less critical
        "I3": 0.5,  # Packing for binding sites
        "I4": 0.8,  # Fold/topology important
    },
    "interface_prediction": {
        "I1": 0.4,  # Local less critical
        "I2": 0.9,  # Distances/orientations critical
        "I3": 0.8,  # Surface/core matters
        "I4": 0.3,  # Long-range less important
    },
    "stability_prediction": {
        "I1": 0.5,  # Local structure
        "I2": 0.8,  # Packing interactions
        "I3": 0.9,  # Core burial critical
        "I4": 0.4,  # Global fold
    },
    "structure_retrieval": {
        "I1": 0.9,  # Local features for matching
        "I2": 0.3,  # Exact distances less critical
        "I3": 0.4,  # Packing
        "I4": 0.6,  # Fold similarity
    },
}

# Approximate compute costs (relative scale 0-1)
DEFAULT_ENCODER_COSTS = {
    "saprot": 0.3,  # Fast token-based
    "foldseek": 0.1,  # Very fast
    "proteinmpnn": 0.4,  # Moderate
    "gvp": 0.6,  # More expensive
    "egnn": 0.7,  # Expensive
    "se3_transformer": 0.9,  # Most expensive
}


@dataclass
class TaskRequirements:
    """Task requirement specification."""

    task_name: str
    w_I1: float  # Local conformational
    w_I2: float  # Metric geometry
    w_I3: float  # Packing
    w_I4: float  # Long-range topology
    description: str = ""

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.w_I1, self.w_I2, self.w_I3, self.w_I4])

    @classmethod
    def from_dict(cls, name: str, reqs: Dict[str, float]) -> "TaskRequirements":
        return cls(
            task_name=name,
            w_I1=reqs.get("I1", 0.5),
            w_I2=reqs.get("I2", 0.5),
            w_I3=reqs.get("I3", 0.5),
            w_I4=reqs.get("I4", 0.5),
        )


@dataclass
class ModelRecommendation:
    """Recommendation for model selection."""

    task_name: str
    recommended_encoder: str
    match_score: float
    all_scores: Dict[str, float]
    reasoning: str
    task_requirements: TaskRequirements
    bottleneck_recommendation: str  # 'A', 'B', or 'C'


class ModelSelector:
    """
    Predictive model selection based on invariant fingerprints.

    Uses empirical fingerprints and task requirements to recommend
    the best encoder for a given task.

    Args:
        fingerprints: Dict of encoder_name -> InvariantFingerprint
        encoder_costs: Optional dict of encoder_name -> cost
        cost_weight: Weight for cost penalty (lambda)
    """

    def __init__(
        self,
        fingerprints: Optional[Dict[str, InvariantFingerprint]] = None,
        encoder_costs: Optional[Dict[str, float]] = None,
        cost_weight: float = 0.1,
    ):
        self.fingerprints = fingerprints or {}
        self.encoder_costs = encoder_costs or DEFAULT_ENCODER_COSTS
        self.cost_weight = cost_weight

        # Task requirements (can be learned or specified)
        self.task_requirements = {
            name: TaskRequirements.from_dict(name, reqs)
            for name, reqs in DEFAULT_TASK_REQUIREMENTS.items()
        }

    def add_fingerprint(self, fingerprint: InvariantFingerprint) -> None:
        """Add an encoder fingerprint."""
        self.fingerprints[fingerprint.encoder_name] = fingerprint

    def add_task_requirements(self, requirements: TaskRequirements) -> None:
        """Add custom task requirements."""
        self.task_requirements[requirements.task_name] = requirements

    def compute_match_score(
        self,
        fingerprint: InvariantFingerprint,
        requirements: TaskRequirements,
    ) -> float:
        """
        Compute match score between encoder and task.

        Match(E, T) = sum(w_Tk * p_Ik) - lambda * Cost(E)

        Args:
            fingerprint: Encoder fingerprint
            requirements: Task requirements

        Returns:
            Match score (higher is better)
        """
        # Weighted sum of invariant matches
        match = np.sum(requirements.vector * fingerprint.vector)

        # Cost penalty
        cost = self.encoder_costs.get(fingerprint.encoder_name.lower(), 0.5)
        penalty = self.cost_weight * cost

        return match - penalty

    def recommend(
        self,
        task_name: str,
        available_encoders: Optional[List[str]] = None,
    ) -> ModelRecommendation:
        """
        Recommend best encoder for a task.

        Args:
            task_name: Name of the task
            available_encoders: List of available encoder names (default: all)

        Returns:
            ModelRecommendation with best encoder and reasoning
        """
        if task_name not in self.task_requirements:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.task_requirements.keys())}")

        requirements = self.task_requirements[task_name]

        # Filter available encoders
        if available_encoders:
            fingerprints = {
                name: fp for name, fp in self.fingerprints.items()
                if name in available_encoders
            }
        else:
            fingerprints = self.fingerprints

        if not fingerprints:
            raise ValueError("No encoder fingerprints available")

        # Compute scores for all encoders
        scores = {}
        for name, fp in fingerprints.items():
            scores[name] = self.compute_match_score(fp, requirements)

        # Find best
        best_encoder = max(scores.keys(), key=lambda x: scores[x])
        best_fp = fingerprints[best_encoder]

        # Determine bottleneck class recommendation
        bottleneck_rec = self._recommend_bottleneck_class(requirements)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            task_name, requirements, best_fp, scores
        )

        return ModelRecommendation(
            task_name=task_name,
            recommended_encoder=best_encoder,
            match_score=scores[best_encoder],
            all_scores=scores,
            reasoning=reasoning,
            task_requirements=requirements,
            bottleneck_recommendation=bottleneck_rec,
        )

    def _recommend_bottleneck_class(
        self,
        requirements: TaskRequirements,
    ) -> str:
        """
        Recommend bottleneck class based on task requirements.

        Based on theoretical predictions:
        - Class A (Token): Strong I1, efficient
        - Class B (Graph): Strong I3, moderate I2
        - Class C (Equivariant): Strong I2, expensive
        """
        w = requirements.vector

        # Heuristic rules
        if w[0] > 0.7 and w[1] < 0.5:  # High I1, low I2
            return "A"
        elif w[1] > 0.7:  # High I2 (metric geometry)
            return "C"
        elif w[2] > 0.7:  # High I3 (packing)
            return "B"
        elif w[3] > 0.7:  # High I4 (topology)
            return "A"  # Or hybrid
        else:
            return "B"  # Default to graph

    def _generate_reasoning(
        self,
        task_name: str,
        requirements: TaskRequirements,
        best_fp: InvariantFingerprint,
        scores: Dict[str, float],
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        # Find strongest requirement
        req_names = ["I1 (local)", "I2 (metric)", "I3 (packing)", "I4 (topology)"]
        strongest_idx = np.argmax(requirements.vector)
        strongest_req = req_names[strongest_idx]

        # Find how well best encoder satisfies it
        satisfaction = best_fp.vector[strongest_idx]

        reasoning = f"""
Task '{task_name}' requires:
- Strongest: {strongest_req} (weight={requirements.vector[strongest_idx]:.2f})

Recommended: {best_fp.encoder_name}
- {strongest_req} preservation: {satisfaction:.2f}
- Overall match score: {scores[best_fp.encoder_name]:.3f}

Alternative scores: {', '.join(f'{k}: {v:.3f}' for k, v in sorted(scores.items(), key=lambda x: -x[1])[:3])}
"""
        return reasoning.strip()

    def learn_task_requirements(
        self,
        fingerprints: List[InvariantFingerprint],
        task_performances: Dict[str, float],
        task_name: str,
    ) -> TaskRequirements:
        """
        Learn task requirement vector from empirical data.

        Fits: task_score ~ sum(w_k * p_Ik) using ridge regression.

        Args:
            fingerprints: List of encoder fingerprints
            task_performances: Dict of encoder_name -> performance
            task_name: Name for the task

        Returns:
            Learned TaskRequirements
        """
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            raise ImportError("sklearn required for learning. Install with: pip install scikit-learn")

        # Build feature matrix
        X = np.array([fp.vector for fp in fingerprints])
        y = np.array([task_performances[fp.encoder_name] for fp in fingerprints])

        # Fit ridge regression
        model = Ridge(alpha=0.1, fit_intercept=False)
        model.fit(X, y)

        # Extract weights (normalize to [0, 1])
        weights = model.coef_
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

        requirements = TaskRequirements(
            task_name=task_name,
            w_I1=float(weights[0]),
            w_I2=float(weights[1]),
            w_I3=float(weights[2]),
            w_I4=float(weights[3]),
            description=f"Learned from {len(fingerprints)} encoders",
        )

        # Add to stored requirements
        self.task_requirements[task_name] = requirements

        return requirements

    def generate_decision_table(
        self,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate empirical decision table.

        Args:
            tasks: List of tasks (default: all known tasks)

        Returns:
            Decision table mapping task -> {encoder -> score, recommended -> name}
        """
        tasks = tasks or list(self.task_requirements.keys())

        table = {}

        for task in tasks:
            try:
                rec = self.recommend(task)
                table[task] = {
                    **rec.all_scores,
                    "recommended": rec.recommended_encoder,
                    "bottleneck_class": rec.bottleneck_recommendation,
                }
            except ValueError:
                continue

        return table

    def validate_predictions(
        self,
        ground_truth: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Validate predictions against ground truth performance.

        Args:
            ground_truth: Dict of task -> {encoder -> actual_performance}

        Returns:
            Validation statistics
        """
        results = {
            "correct_predictions": 0,
            "total_predictions": 0,
            "rank_correlation": [],
            "per_task": {},
        }

        for task, performances in ground_truth.items():
            if task not in self.task_requirements:
                continue

            try:
                rec = self.recommend(task, available_encoders=list(performances.keys()))
            except ValueError:
                continue

            # Check if prediction matches actual best
            actual_best = max(performances.keys(), key=lambda x: performances[x])
            predicted_best = rec.recommended_encoder

            correct = actual_best == predicted_best
            results["total_predictions"] += 1
            if correct:
                results["correct_predictions"] += 1

            # Compute rank correlation
            predicted_ranks = list(sorted(rec.all_scores.keys(), key=lambda x: -rec.all_scores[x]))
            actual_ranks = list(sorted(performances.keys(), key=lambda x: -performances[x]))

            try:
                from scipy.stats import spearmanr
                corr, _ = spearmanr(
                    [predicted_ranks.index(e) for e in performances.keys()],
                    [actual_ranks.index(e) for e in performances.keys()],
                )
                results["rank_correlation"].append(corr)
            except ImportError:
                pass

            results["per_task"][task] = {
                "predicted": predicted_best,
                "actual": actual_best,
                "correct": correct,
            }

        results["accuracy"] = (
            results["correct_predictions"] / max(results["total_predictions"], 1)
        )
        results["mean_rank_correlation"] = (
            np.mean(results["rank_correlation"]) if results["rank_correlation"] else None
        )

        return results


def create_default_selector() -> ModelSelector:
    """
    Create a model selector with default configurations.

    Returns:
        ModelSelector with default task requirements
    """
    return ModelSelector()


def print_decision_table(table: Dict[str, Dict[str, Any]]) -> None:
    """
    Pretty print a decision table.

    Args:
        table: Decision table from ModelSelector.generate_decision_table()
    """
    # Get all encoders
    encoders = set()
    for task_data in table.values():
        encoders.update(k for k in task_data.keys() if k not in ["recommended", "bottleneck_class"])
    encoders = sorted(encoders)

    # Header
    header = f"{'Task':<25} | " + " | ".join(f"{e:>10}" for e in encoders) + " | Recommended"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Rows
    for task, data in table.items():
        scores = [f"{data.get(e, 0):>10.3f}" for e in encoders]
        rec = data.get("recommended", "N/A")
        row = f"{task:<25} | " + " | ".join(scores) + f" | {rec}"
        print(row)

    print("=" * len(header))
