"""
Statistical testing for InvariantsBench.

Provides publication-grade statistical comparisons between encoders:
- Paired Wilcoxon signed-rank tests (per-task encoder comparisons)
- Bootstrap confidence intervals (95% CI on mean scores)
- Benjamini-Hochberg multiple comparison correction
- Effect size computation (rank-biserial correlation)
- Summary table generation for paper appendix

Usage:
    from src.analysis.statistical_tests import BenchmarkStatistics
    stats = BenchmarkStatistics.from_result_dirs({
        "DualStream": "outputs/i5_i8_validation",
        "ESM2-650M": "outputs/scope_benchmark_v2/esm2_650m",
    })
    table = stats.pairwise_comparison("DualStream", "ESM2-650M", probe="mlp_2")
    stats.print_summary_table(probe="mlp_2")
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Invariant family membership for grouping
TASK_TO_FAMILY = {
    "phi_angle_bins": "I1",
    "psi_angle_bins": "I1",
    "local_curvature": "I1",
    "distance_bins": "I2",
    "contact_prediction": "I2",
    "contact_order": "I4",
    "contact_density": "I3",
    "core_surface": "I3",
    "amino_acid_identity": "I5",
    "hydrophobicity": "I5",
    "radius_of_gyration": "I6",
    "helix_fraction": "I6",
    "sheet_fraction": "I6",
    "coil_fraction": "I6",
    "asphericity": "I6",
    "bfactor_proxy": "I7",
    "backbone_entropy_proxy": "I7",
    "charge_distribution": "I8",
    "hydrogen_bond_count": "I8",
}


@dataclass
class ComparisonResult:
    """Result of a pairwise statistical comparison on one task."""

    task: str
    family: str
    encoder_a: str
    encoder_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    ci_a: Tuple[float, float]
    ci_b: Tuple[float, float]
    delta: float  # mean_b - mean_a
    p_value: float
    p_adjusted: Optional[float]  # after BH correction
    significant: bool  # at alpha=0.05 after correction
    effect_size: float  # rank-biserial r
    winner: str  # encoder name or "tie"


@dataclass
class FamilySummary:
    """Aggregated summary for one invariant family."""

    family: str
    tasks: List[str]
    mean_scores: Dict[str, float]  # encoder -> family-mean
    wins: Dict[str, int]  # encoder -> win count
    significant_wins: Dict[str, int]


class BenchmarkStatistics:
    """
    Statistical analysis engine for InvariantsBench results.

    Loads per-seed results from multiple encoders and provides
    publication-grade statistical comparisons.
    """

    def __init__(self):
        # encoder_name -> probe_name -> task_name -> list of seed scores
        self.data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    @classmethod
    def from_result_dirs(
        cls,
        encoder_dirs: Dict[str, str],
    ) -> "BenchmarkStatistics":
        """
        Load results from benchmark output directories.

        Args:
            encoder_dirs: {encoder_name: path_to_output_dir}
                Each dir should contain seed_0.json, seed_1.json, ...
                OR an aggregated_results.json with raw scores.
        """
        stats = cls()

        for encoder_name, dir_path in encoder_dirs.items():
            stats.data[encoder_name] = {}
            dir_path = Path(dir_path)

            # Try loading from individual seed files first
            seed_files = sorted(dir_path.glob("seed_*.json"))
            if seed_files:
                # Merge per-seed results
                for seed_file in seed_files:
                    with open(seed_file) as f:
                        seed_data = json.load(f)

                    for probe_name, tasks in seed_data.items():
                        if probe_name not in stats.data[encoder_name]:
                            stats.data[encoder_name][probe_name] = {}

                        for task, score in tasks.items():
                            if task not in stats.data[encoder_name][probe_name]:
                                stats.data[encoder_name][probe_name][task] = []
                            stats.data[encoder_name][probe_name][task].append(score)

            # Fallback: aggregated_results.json
            agg_path = dir_path / "aggregated_results.json"
            if agg_path.exists() and not seed_files:
                with open(agg_path) as f:
                    agg = json.load(f)

                results = agg.get("results", agg)
                for probe_name, tasks in results.items():
                    if probe_name in ("encoder", "protocol"):
                        continue
                    if probe_name not in stats.data[encoder_name]:
                        stats.data[encoder_name][probe_name] = {}

                    for task, score_data in tasks.items():
                        if isinstance(score_data, dict) and "raw" in score_data:
                            stats.data[encoder_name][probe_name][task] = score_data[
                                "raw"
                            ]
                        elif isinstance(score_data, (int, float)):
                            stats.data[encoder_name][probe_name][task] = [score_data]

            # Also try comparison_results.json (TriStream benchmark format)
            comp_path = dir_path / "comparison_results.json"
            if comp_path.exists():
                with open(comp_path) as f:
                    comp = json.load(f)

                summary = comp.get("summary", {})
                if encoder_name in summary:
                    for probe_name, tasks in summary[encoder_name].items():
                        if probe_name not in stats.data[encoder_name]:
                            stats.data[encoder_name][probe_name] = {}
                        for task, score_data in tasks.items():
                            if isinstance(score_data, dict) and "raw" in score_data:
                                stats.data[encoder_name][probe_name][task] = (
                                    score_data["raw"]
                                )

        return stats

    @classmethod
    def from_comparison_json(
        cls, path: str
    ) -> "BenchmarkStatistics":
        """Load from a comparison_results.json with multiple encoders."""
        stats = cls()

        with open(path) as f:
            comp = json.load(f)

        summary = comp.get("summary", {})
        for encoder_name, probes in summary.items():
            stats.data[encoder_name] = {}
            for probe_name, tasks in probes.items():
                stats.data[encoder_name][probe_name] = {}
                for task, score_data in tasks.items():
                    if isinstance(score_data, dict) and "raw" in score_data:
                        stats.data[encoder_name][probe_name][task] = score_data["raw"]

        return stats

    @property
    def encoders(self) -> List[str]:
        return list(self.data.keys())

    @property
    def probes(self) -> List[str]:
        all_probes = set()
        for enc_data in self.data.values():
            all_probes.update(enc_data.keys())
        return sorted(all_probes)

    def tasks_for_probe(self, probe: str) -> List[str]:
        """Get all tasks available for a given probe across all encoders."""
        tasks = set()
        for enc_data in self.data.values():
            if probe in enc_data:
                tasks.update(enc_data[probe].keys())
        return sorted(tasks)

    def get_scores(
        self, encoder: str, probe: str, task: str
    ) -> Optional[List[float]]:
        """Get raw seed scores for a specific encoder/probe/task."""
        return self.data.get(encoder, {}).get(probe, {}).get(task)

    # ──────────────────────────────────────────────────────
    # Bootstrap confidence intervals
    # ──────────────────────────────────────────────────────

    @staticmethod
    def bootstrap_ci(
        scores: List[float],
        n_bootstrap: int = 10000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for the mean.

        Returns (lower, upper) bounds at (1-alpha) confidence level.
        """
        rng = np.random.RandomState(seed)
        arr = np.array(scores)
        n = len(arr)

        if n < 2:
            return (arr[0], arr[0]) if n == 1 else (0.0, 0.0)

        boot_means = np.array(
            [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)]
        )

        lower = np.percentile(boot_means, 100 * alpha / 2)
        upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
        return (float(lower), float(upper))

    # ──────────────────────────────────────────────────────
    # Wilcoxon signed-rank test
    # ──────────────────────────────────────────────────────

    @staticmethod
    def wilcoxon_test(
        scores_a: List[float], scores_b: List[float]
    ) -> Tuple[float, float]:
        """
        Paired Wilcoxon signed-rank test.

        Returns (statistic, p_value). Falls back to permutation test
        if scipy is unavailable or sample size is too small.
        """
        a = np.array(scores_a)
        b = np.array(scores_b)

        # Need paired samples of same length
        n = min(len(a), len(b))
        if n < 2:
            return (0.0, 1.0)

        a, b = a[:n], b[:n]
        diffs = b - a

        # Remove zeros
        nonzero = diffs[diffs != 0]
        if len(nonzero) < 2:
            return (0.0, 1.0)

        try:
            from scipy.stats import wilcoxon

            stat, p = wilcoxon(nonzero, alternative="two-sided")
            return (float(stat), float(p))
        except ImportError:
            # Manual approximate p-value via permutation
            return BenchmarkStatistics._permutation_test(nonzero)

    @staticmethod
    def _permutation_test(
        diffs: np.ndarray, n_perms: int = 10000, seed: int = 42
    ) -> Tuple[float, float]:
        """Permutation test fallback when scipy unavailable."""
        rng = np.random.RandomState(seed)
        observed = np.abs(np.mean(diffs))
        n = len(diffs)

        count = 0
        for _ in range(n_perms):
            signs = rng.choice([-1, 1], size=n)
            perm_mean = np.abs(np.mean(diffs * signs))
            if perm_mean >= observed:
                count += 1

        p = (count + 1) / (n_perms + 1)
        return (observed, p)

    # ──────────────────────────────────────────────────────
    # Effect size (rank-biserial correlation)
    # ──────────────────────────────────────────────────────

    @staticmethod
    def rank_biserial(scores_a: List[float], scores_b: List[float]) -> float:
        """
        Rank-biserial correlation effect size for paired samples.

        r = (n_concordant - n_discordant) / n_total
        Range: [-1, 1]. Positive means B > A.
        """
        a = np.array(scores_a)
        b = np.array(scores_b)
        n = min(len(a), len(b))

        if n < 1:
            return 0.0

        diffs = b[:n] - a[:n]
        n_pos = np.sum(diffs > 0)
        n_neg = np.sum(diffs < 0)
        n_total = n_pos + n_neg

        if n_total == 0:
            return 0.0

        return float(n_pos - n_neg) / n_total

    # ──────────────────────────────────────────────────────
    # Benjamini-Hochberg correction
    # ──────────────────────────────────────────────────────

    @staticmethod
    def benjamini_hochberg(
        p_values: List[float], alpha: float = 0.05
    ) -> List[Tuple[float, bool]]:
        """
        Benjamini-Hochberg FDR correction.

        Returns list of (adjusted_p, is_significant) tuples.
        """
        n = len(p_values)
        if n == 0:
            return []

        # Sort by p-value
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])

        adjusted = [0.0] * n
        significant = [False] * n

        # BH procedure
        prev_adj = 1.0
        for rank_idx in range(n - 1, -1, -1):
            orig_idx, p = indexed[rank_idx]
            rank = rank_idx + 1  # 1-indexed rank
            adj_p = min(p * n / rank, prev_adj)
            adj_p = min(adj_p, 1.0)
            adjusted[orig_idx] = adj_p
            significant[orig_idx] = adj_p < alpha
            prev_adj = adj_p

        return list(zip(adjusted, significant))

    # ──────────────────────────────────────────────────────
    # Pairwise comparison
    # ──────────────────────────────────────────────────────

    def pairwise_comparison(
        self,
        encoder_a: str,
        encoder_b: str,
        probe: str = "mlp_2",
        alpha: float = 0.05,
    ) -> List[ComparisonResult]:
        """
        Full pairwise comparison between two encoders on all shared tasks.

        Runs Wilcoxon test per task, then BH correction across tasks.
        """
        tasks = self.tasks_for_probe(probe)
        results = []
        raw_p_values = []

        for task in tasks:
            scores_a = self.get_scores(encoder_a, probe, task)
            scores_b = self.get_scores(encoder_b, probe, task)

            if scores_a is None or scores_b is None:
                continue

            mean_a = np.mean(scores_a)
            mean_b = np.mean(scores_b)
            std_a = np.std(scores_a)
            std_b = np.std(scores_b)
            ci_a = self.bootstrap_ci(scores_a)
            ci_b = self.bootstrap_ci(scores_b)

            _, p_val = self.wilcoxon_test(scores_a, scores_b)
            effect = self.rank_biserial(scores_a, scores_b)

            delta = mean_b - mean_a
            winner = encoder_b if delta > 0 else encoder_a if delta < 0 else "tie"

            results.append(
                ComparisonResult(
                    task=task,
                    family=TASK_TO_FAMILY.get(task, "?"),
                    encoder_a=encoder_a,
                    encoder_b=encoder_b,
                    mean_a=float(mean_a),
                    mean_b=float(mean_b),
                    std_a=float(std_a),
                    std_b=float(std_b),
                    ci_a=ci_a,
                    ci_b=ci_b,
                    delta=float(delta),
                    p_value=p_val,
                    p_adjusted=None,
                    significant=False,
                    effect_size=effect,
                    winner=winner,
                )
            )
            raw_p_values.append(p_val)

        # Apply BH correction
        if raw_p_values:
            corrections = self.benjamini_hochberg(raw_p_values, alpha)
            for i, (adj_p, sig) in enumerate(corrections):
                results[i].p_adjusted = adj_p
                results[i].significant = sig

        return results

    # ──────────────────────────────────────────────────────
    # Family-level aggregation
    # ──────────────────────────────────────────────────────

    def family_summary(
        self,
        probe: str = "mlp_2",
    ) -> Dict[str, FamilySummary]:
        """Compute per-family aggregated scores for all encoders."""
        families: Dict[str, List[str]] = {}
        for task, fam in TASK_TO_FAMILY.items():
            if fam not in families:
                families[fam] = []
            families[fam].append(task)

        summaries = {}
        for fam, tasks in sorted(families.items()):
            available_tasks = [
                t for t in tasks if t in self.tasks_for_probe(probe)
            ]
            if not available_tasks:
                continue

            mean_scores = {}
            for enc in self.encoders:
                task_means = []
                for task in available_tasks:
                    scores = self.get_scores(enc, probe, task)
                    if scores:
                        task_means.append(np.mean(scores))
                if task_means:
                    mean_scores[enc] = float(np.mean(task_means))

            summaries[fam] = FamilySummary(
                family=fam,
                tasks=available_tasks,
                mean_scores=mean_scores,
                wins={},
                significant_wins={},
            )

        return summaries

    # ──────────────────────────────────────────────────────
    # Summary table for paper
    # ──────────────────────────────────────────────────────

    def summary_table(
        self,
        probe: str = "mlp_2",
        format: str = "text",
    ) -> str:
        """
        Generate publication-ready summary table.

        Args:
            probe: Which probe to use
            format: "text", "latex", or "markdown"

        Returns:
            Formatted table string
        """
        tasks = self.tasks_for_probe(probe)
        encoders = self.encoders

        if format == "latex":
            return self._latex_table(tasks, encoders, probe)
        elif format == "markdown":
            return self._markdown_table(tasks, encoders, probe)
        else:
            return self._text_table(tasks, encoders, probe)

    def _get_cell(
        self, encoder: str, probe: str, task: str, best_mean: float
    ) -> str:
        """Format a single table cell with mean +/- std, bold if best."""
        scores = self.get_scores(encoder, probe, task)
        if not scores:
            return "  ---  "

        mean = np.mean(scores)
        std = np.std(scores)
        is_best = abs(mean - best_mean) < 1e-6
        text = f"{mean:.3f}±{std:.3f}"

        if is_best:
            text = f"**{text}**"

        return text

    def _text_table(
        self, tasks: List[str], encoders: List[str], probe: str
    ) -> str:
        """Plain text table."""
        lines = []

        # Header
        enc_width = max(len(e) for e in encoders)
        task_width = max(len(t) for t in tasks)

        header = f"{'Task':<{task_width}} | {'Family':>6}"
        for enc in encoders:
            header += f" | {enc:>{enc_width + 8}}"
        lines.append("=" * len(header))
        lines.append(header)
        lines.append("=" * len(header))

        # Rows
        for task in tasks:
            family = TASK_TO_FAMILY.get(task, "?")

            # Find best mean
            means = {}
            for enc in encoders:
                scores = self.get_scores(enc, probe, task)
                if scores:
                    means[enc] = np.mean(scores)

            best_mean = max(means.values()) if means else 0.0

            row = f"{task:<{task_width}} | {family:>6}"
            for enc in encoders:
                scores = self.get_scores(enc, probe, task)
                if scores:
                    m = np.mean(scores)
                    s = np.std(scores)
                    cell = f"{m:.3f}±{s:.3f}"
                    if abs(m - best_mean) < 1e-6:
                        cell = f"*{cell}*"
                    else:
                        cell = f" {cell} "
                else:
                    cell = "    ---    "
                row += f" | {cell:>{enc_width + 8}}"
            lines.append(row)

        lines.append("=" * len(lines[0]))
        return "\n".join(lines)

    def _markdown_table(
        self, tasks: List[str], encoders: List[str], probe: str
    ) -> str:
        """Markdown table for README/docs."""
        lines = []

        header = f"| Task | Family |"
        sep = f"|------|--------|"
        for enc in encoders:
            header += f" {enc} |"
            sep += "---------|"
        lines.append(header)
        lines.append(sep)

        for task in tasks:
            family = TASK_TO_FAMILY.get(task, "?")

            means = {}
            for enc in encoders:
                scores = self.get_scores(enc, probe, task)
                if scores:
                    means[enc] = np.mean(scores)

            best_mean = max(means.values()) if means else 0.0

            row = f"| {task} | {family} |"
            for enc in encoders:
                scores = self.get_scores(enc, probe, task)
                if scores:
                    m = np.mean(scores)
                    s = np.std(scores)
                    cell = f"{m:.3f}±{s:.3f}"
                    if abs(m - best_mean) < 1e-6:
                        cell = f"**{cell}**"
                else:
                    cell = "---"
                row += f" {cell} |"
            lines.append(row)

        return "\n".join(lines)

    def _latex_table(
        self, tasks: List[str], encoders: List[str], probe: str
    ) -> str:
        """LaTeX table for paper."""
        n_enc = len(encoders)
        lines = []
        lines.append(f"\\begin{{tabular}}{{ll{'c' * n_enc}}}")
        lines.append("\\toprule")

        # Header
        header = "Task & Family"
        for enc in encoders:
            safe_name = enc.replace("_", "\\_")
            header += f" & {safe_name}"
        header += " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        current_family = None
        for task in tasks:
            family = TASK_TO_FAMILY.get(task, "?")

            # Add separator between families
            if current_family and family != current_family:
                lines.append("\\midrule")
            current_family = family

            means = {}
            for enc in encoders:
                scores = self.get_scores(enc, probe, task)
                if scores:
                    means[enc] = np.mean(scores)

            best_mean = max(means.values()) if means else 0.0

            safe_task = task.replace("_", "\\_")
            row = f"{safe_task} & {family}"
            for enc in encoders:
                scores = self.get_scores(enc, probe, task)
                if scores:
                    m = np.mean(scores)
                    s = np.std(scores)
                    cell = f"${m:.3f}_{{\\pm{s:.3f}}}$"
                    if abs(m - best_mean) < 1e-6:
                        cell = f"\\textbf{{{cell}}}"
                else:
                    cell = "---"
                row += f" & {cell}"
            row += " \\\\"
            lines.append(row)

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────
    # Printing helpers
    # ──────────────────────────────────────────────────────

    def print_comparison(
        self,
        encoder_a: str,
        encoder_b: str,
        probe: str = "mlp_2",
    ) -> None:
        """Pretty-print pairwise comparison results."""
        results = self.pairwise_comparison(encoder_a, encoder_b, probe)

        print(f"\n{'='*80}")
        print(f"  {encoder_a} vs {encoder_b}  (probe: {probe})")
        print(f"{'='*80}")
        print(
            f"  {'Task':<25} {'Fam':>4} {'MeanA':>8} {'MeanB':>8} "
            f"{'Delta':>8} {'p_adj':>8} {'Sig':>4} {'Effect':>7} {'Winner'}"
        )
        print("-" * 100)

        a_wins = 0
        b_wins = 0
        sig_a = 0
        sig_b = 0

        for r in results:
            sig_str = "***" if r.significant else ""
            print(
                f"  {r.task:<25} {r.family:>4} {r.mean_a:>8.4f} {r.mean_b:>8.4f} "
                f"{r.delta:>+8.4f} {r.p_adjusted:>8.4f} {sig_str:>4} "
                f"{r.effect_size:>+7.2f} {r.winner}"
            )
            if r.delta > 0:
                b_wins += 1
                if r.significant:
                    sig_b += 1
            elif r.delta < 0:
                a_wins += 1
                if r.significant:
                    sig_a += 1

        print("-" * 100)
        print(f"  {encoder_a} wins: {a_wins} ({sig_a} significant)")
        print(f"  {encoder_b} wins: {b_wins} ({sig_b} significant)")
        print()

    def print_summary_table(self, probe: str = "mlp_2") -> None:
        """Print the summary table to stdout."""
        print(self.summary_table(probe, format="text"))

    # ──────────────────────────────────────────────────────
    # Export helpers
    # ──────────────────────────────────────────────────────

    def export_comparison_json(
        self,
        encoder_a: str,
        encoder_b: str,
        probe: str = "mlp_2",
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export comparison results as JSON-serializable dict."""
        results = self.pairwise_comparison(encoder_a, encoder_b, probe)

        output = {
            "encoder_a": encoder_a,
            "encoder_b": encoder_b,
            "probe": probe,
            "tasks": {},
        }

        for r in results:
            output["tasks"][r.task] = {
                "family": r.family,
                "mean_a": r.mean_a,
                "mean_b": r.mean_b,
                "delta": r.delta,
                "ci_a": list(r.ci_a),
                "ci_b": list(r.ci_b),
                "p_value": r.p_value,
                "p_adjusted": r.p_adjusted,
                "significant": r.significant,
                "effect_size": r.effect_size,
                "winner": r.winner,
            }

        if path:
            with open(path, "w") as f:
                json.dump(output, f, indent=2)

        return output
