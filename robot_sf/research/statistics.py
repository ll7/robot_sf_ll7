"""
Statistical analyzer for research reporting (User Story 1)
Implements: paired_t_test, welch_t_test, cohen_d, cohen_d_independent, evaluate_hypothesis,
format_test_results, validate_sample_size, HypothesisEvaluator, compare_to_threshold,
export_hypothesis_json
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def _finite_values(values: list[float]) -> list[float]:
    """Return numeric finite samples, excluding NaN and infinite values."""
    finite: list[float] = []
    for value in values:
        numeric = float(value)
        if math.isfinite(numeric):
            finite.append(numeric)
    return finite


def _finite_pairs(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    """Return pair-aligned finite samples, dropping incomplete pairs."""
    finite_x: list[float] = []
    finite_y: list[float] = []
    for x_value, y_value in zip(x, y, strict=True):
        numeric_x = float(x_value)
        numeric_y = float(y_value)
        if math.isfinite(numeric_x) and math.isfinite(numeric_y):
            finite_x.append(numeric_x)
            finite_y.append(numeric_y)
    return finite_x, finite_y


def paired_t_test(x: list[float], y: list[float]) -> dict[str, Any]:
    """Perform a paired t-test between two related samples.

    Args:
        x: First sample (baseline measurements)
        y: Second sample (treatment measurements, must match length of x)

    Returns:
        Dictionary containing:
            - t_stat: Test statistic (None if insufficient data)
            - p_value: Two-tailed p-value (None if insufficient data)
            - n: Effective finite paired sample size

    Note:
        Requires len(x) == len(y) and at least two finite pairs for valid results.
        Uses scipy.stats.ttest_rel for computation.
    """
    if len(x) != len(y) or len(x) < 2:
        return {"t_stat": None, "p_value": None, "n": min(len(x), len(y))}
    finite_x, finite_y = _finite_pairs(x, y)
    if len(finite_x) < 2:
        return {"t_stat": None, "p_value": None, "n": len(finite_x)}
    t_stat, p_value = stats.ttest_rel(finite_x, finite_y)
    return {"t_stat": float(t_stat), "p_value": float(p_value), "n": len(finite_x)}


def welch_t_test(x: list[float], y: list[float]) -> dict[str, Any]:
    """Perform an independent (Welch's) t-test between two samples.

    Args:
        x: First sample (baseline measurements)
        y: Second sample (candidate measurements)

    Returns:
        Dictionary containing:
            - t_stat: Test statistic (None if insufficient data)
            - p_value: Two-tailed p-value (None if insufficient data)
            - n: Total effective finite sample size
            - n_x / n_y: Individual effective sample sizes for reference

    Note:
        Requires at least two finite values in each sample for valid results.
        Uses scipy.stats.ttest_ind with equal_var=False (Welch's test).
    """

    finite_x = _finite_values(x)
    finite_y = _finite_values(y)
    if len(finite_x) < 2 or len(finite_y) < 2:
        return {
            "t_stat": None,
            "p_value": None,
            "n": len(finite_x) + len(finite_y),
            "n_x": len(finite_x),
            "n_y": len(finite_y),
        }

    t_stat, p_value = stats.ttest_ind(finite_x, finite_y, equal_var=False)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n": len(finite_x) + len(finite_y),
        "n_x": len(finite_x),
        "n_y": len(finite_y),
    }


def cohen_d(x: list[float], y: list[float]) -> float | None:
    """Compute Cohen's d effect size for paired samples.

    Args:
        x: First sample (baseline measurements)
        y: Second sample (treatment measurements, must match length of x)

    Returns:
        Cohen's d value (standardized mean difference) or None if:
            - Samples have different lengths
            - Sample size < 2
            - Standard deviation of differences is zero

    Note:
        Effect size interpretation (Cohen 1988):
            - |d| < 0.2: negligible
            - 0.2 <= |d| < 0.5: small
            - 0.5 <= |d| < 0.8: medium
            - |d| >= 0.8: large
    """
    if len(x) != len(y) or len(x) < 2:
        return None
    finite_x, finite_y = _finite_pairs(x, y)
    if len(finite_x) < 2:
        return None
    diff = np.array(finite_x) - np.array(finite_y)
    return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else None


def cohen_d_independent(x: list[float], y: list[float]) -> float | None:
    """Compute Cohen's d for two independent samples using pooled variance.

    Returns:
        Cohen's d value (standardized mean difference) or None when inputs are
        too small (fewer than two finite values per sample) or when the pooled
        variance is zero.
    """

    finite_x = _finite_values(x)
    finite_y = _finite_values(y)
    if len(finite_x) < 2 or len(finite_y) < 2:
        return None

    x_arr = np.array(finite_x)
    y_arr = np.array(finite_y)
    var_x = np.var(x_arr, ddof=1)
    var_y = np.var(y_arr, ddof=1)
    pooled_denom = np.sqrt(
        ((len(x_arr) - 1) * var_x + (len(y_arr) - 1) * var_y) / (len(x_arr) + len(y_arr) - 2)
    )
    if pooled_denom == 0:
        return None
    return float((np.mean(x_arr) - np.mean(y_arr)) / pooled_denom)


def evaluate_hypothesis(
    baseline: list[float], pretrained: list[float], threshold: float = 40.0
) -> dict[str, Any]:
    """Evaluate the hypothesis that pre-training reduces PPO timesteps by >= threshold percent.

    Args:
        baseline: Baseline timesteps to convergence (list of floats)
        pretrained: Pre-trained timesteps to convergence (list of floats)
        threshold: Minimum improvement percentage for hypothesis to pass (default: 40.0)

    Returns:
        Dictionary matching HypothesisDefinition schema with keys:
            - description: Hypothesis statement
            - metric: Metric name being evaluated
            - threshold_value: Threshold percentage
            - threshold_type: Always "min" for this metric
            - decision: "PASS", "FAIL", or "INCOMPLETE"
            - measured_value: Actual improvement percentage (may be None)
            - note: Explanation of decision

    Decision Logic:
        - PASS: improvement_pct >= threshold
        - FAIL: improvement_pct < threshold or negative (degradation)
        - INCOMPLETE: Insufficient finite data or zero baseline mean
    """
    finite_baseline = _finite_values(baseline)
    finite_pretrained = _finite_values(pretrained)
    sample_counts = {
        "n_baseline": len(finite_baseline),
        "n_pretrained": len(finite_pretrained),
    }
    if not finite_baseline or not finite_pretrained:
        return {
            "decision": "INCOMPLETE",
            "note": "Insufficient finite data for hypothesis evaluation",
            **sample_counts,
        }
    mean_base = float(np.mean(finite_baseline))
    mean_pre = float(np.mean(finite_pretrained))
    improvement_pct = 100 * (mean_base - mean_pre) / mean_base if mean_base > 0 else None
    if improvement_pct is None:
        return {"decision": "INCOMPLETE", "note": "Baseline mean is zero", **sample_counts}
    if improvement_pct >= threshold:
        decision = "PASS"
        note = f"Improvement {improvement_pct:.1f}% >= threshold {threshold}%"
    elif improvement_pct < 0:
        decision = "FAIL"
        note = "Pre-training degraded performance"
    else:
        decision = "FAIL"
        note = f"Improvement {improvement_pct:.1f}% < threshold {threshold}%"
    return {
        "description": f"Pre-training reduces timesteps by >= {threshold}%",
        "metric": "timesteps_to_convergence",
        "threshold_value": threshold,
        "threshold_type": "min",
        "decision": decision,
        "measured_value": improvement_pct,
        "note": note,
        **sample_counts,
    }


def validate_sample_size(x: list[float], y: list[float]) -> dict[str, Any]:
    """Validate that samples are suitable for paired tests.

    Returns:
        Dictionary with a 'valid' flag and auxiliary fields. When invalid,
        includes a 'reason' key (e.g., 'mismatched_lengths',
        'insufficient_finite_samples'). Criteria: lengths equal and at least
        two finite pairs.
    """
    if len(x) != len(y):
        return {"valid": False, "reason": "mismatched_lengths", "n_x": len(x), "n_y": len(y)}
    finite_x, _ = _finite_pairs(x, y)
    if len(finite_x) < 2:
        return {"valid": False, "reason": "insufficient_finite_samples", "n": len(finite_x)}
    return {"valid": True, "n": len(finite_x)}


def format_test_results(
    t_test: dict[str, Any], effect_size: float | None, alpha: float = 0.05
) -> dict[str, Any]:
    """Format statistical test results into standardized structure.

    Args:
        t_test: Result dict from paired_t_test
        effect_size: Cohen's d value (may be None)
        alpha: Significance threshold

    Returns:
        Dict with standardized keys: t_stat, p_value, effect_size, significance, interpretation
    """
    t_stat = t_test.get("t_stat")
    p_value = t_test.get("p_value")
    n = t_test.get("n")
    if t_stat is None or p_value is None:
        return {
            "t_stat": None,
            "p_value": None,
            "effect_size": effect_size,
            "significance": "N/A",
            "n": n,
            "interpretation": "Insufficient data for statistical test",
        }
    significance = "significant" if p_value < alpha else "not_significant"
    # Rough interpretation of Cohen's d (Cohen 1988)
    if effect_size is None:
        eff_label = "effect_size_unavailable"
    elif abs(effect_size) < 0.2:
        eff_label = "negligible"
    elif abs(effect_size) < 0.5:
        eff_label = "small"
    elif abs(effect_size) < 0.8:
        eff_label = "medium"
    else:
        eff_label = "large"
    interpretation = (
        f"p={p_value:.3f} ({significance}), effect size={effect_size:.2f} ({eff_label}), n={n}"
        if effect_size is not None
        else f"p={p_value:.3f} ({significance}), n={n}, effect size unavailable"
    )
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "effect_size": effect_size,
        "significance": significance,
        "effect_size_label": eff_label,
        "n": n,
        "interpretation": interpretation,
    }


def compare_to_threshold(
    baseline: list[float], treatment: list[float], threshold: float
) -> dict[str, Any]:
    """Compute improvement percentage and compare to threshold.

    Returns:
        Dictionary with 'improvement_pct', 'decision' (PASS/FAIL/INCOMPLETE),
        summary statistics such as baseline/treatment means, and effective
        finite sample counts.
    """
    finite_baseline = _finite_values(baseline)
    finite_treatment = _finite_values(treatment)
    sample_counts = {
        "n_baseline": len(finite_baseline),
        "n_treatment": len(finite_treatment),
    }
    if not finite_baseline or not finite_treatment:
        return {
            "decision": "INCOMPLETE",
            "improvement_pct": None,
            "threshold": threshold,
            **sample_counts,
        }
    mean_base = float(np.mean(finite_baseline))
    mean_treat = float(np.mean(finite_treatment))
    if mean_base <= 0:
        return {
            "decision": "INCOMPLETE",
            "improvement_pct": None,
            "threshold": threshold,
            **sample_counts,
        }
    improvement_pct = 100.0 * (mean_base - mean_treat) / mean_base
    # Decision: PASS if meets threshold, otherwise FAIL
    decision = "PASS" if improvement_pct >= threshold else "FAIL"
    return {
        "decision": decision,
        "improvement_pct": improvement_pct,
        "threshold": threshold,
        "baseline_mean": mean_base,
        "treatment_mean": mean_treat,
        **sample_counts,
    }


class HypothesisEvaluator:
    """Evaluate hypothesis results for ablation variants.

    Hypothesis: pre-training reduces timesteps by >= threshold percent.
    """

    def __init__(self, threshold: float = 40.0):
        """Initialize the evaluator with a pass threshold.

        Args:
            threshold: Minimum improvement percentage required to pass the
                hypothesis (default: 40.0).
        """
        self.threshold = threshold

    def evaluate_variant(
        self, baseline: list[float], pretrained: list[float], variant_id: str
    ) -> dict[str, Any]:
        """Evaluate a single variant and attach its identifier.

        Args:
            baseline: Baseline timesteps to convergence.
            pretrained: Pre-trained timesteps to convergence.
            variant_id: Identifier of the evaluated variant.

        Returns:
            Dictionary with threshold comparison results (decision,
            improvement percentage, summary stats) including the
            provided `variant_id` key.
        """
        result = compare_to_threshold(baseline, pretrained, self.threshold)
        result["variant_id"] = variant_id
        return result

    def export_hypothesis_json(self, path: str | Path, results: list[dict[str, Any]]) -> Path:
        """Export hypothesis evaluation results as a JSON file.

        Args:
            path: Output file path.
            results: List of per-variant hypothesis results to serialize.

        Returns:
            Path to the written JSON file (ensured parent directories created).
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "1.0.0", "variants": results}, f, indent=2)
        return out_path


def export_hypothesis_json(path: str | Path, results: list[dict[str, Any]]) -> Path:
    """Helper to export hypothesis results outside the evaluator context.

    Args:
        path: Output file path.
        results: List of per-variant hypothesis results to serialize.

    Returns:
        Path to the written JSON file.
    """
    evaluator = HypothesisEvaluator()  # threshold unused for export
    return evaluator.export_hypothesis_json(path, results)
