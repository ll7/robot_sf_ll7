"""
Statistical analyzer for research reporting (User Story 1)
Implements: paired_t_test, cohen_d, evaluate_hypothesis
"""

from typing import Any, Optional

import numpy as np
from scipy import stats


def paired_t_test(x: list[float], y: list[float]) -> dict[str, Any]:
    """
    Perform a paired t-test between two related samples.
    Returns dict with t_stat, p_value, n.
    """
    if len(x) != len(y) or len(x) < 2:
        return {"t_stat": None, "p_value": None, "n": min(len(x), len(y))}
    t_stat, p_value = stats.ttest_rel(x, y)
    return {"t_stat": float(t_stat), "p_value": float(p_value), "n": len(x)}


def cohen_d(x: list[float], y: list[float]) -> Optional[float]:
    """
    Compute Cohen's d effect size for paired samples.
    """
    if len(x) != len(y) or len(x) < 2:
        return None
    diff = np.array(x) - np.array(y)
    return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else None


def evaluate_hypothesis(
    baseline: list[float], pretrained: list[float], threshold: float = 40.0
) -> dict[str, Any]:
    """
    Evaluate the hypothesis that pre-training reduces PPO timesteps by >= threshold percent.
    Returns a dict matching HypothesisDefinition in the data model.
    """
    if not baseline or not pretrained or min(len(baseline), len(pretrained)) < 1:
        return {"decision": "INCOMPLETE", "note": "Insufficient data for hypothesis evaluation"}
    mean_base = float(np.mean(baseline))
    mean_pre = float(np.mean(pretrained))
    improvement_pct = 100 * (mean_base - mean_pre) / mean_base if mean_base > 0 else None
    if improvement_pct is None:
        return {"decision": "INCOMPLETE", "note": "Baseline mean is zero"}
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
    }
