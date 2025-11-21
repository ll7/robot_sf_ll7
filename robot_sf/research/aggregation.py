"""
Metric aggregation engine for research reporting (User Story 1)
Implements: aggregate_metrics, bootstrap_ci, export_metrics_json, export_metrics_csv
"""

from typing import Any

import numpy as np
import pandas as pd


def aggregate_metrics(
    metric_records: list[dict[str, Any]],
    group_by: str = "policy_type",
    ci_samples: int = 1000,
    ci_confidence: float = 0.95,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Aggregates metrics across seeds and variants, computes mean, median, p95, std, and bootstrap CIs.
    Args:
        metric_records: List of per-seed metric dicts (see MetricRecord in data model)
        group_by: Field to group by (e.g., 'policy_type', 'variant_id')
        ci_samples: Number of bootstrap samples
        ci_confidence: Confidence level for CIs
        seed: Optional random seed for reproducibility
    Returns:
        List of aggregated metric dicts (see AggregatedMetrics in data model)
    """
    # Group by condition and metric
    df = pd.DataFrame(metric_records)
    if df.empty or group_by not in df.columns:
        return []
    results = []
    np.random.seed(seed)
    # Only aggregate numeric columns
    metric_cols = [
        col
        for col in df.columns
        if col not in (group_by, "seed", "variant_id", "policy_type")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    for condition, group in df.groupby(group_by):
        for metric_name in metric_cols:
            values = group[metric_name].dropna().to_numpy()
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            median = float(np.median(values))
            p95 = float(np.percentile(values, 95))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci_low, ci_high = None, None
            if len(values) > 1:
                boot_samples = [
                    np.mean(np.random.choice(values, size=len(values), replace=True))
                    for _ in range(ci_samples)
                ]
                alpha = 1 - ci_confidence
                ci_low = float(np.percentile(boot_samples, 100 * alpha / 2))
                ci_high = float(np.percentile(boot_samples, 100 * (1 - alpha / 2)))
            results.append(
                {
                    "metric_name": metric_name,
                    "condition": condition,
                    "mean": mean,
                    "median": median,
                    "p95": p95,
                    "std": std,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_confidence": ci_confidence,
                    "sample_size": len(values),
                    "effect_size": None,  # To be filled in by statistics module if needed
                }
            )
    return results


def export_metrics_json(aggregated_metrics: list[dict[str, Any]], path: str) -> None:
    """Export aggregated metrics to JSON file."""
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": "1.0.0", "metrics": aggregated_metrics}, f, indent=2)


def export_metrics_csv(aggregated_metrics: list[dict[str, Any]], path: str) -> None:
    """Export aggregated metrics to CSV file."""
    df = pd.DataFrame(aggregated_metrics)
    df.to_csv(path, index=False)


def bootstrap_ci(
    values: list[float],
    ci_samples: int = 1000,
    ci_confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float | None, float | None]:
    """Compute bootstrap confidence interval for a list of values."""
    np.random.seed(seed)
    if len(values) < 2:
        return None, None
    boot_samples = [
        np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(ci_samples)
    ]
    alpha = 1 - ci_confidence
    ci_low = float(np.percentile(boot_samples, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_samples, 100 * (1 - alpha / 2)))
    return ci_low, ci_high
