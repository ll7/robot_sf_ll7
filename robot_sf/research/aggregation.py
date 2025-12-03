"""Aggregation utilities for research reporting.

Functions:
- ``aggregate_metrics``: compute mean/median/p95/std and bootstrap CIs per metric & condition.
- ``bootstrap_ci`` helper (internal) for non-parametric confidence intervals.
- ``extract_seed_metrics``: parse tracker manifests into per-seed records.
- ``compute_completeness_score``: derive completeness % for multi-seed experiments.
- ``export_metrics_json`` / ``export_metrics_csv``: persistent artifacts for downstream analysis.

Design notes:
All functions are pure (except exports) to keep unit tests deterministic. They are
invoked by ``ReportOrchestrator`` and ablation workflows. Avoid adding large
dependencies or side effects here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Performance: only needed for type checking
    from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robot_sf.research.logging_config import get_logger

logger = get_logger(__name__)


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
    """Export aggregated metrics to JSON file.

    Args:
        aggregated_metrics: List of aggregated metric dictionaries from aggregate_metrics()
        path: Output file path (will be created with parent directories)

    Note:
        Output includes schema_version key for validation compatibility.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": "1.0.0", "metrics": aggregated_metrics}, f, indent=2)


def export_metrics_csv(aggregated_metrics: list[dict[str, Any]], path: str) -> None:
    """Export aggregated metrics to CSV file for external analysis tools.

    Args:
        aggregated_metrics: List of aggregated metric dictionaries from aggregate_metrics()
        path: Output CSV file path

    Note:
        Output is tab-delimited with header row. Compatible with pandas.read_csv().
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(aggregated_metrics)
    df.to_csv(path, index=False, sep="\t")


def bootstrap_ci(
    values: list[float],
    ci_samples: int = 1000,
    ci_confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float | None, float | None]:
    """Compute bootstrap confidence interval for a list of values.

    Args:
        values: Numeric values to compute CI from (must have len >= 2)
        ci_samples: Number of bootstrap resamples (default: 1000)
        ci_confidence: Confidence level (default: 0.95 for 95% CI)
        seed: Optional random seed for reproducibility

    Returns:
        Tuple of (ci_low, ci_high) or (None, None) if insufficient data

    Note:
        Uses percentile method. Returns None for single-value inputs.
    """
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


def _load_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    """Load a manifest that may be JSON or JSONL format.

    For JSONL inputs we take the last non-empty line to reflect the most recent
    tracker record. This supports both single-shot manifests and append-mode logs.

    Args:
        manifest_path: Path to manifest file (.json or .jsonl extension)

    Returns:
        Parsed manifest payload as dictionary

    Raises:
        ValueError: If file is empty or unparseable
        json.JSONDecodeError: If JSON is malformed
    """

    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix == ".jsonl":
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Empty manifest file: {manifest_path}")
        return json.loads(lines[-1])
    return json.loads(text)


def extract_seed_metrics(
    manifest_paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract per-seed metrics from tracker manifests.

    Args:
        manifest_paths: Paths to manifest files. Supports JSON or JSONL files.

    Returns:
        Tuple of (metric_records, failures). Failures contain seed/policy metadata
        and the reason parsing was skipped.
    """

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for raw_path in manifest_paths:
        manifest_path = Path(raw_path)
        try:
            payload = _load_manifest_payload(manifest_path)
            seed = int(payload.get("seed")) if payload.get("seed") is not None else None
            metrics = payload.get("metrics") or payload.get("summary", {}).get("metrics")
            if metrics is None:
                raise KeyError("metrics not found")

            policy_type = payload.get("policy_type", "unknown")
            record: dict[str, Any] = {
                "seed": seed,
                "policy_type": policy_type,
                "variant_id": payload.get("variant_id"),
            }

            if "success_rate" in metrics:
                record["success_rate"] = float(metrics["success_rate"])
            if "collision_rate" in metrics:
                record["collision_rate"] = float(metrics["collision_rate"])

            timesteps = metrics.get("timesteps_to_convergence")
            timesteps = timesteps or metrics.get("avg_timesteps")
            timesteps = timesteps or metrics.get("total_timesteps")
            if timesteps is not None:
                record["timesteps_to_convergence"] = float(timesteps)

            if "final_reward_mean" in metrics:
                record["final_reward_mean"] = float(metrics["final_reward_mean"])
            if "run_duration_seconds" in metrics:
                record["run_duration_seconds"] = float(metrics["run_duration_seconds"])

            if len(record.keys() - {"seed", "policy_type", "variant_id"}) == 0:
                raise ValueError("no numeric metrics found")

            records.append(record)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failure = {
                "seed": payload.get("seed") if "payload" in locals() else None,
                "policy_type": payload.get("policy_type") if "payload" in locals() else None,
                "path": str(manifest_path),
                "reason": str(exc),
            }
            failures.append(failure)
            logger.warning(
                "Skipping manifest due to parse failure",
                seed=failure["seed"],
                policy_type=failure["policy_type"],
                path=failure["path"],
                error=failure["reason"],
            )

    return records, failures


def compute_completeness_score(
    expected_seeds: Sequence[int | str],
    completed_seeds: Sequence[int | str],
    *,
    failed_seeds: Sequence[int | str] | None = None,
) -> dict[str, Any]:
    """Compute completeness as percentage of expected seeds that completed.

    Args:
        expected_seeds: Declared/target seed list.
        completed_seeds: Seeds with usable metrics.
        failed_seeds: Optional seeds that explicitly failed (logged separately).

    Returns:
        Dict with score (0-100), completed/missing counts, and seed lists.
    """

    def _seed_sort_key(value: str) -> tuple[int, str]:
        """TODO docstring. Document this function.

        Args:
            value: TODO docstring.

        Returns:
            TODO docstring.
        """
        try:
            return (0, str(int(value)))
        except ValueError:
            return (1, value)

    expected_set = {str(seed) for seed in expected_seeds}
    completed_set = {str(seed) for seed in completed_seeds} & expected_set
    failed_set = {str(seed) for seed in failed_seeds} if failed_seeds else set()
    missing_set = expected_set - completed_set - failed_set

    score = 0.0
    if expected_set:
        score = round(len(completed_set) / len(expected_set) * 100, 1)

    if not missing_set and not failed_set:
        status = "PASS"
    elif completed_set:
        status = "PARTIAL"
    else:
        status = "FAIL"

    return {
        "score": score,
        "expected": len(expected_set),
        "completed": len(completed_set),
        "missing_seeds": sorted(missing_set, key=_seed_sort_key),
        "failed_seeds": sorted(failed_set, key=_seed_sort_key) if failed_set else [],
        "status": status,
    }
