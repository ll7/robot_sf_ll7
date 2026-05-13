"""Baseline comparison helpers for manual-control sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MetricDirection(StrEnum):
    """Optimization direction for a baseline comparison metric."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class BaselineMetric:
    """Frozen policy-to-beat value for one comparable metric."""

    name: str
    value: float
    direction: MetricDirection
    tolerance: float = 0.0

    def is_beaten_by(self, candidate_value: float) -> bool:
        """Return whether ``candidate_value`` beats this baseline metric."""
        if self.direction == MetricDirection.HIGHER_IS_BETTER:
            return candidate_value > self.value + self.tolerance
        return candidate_value < self.value - self.tolerance


@dataclass(frozen=True)
class PolicyBaseline:
    """Frozen policy-to-beat metadata for a manual-control session."""

    policy_id: str
    source: str
    primary_metric: str
    metrics: dict[str, BaselineMetric]
    metadata: dict[str, Any] = field(default_factory=dict)

    def compare(self, candidate_metrics: dict[str, float]) -> BaselineComparison:
        """Compare human/manual metrics against the frozen baseline.

        Returns
        -------
        BaselineComparison
            Per-metric and primary-metric comparison result.
        """
        if self.primary_metric not in self.metrics:
            raise KeyError(f"primary metric {self.primary_metric!r} is missing from baseline")
        if self.primary_metric not in candidate_metrics:
            raise KeyError(f"candidate metric {self.primary_metric!r} is missing")

        metric_results: dict[str, bool] = {}
        for name, baseline_metric in self.metrics.items():
            if name in candidate_metrics:
                metric_results[name] = baseline_metric.is_beaten_by(candidate_metrics[name])

        return BaselineComparison(
            policy_id=self.policy_id,
            source=self.source,
            primary_metric=self.primary_metric,
            beat_baseline=metric_results[self.primary_metric],
            metric_results=metric_results,
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return JSON-compatible baseline metadata for session manifests.

        Returns
        -------
        dict[str, Any]
            Serializable baseline metadata.
        """
        return {
            "policy_id": self.policy_id,
            "source": self.source,
            "primary_metric": self.primary_metric,
            "metrics": {
                name: {
                    "value": metric.value,
                    "direction": metric.direction.value,
                    "tolerance": metric.tolerance,
                }
                for name, metric in self.metrics.items()
            },
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BaselineComparison:
    """Result of comparing manual-control metrics against a policy baseline."""

    policy_id: str
    source: str
    primary_metric: str
    beat_baseline: bool
    metric_results: dict[str, bool]

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return JSON-compatible comparison metadata for attempt summaries.

        Returns
        -------
        dict[str, Any]
            Serializable baseline-comparison metadata.
        """
        return {
            "policy_id": self.policy_id,
            "source": self.source,
            "primary_metric": self.primary_metric,
            "beat_baseline": self.beat_baseline,
            "metric_results": self.metric_results,
        }
