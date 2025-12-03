"""Rule-based recommendation engine for telemetry snapshots."""

from __future__ import annotations

import contextlib
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from robot_sf.telemetry.models import (
    PerformanceRecommendation,
    RecommendationSeverity,
    TelemetrySnapshot,
)

if TYPE_CHECKING:  # pragma: no cover - hints only
    from collections.abc import Iterable

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil not installed
    psutil = None  # type: ignore


@dataclass(slots=True)
class RecommendationRules:
    """Thresholds used by the recommendation engine."""

    throughput_baseline: float = 0.01  # ~1 step / 100s (pipeline granularity)
    throughput_warning_ratio: float = 0.6
    throughput_critical_ratio: float = 0.3
    cpu_process_warning: float = 85.0
    cpu_process_critical: float = 95.0
    cpu_system_warning: float = 90.0
    memory_warning_ratio: float = 0.80
    memory_critical_ratio: float = 0.95
    memory_min_absolute_mb: float = 1024.0


class RecommendationEngine:
    """Analyze telemetry snapshots and emit structured recommendations."""

    def __init__(
        self,
        *,
        rules: RecommendationRules | None = None,
        total_memory_mb: float | None = None,
    ) -> None:
        """TODO docstring. Document this function.

        Args:
            rules: TODO docstring.
            total_memory_mb: TODO docstring.
        """
        self._rules = rules or RecommendationRules()
        self._snapshots: list[TelemetrySnapshot] = []
        self._total_memory_mb = total_memory_mb or self._detect_total_memory()

    def observe_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        """Record a telemetry snapshot for later analysis."""

        self._snapshots.append(snapshot)

    def summary(self) -> dict[str, float | int | None]:
        """Return aggregated telemetry statistics for manifest summaries."""

        cpu_process_avg = _avg(snapshot.cpu_percent_process for snapshot in self._snapshots)
        cpu_system_avg = _avg(snapshot.cpu_percent_system for snapshot in self._snapshots)
        steps_avg = _avg(snapshot.steps_per_sec for snapshot in self._snapshots)
        memory_max = _max(snapshot.memory_rss_mb for snapshot in self._snapshots)
        return {
            "telemetry_samples": len(self._snapshots),
            "avg_cpu_percent_process": cpu_process_avg,
            "avg_cpu_percent_system": cpu_system_avg,
            "max_memory_rss_mb": memory_max,
            "avg_steps_per_sec": steps_avg,
        }

    def generate_recommendations(self) -> list[PerformanceRecommendation]:
        """Evaluate all rules and return triggered recommendations.

        Returns:
            list[PerformanceRecommendation]: All recommendations triggered by the
            currently observed snapshots. Returns an empty list when no rules
            are met or no snapshots have been observed.
        """

        if not self._snapshots:
            return []
        recommendations: list[PerformanceRecommendation] = []
        recommendations.extend(self._evaluate_throughput())
        recommendations.extend(self._evaluate_cpu())
        recommendations.extend(self._evaluate_memory())
        recommendations.extend(self._evaluate_gpu_idle())
        return recommendations

    def _evaluate_throughput(self) -> list[PerformanceRecommendation]:
        """Check average throughput against baseline and emit recommendations.

        Returns:
            list[PerformanceRecommendation]: Recommendations when throughput is low.
        """
        rules = self._rules
        baseline = rules.throughput_baseline
        average = _avg(snapshot.steps_per_sec for snapshot in self._snapshots)
        if baseline <= 0 or average is None:
            return []
        ratio = average / baseline
        severity: RecommendationSeverity | None = None
        if ratio <= rules.throughput_critical_ratio:
            severity = RecommendationSeverity.CRITICAL
        elif ratio <= rules.throughput_warning_ratio:
            severity = RecommendationSeverity.WARNING
        if severity is None:
            return []
        return [
            PerformanceRecommendation(
                trigger="low_throughput",
                severity=severity,
                message=(
                    "Pipeline throughput fell below the documented baseline."
                    " Consider reducing optional steps or enabling demo mode."
                ),
                suggested_actions=(
                    "Lower rollout workloads (e.g., demo mode or fewer episodes)",
                    "Verify simulator backend and hardware acceleration",
                ),
                evidence={
                    "avg_steps_per_sec": round(average, 5),
                    "baseline": baseline,
                    "ratio": round(ratio, 3),
                },
                timestamp_ms=self._last_timestamp(),
            )
        ]

    def _evaluate_cpu(self) -> list[PerformanceRecommendation]:
        """Evaluate CPU metrics and emit warnings/criticals if saturated.

        Returns:
            list[PerformanceRecommendation]: CPU-related recommendations.
        """
        rules = self._rules
        process_max = _max(snapshot.cpu_percent_process for snapshot in self._snapshots)
        system_max = _max(snapshot.cpu_percent_system for snapshot in self._snapshots)
        metrics = [value for value in (process_max, system_max) if value is not None]
        if not metrics:
            return []
        highest = max(metrics)
        severity: RecommendationSeverity | None = None
        if highest >= rules.cpu_process_critical or highest >= rules.cpu_system_warning + 5:
            severity = RecommendationSeverity.CRITICAL
        elif highest >= min(rules.cpu_process_warning, rules.cpu_system_warning):
            severity = RecommendationSeverity.WARNING
        if severity is None:
            return []
        return [
            PerformanceRecommendation(
                trigger="cpu_saturation",
                severity=severity,
                message="CPU utilization remained high during the run.",
                suggested_actions=(
                    "Reduce concurrent training runs on this host",
                    "Lower --num-envs or trajectory counts to decrease load",
                ),
                evidence={
                    "max_cpu_percent_process": process_max,
                    "max_cpu_percent_system": system_max,
                },
                timestamp_ms=self._last_timestamp(),
            )
        ]

    def _evaluate_memory(self) -> list[PerformanceRecommendation]:
        """Evaluate memory usage against absolute and ratio thresholds.

        Returns:
            list[PerformanceRecommendation]: Memory pressure recommendations.
        """
        memory_max = _max(snapshot.memory_rss_mb for snapshot in self._snapshots)
        if memory_max is None:
            return []
        rules = self._rules
        total_memory = self._total_memory_mb
        severity: RecommendationSeverity | None = None
        if total_memory:
            ratio = memory_max / total_memory
            if ratio >= rules.memory_critical_ratio:
                severity = RecommendationSeverity.CRITICAL
            elif ratio >= rules.memory_warning_ratio:
                severity = RecommendationSeverity.WARNING
        elif memory_max >= rules.memory_min_absolute_mb:
            severity = RecommendationSeverity.WARNING
        if severity is None:
            return []
        evidence = {"max_memory_rss_mb": round(memory_max, 1)}
        if total_memory:
            evidence["system_memory_mb"] = round(total_memory, 1)
        return [
            PerformanceRecommendation(
                trigger="memory_pressure",
                severity=severity,
                message="Process memory usage is nearing system capacity.",
                suggested_actions=(
                    "Reduce dataset size or number of buffered episodes",
                    "Close other memory-intensive applications",
                ),
                evidence=evidence,
                timestamp_ms=self._last_timestamp(),
            )
        ]

    def _evaluate_gpu_idle(self) -> list[PerformanceRecommendation]:
        """Detect idle GPU while CPU remains busy and emit an info hint.

        Returns:
            list[PerformanceRecommendation]: GPU idle recommendation if applicable.
        """
        gpu_utils = [
            snapshot.gpu_util_percent
            for snapshot in self._snapshots
            if snapshot.gpu_util_percent is not None
        ]
        cpu_process_avg = _avg(snapshot.cpu_percent_process for snapshot in self._snapshots)
        if not gpu_utils or cpu_process_avg is None:
            return []
        avg_gpu = statistics.mean(gpu_utils)
        if avg_gpu >= 10 or cpu_process_avg <= 50:
            return []
        return [
            PerformanceRecommendation(
                trigger="gpu_idle",
                severity=RecommendationSeverity.INFO,
                message="GPU metrics remained idle while CPU was saturated.",
                suggested_actions=(
                    "Verify CUDA is available and the correct backend is selected",
                    "Ensure training scripts request GPU acceleration",
                ),
                evidence={
                    "avg_gpu_util_percent": round(avg_gpu, 2),
                    "avg_cpu_percent_process": round(cpu_process_avg, 2),
                },
                timestamp_ms=self._last_timestamp(),
            )
        ]

    def _last_timestamp(self) -> int:
        """Return the last snapshot timestamp or current time in milliseconds.

        Returns:
            int: Milliseconds since epoch.
        """
        last = self._snapshots[-1]
        if last.timestamp_ms:
            return last.timestamp_ms
        return int(datetime.now(UTC).timestamp() * 1000)

    @staticmethod
    def _detect_total_memory() -> float | None:
        """Detect total system memory in MB if psutil is available.

        Returns:
            float | None: Total memory (MB) or None when unavailable.
        """
        if psutil is None:  # pragma: no cover - psutil missing
            return None
        with contextlib.suppress(psutil.Error):  # type: ignore[attr-defined]
            virtual = psutil.virtual_memory()
            return float(virtual.total) / (1024**2)
        return None


def _avg(values: Iterable[float | None]) -> float | None:
    """Average of non-None values or None when empty.

    Args:
        values: Iterable containing optional floats.

    Returns:
        float | None: Mean of values or None.
    """
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(statistics.fmean(filtered))


def _max(values: Iterable[float | None]) -> float | None:
    """Maximum of non-None values or None when empty.

    Args:
        values: Iterable containing optional floats.

    Returns:
        float | None: Max of values or None.
    """
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(max(filtered))
