"""Conservative Robot-SF vs CARLA oracle replay metric comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_PARITY_METRICS = (
    "success",
    "collision",
    "ttc_min_s",
    "min_distance_m",
    "comfort",
    "jerk",
    "curvature",
    "intervention_rate",
    "snqi",
)
"""Trajectory-level metrics considered by the first CARLA parity adapter."""


DEGRADED_MODES = {"fallback", "degraded", "not_available", "not-available", "failed"}
"""Replay modes that must not be treated as parity evidence."""


@dataclass(frozen=True)
class MetricParityRow:
    """Comparison status for one Robot-SF vs CARLA metric."""

    metric: str
    status: str
    robot_sf_value: Any = None
    carla_value: Any = None
    delta: float | None = None
    reason: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible row.

        Returns
        -------
        dict[str, Any]
            Serializable metric comparison row.
        """
        return {
            "metric": self.metric,
            "status": self.status,
            "robot_sf_value": self.robot_sf_value,
            "carla_value": self.carla_value,
            "delta": self.delta,
            "reason": self.reason,
        }


def compare_oracle_replay_metrics(
    robot_sf_record: dict[str, Any],
    carla_record: dict[str, Any],
    *,
    metric_names: Iterable[str] = DEFAULT_PARITY_METRICS,
) -> dict[str, Any]:
    """Compare Robot-SF and CARLA replay metrics without overclaiming parity.

    Returns
    -------
    dict[str, Any]
        Namespaced parity report with comparable, unavailable, match, or mismatch rows.
    """
    carla_mode = str(carla_record.get("mode", carla_record.get("status", "native"))).lower()
    if carla_mode in DEGRADED_MODES:
        return {
            "comparison_schema": "carla_oracle_replay_parity_v1",
            "status": "unavailable",
            "reason": f"CARLA replay mode is not native/comparable: {carla_mode}",
            "metrics": [
                MetricParityRow(
                    metric=name,
                    status="unavailable",
                    reason=f"CARLA replay mode is not native/comparable: {carla_mode}",
                ).to_json_dict()
                for name in metric_names
            ],
        }

    rows = [
        _compare_metric(name, _metrics(robot_sf_record), _metrics(carla_record))
        for name in metric_names
    ]
    comparable_count = sum(1 for row in rows if row.status in {"comparable", "match", "mismatch"})
    return {
        "comparison_schema": "carla_oracle_replay_parity_v1",
        "status": "comparable" if comparable_count else "unavailable",
        "reason": None if comparable_count else "no comparable metric fields were available",
        "metrics": [row.to_json_dict() for row in rows],
    }


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Return the metric mapping from a trajectory/episode record.

    Returns
    -------
    dict[str, Any]
        Metric dictionary from the nested ``metrics`` key, or the record itself.
    """
    metrics = record.get("metrics", record)
    return metrics if isinstance(metrics, dict) else {}


def _compare_metric(
    metric: str,
    robot_metrics: dict[str, Any],
    carla_metrics: dict[str, Any],
) -> MetricParityRow:
    """Compare one metric conservatively.

    Returns
    -------
    MetricParityRow
        Comparison result for one metric, including unavailable reasons when needed.
    """
    if metric not in robot_metrics:
        return MetricParityRow(
            metric=metric, status="unavailable", reason="missing Robot-SF metric"
        )
    if metric not in carla_metrics:
        return MetricParityRow(metric=metric, status="unavailable", reason="missing CARLA metric")

    robot_value = robot_metrics[metric]
    carla_value = carla_metrics[metric]
    if isinstance(robot_value, bool) and isinstance(carla_value, bool):
        return MetricParityRow(
            metric=metric,
            status="match" if robot_value == carla_value else "mismatch",
            robot_sf_value=robot_value,
            carla_value=carla_value,
        )

    if _is_number(robot_value) and _is_number(carla_value):
        return MetricParityRow(
            metric=metric,
            status="comparable",
            robot_sf_value=float(robot_value),
            carla_value=float(carla_value),
            delta=float(carla_value) - float(robot_value),
        )

    return MetricParityRow(
        metric=metric,
        status="unavailable",
        robot_sf_value=robot_value,
        carla_value=carla_value,
        reason="metric values are not numeric or boolean",
    )


def _is_number(value: Any) -> bool:
    """Return whether a value can be compared as a finite scalar number.

    Returns
    -------
    bool
        True for non-boolean ``int`` or ``float`` values.
    """
    return isinstance(value, int | float) and not isinstance(value, bool)
