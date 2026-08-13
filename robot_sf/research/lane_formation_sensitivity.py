"""Diagnostic sensitivity surface for the lane-formation non-result.

This module reuses the canonical emergent-phenomena Social Force Model (SFM)
harness to run a bounded bidirectional-corridor surface over geometry,
population, and duration.  It is deliberately diagnostic-only: it does not
change released defaults, metric semantics, or benchmark/paper claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.research.emergent_phenomena import (
    LITERATURE_CALIBRATION,
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioConfig,
    derive_phenomenon_verdict,
    lane_purity,
    lane_segregation_index,
    released_default_config,
    run_scenario,
    simulator_config_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pysocialforce.config import SimulatorConfig

    from robot_sf.research.emergent_phenomena import SpeedCalibration

__all__ = [
    "DEFAULT_HALF_WIDTHS_M",
    "DEFAULT_LENGTHS_M",
    "DEFAULT_PEDESTRIAN_COUNTS",
    "DEFAULT_SEEDS",
    "DEFAULT_STEPS",
    "DEFAULT_THRESHOLD_GRID",
    "DiagnosticRunStatus",
    "ThresholdSpec",
    "build_corridor_surface",
    "build_threshold_grid",
    "diagnostic_manifest",
    "run_lane_formation_sensitivity",
    "summarize_sensitivity_rows",
]

DEFAULT_SEEDS: tuple[int, ...] = (6962, 6963)
DEFAULT_LENGTHS_M: tuple[float, ...] = (16.0, 24.0)
DEFAULT_HALF_WIDTHS_M: tuple[float, ...] = (1.75, 2.5)
DEFAULT_PEDESTRIAN_COUNTS: tuple[int, ...] = (16, 24)
DEFAULT_STEPS: tuple[int, ...] = (200, 400)

DEFAULT_THRESHOLD_GRID: dict[str, tuple[float, ...]] = {
    "lane_segregation_index": (0.15, 0.3, 0.5),
    "lane_purity": (0.4, 0.6, 0.8),
}
SUPPORTED_METRICS = frozenset(DEFAULT_THRESHOLD_GRID)


@dataclass(frozen=True)
class ThresholdSpec:
    """One explicit measurement threshold to apply to diagnostic rows."""

    metric: str
    threshold: float
    label: str


@dataclass(frozen=True)
class DiagnosticRunStatus:
    """Execution status contract for one diagnostic cell."""

    status: str
    execution_mode: str
    reason: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable execution-status mapping.

        Returns:
            Plain mapping with status, execution mode, and optional reason.
        """
        return {
            "status": self.status,
            "execution_mode": self.execution_mode,
            "reason": self.reason,
        }


def _validate_numeric_axis(name: str, values: Sequence[float], *, positive: bool = True) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    for value in values:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{name} contains non-finite value {value!r}")
        if positive and numeric <= 0.0:
            raise ValueError(f"{name} values must be positive; got {value!r}")


def _validate_int_axis(name: str, values: Sequence[int]) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} values must be positive integers; got {value!r}")


def build_threshold_grid(
    thresholds: Mapping[str, Sequence[float]] | None = None,
) -> list[ThresholdSpec]:
    """Build explicit threshold calibration specs.

    Args:
        thresholds: Mapping of metric name to candidate threshold values.
            Defaults to conservative lane-formation diagnostic thresholds.

    Returns:
        Sorted threshold specifications.  Labels are stable and safe to persist.
    """
    source = thresholds if thresholds is not None else DEFAULT_THRESHOLD_GRID
    specs: list[ThresholdSpec] = []
    for metric, values in sorted(source.items()):
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"unsupported threshold metric {metric!r}; expected one of "
                f"{sorted(SUPPORTED_METRICS)}"
            )
        _validate_numeric_axis(f"thresholds[{metric}]", values)
        for value in sorted(float(v) for v in values):
            specs.append(
                ThresholdSpec(
                    metric=metric,
                    threshold=value,
                    label=f"{metric}>={value:g}",
                )
            )
    return specs


def build_corridor_surface(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    lengths_m: Sequence[float] = DEFAULT_LENGTHS_M,
    half_widths_m: Sequence[float] = DEFAULT_HALF_WIDTHS_M,
    pedestrian_counts: Sequence[int] = DEFAULT_PEDESTRIAN_COUNTS,
    steps: Sequence[int] = DEFAULT_STEPS,
) -> list[ScenarioConfig]:
    """Build the bounded bidirectional-corridor scenario surface.

    The axes are intentionally plain ``ScenarioConfig`` fields so downstream
    reviewers can distinguish geometry/population/duration explanations without
    inventing a parallel scenario representation.

    Returns:
        Scenario configurations spanning the Cartesian product of the requested
        sensitivity axes.
    """
    _validate_int_axis("seeds", seeds)
    _validate_numeric_axis("lengths_m", lengths_m)
    _validate_numeric_axis("half_widths_m", half_widths_m)
    _validate_int_axis("pedestrian_counts", pedestrian_counts)
    _validate_int_axis("steps", steps)

    scenarios: list[ScenarioConfig] = []
    for length, half_width, n_pedestrians, n_steps, seed in product(
        lengths_m, half_widths_m, pedestrian_counts, steps, seeds
    ):
        scenarios.append(
            ScenarioConfig(
                name="bidirectional_corridor",
                length=float(length),
                half_width=float(half_width),
                n_pedestrians=int(n_pedestrians),
                seed=int(seed),
                n_steps=int(n_steps),
                extra={
                    "density_peds_per_m2": float(n_pedestrians)
                    / max(1e-9, float(length) * 2.0 * float(half_width)),
                },
            )
        )
    return scenarios


def _threshold_evaluations(
    metrics: Mapping[str, float], threshold_specs: Sequence[ThresholdSpec]
) -> dict[str, dict[str, float | bool | str]]:
    evaluations: dict[str, dict[str, float | bool | str]] = {}
    for spec in threshold_specs:
        value = metrics.get(spec.metric)
        if value is None:
            evaluations[spec.label] = {
                "metric": spec.metric,
                "threshold": spec.threshold,
                "available": False,
                "meets_threshold": False,
                "margin": float("nan"),
            }
            continue
        numeric = float(value)
        evaluations[spec.label] = {
            "metric": spec.metric,
            "threshold": spec.threshold,
            "available": True,
            "meets_threshold": bool(numeric >= spec.threshold),
            "margin": float(numeric - spec.threshold),
        }
    return evaluations


def _scenario_record(
    scenario: ScenarioConfig,
    calibration: SpeedCalibration,
    *,
    threshold_specs: Sequence[ThresholdSpec],
    sim_config: SimulatorConfig | None,
) -> dict[str, Any]:
    """Run one native SFM corridor cell and serialize the diagnostic row.

    Returns:
        One machine-readable diagnostic row for the scenario/calibration cell.
    """
    status = DiagnosticRunStatus(
        status="computed",
        execution_mode="native",
        reason="canonical emergent_phenomena run_scenario completed",
    )
    result = run_scenario(scenario, calibration, sim_config=sim_config)
    metrics = {
        "lane_segregation_index": float(lane_segregation_index(result.trajectory)),
        "lane_purity": float(lane_purity(result.trajectory)),
    }
    # Cross-check the canonical run_scenario order-parameter dispatch without
    # changing its metric semantics.
    for metric_name, metric_value in result.order_parameters.items():
        if metric_name in metrics and not np.isclose(metrics[metric_name], metric_value):
            raise RuntimeError(
                f"Metric mismatch for {metric_name}: direct={metrics[metric_name]!r}, "
                f"run_scenario={metric_value!r}"
            )

    duration_secs = float(scenario.n_steps * result.trajectory.dt)
    row = {
        "record_type": "lane_formation_sensitivity_cell.v1",
        "issue": "robot_sf_ll7#6962",
        "claim_boundary": "diagnostic_only_not_benchmark_or_paper_evidence",
        "scenario": scenario.name,
        "calibration": calibration.name,
        "seed": int(scenario.seed),
        "geometry": {
            "length_m": float(scenario.length),
            "half_width_m": float(scenario.half_width),
            "corridor_width_m": float(2.0 * scenario.half_width),
        },
        "population": {
            "n_pedestrians": int(scenario.n_pedestrians),
            "density_peds_per_m2": float(scenario.extra["density_peds_per_m2"]),
        },
        "duration": {
            "n_steps": int(scenario.n_steps),
            "dt_secs": float(result.trajectory.dt),
            "duration_secs": duration_secs,
        },
        "desired_speed": {
            "calibration": calibration.name,
            "desired_speed_mean_mps": float(calibration.desired_speed_mean),
            "desired_speed_std_mps": float(calibration.desired_speed_std),
            "realized_mean_max_speed_mps": float(result.max_speeds.mean()),
        },
        "metrics": metrics,
        "canonical_verdict": derive_phenomenon_verdict(scenario.name, metrics),
        "threshold_evaluations": _threshold_evaluations(metrics, threshold_specs),
        "execution": status.as_dict(),
    }
    return row


def summarize_sensitivity_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rows by geometry/population/duration/calibration.

    The summary keeps every varied axis visible and reports threshold hit rates
    across seeds, which is the intended calibration support for this diagnostic.

    Returns:
        Aggregate rows keyed by the varied cell axes and calibration.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["calibration"],
            row["geometry"]["length_m"],
            row["geometry"]["half_width_m"],
            row["population"]["n_pedestrians"],
            row["duration"]["n_steps"],
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (calibration, length, half_width, n_pedestrians, n_steps), recs in sorted(groups.items()):
        metric_names = sorted({name for rec in recs for name in rec["metrics"]})
        threshold_labels = sorted({label for rec in recs for label in rec["threshold_evaluations"]})
        summaries.append(
            {
                "record_type": "lane_formation_sensitivity_summary.v1",
                "calibration": calibration,
                "geometry": {
                    "length_m": length,
                    "half_width_m": half_width,
                    "corridor_width_m": float(2.0 * half_width),
                },
                "population": {
                    "n_pedestrians": n_pedestrians,
                    "density_peds_per_m2": float(
                        n_pedestrians / max(1e-9, length * 2.0 * half_width)
                    ),
                },
                "duration": {"n_steps": n_steps},
                "n_seeds": len(recs),
                "seeds": sorted(int(rec["seed"]) for rec in recs),
                "metric_stats": {
                    name: _stats([float(rec["metrics"][name]) for rec in recs])
                    for name in metric_names
                },
                "threshold_hit_rates": {
                    label: float(
                        np.mean(
                            [
                                bool(rec["threshold_evaluations"][label]["meets_threshold"])
                                for rec in recs
                            ]
                        )
                    )
                    for label in threshold_labels
                },
                "execution_status_counts": _counts(
                    str(rec["execution"]["execution_mode"]) for rec in recs
                ),
            }
        )
    return summaries


def _stats(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def run_lane_formation_sensitivity(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    lengths_m: Sequence[float] = DEFAULT_LENGTHS_M,
    half_widths_m: Sequence[float] = DEFAULT_HALF_WIDTHS_M,
    pedestrian_counts: Sequence[int] = DEFAULT_PEDESTRIAN_COUNTS,
    steps: Sequence[int] = DEFAULT_STEPS,
    calibrations: Sequence[SpeedCalibration] | None = None,
    thresholds: Mapping[str, Sequence[float]] | None = None,
    sim_config: SimulatorConfig | None = None,
) -> dict[str, Any]:
    """Run the bounded lane-formation diagnostic surface.

    Returns:
        Payload containing a manifest, per-run rows, and aggregate summaries.

    Raises:
        ValueError: if any axis is empty/invalid.
        RuntimeError: if the canonical metric cross-check fails.
    """
    if sim_config is None:
        sim_config = released_default_config()
    scenarios = build_corridor_surface(
        seeds=seeds,
        lengths_m=lengths_m,
        half_widths_m=half_widths_m,
        pedestrian_counts=pedestrian_counts,
        steps=steps,
    )
    if calibrations is None:
        calibrations = (RELEASED_DEFAULT_CALIBRATION, LITERATURE_CALIBRATION)
    if not calibrations:
        raise ValueError("calibrations must contain at least one value")

    threshold_specs = build_threshold_grid(thresholds)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for calibration in calibrations:
            rows.append(
                _scenario_record(
                    scenario,
                    calibration,
                    threshold_specs=threshold_specs,
                    sim_config=sim_config,
                )
            )

    manifest = diagnostic_manifest(
        rows=rows,
        threshold_specs=threshold_specs,
        sim_config=sim_config,
        axes={
            "seeds": [int(seed) for seed in seeds],
            "lengths_m": [float(value) for value in lengths_m],
            "half_widths_m": [float(value) for value in half_widths_m],
            "pedestrian_counts": [int(value) for value in pedestrian_counts],
            "steps": [int(value) for value in steps],
            "calibrations": [cal.name for cal in calibrations],
        },
    )
    return {
        "schema_version": "lane_formation_sensitivity_diagnostic.v1",
        "manifest": manifest,
        "rows": rows,
        "summary": summarize_sensitivity_rows(rows),
    }


def diagnostic_manifest(
    *,
    rows: Sequence[dict[str, Any]],
    threshold_specs: Sequence[ThresholdSpec],
    sim_config: SimulatorConfig,
    axes: Mapping[str, Any],
) -> dict[str, Any]:
    """Build provenance-safe manifest metadata for a diagnostic run.

    Returns:
        Manifest dict with issue scope, reused harness provenance, axes,
        threshold specs, and explicit execution-mode policy.
    """
    execution_status_counts = _counts(
        f"{row.get('execution', {}).get('execution_mode', 'missing')}:{row.get('execution', {}).get('status', 'missing')}"
        for row in rows
    )
    invalid_execution = _counts(
        f"{row.get('execution', {}).get('execution_mode', 'missing')}:{row.get('execution', {}).get('status', 'missing')}"
        for row in rows
        if row.get("execution", {}).get("execution_mode") != "native"
        or row.get("execution", {}).get("status") != "computed"
    )
    return {
        "schema_version": "lane_formation_sensitivity_manifest.v1",
        "issue": "robot_sf_ll7#6962",
        "purpose": (
            "Bounded diagnostic surface to separate corridor geometry, population, duration, "
            "and measurement-threshold explanations for a lane-formation non-result."
        ),
        "claim_boundary": "diagnostic_only_not_benchmark_or_paper_evidence",
        "released_defaults_changed": False,
        "metric_semantics_changed": False,
        "canonical_harness": "robot_sf.research.emergent_phenomena",
        "reused_entry_points": [
            "ScenarioConfig",
            "run_scenario",
            "lane_segregation_index",
            "lane_purity",
        ],
        "axes": dict(axes),
        "threshold_specs": [
            {"metric": spec.metric, "threshold": spec.threshold, "label": spec.label}
            for spec in threshold_specs
        ],
        "simulator_config": simulator_config_snapshot(sim_config),
        "execution_policy": {
            "allowed_success_modes": ["native"],
            "fallback_degraded_unavailable_policy": "explicit_status_and_fail_closed",
            "execution_status_counts": execution_status_counts,
            "non_native_rows": invalid_execution,
        },
        "reproducibility": {
            "native_only": True,
            "cross_platform_numeric_drift_possible": True,
            "note": (
                "Compare cells within this exact run first; historical evidence generated on a "
                "different platform or architecture is a qualified comparator, not a byte/value "
                "identity guarantee."
            ),
        },
        "row_count": len(rows),
    }
