"""Threshold-calibration and sustained-flow reference diagnostics.

The issue-6962 sensitivity screen varied the measurement cell but did not
establish whether the lane metrics respond to a known separated-lane control
or whether a finite corridor loses its population before the observation
window.  This module adds two explicitly different references:

* deterministic synthetic mixed/separated trajectories audit the metric and
  sampling contract;
* a native Social Force Model flow recycles agents at the corridor boundaries
  after an explicit warm-up, keeping the population available for a sustained
  observation window.

The separated native condition is a positive control initialized in separated
lanes.  It is not evidence that lanes emerge spontaneously and must never be
reported as such.  Released defaults and metric semantics are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pysocialforce as pysf

from robot_sf.research.emergent_phenomena import (
    LITERATURE_CALIBRATION,
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioConfig,
    SpeedCalibration,
    TrajectoryRecord,
    build_bidirectional_corridor,
    lane_purity,
    lane_segregation_index,
    released_default_config,
    simulator_config_snapshot,
)

__all__ = [
    "DEFAULT_REFERENCE_CALIBRATIONS",
    "DEFAULT_REFERENCE_CONDITIONS",
    "DEFAULT_REFERENCE_SEEDS",
    "ReferenceProtocol",
    "metric_reference_audit",
    "run_native_reference",
    "run_reference_campaign",
    "summarize_reference_rows",
]

SCHEMA_VERSION = "lane_formation_reference_diagnostic.v1"
CLAIM_BOUNDARY = "diagnostic_only_not_benchmark_or_paper_evidence"
DEFAULT_REFERENCE_SEEDS: tuple[int, ...] = (5149, 5150, 5151)
DEFAULT_REFERENCE_CONDITIONS: tuple[str, ...] = (
    "mixed_sustained_flow",
    "separated_lane_control",
)
DEFAULT_REFERENCE_CALIBRATIONS: tuple[SpeedCalibration, ...] = (
    RELEASED_DEFAULT_CALIBRATION,
    LITERATURE_CALIBRATION,
)
DEFAULT_SAMPLING_STRIDES: tuple[int, ...] = (1, 2, 4)
REFERENCE_CLEAR_FLOOR = 0.8


@dataclass(frozen=True)
class ReferenceProtocol:
    """Frozen protocol for one native reference condition.

    ``warmup_steps`` are simulated and discarded.  The next
    ``observation_steps`` are recorded while agents that cross a corridor
    boundary are deterministically reintroduced at the opposite entry.  The
    recycling is a diagnostic control for sustained occupancy, not a change to
    the released simulator or scenario defaults.
    """

    length_m: float = 24.0
    half_width_m: float = 2.5
    n_pedestrians: int = 24
    warmup_steps: int = 100
    observation_steps: int = 200
    recycle_margin_m: float = 0.2
    lane_offset_m: float = 0.85
    entry_y_span_m: float = 1.2

    def validate(self) -> None:
        """Fail closed on a physically or computationally invalid protocol."""
        for name in ("length_m", "half_width_m", "recycle_margin_m", "entry_y_span_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.n_pedestrians < 2 or self.n_pedestrians % 2:
            raise ValueError("n_pedestrians must be an even integer >= 2")
        for name in ("warmup_steps", "observation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.lane_offset_m) or self.lane_offset_m <= 0.0:
            raise ValueError("lane_offset_m must be finite and positive")
        if self.lane_offset_m >= self.half_width_m:
            raise ValueError("lane_offset_m must be inside the corridor")
        if self.entry_y_span_m > self.half_width_m:
            raise ValueError("entry_y_span_m must not exceed half_width_m")

    def as_dict(self) -> dict[str, float | int]:
        """Return stable JSON-ready protocol metadata."""
        self.validate()
        return {
            "length_m": float(self.length_m),
            "half_width_m": float(self.half_width_m),
            "n_pedestrians": int(self.n_pedestrians),
            "warmup_steps": int(self.warmup_steps),
            "observation_steps": int(self.observation_steps),
            "recycle_margin_m": float(self.recycle_margin_m),
            "lane_offset_m": float(self.lane_offset_m),
            "entry_y_span_m": float(self.entry_y_span_m),
        }


def _validate_condition(condition: str) -> None:
    if condition not in DEFAULT_REFERENCE_CONDITIONS:
        raise ValueError(
            f"unsupported reference condition {condition!r}; expected one of "
            f"{list(DEFAULT_REFERENCE_CONDITIONS)}"
        )


def _validate_sampling_strides(strides: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in strides)
    if not values or any(value < 1 for value in values):
        raise ValueError("sampling_strides must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("sampling_strides must be unique")
    return tuple(sorted(values))


def _metric_values(trajectory: TrajectoryRecord) -> dict[str, float]:
    return {
        "lane_segregation_index": float(lane_segregation_index(trajectory)),
        "lane_purity": float(lane_purity(trajectory)),
    }


def _sample_trajectory(trajectory: TrajectoryRecord, stride: int) -> TrajectoryRecord:
    indices = np.arange(0, trajectory.positions.shape[0], stride, dtype=int)
    if indices.size < 2:
        raise ValueError("sampling stride leaves fewer than two trajectory samples")
    return TrajectoryRecord(
        positions=trajectory.positions[indices],
        velocities=trajectory.velocities[indices],
        desired_directions=trajectory.desired_directions,
        times=trajectory.times[indices],
        dt=trajectory.dt * stride,
    )


def _synthetic_reference_trajectory(condition: str, *, steps: int, seed: int) -> TrajectoryRecord:
    """Build a deterministic metric-only mixed/separated reference fixture.

    Returns:
        Synthetic trajectory with paired directional assignments.
    """
    _validate_condition(condition)
    if steps < 4:
        raise ValueError("steps must be at least four")
    n_per_direction = 12
    n = 2 * n_per_direction
    dt = 0.1
    times = np.arange(steps + 1, dtype=float) * dt
    directions = np.concatenate(
        (np.ones(n_per_direction, dtype=float), -np.ones(n_per_direction, dtype=float))
    )
    positions = np.zeros((steps + 1, n, 2), dtype=float)
    positions[:, :, 0] = np.where(directions > 0.0, 1.0, 23.0)[None, :]
    positions[:, :, 0] += directions[None, :] * 0.35 * np.arange(steps + 1)[:, None]

    phase = np.arange(steps + 1, dtype=float)[:, None] * 0.11
    agent_phase = np.arange(n_per_direction, dtype=float)[None, :] * 0.31
    if condition == "separated_lane_control":
        lateral = 0.9 * np.ones((steps + 1, n_per_direction), dtype=float)
        lateral += 0.03 * np.sin(phase + agent_phase + float(seed % 17))
        positions[:, :, 1] = np.concatenate((lateral, -lateral), axis=1)
    else:
        # Each direction receives the same lateral samples.  This is a mixed
        # flow by construction, so the direction/lateral correlation is zero.
        lateral = np.linspace(-1.0, 1.0, n_per_direction)[None, :]
        lateral = lateral + 0.02 * np.sin(phase + agent_phase + float(seed % 17))
        positions[:, :, 1] = np.concatenate((lateral, lateral), axis=1)

    velocities = np.zeros_like(positions)
    velocities[:, :, 0] = directions[None, :] * 0.35
    velocities[:, :, 1] = np.gradient(positions[:, :, 1], dt, axis=0)
    desired_directions = np.column_stack((directions, np.zeros(n, dtype=float)))
    return TrajectoryRecord(
        positions=positions,
        velocities=velocities,
        desired_directions=desired_directions,
        times=times,
        dt=dt,
    )


def _reference_thresholds(metrics: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    thresholds = {
        "lane_segregation_index>=0.15": ("lane_segregation_index", 0.15),
        "lane_segregation_index>=0.5": ("lane_segregation_index", 0.5),
        "lane_purity>=0.4": ("lane_purity", 0.4),
        "lane_purity>=0.8": ("lane_purity", 0.8),
    }
    return {
        label: {
            "metric": metric,
            "threshold": threshold,
            "meets_threshold": bool(metrics[metric] >= threshold),
            "margin": float(metrics[metric] - threshold),
        }
        for label, (metric, threshold) in thresholds.items()
    }


def metric_reference_audit(
    *,
    sampling_strides: tuple[int, ...] | list[int] = DEFAULT_SAMPLING_STRIDES,
    steps: int = 120,
    seed: int = 6969,
) -> dict[str, Any]:
    """Audit metric separation and sampling stability on known fixtures.

    The audit is deliberately synthetic: it tests the measurement contract,
    not the physical model.  A separated fixture must clear the positive
    reference floor while a paired mixed fixture must stay below the clear
    threshold at every requested sampling stride.

    Returns:
        Machine-readable fixture rows and fail-closed audit checks.
    """
    strides = _validate_sampling_strides(sampling_strides)
    records: list[dict[str, Any]] = []
    for condition in DEFAULT_REFERENCE_CONDITIONS:
        trajectory = _synthetic_reference_trajectory(condition, steps=steps, seed=seed)
        for stride in strides:
            metrics = _metric_values(_sample_trajectory(trajectory, stride))
            records.append(
                {
                    "record_type": "lane_metric_reference.v1",
                    "fixture": condition,
                    "sampling_stride": stride,
                    "metrics": metrics,
                    "threshold_evaluations": _reference_thresholds(metrics),
                    "execution": {
                        "status": "computed",
                        "execution_mode": "synthetic_reference",
                    },
                }
            )

    separated = [row for row in records if row["fixture"] == "separated_lane_control"]
    mixed = [row for row in records if row["fixture"] == "mixed_sustained_flow"]
    separated_lsi = [row["metrics"]["lane_segregation_index"] for row in separated]
    mixed_lsi = [row["metrics"]["lane_segregation_index"] for row in mixed]
    separated_purity = [row["metrics"]["lane_purity"] for row in separated]
    mixed_purity = [row["metrics"]["lane_purity"] for row in mixed]
    checks = {
        "separated_lane_control_clears_reference_floor": bool(
            min(separated_lsi) >= REFERENCE_CLEAR_FLOOR
            and min(separated_purity) >= REFERENCE_CLEAR_FLOOR
        ),
        "mixed_flow_stays_below_clear_threshold": bool(
            max(mixed_lsi) < 0.5 and max(mixed_purity) < 0.5
        ),
        "separated_lsi_sampling_spread_le_0_02": bool(
            max(separated_lsi) - min(separated_lsi) <= 0.02
        ),
        "mixed_lsi_sampling_spread_le_0_02": bool(max(mixed_lsi) - min(mixed_lsi) <= 0.02),
    }
    return {
        "schema_version": "lane_metric_reference_audit.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "reference_floor": REFERENCE_CLEAR_FLOOR,
        "clear_threshold": 0.5,
        "sampling_strides": list(strides),
        "records": records,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _apply_separated_lane_initialization(
    state: np.ndarray, desired_directions: np.ndarray, protocol: ReferenceProtocol
) -> None:
    plus = desired_directions[:, 0] > 0.0
    state[plus, 1] = protocol.lane_offset_m
    state[~plus, 1] = -protocol.lane_offset_m
    state[:, 5] = state[:, 1]


def _recycle_exited_agents(
    sim: pysf.Simulator,
    *,
    desired_directions: np.ndarray,
    protocol: ReferenceProtocol,
    condition: str,
    rng: np.random.Generator,
) -> int:
    state = sim.peds.state.copy()
    max_speeds = np.asarray(sim.peds.max_speeds, dtype=float)
    recycled = 0
    plus = desired_directions[:, 0] > 0.0
    for index, direction_is_plus in enumerate(plus):
        crossed_boundary = (
            direction_is_plus and state[index, 0] > protocol.length_m + protocol.recycle_margin_m
        ) or (not direction_is_plus and state[index, 0] < -protocol.recycle_margin_m)
        if not crossed_boundary:
            continue
        entry_x = 0.5 if direction_is_plus else protocol.length_m - 0.5
        if condition == "separated_lane_control":
            entry_y = protocol.lane_offset_m if direction_is_plus else -protocol.lane_offset_m
        else:
            entry_y = float(rng.uniform(-protocol.entry_y_span_m, protocol.entry_y_span_m))
        state[index, 0] = entry_x
        state[index, 1] = entry_y
        state[index, 2] = max_speeds[index] if direction_is_plus else -max_speeds[index]
        state[index, 3] = 0.0
        state[index, 4] = protocol.length_m if direction_is_plus else 0.0
        state[index, 5] = entry_y
        recycled += 1
    if recycled:
        sim.peds.state = state
    return recycled


def run_native_reference(
    *,
    protocol: ReferenceProtocol,
    condition: str,
    seed: int,
    calibration: SpeedCalibration,
    sampling_strides: tuple[int, ...] | list[int] = DEFAULT_SAMPLING_STRIDES,
    sim_config: Any | None = None,
) -> dict[str, Any]:
    """Run one native warm-up/recycled-flow reference condition.

    Returns:
        A compact native row.  The trajectory itself remains in memory so raw
        trajectory bytes are not promoted as durable evidence.
    """
    protocol.validate()
    _validate_condition(condition)
    strides = _validate_sampling_strides(sampling_strides)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if sim_config is None:
        sim_config = released_default_config()

    scenario = ScenarioConfig(
        name="bidirectional_corridor",
        length=protocol.length_m,
        half_width=protocol.half_width_m,
        n_pedestrians=protocol.n_pedestrians,
        seed=int(seed),
        n_steps=protocol.observation_steps,
    )
    state, obstacles, desired_directions = build_bidirectional_corridor(scenario, calibration)
    if condition == "separated_lane_control":
        _apply_separated_lane_initialization(state, desired_directions, protocol)

    sim = pysf.Simulator(state=state.copy(), obstacles=obstacles, config=sim_config)
    rng = np.random.default_rng(int(seed) + 6969000)
    recycled_agents = 0

    def step_and_recycle() -> None:
        nonlocal recycled_agents
        sim.step()
        recycled_agents += _recycle_exited_agents(
            sim,
            desired_directions=desired_directions,
            protocol=protocol,
            condition=condition,
            rng=rng,
        )

    for _ in range(protocol.warmup_steps):
        step_and_recycle()

    positions = [sim.peds.pos().copy()]
    velocities = [sim.peds.vel().copy()]
    for _ in range(protocol.observation_steps):
        step_and_recycle()
        positions.append(sim.peds.pos().copy())
        velocities.append(sim.peds.vel().copy())

    dt = float(sim_config.scene_config.dt_secs)
    trajectory = TrajectoryRecord(
        positions=np.asarray(positions, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        desired_directions=desired_directions,
        times=np.arange(len(positions), dtype=float) * dt,
        dt=dt,
    )
    metrics = _metric_values(trajectory)
    sampling_metrics = {
        str(stride): _metric_values(_sample_trajectory(trajectory, stride)) for stride in strides
    }
    return {
        "record_type": "lane_formation_reference_native.v1",
        "issue": "robot_sf_ll7#6969",
        "claim_boundary": CLAIM_BOUNDARY,
        "condition": condition,
        "calibration": calibration.name,
        "seed": int(seed),
        "protocol": protocol.as_dict(),
        "metrics": metrics,
        "threshold_evaluations": _reference_thresholds(metrics),
        "sampling_metrics": sampling_metrics,
        "recycled_agents": int(recycled_agents),
        "execution": {
            "status": "computed",
            "execution_mode": "native",
            "warmup_steps_discarded": protocol.warmup_steps,
            "observation_steps_recorded": protocol.observation_steps,
        },
        "simulator_config": simulator_config_snapshot(sim_config),
        "positive_control_is_not_emergence_claim": condition == "separated_lane_control",
    }


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def summarize_reference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize native reference rows without selecting a winning regime.

    Returns:
        Deterministically sorted summaries grouped by condition and calibration.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["condition"], row["calibration"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (condition, calibration), records in sorted(groups.items()):
        metrics = {
            name: _stats([float(record["metrics"][name]) for record in records])
            for name in ("lane_segregation_index", "lane_purity")
        }
        sampling_spreads = {}
        for name in metrics:
            per_run = [
                max(
                    float(record["sampling_metrics"][stride][name])
                    for stride in record["sampling_metrics"]
                )
                - min(
                    float(record["sampling_metrics"][stride][name])
                    for stride in record["sampling_metrics"]
                )
                for record in records
            ]
            sampling_spreads[name] = {
                "max_per_run": float(max(per_run)),
                "mean_per_run": float(np.mean(per_run)),
            }
        summaries.append(
            {
                "record_type": "lane_formation_reference_summary.v1",
                "condition": condition,
                "calibration": calibration,
                "n_seeds": len(records),
                "seeds": sorted(int(record["seed"]) for record in records),
                "metric_stats": metrics,
                "sampling_metric_spread": sampling_spreads,
                "recycled_agents": {
                    "total": int(sum(record["recycled_agents"] for record in records)),
                    "mean_per_run": float(
                        np.mean([record["recycled_agents"] for record in records])
                    ),
                },
                "execution_status_counts": {
                    "native:computed": len(records),
                },
                "interpretation": (
                    "controlled positive reference, not spontaneous-emergence evidence"
                    if condition == "separated_lane_control"
                    else "sustained mixed-flow measurement control"
                ),
            }
        )
    return summaries


def run_reference_campaign(
    *,
    protocol: ReferenceProtocol = ReferenceProtocol(),
    seeds: tuple[int, ...] | list[int] = DEFAULT_REFERENCE_SEEDS,
    conditions: tuple[str, ...] | list[str] = DEFAULT_REFERENCE_CONDITIONS,
    calibrations: tuple[SpeedCalibration, ...]
    | list[SpeedCalibration] = DEFAULT_REFERENCE_CALIBRATIONS,
    sampling_strides: tuple[int, ...] | list[int] = DEFAULT_SAMPLING_STRIDES,
    sim_config: Any | None = None,
) -> dict[str, Any]:
    """Run the metric audit and bounded native reference campaign.

    Returns:
        Payload containing the manifest, metric audit, native rows, and summaries.
    """
    protocol.validate()
    if not seeds or any(isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must contain at least one integer")
    for condition in conditions:
        _validate_condition(condition)
    if not calibrations:
        raise ValueError("calibrations must contain at least one value")
    strides = _validate_sampling_strides(sampling_strides)
    rows = [
        run_native_reference(
            protocol=protocol,
            condition=condition,
            seed=int(seed),
            calibration=calibration,
            sampling_strides=strides,
            sim_config=sim_config,
        )
        for condition in conditions
        for calibration in calibrations
        for seed in seeds
    ]
    if any(
        row["execution"]["execution_mode"] != "native" or row["execution"]["status"] != "computed"
        for row in rows
    ):
        raise RuntimeError("reference campaign contains a non-native or non-computed row")
    config = sim_config if sim_config is not None else released_default_config()
    manifest = {
        "schema_version": "lane_formation_reference_manifest.v1",
        "issue": "robot_sf_ll7#6969",
        "purpose": (
            "Calibrate lane-metric references and test an explicit warm-up/sustained-flow "
            "control before any Social Force Model parameter screen."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "released_defaults_changed": False,
        "metric_semantics_changed": False,
        "answerability_state": "diagnostic_only",
        "answerability_reason": (
            "The positive control is initialized in separated lanes; the native mixed-flow "
            "condition is a sustained measurement control. Neither establishes spontaneous "
            "lane formation or justifies parameter tuning."
        ),
        "canonical_harness": [
            "robot_sf.research.emergent_phenomena",
            "robot_sf.research.lane_formation_sensitivity",
            "robot_sf.research.lane_formation_reference",
        ],
        "protocol": protocol.as_dict(),
        "seeds": [int(seed) for seed in seeds],
        "conditions": list(conditions),
        "calibrations": [calibration.name for calibration in calibrations],
        "sampling_strides": list(strides),
        "simulator_config": simulator_config_snapshot(config),
        "execution_policy": {
            "allowed_native_success": ["native:computed"],
            "fallback_degraded_unavailable_policy": "explicit_status_and_fail_closed",
            "row_count": len(rows),
        },
        "positive_control_policy": (
            "separated_lane_control is a metric/trajectory control and cannot be interpreted "
            "as spontaneous lane emergence"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest,
        "metric_audit": metric_reference_audit(sampling_strides=strides),
        "rows": rows,
        "summary": summarize_reference_rows(rows),
    }
