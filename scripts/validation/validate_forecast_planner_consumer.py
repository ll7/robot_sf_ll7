#!/usr/bin/env python3
"""Validate forecast_variant consumption by a real planner path.

This smoke drives ``PredictionPlannerAdapter.plan()`` on one deterministic
motion-rich SocNav observation for each baseline forecast variant.  It is smoke
evidence for planner consumption, not benchmark evidence of planner benefit.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.common.forecast_variants import FORECAST_VARIANT_CHOICES
from robot_sf.planner.socnav import PredictionPlannerAdapter, SocNavPlannerConfig

SCHEMA_VERSION = "ForecastPlannerConsumerSmoke.v1"
ISSUE = 2960
DEFAULT_VARIANTS = FORECAST_VARIANT_CHOICES
DT_S = 0.5
COLLISION_DISTANCE_M = 0.5
NEAR_MISS_DISTANCE_M = 1.5
STOP_SPEED_MPS = 0.05


@dataclass(frozen=True)
class VariantSmokeRow:
    """Compact same-seed smoke row for one forecast variant."""

    variant: str
    execution_mode: str
    classification: str
    collision: bool
    near_miss: bool
    min_distance_m: float
    stop_yield_timing_steps: int
    progress_m: float
    false_positive_stops: int
    runtime_s: float
    linear_velocity: float
    angular_velocity: float
    prediction_changed_vs_none: bool


def _make_motion_rich_observation() -> dict[str, Any]:
    """Return one deterministic SocNav observation with moving nearby pedestrians."""
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([0.0], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([0.5], dtype=np.float32),
        },
        "goal": {
            "current": np.array([4.0, 0.0], dtype=np.float32),
            "next": np.array([4.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.array([[0.9, 0.15], [1.15, -0.05], [2.2, 0.45]], dtype=np.float32),
            "velocities": np.array([[-0.25, 0.0], [-0.05, 0.1], [-0.2, -0.05]], dtype=np.float32),
            "radius": np.array([0.35, 0.35, 0.35], dtype=np.float32),
            "count": np.array([3.0], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([DT_S], dtype=np.float32), "time_s": np.array([0.0])},
    }


def _planner_config(variant: str) -> SocNavPlannerConfig:
    """Build the config-first planner settings used by the smoke."""
    return SocNavPlannerConfig(
        forecast_variant=variant,
        forecast_variant_horizons_s=(0.5, 1.0, 1.5, 2.0),
        forecast_variant_dt_s=0.5,
        forecast_variant_risk_distance_m=1.4,
        predictive_checkpoint_path="output/missing_issue_2960_forecast_smoke_model.pt",
        predictive_horizon_steps=4,
        predictive_rollout_dt=DT_S,
        predictive_max_agents=4,
        predictive_candidate_speeds=(0.0, 0.25, 0.5, 0.75),
        predictive_candidate_heading_deltas=(-np.pi / 8, 0.0, np.pi / 8),
        predictive_collision_weight=1.2,
        predictive_ttc_weight=0.8,
        predictive_ttc_distance=1.2,
    )


def _consumed_future(
    adapter: PredictionPlannerAdapter, observation: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the future trajectories consumed by the planner scoring path."""
    state, mask, _robot_pos, _robot_heading = adapter._build_model_input(observation)
    return adapter._predict_trajectories(state, mask), mask


def _min_distance_for_action(
    future_peds: np.ndarray,
    *,
    mask: np.ndarray,
    linear_velocity: float,
    angular_velocity: float,
    dt_s: float = DT_S,
) -> float:
    """Compute minimum robot-pedestrian distance over the smoke rollout."""
    robot = np.array([0.0, 0.0], dtype=np.float32)
    heading = 0.0
    min_distance = float("inf")
    for step_index in range(future_peds.shape[1]):
        heading += float(angular_velocity) * dt_s
        robot = (
            robot
            + np.array(
                [np.cos(heading), np.sin(heading)],
                dtype=np.float32,
            )
            * float(linear_velocity)
            * dt_s
        )
        for ped_index in range(future_peds.shape[0]):
            if float(mask[ped_index]) <= 0.5:
                continue
            distance = float(np.linalg.norm(robot - future_peds[ped_index, step_index]))
            min_distance = min(min_distance, distance)
    return min_distance


def _classify_variant(
    *,
    variant: str,
    execution_mode: str,
    prediction_changed_vs_none: bool,
) -> str:
    """Classify one variant under fail-closed smoke semantics."""
    if execution_mode == "blocked":
        return "blocked"
    if execution_mode == "degraded":
        return "degraded"
    if variant == "none":
        return "native"
    if execution_mode == "native" and prediction_changed_vs_none:
        return "native"
    if execution_mode == "native":
        return "degraded"
    return "diagnostic_only"


def run_smoke(variants: tuple[str, ...] = DEFAULT_VARIANTS) -> dict[str, Any]:
    """Run the forecast planner-consumer smoke and return a JSON-ready report."""
    observation = _make_motion_rich_observation()
    reference_adapter = PredictionPlannerAdapter(_planner_config("none"), allow_fallback=True)
    reference_future, reference_mask = _consumed_future(reference_adapter, observation)
    valid_reference_rows = reference_mask > 0.5

    rows: list[VariantSmokeRow] = []
    for variant in variants:
        start = time.perf_counter()
        adapter = PredictionPlannerAdapter(_planner_config(variant), allow_fallback=True)
        future, mask = _consumed_future(adapter, observation)
        linear_velocity, angular_velocity = adapter.plan(observation)
        runtime_s = time.perf_counter() - start
        min_distance = _min_distance_for_action(
            future,
            mask=mask,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
        stopped = float(linear_velocity) <= STOP_SPEED_MPS
        valid_rows = (mask > 0.5) & valid_reference_rows
        prediction_changed = not np.allclose(
            future[valid_rows],
            reference_future[valid_rows],
            atol=1e-6,
        )
        execution_mode = adapter.get_forecast_variant_execution_mode()
        classification = _classify_variant(
            variant=variant,
            execution_mode=execution_mode,
            prediction_changed_vs_none=prediction_changed,
        )
        rows.append(
            VariantSmokeRow(
                variant=variant,
                execution_mode=execution_mode,
                classification=classification,
                collision=bool(min_distance <= COLLISION_DISTANCE_M),
                near_miss=bool(min_distance <= NEAR_MISS_DISTANCE_M),
                min_distance_m=min_distance,
                stop_yield_timing_steps=int(stopped),
                progress_m=max(0.0, float(linear_velocity) * DT_S),
                false_positive_stops=int(stopped and min_distance > NEAR_MISS_DISTANCE_M),
                runtime_s=runtime_s,
                linear_velocity=float(linear_velocity),
                angular_velocity=float(angular_velocity),
                prediction_changed_vs_none=bool(prediction_changed),
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "claim_boundary": (
            "Smoke evidence that forecast_variant can feed a real PredictionPlannerAdapter "
            "consumer. This is not nominal or benchmark evidence and does not claim any "
            "forecast variant improves safety, success, or runtime."
        ),
        "fixture": {
            "seed": 2960,
            "scenario": "deterministic_motion_rich_socnav_smoke",
            "same_seed_across_variants": True,
            "dt_s": DT_S,
        },
        "required_metrics": [
            "collision",
            "near_miss",
            "min_distance_m",
            "stop_yield_timing_steps",
            "progress_m",
            "false_positive_stops",
            "runtime_s",
        ],
        "variant_results": [asdict(row) for row in rows],
        "result_classification": "smoke",
        "limitations": [
            "Single deterministic SocNav observation; not a statistically powered benchmark.",
            "Planner actions are local one-step commands, not full episode success evidence.",
            "Rows classify planner-consumption mechanics, not forecast quality or navigation benefit.",
        ],
    }


def format_markdown(report: dict[str, Any]) -> str:
    """Format a compact Markdown report."""
    lines = [
        "# Issue #2960 Forecast Planner Consumer Smoke",
        "",
        report["claim_boundary"],
        "",
        "## Fixture",
        "",
        f"- Seed: {report['fixture']['seed']}",
        f"- Scenario: {report['fixture']['scenario']}",
        f"- Same seed across variants: {report['fixture']['same_seed_across_variants']}",
        "",
        "## Variant Results",
        "",
        "| variant | class | mode | collision | near miss | min distance | progress | stop steps | false-positive stops | runtime | changed vs none |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["variant_results"]:
        lines.append(
            "| {variant} | {classification} | {execution_mode} | {collision} | {near_miss} | "
            "{min_distance_m:.3f} | {progress_m:.3f} | {stop_yield_timing_steps} | "
            "{false_positive_stops} | {runtime_s:.6f} | {prediction_changed_vs_none} |".format(
                **row
            )
        )
    lines.extend(["", "## Limitations"])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/forecast_planner_consumer_smoke"),
    )
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    args = parser.parse_args()

    variants = tuple(str(variant) for variant in args.variants)
    unsupported = sorted(set(variants) - set(FORECAST_VARIANT_CHOICES))
    if unsupported:
        parser.error(f"unsupported variants: {unsupported}")

    report = run_smoke(variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "forecast_planner_consumer_smoke.json"
    md_path = args.output_dir / "forecast_planner_consumer_smoke.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
