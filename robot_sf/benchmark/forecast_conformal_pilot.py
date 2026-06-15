"""Smoke-only conformal forecast tube diagnostics for ForecastBatch.v1 artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.forecast_batch import FORECAST_BATCH_SCHEMA_VERSION, ForecastBatch

if TYPE_CHECKING:
    from robot_sf.benchmark.forecast_metrics import GroundTruthPositions

FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION = "ForecastConformalPilot.v1"


def build_forecast_conformal_pilot_report(
    calibration_cases: list[dict[str, Any]],
    evaluation_cases: list[dict[str, Any]],
    *,
    report_id: str,
    coverage_target: float = 0.9,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a split-conformal deterministic tube smoke report.

    Args:
        calibration_cases: Cases with ``batch`` and ``ground_truth`` fields used only to fit radii.
        evaluation_cases: Held-out cases with ``batch`` and ``ground_truth`` fields used only to
            measure empirical coverage.
        report_id: Stable identifier for the generated report.
        coverage_target: Desired empirical coverage for conformal tubes.
        generated_at_utc: Optional deterministic timestamp.

    Returns:
        JSON-compatible conformal pilot report.
    """
    if not 0.0 < coverage_target < 1.0:
        raise ValueError("coverage_target must be between 0 and 1")
    if not calibration_cases:
        return _empty_report(
            report_id=report_id,
            coverage_target=coverage_target,
            generated_at_utc=generated_at_utc,
            reason="no calibration cases supplied",
            calibration_cases=calibration_cases,
            evaluation_cases=evaluation_cases,
        )
    if not evaluation_cases:
        return _empty_report(
            report_id=report_id,
            coverage_target=coverage_target,
            generated_at_utc=generated_at_utc,
            reason="no held-out evaluation cases supplied",
            calibration_cases=calibration_cases,
            evaluation_cases=evaluation_cases,
        )

    calibration_scores = _collect_scores(calibration_cases, split_name="calibration")
    evaluation_scores = _collect_scores(evaluation_cases, split_name="evaluation")
    group_keys = sorted(set(calibration_scores) | set(evaluation_scores))
    pilot_rows = [
        _build_pilot_row(
            group_key=group_key,
            calibration_scores=calibration_scores.get(group_key, []),
            evaluation_scores=evaluation_scores.get(group_key, []),
            coverage_target=coverage_target,
        )
        for group_key in group_keys
    ]
    limitation_rows = _limitation_rows(pilot_rows)
    if not pilot_rows:
        limitation_rows.append(
            {
                "scenario_family": "all",
                "horizon_s": 0.0,
                "observation_tier": "all",
                "predictor_family": "all",
                "reason": "no deterministic forecast and ground-truth score pairs were available",
            }
        )
    return {
        "schema_version": FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "source_schema_version": FORECAST_BATCH_SCHEMA_VERSION,
        "method": {
            "name": "split_conformal_deterministic_tube",
            "coverage_target": float(coverage_target),
            "score": "euclidean deterministic forecast residual by actor and horizon",
            "set_size_proxy": "pi * conformal_radius_m^2 per horizon",
        },
        "split_provenance": {
            "calibration_case_count": len(calibration_cases),
            "evaluation_case_count": len(evaluation_cases),
            "calibration_split_ids": _split_ids(calibration_cases, default="calibration"),
            "evaluation_split_ids": _split_ids(evaluation_cases, default="heldout_evaluation"),
            "held_out_required": True,
        },
        "pilot_rows": pilot_rows,
        "limitation_rows": limitation_rows,
        "recommendation": _recommendation(pilot_rows, limitation_rows),
        "claim_boundary": (
            "Conformal pilot rows are smoke evidence only. Simulator-held-out coverage and "
            "deterministic tube size do not prove planner safety, real-world coverage, or "
            "closed-loop navigation benefit."
        ),
    }


def format_forecast_conformal_pilot_markdown(report: dict[str, Any]) -> str:
    """Format a compact Markdown conformal pilot report.

    Returns:
        Markdown report text with pilot rows, limitations, and claim boundary.
    """
    recommendation = report["recommendation"]
    lines = [
        "# Forecast Conformal Pilot Report",
        "",
        f"- Report id: {report['report_id']}",
        f"- Decision: {recommendation['decision']}",
        f"- Claim status: {recommendation['claim_status']}",
        f"- Calibration cases: {report['split_provenance']['calibration_case_count']}",
        f"- Held-out evaluation cases: {report['split_provenance']['evaluation_case_count']}",
        f"- Pilot rows: {len(report['pilot_rows'])}",
        f"- Limitation rows: {len(report['limitation_rows'])}",
        "",
        "| scenario_family | horizon_s | observation_tier | predictor_family | radius_m | coverage | set_size | status |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["pilot_rows"]:
        lines.append(
            "| {scenario_family} | {horizon_s:g} | {observation_tier} | {predictor_family} | "
            "{radius} | {coverage} | {set_size} | {status} |".format(
                scenario_family=row["scenario_family"],
                horizon_s=float(row["horizon_s"]),
                observation_tier=row["observation_tier"],
                predictor_family=row["predictor_family"],
                radius=_format_optional_float(row["conformal_radius_m"]),
                coverage=_format_optional_float(row["empirical_coverage"]),
                set_size=_format_optional_float(row["mean_set_size_proxy_m2"]),
                status=row["coverage_status"],
            )
        )
    if report["limitation_rows"]:
        lines.extend(["", "## Limitations", ""])
        for row in report["limitation_rows"]:
            lines.append(
                "- {scenario_family} / {horizon_s:g}s / {observation_tier} / "
                "{predictor_family}: {reason}".format(**row)
            )
    lines.extend(["", report["claim_boundary"]])
    return "\n".join(lines) + "\n"


def write_forecast_conformal_pilot_report(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write conformal pilot JSON and optional Markdown artifacts.

    Returns:
        Mapping of artifact kind to written path.
    """
    _require_conformal_pilot_report(report)
    json_content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_content = (
        None if markdown_path is None else format_forecast_conformal_pilot_markdown(report)
    )
    paths = {"json": Path(json_path)}
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json_content, encoding="utf-8")
    if markdown_path is not None:
        paths["markdown"] = Path(markdown_path)
        paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
        paths["markdown"].write_text(markdown_content or "", encoding="utf-8")
    return paths


def _collect_scores(
    cases: list[dict[str, Any]],
    *,
    split_name: str,
) -> dict[tuple[str, float, str, str], list[dict[str, Any]]]:
    """Collect deterministic residual scores grouped by comparable forecast provenance.

    Args:
        cases: Calibration or held-out evaluation cases with batch/truth payloads.
        split_name: Fallback split label used when a case omits ``split_id``.

    Returns:
        Residual-score rows keyed by scenario family, horizon, observation tier, and predictor
        family so radii and coverage are never mixed across incompatible forecast settings.
    """
    grouped: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case_index, case in enumerate(cases):
        batch = _case_batch(case, case_index=case_index, split_name=split_name)
        truth = _case_ground_truth(case, case_index=case_index, split_name=split_name)
        scenario_family = _scenario_family(batch)
        for forecast in batch.forecasts:
            if forecast.deterministic is None:
                continue
            actor_truth = _truth_array(
                truth,
                actor_id=forecast.actor_id,
                expected_steps=len(batch.provenance.horizons_s),
            )
            if actor_truth is None:
                continue
            for horizon_index, horizon_s in enumerate(batch.provenance.horizons_s):
                residual = float(
                    np.linalg.norm(
                        forecast.deterministic[horizon_index] - actor_truth[horizon_index]
                    )
                )
                # Group by the report dimensions that define a comparable conformal population.
                grouped[
                    (
                        scenario_family,
                        float(horizon_s),
                        batch.provenance.observation_tier,
                        batch.provenance.predictor_family,
                    )
                ].append(
                    {
                        "residual_m": residual,
                        "scenario_id": batch.provenance.scenario_id,
                        "actor_id": forecast.actor_id,
                        "split_id": _case_split_id(case, default=split_name),
                        "collision_relevance": _collision_relevance(forecast.uncertainty_metadata),
                    }
                )
    return grouped


def _build_pilot_row(
    *,
    group_key: tuple[str, float, str, str],
    calibration_scores: list[dict[str, Any]],
    evaluation_scores: list[dict[str, Any]],
    coverage_target: float,
) -> dict[str, Any]:
    """Build one conformal pilot row from calibration and held-out residual groups.

    Args:
        group_key: Scenario family, horizon, observation tier, and predictor family tuple.
        calibration_scores: Residuals used only to fit the conformal radius.
        evaluation_scores: Held-out residuals used only to measure empirical coverage.
        coverage_target: Requested split-conformal coverage target.

    Returns:
        JSON-compatible pilot row with denominators, radius, set-size proxy, coverage, and
        recommendation fields.
    """
    scenario_family, horizon_s, observation_tier, predictor_family = group_key
    radius = _conformal_radius(
        [float(score["residual_m"]) for score in calibration_scores],
        coverage_target=coverage_target,
    )
    covered = []
    if radius is not None:
        covered = [float(score["residual_m"]) <= radius for score in evaluation_scores]
    empirical_coverage = None if not covered else sum(covered) / len(covered)
    coverage_gap = None if empirical_coverage is None else empirical_coverage - coverage_target
    status = _coverage_status(
        calibration_count=len(calibration_scores),
        evaluation_count=len(evaluation_scores),
        coverage_gap=coverage_gap,
    )
    return {
        "scenario_family": scenario_family,
        "horizon_s": float(horizon_s),
        "observation_tier": observation_tier,
        "predictor_family": predictor_family,
        "coverage_target": float(coverage_target),
        "conformal_radius_m": radius,
        "mean_set_size_proxy_m2": None if radius is None else math.pi * radius * radius,
        "calibration_denominator": len(calibration_scores),
        "evaluation_denominator": len(evaluation_scores),
        "empirical_coverage": empirical_coverage,
        "coverage_gap": coverage_gap,
        "miss_count": None if not covered else int(len(covered) - sum(covered)),
        "collision_relevance_available_count": sum(
            1 for score in evaluation_scores if score["collision_relevance"] is not None
        ),
        "coverage_status": status,
        "recommendation": _row_recommendation(status),
    }


def _conformal_radius(scores: list[float], *, coverage_target: float) -> float | None:
    finite_scores = sorted(score for score in scores if np.isfinite(score))
    if not finite_scores:
        return None
    rank = math.ceil((len(finite_scores) + 1) * coverage_target)
    if rank > len(finite_scores):
        return None
    index = min(max(rank - 1, 0), len(finite_scores) - 1)
    return float(finite_scores[index])


def _coverage_status(
    *,
    calibration_count: int,
    evaluation_count: int,
    coverage_gap: float | None,
) -> str:
    if calibration_count <= 0:
        return "unavailable_no_calibration_denominator"
    if evaluation_count <= 0 or coverage_gap is None:
        return "unavailable_no_evaluation_denominator"
    if coverage_gap < 0.0:
        return "under_covered_heldout"
    return "covered_heldout_smoke"


def _limitation_rows(pilot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limitations = []
    for row in pilot_rows:
        if row["coverage_status"].startswith("unavailable"):
            reason = row["coverage_status"]
        elif row["coverage_status"] == "under_covered_heldout":
            reason = "held-out coverage below target"
        else:
            reason = None
        if reason is not None:
            limitations.append(
                {
                    "scenario_family": row["scenario_family"],
                    "horizon_s": row["horizon_s"],
                    "observation_tier": row["observation_tier"],
                    "predictor_family": row["predictor_family"],
                    "reason": reason,
                }
            )
    return limitations


def _recommendation(
    pilot_rows: list[dict[str, Any]],
    limitation_rows: list[dict[str, Any]],
) -> dict[str, str]:
    if not pilot_rows:
        return {
            "decision": "wait",
            "claim_status": "blocked",
            "reason": "no conformal pilot rows were produced",
        }
    if any(row["coverage_status"].startswith("unavailable") for row in pilot_rows):
        return {
            "decision": "wait",
            "claim_status": "diagnostic-only",
            "reason": "one or more groups lack calibration or held-out evaluation denominators",
        }
    if limitation_rows:
        return {
            "decision": "revise",
            "claim_status": "diagnostic-only",
            "reason": "one or more held-out groups missed the coverage target",
        }
    return {
        "decision": "continue",
        "claim_status": "smoke-only",
        "reason": "all held-out groups reached the requested simulator coverage target",
    }


def _row_recommendation(status: str) -> str:
    if status == "covered_heldout_smoke":
        return "continue"
    if status.startswith("unavailable"):
        return "wait"
    return "revise"


def _case_batch(case: dict[str, Any], *, case_index: int, split_name: str) -> ForecastBatch:
    if not isinstance(case, dict):
        raise ValueError(f"{split_name}_cases[{case_index}] must be a mapping")
    if "batch" not in case:
        raise ValueError(f"{split_name}_cases[{case_index}].batch is required")
    batch = case["batch"]
    if isinstance(batch, ForecastBatch):
        return batch
    if isinstance(batch, dict):
        return ForecastBatch.from_dict(batch)
    raise ValueError(f"{split_name}_cases[{case_index}].batch must be a ForecastBatch mapping")


def _case_ground_truth(
    case: dict[str, Any],
    *,
    case_index: int,
    split_name: str,
) -> GroundTruthPositions:
    ground_truth = case.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError(f"{split_name}_cases[{case_index}].ground_truth must be a mapping")
    return ground_truth


def _truth_array(
    ground_truth: GroundTruthPositions,
    *,
    actor_id: str,
    expected_steps: int,
) -> np.ndarray | None:
    if actor_id not in ground_truth:
        return None
    array = np.asarray(ground_truth[actor_id], dtype=float)
    if array.shape != (expected_steps, 2):
        raise ValueError("ground_truth trajectories must align with horizons_s and shape (T, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError("ground_truth trajectories must contain only finite values")
    return array


def _scenario_family(batch: ForecastBatch) -> str:
    family = batch.metadata.get("scenario_family")
    if family is not None:
        return str(family)
    return batch.provenance.scenario_id.split("_seed_", maxsplit=1)[0]


def _collision_relevance(metadata: dict[str, Any] | None) -> float | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("collision_relevance")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _split_ids(cases: list[dict[str, Any]], *, default: str) -> list[str]:
    return sorted(
        {_case_split_id(case, default=default) for case in cases if isinstance(case, dict)}
    )


def _case_split_id(case: dict[str, Any], *, default: str) -> str:
    value = case.get("split_id")
    return default if value is None else str(value)


def _empty_report(
    *,
    report_id: str,
    coverage_target: float,
    generated_at_utc: str | None,
    reason: str,
    calibration_cases: list[dict[str, Any]],
    evaluation_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "source_schema_version": FORECAST_BATCH_SCHEMA_VERSION,
        "method": {
            "name": "split_conformal_deterministic_tube",
            "coverage_target": float(coverage_target),
            "score": "euclidean deterministic forecast residual by actor and horizon",
            "set_size_proxy": "pi * conformal_radius_m^2 per horizon",
        },
        "split_provenance": {
            "calibration_case_count": len(calibration_cases),
            "evaluation_case_count": len(evaluation_cases),
            "calibration_split_ids": _split_ids(calibration_cases, default="calibration"),
            "evaluation_split_ids": _split_ids(
                evaluation_cases,
                default="heldout_evaluation",
            ),
            "held_out_required": True,
        },
        "pilot_rows": [],
        "limitation_rows": [
            {
                "scenario_family": "all",
                "horizon_s": 0.0,
                "observation_tier": "all",
                "predictor_family": "all",
                "reason": reason,
            }
        ],
        "recommendation": {
            "decision": "wait",
            "claim_status": "blocked",
            "reason": reason,
        },
        "claim_boundary": "No conformal pilot claim is supported without held-out split evidence.",
    }


def _require_conformal_pilot_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("report must be a mapping")
    if report.get("schema_version") != FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION:
        raise ValueError("report must use ForecastConformalPilot.v1")


def _format_optional_float(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.6g}"


__all__ = [
    "FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION",
    "build_forecast_conformal_pilot_report",
    "format_forecast_conformal_pilot_markdown",
    "write_forecast_conformal_pilot_report",
]
