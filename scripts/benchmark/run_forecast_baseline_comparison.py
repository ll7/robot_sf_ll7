#!/usr/bin/env python3
"""Run the #2915 forecast-baseline comparison campaign.

This is analysis-only evidence. It compares existing deterministic forecast
baselines on identical durable trace fixtures and writes schema-valid
ForecastBatch.v1 JSONL plus compact comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from robot_sf.benchmark.forecast_baseline_comparison import compare_forecast_baselines
from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
)
from robot_sf.benchmark.pedestrian_forecast import (
    BASELINE_FUNCTIONS,
    ForecastBaselineFunction,
    NeighborContext,
    PedestrianState,
    _call_baseline_with_neighbors,
    compute_batch_forecast_metrics,
    is_pedestrian_actor,
)
from robot_sf.benchmark.schemas.forecast_batch_schema import ForecastBatchSchema

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/research/forecast_baseline_comparison_issue_2915.yaml"
SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/forecast_batch.schema.v1.json"
_CV_EVAL_PATH = REPO_ROOT / "scripts/benchmark/run_cv_forecast_eval.py"
_CV_EVAL_SPEC = importlib.util.spec_from_file_location(
    "run_cv_forecast_eval_issue2915", _CV_EVAL_PATH
)
if _CV_EVAL_SPEC is None or _CV_EVAL_SPEC.loader is None:
    raise RuntimeError(f"could not import {_CV_EVAL_PATH}")
_CV_EVAL = importlib.util.module_from_spec(_CV_EVAL_SPEC)
_CV_EVAL_SPEC.loader.exec_module(_CV_EVAL)
TRACE_CANDIDATES = _CV_EVAL.TRACE_CANDIDATES
MISSING_FAMILIES = _CV_EVAL.MISSING_FAMILIES
_actor_class_counts = _CV_EVAL._actor_class_counts
_compute_dt_s = _CV_EVAL._compute_dt_s
_extract_trace_steps = _CV_EVAL._extract_trace_steps
_load_trace = _CV_EVAL._load_trace
_sanitize_metrics = _CV_EVAL._sanitize_metrics
_trace_has_motion = _CV_EVAL._trace_has_motion
_trace_metadata_coverage = _CV_EVAL._trace_metadata_coverage
PRIMARY_METRIC_MAP = {
    "miss_rate": "mean_miss_rate",
    "calibration_error": "mean_calibration_error",
    "collision_relevant_forecast_error": "mean_collision_relevance_error",
    "planner_relevant_risk_error": "mean_collision_relevance_error",
}
SECONDARY_METRIC_MAP = {
    "ade": "mean_ade",
    "fde": "mean_ade",
}


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    if data.get("schema_version") != "forecast_baseline_comparison_config.v1":
        raise ValueError("config schema_version must be forecast_baseline_comparison_config.v1")
    return data


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ["<git status failed>"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def _display_path(path: pathlib.Path) -> str:
    """Return repo-relative paths when possible, otherwise an absolute path."""
    return str(
        path.relative_to(REPO_ROOT)
        if path.is_absolute() and path.is_relative_to(REPO_ROOT)
        else path
    )


def _resolve_output_path(config: Mapping[str, Any], key: str) -> pathlib.Path:
    output = config.get("output")
    if not isinstance(output, dict) or key not in output:
        raise ValueError(f"config output.{key} is required")
    path = pathlib.Path(str(output[key]))
    return path if path.is_absolute() else REPO_ROOT / path


def _selected_candidates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    families = set(config.get("scenario_families") or [])
    seeds = set(config.get("seeds") or [])
    candidates = [
        dict(candidate)
        for candidate in TRACE_CANDIDATES
        if (not families or candidate["family"] in families)
        and (not seeds or candidate.get("seed") in seeds)
    ]
    if not candidates:
        raise ValueError("config selected no trace candidates")
    return candidates


def _baseline_function(name: str) -> ForecastBaselineFunction:
    if name not in BASELINE_FUNCTIONS:
        raise ValueError(f"unknown baseline {name!r}; available={sorted(BASELINE_FUNCTIONS)}")
    return BASELINE_FUNCTIONS[name]


def _metric_average(
    metrics: Mapping[str, float], prefix: str, horizons_s: Sequence[float]
) -> float | None:
    values = [
        float(metrics[f"{prefix}_{horizon:g}s"])
        for horizon in horizons_s
        if f"{prefix}_{horizon:g}s" in metrics
    ]
    if not values:
        return None
    return float(np.mean(values))


def _metric_fde(metrics: Mapping[str, float], horizons_s: Sequence[float]) -> float | None:
    for horizon in sorted((float(h) for h in horizons_s), reverse=True):
        key = f"mean_ade_{horizon:g}s"
        if key in metrics:
            return float(metrics[key])
    return None


def _flatten_metrics(metrics: Mapping[str, float], config: Mapping[str, Any]) -> dict[str, float]:
    metric_config = config.get("metrics") if isinstance(config.get("metrics"), dict) else {}
    horizons_s = [float(h) for h in config.get("horizons_s", [])]
    flattened: dict[str, float] = {}
    for metric in metric_config.get("primary", []):
        prefix = PRIMARY_METRIC_MAP.get(str(metric))
        if prefix is None:
            continue
        value = _metric_average(metrics, prefix, horizons_s)
        if value is not None:
            flattened[str(metric)] = value
    for metric in metric_config.get("secondary", []):
        if str(metric) == "fde":
            value = _metric_fde(metrics, horizons_s)
        else:
            value = _metric_average(metrics, SECONDARY_METRIC_MAP.get(str(metric), ""), horizons_s)
        if value is not None:
            flattened[str(metric)] = value
    return flattened


def _trace_eligible(trace: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[bool, str | None]:
    frames = trace.get("frames") or trace.get("steps") or []
    if len(frames) < 3:
        return False, "insufficient_frames"
    if not _trace_has_motion(dict(trace)):
        return False, "limited_no_pedestrian_motion"
    if not any(
        is_pedestrian_actor(p.get("actor_type")) for f in frames for p in f.get("pedestrians", [])
    ):
        return False, "no_pedestrian_actors"
    max_horizon = max(float(h) for h in config.get("horizons_s", [0.5]))
    dt_s = _compute_dt_s(dict(trace))
    if round(max_horizon / dt_s) >= len(frames):
        return False, "insufficient_frames_for_requested_horizons"
    return True, None


def _neighbor_contexts(
    pedestrian: Mapping[str, Any], frame: Mapping[str, Any]
) -> list[NeighborContext]:
    ego_id = int(pedestrian["id"])
    neighbors: list[NeighborContext] = []
    for other in frame.get("pedestrians", []):
        if int(other["id"]) == ego_id:
            continue
        neighbors.append(
            NeighborContext(
                position=np.asarray(other["position"], dtype=float),
                velocity=np.asarray(other["velocity"], dtype=float),
                actor_type=str(other.get("actor_type") or "pedestrian"),
            )
        )
    return neighbors


def _first_forecastable_frame(trace_steps: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for frame in trace_steps:
        if any(is_pedestrian_actor(p.get("actor_type")) for p in frame.get("pedestrians", [])):
            return frame
    return None


def _build_forecast_batch(
    *,
    trace: Mapping[str, Any],
    trace_steps: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    baseline_name: str,
    baseline_function: ForecastBaselineFunction,
    config: Mapping[str, Any],
    timestamp: str,
) -> ForecastBatch:
    horizons_s = [float(h) for h in config.get("horizons_s", [])]
    frame = _first_forecastable_frame(trace_steps)
    if frame is None:
        raise ValueError("trace has no forecastable pedestrian frame")

    forecasts: list[ActorForecast] = []
    actor_ids: list[str] = []
    actor_classes: dict[str, str] = {}
    actor_mask: list[bool] = []
    for pedestrian in frame.get("pedestrians", []):
        actor_id = str(pedestrian["id"])
        is_included = is_pedestrian_actor(pedestrian.get("actor_type"))
        actor_ids.append(actor_id)
        actor_mask.append(is_included)
        actor_classes[actor_id] = str(pedestrian.get("actor_type") or "pedestrian")
        if not is_included:
            continue
        state = PedestrianState.from_trace(dict(pedestrian))
        ped_forecast = _call_baseline_with_neighbors(
            baseline_function,
            state,
            horizons_s,
            _neighbor_contexts(pedestrian, frame),
        )
        forecasts.append(
            ActorForecast(
                actor_id=actor_id,
                deterministic=np.asarray([p.mean for p in ped_forecast.predictions], dtype=float),
                gaussian=[
                    {
                        "mean": p.mean.tolist(),
                        "cov": p.covariance.tolist(),
                    }
                    for p in ped_forecast.predictions
                ],
                uncertainty_metadata={
                    "baseline": baseline_name,
                    "horizons_s": horizons_s,
                    "per_horizon_metadata": [dict(p.metadata) for p in ped_forecast.predictions],
                },
            )
        )

    provenance = ForecastBatchProvenance(
        predictor_id=baseline_name,
        predictor_family="deterministic_forecast_baseline",
        observation_tier="trace_fixture",
        frame=CoordinateFrame(name="world"),
        dt_s=_compute_dt_s(dict(trace)),
        horizons_s=horizons_s,
        scenario_id=str(candidate.get("scenario_id") or candidate["family"]),
        seed=int(candidate.get("seed") or 0),
        fallback_status="native",
        degraded_status="not_degraded",
        actor_ids=actor_ids,
        actor_mask=actor_mask,
        actor_mask_metadata={
            "policy": "pedestrian/person actors included; non-pedestrian actors excluded",
            "history_steps": int(config.get("observation_history_steps", 0)),
            "sample_stride_steps": int(config.get("sample_stride_steps", 0)),
        },
        feature_schema={
            "source": "simulation_trace_export.v1",
            "features": [
                "position",
                "velocity",
                "intent_label",
                "signal_state",
                "neighbor_context",
            ],
        },
        timestamp=timestamp,
        actor_classes=actor_classes,
        oracle_state=False,
    )
    return ForecastBatch(
        provenance=provenance,
        forecasts=forecasts,
        metadata={
            "issue": 2915,
            "evidence_tier": "analysis_only",
            "trace_path": candidate["path"],
            "scenario_family": candidate["family"],
            "trace_label": candidate["label"],
        },
    )


def _validate_batch_schema(batch: ForecastBatch, schema: ForecastBatchSchema) -> None:
    schema.validate_forecast_batch_data(batch.to_dict())


def _write_jsonl(path: pathlib.Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True))
            stream.write("\n")


def _write_csv(
    path: pathlib.Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _run_candidate_baseline(
    *,
    candidate: Mapping[str, Any],
    baseline_name: str,
    baseline_function: ForecastBaselineFunction,
    config: Mapping[str, Any],
    timestamp: str,
    schema: ForecastBatchSchema,
) -> tuple[dict[str, Any], ForecastBatch | None]:
    rel_path = pathlib.Path(str(candidate["path"]))
    abs_path = REPO_ROOT / rel_path
    row: dict[str, Any] = {
        "scenario_family": candidate["family"],
        "trace_label": candidate["label"],
        "trace_path": str(rel_path),
        "scenario_id": candidate.get("scenario_id", ""),
        "planner_id": candidate.get("planner_id", ""),
        "seed": candidate.get("seed"),
        "baseline": baseline_name,
        "evidence_tier": "analysis_only",
        "row_status": "evaluated",
    }
    if not abs_path.exists():
        row["row_status"] = "trace_file_missing"
        row["exclusion_reason"] = "trace file missing"
        return row, None

    trace = _load_trace(abs_path)
    trace_steps = _extract_trace_steps(trace)
    row["frame_count"] = len(trace.get("frames") or trace.get("steps") or [])
    row["actor_class_counts"] = json.dumps(_actor_class_counts(trace), sort_keys=True)
    row["metadata_coverage"] = json.dumps(_trace_metadata_coverage(trace), sort_keys=True)
    eligible, reason = _trace_eligible(trace, config)
    if not eligible:
        row["row_status"] = reason or "not_available"
        row["exclusion_reason"] = reason or "not available"
        return row, None

    metrics_config = config.get("metrics") if isinstance(config.get("metrics"), dict) else {}
    try:
        metrics = compute_batch_forecast_metrics(
            trace_steps,
            horizons_s=[float(h) for h in config.get("horizons_s", [])],
            dt_s=_compute_dt_s(trace),
            confidence_level=float(metrics_config.get("confidence_level", 0.95)),
            collision_distance_m=float(metrics_config.get("collision_distance_m", 0.8)),
            baseline_function=baseline_function,
        )
        batch = _build_forecast_batch(
            trace=trace,
            trace_steps=trace_steps,
            candidate=candidate,
            baseline_name=baseline_name,
            baseline_function=baseline_function,
            config=config,
            timestamp=timestamp,
        )
        _validate_batch_schema(batch, schema)
    except Exception as exc:
        row["row_status"] = "failed"
        row["exclusion_reason"] = str(exc)
        return row, None

    flattened = _flatten_metrics(metrics, config)
    row.update(flattened)
    row["forecast_evaluable_samples"] = metrics.get("forecast_evaluable_samples", 0.0)
    row["raw_metrics"] = json.dumps(_sanitize_metrics(metrics), sort_keys=True)
    return row, batch


def _aggregate_by_family(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    metric_names = list(config.get("metrics", {}).get("primary", [])) + list(
        config.get("metrics", {}).get("secondary", [])
    )
    keys = sorted({(str(r["scenario_family"]), str(r["baseline"])) for r in rows})
    aggregates: list[dict[str, Any]] = []
    for family, baseline in keys:
        group = [
            r
            for r in rows
            if r["scenario_family"] == family
            and r["baseline"] == baseline
            and r["row_status"] == "evaluated"
        ]
        excluded = [
            r
            for r in rows
            if r["scenario_family"] == family
            and r["baseline"] == baseline
            and r["row_status"] != "evaluated"
        ]
        out: dict[str, Any] = {
            "scenario_family": family,
            "baseline": baseline,
            "evaluated_rows": len(group),
            "excluded_rows": len(excluded),
            "row_status": "evaluated" if group else "not_available",
        }
        for metric in metric_names:
            values = [float(r[metric]) for r in group if r.get(metric) not in (None, "")]
            if values:
                out[metric] = float(np.mean(values))
        aggregates.append(out)
    return aggregates


def _missing_family_rows(
    config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    baselines: Sequence[str],
) -> list[dict[str, Any]]:
    selected = {str(family) for family in config.get("scenario_families", [])}
    covered = {str(candidate["family"]) for candidate in candidates}
    missing_reasons = {item["family"]: item["reason"] for item in MISSING_FAMILIES}
    rows: list[dict[str, Any]] = []
    for family in sorted(selected - covered):
        for baseline in baselines:
            rows.append(
                {
                    "scenario_family": family,
                    "trace_label": "",
                    "trace_path": "",
                    "scenario_id": family,
                    "planner_id": "",
                    "seed": "",
                    "baseline": baseline,
                    "evidence_tier": "analysis_only",
                    "row_status": "not_available",
                    "exclusion_reason": missing_reasons.get(
                        family, "no configured durable trace candidate"
                    ),
                }
            )
    return rows


def _row_metadata_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    summary = {
        "rows_with_metadata": 0,
        "rows_without_metadata": 0,
        "rows_with_signal_metadata": 0,
        "rows_with_intent_metadata": 0,
        "rows_total": len(rows),
    }
    for row in rows:
        raw = row.get("metadata_coverage", {})
        coverage = json.loads(raw) if isinstance(raw, str) and raw else raw
        if not isinstance(coverage, dict):
            summary["rows_without_metadata"] += 1
            continue
        has_signal = bool(coverage.get("has_signal_metadata"))
        has_intent = bool(coverage.get("has_intent_metadata"))
        if has_signal:
            summary["rows_with_signal_metadata"] += 1
        if has_intent:
            summary["rows_with_intent_metadata"] += 1
        if coverage.get("metadata_presence") == "present":
            summary["rows_with_metadata"] += 1
        else:
            summary["rows_without_metadata"] += 1
    return summary


def _strongest_by_family(
    aggregates: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    primary = [str(m) for m in config.get("metrics", {}).get("primary", [])]
    secondary = [str(m) for m in config.get("metrics", {}).get("secondary", [])]
    families = sorted({str(row["scenario_family"]) for row in aggregates})
    result: dict[str, Any] = {}
    for family in families:
        group = [
            row
            for row in aggregates
            if row["scenario_family"] == family and row.get("row_status") == "evaluated"
        ]
        if not group:
            result[family] = {"best": None, "reason": "no evaluated baseline rows"}
            continue
        metrics = primary + secondary
        comparable = [
            row for row in group if any(row.get(metric) is not None for metric in metrics)
        ]
        if not comparable:
            result[family] = {"best": None, "reason": "no comparable metrics"}
            continue

        def score(row: Mapping[str, Any]) -> tuple[float, ...]:
            return tuple(float(row.get(metric, float("inf"))) for metric in metrics)

        winner = sorted(comparable, key=lambda row: (score(row), str(row["baseline"])))[0]
        comparison_payload = compare_forecast_baselines(
            {
                str(row["baseline"]): {
                    metric: float(row[metric])
                    for metric in metrics
                    if row.get(metric) not in (None, "")
                }
                for row in comparable
            },
            metrics=metrics,
            lower_is_better_metrics=metrics,
        ).to_dict()
        result[family] = {
            "best": winner["baseline"],
            "primary_score_order": primary,
            "comparison": comparison_payload,
        }
    return result


def _write_markdown(
    path: pathlib.Path,
    *,
    summary: Mapping[str, Any],
    aggregates: Sequence[Mapping[str, Any]],
    strongest: Mapping[str, Any],
) -> None:
    lines = [
        "# Forecast Baseline Comparison (#2915)",
        "",
        "## Claim Boundary",
        "",
        "Analysis-only diagnostic evidence. This compares deterministic forecast baselines on a bounded set of durable Robot SF trace fixtures. It is not planner-promotion or paper-facing benchmark evidence.",
        "",
        "## Reproducibility",
        "",
        f"- Issue: #{summary['issue']}",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Git HEAD: `{summary['git_head']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Command: `{summary['command']}`",
        f"- Evidence tier: `{summary['evidence_tier']}`",
        "",
        "## Strongest Baseline By Family",
        "",
        "| Scenario family | Strongest baseline | Note |",
        "| --- | --- | --- |",
    ]
    for family, payload in strongest.items():
        best = payload.get("best") or "not available"
        note = payload.get("reason") or "ranked on primary metrics, then ADE/FDE"
        lines.append(f"| {family} | {best} | {note} |")
    lines.extend(
        [
            "",
            "## Aggregate Rows",
            "",
            "| Scenario family | Baseline | Status | Evaluated rows | Excluded rows | Miss rate | Calibration error | Collision relevance error | Planner risk error | ADE | FDE |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(
            "| {scenario_family} | {baseline} | {row_status} | {evaluated_rows} | {excluded_rows} | {miss_rate} | {calibration_error} | {collision_relevant_forecast_error} | {planner_relevant_risk_error} | {ade} | {fde} |".format(
                **{
                    key: row.get(key, "")
                    for key in [
                        "scenario_family",
                        "baseline",
                        "row_status",
                        "evaluated_rows",
                        "excluded_rows",
                        "miss_rate",
                        "calibration_error",
                        "collision_relevant_forecast_error",
                        "planner_relevant_risk_error",
                        "ade",
                        "fde",
                    ]
                }
            )
        )
    lines.extend(
        [
            "",
            "## Learned-Predictor Gap",
            "",
            "Residual learned-predictor gap remains diagnostic-only: learned predictor expansion should stay gated until this comparison is repeated on broader fixed-scope trace coverage with non-degenerate crossing, bottleneck, and dense-interaction families.",
            "",
            "## Limitations",
            "",
            "- Rows with missing files, no pedestrian motion, insufficient frames, or execution errors are excluded fail-closed and not ranked as successful evidence.",
            "- `planner_relevant_risk_error` is the same collision-relevance forecast error surface used by the existing pedestrian forecast metric path; it is a proxy for planner-risk relevance, not closed-loop planner evidence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: pathlib.Path,
    *,
    date: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run the forecast-baseline comparison and write configured evidence artifacts."""
    config = _load_config(config_path)
    generated_at = generated_at_utc or dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")
    git_status_at_start = _git_status_short()
    evidence_dir = _resolve_output_path(config, "evidence_dir")
    forecast_dir = _resolve_output_path(config, "forecast_jsonl_dir")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    forecast_dir.mkdir(parents=True, exist_ok=True)
    schema = ForecastBatchSchema(SCHEMA_PATH)

    baselines = [str(name) for name in config.get("baselines", [])]
    candidates = _selected_candidates(config)
    rows: list[dict[str, Any]] = []
    batches_by_baseline: dict[str, list[dict[str, Any]]] = {baseline: [] for baseline in baselines}
    for candidate in candidates:
        for baseline in baselines:
            row, batch = _run_candidate_baseline(
                candidate=candidate,
                baseline_name=baseline,
                baseline_function=_baseline_function(baseline),
                config=config,
                timestamp=generated_at,
                schema=schema,
            )
            rows.append(row)
            if batch is not None:
                batches_by_baseline[baseline].append(batch.to_dict())
    rows.extend(_missing_family_rows(config, candidates, baselines))

    for baseline, payloads in batches_by_baseline.items():
        _write_jsonl(forecast_dir / f"{baseline}.forecast_batch.v1.jsonl", payloads)

    aggregates = _aggregate_by_family(rows, config)
    strongest = _strongest_by_family(aggregates, config)
    metric_names = list(config.get("metrics", {}).get("primary", [])) + list(
        config.get("metrics", {}).get("secondary", [])
    )
    fieldnames = [
        "scenario_family",
        "baseline",
        "row_status",
        "evaluated_rows",
        "excluded_rows",
        *metric_names,
    ]
    _write_csv(_resolve_output_path(config, "comparison_table"), aggregates, fieldnames)

    missing_selected = [
        item
        for item in MISSING_FAMILIES
        if item["family"] in set(config.get("scenario_families", []))
    ]
    summary = {
        "issue": int(config.get("issue", 2915)),
        "evidence_tier": str(config.get("evidence_tier", "analysis_only")),
        "generated_at_utc": generated_at,
        "date": date,
        "config_path": _display_path(config_path),
        "command": " ".join(sys.argv),
        "git_head": _git_head(),
        "git_status_short_at_generation": git_status_at_start,
        "baselines": baselines,
        "scenario_families": list(config.get("scenario_families", [])),
        "selected_trace_count": len(candidates),
        "row_status_counts": {
            status: sum(1 for row in rows if row["row_status"] == status)
            for status in sorted({str(row["row_status"]) for row in rows})
        },
        "metadata_summary": _row_metadata_summary(rows),
        "missing_selected_families": missing_selected,
        "strongest_by_family": strongest,
        "learned_predictor_gap": (
            "diagnostic residual gap remains; do not unblock learned predictor promotion "
            "without broader fixed-scope trace coverage"
        ),
        "forecast_jsonl_files": [
            _display_path(forecast_dir / f"{baseline}.forecast_batch.v1.jsonl")
            for baseline in baselines
        ],
    }
    report = {
        "summary": summary,
        "rows": rows,
        "aggregates": aggregates,
        "strongest_by_family": strongest,
    }
    report_json = _resolve_output_path(config, "comparison_report_json")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_json = _resolve_output_path(config, "summary_json")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(
        _resolve_output_path(config, "comparison_report_md"),
        summary=summary,
        aggregates=aggregates,
        strongest=strongest,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--date", default=None, help="Evidence date label for reproducibility.")
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Fixed ISO-8601 UTC timestamp for reproducible evidence.",
    )
    args = parser.parse_args(argv)
    report = run(
        args.config if args.config.is_absolute() else REPO_ROOT / args.config,
        date=args.date,
        generated_at_utc=args.generated_at_utc,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
