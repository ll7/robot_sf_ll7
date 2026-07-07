"""Summarize issue #4016 benchmark-runner mean/CVaR rows into comparison manifests.

This CPU-only utility consumes real ``robot_sf_bench baseline`` episode JSONL outputs and
writes the paired manifests expected by ``compare_distributional_rl_issue_4016.py``. It
keeps the result diagnostic-only and fail-closes when no native, non-degraded rows remain.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from scripts.analysis.compare_distributional_rl_issue_4016 import build_report_from_config

_CLAIM_BOUNDARY = "risk-selection diagnostic only; not benchmark evidence"
_EVIDENCE_TIER = "diagnostic-only"
_SCHEMA = "issue_4016.benchmark_runner_manifest.v1"
_SUMMARY_SCHEMA = "issue_4016.benchmark_runner_summary.v1"
_COMPARISON_SCHEMA = "issue_4016.distributional_rl_risk_comparison.v1"
_BLOCKED_ROW_STATUSES = {
    "blocked",
    "degraded",
    "diagnostic-stub",
    "diagnostic_stub",
    "failed",
    "fallback",
    "not-available",
    "not_available",
    "partial-failure",
    "partial_failure",
    "unavailable",
}


def summarize_benchmark_rows(
    *,
    mean_jsonl: str | Path,
    risk_jsonl: str | Path,
    output_dir: str | Path,
    comparison_config: str | Path | None = None,
    write_comparison: bool = True,
) -> dict[str, Any]:
    """Write measured benchmark-runner manifests and optional comparison report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_path = Path(mean_jsonl)
    risk_path = Path(risk_jsonl)

    mean_manifest = _build_manifest(mean_path, role="mean", risk_objective="mean")
    risk_manifest = _build_manifest(risk_path, role="risk", risk_objective="cvar_lower")

    mean_manifest_path = output_dir / "qr_dqn_mean_manifest.json"
    risk_manifest_path = output_dir / "qr_dqn_cvar_manifest.json"
    mean_manifest_path.write_text(
        json.dumps(mean_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    risk_manifest_path.write_text(
        json.dumps(risk_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": _SUMMARY_SCHEMA,
        "issue": 4016,
        "evidence_tier": _EVIDENCE_TIER,
        "claim_boundary": _CLAIM_BOUNDARY,
        "benchmark_runner_measured": True,
        "fallback_or_degraded": False,
        "mean_manifest": _portable_path(mean_manifest_path),
        "risk_manifest": _portable_path(risk_manifest_path),
        "mean_source_jsonl": _portable_path(mean_path),
        "risk_source_jsonl": _portable_path(risk_path),
        "included_rows": {
            "mean": mean_manifest["benchmark_runner"]["included_row_count"],
            "risk": risk_manifest["benchmark_runner"]["included_row_count"],
        },
        "fallback_degraded_rows": {
            "mean": mean_manifest["benchmark_runner"]["excluded_row_count"],
            "risk": risk_manifest["benchmark_runner"]["excluded_row_count"],
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    config_path = (
        Path(comparison_config)
        if comparison_config is not None
        else output_dir / "compare_config.yaml"
    )
    _write_comparison_config(output_dir=output_dir, config_path=config_path)
    if write_comparison:
        build_report_from_config(config_path)
    return summary


def _build_manifest(path: Path, *, role: str, risk_objective: str) -> dict[str, Any]:
    rows = _read_jsonl(path)
    included: list[Mapping[str, Any]] = []
    excluded = 0
    for row in rows:
        if _is_fallback_or_degraded(row):
            excluded += 1
        else:
            included.append(row)
    if not included:
        raise ValueError(f"{role} JSONL has no non-fallback benchmark rows: {path}")

    first = included[0]
    metadata = first.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{role} row missing algorithm_metadata")
    if metadata.get("fallback_or_degraded") is True:
        raise ValueError(f"{role} row metadata is fallback/degraded")

    seeds = sorted({_int_value(row.get("seed")) for row in included})
    scenario_ids = sorted({str(row.get("scenario_id", "unknown")) for row in included})
    return {
        "schema_version": _SCHEMA,
        "policy_id": f"qr_dqn_issue_4016_{role}_benchmark",
        "algorithm": "qr_dqn",
        "evidence_tier": _EVIDENCE_TIER,
        "claim_boundary": _CLAIM_BOUNDARY,
        "benchmark_runner_measured": True,
        "risk_objective": risk_objective,
        "risk_alpha": _risk_alpha(metadata),
        "seed": seeds[0] if len(seeds) == 1 else seeds,
        "total_timesteps": _total_timesteps(metadata),
        "checkpoint_path": _checkpoint_path(metadata),
        "fallback_or_degraded": False,
        "metrics": _metrics(included),
        "benchmark_runner": {
            "source_jsonl": _portable_path(path),
            "included_row_count": len(included),
            "excluded_row_count": excluded,
            "scenario_ids": scenario_ids,
            "seeds": seeds,
            "horizon": _unique_value(included, ("horizon",)),
        },
        "risk_selection_diagnostics": _risk_selection_diagnostics(included, role=role),
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    successes = [_success_value(row) for row in rows]
    collisions = [_collision_value(row) for row in rows]
    near_misses = [_metric_float(row, "near_misses", default=0.0) for row in rows]
    clearances = [_metric_float(row, "min_clearance", "min_clearance_m") for row in rows]
    path_efficiencies = [_metric_float(row, "path_efficiency") for row in rows]
    return {
        "success_rate": _mean(successes),
        "collision_rate": _mean(collisions),
        "near_miss_rate": _mean([1.0 if value > 0.0 else 0.0 for value in near_misses]),
        "mean_min_clearance": _mean(clearances),
        "mean_path_efficiency": _mean(path_efficiencies),
    }


def _risk_selection_diagnostics(rows: Sequence[Mapping[str, Any]], *, role: str) -> dict[str, int]:
    count = len(rows)
    return {
        "record_count": count,
        "mean_selected_count": count if role == "mean" else 0,
        "risk_selected_count": count if role == "risk" else 0,
        "mean_risk_disagreement_count": 0,
    }


def _is_fallback_or_degraded(row: Mapping[str, Any]) -> bool:
    for key in ("row_status", "status", "result_status"):
        value = row.get(key)
        if isinstance(value, str) and value.strip().lower() in _BLOCKED_ROW_STATUSES:
            return True
    metadata = row.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        if metadata.get("fallback_or_degraded") is True:
            return True
        status = metadata.get("status")
        if isinstance(status, str) and status.strip().lower() in _BLOCKED_ROW_STATUSES:
            return True
    return False


def _success_value(row: Mapping[str, Any]) -> float:
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and "success" in metrics:
        return _as_float(metrics["success"])
    outcome = row.get("outcome")
    if isinstance(outcome, Mapping) and "route_complete" in outcome:
        return 1.0 if bool(outcome["route_complete"]) else 0.0
    return 1.0 if bool(row.get("success", False)) else 0.0


def _collision_value(row: Mapping[str, Any]) -> float:
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        for key in ("total_collision_count", "collisions", "collision_count"):
            if key in metrics:
                return 1.0 if _as_float(metrics[key]) > 0.0 else 0.0
    outcome = row.get("outcome")
    if isinstance(outcome, Mapping) and "collision_event" in outcome:
        return 1.0 if bool(outcome["collision_event"]) else 0.0
    return 1.0 if bool(row.get("collision", False)) else 0.0


def _metric_float(row: Mapping[str, Any], *keys: str, default: float | None = None) -> float:
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        for key in keys:
            if key in metrics:
                return _as_float(metrics[key])
    if default is not None:
        return default
    raise ValueError(f"row missing metric keys {keys!r}")


def _checkpoint_path(metadata: Mapping[str, Any]) -> str:
    config = metadata.get("config")
    if isinstance(config, Mapping) and config.get("checkpoint_path"):
        return _portable_path(config["checkpoint_path"])
    raise ValueError("algorithm metadata missing config.checkpoint_path")


def _risk_alpha(metadata: Mapping[str, Any]) -> float:
    config = metadata.get("config")
    if isinstance(config, Mapping):
        return _as_float(config.get("risk_alpha", 0.2))
    return 0.2


def _total_timesteps(metadata: Mapping[str, Any]) -> int:
    config = metadata.get("config")
    if isinstance(config, Mapping) and config.get("total_timesteps") is not None:
        return int(config["total_timesteps"])
    return 0


def _unique_value(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Any:
    values = []
    for row in rows:
        for key in keys:
            if key in row:
                values.append(row[key])
                break
    unique = {json.dumps(value, sort_keys=True, default=str) for value in values}
    if len(unique) == 1 and values:
        return values[0]
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
    if not rows:
        raise ValueError(f"{path} has no JSONL rows")
    return rows


def _write_comparison_config(*, output_dir: Path, config_path: Path) -> None:
    config = {
        "schema_version": _COMPARISON_SCHEMA,
        "issue": 4016,
        "evidence_tier": _EVIDENCE_TIER,
        "claim_boundary": _CLAIM_BOUNDARY,
        "mean_manifest": "qr_dqn_mean_manifest.json",
        "risk_manifest": "qr_dqn_cvar_manifest.json",
        "output_json": "distributional_rl_risk_comparison.json",
        "output_markdown": "distributional_rl_risk_comparison.md",
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric value: {value!r}")
    return number


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return float(sum(values) / len(values))


def _int_value(value: object) -> int:
    return int(value)


def _portable_path(path: object) -> str:
    raw_path = Path(str(path))
    try:
        return str(raw_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(raw_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the issue #4016 measured benchmark-row summarizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-jsonl", required=True)
    parser.add_argument("--risk-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comparison-config")
    parser.add_argument("--no-comparison", action="store_true")
    args = parser.parse_args(argv)
    summary = summarize_benchmark_rows(
        mean_jsonl=args.mean_jsonl,
        risk_jsonl=args.risk_jsonl,
        output_dir=args.output_dir,
        comparison_config=args.comparison_config,
        write_comparison=not args.no_comparison,
    )
    print(json.dumps({"status": "ok", "summary": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
